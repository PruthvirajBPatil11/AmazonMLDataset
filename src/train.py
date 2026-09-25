"""
End-to-end training script.

Steps:
  1. Load train sources + ground truth.
  2. Stratified holdout split on Source 1 (by country + match-count bucket) so we
     have a trustworthy local F0.5 proxy for the real leaderboard.
  3. Run blocking on the TRAIN split, report candidate recall / reduction ratio
     (the ceiling on everything downstream -- always check this before touching
     the model).
  4. Build labeled pairs: every (source1, candidate) pair from blocking is a
     training example; label = 1 if it's in ground truth, else 0. This gives the
     model real negatives (near-miss candidates that blocking thought were
     plausible), which is what it actually needs to learn precision.
  5. Train LightGBM (binary classification, MIT-licensed, tiny model -- easily
     satisfies the <=8B param / permissive-license constraint).
  6. Save the model + report train-side metrics. Threshold tuning happens in
     infer.py against the held-out validation split.

Run from src/:
    python3 train.py --data-dir ../dataset/train --model-out ../model.txt
"""
import argparse
import random
import sys

import lightgbm as lgb
import pandas as pd

from blocking import build_index, build_record_keys, generate_candidates
from features import pair_features, FEATURE_COLUMNS

RNG = random.Random(42)


def load_sources(data_dir):
    s1 = pd.read_csv(f'{data_dir}/train_source1.tsv', sep='\t', dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f'{data_dir}/train_source2.tsv', sep='\t', dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f'{data_dir}/train_source3.tsv', sep='\t', dtype=str, keep_default_na=False)
    gt = pd.read_csv(f'{data_dir}/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
    return s1, s2, s3, gt


def match_bucket(n):
    if n == 0:
        return 0
    if n <= 2:
        return 1
    if n <= 4:
        return 2
    return 3


def stratified_split(s1_df, gt_df, holdout_frac=0.2, seed=42):
    """Split Source 1 entities into train/holdout, stratified by
    country x match-count-bucket, so rare strata (e.g. singletons, high-cardinality
    matches) are represented in both splits."""
    gt_map = dict(zip(gt_df.source1_entity_id, gt_df.matched_entity_ids))
    n_matches = s1_df['entity_id'].map(lambda e: 0 if not gt_map.get(e) else len(gt_map[e].split(',')))
    strata = list(zip(s1_df['country'], n_matches.map(match_bucket)))

    rng = random.Random(seed)
    from collections import defaultdict
    groups = defaultdict(list)
    for i, key in enumerate(strata):
        groups[key].append(i)

    holdout_idx = set()
    for key, idxs in groups.items():
        rng.shuffle(idxs)
        k = int(len(idxs) * holdout_frac)
        holdout_idx.update(idxs[:k])

    is_holdout = s1_df.index.isin(holdout_idx)
    return s1_df[~is_holdout].reset_index(drop=True), s1_df[is_holdout].reset_index(drop=True)


def compute_blocking_recall(s1_df, gt_df, index2, index3, max_per_key=2000):
    gt_map = dict(zip(gt_df.source1_entity_id, gt_df.matched_entity_ids))
    total_true = 0
    total_found = 0
    total_candidates = 0
    n_entities = 0
    candidate_map = {}
    for eid, rec, cands in generate_candidates(s1_df, index2, index3, max_per_key):
        candidate_map[eid] = (rec, cands)
        true_ids = set(gt_map.get(eid, '').split(',')) - {''}
        total_true += len(true_ids)
        total_found += len(true_ids & cands)
        total_candidates += len(cands)
        n_entities += 1
    recall = total_found / total_true if total_true else 1.0
    avg_cands = total_candidates / n_entities if n_entities else 0.0
    print(f'[blocking] entities={n_entities} avg_candidates/entity={avg_cands:.1f} '
          f'pair_recall={recall:.4f} ({total_found}/{total_true} true matches recovered)',
          file=sys.stderr)
    return candidate_map


def build_training_frame(s1_df, gt_df, candidate_map, index2, index3, neg_per_pos=5):
    gt_map = dict(zip(gt_df.source1_entity_id, gt_df.matched_entity_ids))
    rows = []
    for _, s1_row in s1_df.iterrows():
        eid = s1_row['entity_id']
        rec1, cands = candidate_map[eid]
        true_ids = set(gt_map.get(eid, '').split(',')) - {''}
        pos = [c for c in cands if c in true_ids]
        neg = [c for c in cands if c not in true_ids]
        # cap negatives per positive-bearing entity to keep class balance sane;
        # for pure-singleton entities (no positives) keep a small fixed sample
        # so the model still sees "everything here is a non-match" examples
        if pos:
            k_neg = min(len(neg), max(1, len(pos) * neg_per_pos))
        else:
            k_neg = min(len(neg), neg_per_pos)
        neg_sample = RNG.sample(neg, k_neg) if neg else []

        for cid in pos + neg_sample:
            rec2 = index2.records.get(cid) or index3.records.get(cid)
            if rec2 is None:
                continue
            feats = pair_features(rec1, rec2)
            feats['source1_entity_id'] = eid
            feats['candidate_entity_id'] = cid
            feats['label'] = int(cid in true_ids)
            rows.append(feats)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='../dataset/train')
    ap.add_argument('--model-out', default='../model.txt')
    ap.add_argument('--holdout-frac', type=float, default=0.2)
    ap.add_argument('--max-per-key', type=int, default=2000)
    ap.add_argument('--neg-per-pos', type=int, default=5)
    ap.add_argument('--sample-n', type=int, default=0,
                     help='If >0, subsample this many Source 1 rows for a fast dev run.')
    args = ap.parse_args()

    print('[load] reading sources...', file=sys.stderr)
    s1, s2, s3, gt = load_sources(args.data_dir)

    if args.sample_n:
        s1_ids = set(s1['entity_id'].sample(args.sample_n, random_state=42))
        s1 = s1[s1['entity_id'].isin(s1_ids)].reset_index(drop=True)
        gt = gt[gt['source1_entity_id'].isin(s1_ids)].reset_index(drop=True)

    train_s1, holdout_s1 = stratified_split(s1, gt, args.holdout_frac)
    print(f'[split] train={len(train_s1)} holdout={len(holdout_s1)}', file=sys.stderr)

    print('[index] building inverted indices over source2/source3...', file=sys.stderr)
    index2 = build_index(s2)
    index3 = build_index(s3)

    print('[blocking] train split...', file=sys.stderr)
    train_candidates = compute_blocking_recall(train_s1, gt, index2, index3, args.max_per_key)
    print('[blocking] holdout split...', file=sys.stderr)
    holdout_candidates = compute_blocking_recall(holdout_s1, gt, index2, index3, args.max_per_key)

    print('[features] building labeled training frame...', file=sys.stderr)
    train_frame = build_training_frame(train_s1, gt, train_candidates, index2, index3, args.neg_per_pos)
    print(f'[features] {len(train_frame)} labeled pairs, '
          f'positive rate={train_frame["label"].mean():.4f}', file=sys.stderr)

    X = train_frame[FEATURE_COLUMNS]
    y = train_frame['label']

    print('[train] fitting LightGBM...', file=sys.stderr)
    model = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y)
    model.booster_.save_model(args.model_out)
    print(f'[train] model saved to {args.model_out}', file=sys.stderr)

    importances = sorted(zip(FEATURE_COLUMNS, model.feature_importances_),
                          key=lambda x: -x[1])
    print('[train] feature importances:', file=sys.stderr)
    for name, imp in importances:
        print(f'    {name}: {imp}', file=sys.stderr)

    # persist holdout candidate pairs + labels for infer.py / threshold tuning
    holdout_frame = build_training_frame(holdout_s1, gt, holdout_candidates, index2, index3,
                                          neg_per_pos=10 ** 6)  # keep ALL candidates for honest eval
    holdout_frame.to_csv(args.model_out + '.holdout_pairs.csv', index=False)
    print(f'[train] holdout pairs saved to {args.model_out}.holdout_pairs.csv '
          f'({len(holdout_frame)} rows) -- use with infer.py --tune-threshold',
          file=sys.stderr)


if __name__ == '__main__':
    main()
