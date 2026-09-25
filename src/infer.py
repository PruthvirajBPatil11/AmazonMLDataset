"""
Two modes:

  tune    -- load the holdout_pairs.csv produced by train.py (all blocking
             candidates for the holdout split, labeled), score them with the
             trained model, and sweep a probability threshold to maximize
             macro F0.5 (the actual competition metric, not F1/accuracy).
             Also reports blocking recall ceiling on holdout for reference.

  predict -- run the full pipeline (blocking -> feature -> model -> threshold)
             on a test set with no ground truth, and write
             output/matching_results.tsv and output/candidate_pairs.tsv in the
             exact format the challenge requires.

Usage:
    python3 infer.py tune --model model.txt --holdout-pairs model.txt.holdout_pairs.csv
    python3 infer.py predict --model model.txt --data-dir ../dataset/test \
        --threshold 0.42 --out-dir ../output
"""
import argparse
import os
import sys

import lightgbm as lgb
import pandas as pd

from blocking import build_index, generate_candidates
from features import pair_features, FEATURE_COLUMNS
from score import f_half_macro, parse_id_list


def cmd_tune(args):
    df = pd.read_csv(args.holdout_pairs)
    booster = lgb.Booster(model_file=args.model)
    df['prob'] = booster.predict(df[FEATURE_COLUMNS])

    truth_map = {}
    for eid, g in df[df.label == 1].groupby('source1_entity_id'):
        truth_map[eid] = set(g['candidate_entity_id'])
    # entities present in holdout with zero true matches also need to be in truth_map
    # (as empty sets) so singletons are scored correctly
    for eid in df['source1_entity_id'].unique():
        truth_map.setdefault(eid, set())

    best_t, best_score = 0.5, -1.0
    for t in [i / 100 for i in range(1, 100)]:
        kept = df[df.prob >= t]
        pred_map = {}
        for eid, g in kept.groupby('source1_entity_id'):
            pred_map[eid] = set(g['candidate_entity_id'])
        score = f_half_macro(pred_map, truth_map)
        if score > best_score:
            best_score, best_t = score, t

    print(f'[tune] best threshold={best_t:.2f}  holdout macro F0.5={best_score:.4f}', file=sys.stderr)

    # also report blocking recall ceiling for context
    all_true = sum(len(v) for v in truth_map.values())
    all_candidates_true = df[df.label == 1]['candidate_entity_id'].nunique()
    print(f'[tune] (for reference) {all_true} true matches present among candidates '
          f'in holdout_pairs.csv -- this file already reflects blocking recall.',
          file=sys.stderr)
    return best_t


def cmd_predict(args):
    print('[load] reading test sources...', file=sys.stderr)
    s1 = pd.read_csv(f'{args.data_dir}/test_source1.tsv', sep='\t', dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f'{args.data_dir}/test_source2.tsv', sep='\t', dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f'{args.data_dir}/test_source3.tsv', sep='\t', dtype=str, keep_default_na=False)

    print('[index] building inverted indices...', file=sys.stderr)
    index2 = build_index(s2)
    index3 = build_index(s3)

    booster = lgb.Booster(model_file=args.model)

    os.makedirs(args.out_dir, exist_ok=True)
    match_rows = []
    cand_rows = []

    print('[predict] scoring candidates for every Source 1 test entity...', file=sys.stderr)
    n = 0
    for eid, rec1, cands in generate_candidates(s1, index2, index3, args.max_per_key):
        n += 1
        if n % 200000 == 0:
            print(f'[predict] {n} entities processed...', file=sys.stderr)

        cand_rows.append((eid, ','.join(sorted(cands))))

        if not cands:
            match_rows.append((eid, ''))
            continue

        feat_rows = []
        cand_ids = []
        for cid in cands:
            rec2 = index2.records.get(cid) or index3.records.get(cid)
            if rec2 is None:
                continue
            feat_rows.append(pair_features(rec1, rec2))
            cand_ids.append(cid)

        if not feat_rows:
            match_rows.append((eid, ''))
            continue

        X = pd.DataFrame(feat_rows)[FEATURE_COLUMNS]
        probs = booster.predict(X)
        kept = [cid for cid, p in zip(cand_ids, probs) if p >= args.threshold]

        # safety cap: training data never showed more than 11 true matches for
        # one Source 1 entity: if the model wants to keep far more than that,
        # keep only the top-N by probability instead (protects precision on
        # pathological candidate sets)
        if len(kept) > args.max_matches:
            ranked = sorted(zip(cand_ids, probs), key=lambda x: -x[1])
            kept = [cid for cid, p in ranked[:args.max_matches] if p >= args.threshold]

        match_rows.append((eid, ','.join(sorted(kept))))

    match_df = pd.DataFrame(match_rows, columns=['source1_entity_id', 'matched_entity_ids'])
    cand_df = pd.DataFrame(cand_rows, columns=['source1_entity_id', 'candidate_entity_ids'])

    match_path = os.path.join(args.out_dir, 'matching_results.tsv')
    cand_path = os.path.join(args.out_dir, 'candidate_pairs.tsv')
    match_df.to_csv(match_path, sep='\t', index=False)
    cand_df.to_csv(cand_path, sep='\t', index=False)
    print(f'[predict] wrote {match_path} and {cand_path}', file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('tune')
    t.add_argument('--model', required=True)
    t.add_argument('--holdout-pairs', required=True)

    p = sub.add_parser('predict')
    p.add_argument('--model', required=True)
    p.add_argument('--data-dir', default='../dataset/test')
    p.add_argument('--out-dir', default='../output')
    p.add_argument('--threshold', type=float, required=True)
    p.add_argument('--max-per-key', type=int, default=2000)
    p.add_argument('--max-matches', type=int, default=15)

    args = ap.parse_args()
    if args.cmd == 'tune':
        cmd_tune(args)
    else:
        cmd_predict(args)


if __name__ == '__main__':
    main()
