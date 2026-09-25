# Business Entity Resolution — Baseline Pipeline

Tested end-to-end on real data from the challenge (2.2M / 5.0M / 5.3M row sources).
Full-scale indexing of Source 2 + Source 3 takes ~13 minutes total on a single
core (~13k rows/sec); plan accordingly on your first full run.

## Setup

```bash
pip install -r requirements.txt
```

## Expected data layout

```
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

## 1. Train

```bash
cd src
python3 train.py --data-dir ../dataset/train --model-out ../model.txt
```

This will:
- split Source 1 into train/holdout (stratified by country x match-count, 80/20)
- build inverted-index blocking over Source 2 + Source 3
- **print blocking pair-recall** on both splits — check this first. If it's well
  below ~0.95, the model's ceiling is capped no matter how good the classifier is;
  go tighten/loosen blocking keys in `blocking.py` before touching the model.
- build labeled pairs (positive = true match, negative = other blocked candidates)
- train a LightGBM classifier (small, MIT-licensed, well under the 8B param limit)
- save `model.txt` and `model.txt.holdout_pairs.csv` (all holdout candidates,
  scored later for threshold tuning)

For a fast dev run on a subsample first: add `--sample-n 3000`.

## 2. Tune the decision threshold

```bash
python3 infer.py tune --model ../model.txt --holdout-pairs ../model.txt.holdout_pairs.csv
```

Sweeps a probability threshold and reports the one that **maximizes holdout
macro F0.5** — the actual competition metric, not F1 or accuracy. F0.5 weights
precision 2x recall, so the chosen threshold will typically be higher/stricter
than you'd pick for F1.

## 3. Predict on test set

```bash
python3 infer.py predict \
    --model ../model.txt \
    --data-dir ../dataset/test \
    --threshold <value from step 2> \
    --out-dir ../output
```

Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` in the
exact required format.

## 4. Validate before submitting

Run the challenge's own `utils/validate_submission.py` against the output
before uploading:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Pipeline overview

| Stage | File | What it does |
|---|---|---|
| Normalization | `normalize.py` | Unicode/diacritic cleanup, Devanagari→Latin transliteration (offline, rule-based — not an external lookup), legal-suffix canonicalization, address abbreviation expansion, unit/plot/shop token extraction |
| Blocking | `blocking.py` | Inverted-index candidate generation over 4 independent key types (name token, name prefix, unit token, soundex), unioned for recall |
| Features | `features.py` | ~19 pairwise similarity features per candidate pair (name/address fuzzy ratios, token Jaccard, unit-token overlap, country match) |
| Training | `train.py` | Stratified split, blocking-recall diagnostics, labeled pair construction, LightGBM training |
| Inference | `infer.py` | Threshold tuning against true F0.5, full test-set prediction, output writing |
| Scoring | `score.py` | Exact macro-average F0.5 implementation matching the challenge spec (singletons scored correctly) |

## Known limitations / where to improve next

This is a **baseline**, not a tuned final submission. In particular:

1. **Blocking `max_per_key` (default 2000)** needs re-tuning at full 5M-row
   scale — on the small smoke-test sample used to validate this code, some
   candidate sets were larger than they should be because common tokens are
   proportionally more common in a small pool. Watch the printed
   `avg_candidates/entity` on a full run and tighten if it's too large to
   score efficiently, or recall drops if too tight.
2. **Address parsing is deliberately unstructured** (token-set similarity only,
   no city/state/PIN field extraction) because component order varies too much
   across sources/countries in EDA, and this generalizes to the unseen France
   test-set country. A more structured, country-aware address parser could
   improve precision further if you're willing to add per-country logic that
   degrades gracefully on unseen countries.
3. **Devanagari transliteration** uses the ITRANS phonetic scheme via
   `indic_transliteration`. It's an approximation, not a lookup ("होटल" →
   "hotala" vs. target "hotel") — good enough to lift similarity scores from
   near-zero into fuzzy-matchable range, but a dedicated fuzzy phonetic
   distance metric tuned on this specific gap could do better.
4. **No feature currently captures word-substitution cases** like
   "Hotel Enterprises Limited" vs "Hotel Limited Services" well — both share
   two tokens but reorder/substitute the third. Consider adding a
   character-n-gram cosine similarity feature for these harder cases.
5. Negative sampling in `train.py` is a fixed ratio relative to positives; for
   singleton-heavy behavior, consider explicitly upweighting singleton
   entities during training since they're 5.6% of ground truth but easy for
   the model to ignore if under-sampled.
