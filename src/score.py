"""
Local F_0.5 scorer, matching the challenge's exact definition:
  - per-Source-1-entity F0.5, then macro-averaged across all entities
  - singleton (no true matches) scores 1.0 if predicted empty, else 0.0

Usage as a library:
    from score import f_half_macro
    score = f_half_macro(pred_map, truth_map)
where pred_map / truth_map: {source1_entity_id: set(matched_ids)}
"""


def f_half_entity(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    precision = tp / len(pred)
    recall = tp / len(truth)
    if precision == 0 and recall == 0:
        return 0.0
    beta2 = 0.25
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def f_half_macro(pred_map: dict, truth_map: dict) -> float:
    scores = []
    for s1_id, truth in truth_map.items():
        pred = pred_map.get(s1_id, set())
        scores.append(f_half_entity(pred, truth))
    return sum(scores) / len(scores) if scores else 0.0


def parse_id_list(s: str) -> set:
    if not s:
        return set()
    return set(x for x in s.split(',') if x)
