"""
Pairwise feature engineering.

Takes two record dicts (as produced by blocking.build_record_keys, which already
carries normalized text/tokens so we don't recompute normalization per pair) and
returns a flat dict of numeric features for the classifier.
"""
from rapidfuzz import fuzz


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _safe_ratio(fn, a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fn(a, b) / 100.0


def pair_features(rec1: dict, rec2: dict) -> dict:
    name1, name2 = rec1['norm_name'], rec2['norm_name']
    addr1, addr2 = rec1['norm_addr'], rec2['norm_addr']

    feats = {
        # --- name similarity, several complementary metrics ---
        'name_ratio': _safe_ratio(fuzz.ratio, name1, name2),
        'name_token_sort_ratio': _safe_ratio(fuzz.token_sort_ratio, name1, name2),
        'name_token_set_ratio': _safe_ratio(fuzz.token_set_ratio, name1, name2),
        'name_partial_ratio': _safe_ratio(fuzz.partial_ratio, name1, name2),
        'name_jaccard': _jaccard(rec1['name_tokens'], rec2['name_tokens']),
        'name_sig_jaccard': _jaccard(rec1['sig_name_tokens'], rec2['sig_name_tokens']),
        'name_prefix_match': float(
            bool(rec1['name_prefix']) and rec1['name_prefix'] == rec2['name_prefix']
        ),
        'soundex_match': float(
            bool(rec1['soundex']) and rec1['soundex'] == rec2['soundex']
        ),
        'name_len_ratio': (
            min(len(name1), len(name2)) / max(len(name1), len(name2))
            if name1 and name2 else 0.0
        ),

        # --- address similarity ---
        'addr_ratio': _safe_ratio(fuzz.ratio, addr1, addr2),
        'addr_token_sort_ratio': _safe_ratio(fuzz.token_sort_ratio, addr1, addr2),
        'addr_token_set_ratio': _safe_ratio(fuzz.token_set_ratio, addr1, addr2),
        'addr_jaccard': _jaccard(rec1['addr_tokens'], rec2['addr_tokens']),
        'addr_both_empty': float(not addr1 and not addr2),
        'addr_one_empty': float(bool(addr1) != bool(addr2)),

        # --- the strongest India signal from EDA: shared unit/plot/shop tokens ---
        'unit_token_overlap': _jaccard(rec1['unit_tokens'], rec2['unit_tokens']),
        'unit_token_shared_count': float(len(rec1['unit_tokens'] & rec2['unit_tokens'])),

        # --- cross features ---
        'country_match': float(rec1['country'] == rec2['country']),
        'name_x_addr': 0.0,  # filled below
    }
    feats['name_x_addr'] = feats['name_token_set_ratio'] * feats['addr_token_set_ratio']
    return feats


FEATURE_COLUMNS = [
    'name_ratio', 'name_token_sort_ratio', 'name_token_set_ratio', 'name_partial_ratio',
    'name_jaccard', 'name_sig_jaccard', 'name_prefix_match', 'soundex_match',
    'name_len_ratio', 'addr_ratio', 'addr_token_sort_ratio', 'addr_token_set_ratio',
    'addr_jaccard', 'addr_both_empty', 'addr_one_empty', 'unit_token_overlap',
    'unit_token_shared_count', 'country_match', 'name_x_addr',
]
