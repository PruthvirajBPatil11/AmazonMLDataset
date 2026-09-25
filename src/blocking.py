"""
Candidate generation (blocking) stage.

Strategy: build several cheap inverted indices over Source 2 + Source 3, keyed by
different normalized signals, then union the lookups for each Source 1 entity.
Using several *independent, weak* keys and unioning them gives much better recall
than one strict key, at the cost of a larger (but still tractable) candidate set --
appropriate here since the matching model downstream is what drives precision.

Blocking keys used (all computed on already-normalized text, see normalize.py):
  1. name_token key       -- any shared significant name token (excluding a small
                              stopword-ish set of ultra-common business words)
  2. name_prefix key      -- first 4 chars of the sorted, concatenated name tokens
                              (catches word-order transpositions + typos in later chars)
  3. unit_token key       -- shared alphanumeric unit/plot/shop/house-number token
                              (the strongest India signal found in EDA)
  4. soundex key          -- phonetic key on the first name token (catches spelling
                              variants and transliteration residue that survives
                              partial normalization)

No country-specific branching: every key is computed the same way regardless of
the country label, so behavior degrades gracefully on unseen labels (e.g. France
in test) instead of falling through a country if/else with no matching branch.
"""
from collections import defaultdict

import jellyfish

from normalize import (
    normalize_name, normalize_address, name_tokens, address_tokens,
    extract_unit_tokens,
)

# Extremely common business-name words carry ~no discriminative power as a lone
# blocking key (they would create huge, useless blocks). Excluded from key #1 only;
# still used normally in downstream similarity features.
_STOP_NAME_TOKENS = {
    'pvt', 'ltd', 'llc', 'llp', 'inc', 'corp', 'co', 'and', 'the', 'of',
    'services', 'enterprises', 'trading', 'private', 'limited', 'company',
    'group', 'solutions', 'consultants', 'india', 'international',
}


def build_record_keys(entity_id, business_name, business_address, country):
    """Compute normalized text + all blocking keys for one record.
    Returns a dict; used identically for source1 (query side) and source2/3
    (index side)."""
    norm_name = normalize_name(business_name)
    norm_addr = normalize_address(business_address)
    n_tok = name_tokens(norm_name)
    a_tok = address_tokens(norm_addr)
    unit_tok = extract_unit_tokens(norm_addr)

    sig_tokens = n_tok - _STOP_NAME_TOKENS
    name_prefix = ''.join(sorted(n_tok))[:4] if n_tok else ''
    first_tok = min(n_tok) if n_tok else ''
    soundex = jellyfish.soundex(first_tok) if first_tok else ''

    return {
        'entity_id': entity_id,
        'country': country,
        'norm_name': norm_name,
        'norm_addr': norm_addr,
        'name_tokens': n_tok,
        'addr_tokens': a_tok,
        'unit_tokens': unit_tok,
        'sig_name_tokens': sig_tokens,
        'name_prefix': name_prefix,
        'soundex': soundex,
    }


class BlockingIndex:
    """Inverted index over one or more record sources (S2, S3)."""

    def __init__(self):
        self.by_name_token = defaultdict(set)
        self.by_name_prefix = defaultdict(set)
        self.by_unit_token = defaultdict(set)
        self.by_soundex = defaultdict(set)
        self.records = {}  # entity_id -> key dict

    def add(self, rec):
        eid = rec['entity_id']
        self.records[eid] = rec
        for tok in rec['sig_name_tokens']:
            self.by_name_token[tok].add(eid)
        if rec['name_prefix']:
            self.by_name_prefix[rec['name_prefix']].add(eid)
        for tok in rec['unit_tokens']:
            self.by_unit_token[tok].add(eid)
        if rec['soundex']:
            self.by_soundex[rec['soundex']].add(eid)

    def candidates_for(self, rec, max_per_key=2000):
        """Union candidate ids across all key types. max_per_key guards against a
        pathological ultra-common key blowing up candidate count (e.g. a soundex
        bucket with 50k entries contributes nothing useful and is skipped)."""
        out = set()
        for tok in rec['sig_name_tokens']:
            bucket = self.by_name_token.get(tok)
            if bucket and len(bucket) <= max_per_key:
                out |= bucket
        if rec['name_prefix']:
            bucket = self.by_name_prefix.get(rec['name_prefix'])
            if bucket and len(bucket) <= max_per_key:
                out |= bucket
        for tok in rec['unit_tokens']:
            bucket = self.by_unit_token.get(tok)
            if bucket and len(bucket) <= max_per_key:
                out |= bucket
        if rec['soundex']:
            bucket = self.by_soundex.get(rec['soundex'])
            if bucket and len(bucket) <= max_per_key:
                out |= bucket
        return out


def build_index(df):
    """df: DataFrame with entity_id, business_name, business_address, country."""
    idx = BlockingIndex()
    for row in df.itertuples(index=False):
        rec = build_record_keys(row.entity_id, row.business_name, row.business_address, row.country)
        idx.add(rec)
    return idx


def generate_candidates(source1_df, index2, index3, max_per_key=2000):
    """For every Source 1 row, look up candidates in both indices.
    Yields (source1_entity_id, source1_rec, candidate_id_set)."""
    for row in source1_df.itertuples(index=False):
        rec = build_record_keys(row.entity_id, row.business_name, row.business_address, row.country)
        cands = index2.candidates_for(rec, max_per_key) | index3.candidates_for(rec, max_per_key)
        yield row.entity_id, rec, cands
