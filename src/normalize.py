"""
Text normalization for business names and addresses.

Handles:
 - Unicode/diacritic cleanup (NFKD strip)
 - Devanagari -> Latin phonetic transliteration (offline, rule-based library;
   no external API / lookup service involved)
 - Legal-suffix canonicalization (Pvt/Private, Ltd/Limited, Corp/Corporation, ...)
 - Case folding, punctuation stripping, common abbreviation expansion
 - Address token cleanup + extraction of alphanumeric unit/plot/shop identifiers
   (these turned out to be the strongest single blocking signal for India records
   in EDA: e.g. "Wz-187C Shop No.13" persists verbatim-ish across all 3 sources
   even when everything else about the address differs)
"""
import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

# Canonical legal-suffix mapping. Keys are regex word-boundary patterns (already
# lowercased), values are the canonical token we replace them with. Order matters:
# longer/more specific patterns first so e.g. "private limited" doesn't get
# double-mangled by a bare "ltd" rule.
_SUFFIX_MAP = [
    (r'\bprivate limited\b', 'pvt ltd'),
    (r'\bpvt\.? ltd\.?\b', 'pvt ltd'),
    (r'\bpvt\.?\b', 'pvt'),
    (r'\bprivate\b', 'pvt'),
    (r'\blimited\b', 'ltd'),
    (r'\bltd\.?\b', 'ltd'),
    (r'\bllp\.?\b', 'llp'),
    (r'\bl\.l\.p\.?\b', 'llp'),
    (r'\bcorporation\b', 'corp'),
    (r'\bcorp\.?\b', 'corp'),
    (r'\bincorporated\b', 'inc'),
    (r'\binc\.?\b', 'inc'),
    (r'\bl\.l\.c\.?\b', 'llc'),
    (r'\bllc\.?\b', 'llc'),
    (r'\bcompany\b', 'co'),
    (r'\bco\.?\b', 'co'),
    (r'\b&\b', 'and'),
    (r'\bm/s\.?\b', ''),  # Indian "Messrs" business-name prefix, adds no signal
]

_ADDR_ABBR_MAP = [
    (r'\brd\.?\b', 'road'),
    (r'\bst\.?\b', 'street'),
    (r'\bave\.?\b', 'avenue'),
    (r'\bdr\.?\b', 'drive'),
    (r'\bln\.?\b', 'lane'),
    (r'\bblvd\.?\b', 'boulevard'),
    (r'\bapt\.?\b', 'apartment'),
    (r'\bhno\.?\b', 'house no'),
    (r'\bh\.no\.?\b', 'house no'),
    (r'\bsno\.?\b', 'survey no'),
    (r'\bs\.no\.?\b', 'survey no'),
    (r'\bp\.?\s*no\.?\b', 'plot no'),
    (r'\bnr\.?\b', 'near'),
]

# OCR / typo confusable substitutions seen in the data (e.g. "Cardio1ogy").
# Applied only inside alpha runs, conservatively, as an extra similarity feature
# rather than a destructive rewrite (see features.py: ocr_normalize variant).
_OCR_CONFUSABLES = str.maketrans({'1': 'l', '0': 'o'})


def strip_diacritics(s: str) -> str:
    """NFKD-normalize and drop combining marks, e.g. 'Énterprises' -> 'Enterprises'."""
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def transliterate_devanagari(s: str) -> str:
    """Convert any Devanagari script substring to a Latin phonetic approximation.
    Offline, rule-based (indic_transliteration package) — not a lookup service.
    """
    if not DEVANAGARI_RE.search(s):
        return s

    def _conv(match):
        return transliterate(match.group(0), sanscript.DEVANAGARI, sanscript.ITRANS)

    # transliterate only the devanagari runs, leave any latin text untouched
    return DEVANAGARI_RE_RUN.sub(_conv, s)


DEVANAGARI_RE_RUN = re.compile(r'[\u0900-\u097F][\u0900-\u097F\s]*[\u0900-\u097F]|[\u0900-\u097F]')


def clean_punct(s: str) -> str:
    s = re.sub(r'[^\w\s&]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_name(raw: str) -> str:
    """Full name normalization pipeline. Returns lowercase, suffix-canonical,
    punctuation-stripped string."""
    if not raw:
        return ''
    s = raw
    s = transliterate_devanagari(s)
    s = strip_diacritics(s)
    s = s.lower()
    s = re.sub(r'^[\-\s]+', '', s)          # strip leading "-- " junk seen in source2
    s = clean_punct(s)
    for pattern, repl in _SUFFIX_MAP:
        s = re.sub(pattern, repl, s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_name_ocr(raw: str) -> str:
    """Variant with OCR-confusable substitution, used as an auxiliary feature
    (not the primary key) since it's a lossy transform."""
    return normalize_name(raw).translate(_OCR_CONFUSABLES)


def normalize_address(raw: str) -> str:
    """Full address normalization. Lowercase, diacritic-stripped, abbreviation-
    expanded, punctuation-stripped. Deliberately does NOT try to parse fixed
    city/state/PIN positions -- EDA showed component order varies too much
    across sources and countries (including unseen France in test) for that
    to be reliable."""
    if not raw:
        return ''
    s = raw
    s = transliterate_devanagari(s)
    s = strip_diacritics(s)
    s = s.lower()
    s = clean_punct(s)
    for pattern, repl in _ADDR_ABBR_MAP:
        s = re.sub(pattern, repl, s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


_UNIT_TOKEN_RE = re.compile(r'\b[a-z]{0,3}-?\d+[a-z]?\b')


def extract_unit_tokens(normalized_addr: str) -> set:
    """Pull out alphanumeric unit/plot/shop/house-number style tokens, e.g.
    'wz-187c', 'shop no 13', house/door numbers. These proved to be the
    strongest single blocking signal for India addresses in EDA: they persist
    near-verbatim across sources even when locality phrasing diverges wildly."""
    return set(_UNIT_TOKEN_RE.findall(normalized_addr))


def name_tokens(normalized_name: str) -> set:
    return set(t for t in normalized_name.split() if len(t) > 1)


def address_tokens(normalized_addr: str) -> set:
    return set(t for t in normalized_addr.split() if len(t) > 1)
