"""ISO queue substation_name -> HIFLD anchors matcher.

Match passes per queue row:
  1. Pre-filter HIFLD substations to the same state (kills 99% of false-positive
     candidates and makes the fuzzy step ~50x faster).
  2. Normalize both names via the same rule chain (strip utility/ISO prefixes,
     voltage tokens, parentheticals, common suffix words like SUBSTATION/SUB/
     STATION/SWITCHYARD/LINE, punctuation, collapse whitespace).
  3. Pass A: exact match on the normalized form -> match_method = 'exact'.
  4. Pass B: rapidfuzz token_set_ratio over remaining candidates.
     >=90 -> 'fuzzy_high'   (auto-accept, very high precision)
     85-89 -> 'fuzzy_medium' (accept, manual review recommended for top sites)
     70-84 -> 'fuzzy_low'   (write but flag — analyst should verify)
     <70  -> 'unmatched'    (no anchor recorded)
  5. Pass C: endpoint-split on hyphenated/arrow names (e.g. "Helm-Kerman").
  6. Tie-breaking: highest score wins. Ties broken by higher token_sort_ratio,
     then by shorter normalized-name length (more specific match preferred).

Rows that reach 'unmatched' are excluded from anchor_queue_stats entirely.
New substations not yet in HIFLD (e.g. post-2022 ERCOT builds) will be
unmatched — that is correct and preferable to a county-level geographic proxy.

Why no geographic verification: the gridstatus library does not expose POI
coordinates from any of the queue feeds. When we get coordinates from a future
direct-from-ISO ingestion, the right architecture is to require BOTH the name
match AND the geographic match (anchor within ~10mi of POI) before accepting.
"""
import re

import pandas as pd
from rapidfuzz import fuzz, process

# Match score thresholds -- adjust based on validation results.
THRESHOLD_HIGH = 90
THRESHOLD_MEDIUM = 85
THRESHOLD_LOW = 70

# Policy: fuzzy_low matches are NOT auto-accepted because validation showed
# ~75% false positive rate in that tier. fuzzy_low rows are recorded with
# anchor_id=None (so they're excluded from enrichment aggregation by default)
# but anchor_name is still populated with the candidate so analysts can review
# and promote individual matches via an override workflow.
# Flip to True only if you want fuzzy_low to count toward scoring without
# manual review -- not recommended for production.
AUTO_ACCEPT_FUZZY_LOW = False

# Explicit wrong-pair blocklist: (normalized_query_fragment, normalized_anchor_fragment)
# If BOTH fragments appear in their respective normalized names, the match is rejected
# even if the fuzzy score is above threshold. Catches directional antonym mismatches
# (North ≠ South, Creek ≠ Reef) that token_set_ratio can't distinguish.
MATCH_BLOCKLIST = [
    # N. Lebanon PA queue entry → South Lebanon Substation PA (opposite directions)
    ('LEBANON', 'SOUTH LEBANON'),
    # Granite Creek AZ → Granite Reef AZ (different geographic features, different locations)
    ('GRANITE CREEK', 'GRANITE REEF'),
]


def _is_blocklisted(query_norm: str, anchor_norm: str) -> bool:
    """Return True if this query→anchor pair is a known wrong match."""
    q, a = query_norm.upper(), anchor_norm.upper()
    for q_frag, a_frag in MATCH_BLOCKLIST:
        if q_frag in q and a_frag in a:
            return True
    return False

# Leading utility / ISO routing prefixes that aren't part of the substation name.
# Order matters - longer prefixes first so partial matches don't truncate.
PREFIX_PATTERNS = [
    r'^NYISO\s+',
    r'^NGRID\s+',
    r'^RIE\s+FEEDER\s+\S+,?\s*',
    r'^NSP\s+',
    r'^CONED\s+',
    r'^PSEG\s+',
    r'^PG&?E\s+',      # PG&E or PGE
    r'^SCE\s+',
    r'^SDGE\s+',
    r'^APS\s+',        # Arizona Public Service prefix (AZ queue feeds)
    r'^AEP\s+\d*\s*', # AEP utility code + optional bus number (ERCOT/MISO)
    r'^AEP\s+',
    r'^PROPOSED\s+',
    r'^NEW\s+',
    # ERCOT GIS Report leading bus numbers: "59903 Bearkat 345kV" -> "Bearkat"
    # Also handles "#79501 Kingfisher" style with hash prefix
    r'^#?\d{4,6}\s+',
    # ERCOT "tap NkV NNNNN Name" prefix pattern: "tap 345kV 8905 Name"
    r'^TAP\s+\d+\s*K?V\s+\d+\s+',
    r'^TAP\s+\d+\s*K?V\s+',
    r'^TAP\s+',
    # ERCOT "Bus # NNNN, Location" — bus address tag, not a substation name
    r'^BUS\s*#?\s*\d+\s*,?\s*',
    # Embedded utility operator codes (ERCOT/MISO): SHECO, LCRA, AEP, NWP
    r'\bSHECO\s+',
    r'\bLCRA\s+',
    r'\b\(AEP\)',
    r'\bNWP\b',
    # XML encoding artifacts from ERCOT data export
    r'_X[0-9A-F]{4}_',
]

# Voltage tokens to strip. Catches "230 kV", "230kV", "13.8KV", trailing "345".
# IMPORTANT: fraction voltages (500/230kV) must come BEFORE single-number patterns
# so "138/230 kV" is consumed as a unit before "230 kV" gets matched alone.
VOLTAGE_PATTERNS = [
    r'\b\d+/\d+(\.\d+)?\s*K?V?\b',     # "500/230kV", "138/230 kV" — fraction first
    r'\b\d+(\.\d+)?\s*-\s*K?V\b',      # "345- kV" (malformed dash before unit)
    r'\b\d+(\.\d+)?\s*K?V\b',          # "230 kV", "13.8KV", "230KV"
    r'\b\d{3,4}\b(?=\s*(?:LINE|BUS|$))',  # trailing voltage like "CAYUGA 345"
    r'\b\d+(\.\d+)?KV\b',              # compact form "345KV" without space
]

# Suffix tokens that describe the asset type, not its name.
# IMPORTANT: longer/multi-word phrases must come BEFORE shorter ones so
# "POWER PLANT" is consumed as a single phrase before "PLANT" alone matches.
SUFFIX_PATTERNS = [
    r'\bPOWER\s+PLANT\b',
    r'\bGENERATING\s+STATION\b',
    r'\bGEN\s+STATION\b',
    r'\bSWITCHING\s+STATION\b',
    r'\bSUBSTATION\b',
    r'\bSWITCHYARD\b',
    r'\bSTATION\b',
    r'\bSUB\b',
    r'\bYARD\b',
    r'\bPLANT\b',
    r'\bLINE\b',
    r'\bTAP\b',
    r'\bBUS\s+\w+\b',                  # "BUS E", "BUS 1"
    r'\bT[123]\b',                     # transformer number "T3"
    r'\bUNIT\s+\d+\b',
    r'\b#\d+\b',
    r'\bSS\b',                         # "SS" = SubStation (ERCOT/MISO shorthand)
    r'\bDP\b',                         # "DP" = Distribution Point (PJM/VA)
    r'\bSTREET\b',                     # street addresses (ISONE: "Meadow Street 552")
    r'\bAVENUE\b',
    r'\bROAD\b',
    r'\b\d{2,4}\b',                    # bare numbers after voltage strip (bus IDs, section numbers)
    r'\bSWITCH\b',                     # "SWITCH" standalone
    r'\bII\b',                         # "POSSUM POINT II" -> "POSSUM POINT"
    r'\bNO\s*\.\s*\d+\b',             # "NO. 2", "NO.1"
]


def _strip_prefixes(s: str) -> str:
    for pat in PREFIX_PATTERNS:
        s = re.sub(pat, '', s, flags=re.IGNORECASE)
    return s


def normalize_name(raw):
    """Apply the full normalization chain to a substation name string.

    Order matters: prefixes are stripped BEFORE comma-clauses because some ISO
    feeds put the substation name AFTER a comma (e.g. "RIE feeder 26W7,
    Woonsocket Substation"). After the prefix is stripped the comma is gone.
    For names where the part after the comma is just metadata (e.g. county,
    state) the substation name is already on the left side and unaffected.
    """
    if raw is None or pd.isna(raw):
        return ''
    s = str(raw).upper()
    s = re.sub(r'\([^)]*\)', '', s)                      # strip parentheticals
    s = _strip_prefixes(s)
    s = re.sub(r',.*$', '', s)                           # strip trailing comma clauses
    for pat in VOLTAGE_PATTERNS:
        s = re.sub(pat, '', s)
    # Second prefix pass: ERCOT names like "138kV 11273 NELSON" expose the bus
    # number only after the voltage token is removed. Collapse whitespace first
    # so the leading-digit pattern fires on the now-exposed bus number.
    s = re.sub(r'\s+', ' ', s).strip()
    s = _strip_prefixes(s)
    for pat in SUFFIX_PATTERNS:
        s = re.sub(pat, '', s)
    s = re.sub(r"[^\w\s\-]", ' ', s)                     # punctuation -> space, keep dashes
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _is_line_poi(raw_name: str) -> bool:
    """True if a queue substation_name looks like a transmission line POI.

    Line POIs ("Helm-Kerman 70 kV Line", "A-B 345 kV") should not be matched
    as if they were single substations — they name two endpoints of a line tap.
    When detected, match_one skips Pass B (full-name fuzzy) and goes directly
    to Pass C (endpoint-split), which tries each half separately.

    True substations with explicit keywords ("Panoche Substation", "Birds
    Landing Switchyard") are excluded even if they contain a hyphen.
    """
    if not raw_name or pd.isna(raw_name):
        return False
    s = str(raw_name)
    s_upper = s.upper()
    # A name explicitly calling itself a substation/switchyard is NOT a line POI.
    if re.search(r'\b(SUBSTATION|SWITCHYARD)\b', s_upper):
        return False
    # Strip parentheticals before checking for LINE keyword.
    s_no_paren = re.sub(r'\([^)]*\)', '', s)
    if re.search(r'\bLINE\b', s_no_paren, re.IGNORECASE):
        return True
    # "NameA - NameB" or "NameA TO NameB" with ≥4-char tokens on each side.
    # Strip voltage tokens first so "A 345kV - B 230kV" reduces to "A - B".
    s_no_volt = s_no_paren
    for pat in VOLTAGE_PATTERNS:
        s_no_volt = re.sub(pat, '', s_no_volt)
    if re.search(r'\b([A-Za-z]{4,})\s*[-–]\s*([A-Za-z]{4,})\b', s_no_volt):
        return True
    if re.search(r'\b([A-Za-z]{4,})\s+TO\s+([A-Za-z]{4,})\b', s_no_volt, re.IGNORECASE):
        return True
    return False


def _extract_kv(raw_name: str) -> 'float | None':
    """Extract the first kV value from a queue substation name.

    "Birds Landing 230 kV" -> 230.0
    "Helm-Kerman 70 kV Line" -> 70.0
    "Panoche Substation" -> None
    """
    if not raw_name or pd.isna(raw_name):
        return None
    m = re.search(r'\b(\d+(?:\.\d+)?)\s*k[vV]\b', str(raw_name), re.IGNORECASE)
    return float(m.group(1)) if m else None


def build_anchor_index(anchors_df, name_col='NAME', state_col='STATE', volt_col=None):
    """Pre-normalize all anchor names per state for fast lookup.

    Returns dict: {state: DataFrame[name_norm, name_raw, anchor_id, max_volt]}.
    Drops anchors with placeholder names (UNKNOWN*) — they can never match.

    volt_col: optional column in anchors_df containing max voltage in kV.
              Included in the index so match_one can surface voltage info.
    """
    keep = [name_col, state_col]
    if volt_col and volt_col in anchors_df.columns:
        keep.append(volt_col)
    df = anchors_df[keep].copy()
    col_names = ['name_raw', 'state']
    if volt_col and volt_col in anchors_df.columns:
        col_names.append('max_volt')
    df.columns = col_names
    if 'max_volt' not in df.columns:
        df['max_volt'] = None
    df = df.dropna(subset=['name_raw', 'state'])
    df = df[~df['name_raw'].str.startswith('UNKNOWN', na=False)]
    df['anchor_id'] = df.index.astype(str)
    df['name_norm'] = df['name_raw'].apply(normalize_name)
    df = df[df['name_norm'].str.len() > 0]
    return {state: g.reset_index(drop=True) for state, g in df.groupby('state')}


# Tokens too short or generic to count as meaningful overlap evidence.
_OVERLAP_STOPWORDS = {
    'THE', 'AND', 'FOR', 'NEW', 'OLD', 'EAST', 'WEST', 'NORTH', 'SOUTH',
    'UNIT', 'LINE', 'AREA', 'ROAD', 'PARK', 'LAKE', 'CITY',
}

def _meaningful_tokens(norm: str) -> set:
    """Return tokens ≥4 chars that aren't generic direction/article words."""
    return {t for t in norm.split() if len(t) >= 4 and t not in _OVERLAP_STOPWORDS}


def _has_meaningful_overlap(query_norm: str, anchor_norm: str) -> bool:
    """True if at least one meaningful token is shared between both names.

    Prevents token_set_ratio false positives where a short generic token
    (e.g. 'PGE', 'OAK', 'AMO') from one endpoint fires a perfect score
    against a completely different substation that happens to contain that word.
    """
    return bool(_meaningful_tokens(query_norm) & _meaningful_tokens(anchor_norm))


def _best_fuzzy(query_norm, candidates):
    """Return (row, score) for the best fuzzy match, or (None, score) if none."""
    if not query_norm:
        return None, 0
    choices = candidates['name_norm'].tolist()
    best = process.extractOne(query_norm, choices, scorer=fuzz.token_set_ratio)
    if best is None:
        return None, 0
    matched_norm, score, idx = best
    return candidates.iloc[idx], score


def match_one(query_name, query_state, anchor_index):
    """Match a single queue substation_name to an anchor.

    Returns dict with anchor_id, anchor_name, score, method, poi_type, query_kv.

    poi_type:  'substation' | 'line_poi' | 'unknown'
    query_kv:  kV extracted from the raw queue name (None if not present)

    Line POI detection (Owen): queue entries like "Helm-Kerman 70 kV Line" are
    transmission-line tap points, not single substations. For these we skip Pass
    B (full-name fuzzy) which would false-match the compound "A-B" string against
    either endpoint alone. Pass C (endpoint-split) handles them correctly.
    """
    query_kv = _extract_kv(query_name)
    is_line   = _is_line_poi(query_name)
    poi_type  = 'line_poi' if is_line else 'unknown'

    if not query_state or pd.isna(query_state) or query_state not in anchor_index:
        return {'anchor_id': None, 'anchor_name': None, 'score': 0,
                'method': 'no_state_filter', 'poi_type': poi_type, 'query_kv': query_kv}

    query_norm = normalize_name(query_name)
    if not query_norm:
        return {'anchor_id': None, 'anchor_name': None, 'score': 0,
                'method': 'empty_after_norm', 'poi_type': poi_type, 'query_kv': query_kv}

    candidates = anchor_index[query_state]

    # Pass A: exact match on the normalized form (always attempted, even for line POIs).
    exact = candidates[candidates['name_norm'] == query_norm]
    if len(exact) > 0:
        row = exact.iloc[0]
        return {'anchor_id': row['anchor_id'], 'anchor_name': row['name_raw'],
                'score': 100, 'method': 'exact', 'poi_type': poi_type, 'query_kv': query_kv}

    # Pass B: fuzzy match on the full normalized name.
    # Skipped for line POIs: token_set_ratio false-positives on "A-B" compound
    # names because it ignores word order and can match just one endpoint.
    if not is_line:
        row, score = _best_fuzzy(query_norm, candidates)
        if (score >= THRESHOLD_HIGH
                and _has_meaningful_overlap(query_norm, row['name_norm'])
                and not _is_blocklisted(query_norm, row['name_norm'])):
            return {'anchor_id': row['anchor_id'], 'anchor_name': row['name_raw'],
                    'score': score, 'method': 'fuzzy_high',
                    'poi_type': poi_type, 'query_kv': query_kv}
    else:
        row, score = None, 0  # skip; needed for Pass D fallback below

    # Pass C: endpoint-split fallback for line-tap names.
    # Many ISOs name interconnections as "EndpointA - EndpointB" or
    # "EndpointA to EndpointB". Try each half separately and take the best.
    separators = ['-', ' TO ']
    endpoints = []
    for sep in separators:
        if sep in query_norm:
            for half in query_norm.split(sep):
                half = half.strip()
                if len(half) >= 3:
                    endpoints.append(half)
    if endpoints:
        best_endpoint = (None, 0, '')
        for half in endpoints:
            r, s = _best_fuzzy(half, candidates)
            if s > best_endpoint[1]:
                best_endpoint = (r, s, half)
        r, s, half = best_endpoint
        if (s >= THRESHOLD_HIGH and r is not None
                and _has_meaningful_overlap(half, r['name_norm'])
                and not _is_blocklisted(half, r['name_norm'])):
            return {'anchor_id': r['anchor_id'], 'anchor_name': r['name_raw'],
                    'score': s, 'method': 'fuzzy_high_endpoint',
                    'poi_type': poi_type, 'query_kv': query_kv}

    # Pass D: lower-confidence buckets on the original full match (non-line-POI only).
    if not is_line and row is not None:
        if (score >= THRESHOLD_MEDIUM
                and _has_meaningful_overlap(query_norm, row['name_norm'])
                and not _is_blocklisted(query_norm, row['name_norm'])):
            return {'anchor_id': row['anchor_id'], 'anchor_name': row['name_raw'],
                    'score': score, 'method': 'fuzzy_medium',
                    'poi_type': poi_type, 'query_kv': query_kv}
        if score >= THRESHOLD_LOW:
            # fuzzy_low: do NOT auto-accept. anchor_id=None so it's excluded from
            # enrichment aggregation. anchor_name still surfaces the candidate so
            # the analyst override workflow can review and promote individual rows.
            accepted_id = row['anchor_id'] if AUTO_ACCEPT_FUZZY_LOW else None
            return {'anchor_id': accepted_id, 'anchor_name': row['name_raw'],
                    'score': score, 'method': 'fuzzy_low',
                    'poi_type': poi_type, 'query_kv': query_kv}
    return {'anchor_id': None, 'anchor_name': None, 'score': score,
            'method': 'unmatched', 'poi_type': poi_type, 'query_kv': query_kv}


def match_all(queue_df, anchor_index, name_col='substation_name', state_col='state'):
    """Match every queue row to an anchor. Returns queue_df with 6 new columns.

    Added vs original: poi_type ('substation'|'line_poi'|'unknown') and
    query_kv (kV extracted from raw substation name, or None).
    """
    out = queue_df.copy()
    results = [
        match_one(r[name_col], r[state_col], anchor_index)
        for _, r in queue_df[[name_col, state_col]].iterrows()
    ]
    out['anchor_id']    = [r['anchor_id']   for r in results]
    out['anchor_name']  = [r['anchor_name'] for r in results]
    out['match_score']  = [r['score']       for r in results]
    out['match_method'] = [r['method']      for r in results]
    out['poi_type']     = [r['poi_type']    for r in results]
    out['query_kv']     = [r['query_kv']    for r in results]
    return out
