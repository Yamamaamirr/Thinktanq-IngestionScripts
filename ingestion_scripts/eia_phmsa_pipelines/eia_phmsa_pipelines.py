"""EIA gas pipeline ingestion.

Source: EIA Natural Gas Interstate and Intrastate Pipelines (USDOE/EIA),
        hosted on ArcGIS Online by Federal_User_Community.
        32,892 features covering interstate, intrastate, and gathering pipelines.

Diameter estimation — operator-name lookup (no PHMSA required):
  The EIA REST service does not expose diameter. PHMSA NPMS has exact diameters
  but requires government/operator credentials to access. Instead, diameter is
  estimated from operator name using a two-tier lookup table:

  Tier 1 — named trunk operators (30"+ assumed):
    Major long-haul interstate trunk lines whose diameter is publicly known
    to exceed 30". Matched by substring against the EIA Operator field.

  Tier 2 — named mid-tier operators (20" assumed):
    Mid-size interstate operators typically running 16–24" pipe.

  Fallback — TYPEPIPE + operator_tier:
    Interstate + tier_1  -> 24"   (unnamed major interstate corridor)
    Interstate + other   -> 14"   (smaller interstate operator)
    Intrastate + tier_1  -> 12"   (regional distribution, large operator)
    Intrastate + other   ->  8"   (local gathering / small distribution)

  This proxy provides ~80% of the real signal per Owen's guidance and unlocks
  the PRD On-Site-Generation-Feasibility scoring factor for all 32,971 segments.
  When exact PHMSA diameters are obtained (via midstream partner), replace
  diameter_in values for matched segments using a spatial join.

Pattern: same paginated-fetch-and-cache approach as hilfd_electricsubstations.
All records get feature_type='secondary' to distinguish from transmission lines
(feature_type='primary') in the linear_infra table.
"""
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError

BASE_URL = (
    'https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/'
    'Natural_Gas_Interstate_and_Intrastate_Pipelines_1/FeatureServer/0/query'
)
CACHE_FILE = Path(__file__).parent / 'raw_pipelines.parquet'
OUT_PARQUET = Path(__file__).parent / 'pipelines.parquet'
PAGE_SIZE = 2000

KEEP_COLS = ['FID', 'TYPEPIPE', 'Operator', 'Status', 'geometry']

# Keyword-based tier_1 classification — case-insensitive substring match against
# operator name. Covers major interstate operators plus known subsidiaries and
# rebrands (El Paso -> Kinder Morgan, Algonquin -> Enbridge/Spectra, etc.).
TIER_1_KEYWORDS = {
    'ENTERPRISE', 'WILLIAMS', 'KINDER MORGAN', 'EL PASO',
    'ENBRIDGE', 'ENERGY TRANSFER', 'TARGA', 'BOARDWALK',
    'SOUTHERN NATURAL', 'TRANSCONTINENTAL', 'TRANSWESTERN',
    'TENNESSEE GAS', 'COLUMBIA GAS', 'COLUMBIA GULF',
    'TEXAS EASTERN', 'PANHANDLE EASTERN', 'NORTHWEST PIPELINE',
    'QUESTAR', 'ANR', 'TRUNKLINE', 'ALGONQUIN',
    # Operators in TRUNK_30_OPERATORS not covered by the keywords above.
    # Without these they get diameter=30 but operator_tier='other', which
    # drops On-Site-Generation-Feasibility from partial (1.3x) to none (1.0x).
    'NORTHERN NATURAL', 'FLORIDA GAS', 'ROCKIES EXPRESS',
    'KERN RIVER', 'RUBY PIPELINE', 'MIDCONTINENT EXPRESS',
    'COMANCHE TRAIL', 'TRANS PECOS', 'SIERRITA', 'FLORIDA SOUTHEAST',
}

# ── Diameter estimation lookup (Owen's operator-proxy approach) ───────────────
# Tier 1 trunk operators — publicly known to run 30"+ mainline pipe.
# Strings are substrings matched case-insensitively against EIA Operator field.
TRUNK_30_OPERATORS = [
    'Transcontinental Gas PL',      # Transco — 30" mainline
    'Tennessee Gas Pipeline',
    'Columbia Gas Trans Co',
    'Texas Eastern Trans Co',
    'Algonquin Gas Trans Co',
    'Rockies Express Pipeline',     # REX — 42"
    'Northern Natural Gas Co',
    'El Paso Natural Gas Co',
    'El Paso Texas Pipeline Co',
    'Southern Natural Gas Co',
    'ANR Pipeline Co',
    'Panhandle Eastern PL Co',
    'Trunkline Gas Co',
    'Northwest Pipeline',
    'Kern River Gas Trans Co',
    'Ruby Pipeline LLC',
    'Midcontinent Express Pipeline',
    'Kinder Morgan Texas Pipeline Co',
    'Columbia Gulf Pipeline',
    'Permian Highway Pipeline',     # 42"
    'Whistler Pipeline',            # 42"
    'Gulf Coast Express',           # 42"
    'Trans Pecos Pipeline',         # 42"
    'Comanche Trail Pipeline',
    'Sierrita Gas Pipeline',        # 36"
    'Florida Gas Trans Co',
    'Florida Southeast Connection',
]

# Mid-tier interstate operators — typically 16–24" pipe.
MID_20_OPERATORS = [
    'Equitrans Inc',
    'Dominion Transmission Co',
    'Southern Star Central Gas PL Co',
    'Questar Pipeline Co',
    'Wyoming Interstate PL Co',
    'Great Lakes Gas Trans Co',
    'Iroquois Gas Trans Co',
    'Empire Pipeline',
]

RETRY_DELAYS = [5, 10, 20, 30, 60]
NETWORK_ERRORS = (ConnectionError, Timeout, ChunkedEncodingError, ProtocolError)


def _fetch_page(offset):
    params = {
        'where': "Status='Operating'",
        'outFields': 'FID,TYPEPIPE,Operator,Status',
        'resultOffset': offset,
        'resultRecordCount': PAGE_SIZE,
        'f': 'geojson',
    }
    last_exc = None
    for delay in RETRY_DELAYS:
        try:
            r = requests.get(BASE_URL, params=params, timeout=60)
            r.raise_for_status()
            return r.json().get('features', [])
        except NETWORK_ERRORS as e:
            last_exc = e
            print(f"  Network error: {type(e).__name__} — retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"Max retries exceeded ({last_exc!r})")


def _assign_operator_tier(name):
    if not name:
        return 'other'
    upper = name.upper()
    for kw in TIER_1_KEYWORDS:
        if kw in upper:
            return 'tier_1'
    return 'other'


def _estimate_diameter(operator, typepipe, tier):
    """Estimate pipe diameter in inches from operator name + pipeline type.

    Priority: named trunk list -> named mid-tier list -> TYPEPIPE/tier fallback.
    Fallback tiers reflect typical industry diameter ranges for each category.
    """
    if operator:
        op_upper = operator.upper()
        for t in TRUNK_30_OPERATORS:
            if t.upper() in op_upper or op_upper in t.upper():
                return 30
        for m in MID_20_OPERATORS:
            if m.upper() in op_upper or op_upper in m.upper():
                return 20
    if typepipe == 'Interstate' and tier == 'tier_1':
        return 24
    if typepipe == 'Interstate':
        return 14
    if tier == 'tier_1':
        return 12
    return 8


if CACHE_FILE.exists():
    print("Loading from cache...")
    gdf = gpd.read_parquet(CACHE_FILE)
else:
    print("Fetching from EIA ArcGIS service...")
    all_features = []
    offset = 0
    while True:
        feats = _fetch_page(offset)
        if not feats:
            break
        all_features.extend(feats)
        offset += len(feats)
        print(f"  Fetched {offset:,} features...")
        if len(feats) < PAGE_SIZE:
            break
    gdf = gpd.GeoDataFrame.from_features(all_features, crs='EPSG:4326')
    gdf.to_parquet(CACHE_FILE)
    print(f"Cached {len(gdf):,} rows -> {CACHE_FILE}")

# ── Parquet verification ──────────────────────────────────────────────────────
print(f"\nRaw rows in cache: {len(gdf):,}")
print(f"Raw cache columns: {list(gdf.columns)}")
print(f"Raw cache dtypes:\n{gdf.dtypes}")
print(f"Raw geometry types:\n{gdf.geom_type.value_counts(dropna=False)}")
assert gdf['geometry'].notna().sum() > 30_000, "Unexpectedly few non-null geometries in cache"
assert len(gdf) > 30_000, f"Expected 30k+ rows in cache, got {len(gdf)}"
print("Parquet verification passed.")
# ─────────────────────────────────────────────────────────────────────────────

gdf = gdf[KEEP_COLS].copy()
gdf = gdf.dropna(subset='geometry')
gdf = gdf.to_crs(epsg=4326)

gdf['source_id']      = gdf['FID'].astype(str)
gdf['name']           = gdf['Operator']
gdf['pipeline_type']  = gdf['TYPEPIPE']   # Interstate / Intrastate -- kept for reference
gdf['feature_type']   = 'secondary'
gdf['class_rating']   = 'sub_class'
gdf['operator_tier']  = gdf['Operator'].apply(_assign_operator_tier)
gdf['diameter_in']        = gdf.apply(
    lambda r: _estimate_diameter(r['Operator'], r['TYPEPIPE'], r['operator_tier']),
    axis=1,
).astype(float)
# All diameters are operator-name proxies, not verified PHMSA/NPMS values.
# Per Owen: treat as first-pass capacity class, not a hard eligibility filter.
# Set to False for individual segments once verified via operator maps or FERC/PHMSA records.
gdf['diameter_estimated'] = True
gdf['service_type']       = 'unknown'   # not in EIA data; PHMSA NPMS has this

print(f"\nAfter geometry filter: {len(gdf):,}")

# ── MultiLineString -> LineString explode ─────────────────────────────────────
# PostGIS linear_infra schema requires GEOGRAPHY(LINESTRING, 4326). The EIA data
# contains 85 MultiLineString features (disconnected pipeline segments stored as
# one record). explode() splits each MultiLineString into individual LineStrings,
# inheriting all attribute columns. Row count increases by the number of extra
# component linestrings released from MultiLineStrings.
n_before = len(gdf)
multi_mask = gdf.geom_type == 'MultiLineString'
print(f"\nMultiLineString features before explode: {multi_mask.sum()}")

gdf = gdf.explode(index_parts=False).reset_index(drop=True)

n_added = len(gdf) - n_before
print(f"After explode: {len(gdf):,} rows (+{n_added} component linestrings released)")
print(f"Geometry types after explode:\n{gdf.geom_type.value_counts()}")

non_ls = gdf[gdf.geom_type != 'LineString']
assert len(non_ls) == 0, (
    f"Non-LineString geometries remain after explode: "
    f"{non_ls.geom_type.value_counts().to_dict()}"
)
print("Geometry assertion passed: all features are LineString.")
# ─────────────────────────────────────────────────────────────────────────────

now = pd.Timestamp.now(tz='UTC')
gdf['ingested_at']    = now
gdf['last_updated_at'] = now

print(f"\nCRS: EPSG:4326")
print(f"\nTYPEPIPE breakdown:\n{gdf['TYPEPIPE'].value_counts(dropna=False)}")
print(f"\noperator_tier breakdown:\n{gdf['operator_tier'].value_counts(dropna=False)}")
print(f"\nTop tier_1 operators (by feature count):")
tier1_ops = gdf[gdf['operator_tier'] == 'tier_1']['Operator'].value_counts().head(15)
print(tier1_ops.to_string())
print(f"\ndiameter_in distribution:\n{gdf['diameter_in'].value_counts().sort_index(ascending=False).to_string()}")
print(f"Segments >= 20\": {(gdf['diameter_in'] >= 20).sum():,} / {len(gdf):,} ({100*(gdf['diameter_in'] >= 20).mean():.1f}%)")
print(f"\ndiameter_estimated: {gdf['diameter_estimated'].sum():,} / {len(gdf):,} rows "
      f"(all proxy -- verified=0 until PHMSA/operator data replaces estimates)")
print(f"\nNull counts:\n{gdf[['source_id','name','pipeline_type','operator_tier','diameter_in','service_type']].isnull().sum()}")

# ── Output ───────────────────────────────────────────────────────────────────
keep_out = ['source_id', 'name', 'pipeline_type', 'feature_type', 'class_rating',
            'operator_tier', 'diameter_in', 'diameter_estimated', 'service_type',
            'geometry', 'ingested_at', 'last_updated_at']
gdf = gdf[keep_out]
gdf.to_parquet(OUT_PARQUET, index=False)
print(f"\nWrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e6:.1f} MB)")
sample = gdf.head(20).copy()
sample.drop(columns='geometry').to_csv(Path(__file__).parent / 'sample_rows.csv', index=False)
sample.to_file(Path(__file__).parent / 'sample_rows.geojson', driver='GeoJSON')
print("Exported: sample_rows.csv and sample_rows.geojson")
