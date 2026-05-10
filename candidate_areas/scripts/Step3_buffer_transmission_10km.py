"""
Step 3 — Buffer 230kV+ transmission lines at 10 km for target states.

Reads:  ingestion_scripts/hifld_transmission_lines/hifld_transmission_lines.parquet
        Census TIGER 2025 state boundaries (downloaded + cached on first run)
Writes: ingestion_scripts/hifld_transmission_lines/transmission_lines_345kv_target.parquet  (lines, QGIS verification)
        ingestion_scripts/hifld_transmission_lines/transmission_buffers_345kv_plus.parquet
        ingestion_scripts/hifld_transmission_lines/transmission_buffers_230kv.parquet
        ingestion_scripts/hifld_transmission_lines/transmission_buffers_10km.parquet  (combined)
        ingestion_scripts/census_tiger/cache/tl_2025_us_state.zip  (cache)
        ingestion_scripts/census_tiger/state_boundaries.parquet     (reusable)

Run from project root:  python candidate_areas/Step3_buffer_transmission_10km.py

Voltage tier treatment (Owen Rec #6):
  Tier 1 — class_1 (500kV+ / DC):    voltage_tier=1, transmission_signal_score=100
             Highest signal. Regional backbone.
  Tier 2 — class_2 (345kV):           voltage_tier=2, transmission_signal_score=69
             Strong signal. High-capacity sub-regional.
  Tier 3 — class_3 (220-287kV):       voltage_tier=3, transmission_signal_score=46
             Secondary signal. Sub-regional / metro feeds.
  Excluded — sub_class (100-161kV):   Not buffered. Scoring attribute only.

  transmission_signal_score is normalized from PRD class_weight_mapping
  (max weight 500): class_1=100, class_2=round(345/500*100)=69, class_3=round(230/500*100)=46.

Separate output files (Owen Rec #6 — keep-zone protection):
  transmission_buffers_345kv_plus.parquet — Tier 1+2 (primary) ONLY.
    This is the file used by Step4_build_keep_zones.py for the keep-zone union.
    Only 345kV+ lines anchor the search zone.

  transmission_buffers_230kv.parquet — Tier 3 (secondary) ONLY.
    NOT used in the keep-zone union. Used in scoring only
    (infrastructure_proximity_score in Step9). 230kV proximity is a scoring
    signal, not a keep-zone expansion trigger.

  transmission_buffers_10km.parquet — Combined convenience file (all tiers).
    Useful for QGIS visualization and ad-hoc analysis. Not consumed by any
    pipeline step directly.

Buffer radius: 10 km (10,000 m) along line geometry — produces corridor polygons.
Projection:    EPSG:5070 (NAD83 / Conus Albers) for accurate metric buffering.
"""

import geopandas as gpd
import requests
from pathlib import Path

TARGET_STATES = ['CA', 'TX', 'AZ', 'NV', 'VA']
STATE_FIPS = {'CA': '06', 'TX': '48', 'AZ': '04', 'NV': '32', 'VA': '51'}

BUFFER_M      = 10_000
TIGER_URL     = 'https://www2.census.gov/geo/tiger/TIGER2025/STATE/tl_2025_us_state.zip'
CACHE_DIR     = Path('ingestion_scripts/census_tiger/cache')
STATE_ZIP     = CACHE_DIR / 'tl_2025_us_state.zip'
STATE_PARQUET = Path('ingestion_scripts/census_tiger/state_boundaries.parquet')
TX_SRC        = Path('ingestion_scripts/hifld_transmission_lines/hifld_transmission_lines.parquet')

OUT_LINES    = Path('ingestion_scripts/hifld_transmission_lines/transmission_lines_345kv_target.parquet')
OUT_PRIMARY  = Path('ingestion_scripts/hifld_transmission_lines/transmission_buffers_345kv_plus.parquet')
OUT_230      = Path('ingestion_scripts/hifld_transmission_lines/transmission_buffers_230kv.parquet')
OUT_COMBINED = Path('ingestion_scripts/hifld_transmission_lines/transmission_buffers_10km.parquet')

# PRD-derived voltage tier and score (normalized to 0-100 against max weight 500)
VOLTAGE_TIER  = {'class_1': 1, 'class_2': 2, 'class_3': 3}
TX_WEIGHT     = {'class_1': 500, 'class_2': 345, 'class_3': 230}
TX_SCORE      = {'class_1': 100, 'class_2': 69, 'class_3': 46}   # round(weight/500*100)
SIGNAL_TIER   = {'class_1': 'primary', 'class_2': 'primary', 'class_3': 'secondary'}

KEEP_COLS = ['voltage_kv', 'volt_class', 'class_rating', 'voltage_tier',
             'signal_tier', 'tx_weight', 'transmission_signal_score',
             'owner', 'substation_from', 'substation_to', 'state_code', 'geometry']

# ── Step 1: Download state boundaries (cached) ────────────────────────────────

CACHE_DIR.mkdir(parents=True, exist_ok=True)

if STATE_ZIP.exists():
    print(f'State boundary zip already cached: {STATE_ZIP}')
else:
    print('Downloading Census TIGER state boundaries ...')
    r = requests.get(TIGER_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(STATE_ZIP, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    print(f'  Downloaded: {STATE_ZIP.stat().st_size / 1e6:.1f} MB')

# ── Step 2: Load + filter to target states ────────────────────────────────────

print('Loading state boundaries ...')
states = gpd.read_file(STATE_ZIP)
states = states.to_crs(epsg=4326)

target = states[states['STUSPS'].isin(TARGET_STATES)][['STUSPS', 'NAME', 'geometry']].copy()
target = target.rename(columns={'STUSPS': 'state_code', 'NAME': 'state_name'})

print(f'  Target states loaded: {list(target["state_code"])}')

target.to_parquet(STATE_PARQUET, index=False)
print(f'  Saved: {STATE_PARQUET}')

target_dissolved = target.dissolve()

# ── Step 3: Load transmission lines, filter 220kV+ ───────────────────────────

print('Loading transmission lines ...')
tx = gpd.read_parquet(TX_SRC)
print(f'  Full dataset: {len(tx):,} segments')

tx_hv = tx[tx['class_rating'].isin(['class_1', 'class_2', 'class_3'])].copy()
print(f'  After 220kV+ filter: {len(tx_hv):,} segments')

# ── Step 4: Spatial filter to target states ───────────────────────────────────

print('Spatially filtering to target states ...')
tx_hv = tx_hv.to_crs(epsg=4326)
tx_target = tx_hv[tx_hv.intersects(target_dissolved.geometry.iloc[0])].copy()
print(f'  After spatial filter: {len(tx_target):,} segments')

tx_target = tx_target.sjoin(
    target[['state_code', 'geometry']],
    how='left',
    predicate='intersects'
)
tx_target = tx_target.drop_duplicates(subset=tx_target.columns.difference(['state_code', 'index_right']))
tx_target = tx_target.drop(columns=['index_right'], errors='ignore')

print('  Segments per state:')
for state in TARGET_STATES:
    n = (tx_target['state_code'] == state).sum()
    print(f'    {state}: {n:,}')

# ── Step 5a: Save target-state LINE geometry (for QGIS buffer verification) ───

line_cols = ['voltage_kv', 'volt_class', 'class_rating', 'owner',
             'substation_from', 'substation_to', 'state_code', 'geometry']
lines_345 = tx_target[tx_target['class_rating'].isin(['class_1', 'class_2'])].copy()
lines_345 = lines_345[[c for c in line_cols if c in lines_345.columns]]
lines_345.to_parquet(OUT_LINES, index=False)
print(f'Saved: {OUT_LINES}  ({len(lines_345):,} rows — LINE geometry, 345kV+ target states)')

# ── Step 5b: Tag tier fields, project, buffer ─────────────────────────────────

tx_target['voltage_tier']             = tx_target['class_rating'].map(VOLTAGE_TIER)
tx_target['signal_tier']              = tx_target['class_rating'].map(SIGNAL_TIER)
tx_target['tx_weight']                = tx_target['class_rating'].map(TX_WEIGHT)
tx_target['transmission_signal_score'] = tx_target['class_rating'].map(TX_SCORE)

print(f'Projecting to EPSG:5070 and buffering {BUFFER_M/1000:.0f} km ...')
tx_proj = tx_target.to_crs(epsg=5070)
tx_proj['geometry'] = tx_proj.geometry.buffer(BUFFER_M)

out_df = tx_proj[[c for c in KEEP_COLS if c in tx_proj.columns]]

# ── Step 6: Write three output files ──────────────────────────────────────────

primary  = out_df[out_df['signal_tier'] == 'primary'].copy()
secondary = out_df[out_df['signal_tier'] == 'secondary'].copy()

primary.to_parquet(OUT_PRIMARY, index=False)
secondary.to_parquet(OUT_230, index=False)
out_df.to_parquet(OUT_COMBINED, index=False)

print(f'\nSaved: {OUT_PRIMARY}  ({len(primary):,} rows — Tier 1+2, keep-zone input)')
print(f'Saved: {OUT_230}  ({len(secondary):,} rows — Tier 3, scoring only)')
print(f'Saved: {OUT_COMBINED}  ({len(out_df):,} rows — combined, QGIS/analysis only)')

# ── Step 7: Verify ────────────────────────────────────────────────────────────

for label, path, expected_tiers in [
    ('345kV+ (primary)', OUT_PRIMARY,  {'primary'}),
    ('230kV (secondary)', OUT_230,     {'secondary'}),
    ('Combined',          OUT_COMBINED, {'primary', 'secondary'}),
]:
    chk = gpd.read_parquet(path)
    print(f'\n--- {label}: {path.name} ---')
    print(f'  Rows  : {len(chk):,}')
    print(f'  CRS   : EPSG:{chk.crs.to_epsg()}')
    print(f'  Geoms : {chk.geometry.geom_type.value_counts().to_dict()}')
    print(f'  class_rating     : {chk["class_rating"].value_counts().to_dict()}')
    print(f'  voltage_tier     : {sorted(chk["voltage_tier"].unique().tolist())}')
    print(f'  signal_tier      : {chk["signal_tier"].value_counts().to_dict()}')
    print(f'  tx_signal_score  : {sorted(chk["transmission_signal_score"].unique().tolist())}')

    all_pass = True
    checks = {
        'CRS is 5070':                chk.crs.to_epsg() == 5070,
        'No null geometries':         chk.geometry.isna().sum() == 0,
        'All Polygon geoms':          chk.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).all(),
        'Only target states':         set(chk['state_code'].dropna().unique()) <= set(TARGET_STATES),
        'signal_tier set correct':    set(chk['signal_tier'].unique()) == expected_tiers,
        'No null voltage_tier':       chk['voltage_tier'].notna().all(),
        'No null tx_signal_score':    chk['transmission_signal_score'].notna().all(),
    }
    for clabel, result in checks.items():
        status = 'PASS' if result else 'FAIL'
        if not result:
            all_pass = False
        print(f'  [{status}] {clabel}')

    if label == '345kV+ (primary)':
        print()
        print('  Per state area summary (primary only — this is the keep-zone input):')
        for state in TARGET_STATES:
            s = chk[chk['state_code'] == state]
            total_km2 = s.geometry.area.sum() / 1e6
            dissolved_km2 = s.dissolve().geometry.area.sum() / 1e6
            overlap_pct = 100 * (1 - dissolved_km2 / total_km2) if total_km2 > 0 else 0
            print(f'    {state}: {len(s):>4} segs | {total_km2:>8,.0f} km2 raw | '
                  f'{dissolved_km2:>8,.0f} km2 dissolved | {overlap_pct:.0f}% overlap')

print()
print('All outputs verified.' if all_pass else 'WARNING: one or more checks failed.')
