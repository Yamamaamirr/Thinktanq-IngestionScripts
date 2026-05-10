"""
Step 5b — Build open water exclusion layer (lakes, reservoirs, rivers, playas).

Reads:  ingestion_scripts/census_tiger/state_boundaries.parquet  (clip boundary)
Writes: candidate_areas/exclusion_layers/exclusion_water.parquet

Run from project root:  python candidate_areas/Step5b_exclusion_water.py

What this produces:
  Polygon layer of all significant open water bodies (lakes, reservoirs, streams,
  intermittent lakes, dry lake beds) within the 5 target states. Used in Stage 4
  to subtract from CDL candidate patches alongside FEMA and other exclusions.

Source: Esri USA Water Bodies (services.arcgis.com) — National Atlas-derived,
        updated 2018. Covers all named water bodies at national scale.

Why download nationally then clip:
  Multi-state water bodies (Lake Mead = AZ-NV, Lake Tahoe = CA-NV, Lake Texoma =
  OK-TX) have compound STATE values that make SQL filtering fragile. Downloading
  the full national dataset (12,457 features at 50-acre min) and spatially
  clipping to target state boundaries is simpler and catches everything.

Feature types included:
  Lake, Reservoir, Stream, Lake Intermittent, Lake Dry, Reservoir Intermittent
  (all open water types that cannot support 50-acre development sites)

Feature types excluded:
  Swamp or Marsh  — NWI wetlands layer (Step 5d) handles all wetlands
  Glacier         — not present in CA / TX / AZ / NV / VA
  Canal           — man-made, too narrow to affect 50-acre candidates

Size floor: SQMI >= 0.078 (50 acres = 0.078 sq mi)
  Matches candidate minimum size. Sub-50-acre water bodies inside a candidate
  leave enough remaining area to pass the Stage 4 area re-filter anyway. Including
  them would add thousands of tiny holes to eligible_zones for no benefit.

CRS: fetched in EPSG:4326, clipped and saved in EPSG:5070.

Attributes stored:
  feature    TEXT    water body type (Lake / Reservoir / Stream etc.)
  name       TEXT    water body name — useful for QGIS labelling
  sqmi       FLOAT   area in square miles (reference only)
"""

import time
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from pathlib import Path

WATER_URL = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
    "USA_Water_Bodies/FeatureServer/0/query"
)

STATE_SRC = Path('ingestion_scripts/census_tiger/state_boundaries.parquet')
OUT       = Path('candidate_areas/exclusion_layers/exclusion_water.parquet')

TARGET_STATES = ['CA', 'TX', 'AZ', 'NV', 'VA']
PAGE_SIZE     = 1000
MIN_SQMI      = 0.078   # 50 acres

EXCLUDE_TYPES = ('Swamp or Marsh', 'Glacier', 'Canal')

WHERE_CLAUSE = (
    f"FEATURE NOT IN ('Swamp or Marsh', 'Glacier', 'Canal') "
    f"AND SQMI >= {MIN_SQMI}"
)

FEATURE_LABELS = {
    'Lake':                   'Lake',
    'Reservoir':              'Reservoir',
    'Stream':                 'Stream/River',
    'Lake Intermittent':      'Lake (intermittent)',
    'Lake Dry':               'Dry lake / Playa',
    'Reservoir Intermittent': 'Reservoir (intermittent)',
}


def fetch_page(offset):
    params = {
        'where':             WHERE_CLAUSE,
        'outFields':         'OBJECTID,FEATURE,NAME,SQMI',
        'returnGeometry':    'true',
        'outSR':             4326,
        'resultOffset':      offset,
        'resultRecordCount': PAGE_SIZE,
        'f':                 'json',
    }
    for attempt in range(5):
        try:
            r = requests.get(WATER_URL, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 ** attempt * 3
            print(f'  Retry {attempt+1}/5 after {wait}s ({e})')
            time.sleep(wait)
    raise RuntimeError(f'Water Bodies API failed after 5 retries at offset {offset}')


def esri_rings_to_shapely(rings):
    try:
        return shape({'type': 'Polygon', 'coordinates': rings})
    except Exception:
        return None


# ── Load state boundaries for spatial clip ────────────────────────────────────

print('Loading state boundaries ...')
states = gpd.read_parquet(STATE_SRC)
states = states[states['state_code'].isin(TARGET_STATES)].copy()
states_dissolved = states.dissolve().reset_index(drop=True)
print(f'  {len(states)} states loaded — dissolved into single clip boundary')

# ── Download full national dataset ────────────────────────────────────────────

print(f'\nDownloading USA Water Bodies nationally (filter: open water >= 50 acres) ...')
all_rows = []
offset   = 0
page_num = 0

while True:
    page_num += 1
    data     = fetch_page(offset)
    features = data.get('features', [])
    print(f'  Page {page_num}: {len(features)} features (offset {offset})')

    for feat in features:
        geom_json = feat.get('geometry')
        attrs     = feat.get('attributes', {})
        if not geom_json or not geom_json.get('rings'):
            continue
        geom = esri_rings_to_shapely(geom_json['rings'])
        if geom is None or geom.is_empty:
            continue
        all_rows.append({
            'objectid': attrs.get('OBJECTID'),
            'feature':  attrs.get('FEATURE'),
            'name':     attrs.get('NAME'),
            'sqmi':     attrs.get('SQMI'),
            'geometry': geom,
        })

    if not data.get('exceededTransferLimit', False) or len(features) == 0:
        break
    offset += PAGE_SIZE

print(f'\nTotal national features fetched: {len(all_rows):,}')

# ── Build GeoDataFrame ────────────────────────────────────────────────────────

gdf_national = gpd.GeoDataFrame(
    pd.DataFrame(all_rows)[['feature', 'name', 'sqmi', 'geometry']],
    geometry='geometry',
    crs='EPSG:4326',
)

# ── Spatial clip to target states ─────────────────────────────────────────────

print('Clipping to target states (spatial intersection) ...')
states_4326 = states_dissolved.to_crs(epsg=4326)

# Use spatial index for speed
gdf_clipped = gdf_national[
    gdf_national.intersects(states_4326.geometry.iloc[0])
].copy()

print(f'  National: {len(gdf_national):,}  ->  Target states: {len(gdf_clipped):,}')

# ── Reproject to EPSG:5070 ────────────────────────────────────────────────────

print('Reprojecting to EPSG:5070 ...')
gdf_clipped = gdf_clipped.to_crs(epsg=5070)

n_invalid = (~gdf_clipped.geometry.is_valid).sum()
if n_invalid:
    print(f'  Fixing {n_invalid} invalid geometries ...')
    gdf_clipped['geometry'] = gdf_clipped.geometry.make_valid()
    print(f'  Invalid after fix: {(~gdf_clipped.geometry.is_valid).sum()}')

# ── Save ─────────────────────────────────────────────────────────────────────

OUT.parent.mkdir(parents=True, exist_ok=True)
gdf_clipped.to_parquet(OUT, index=False)

# ── Verify ────────────────────────────────────────────────────────────────────

check = gpd.read_parquet(OUT)
print(f'\nSaved: {OUT}')
print(f'  Rows          : {len(check):,}')
print(f'  CRS           : EPSG:{check.crs.to_epsg()}')
print(f'  Columns       : {list(check.columns)}')
print(f'  Null geometry : {check.geometry.isna().sum()}')
print(f'  Invalid geoms : {(~check.geometry.is_valid).sum()}')
print()

print('Feature type breakdown:')
breakdown = check.groupby('feature').agg(
    count=('feature', 'size'),
    total_sqmi=('sqmi', 'sum'),
).reset_index().sort_values('count', ascending=False)
for _, r in breakdown.iterrows():
    label = FEATURE_LABELS.get(r['feature'], r['feature'])
    print(f'  {label:25} | {r["count"]:>5} features | {r["total_sqmi"]:>8,.1f} sqmi')
print()

print('Spot-check key water bodies (Lake Mead, Lake Tahoe, Lake Travis):')
for name in ['Lake Mead', 'Lake Tahoe', 'Lake Travis', 'Shasta Lake', 'Lake Texoma']:
    row = check[check['name'] == name]
    if len(row):
        r = row.iloc[0]
        print(f'  FOUND  {name:20} | {r["feature"]:12} | {r["sqmi"]:.1f} sqmi')
    else:
        print(f'  MISS   {name}')
print()

total_km2 = check.geometry.area.sum() / 1e6
print(f'Total geometry area: {total_km2:,.0f} km2')
print()

checks = {
    'Has rows'           : len(check) > 0,
    'CRS is EPSG:5070'   : check.crs.to_epsg() == 5070,
    'No null geometry'   : check.geometry.isna().sum() == 0,
    'No invalid geometry': (~check.geometry.is_valid).sum() == 0,
    'No wetlands/glaciers': (~check['feature'].isin(EXCLUDE_TYPES)).all(),
    'All >= 50 acres'    : (check['sqmi'] >= MIN_SQMI).all(),
    'All polygon geoms'  : check.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).all(),
}
all_pass = True
for label, result in checks.items():
    status = 'PASS' if result else 'FAIL'
    if not result:
        all_pass = False
    print(f'  [{status}] {label}')

print('\nAll checks passed.' if all_pass else '\nWARNING: one or more checks failed.')
