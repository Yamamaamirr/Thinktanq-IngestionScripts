"""
Step 5a — Build FEMA hard exclusion layer (floodway + Zone VE coastal).

Reads:  ingestion_scripts/census_tiger/state_boundaries.parquet  (bboxes)
Writes: candidate_areas/exclusion_layers/exclusion_fema.parquet

Run from project root:  python candidate_areas/Step5a_exclusion_fema.py

HARD EXCLUSION ONLY (Owen Rec #8):
  This file contains only the two categories Owen designates as hard exclusions.
  Plain Zone AE (100-yr floodplain with no floodway subtype) is intentionally
  excluded — it is a review flag (fema_ae_flag) applied in Step9 scoring, not a
  spatial disqualifier. It can be developed with fill/elevation engineering.

  What IS included (hard exclusions):
    Regulatory floodway  — ZONE_SUBTY IN ('Floodway', 'Administrative Floodway',
                           'Riverine Floodway Shown in Coastal Zone')
                           The active river/stream channel. Legally constrained,
                           physically unbuildable. Includes A+floodway and AE+floodway.
    Zone VE (coastal)    — FLD_ZONE = 'VE'
                           Coastal high-hazard with wave action. Extreme risk,
                           not viable for large-format infrastructure.

  What is NOT included (review flags, handled in Step9):
    Plain Zone AE        — 100-yr floodplain, no floodway subtype.
                           Buildable with engineering. Score penalty, not removal.
    Radar proximity      — review flag (Owen Rec #9).

  Full row breakdown in output:
    A  + Administrative Floodway     →  45 rows  (regulatory floodway)
    AE + Administrative Floodway     →  16 rows  (regulatory floodway)
    AE + Floodway                    → 4693 rows  (regulatory floodway)
    VE + Coastal Floodplain          →  105 rows  (coastal high-hazard)
    VE + Riverine Floodway           →   28 rows  (coastal high-hazard)
    VE + (no subtype)                → 2403 rows  (coastal high-hazard)
    TOTAL                            → 7290 rows

Source: FEMA ArcGIS FeatureServer (USA_Flood_Hazard_Reduced_Set_gdb).
        Data last updated: December 27, 2024. No re-fetch needed unless
        refreshing the dataset.

CRS: fetched in EPSG:4326, saved in EPSG:5070 (consistent with all layers).

Attributes stored:
  fld_zone   TEXT   A / AE / VE
  zone_subty TEXT   Floodway / Administrative Floodway / Coastal Floodplain /
                    Riverine Floodway Shown in Coastal Zone / null (plain VE)
"""

import time
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from pathlib import Path

FEMA_URL = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
    "USA_Flood_Hazard_Reduced_Set_gdb/FeatureServer/0/query"
)

STATE_SRC = Path('ingestion_scripts/census_tiger/state_boundaries.parquet')
OUT       = Path('candidate_areas/exclusion_layers/exclusion_fema.parquet')

TARGET_STATES = ['CA', 'TX', 'AZ', 'NV', 'VA']
PAGE_SIZE     = 2000   # requested per page; service caps at ~750, pagination uses exceededTransferLimit

WHERE_CLAUSE = (
    "ZONE_SUBTY IN ('Floodway', 'Administrative Floodway') "
    "OR FLD_ZONE IN ('V', 'VE')"
)


def fetch_page(bbox_4326, offset):
    """Fetch one page of FEMA features within a state bbox."""
    minx, miny, maxx, maxy = bbox_4326
    params = {
        'where':             WHERE_CLAUSE,
        'geometry':          f'{minx},{miny},{maxx},{maxy}',
        'geometryType':      'esriGeometryEnvelope',
        'inSR':              4326,
        'spatialRel':        'esriSpatialRelIntersects',
        'outFields':         'OBJECTID,FLD_ZONE,ZONE_SUBTY',
        'returnGeometry':    'true',
        'outSR':             4326,
        'resultOffset':      offset,
        'resultRecordCount': PAGE_SIZE,
        'f':                 'json',
    }
    for attempt in range(5):
        try:
            r = requests.get(FEMA_URL, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 ** attempt * 3
            print(f'    Retry {attempt+1}/5 after {wait}s ({e})')
            time.sleep(wait)
    raise RuntimeError(f'FEMA API failed after 5 retries at offset {offset}')


def esri_rings_to_shapely(rings):
    """Convert Esri polygon rings list to a shapely geometry."""
    return shape({'type': 'Polygon', 'coordinates': rings})


# ── Load state boundaries ─────────────────────────────────────────────────────

print('Loading state boundaries ...')
states = gpd.read_parquet(STATE_SRC)
states = states[states['state_code'].isin(TARGET_STATES)].copy()
print(f'  {len(states)} states loaded')

# ── Fetch per state ───────────────────────────────────────────────────────────

all_frames = []

for _, state_row in states.iterrows():
    code = state_row['state_code']
    bbox = state_row.geometry.bounds  # (minx, miny, maxx, maxy) in 4326

    print(f'\n{code}: bbox = ({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f})')

    offset      = 0
    state_rows  = []
    page_num    = 0

    while True:
        page_num += 1
        data = fetch_page(bbox, offset)

        features = data.get('features', [])
        print(f'  Page {page_num}: {len(features)} features (offset {offset})')

        for feat in features:
            geom_json = feat.get('geometry')
            attrs     = feat.get('attributes', {})
            if not geom_json or not geom_json.get('rings'):
                continue
            try:
                geom = esri_rings_to_shapely(geom_json['rings'])
            except Exception:
                continue
            state_rows.append({
                'objectid':  attrs.get('OBJECTID'),
                'fld_zone':  attrs.get('FLD_ZONE'),
                'zone_subty': attrs.get('ZONE_SUBTY'),
                'geometry':  geom,
            })

        # Stop when the service signals no more records exist for this bbox.
        # exceededTransferLimit=True means more pages remain; absent/False means done.
        if not data.get('exceededTransferLimit', False) or len(features) == 0:
            break
        offset += PAGE_SIZE

    print(f'  {code} total raw features: {len(state_rows)}')
    if state_rows:
        all_frames.append(pd.DataFrame(state_rows))

# ── Combine and deduplicate ───────────────────────────────────────────────────

print('\nCombining all states ...')
combined_df = pd.concat(all_frames, ignore_index=True)
print(f'  Total before dedup: {len(combined_df):,}')

# Features on state borders may appear in two state fetches — drop by OBJECTID
combined_df = combined_df.drop_duplicates(subset=['objectid'])
print(f'  Total after dedup : {len(combined_df):,}')

# ── Build GeoDataFrame and reproject ─────────────────────────────────────────

print('Building GeoDataFrame ...')
gdf = gpd.GeoDataFrame(
    combined_df[['fld_zone', 'zone_subty', 'geometry']],
    geometry='geometry',
    crs='EPSG:4326',
)

print('Reprojecting to EPSG:5070 ...')
gdf = gdf.to_crs(epsg=5070)

# Fix any invalid geometries before saving — GEOS dissolve will crash on them
n_invalid = (~gdf.geometry.is_valid).sum()
if n_invalid:
    print(f'  Fixing {n_invalid} invalid geometries with make_valid() ...')
    gdf['geometry'] = gdf.geometry.make_valid()
    print(f'  Invalid after fix: {(~gdf.geometry.is_valid).sum()}')

# ── Save ─────────────────────────────────────────────────────────────────────

OUT.parent.mkdir(parents=True, exist_ok=True)
gdf.to_parquet(OUT, index=False)

# ── Verify ────────────────────────────────────────────────────────────────────

check = gpd.read_parquet(OUT)
print(f'\nSaved: {OUT}')
print(f'  Rows          : {len(check):,}')
print(f'  CRS           : EPSG:{check.crs.to_epsg()}')
print(f'  Columns       : {list(check.columns)}')
print(f'  Null geometry : {check.geometry.isna().sum()}')
print(f'  Invalid geoms : {(~check.geometry.is_valid).sum()}')
print()

print('Zone breakdown:')
print(check.groupby(['fld_zone', 'zone_subty'], dropna=False).size().to_string())
print()

print('Area summary:')
total_km2 = check.geometry.area.sum() / 1e6
print(f'  Total raw area (with overlaps): {total_km2:,.0f} km2')
try:    
    dissolved_km2 = check.dissolve().geometry.area.sum() / 1e6
    print(f'  Dissolved area (unique)        : {dissolved_km2:,.0f} km2')
except Exception as e:
    print(f'  Dissolve skipped (geometry complexity): {e}')
print()

checks = {
    'Has rows'              : len(check) > 0,
    'CRS is EPSG:5070'      : check.crs.to_epsg() == 5070,
    'No null geometry'      : check.geometry.isna().sum() == 0,
    'Only target zones'     : set(check['fld_zone'].dropna()) <= {'A', 'AE', 'V', 'VE'},
    'All polygon geometries': check.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).all(),
}
all_pass = True
for label, result in checks.items():
    status = 'PASS' if result else 'FAIL'
    if not result:
        all_pass = False
    print(f'  [{status}] {label}')

print('\nAll checks passed.' if all_pass else '\nWARNING: one or more checks failed.')
