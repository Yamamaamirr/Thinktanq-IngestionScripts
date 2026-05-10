"""
Ingest USFWS NWI wetlands from locally downloaded state GDB files.

Reads:  enrichment/usfws_wetlands/data/{STATE}_geodatabase_wetlands/{STATE}_geodatabase_wetlands.gdb
        ingestion_scripts/census_tiger/state_boundaries.parquet  (clip boundary)
Writes: ingestion_scripts/usfws_nwi/nwi_wetlands.parquet

Run from project root:  python ingestion_scripts/usfws_nwi/ingest_nwi_gdb.py

Source: USFWS NWI state GDB downloads (last updated Nov 24 2025).
  Data is already in EPSG:5070 (NAD83 Conus Albers) — no reprojection needed.
  Each state file extends slightly beyond state borders (mapped by quad), so
  we clip to exact state boundaries using state_boundaries.parquet.

Filter: WETLAND_TYPE excludes Lake, Riverine, Estuarine and Marine Deepwater
  — open water already captured in exclusion_water.parquet.

Attributes stored:
  attribute     TEXT   Cowardin code e.g. PFO1A, PEM3C
  wetland_type  TEXT   e.g. Freshwater Forested/Shrub Wetland
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path

GDB_BASE  = Path('enrichment/usfws_wetlands/data')
STATE_SRC = Path('ingestion_scripts/census_tiger/state_boundaries.parquet')
OUT       = Path('ingestion_scripts/usfws_nwi/nwi_wetlands.parquet')

TARGET_STATES = ['VA', 'NV', 'AZ', 'CA', 'TX']
EXCLUDE_TYPES = {'Lake', 'Riverine', 'Estuarine and Marine Deepwater'}

# ── Load state boundaries for clip (reproject to EPSG:5070 to match GDB) ─────

print('Loading state boundaries ...')
states = gpd.read_parquet(STATE_SRC)
states = states[states['state_code'].isin(TARGET_STATES)].copy()
states_5070 = states.to_crs(epsg=5070)
print(f'  {len(states)} states loaded, reprojected to EPSG:5070')

# ── Read and filter each state GDB ───────────────────────────────────────────

all_gdfs = []

for state in TARGET_STATES:
    gdb   = GDB_BASE / f'{state}_geodatabase_wetlands' / f'{state}_geodatabase_wetlands.gdb'
    layer = f'{state}_Wetlands'

    # Get state bbox in EPSG:5070 to pre-filter when reading (avoids OOM on TX)
    state_geom_5070 = states_5070[states_5070['state_code'] == state].geometry.iloc[0]
    bbox = state_geom_5070.bounds  # (minx, miny, maxx, maxy) in EPSG:5070

    print(f'\n{state}: reading {layer} (bbox pre-filter) ...')
    gdf = gpd.read_file(gdb, layer=layer, bbox=bbox)
    print(f'  Raw rows after bbox: {len(gdf):,}')

    # Filter out open water types (already in exclusion_water.parquet)
    gdf = gdf[~gdf['WETLAND_TYPE'].isin(EXCLUDE_TYPES)].copy()
    print(f'  After type filter: {len(gdf):,}')

    # Keep only needed columns, rename to match pipeline convention
    gdf = gdf[['ATTRIBUTE', 'WETLAND_TYPE', 'geometry']].copy()
    gdf.columns = ['attribute', 'wetland_type', 'geometry']

    # Precise clip to exact state boundary
    gdf = gdf[gdf.intersects(state_geom_5070)].copy()
    print(f'  After state clip: {len(gdf):,}')

    all_gdfs.append(gdf)

# ── Combine all states ────────────────────────────────────────────────────────

print('\nCombining all states ...')
combined = pd.concat(all_gdfs, ignore_index=True)
combined = gpd.GeoDataFrame(combined, geometry='geometry', crs='EPSG:5070')
print(f'  Total rows: {len(combined):,}')

# Fix any invalid geometries
n_inv = (~combined.geometry.is_valid).sum()
if n_inv:
    print(f'  Fixing {n_inv} invalid geometries ...')
    combined['geometry'] = combined.geometry.make_valid()
    combined = combined[~combined.geometry.is_empty]

# ── Save ─────────────────────────────────────────────────────────────────────

OUT.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(OUT, index=False)

# ── Verify ────────────────────────────────────────────────────────────────────

check = gpd.read_parquet(OUT)
print(f'\nSaved: {OUT}')
print(f'  Rows     : {len(check):,}')
print(f'  CRS      : EPSG:{check.crs.to_epsg()}')
print(f'  Columns  : {list(check.columns)}')
print(f'  Null geom: {check.geometry.isna().sum()}')
print(f'  Invalid  : {(~check.geometry.is_valid).sum()}')
print(f'  Area     : {check.geometry.area.sum()/1e6:,.0f} km2')

print('\nWETLAND_TYPE breakdown:')
for wt, cnt in check['wetland_type'].value_counts().items():
    print(f'  {cnt:>9,}  {wt}')

print('\nCowardin prefix breakdown:')
LABELS = {
    'PFO': 'Palustrine Forested  <- hard exclude',
    'PEM': 'Palustrine Emergent',
    'PSS': 'Palustrine Scrub-Shrub',
    'PAB': 'Palustrine Aquatic Bed',
    'PUB': 'Palustrine Unconsolidated Bottom',
    'E1 ': 'Estuarine Subtidal',
    'E2 ': 'Estuarine Intertidal',
}
for pfx, cnt in check['attribute'].str[:3].value_counts().items():
    print(f'  {cnt:>9,}  {pfx}  {LABELS.get(pfx,"")}')

checks = {
    'Has rows'            : len(check) > 0,
    'CRS EPSG:5070'       : check.crs.to_epsg() == 5070,
    'No null geometry'    : check.geometry.isna().sum() == 0,
    'No invalid geometry' : (~check.geometry.is_valid).sum() == 0,
    'No excluded types'   : len(check[check['wetland_type'].isin(EXCLUDE_TYPES)]) == 0,
    'Has PFO rows'        : check['attribute'].str.startswith('PFO').any(),
    'Has PEM rows'        : check['attribute'].str.startswith('PEM').any(),
    'All polygon geoms'   : check.geometry.geom_type.isin(['Polygon','MultiPolygon']).all(),
}
print()
all_pass = True
for lbl, result in checks.items():
    status = 'PASS' if result else 'FAIL'
    if not result:
        all_pass = False
    print(f'  [{status}] {lbl}')

print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
