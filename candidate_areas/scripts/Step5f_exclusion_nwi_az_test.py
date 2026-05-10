"""
NWI wetland exclusion layer — AZ test run.

Loads Az_wetlands.parquet, clips to keep_zones, runs make_valid,
simplifies at 50m (same tolerance as slope), saves result.

This is a single-state test before the full 5-state script.

Output: candidate_areas/exclusion_layers/exclusion_nwi_az_test.parquet
"""

import geopandas as gpd
from shapely.validation import make_valid
from shapely import get_coordinates
from pathlib import Path

NWI_PATH   = Path('enrichment/usfws_wetlands/data/Az_wetlands.parquet')
KEEP_PATH  = Path('candidate_areas/outputs/keep_zones.parquet')
OUT_PATH   = Path('candidate_areas/exclusion_layers/exclusion_nwi_az_test.parquet')
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SIMPLIFY_TOLERANCE = None   # no simplification — NWI boundaries kept as-is

# ---- Load keep_zones --------------------------------------------------------

print('Loading keep_zones ...')
keep_gdf  = gpd.read_parquet(KEEP_PATH)
keep_geom = keep_gdf.geometry.iloc[0]
keep_bbox = keep_gdf.total_bounds   # [xmin, ymin, xmax, ymax]
print(f'  keep_zones area: {keep_geom.area/1e6:,.0f} km2 | CRS: EPSG:{keep_gdf.crs.to_epsg()}')

# ---- Load AZ wetlands -------------------------------------------------------

print(f'\nLoading {NWI_PATH.name} ...')
gdf = gpd.read_parquet(NWI_PATH)
print(f'  Rows:       {len(gdf):,}')
print(f'  CRS:        EPSG:{gdf.crs.to_epsg()}')
print(f'  File area:  {gdf.geometry.area.sum()/1e6:,.1f} km2')
print(f'  Invalid:    {(~gdf.geometry.is_valid).sum()}')

# ---- Bbox pre-filter --------------------------------------------------------

print('\nApplying bbox pre-filter ...')
xmin, ymin, xmax, ymax = keep_bbox
in_bbox = gdf.cx[xmin:xmax, ymin:ymax]
print(f'  Before: {len(gdf):,} rows')
print(f'  After bbox filter: {len(in_bbox):,} rows')

if len(in_bbox) == 0:
    print('  WARNING: no AZ wetlands overlap keep_zones bbox — check CRS.')
    raise SystemExit(1)

# ---- Precise clip to keep_zones ---------------------------------------------

print('\nClipping to keep_zones ...')
clipped = gpd.clip(in_bbox[['geometry']], keep_geom)
clipped = clipped[~clipped.geometry.is_empty].copy()
print(f'  After precise clip: {len(clipped):,} rows')
print(f'  Clipped area: {clipped.geometry.area.sum()/1e6:,.2f} km2')

if len(clipped) == 0:
    print('  WARNING: no AZ wetlands fall within keep_zones.')
    raise SystemExit(1)

# ---- make_valid -------------------------------------------------------------

print('\nmake_valid ...')
invalid_before = (~clipped.geometry.is_valid).sum()
print(f'  Invalid before: {invalid_before}')
clipped['geometry'] = clipped.geometry.apply(make_valid)
invalid_after = (~clipped.geometry.is_valid).sum()
print(f'  Invalid after:  {invalid_after}')

# drop anything that became empty after make_valid
clipped = clipped[~clipped.geometry.is_empty].copy()

# ---- Simplify ---------------------------------------------------------------

print('\nSkipping simplification — NWI boundaries kept as-is.')
verts = sum(len(get_coordinates(g)) for g in clipped.geometry)
print(f'  Total vertices: {verts:,}')

# ---- Final stats & checks ---------------------------------------------------

area_km2 = clipped.geometry.area.sum() / 1e6
print(f'\n=== RESULT ===')
print(f'  Rows:          {len(clipped):,}')
print(f'  Area:          {area_km2:.2f} km2')
print(f'  CRS:           EPSG:{clipped.crs.to_epsg()}')
print(f'  Invalid geoms: {(~clipped.geometry.is_valid).sum()}')
print(f'  Empty geoms:   {clipped.geometry.is_empty.sum()}')

checks = {
    'Has rows'         : len(clipped) > 0,
    'CRS EPSG:5070'    : clipped.crs.to_epsg() == 5070,
    'No null geoms'    : clipped.geometry.isna().sum() == 0,
    'No empty geoms'   : clipped.geometry.is_empty.sum() == 0,
    'No invalid geoms' : (~clipped.geometry.is_valid).sum() == 0,
    'Area > 0'         : area_km2 > 0,
    'Area < keep_zones': area_km2 < keep_geom.area / 1e6,
}

print('\n=== CHECKS ===')
all_pass = True
for lbl, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
    if not ok:
        all_pass = False

# ---- Save -------------------------------------------------------------------

clipped.to_parquet(OUT_PATH, index=False)
print(f'\nSaved: {OUT_PATH}')
print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
print('\nNext: load exclusion_nwi_az_test.parquet in QGIS and verify wetlands')
print('sit within keep_zones and cover expected AZ wetland areas.')
