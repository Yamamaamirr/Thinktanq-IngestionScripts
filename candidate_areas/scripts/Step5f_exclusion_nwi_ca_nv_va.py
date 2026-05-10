"""
NWI wetland exclusion layer — CA, NV, VA.

Clips each state's NWI wetlands to keep_zones, runs make_valid, saves.
No simplification — boundaries kept as-is (same as AZ test).

Outputs:
  candidate_areas/exclusion_layers/exclusion_nwi_ca.parquet
  candidate_areas/exclusion_layers/exclusion_nwi_nv.parquet
  candidate_areas/exclusion_layers/exclusion_nwi_va.parquet
"""

import geopandas as gpd
from shapely.validation import make_valid
from shapely import get_coordinates
from pathlib import Path

STATES = {
    'CA': Path('enrichment/usfws_wetlands/data/Ca_wetlands.parquet'),
    'NV': Path('enrichment/usfws_wetlands/data/Nv_wetlands.parquet'),
    'VA': Path('enrichment/usfws_wetlands/data/Va_wetlands.parquet'),
}

KEEP_PATH = Path('candidate_areas/outputs/keep_zones.parquet')
OUT_DIR   = Path('candidate_areas/exclusion_layers')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load keep_zones --------------------------------------------------------

print('Loading keep_zones ...')
keep_gdf  = gpd.read_parquet(KEEP_PATH)
keep_geom = keep_gdf.geometry.iloc[0]
keep_bbox = keep_gdf.total_bounds
print(f'  Area: {keep_geom.area/1e6:,.0f} km2 | CRS: EPSG:{keep_gdf.crs.to_epsg()}\n')

# ---- Process each state -----------------------------------------------------

for state, nwi_path in STATES.items():
    print(f'{"="*50}')
    print(f' {state}')
    print(f'{"="*50}')

    print(f'Loading {nwi_path.name} ...')
    gdf = gpd.read_parquet(nwi_path)
    print(f'  Rows:      {len(gdf):,}')
    print(f'  CRS:       EPSG:{gdf.crs.to_epsg()}')
    print(f'  Area:      {gdf.geometry.area.sum()/1e6:,.1f} km2')
    print(f'  Invalid:   {(~gdf.geometry.is_valid).sum()}')

    # Bbox pre-filter
    xmin, ymin, xmax, ymax = keep_bbox
    in_bbox = gdf.cx[xmin:xmax, ymin:ymax]
    print(f'\n  Bbox filter: {len(gdf):,} -> {len(in_bbox):,} rows')

    if len(in_bbox) == 0:
        print(f'  WARNING: no {state} wetlands overlap keep_zones bbox — skipping.')
        continue

    # Precise clip
    print(f'  Clipping to keep_zones ...')
    clipped = gpd.clip(in_bbox[['geometry']], keep_geom)
    clipped = clipped[~clipped.geometry.is_empty].copy()
    print(f'  After clip: {len(clipped):,} rows | {clipped.geometry.area.sum()/1e6:,.2f} km2')

    if len(clipped) == 0:
        print(f'  WARNING: no {state} wetlands fall within keep_zones — skipping.')
        continue

    # make_valid
    invalid_before = (~clipped.geometry.is_valid).sum()
    print(f'\n  make_valid: {invalid_before} invalid before ...')
    clipped['geometry'] = clipped.geometry.apply(make_valid)
    clipped = clipped[~clipped.geometry.is_empty].copy()
    print(f'  Invalid after: {(~clipped.geometry.is_valid).sum()}')

    # Vertex count
    verts = sum(len(get_coordinates(g)) for g in clipped.geometry)
    print(f'  Total vertices: {verts:,}')

    # Checks
    area_km2 = clipped.geometry.area.sum() / 1e6
    checks = {
        'Has rows'         : len(clipped) > 0,
        'CRS EPSG:5070'    : clipped.crs.to_epsg() == 5070,
        'No null geoms'    : clipped.geometry.isna().sum() == 0,
        'No empty geoms'   : clipped.geometry.is_empty.sum() == 0,
        'No invalid geoms' : (~clipped.geometry.is_valid).sum() == 0,
        'Area > 0'         : area_km2 > 0,
        'Area < keep_zones': area_km2 < keep_geom.area / 1e6,
    }

    print(f'\n  === CHECKS ===')
    all_pass = True
    for lbl, ok in checks.items():
        print(f'    [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False

    # Save
    out_path = OUT_DIR / f'exclusion_nwi_{state.lower()}.parquet'
    clipped.to_parquet(out_path, index=False)
    print(f'\n  Saved: {out_path}')
    print(f'  {"All checks passed." if all_pass else "WARNING: some checks failed."}\n')

print('Done.')
