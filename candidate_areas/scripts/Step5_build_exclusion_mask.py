"""
Step 5 - Build exclusion mask from all hard-exclusion layers.

Strategy per layer:
  FEMA / Water / PADUS  — small row counts, hierarchical unary_union -> 1 row each
  Slope (simpslope)     — 79 huge raster-derived polygons; skip union (takes hours),
                          keep 79 rows as-is
  NWI AZ/CA/NV/VA       — wetland polygons (simple shapes, not raster blobs);
                          hierarchical unary_union -> 1 row per state
  NWI TX                — 992k rows, 10 parquet row groups; per-group
                          hierarchical union -> 10 rows

Output: candidate_areas/outputs/exclusion_mask.parquet  (EPSG:5070, ~96 rows)

Run:  python candidate_areas/Step5_build_exclusion_mask.py
"""

import geopandas as gpd
import pyarrow.parquet as pq
import pandas as pd
from shapely.ops import unary_union
from shapely.validation import make_valid
from shapely import from_wkb
from pathlib import Path

# ---- Paths ------------------------------------------------------------------

UNION_LAYERS = {
    'fema'  : Path('candidate_areas/exclusion_layers/exclusion_fema.parquet'),
    'water' : Path('candidate_areas/exclusion_layers/exclusion_water.parquet'),
    'padus' : Path('candidate_areas/exclusion_layers/exclusion_padus.parquet'),
}

SLOPE_PATH = Path('candidate_areas/exclusion_layers/exclusion_slope.parquet')

NWI_STATE_PATHS = {
    'nwi_az': Path('candidate_areas/exclusion_layers/exclusion_nwi_az_test.parquet'),
    'nwi_ca': Path('candidate_areas/exclusion_layers/exclusion_nwi_ca.parquet'),
    'nwi_nv': Path('candidate_areas/exclusion_layers/exclusion_nwi_nv.parquet'),
    'nwi_va': Path('candidate_areas/exclusion_layers/exclusion_nwi_va.parquet'),
}

NWI_TX_PATH = Path('candidate_areas/exclusion_layers/exclusion_nwi_tx.parquet')
OUT_PATH    = Path('candidate_areas/outputs/exclusion_mask.parquet')

CHUNK_SIZE = 5_000

# ---- Helper -----------------------------------------------------------------

def hierarchical_union(geoms, chunk_size=CHUNK_SIZE):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if len(geoms) == 0:
        return None
    if len(geoms) <= chunk_size:
        return unary_union(geoms)
    n_chunks = (len(geoms) - 1) // chunk_size + 1
    chunk_unions = []
    for i in range(0, len(geoms), chunk_size):
        cu = unary_union(geoms[i:i + chunk_size])
        chunk_unions.append(cu)
        print(f'    chunk {len(chunk_unions)}/{n_chunks} ({min(i+chunk_size,len(geoms)):,}/{len(geoms):,})')
    print(f'    Final union of {len(chunk_unions)} chunks ...')
    return unary_union(chunk_unions)

# ---- Collect output rows ----------------------------------------------------

layer_records = []   # {'layer': str, 'geometry': shapely_geom}

# ---- 1. FEMA / Water / PADUS — union ----------------------------------------

for name, path in UNION_LAYERS.items():
    print(f'\n{"="*50}')
    print(f' {name.upper()} (union)')
    print(f'{"="*50}')
    gdf = gpd.read_parquet(path)
    print(f'  {len(gdf):,} rows | {gdf.geometry.area.sum()/1e6:,.0f} km2')

    invalid = (~gdf.geometry.is_valid).sum()
    if invalid > 0:
        print(f'  make_valid: {invalid} invalid ...')
        gdf['geometry'] = gdf.geometry.apply(make_valid)
    gdf = gdf[~gdf.geometry.is_empty].copy()

    print(f'  Unioning {len(gdf):,} geometries ...')
    geom = hierarchical_union(list(gdf.geometry))
    print(f'  Done: {geom.geom_type} | {geom.area/1e6:,.0f} km2')
    layer_records.append({'layer': name, 'geometry': geom})
    del gdf

# ---- 2. Slope — as-is (no union) --------------------------------------------

print(f'\n{"="*50}')
print(f' SLOPE (as-is, no union — raster polygons too complex for GEOS union)')
print(f'{"="*50}')
gdf = gpd.read_parquet(SLOPE_PATH)
print(f'  {len(gdf):,} rows | {gdf.geometry.area.sum()/1e6:,.0f} km2')

invalid = (~gdf.geometry.is_valid).sum()
if invalid > 0:
    print(f'  make_valid: {invalid} invalid ...')
    gdf['geometry'] = gdf.geometry.apply(make_valid)
gdf = gdf[~gdf.geometry.is_empty][['geometry']].copy()
gdf['layer'] = 'slope'
layer_records.extend(gdf[['layer', 'geometry']].to_dict('records'))
print(f'  Kept {len(gdf):,} rows')
del gdf

# ---- 3. NWI AZ / CA / NV / VA — hierarchical union -------------------------
# Wetland polygons are simple shapes (not raster blobs), so union is fast.

for name, path in NWI_STATE_PATHS.items():
    print(f'\n{"="*50}')
    print(f' {name.upper()} (hierarchical union)')
    print(f'{"="*50}')
    gdf = gpd.read_parquet(path)
    print(f'  {len(gdf):,} rows | {gdf.geometry.area.sum()/1e6:,.0f} km2')

    invalid = (~gdf.geometry.is_valid).sum()
    if invalid > 0:
        print(f'  make_valid: {invalid} invalid ...')
        gdf['geometry'] = gdf.geometry.apply(make_valid)
    gdf = gdf[~gdf.geometry.is_empty].copy()

    print(f'  Unioning {len(gdf):,} geometries (chunk_size={CHUNK_SIZE:,}) ...')
    geom = hierarchical_union(list(gdf.geometry))
    print(f'  Done: {geom.geom_type} | {geom.area/1e6:,.0f} km2')
    layer_records.append({'layer': name, 'geometry': geom})
    del gdf

# ---- 4. NWI TX — per row-group hierarchical union ---------------------------
# 992k rows across 10 parquet row groups. Union within each group -> 10 rows.

print(f'\n{"="*50}')
print(f' NWI_TX (per row-group hierarchical union)')
print(f'{"="*50}')
pf = pq.ParquetFile(NWI_TX_PATH)
n_groups = pf.metadata.num_row_groups
print(f'  {pf.metadata.num_rows:,} rows across {n_groups} row groups')

for i in range(n_groups):
    tbl   = pf.read_row_group(i, columns=['geometry'])
    raw   = tbl.column('geometry').to_pylist()
    geoms = [make_valid(g) for g in from_wkb(raw)]
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    del tbl, raw

    print(f'  Row group {i+1:2d}/{n_groups}: {len(geoms):,} geoms -> unioning ...')
    group_geom = hierarchical_union(geoms, chunk_size=CHUNK_SIZE)
    area = group_geom.area / 1e6 if group_geom else 0
    print(f'    Done: {group_geom.geom_type} | {area:,.0f} km2')
    layer_records.append({'layer': f'nwi_tx_g{i:02d}', 'geometry': group_geom})
    del geoms, group_geom

# ---- Build output GeoDataFrame ----------------------------------------------

print(f'\nBuilding exclusion_mask GeoDataFrame ...')
mask_gdf = gpd.GeoDataFrame(layer_records, crs='EPSG:5070')
mask_gdf = mask_gdf[mask_gdf.geometry.notna() & ~mask_gdf.geometry.is_empty].copy()

total_area = mask_gdf.geometry.area.sum() / 1e6
print(f'  {len(mask_gdf)} rows | {total_area:,.0f} km2')
for _, row in mask_gdf.iterrows():
    print(f'    {row["layer"]:18s}: {row.geometry.area/1e6:>8,.0f} km2 | {row.geometry.geom_type}')

# ---- Checks -----------------------------------------------------------------

checks = {
    'Has rows'         : len(mask_gdf) > 0,
    'CRS EPSG:5070'    : mask_gdf.crs.to_epsg() == 5070,
    'No null geoms'    : mask_gdf.geometry.isna().sum() == 0,
    'No empty geoms'   : mask_gdf.geometry.is_empty.sum() == 0,
    'No invalid geoms' : (~mask_gdf.geometry.is_valid).sum() == 0,
    'Area > 0'         : total_area > 0,
    'Area < keep_zones': total_area < 932_456,
    'Has fema'         : 'fema' in mask_gdf['layer'].values,
    'Has water'        : 'water' in mask_gdf['layer'].values,
    'Has padus'        : 'padus' in mask_gdf['layer'].values,
    'Has slope'        : 'slope' in mask_gdf['layer'].values,
    'Has nwi layers'   : any('nwi' in l for l in mask_gdf['layer'].values),
}

print('\n=== CHECKS ===')
all_pass = True
for lbl, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
    if not ok:
        all_pass = False

# ---- Save -------------------------------------------------------------------

mask_gdf.to_parquet(OUT_PATH, index=False)
print(f'\nSaved: {OUT_PATH}')
print(f'  {len(mask_gdf)} rows | {total_area:,.0f} km2')
print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
print('\nNext: verify exclusion_mask.parquet in QGIS, then run Step6.')
