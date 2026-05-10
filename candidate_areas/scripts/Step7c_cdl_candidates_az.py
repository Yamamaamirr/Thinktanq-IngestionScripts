"""
Step 7c - Per-group connected components → candidate polygons (AZ)

For each land-cover group:
  5a: binary group mask from masked CDL
  5b: binary_opening  (3x3) — remove single-pixel specks
  5c: binary_closing  (3x3) — fill micro-holes
  5d: label() 8-connectivity — find contiguous blobs
  5e: bincount filter — drop blobs < 225 pixels (< 50 acres at 30m)
  5f: polygonize surviving label IDs
  5g: add attributes (area_acres from pixel count, centroid in EPSG:4326)

Intermediate outputs (verify each in QGIS):
  candidate_areas/data/cdl/processed/az/cdl_components_az_cropland.tif
  candidate_areas/data/cdl/processed/az/cdl_components_az_grassland.tif
  candidate_areas/data/cdl/processed/az/cdl_components_az_shrub_barren.tif
  candidate_areas/data/cdl/processed/az/cdl_components_az_forest.tif

Final outputs:
  candidate_areas/outputs/states/az/cdl_candidates_az.parquet
  candidate_areas/outputs/states/az/cdl_candidates_az.fgb

Run: python candidate_areas/Step7c_cdl_candidates_az.py
"""

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
import pandas as pd
from scipy.ndimage import label, binary_opening, binary_closing
from shapely.geometry import shape
from shapely.validation import make_valid
from pathlib import Path
import uuid
from datetime import date
import gc

MASKED_CDL_PATH = Path('candidate_areas/data/cdl/processed/az/cdl_masked_az.tif')
OUT_PARQUET     = Path('candidate_areas/outputs/states/az/cdl_candidates_az.parquet')
OUT_FGB         = Path('candidate_areas/outputs/states/az/cdl_candidates_az.fgb')
OUT_DIR         = Path('candidate_areas/data/cdl')

STATE       = 'AZ'
PIXEL_AREA  = 30 * 30          # m² per pixel at 30m resolution
ACRES_PER_M2 = 1 / 4046.856   # conversion factor
MIN_PIXELS  = 225              # ~50 acres at 30m (900 m²/px × 225 = 202,500 m²)
STRUCTURE   = np.ones((3, 3), dtype=bool)  # 3x3 structuring element

# Land-cover groups: name → CDL class codes
GROUPS = {
    'cropland'    : list(range(1, 62)),          # 1-60 active + 61 fallow
    'grassland'   : [176],
    'shrub_barren': [152, 131],
    'forest'      : [141, 142, 143],
}

# Human-readable labels per group
GROUP_LABELS = {
    'cropland'    : 'Cropland / Fallow',
    'grassland'   : 'Grassland / Pasture',
    'shrub_barren': 'Shrubland / Barren',
    'forest'      : 'Forest',
}

# ---- Load masked CDL --------------------------------------------------------

print('Loading masked CDL raster ...')
with rasterio.open(MASKED_CDL_PATH) as src:
    masked_cdl  = src.read(1)
    transform   = src.transform
    crs         = src.crs
    nodata      = src.nodata or 0
    profile     = src.profile.copy()

pixel_area_acres = PIXEL_AREA * ACRES_PER_M2
print(f'  Shape: {masked_cdl.shape}')
print(f'  CRS:   {crs}')
print(f'  Pixel area: {PIXEL_AREA} m² = {pixel_area_acres:.4f} acres')
print(f'  Min patch size: {MIN_PIXELS} pixels = {MIN_PIXELS * pixel_area_acres:.0f} acres')

# ---- Process each land-cover group ------------------------------------------

all_candidates = []

for group_name, codes in GROUPS.items():
    print(f'\n{"="*55}')
    print(f' Group: {group_name.upper()}  ({GROUP_LABELS[group_name]})')
    print(f' Classes: {codes[:5]}{"..." if len(codes) > 5 else ""}')
    print(f'{"="*55}')

    # 5a: Binary group mask
    group_mask = np.isin(masked_cdl, codes) & (masked_cdl != nodata)
    n_raw = int(group_mask.sum())
    print(f'  5a Raw pixels: {n_raw:,} ({n_raw * pixel_area_acres / 640:,.0f} sq miles)')

    if n_raw == 0:
        print(f'  No pixels for this group — skipping.')
        continue

    # 5b: binary_opening — remove single-pixel specks
    group_mask = binary_opening(group_mask, structure=STRUCTURE)
    print(f'  5b After opening:  {int(group_mask.sum()):,} pixels')

    # 5c: binary_closing — fill micro-holes
    group_mask = binary_closing(group_mask, structure=STRUCTURE)
    print(f'  5c After closing:  {int(group_mask.sum()):,} pixels')

    # 5d: Connected components (8-connectivity)
    labeled_array, n_components = label(group_mask, structure=np.ones((3,3)))
    print(f'  5d Connected components: {n_components:,}')
    del group_mask

    # 5e: Filter by pixel count using bincount
    counts = np.bincount(labeled_array.ravel())   # counts[i] = pixels in label i
    counts[0] = 0                                  # label 0 = background, always remove

    surviving_labels = np.where(counts >= MIN_PIXELS)[0]
    n_surviving = len(surviving_labels)
    print(f'  5e Surviving blobs (>= {MIN_PIXELS}px / ~50 acres): {n_surviving:,}')

    if n_surviving == 0:
        print(f'  No blobs pass size filter — skipping.')
        del labeled_array, counts
        continue

    # Build filtered label raster (0 = removed, keep surviving label IDs)
    remove_mask = counts < MIN_PIXELS
    remove_mask[0] = True   # always remove background
    filtered_labels = labeled_array.copy()
    filtered_labels[remove_mask[labeled_array]] = 0
    del labeled_array, counts

    # Save intermediate raster for QGIS verification
    comp_path = OUT_DIR / f'cdl_components_az_{group_name}.tif'
    comp_profile = profile.copy()
    comp_profile.update({'dtype': 'int32', 'nodata': 0, 'compress': 'lzw'})
    with rasterio.open(comp_path, 'w', **comp_profile) as dst:
        dst.write(filtered_labels.astype(np.int32), 1)
    print(f'  Saved component raster: {comp_path.name} ({comp_path.stat().st_size/1e6:.1f} MB)')

    # 5f: Polygonize surviving label IDs
    print(f'  5f Polygonizing {n_surviving:,} blobs ...')
    geoms = []
    label_ids = []
    pixel_counts = []

    for poly_geom, label_val in shapes(
            filtered_labels.astype(np.int32),
            mask=(filtered_labels > 0).astype(np.uint8),
            transform=transform):
        lv = int(label_val)
        if lv == 0:
            continue
        geoms.append(make_valid(shape(poly_geom)))
        label_ids.append(lv)

    del filtered_labels

    if not geoms:
        print(f'  No polygons produced — skipping.')
        continue

    # 5g: Build GeoDataFrame with attributes
    # Area from polygon geometry — since we don't simplify, polygon area is
    # pixel-perfect. Using pixel count per label would double-count fragments
    # when 8-connectivity blobs produce multiple 4-connectivity polygons in shapes().
    gdf = gpd.GeoDataFrame(
        {'label_id': label_ids, 'geometry': geoms},
        crs=crs
    )

    gdf['area_m2']     = gdf.geometry.area
    gdf['pixel_count'] = (gdf['area_m2'] / PIXEL_AREA).round().astype(int)
    gdf['area_acres']  = gdf['area_m2'] * ACRES_PER_M2

    # Add metadata
    gdf['cdl_group']       = group_name
    gdf['cdl_group_label'] = GROUP_LABELS[group_name]
    gdf['state']           = STATE

    # Centroid in EPSG:4326 — compute in projected CRS first, then reproject
    centroids_5070 = gdf.geometry.centroid
    centroids_4326 = gpd.GeoSeries(centroids_5070, crs=crs).to_crs(4326)
    gdf['centroid_lon'] = centroids_4326.x.round(6)
    gdf['centroid_lat'] = centroids_4326.y.round(6)

    print(f'  5g Polygons produced: {len(gdf):,}')
    print(f'     Area range: {gdf.area_acres.min():.0f} – {gdf.area_acres.max():,.0f} acres')
    print(f'     Total area: {gdf.area_acres.sum()/640:,.0f} sq miles')

    all_candidates.append(gdf)
    del gdf, geoms, label_ids
    gc.collect()

# ---- Combine all groups -----------------------------------------------------

print(f'\nCombining all groups ...')
if not all_candidates:
    raise SystemExit('ERROR: No candidate polygons produced for any group.')

result = gpd.GeoDataFrame(
    pd.concat(all_candidates, ignore_index=True), crs=crs)

# Re-filter >= 50 acres (morphology/polygonize can create tiny artifacts)
before = len(result)
result = result[result['area_acres'] >= 50].copy()
print(f'  After re-filter: {before:,} -> {len(result):,} candidates')

# Add candidate_id and snapshot_date
result['candidate_id']   = [str(uuid.uuid4()) for _ in range(len(result))]
result['snapshot_date']  = date.today().isoformat()
result['candidate_type'] = 'greenfield'

print(f'  Total candidates: {len(result):,}')
print(f'  Total area: {result.area_acres.sum():,.0f} acres '
      f'({result.area_acres.sum()/640:,.0f} sq miles)')
print(f'\n  By group:')
for g, grp in result.groupby('cdl_group'):
    print(f'    {g:<15} {len(grp):>6,} polygons | '
          f'{grp.area_acres.sum():>12,.0f} acres')

# ---- Save -------------------------------------------------------------------

result.to_parquet(OUT_PARQUET, index=False)
print(f'\nSaved parquet: {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e6:.1f} MB)')

result.to_file(str(OUT_FGB), driver='FlatGeobuf')
print(f'Saved FGB:     {OUT_FGB} ({OUT_FGB.stat().st_size/1e6:.1f} MB)')

# ---- Checks -----------------------------------------------------------------

print('\n=== CHECKS ===')
checks = {
    'Has candidates'      : len(result) > 0,
    'CRS EPSG:5070'       : result.crs.to_epsg() == 5070,
    'No null geometry'    : result.geometry.isna().sum() == 0,
    'No empty geometry'   : (~result.geometry.is_empty).all(),
    'All >= 50 acres'     : (result.area_acres >= 50).all(),
    'Has cdl_group'       : 'cdl_group' in result.columns,
    'Has candidate_id'    : 'candidate_id' in result.columns,
    'Has centroid_lat'    : 'centroid_lat' in result.columns,
}
all_pass = True
for lbl, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
    if not ok:
        all_pass = False

print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
print('\nNext: load cdl_candidates_az.fgb in QGIS.')
print('Also load cdl_components_az_*.tif rasters to verify individual groups.')
print('Verify polygons sit inside keep_zones_final with natural CDL boundaries.')
