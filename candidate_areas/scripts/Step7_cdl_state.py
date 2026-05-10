"""
Step 7 - CDL candidate generation for a single state.

Runs all steps for one state:
  1. Load CDL raster, get authoritative grid
  2. Filter keep_zones_final to state bbox
  3. Rasterize keep_zones_final onto CDL grid -> buildable_mask_{state}.tif
  4. Apply mask to CDL -> cdl_masked_{state}.tif
  5. Per land-cover group: open -> close -> label -> filter -> polygonize
     Saves cdl_components_{state}_{group}.tif per group
  6. Combine groups, final filter >= 50 acres
  7. Save cdl_candidates_{state}.parquet + .fgb

Usage:
  python candidate_areas/Step7_cdl_state.py --state NV
  python candidate_areas/Step7_cdl_state.py --state VA
  python candidate_areas/Step7_cdl_state.py --state AZ
  python candidate_areas/Step7_cdl_state.py --state CA
  python candidate_areas/Step7_cdl_state.py --state TX
"""

import argparse
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
import geopandas as gpd
import pandas as pd
from scipy.ndimage import label, binary_opening, binary_closing
from shapely.geometry import shape
from shapely.validation import make_valid
from pathlib import Path
import uuid
from datetime import date
import gc

# ---- Config -----------------------------------------------------------------

CDL_FILES = {
    'AZ': 'candidate_areas/data/cdl/source/az/CDL_2025_04.tif',
    'NV': 'candidate_areas/data/cdl/source/nv/CDL_2025_32.tif',
    'VA': 'candidate_areas/data/cdl/source/va/CDL_2025_51.tif',
    'CA': 'candidate_areas/data/cdl/source/ca/CDL_2025_06.tif',
    'TX': 'candidate_areas/data/cdl/source/tx/CDL_2025_48.tif',
}

KZF_PATH  = Path('candidate_areas/outputs/keep_zones_final.parquet')
OUT_DIR   = Path('candidate_areas/data/cdl/processed')   # intermediate rasters per state
CAND_DIR  = Path('candidate_areas/outputs/states')        # final candidates per state

PIXEL_AREA   = 30 * 30           # m² per pixel
ACRES_PER_M2 = 1 / 4046.856
MIN_PIXELS   = 225               # ~50 acres at 30m
NODATA       = 0
STRUCTURE    = np.ones((3, 3), dtype=bool)

GROUPS = {
    'cropland'    : list(range(1, 62)),
    'grassland'   : [176],
    'shrub_barren': [152, 131],
    'forest'      : [141, 142, 143],
}
GROUP_LABELS = {
    'cropland'    : 'Cropland / Fallow',
    'grassland'   : 'Grassland / Pasture',
    'shrub_barren': 'Shrubland / Barren',
    'forest'      : 'Forest',
}

# ---- Parse args -------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--state', required=True, choices=list(CDL_FILES.keys()))
args = parser.parse_args()
STATE = args.state

CDL_PATH       = Path(CDL_FILES[STATE])
state_lower    = STATE.lower()
STATE_OUT_DIR  = OUT_DIR / state_lower                          # processed/{state}/
STATE_OUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_CAND_DIR = CAND_DIR / state_lower                         # outputs/states/{state}/
STATE_CAND_DIR.mkdir(parents=True, exist_ok=True)
MASK_PATH      = STATE_OUT_DIR / f'buildable_mask_{state_lower}.tif'
MASKED_CDL_PATH= STATE_OUT_DIR / f'cdl_masked_{state_lower}.tif'
OUT_PARQUET    = STATE_CAND_DIR / f'cdl_candidates_{state_lower}.parquet'
OUT_FGB        = STATE_CAND_DIR / f'cdl_candidates_{state_lower}.fgb'

print(f'\n{"="*55}')
print(f' CDL CANDIDATE GENERATION — {STATE}')
print(f'{"="*55}')

if not CDL_PATH.exists():
    raise SystemExit(f'ERROR: CDL file not found: {CDL_PATH}')

# ---- Step 1: Load CDL grid --------------------------------------------------

print(f'\nStep 1: Loading CDL raster ...')
with rasterio.open(CDL_PATH) as src:
    cdl_transform = src.transform
    cdl_shape     = (src.height, src.width)
    cdl_crs       = src.crs
    cdl_bounds    = src.bounds
    cdl_profile   = src.profile.copy()

print(f'  Shape: {cdl_shape[0]:,} x {cdl_shape[1]:,}')
print(f'  CRS:   {cdl_crs}')
print(f'  Pixel: {cdl_transform.a:.0f}m x {abs(cdl_transform.e):.0f}m')

# ---- Step 2: Filter keep_zones_final to state bbox -------------------------

print(f'\nStep 2: Filtering keep_zones_final to {STATE} bbox ...')
kzf = gpd.read_parquet(KZF_PATH)
xmin, ymin, xmax, ymax = cdl_bounds.left, cdl_bounds.bottom, cdl_bounds.right, cdl_bounds.top
kzf_state = kzf.cx[xmin:xmax, ymin:ymax]
print(f'  Rows: {len(kzf_state):,} | Area: {kzf_state.geometry.area.sum()/1e6:,.1f} km2')
del kzf

if len(kzf_state) == 0:
    raise SystemExit(f'ERROR: No keep_zones_final rows overlap {STATE} CDL bounds.')

# ---- Step 3: Rasterize buildable mask ---------------------------------------

print(f'\nStep 3: Rasterizing buildable mask ...')
shapes_iter = ((geom, 1) for geom in kzf_state.geometry
               if geom is not None and not geom.is_empty)

buildable_mask = rasterize(
    shapes=shapes_iter,
    out_shape=cdl_shape,
    transform=cdl_transform,
    fill=0,
    dtype=np.uint8,
    all_touched=False,
)
del kzf_state

n_buildable = int((buildable_mask == 1).sum())
area_km2    = n_buildable * (cdl_transform.a ** 2) / 1e6
print(f'  Buildable pixels: {n_buildable:,} | Area: {area_km2:,.1f} km2')

profile_mask = {
    'driver': 'GTiff', 'dtype': 'uint8', 'width': cdl_shape[1],
    'height': cdl_shape[0], 'count': 1, 'crs': cdl_crs,
    'transform': cdl_transform, 'compress': 'lzw', 'nodata': 255,
}
with rasterio.open(MASK_PATH, 'w', **profile_mask) as dst:
    dst.write(buildable_mask, 1)
print(f'  Saved: {MASK_PATH.name} ({MASK_PATH.stat().st_size/1e6:.1f} MB)')

# ---- Step 4: Apply mask to CDL ----------------------------------------------

print(f'\nStep 4: Applying buildable mask to CDL ...')
with rasterio.open(CDL_PATH) as src:
    cdl_array = src.read(1)

masked_cdl = np.where(buildable_mask == 1, cdl_array, NODATA).astype(np.uint8)
del cdl_array, buildable_mask

n_valid = int((masked_cdl != NODATA).sum())
print(f'  Valid pixels: {n_valid:,}')

out_profile = cdl_profile.copy()
out_profile.update({'dtype': 'uint8', 'nodata': NODATA, 'compress': 'lzw'})
with rasterio.open(MASKED_CDL_PATH, 'w', **out_profile) as dst:
    dst.write(masked_cdl, 1)
print(f'  Saved: {MASKED_CDL_PATH.name} ({MASKED_CDL_PATH.stat().st_size/1e6:.1f} MB)')

pixel_area_acres = PIXEL_AREA * ACRES_PER_M2

# ---- Step 5: Per-group connected components ---------------------------------

all_candidates = []

for group_name, codes in GROUPS.items():
    print(f'\n--- {group_name.upper()} ({GROUP_LABELS[group_name]}) ---')

    group_mask = np.isin(masked_cdl, codes) & (masked_cdl != NODATA)
    n_raw = int(group_mask.sum())
    print(f'  Raw pixels: {n_raw:,}')

    if n_raw == 0:
        print(f'  No pixels — skipping.')
        continue

    group_mask = binary_opening(group_mask, structure=STRUCTURE)
    group_mask = binary_closing(group_mask, structure=STRUCTURE)
    print(f'  After open+close: {int(group_mask.sum()):,} pixels')

    labeled_array, n_comp = label(group_mask, structure=np.ones((3, 3)))
    print(f'  Components: {n_comp:,}')
    del group_mask

    counts = np.bincount(labeled_array.ravel())
    counts[0] = 0
    surviving = np.where(counts >= MIN_PIXELS)[0]
    print(f'  Surviving (>= {MIN_PIXELS}px): {len(surviving):,}')

    if len(surviving) == 0:
        del labeled_array, counts
        continue

    remove_mask = counts < MIN_PIXELS
    remove_mask[0] = True
    filtered = labeled_array.copy()
    filtered[remove_mask[labeled_array]] = 0
    del labeled_array, counts

    # Save component raster
    comp_path = STATE_OUT_DIR / f'cdl_components_{state_lower}_{group_name}.tif'
    comp_profile = out_profile.copy()
    comp_profile.update({'dtype': 'int32', 'nodata': 0})
    with rasterio.open(comp_path, 'w', **comp_profile) as dst:
        dst.write(filtered.astype(np.int32), 1)
    print(f'  Saved: {comp_path.name} ({comp_path.stat().st_size/1e6:.1f} MB)')

    # Polygonize
    geoms, label_ids = [], []
    for poly_geom, label_val in shapes(
            filtered.astype(np.int32),
            mask=(filtered > 0).astype(np.uint8),
            transform=cdl_transform):
        lv = int(label_val)
        if lv == 0:
            continue
        geoms.append(make_valid(shape(poly_geom)))
        label_ids.append(lv)
    del filtered

    if not geoms:
        continue

    gdf = gpd.GeoDataFrame({'label_id': label_ids, 'geometry': geoms}, crs=cdl_crs)
    gdf['area_m2']     = gdf.geometry.area
    gdf['pixel_count'] = (gdf['area_m2'] / PIXEL_AREA).round().astype(int)
    gdf['area_acres']  = gdf['area_m2'] * ACRES_PER_M2

    gdf['cdl_group']       = group_name
    gdf['cdl_group_label'] = GROUP_LABELS[group_name]
    gdf['state']           = STATE

    centroids_5070 = gdf.geometry.centroid
    centroids_4326 = gpd.GeoSeries(centroids_5070, crs=cdl_crs).to_crs(4326)
    gdf['centroid_lon'] = centroids_4326.x.round(6)
    gdf['centroid_lat'] = centroids_4326.y.round(6)

    print(f'  Polygons: {len(gdf):,} | Area range: {gdf.area_acres.min():.0f}-{gdf.area_acres.max():,.0f} ac')
    all_candidates.append(gdf)
    del gdf, geoms, label_ids
    gc.collect()

del masked_cdl

# ---- Step 6: Combine + final filter -----------------------------------------

print(f'\nCombining groups ...')
result = gpd.GeoDataFrame(pd.concat(all_candidates, ignore_index=True), crs=cdl_crs)
del all_candidates

before = len(result)
result = result[result['area_acres'] >= 50].copy()
print(f'  {before:,} -> {len(result):,} candidates after re-filter')

result['candidate_id']          = [str(uuid.uuid4()) for _ in range(len(result))]
result['snapshot_date']         = date.today().isoformat()
result['candidate_type']        = 'greenfield'
result['route_complexity_score'] = None  # Owen Rec #5 — null until manually reviewed
result['route_complexity_notes'] = None
result['nlcd_class']                = None  # Owen Rec #11
result['nlcd_label']                = None
result['landcover_confidence_score'] = None

print(f'\n=== {STATE} RESULTS ===')
print(f'  Total candidates: {len(result):,}')
print(f'  Total area: {result.area_acres.sum():,.0f} acres')
for g, grp in result.groupby('cdl_group'):
    print(f'    {g:<15} {len(grp):>5,} polys | {grp.area_acres.sum():>12,.0f} acres')

# ---- Step 7: Save -----------------------------------------------------------

result.to_parquet(OUT_PARQUET, index=False)
print(f'\nSaved: {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e6:.1f} MB)')

result.to_file(str(OUT_FGB), driver='FlatGeobuf')
print(f'Saved: {OUT_FGB} ({OUT_FGB.stat().st_size/1e6:.1f} MB)')

# ---- Checks -----------------------------------------------------------------

print('\n=== CHECKS ===')
checks = {
    'Has candidates'  : len(result) > 0,
    'CRS EPSG:5070'   : result.crs.to_epsg() == 5070,
    'No null geom'    : result.geometry.isna().sum() == 0,
    'No empty geom'   : (~result.geometry.is_empty).all(),
    'All >= 50 acres' : (result.area_acres >= 50).all(),
    'Has centroid_lat': 'centroid_lat' in result.columns,
}
all_pass = True
for lbl, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
    if not ok:
        all_pass = False

print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
print(f'\nVerify cdl_candidates_{state_lower}.fgb in QGIS.')
