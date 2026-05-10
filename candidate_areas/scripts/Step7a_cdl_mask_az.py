"""
Step 7a - Rasterize keep_zones_final onto AZ CDL grid → buildable_mask_az.tif

Steps 1-3 only:
  1. Load CDL_2025_04.tif (AZ) — get authoritative grid (transform, shape, CRS)
  2. Filter keep_zones_final.parquet to AZ bbox
  3. Rasterize keep_zones_final onto CDL grid → buildable_mask_az.tif

Output: candidate_areas/data/cdl/processed/az/buildable_mask_az.tif

Verify in QGIS:
  - Load buildable_mask_az.tif alongside keep_zones_final.fgb
  - Pixels with value=1 should sit cleanly inside keep_zones_final
  - No fishnet grid lines visible
  - No coverage outside AZ keep_zones_final areas

Run: python candidate_areas/Step7a_cdl_mask_az.py
"""

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
from pathlib import Path

CDL_PATH  = Path('candidate_areas/data/cdl/source/az/CDL_2025_04.tif')
KZF_PATH  = Path('candidate_areas/outputs/keep_zones_final.parquet')
OUT_PATH  = Path('candidate_areas/data/cdl/processed/az/buildable_mask_az.tif')

# ---- Step 1: Load CDL, get authoritative grid -------------------------------

print('Step 1: Loading CDL raster to get authoritative grid ...')
with rasterio.open(CDL_PATH) as src:
    cdl_transform = src.transform
    cdl_shape     = (src.height, src.width)
    cdl_crs       = src.crs
    cdl_bounds    = src.bounds
    cdl_nodata    = src.nodata

print(f'  CDL shape:     {cdl_shape[0]:,} rows x {cdl_shape[1]:,} cols')
print(f'  CDL CRS:       {cdl_crs}')
print(f'  CDL bounds:    {cdl_bounds}')
print(f'  CDL transform: {cdl_transform}')
print(f'  CDL nodata:    {cdl_nodata}')
print(f'  Pixel size:    {cdl_transform.a:.1f}m x {abs(cdl_transform.e):.1f}m')

# ---- Step 2: Load keep_zones_final, filter to AZ bbox ----------------------

print('\nStep 2: Loading keep_zones_final and filtering to AZ bbox ...')
kzf = gpd.read_parquet(KZF_PATH)
print(f'  Total rows loaded: {len(kzf):,}')

# Filter to rows that overlap CDL bounds (AZ bbox)
xmin, ymin, xmax, ymax = cdl_bounds.left, cdl_bounds.bottom, cdl_bounds.right, cdl_bounds.top
kzf_az = kzf.cx[xmin:xmax, ymin:ymax]
print(f'  Rows overlapping AZ CDL bbox: {len(kzf_az):,}')
print(f'  Area in AZ bbox: {kzf_az.geometry.area.sum()/1e6:,.1f} km2')
print(f'  CRS: EPSG:{kzf_az.crs.to_epsg()}')

if len(kzf_az) == 0:
    raise SystemExit('ERROR: No keep_zones_final rows overlap AZ CDL bounds.')

# ---- Step 3: Rasterize keep_zones_final onto CDL grid ----------------------

print('\nStep 3: Rasterizing keep_zones_final onto CDL grid ...')
print(f'  all_touched=False (pixel center must be inside polygon)')
print(f'  dtype=uint8')

shapes = ((geom, 1) for geom in kzf_az.geometry if geom is not None and not geom.is_empty)

buildable_mask = rasterize(
    shapes=shapes,
    out_shape=cdl_shape,
    transform=cdl_transform,
    fill=0,
    dtype=np.uint8,
    all_touched=False,
)

n_buildable = int((buildable_mask == 1).sum())
area_km2    = n_buildable * (cdl_transform.a ** 2) / 1e6
print(f'  Buildable pixels: {n_buildable:,}')
print(f'  Buildable area:   {area_km2:,.1f} km2')
print(f'  Coverage:         {n_buildable / (cdl_shape[0]*cdl_shape[1]) * 100:.2f}% of AZ CDL raster')

# ---- Save output ------------------------------------------------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

profile = {
    'driver'   : 'GTiff',
    'dtype'    : 'uint8',
    'width'    : cdl_shape[1],
    'height'   : cdl_shape[0],
    'count'    : 1,
    'crs'      : cdl_crs,
    'transform': cdl_transform,
    'compress' : 'lzw',
    'nodata'   : 255,
}

with rasterio.open(OUT_PATH, 'w', **profile) as dst:
    dst.write(buildable_mask, 1)

size_mb = OUT_PATH.stat().st_size / 1e6
print(f'\nSaved: {OUT_PATH} ({size_mb:.1f} MB)')

# ---- Checks -----------------------------------------------------------------

print('\n=== CHECKS ===')
checks = {
    'Has buildable pixels' : n_buildable > 0,
    'Area > 0 km2'         : area_km2 > 0,
    'Area < AZ state area' : area_km2 < 300_000,
    'File saved'           : OUT_PATH.exists(),
    'dtype uint8'          : buildable_mask.dtype == np.uint8,
}
all_pass = True
for lbl, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
    if not ok:
        all_pass = False

print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
print('\nNext: load buildable_mask_az.tif in QGIS alongside keep_zones_final.fgb')
print('Verify: value=1 pixels sit inside keep_zones_final, no fishnet grid lines visible.')
