"""
Step 7b - Apply buildable mask to AZ CDL raster → cdl_masked_az.tif

Step 4 only:
  Load CDL_2025_04.tif + buildable_mask_az.tif
  Apply: masked_cdl = where(mask == 1, cdl, nodata)
  Save: cdl_masked_az.tif

Output: candidate_areas/data/cdl/processed/az/cdl_masked_az.tif

Verify in QGIS:
  - Load cdl_masked_az.tif with CDL colour scheme
  - Only buildable area pixels should have land cover values
  - Areas outside keep_zones_final should be nodata (transparent)
  - Land cover classes should look geographically correct

Run: python candidate_areas/Step7b_cdl_apply_mask_az.py
"""

import numpy as np
import rasterio
from pathlib import Path

CDL_PATH  = Path('candidate_areas/data/cdl/source/az/CDL_2025_04.tif')
MASK_PATH = Path('candidate_areas/data/cdl/processed/az/buildable_mask_az.tif')
OUT_PATH  = Path('candidate_areas/data/cdl/processed/az/cdl_masked_az.tif')

NODATA = 0   # use 0 as nodata — safe for uint8 CDL (class 0 = background)

# ---- Load CDL ---------------------------------------------------------------

print('Loading CDL raster ...')
with rasterio.open(CDL_PATH) as src:
    cdl_array = src.read(1)
    cdl_profile = src.profile.copy()
    cdl_transform = src.transform
    cdl_crs = src.crs

print(f'  CDL shape: {cdl_array.shape}')
print(f'  CDL dtype: {cdl_array.dtype}')
print(f'  CDL unique classes (sample): {np.unique(cdl_array[:100,:100])}')

# ---- Load buildable mask ----------------------------------------------------

print('\nLoading buildable mask ...')
with rasterio.open(MASK_PATH) as src:
    buildable_mask = src.read(1)

print(f'  Mask shape: {buildable_mask.shape}')
print(f'  Buildable pixels: {(buildable_mask == 1).sum():,}')

# Shapes must match
assert cdl_array.shape == buildable_mask.shape, \
    f'Shape mismatch: CDL {cdl_array.shape} vs mask {buildable_mask.shape}'

# ---- Apply mask -------------------------------------------------------------

print('\nApplying buildable mask to CDL ...')
masked_cdl = np.where(buildable_mask == 1, cdl_array, NODATA).astype(np.uint8)

# Stats on masked result
n_valid = int((masked_cdl != NODATA).sum())
unique_classes = np.unique(masked_cdl[masked_cdl != NODATA])
print(f'  Valid (buildable) pixels: {n_valid:,}')
print(f'  Unique CDL classes in buildable area: {len(unique_classes)}')
print(f'  Classes present: {sorted(unique_classes[:20].tolist())} ...')

# ---- Save -------------------------------------------------------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

out_profile = cdl_profile.copy()
out_profile.update({
    'dtype'   : 'uint8',
    'nodata'  : NODATA,
    'compress': 'lzw',
})

with rasterio.open(OUT_PATH, 'w', **out_profile) as dst:
    dst.write(masked_cdl, 1)

size_mb = OUT_PATH.stat().st_size / 1e6
print(f'\nSaved: {OUT_PATH} ({size_mb:.1f} MB)')

# ---- Class breakdown --------------------------------------------------------

print('\n=== CDL CLASS BREAKDOWN IN BUILDABLE AREA ===')
TARGET_CLASSES = {
    'Cropland (1-60)'     : list(range(1, 61)),
    'Fallow/Idle (61)'    : [61],
    'Grassland (176)'     : [176],
    'Shrubland (152)'     : [152],
    'Barren (131)'        : [131],
    'Forest (141-143)'    : [141, 142, 143],
    'Water (111)'         : [111],
    'Developed (121-124)' : [121, 122, 123, 124],
    'Wetland (190,195)'   : [190, 195],
}
pixel_area_km2 = (cdl_transform.a ** 2) / 1e6
for label, codes in TARGET_CLASSES.items():
    mask_cls = np.isin(masked_cdl, codes) & (masked_cdl != NODATA)
    px = int(mask_cls.sum())
    km2 = px * pixel_area_km2
    print(f'  {label:<25} {px:>12,} px  {km2:>10,.1f} km2')

# ---- Checks -----------------------------------------------------------------

print('\n=== CHECKS ===')
checks = {
    'Has valid pixels'     : n_valid > 0,
    'Has CDL classes'      : len(unique_classes) > 0,
    'File saved'           : OUT_PATH.exists(),
    'Shape matches CDL'    : masked_cdl.shape == cdl_array.shape,
    'dtype uint8'          : masked_cdl.dtype == np.uint8,
}
all_pass = True
for lbl, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
    if not ok:
        all_pass = False

print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
print('\nNext: load cdl_masked_az.tif in QGIS.')
print('Style with CDL colour scheme — only buildable pixels should show land cover.')
print('Areas outside keep_zones_final should appear as nodata (transparent).')
