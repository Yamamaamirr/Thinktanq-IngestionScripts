"""
Step 7 - CDL candidate generation for Texas (tiled).

TX CDL raster is 1.66 billion pixels (1.66 GB) — too large to load at once.
Splits into a TILE_ROWS x TILE_COLS grid, processes each tile independently,
combines vector results. Patches split at tile boundaries appear as separate
polygons but are both valid candidates (acceptable for Phase 1 screening).

Intermediate outputs per tile in candidate_areas/data/cdl/tx_tiles/
Final outputs:
  candidate_areas/outputs/states/tx/cdl_candidates_tx.parquet
  candidate_areas/outputs/states/tx/cdl_candidates_tx.fgb

Run: python candidate_areas/Step7_cdl_tx.py
"""

import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.windows import Window
import geopandas as gpd
import pandas as pd
from scipy.ndimage import label, binary_opening, binary_closing
from shapely.geometry import shape, box
from shapely.validation import make_valid
from pathlib import Path
import uuid
from datetime import date
import gc

CDL_PATH  = Path('candidate_areas/data/cdl/source/tx/CDL_2025_48.tif')
KZF_PATH  = Path('candidate_areas/outputs/keep_zones_final.parquet')
OUT_DIR   = Path('candidate_areas/data/cdl')
TILE_DIR  = OUT_DIR / 'tx_tiles'
OUT_PARQUET = Path('candidate_areas/outputs/states/tx/cdl_candidates_tx.parquet')
OUT_FGB     = Path('candidate_areas/outputs/states/tx/cdl_candidates_tx.fgb')

STATE      = 'TX'
TILE_ROWS  = 3       # split TX into 3x3 = 9 tiles (~185M px each, ~185 MB)
TILE_COLS  = 3
PIXEL_AREA = 30 * 30
ACRES_PER_M2 = 1 / 4046.856
MIN_PIXELS = 225
NODATA     = 0
STRUCTURE  = np.ones((3, 3), dtype=bool)

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

TILE_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load CDL metadata ------------------------------------------------------

print(f'\n{"="*55}')
print(f' CDL CANDIDATE GENERATION - TX (tiled {TILE_ROWS}x{TILE_COLS})')
print(f'{"="*55}')

with rasterio.open(CDL_PATH) as src:
    full_height   = src.height
    full_width    = src.width
    cdl_transform = src.transform
    cdl_crs       = src.crs
    cdl_profile   = src.profile.copy()
    cdl_bounds    = src.bounds

print(f'CDL shape: {full_height:,} x {full_width:,} = {full_height*full_width/1e9:.2f}B pixels')
print(f'CRS: {cdl_crs}')

tile_h = full_height // TILE_ROWS
tile_w = full_width  // TILE_COLS
print(f'Tile size: ~{tile_h:,} x {tile_w:,} = {tile_h*tile_w/1e6:.0f}M pixels (~{tile_h*tile_w/1e6:.0f} MB)')

# ---- Load keep_zones_final for TX -------------------------------------------

print('\nLoading keep_zones_final for TX bbox ...')
kzf = gpd.read_parquet(KZF_PATH)
xmin, ymin, xmax, ymax = cdl_bounds.left, cdl_bounds.bottom, cdl_bounds.right, cdl_bounds.top
kzf_tx = kzf.cx[xmin:xmax, ymin:ymax].copy()
del kzf
print(f'  {len(kzf_tx):,} rows | {kzf_tx.geometry.area.sum()/1e6:,.0f} km2')

pixel_area_acres = PIXEL_AREA * ACRES_PER_M2

# ---- Process each tile ------------------------------------------------------

all_candidates = []
total_tiles = TILE_ROWS * TILE_COLS
tile_num = 0

for row in range(TILE_ROWS):
    for col in range(TILE_COLS):
        tile_num += 1

        # Compute window (last tile may be slightly larger to cover remainder)
        row_off = row * tile_h
        col_off = col * tile_w
        t_h = tile_h if row < TILE_ROWS - 1 else full_height - row_off
        t_w = tile_w if col < TILE_COLS - 1 else full_width  - col_off
        window = Window(col_off, row_off, t_w, t_h)

        print(f'\n--- Tile {tile_num}/{total_tiles} (row={row},col={col}) '
              f'{t_h:,}x{t_w:,} ---')

        # Get tile transform and bounds
        tile_transform = rasterio.windows.transform(window, cdl_transform)
        tile_bounds    = rasterio.windows.bounds(window, cdl_transform)
        tb_xmin, tb_ymin, tb_xmax, tb_ymax = tile_bounds

        # Filter keep_zones_final to tile bbox
        kzf_tile = kzf_tx.cx[tb_xmin:tb_xmax, tb_ymin:tb_ymax]
        if len(kzf_tile) == 0:
            print(f'  No keep_zones_final in tile — skipping.')
            continue

        # Rasterize buildable mask for tile
        shapes_iter = ((geom, 1) for geom in kzf_tile.geometry
                       if geom is not None and not geom.is_empty)
        buildable_mask = rasterize(
            shapes=shapes_iter,
            out_shape=(t_h, t_w),
            transform=tile_transform,
            fill=0, dtype=np.uint8, all_touched=False,
        )

        n_buildable = int((buildable_mask == 1).sum())
        if n_buildable == 0:
            del buildable_mask
            continue
        print(f'  Buildable: {n_buildable:,} px ({n_buildable*PIXEL_AREA/1e6:.0f} km2)')

        # Load CDL tile and apply mask
        with rasterio.open(CDL_PATH) as src:
            cdl_tile = src.read(1, window=window)

        masked = np.where(buildable_mask == 1, cdl_tile, NODATA).astype(np.uint8)
        del cdl_tile, buildable_mask

        # Per-group connected components
        tile_profile = cdl_profile.copy()
        tile_profile.update({
            'width': t_w, 'height': t_h,
            'transform': tile_transform,
            'compress': 'lzw', 'nodata': NODATA,
        })

        for group_name, codes in GROUPS.items():
            group_mask = np.isin(masked, codes) & (masked != NODATA)
            if int(group_mask.sum()) == 0:
                continue

            group_mask = binary_opening(group_mask, structure=STRUCTURE)
            group_mask = binary_closing(group_mask, structure=STRUCTURE)

            labeled_array, n_comp = label(group_mask, structure=np.ones((3,3)))
            del group_mask

            counts = np.bincount(labeled_array.ravel())
            counts[0] = 0
            surviving = np.where(counts >= MIN_PIXELS)[0]

            if len(surviving) == 0:
                del labeled_array, counts
                continue

            remove_mask = counts < MIN_PIXELS
            remove_mask[0] = True
            filtered = labeled_array.copy()
            filtered[remove_mask[labeled_array]] = 0
            del labeled_array, counts

            # Polygonize
            geoms, label_ids = [], []
            for poly_geom, label_val in shapes(
                    filtered.astype(np.int32),
                    mask=(filtered > 0).astype(np.uint8),
                    transform=tile_transform):
                lv = int(label_val)
                if lv == 0:
                    continue
                geoms.append(make_valid(shape(poly_geom)))
                label_ids.append(lv)
            del filtered

            if not geoms:
                continue

            gdf = gpd.GeoDataFrame(
                {'label_id': label_ids, 'geometry': geoms}, crs=cdl_crs)
            gdf['area_m2']        = gdf.geometry.area
            gdf['pixel_count']    = (gdf['area_m2'] / PIXEL_AREA).round().astype(int)
            gdf['area_acres']     = gdf['area_m2'] * ACRES_PER_M2
            gdf['cdl_group']      = group_name
            gdf['cdl_group_label']= GROUP_LABELS[group_name]
            gdf['state']          = STATE

            centroids_5070 = gdf.geometry.centroid
            centroids_4326 = gpd.GeoSeries(centroids_5070, crs=cdl_crs).to_crs(4326)
            gdf['centroid_lon'] = centroids_4326.x.round(6)
            gdf['centroid_lat'] = centroids_4326.y.round(6)

            all_candidates.append(gdf)
            del gdf, geoms, label_ids

        del masked
        gc.collect()
        print(f'  Tile {tile_num} done. Running total: {sum(len(g) for g in all_candidates):,} polygons')

# ---- Combine + filter -------------------------------------------------------

print(f'\nCombining all tiles ...')
result = gpd.GeoDataFrame(pd.concat(all_candidates, ignore_index=True), crs=cdl_crs)
del all_candidates

before = len(result)
result = result[result['area_acres'] >= 50].copy()
print(f'  {before:,} -> {len(result):,} after re-filter')

result['candidate_id']          = [str(uuid.uuid4()) for _ in range(len(result))]
result['snapshot_date']         = date.today().isoformat()
result['candidate_type']        = 'greenfield'
result['route_complexity_score'] = None  # Owen Rec #5 — null until manually reviewed
result['route_complexity_notes'] = None
result['nlcd_class']                = None  # Owen Rec #11
result['nlcd_label']                = None
result['landcover_confidence_score'] = None

print(f'\n=== TX RESULTS ===')
print(f'  Total candidates: {len(result):,}')
print(f'  Total area: {result.area_acres.sum():,.0f} acres')
for g, grp in result.groupby('cdl_group'):
    print(f'    {g:<15} {len(grp):>6,} polys | {grp.area_acres.sum():>12,.0f} acres')

# ---- Save -------------------------------------------------------------------

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
    'TX centroids'    : result.centroid_lat.between(25, 37).all(),
}
all_pass = True
for lbl, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
    if not ok:
        all_pass = False

print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
print('\nVerify cdl_candidates_tx.fgb in QGIS.')
