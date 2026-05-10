"""
Step 5e - USGS 3DEP slope exclusion layer for target states.

Downloads raw elevation tiles from the USGS 3DEP ImageServer, computes slope
from elevation gradients, thresholds at SLOPE_THRESHOLD_PCT (15%), vectorizes,
clips to keep_zones, saves as slope_gt15.parquet (EPSG:5070).

Owen Rec #13 dual thresholds:
  >15%  -> hard exclusion (this script, slope_gt15.parquet)
  8-15% -> review flag + scoring penalty (Step9 only, NOT stored here)

Run:  python candidate_areas/Step5e_exclusion_slope.py

Output: candidate_areas/exclusion_layers/exclusion_slope.parquet
"""

import io
import math
import time

import geopandas as gpd
import numpy as np
import requests
import rasterio
from rasterio.features import shapes
from shapely.geometry import box, shape
from shapely.ops import unary_union
from pathlib import Path

# ---- Config ------------------------------------------------------------------

SLOPE_THRESHOLD_PCT = 15.0   # hard exclude per PRD

EXPORT_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer/exportImage"
)

RES_DEG  = 0.0009   # ~100 m at mid-latitudes
MAX_PX   = 2000     # max tile dimension — 2000x2000=4M px stays within server limits

STATES = ['AZ', 'CA', 'TX', 'NV', 'VA']

KEEP_PATH   = Path('candidate_areas/outputs/keep_zones.parquet')
STATES_PATH = Path('ingestion_scripts/census_tiger/state_boundaries.parquet')
OUT_PATH    = Path('candidate_areas/exclusion_layers/exclusion_slope.parquet')

# ---- Helpers -----------------------------------------------------------------

def fetch_elevation_tile(xmin, ymin, xmax, ymax, retries=5):
    """Download raw elevation GeoTIFF bytes from 3DEP. Returns (bytes, w, h) or (None, None, None)."""
    w = max(2, min(MAX_PX, round((xmax - xmin) / RES_DEG)))
    h = max(2, min(MAX_PX, round((ymax - ymin) / RES_DEG)))
    params = {
        'bbox'     : f'{xmin},{ymin},{xmax},{ymax}',
        'bboxSR'   : 4326,
        'size'     : f'{w},{h}',
        'imageSR'  : 4326,
        'format'   : 'tiff',
        'pixelType': 'F32',
        'noData'   : -9999,
        'f'        : 'image',
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(EXPORT_URL, params=params, timeout=180)
            r.raise_for_status()
            ct = r.headers.get('Content-Type', '')
            if 'image' in ct or 'octet-stream' in ct:
                return r.content, w, h
            print(f'    Non-image response: {r.text[:200]}')
        except Exception as exc:
            print(f'    Attempt {attempt} error: {exc}')
        time.sleep(5 * attempt)
    return None, None, None


def compute_slope_pct(elev, transform):
    """Compute slope (%) from elevation array using gradients in metres."""
    nrows, ncols = elev.shape
    centre_lat_deg = transform.f + transform.e * (nrows / 2)
    y_m = abs(transform.e) * 111_000.0
    x_m = abs(transform.a) * 111_000.0 * math.cos(math.radians(centre_lat_deg))
    dz_dy, dz_dx = np.gradient(elev, y_m, x_m)
    return np.sqrt(dz_dx**2 + dz_dy**2) * 100.0


def tile_bboxes(xmin, ymin, xmax, ymax):
    """Yield (x0, y0, x1, y1) non-overlapping tiles covering the bbox."""
    tw = MAX_PX * RES_DEG
    th = MAX_PX * RES_DEG
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            yield (x, y, min(x + tw, xmax), min(y + th, ymax))
            y += th
        x += tw


def steep_polys_from_tile(tiff_bytes, threshold_pct):
    """Return list of shapely polygons where slope > threshold_pct, with verification stats."""
    with rasterio.open(io.BytesIO(tiff_bytes)) as src:
        elev   = src.read(1).astype(np.float32)
        tfm    = src.transform
        nodata = src.nodata if src.nodata is not None else -9999

    valid_mask = elev != nodata
    if not valid_mask.any():
        print('    (all nodata - skipped)')
        return []

    elev_clean = np.where(valid_mask, elev, np.nan)
    slope = compute_slope_pct(elev_clean, tfm)

    s_valid = slope[valid_mask & ~np.isnan(slope)]
    if s_valid.size:
        print(f'    elev {float(elev[valid_mask].min()):.0f}-'
              f'{float(elev[valid_mask].max()):.0f} m | '
              f'slope {float(s_valid.min()):.1f}-{float(s_valid.max()):.1f}% | '
              f'{(s_valid > threshold_pct).mean()*100:.1f}% pixels steep')

    steep_mask = (slope > threshold_pct) & valid_mask & ~np.isnan(slope)
    if not steep_mask.any():
        return []

    return [shape(geom)
            for geom, val in shapes(steep_mask.astype(np.uint8), transform=tfm)
            if val == 1]

# ---- Main --------------------------------------------------------------------

print(f'Threshold: slope > {SLOPE_THRESHOLD_PCT}%')

print('\nLoading keep_zones ...')
keep_gdf  = gpd.read_parquet(KEEP_PATH).to_crs('EPSG:4326')
keep_geom = keep_gdf.geometry.iloc[0]
if not keep_geom.is_valid:
    keep_geom = keep_geom.buffer(0)   # fix self-intersections from unary_union
print(f'  CRS: {keep_gdf.crs} | rows: {len(keep_gdf)} | valid: {keep_geom.is_valid}')

print('Loading state boundaries ...')
states_gdf = (gpd.read_parquet(STATES_PATH)
              .query('state_code in @STATES')
              .to_crs('EPSG:4326'))
print(f'  States loaded: {sorted(states_gdf["state_code"].tolist())}')

all_steep = []

for state in STATES:
    print(f'\n-- {state} --------------------------------------------')
    state_geom = states_gdf.loc[states_gdf['state_code'] == state, 'geometry'].iloc[0]
    state_keep = keep_geom.intersection(state_geom)

    if state_keep.is_empty:
        print('  No keep_zones overlap - skipping.')
        continue

    x0, y0, x1, y1 = state_keep.bounds
    tiles = list(tile_bboxes(x0, y0, x1, y1))
    print(f'  Keep-zone bbox: ({x0:.3f},{y0:.3f}) to ({x1:.3f},{y1:.3f})')
    print(f'  {len(tiles)} tile(s) to download')

    clipped_parts = []
    for ti, (tx0, ty0, tx1, ty1) in enumerate(tiles, 1):
        print(f'  Tile {ti}/{len(tiles)} ({tx0:.3f},{ty0:.3f} to {tx1:.3f},{ty1:.3f}) ...', flush=True)
        data, pw, ph = fetch_elevation_tile(tx0, ty0, tx1, ty1)
        if data is None:
            print('    FAILED - skipping tile.')
            continue
        print(f'    Downloaded {len(data):,} bytes ({pw}x{ph} px)', end=' | ', flush=True)
        polys = steep_polys_from_tile(data, SLOPE_THRESHOLD_PCT)
        print(f'    {len(polys)} raw steep polygons')
        if not polys:
            continue
        # Clip keep_zones to this tile bbox first — tiny local geometry, fast intersection
        tile_box  = box(tx0, ty0, tx1, ty1)
        keep_tile = state_keep.intersection(tile_box)
        if keep_tile.is_empty:
            continue
        # Union + simplify steep polys, then clip to local keep_zones slice
        tile_u    = unary_union(polys).simplify(0.0001, preserve_topology=True)
        clipped   = tile_u.intersection(keep_tile)
        if not clipped.is_empty:
            clipped_parts.append(clipped)
            a_km2 = clipped.area * (111_000**2) / 1e6
            print(f'    Clipped steep in keep_zones: ~{a_km2:.1f} km2')

    if not clipped_parts:
        print(f'  No slope > {SLOPE_THRESHOLD_PCT}% found in {state} keep_zones.')
        continue

    # Skip state-level union — save each tile result as a separate row.
    # The subtraction script handles multi-row GeoDataFrames fine.
    a_km2 = sum(g.area for g in clipped_parts) * (111_000**2) / 1e6
    print(f'  {state} done - ~{a_km2:,.0f} km2 total, {len(clipped_parts)} tile parts')
    all_steep.extend(clipped_parts)

# ---- Assemble and save -------------------------------------------------------

print('\nSaving results ...')
if not all_steep:
    print('WARNING: no steep areas found - check service and threshold.')
else:
    # Keep as multi-row GeoDataFrame — avoids expensive global unary_union.
    # The subtraction script uses gpd.overlay which handles multi-row inputs fine.
    gdf_out  = gpd.GeoDataFrame({'geometry': all_steep}, crs='EPSG:4326')
    gdf_out  = gdf_out[gdf_out.geometry.is_valid & ~gdf_out.geometry.is_empty].copy()
    gdf_out  = gdf_out.to_crs('EPSG:5070')

    area_km2 = gdf_out.geometry.area.sum() / 1e6
    print(f'  Rows: {len(gdf_out)} tile parts | Total steep area: {area_km2:,.1f} km2')

    gdf_out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH}')

    checks = {
        'Has rows'         : len(gdf_out) > 0,
        'CRS EPSG:5070'    : gdf_out.crs.to_epsg() == 5070,
        'No null geometry' : gdf_out.geometry.isna().sum() == 0,
        'No empty geometry': (~gdf_out.geometry.is_empty).all(),
        'Area > 0'         : area_km2 > 0,
        'Area < keep_zones': area_km2 < 932_456,
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
    print('\nVerify slope_gt15.parquet in QGIS - should show steep terrain in')
    print('mountain ranges (Sierra Nevada, Appalachians, AZ/NV mountains).')
    print('Then subtract: python candidate_areas/Step5e_subtract_slope.py')
