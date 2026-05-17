"""
Per-candidate slope_mean_pct and slope_max_pct ingestion.

Uses the same USGS 3DEP ImageServer tile approach as
candidate_areas/scripts/Step5e_exclusion_slope.py, but instead of
thresholding at 15% and saving a polygon mask of the excluded steep
areas, this samples the slope raster per candidate polygon and saves
the mean and max slope value for every candidate.

Output:
  ingestion_scripts/usgs_slope/slope_per_candidate.parquet
    columns: candidate_id, slope_mean_pct, slope_max_pct, n_pixels, ingested_at

Slope is computed exactly the same way as Step5e (elevation gradients
in metres, converted to %), so this is a single re-pass over the same
tiles. The slope values are in the 0-15% range for the surviving
candidates (>15% was already hard-excluded upstream).

Run: python ingestion_scripts/usgs_slope/ingest_slope_per_candidate.py
"""

import io
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import geometry_mask
from shapely.geometry import box

# ---- Config (identical to Step5e for raster compatibility) -------------------

EXPORT_URL = (
    'https://elevation.nationalmap.gov/arcgis/rest/services/'
    '3DEPElevation/ImageServer/exportImage'
)
RES_DEG  = 0.0009   # ~100 m at mid-latitudes (matches Step5e)
MAX_PX   = 2000     # tile dimension cap

STATES = ['AZ', 'CA', 'NV', 'TX', 'VA']

CAND_PATH = Path('candidate_areas/outputs/candidate_areas.parquet')
STATES_PATH = Path('ingestion_scripts/census_tiger/state_boundaries.parquet')
KEEP_PATH = Path('candidate_areas/outputs/keep_zones.parquet')
OUT_PATH = Path('ingestion_scripts/usgs_slope/slope_per_candidate.parquet')

# Per-candidate accumulators.  We don't store every pixel value; we keep
# running stats so memory is bounded regardless of polygon size.
#   sum_slope     : Σ slope values  (for mean)
#   sum_sq_slope  : Σ slope² values (for std if needed later)
#   n_pixels      : pixel count
#   max_slope     : running maximum
class CandStats:
    __slots__ = ('sum_v', 'sum_sq', 'n', 'mx')
    def __init__(self):
        self.sum_v  = 0.0
        self.sum_sq = 0.0
        self.n      = 0
        self.mx     = 0.0
    def update(self, vals):
        if vals.size == 0:
            return
        self.sum_v  += float(vals.sum())
        self.sum_sq += float((vals * vals).sum())
        self.n      += int(vals.size)
        self.mx     = max(self.mx, float(vals.max()))
    def finalize(self):
        if self.n == 0:
            return (None, None, 0)
        return (self.sum_v / self.n, self.mx, self.n)


# ---- Tile helpers (identical to Step5e) --------------------------------------

def fetch_elevation_tile(xmin, ymin, xmax, ymax, retries=5):
    w = max(2, min(MAX_PX, round((xmax - xmin) / RES_DEG)))
    h = max(2, min(MAX_PX, round((ymax - ymin) / RES_DEG)))
    params = {
        'bbox': f'{xmin},{ymin},{xmax},{ymax}',
        'bboxSR': 4326, 'size': f'{w},{h}',
        'imageSR': 4326, 'format': 'tiff',
        'pixelType': 'F32', 'noData': -9999, 'f': 'image',
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
    nrows, ncols = elev.shape
    centre_lat = transform.f + transform.e * (nrows / 2)
    y_m = abs(transform.e) * 111_000.0
    x_m = abs(transform.a) * 111_000.0 * math.cos(math.radians(centre_lat))
    dz_dy, dz_dx = np.gradient(elev, y_m, x_m)
    return np.sqrt(dz_dx**2 + dz_dy**2) * 100.0


def tile_bboxes(xmin, ymin, xmax, ymax):
    tw = MAX_PX * RES_DEG
    th = MAX_PX * RES_DEG
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            yield (x, y, min(x + tw, xmax), min(y + th, ymax))
            y += th
        x += tw


# ---- Main --------------------------------------------------------------------

print('Loading candidates ...')
cands = gpd.read_parquet(CAND_PATH).to_crs('EPSG:4326')
print(f'  Loaded {len(cands):,} candidates')

print('Loading keep_zones ...')
keep_gdf = gpd.read_parquet(KEEP_PATH).to_crs('EPSG:4326')
keep_geom = keep_gdf.geometry.iloc[0]
if not keep_geom.is_valid:
    keep_geom = keep_geom.buffer(0)

print('Loading state boundaries ...')
states_gdf = (gpd.read_parquet(STATES_PATH)
              .query('state_code in @STATES')
              .to_crs('EPSG:4326'))

# Spatial index of candidates for fast tile-candidate intersection
cand_sindex = cands.sindex

# Per-candidate accumulators
stats = {cid: CandStats() for cid in cands['candidate_id'].values}
print(f'  {len(stats):,} per-candidate accumulators ready')

# Map candidate_id -> position in cands DataFrame for fast geometry lookup
id_to_pos = {cid: i for i, cid in enumerate(cands['candidate_id'].values)}

tile_count = 0
hit_count  = 0
fail_count = 0

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

    for ti, (tx0, ty0, tx1, ty1) in enumerate(tiles, 1):
        tile_count += 1
        tile_box = box(tx0, ty0, tx1, ty1)

        # Find candidates whose bbox intersects this tile (cheap bbox prefilter
        # via spatial index, then exact intersects check).
        prefilter_idx = list(cand_sindex.intersection((tx0, ty0, tx1, ty1)))
        if not prefilter_idx:
            print(f'  Tile {ti}/{len(tiles)} ({tx0:.3f},{ty0:.3f}) — '
                  f'no candidates in bbox, skipping download', flush=True)
            continue

        # Tight intersects check (drops candidates whose bbox intersects but geom does not)
        tile_cands_idx = []
        for idx in prefilter_idx:
            if cands.geometry.iloc[idx].intersects(tile_box):
                tile_cands_idx.append(idx)
        if not tile_cands_idx:
            print(f'  Tile {ti}/{len(tiles)} ({tx0:.3f},{ty0:.3f}) — '
                  f'no candidate geometries in tile, skipping download', flush=True)
            continue

        print(f'  Tile {ti}/{len(tiles)} ({tx0:.3f},{ty0:.3f}) — '
              f'{len(tile_cands_idx)} cands, downloading ...', flush=True)

        data, pw, ph = fetch_elevation_tile(tx0, ty0, tx1, ty1)
        if data is None:
            print('    FAILED download — skipping tile.')
            fail_count += 1
            continue

        with rasterio.open(io.BytesIO(data)) as src:
            elev = src.read(1).astype(np.float32)
            tfm  = src.transform
            nodata = src.nodata if src.nodata is not None else -9999
            raster_shape = elev.shape

        valid_mask = elev != nodata
        if not valid_mask.any():
            print('    All nodata — skipped.')
            continue

        elev_clean = np.where(valid_mask, elev, np.nan)
        slope = compute_slope_pct(elev_clean, tfm)

        # For each candidate intersecting this tile, build a polygon mask and
        # sample slope values inside it.
        sampled = 0
        for idx in tile_cands_idx:
            row = cands.iloc[idx]
            cid = row['candidate_id']
            # Build a single-polygon mask in the tile's raster grid.
            try:
                mask = geometry_mask(
                    [row.geometry], out_shape=raster_shape, transform=tfm,
                    invert=True, all_touched=False,
                )
            except Exception as e:
                print(f'    mask error for {cid[:8]}: {type(e).__name__}: {e}')
                continue

            inside = mask & valid_mask & ~np.isnan(slope)
            if not inside.any():
                continue
            vals = slope[inside]
            stats[cid].update(vals)
            sampled += 1
            hit_count += 1

        print(f'    Sampled {sampled:,} candidate(s) from this tile', flush=True)


# ---- Materialize output ------------------------------------------------------

print(f'\nMaterializing per-candidate stats ...')
print(f'  Tiles downloaded: {tile_count} | failures: {fail_count}')
print(f'  Per-tile candidate samples (cumulative): {hit_count:,}')

rows = []
for cid in cands['candidate_id'].values:
    mean_v, max_v, n_px = stats[cid].finalize()
    rows.append({
        'candidate_id': cid,
        'slope_mean_pct': mean_v,
        'slope_max_pct':  max_v,
        'n_pixels':       n_px,
    })

out = pd.DataFrame(rows)
out['ingested_at'] = datetime.now(timezone.utc)

n_with_data = out['slope_mean_pct'].notna().sum()
print(f'  Candidates with slope data: {n_with_data:,} / {len(out):,}')
if n_with_data > 0:
    s = out.loc[out['slope_mean_pct'].notna()]
    print(f'  slope_mean_pct: min={s.slope_mean_pct.min():.2f}, '
          f'median={s.slope_mean_pct.median():.2f}, '
          f'max={s.slope_mean_pct.max():.2f}')
    print(f'  slope_max_pct:  min={s.slope_max_pct.min():.2f}, '
          f'median={s.slope_max_pct.median():.2f}, '
          f'max={s.slope_max_pct.max():.2f}')

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT_PATH, index=False)
size_mb = OUT_PATH.stat().st_size / 1e6
print(f'\nSaved: {OUT_PATH} ({size_mb:.2f} MB)')

# ---- Checks ----
print('\n=== Checks ===')
n = len(out)
checks = {
    'Has all candidates'    : n == len(cands),
    'Unique candidate_ids'  : out['candidate_id'].is_unique,
    'Slope mean range 0-15' : (out['slope_mean_pct'].fillna(0).between(0, 25)).all(),
    'Slope max range 0-25'  : (out['slope_max_pct'].fillna(0).between(0, 100)).all(),
    'Most cands have data'  : n_with_data / n > 0.95,
}
all_pass = True
for lbl, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
    if not ok:
        all_pass = False
print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
