"""
Per-reuse-node slope_mean_pct and slope_max_pct ingestion.

Adapted from ingest_slope_per_candidate.py for the reuse-node layer:
  * Reads reuse_nodes_clean.parquet (6,631 polygons) instead of the
    greenfield candidate set.
  * Tiles over the per-state bounding box of reuse-node polygons
    (with a 0.05-deg pad) instead of using keep_zones, since reuse
    nodes don't have a candidate keep-zone layer.
  * Output keyed on `site_id` (the reuse-node identifier) to match
    StepR3A consumption.

Output:
  ingestion_scripts/usgs_slope/slope_per_reuse_node.parquet
    columns: site_id, slope_mean_pct, slope_max_pct, n_pixels, ingested_at

Tile fetching, slope computation, and per-polygon sampling are
byte-identical to the greenfield ingest, so slope values are directly
comparable between layers.

Run: python ingestion_scripts/usgs_slope/ingest_slope_per_reuse_node.py
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

EXPORT_URL = (
    'https://elevation.nationalmap.gov/arcgis/rest/services/'
    '3DEPElevation/ImageServer/exportImage'
)
RES_DEG  = 0.0009
MAX_PX   = 2000
STATES = ['AZ', 'CA', 'NV', 'TX', 'VA']

REUSE_PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_clean.parquet')
OUT_PATH   = Path('ingestion_scripts/usgs_slope/slope_per_reuse_node.parquet')
BBOX_PAD_DEG = 0.05   # ~5 km padding around state-level reuse-node bbox


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


print(f'Loading reuse nodes: {REUSE_PATH} ...')
reuse = gpd.read_parquet(REUSE_PATH).to_crs('EPSG:4326')
print(f'  Loaded {len(reuse):,} reuse-node polygons')

reuse_sindex = reuse.sindex
stats = {sid: CandStats() for sid in reuse['site_id'].values}
print(f'  {len(stats):,} per-reuse-node accumulators ready')

tile_count = 0
hit_count  = 0
fail_count = 0
skip_count = 0

for state in STATES:
    print(f'\n-- {state} --------------------------------------------')
    state_sub = reuse[reuse['state_abbr'] == state]
    if len(state_sub) == 0:
        print('  No reuse nodes in this state - skipping.')
        continue
    x0, y0, x1, y1 = state_sub.total_bounds
    x0 -= BBOX_PAD_DEG; y0 -= BBOX_PAD_DEG
    x1 += BBOX_PAD_DEG; y1 += BBOX_PAD_DEG

    tiles = list(tile_bboxes(x0, y0, x1, y1))
    print(f'  {len(state_sub):,} reuse nodes, bbox '
          f'({x0:.3f},{y0:.3f}) -> ({x1:.3f},{y1:.3f})')
    print(f'  {len(tiles)} tile(s) to download')

    for ti, (tx0, ty0, tx1, ty1) in enumerate(tiles, 1):
        tile_count += 1
        tile_box = box(tx0, ty0, tx1, ty1)

        prefilter_idx = list(reuse_sindex.intersection((tx0, ty0, tx1, ty1)))
        if not prefilter_idx:
            skip_count += 1
            continue

        tile_idx = [idx for idx in prefilter_idx
                    if reuse.geometry.iloc[idx].intersects(tile_box)]
        if not tile_idx:
            skip_count += 1
            continue

        print(f'  Tile {ti}/{len(tiles)} ({tx0:.2f},{ty0:.2f}) -> {len(tile_idx)} polys ...',
              flush=True)

        data, pw, ph = fetch_elevation_tile(tx0, ty0, tx1, ty1)
        if data is None:
            print('    FAILED download - skipping tile')
            fail_count += 1
            continue

        with rasterio.open(io.BytesIO(data)) as src:
            elev = src.read(1).astype(np.float32)
            tfm  = src.transform
            nodata = src.nodata if src.nodata is not None else -9999
            raster_shape = elev.shape

        valid_mask = elev != nodata
        if not valid_mask.any():
            continue

        elev_clean = np.where(valid_mask, elev, np.nan)
        slope = compute_slope_pct(elev_clean, tfm)

        sampled = 0
        for idx in tile_idx:
            row = reuse.iloc[idx]
            sid = row['site_id']
            try:
                mask = geometry_mask(
                    [row.geometry], out_shape=raster_shape, transform=tfm,
                    invert=True, all_touched=True,
                )
            except Exception:
                continue
            inside = mask & valid_mask & ~np.isnan(slope)
            if not inside.any():
                continue
            vals = slope[inside]
            stats[sid].update(vals)
            sampled += 1
            hit_count += 1

        print(f'    Sampled {sampled} polygon(s)', flush=True)


print(f'\nTiles downloaded: {tile_count}  |  failures: {fail_count}  |  skipped: {skip_count}')
print(f'Per-tile polygon samples (cumulative): {hit_count:,}')

rows = []
for sid in reuse['site_id'].values:
    mean_v, max_v, n_px = stats[sid].finalize()
    rows.append({
        'site_id':        sid,
        'slope_mean_pct': mean_v,
        'slope_max_pct':  max_v,
        'n_pixels':       n_px,
    })
out = pd.DataFrame(rows)
out['ingested_at'] = datetime.now(timezone.utc)

n_with = out.slope_mean_pct.notna().sum()
print(f'\nReuse nodes with slope data: {n_with:,} / {len(out):,} '
      f'({100*n_with/len(out):.1f}%)')
if n_with > 0:
    s = out.loc[out.slope_mean_pct.notna()]
    print(f'  slope_mean_pct: min={s.slope_mean_pct.min():.2f}, '
          f'p10={s.slope_mean_pct.quantile(0.1):.2f}, '
          f'median={s.slope_mean_pct.median():.2f}, '
          f'p90={s.slope_mean_pct.quantile(0.9):.2f}, '
          f'max={s.slope_mean_pct.max():.2f}')
    print(f'  slope_max_pct:  min={s.slope_max_pct.min():.2f}, '
          f'median={s.slope_max_pct.median():.2f}, '
          f'max={s.slope_max_pct.max():.2f}')

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT_PATH, index=False)
print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.2f} MB)')

print('\n=== Checks ===')
checks = {
    'Has all reuse nodes'   : len(out) == len(reuse),
    'Unique site_ids'       : out.site_id.is_unique,
    'slope_mean non-neg or NaN': (out.slope_mean_pct.fillna(0) >= 0).all(),
    'slope_max  non-neg or NaN': (out.slope_max_pct.fillna(0)  >= 0).all(),
    'Most rows have data'   : n_with / len(out) > 0.95,
}
all_pass = True
for lbl, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
    if not ok:
        all_pass = False
print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
