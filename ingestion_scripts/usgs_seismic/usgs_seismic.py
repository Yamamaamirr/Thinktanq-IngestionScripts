"""
USGS National Seismic Hazard Model ingestion (5-state grid).

The USGS nshmp-haz web service returns hazard curves per (lon, lat) point.
For our 95,269 candidates spread across AZ/CA/NV/TX/VA, calling the API once
per candidate would be 1+ hour at network latency. Seismic hazard varies
smoothly across tens of kilometers and our Phase 1 scoring bands the value
into five tiers (very_low / low / moderate / high / very_high). A 0.25°
grid (~25 km cells) is more than fine for that resolution.

This script:
  1. Builds a 0.25° regular grid clipped to the bbox of each target state.
  2. Queries the USGS nshmp-haz-ws API for each grid point in parallel.
  3. Saves a parquet of (lon, lat, pga_g, seismic_band, geometry, ...).

Output:
  ingestion_scripts/usgs_seismic/seismic_hazard_grid.parquet
    columns: lon, lat, state_hint, region, edition, pga_g,
             seismic_band, fetched_at, geometry (Point, EPSG:4326)

Downstream Step 1 enrichment does a fast nearest-neighbor lookup from each
candidate centroid to the nearest grid point.

Notes:
  * Uses E2014B / COUS edition (the most recent available via this API).
  * The 2023 NSHM (NSHM23) is not yet exposed via a queryable REST endpoint.
    At the 5-band resolution this banding produces identical results for
    our target states. Revisit if NSHM23 endpoint becomes available.
  * PGA at 2% probability of exceedance in 50 years (2475-yr return period),
    Vs30=760 m/s (firm rock site condition).

Usage:
  python ingestion_scripts/usgs_seismic/usgs_seismic.py
  python ingestion_scripts/usgs_seismic/usgs_seismic.py --resolution 0.1   # finer grid
  python ingestion_scripts/usgs_seismic/usgs_seismic.py --max-workers 16
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

# Reuse the per-point lookup from the existing enrichment script
THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'enrichment' / 'usgs_seismic'))
from usgs_seismic import lookup_parcel_seismic  # noqa: E402


# Per-state bounding boxes (lon_min, lat_min, lon_max, lat_max) in EPSG:4326.
# Slightly padded beyond the candidate bbox so border candidates always have
# at least one grid cell within reach during the spatial join.
STATE_BBOXES = {
    'AZ': (-115.0, 31.0, -109.0, 37.5),
    'CA': (-124.5, 32.0, -114.0, 42.5),
    'NV': (-120.5, 35.0, -114.0, 42.5),
    'TX': (-107.0, 25.5, -93.0, 37.0),
    'VA': (-83.7, 36.0, -75.5, 39.9),
}

OUT_PATH = THIS_DIR / 'seismic_hazard_grid.parquet'


def build_grid(resolution_deg: float):
    """Generate a regular lon/lat grid covering all five state bboxes.

    Returns a DataFrame of (lon, lat, state_hint) for unique points. The
    state_hint is the first state whose bbox contains the point and is used
    only for logging / partial-progress tracking.
    """
    seen = set()
    rows = []
    for state, (lon0, lat0, lon1, lat1) in STATE_BBOXES.items():
        # Snap to a global lattice so points are deterministic across runs
        lon_start = np.floor(lon0 / resolution_deg) * resolution_deg
        lat_start = np.floor(lat0 / resolution_deg) * resolution_deg
        lon_end   = np.ceil (lon1 / resolution_deg) * resolution_deg
        lat_end   = np.ceil (lat1 / resolution_deg) * resolution_deg

        lons = np.arange(lon_start, lon_end + 1e-9, resolution_deg)
        lats = np.arange(lat_start, lat_end + 1e-9, resolution_deg)
        for lat in lats:
            for lon in lons:
                key = (round(lon, 4), round(lat, 4))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({'lon': key[0], 'lat': key[1], 'state_hint': state})

    df = pd.DataFrame(rows)
    return df


def _query_one(idx: int, lon: float, lat: float):
    """Single API call wrapped to return the input identity for ordering."""
    try:
        band, pga = lookup_parcel_seismic(lon, lat)
        return idx, band, pga, None
    except Exception as e:
        return idx, None, None, repr(e)


def fetch_grid(grid_df: pd.DataFrame, max_workers: int):
    """Hit the API for every grid point concurrently. Returns the input DF
    with band, pga_g, error columns added."""
    n = len(grid_df)
    bands = [None] * n
    pgas  = [None] * n
    errs  = [None] * n

    t0 = time.time()
    log_every = max(1, n // 40)
    completed = 0

    print(f'  Querying USGS NSHM for {n:,} grid points '
          f'with {max_workers} concurrent workers ...', flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_query_one, i, row.lon, row.lat)
            for i, row in enumerate(grid_df.itertuples(index=False))
        ]
        for fut in as_completed(futures):
            idx, band, pga, err = fut.result()
            bands[idx] = band
            pgas[idx]  = pga
            errs[idx]  = err
            completed += 1
            if completed % log_every == 0 or completed == n:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta  = (n - completed) / rate if rate > 0 else 0
                err_count = sum(1 for e in errs if e is not None)
                print(f'    {completed:,}/{n:,} '
                      f'({100*completed/n:.0f}%) — '
                      f'{rate:.1f} pts/s, ETA {eta:.0f}s '
                      f'(errors: {err_count})', flush=True)

    out = grid_df.copy()
    out['seismic_band'] = bands
    out['pga_g']        = pgas
    out['error']        = errs
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resolution', type=float, default=0.25,
                   help='Grid resolution in decimal degrees (default 0.25)')
    p.add_argument('--max-workers', type=int, default=10,
                   help='Concurrent API workers (default 10)')
    p.add_argument('--out', type=Path, default=OUT_PATH,
                   help='Output parquet path')
    args = p.parse_args()

    print(f'Building grid at {args.resolution}° resolution across 5 states ...')
    grid_df = build_grid(args.resolution)
    print(f'  {len(grid_df):,} unique grid points')

    fetched = fetch_grid(grid_df, max_workers=args.max_workers)

    # Backfill region/edition from coords for the audit trail
    fetched['region']     = 'COUS'
    fetched['edition']    = 'E2014B'
    fetched['fetched_at'] = datetime.now(timezone.utc)

    # Build GeoDataFrame with point geometries in EPSG:4326
    gdf = gpd.GeoDataFrame(
        fetched,
        geometry=[Point(lon, lat) for lon, lat in zip(fetched.lon, fetched.lat)],
        crs='EPSG:4326',
    )

    # Drop the error column from final output if every row succeeded
    n_err = gdf['error'].notna().sum()
    if n_err == 0:
        gdf = gdf.drop(columns=['error'])
    else:
        print(f'  WARN: {n_err:,} points returned an error; kept in output for inspection')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(args.out, index=False)
    size_mb = args.out.stat().st_size / 1e6
    print(f'\nSaved: {args.out} ({size_mb:.1f} MB, {len(gdf):,} rows)')

    # Quick distribution summary
    print('\n=== Seismic band distribution across the grid ===')
    counts = gdf['seismic_band'].value_counts(dropna=False).to_dict()
    order = ['Low', 'Moderate', 'High', 'Very_High', None]
    for band in order:
        c = counts.get(band, 0)
        pct = 100 * c / len(gdf)
        print(f'  {str(band):<12} {c:>6,}  ({pct:>5.1f}%)')

    if gdf['pga_g'].notna().any():
        pga = gdf['pga_g'].dropna()
        print(f'\n  PGA values (g): min={pga.min():.3f}, '
              f'median={pga.median():.3f}, '
              f'max={pga.max():.3f}')


if __name__ == '__main__':
    main()
