"""
Targeted backfill for candidates that got no slope data from the tile-based ingest.

The bulk tile-based ingest (ingest_slope_per_candidate.py) leaves a small number
of candidates with NaN slope when their polygons don't catch any valid pixel
centers (3DEP coverage gaps along the VA/MD border for Loudoun + Clarke counties
in our run). For those, we use the per-point USGS 3DEP getSamples API from
enrichment/usgs_slope/usgs_3dep_slope.py to sample a fine grid inside each
polygon and compute slope per-cell.

The output is merged back into slope_per_candidate.parquet so the scoring
engine sees one consistent file.

Run:
  python ingestion_scripts/usgs_slope/fill_null_slope_via_api.py
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Import the per-parcel lookup from the enrichment script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'enrichment' / 'usgs_slope'))
from usgs_3dep_slope import lookup_parcel_slope  # noqa: E402

SLOPE_PATH = Path('ingestion_scripts/usgs_slope/slope_per_candidate.parquet')
CAND_PATH = Path('candidate_areas/outputs/candidate_areas.parquet')


def polygon_to_rings(geom):
    """Convert a Shapely Polygon to Esri-style rings: [[[lon, lat], ...]]."""
    if geom.geom_type == 'MultiPolygon':
        # Take the largest sub-polygon
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != 'Polygon':
        raise ValueError(f'Unsupported geom type: {geom.geom_type}')
    rings = [list(geom.exterior.coords)]
    for interior in geom.interiors:
        rings.append(list(interior.coords))
    return rings


def main():
    print(f'Loading {SLOPE_PATH} ...')
    slope = pd.read_parquet(SLOPE_PATH)
    n_null = slope.slope_mean_pct.isna().sum()
    print(f'  Current null count: {n_null:,}')

    if n_null == 0:
        print('  Nothing to fill. Done.')
        return

    print(f'Loading {CAND_PATH} ...')
    cands = gpd.read_parquet(CAND_PATH).to_crs('EPSG:4326')

    null_ids = slope.loc[slope.slope_mean_pct.isna(), 'candidate_id'].tolist()
    null_cands = cands[cands.candidate_id.isin(null_ids)].copy()
    print(f'  Matched {len(null_cands)} candidate geometries to backfill')

    t0 = time.time()
    results = {}
    for i, row in enumerate(null_cands.itertuples(index=False), 1):
        try:
            rings = polygon_to_rings(row.geometry)
            disq, avg, p70, mx, band = lookup_parcel_slope(rings)
            results[row.candidate_id] = {
                'slope_mean_pct': avg,
                'slope_max_pct': mx,
                'n_pixels': None,  # API path does not expose pixel count
            }
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(f'  {i}/{len(null_cands)} '
                  f'{row.candidate_id[:8]} {row.county_name:<15} '
                  f'mean={avg:>6.2f}% max={mx:>6.2f}%  '
                  f'({rate:.1f}/s, ETA {(len(null_cands)-i)/max(rate,0.001):.0f}s)')
        except Exception as e:
            print(f'  {i}/{len(null_cands)} '
                  f'{row.candidate_id[:8]} ERROR: {type(e).__name__}: {e}')
            results[row.candidate_id] = {
                'slope_mean_pct': None, 'slope_max_pct': None, 'n_pixels': None,
            }

    # Merge back into the parquet
    print('\nMerging results back into slope_per_candidate.parquet ...')
    backfill_df = pd.DataFrame.from_dict(results, orient='index')
    backfill_df.index.name = 'candidate_id'
    backfill_df = backfill_df.reset_index()

    slope_indexed = slope.set_index('candidate_id')
    for cid, row in backfill_df.set_index('candidate_id').iterrows():
        slope_indexed.loc[cid, 'slope_mean_pct'] = row.slope_mean_pct
        slope_indexed.loc[cid, 'slope_max_pct']  = row.slope_max_pct
        if 'n_pixels' in slope_indexed.columns and row.n_pixels is not None:
            slope_indexed.loc[cid, 'n_pixels'] = row.n_pixels

    slope_out = slope_indexed.reset_index()
    slope_out['ingested_at'] = pd.Timestamp.now(tz='UTC')

    n_null_after = slope_out.slope_mean_pct.isna().sum()
    print(f'  Null count before: {n_null:,}')
    print(f'  Null count after:  {n_null_after:,}')
    print(f'  Backfilled:        {n_null - n_null_after:,}')

    slope_out.to_parquet(SLOPE_PATH, index=False)
    print(f'\nSaved: {SLOPE_PATH}')


if __name__ == '__main__':
    main()
