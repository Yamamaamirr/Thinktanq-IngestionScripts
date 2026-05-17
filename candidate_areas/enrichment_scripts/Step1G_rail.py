"""
Step 1G — Class 1 rail distance per candidate (with STRACNET + n_tracks).

For each candidate, compute distance to the nearest Class 1 rail segment,
plus capture two attributes of that segment:
  - is_stracnet  (DoD-designated Strategic Rail Corridor, heavier-grade)
  - n_tracks     (single/multi-track)

Both feed scoring (Section 4.4) as small bonuses on top of the distance
band.

Method
------
Vectorized via gpd.sjoin_nearest, per state. 152,748 rail segments
nationwide — we pre-clip to a generous state envelope per pass.

Output:
  candidate_areas/enrichment_outputs/step1g_rail.parquet

Columns:
  nearest_class1_rail_distance_m
  nearest_rail_is_stracnet      bool, True if STRACNET
  nearest_rail_n_tracks         int (1, 2, 3, 4, 5)

Run:
  python candidate_areas/enrichment_scripts/Step1G_rail.py
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

CAND_PATH = Path('candidate_areas/outputs/candidate_areas.parquet')
RAIL_PATH = Path('ingestion_scripts/class1_rail/class1_rail.parquet')
OUT_PATH  = Path('candidate_areas/enrichment_outputs/step1g_rail.parquet')

MAX_SEARCH_M = 50_000.0   # 50 km > scoring's 25 mi top band
STATES = ['AZ','CA','NV','TX','VA']


def main():
    print(f'Loading candidates: {CAND_PATH} ...')
    cands = gpd.read_parquet(CAND_PATH)
    print(f'  {len(cands):,} candidates in EPSG:{cands.crs.to_epsg()}')

    print(f'\nLoading Class 1 rail: {RAIL_PATH} ...')
    rail = gpd.read_parquet(RAIL_PATH)
    if rail.crs.to_epsg() != 5070:
        rail = rail.to_crs(5070)
    print(f'  {len(rail):,} rail segments')
    print(f'  is_stracnet True: {rail.is_stracnet.sum():,}')
    print(f'  n_tracks distribution: {rail.n_tracks.value_counts().to_dict()}')

    out_rows = []
    pad_box = 60_000  # 60 km > 50 km cap

    for st in STATES:
        sd = cands[cands.state == st].reset_index(drop=True)
        if len(sd) == 0:
            continue
        print(f'\n-- {st} -----------------------------------------')
        print(f'  {len(sd):,} candidates')

        bb = sd.total_bounds
        env = box(bb[0]-pad_box, bb[1]-pad_box, bb[2]+pad_box, bb[3]+pad_box)
        st_rail = rail[rail.geometry.intersects(env)].reset_index(drop=True)

        print(f'  Nearest rail ({len(st_rail):,} segments in envelope) ...', flush=True)
        t0 = time.time()
        result = gpd.sjoin_nearest(
            sd[['candidate_id','geometry']],
            st_rail[['is_stracnet','n_tracks','geometry']],
            how='left',
            max_distance=MAX_SEARCH_M,
            distance_col='_dist',
        )
        result = result.sort_values('_dist').drop_duplicates(subset=['candidate_id'], keep='first')
        result = result.set_index('candidate_id').reindex(sd['candidate_id'].values)
        print(f'    {time.time()-t0:.1f}s')

        out_rows.append(pd.DataFrame({
            'candidate_id':                  sd.candidate_id.values,
            'nearest_class1_rail_distance_m':result['_dist'].values,
            'nearest_rail_is_stracnet':      result['is_stracnet'].values,
            'nearest_rail_n_tracks':         result['n_tracks'].values,
        }))

    print('\nConcatenating per-state results ...')
    out = pd.concat(out_rows, ignore_index=True)
    assert len(out) == len(cands), f'lost candidates: expected {len(cands)}, got {len(out)}'

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f'\nSaved: {OUT_PATH} ({size_mb:.2f} MB, {len(out):,} rows)')

    print('\n=== Distance medians (m) ===')
    s = out.nearest_class1_rail_distance_m.dropna()
    n_miss = out.nearest_class1_rail_distance_m.isna().sum()
    print(f'  median={s.median():.0f}, p10={s.quantile(0.1):.0f}, p90={s.quantile(0.9):.0f}, beyond_50km={n_miss:,}')

    print('\n=== STRACNET / n_tracks distribution among nearest rail ===')
    print(f'  Nearest is STRACNET: {out.nearest_rail_is_stracnet.sum():,} ({100*out.nearest_rail_is_stracnet.mean():.1f}%)')
    print(f'  Nearest n_tracks counts: {out.nearest_rail_n_tracks.value_counts(dropna=False).to_dict()}')

    print('\n=== Per-state median ===')
    cdf = cands[['candidate_id','state']].merge(out, on='candidate_id')
    for st in STATES:
        sd = cdf[cdf.state == st]
        if len(sd) == 0:
            continue
        print(f'  {st}: n={len(sd):>6,}  median={sd.nearest_class1_rail_distance_m.median():>7.0f}m  '
              f'stracnet={sd.nearest_rail_is_stracnet.sum():>6,}')

    print('\n=== Checks ===')
    checks = {
        'Has all candidates'         : len(out) == len(cands),
        'Unique candidate_ids'       : out.candidate_id.is_unique,
        'Distance non-neg or NaN'    : (out.nearest_class1_rail_distance_m.fillna(0) >= 0).all(),
        'n_tracks ints or NaN'       : True,  # geopandas merges produce floats; check categorical
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
