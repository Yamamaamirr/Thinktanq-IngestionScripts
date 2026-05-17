"""
Step 1H — Water service area per candidate.

For each candidate:
  - within_water_service_area     True if candidate centroid is inside any
                                  water service polygon
  - nearest_water_service_distance_m   0 if inside, else distance to nearest
                                       service area boundary
  - nearest_water_service_pop_served   pop_served of the inside-or-nearest district

Method
------
Two-pass per state:
  1) sjoin with predicate='intersects' against candidate centroids — finds the
     district containing each centroid.
  2) For centroids NOT inside any district, sjoin_nearest to get the distance
     to the nearest district boundary plus its pop_served.

No simplification.

Output:
  candidate_areas/enrichment_outputs/step1h_water.parquet

Run:
  python candidate_areas/enrichment_scripts/Step1H_water.py
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

CAND_PATH  = Path('candidate_areas/outputs/candidate_areas.parquet')
WATER_PATH = Path('ingestion_scripts/water_districts/water_districts.parquet')
OUT_PATH   = Path('candidate_areas/enrichment_outputs/step1h_water.parquet')

MAX_SEARCH_M = 30_000.0   # 30 km > scoring's 15 mi top band
STATES = ['AZ','CA','NV','TX','VA']


def main():
    print(f'Loading candidates: {CAND_PATH} ...')
    cands = gpd.read_parquet(CAND_PATH)
    print(f'  {len(cands):,} candidates in EPSG:{cands.crs.to_epsg()}')

    print(f'\nLoading water districts: {WATER_PATH} ...')
    water = gpd.read_parquet(WATER_PATH)
    if water.crs.to_epsg() != 5070:
        water = water.to_crs(5070)
    # Drop any with null/empty geometry
    water = water[water.geometry.notna() & ~water.geometry.is_empty]
    print(f'  {len(water):,} water districts')
    print(f'  pop_served stats: median={water.pop_served.median():.0f}, max={water.pop_served.max():.0f}')

    # Candidate centroids as POINTS for inside/distance ops
    print('\nComputing candidate centroids ...')
    cents = gpd.GeoDataFrame(
        cands[['candidate_id','state']],
        geometry=cands.geometry.centroid,
        crs=cands.crs,
    )

    out_rows = []
    pad_box = 40_000

    for st in STATES:
        sd = cents[cents.state == st].reset_index(drop=True)
        if len(sd) == 0:
            continue
        print(f'\n-- {st} -----------------------------------------')
        print(f'  {len(sd):,} candidate centroids')

        bb = sd.total_bounds
        env = box(bb[0]-pad_box, bb[1]-pad_box, bb[2]+pad_box, bb[3]+pad_box)
        st_water = water[water.geometry.intersects(env)].reset_index(drop=True)
        print(f'  Water districts in envelope: {len(st_water):,}')

        # Pass 1: inside check (point-in-polygon)
        print(f'  Inside check ...', flush=True)
        t0 = time.time()
        inside = gpd.sjoin(
            sd[['candidate_id','geometry']],
            st_water[['pop_served','geometry']],
            how='left', predicate='within',
        )
        # Deduplicate on candidate_id keeping first (overlapping districts: take max pop_served)
        inside = inside.sort_values('pop_served', ascending=False).drop_duplicates(subset=['candidate_id'], keep='first')
        inside = inside.set_index('candidate_id').reindex(sd['candidate_id'].values)
        is_inside = inside.index_right.notna().values  # index_right NaN means no containing polygon
        inside_pop = inside['pop_served'].values
        print(f'    {time.time()-t0:.1f}s ({is_inside.sum():,} centroids inside a district)')

        # Pass 2: nearest distance for outside-points
        outside_idx = np.where(~is_inside)[0]
        d_out = np.full(len(sd), 0.0)
        pop_out = inside_pop.copy()

        if len(outside_idx) > 0:
            outside_pts = sd.iloc[outside_idx]
            print(f'  Nearest distance for {len(outside_idx):,} outside centroids ...', flush=True)
            t0 = time.time()
            near = gpd.sjoin_nearest(
                outside_pts[['candidate_id','geometry']],
                st_water[['pop_served','geometry']],
                how='left',
                max_distance=MAX_SEARCH_M,
                distance_col='_dist',
            )
            near = near.sort_values('_dist').drop_duplicates(subset=['candidate_id'], keep='first')
            near = near.set_index('candidate_id').reindex(outside_pts['candidate_id'].values)
            d_out_subset = near['_dist'].values
            pop_out_subset = near['pop_served'].values
            d_out[outside_idx] = d_out_subset
            pop_out[outside_idx] = pop_out_subset
            print(f'    {time.time()-t0:.1f}s')

        out_rows.append(pd.DataFrame({
            'candidate_id':                       sd.candidate_id.values,
            'within_water_service_area':          is_inside,
            'nearest_water_service_distance_m':   d_out,
            'nearest_water_service_pop_served':   pop_out,
        }))

    print('\nConcatenating per-state results ...')
    out = pd.concat(out_rows, ignore_index=True)
    assert len(out) == len(cands), f'lost candidates: expected {len(cands)}, got {len(out)}'

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f'\nSaved: {OUT_PATH} ({size_mb:.2f} MB, {len(out):,} rows)')

    print('\n=== Inside flag counts ===')
    print(f'  within_water_service_area True: {out.within_water_service_area.sum():,} ({100*out.within_water_service_area.mean():.1f}%)')

    print('\n=== Distance distribution for outside-of-district candidates ===')
    s = out[~out.within_water_service_area].nearest_water_service_distance_m.dropna()
    if len(s):
        print(f'  median={s.median():.0f}m, p10={s.quantile(0.1):.0f}, p90={s.quantile(0.9):.0f}')

    print('\n=== Per-state distribution ===')
    cdf = cands[['candidate_id','state']].merge(out, on='candidate_id')
    for st in STATES:
        sd = cdf[cdf.state == st]
        if len(sd) == 0:
            continue
        n_in = sd.within_water_service_area.sum()
        out_dist = sd[~sd.within_water_service_area].nearest_water_service_distance_m.median()
        print(f'  {st}: n={len(sd):>6,}  inside={n_in:>5,} ({100*n_in/len(sd):4.1f}%)  '
              f'median outside dist={out_dist:.0f}m')

    print('\n=== Checks ===')
    checks = {
        'Has all candidates'                : len(out) == len(cands),
        'Unique candidate_ids'              : out.candidate_id.is_unique,
        'Distance non-neg'                  : (out.nearest_water_service_distance_m.fillna(0) >= 0).all(),
        'Inside iff distance=0'             : ((out.within_water_service_area == False) | (out.nearest_water_service_distance_m == 0)).all(),
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
