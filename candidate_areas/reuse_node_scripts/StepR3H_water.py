"""
StepR3H -- Water service area per reuse node.

Mirror of Step1H_water.py for reuse_nodes_clean.parquet.

Output columns identical to Step1H:
  within_water_service_area, nearest_water_service_distance_m,
  nearest_water_service_pop_served

Run: python candidate_areas/reuse_node_scripts/StepR3H_water.py
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent))
from _r3_helpers import load_reuse_nodes_as_candidates, out_path

WATER_PATH = Path('ingestion_scripts/water_districts/water_districts.parquet')
OUT_PATH   = out_path('stepR3h_water.parquet')

MAX_SEARCH_M = 30_000.0
STATES = ['AZ','CA','NV','TX','VA']


def main():
    print('Loading reuse nodes ...')
    cands = load_reuse_nodes_as_candidates(crs_epsg=5070)
    print(f'  {len(cands):,} reuse nodes')

    print(f'\nLoading water districts: {WATER_PATH} ...')
    water = gpd.read_parquet(WATER_PATH)
    if water.crs.to_epsg() != 5070:
        water = water.to_crs(5070)
    water = water[water.geometry.notna() & ~water.geometry.is_empty]
    print(f'  {len(water):,} districts')

    cents = gpd.GeoDataFrame(
        cands[['candidate_id','state']],
        geometry=cands.geometry.centroid, crs=cands.crs,
    )

    out_rows = []
    pad_box = 40_000
    for st in STATES:
        sd = cents[cents.state == st].reset_index(drop=True)
        if len(sd) == 0:
            continue
        print(f'\n-- {st} : {len(sd):,} centroids -----------')

        bb = sd.total_bounds
        env = box(bb[0]-pad_box, bb[1]-pad_box, bb[2]+pad_box, bb[3]+pad_box)
        st_water = water[water.geometry.intersects(env)].reset_index(drop=True)
        print(f'  Districts in envelope: {len(st_water):,}')

        print(f'  Inside check ...', flush=True)
        t0 = time.time()
        inside = gpd.sjoin(
            sd[['candidate_id','geometry']],
            st_water[['pop_served','geometry']],
            how='left', predicate='within',
        )
        inside = inside.sort_values('pop_served', ascending=False).drop_duplicates(subset=['candidate_id'], keep='first')
        inside = inside.set_index('candidate_id').reindex(sd['candidate_id'].values)
        is_inside = inside.index_right.notna().values
        inside_pop = inside['pop_served'].values
        print(f'    {time.time()-t0:.1f}s ({is_inside.sum():,} inside)')

        outside_idx = np.where(~is_inside)[0]
        d_out = np.full(len(sd), 0.0)
        pop_out = inside_pop.copy()

        if len(outside_idx) > 0:
            outside_pts = sd.iloc[outside_idx]
            print(f'  Nearest for {len(outside_idx):,} outside ...', flush=True)
            t0 = time.time()
            near = gpd.sjoin_nearest(
                outside_pts[['candidate_id','geometry']],
                st_water[['pop_served','geometry']],
                how='left', max_distance=MAX_SEARCH_M, distance_col='_dist',
            )
            near = near.sort_values('_dist').drop_duplicates(subset=['candidate_id'], keep='first')
            near = near.set_index('candidate_id').reindex(outside_pts['candidate_id'].values)
            d_out[outside_idx] = near['_dist'].values
            pop_out[outside_idx] = near['pop_served'].values
            print(f'    {time.time()-t0:.1f}s')

        out_rows.append(pd.DataFrame({
            'candidate_id':                     sd.candidate_id.values,
            'within_water_service_area':        is_inside,
            'nearest_water_service_distance_m': d_out,
            'nearest_water_service_pop_served': pop_out,
        }))

    out = pd.concat(out_rows, ignore_index=True)
    assert len(out) == len(cands)

    out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.2f} MB, {len(out):,} rows)')

    n_in = int(out.within_water_service_area.sum())
    print(f'\n  inside: {n_in:,} ({100*n_in/len(out):.1f}%)')
    s = out[~out.within_water_service_area].nearest_water_service_distance_m.dropna()
    if len(s):
        print(f'  median outside dist: {s.median():.0f}m')

    print('\n=== Checks ===')
    checks = {
        'Has all reuse nodes'    : len(out) == len(cands),
        'Unique candidate_ids'   : out.candidate_id.is_unique,
        'Distance non-neg'       : (out.nearest_water_service_distance_m.fillna(0) >= 0).all(),
        'Inside implies dist=0'  : ((out.within_water_service_area == False) | (out.nearest_water_service_distance_m == 0)).all(),
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
