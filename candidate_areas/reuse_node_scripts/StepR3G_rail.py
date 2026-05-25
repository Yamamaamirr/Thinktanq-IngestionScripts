"""
StepR3G -- Class 1 rail distance per reuse node.

Mirror of Step1G_rail.py for reuse_nodes_clean.parquet.

Output columns identical to Step1G:
  nearest_class1_rail_distance_m, nearest_rail_is_stracnet, nearest_rail_n_tracks

Run: python candidate_areas/reuse_node_scripts/StepR3G_rail.py
"""
from pathlib import Path
import sys
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent))
from _r3_helpers import load_reuse_nodes_as_candidates, out_path

RAIL_PATH = Path('ingestion_scripts/class1_rail/class1_rail.parquet')
OUT_PATH  = out_path('stepR3g_rail.parquet')

MAX_SEARCH_M = 50_000.0
STATES = ['AZ','CA','NV','TX','VA']


def main():
    print('Loading reuse nodes ...')
    cands = load_reuse_nodes_as_candidates(crs_epsg=5070)
    print(f'  {len(cands):,} reuse nodes')

    print(f'\nLoading Class 1 rail: {RAIL_PATH} ...')
    rail = gpd.read_parquet(RAIL_PATH)
    if rail.crs.to_epsg() != 5070:
        rail = rail.to_crs(5070)
    print(f'  {len(rail):,} rail segments')

    out_rows = []
    pad_box = 60_000
    for st in STATES:
        sd = cands[cands.state == st].reset_index(drop=True)
        if len(sd) == 0:
            continue
        print(f'\n-- {st} : {len(sd):,} reuse nodes -----------')

        bb = sd.total_bounds
        env = box(bb[0]-pad_box, bb[1]-pad_box, bb[2]+pad_box, bb[3]+pad_box)
        st_rail = rail[rail.geometry.intersects(env)].reset_index(drop=True)

        print(f'  Nearest ({len(st_rail):,}) ...', flush=True)
        t0 = time.time()
        result = gpd.sjoin_nearest(
            sd[['candidate_id','geometry']],
            st_rail[['is_stracnet','n_tracks','geometry']],
            how='left', max_distance=MAX_SEARCH_M, distance_col='_dist',
        )
        result = result.sort_values('_dist').drop_duplicates(subset=['candidate_id'], keep='first')
        result = result.set_index('candidate_id').reindex(sd['candidate_id'].values)
        print(f'    {time.time()-t0:.1f}s')

        out_rows.append(pd.DataFrame({
            'candidate_id':                   sd.candidate_id.values,
            'nearest_class1_rail_distance_m': result['_dist'].values,
            'nearest_rail_is_stracnet':       result['is_stracnet'].values,
            'nearest_rail_n_tracks':          result['n_tracks'].values,
        }))

    out = pd.concat(out_rows, ignore_index=True)
    assert len(out) == len(cands)

    out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.2f} MB, {len(out):,} rows)')

    s = out.nearest_class1_rail_distance_m.dropna()
    print(f'\n  median={s.median():.0f}m  beyond_50km={out.nearest_class1_rail_distance_m.isna().sum():,}')
    print(f'  Nearest STRACNET: {int(out.nearest_rail_is_stracnet.sum()):,}')

    print('\n=== Checks ===')
    checks = {
        'Has all reuse nodes'    : len(out) == len(cands),
        'Unique candidate_ids'   : out.candidate_id.is_unique,
        'Distance non-neg or NaN': (out.nearest_class1_rail_distance_m.fillna(0) >= 0).all(),
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
