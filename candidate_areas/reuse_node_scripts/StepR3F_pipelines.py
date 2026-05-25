"""
StepR3F -- Pipeline distance, operator tier, estimated diameter per reuse node.

Mirror of Step1F_pipelines.py for reuse_nodes_clean.parquet.

Output columns identical to Step1F:
  nearest_pipeline_distance_m, nearest_pipeline_operator_tier,
  nearest_pipeline_est_diameter_in, pipeline_diameter_estimated,
  nearest_tier1_pipeline_distance_m, nearest_other_pipeline_distance_m

Run: python candidate_areas/reuse_node_scripts/StepR3F_pipelines.py
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

PIPE_PATH = Path('ingestion_scripts/eia_phmsa_pipelines/pipelines.parquet')
OUT_PATH  = out_path('stepR3f_pipelines.parquet')

MAX_SEARCH_M = 80_000.0
STATES = ['AZ','CA','NV','TX','VA']


def vectorized_nearest_with_attrs(cands_gdf, src_gdf, keep_cols=None):
    if len(src_gdf) == 0:
        return None
    keep = ['geometry'] + (keep_cols or [])
    result = gpd.sjoin_nearest(
        cands_gdf[['candidate_id','geometry']],
        src_gdf[keep],
        how='left', max_distance=MAX_SEARCH_M, distance_col='_dist',
    )
    result = result.sort_values('_dist').drop_duplicates(subset=['candidate_id'], keep='first')
    result = result.set_index('candidate_id').reindex(cands_gdf['candidate_id'].values)
    return result


def main():
    print('Loading reuse nodes ...')
    cands = load_reuse_nodes_as_candidates(crs_epsg=5070)
    print(f'  {len(cands):,} reuse nodes')

    print(f'\nLoading pipelines: {PIPE_PATH} ...')
    pipe = gpd.read_parquet(PIPE_PATH)
    if pipe.crs.to_epsg() != 5070:
        pipe = pipe.to_crs(5070)
    print(f'  {len(pipe):,} pipeline segments')

    pipe_tier1 = pipe[pipe.operator_tier == 'tier_1'].reset_index(drop=True)
    pipe_other = pipe[pipe.operator_tier != 'tier_1'].reset_index(drop=True)

    out_rows = []
    pad_box = 90_000
    for st in STATES:
        sd = cands[cands.state == st].reset_index(drop=True)
        if len(sd) == 0:
            continue
        print(f'\n-- {st} : {len(sd):,} reuse nodes -----------')

        bb = sd.total_bounds
        env = box(bb[0]-pad_box, bb[1]-pad_box, bb[2]+pad_box, bb[3]+pad_box)
        st_t1 = pipe_tier1[pipe_tier1.geometry.intersects(env)].reset_index(drop=True)
        st_ot = pipe_other[pipe_other.geometry.intersects(env)].reset_index(drop=True)

        print(f'  Nearest any-tier ({len(st_t1)+len(st_ot):,}) ...', flush=True)
        t0 = time.time()
        st_all = pd.concat([st_t1, st_ot], ignore_index=True)
        st_all_gdf = gpd.GeoDataFrame(st_all, crs=pipe.crs)
        all_res = vectorized_nearest_with_attrs(sd, st_all_gdf, keep_cols=['operator_tier','diameter_in'])
        print(f'    {time.time()-t0:.1f}s')

        print(f'  Nearest tier_1 ({len(st_t1):,}) ...', flush=True)
        t0 = time.time()
        t1_res = vectorized_nearest_with_attrs(sd, st_t1, keep_cols=[])
        d_tier1 = t1_res['_dist'].values if t1_res is not None else np.full(len(sd), np.nan)
        print(f'    {time.time()-t0:.1f}s')

        print(f'  Nearest other ({len(st_ot):,}) ...', flush=True)
        t0 = time.time()
        ot_res = vectorized_nearest_with_attrs(sd, st_ot, keep_cols=[])
        d_other = ot_res['_dist'].values if ot_res is not None else np.full(len(sd), np.nan)
        print(f'    {time.time()-t0:.1f}s')

        out_rows.append(pd.DataFrame({
            'candidate_id':                      sd.candidate_id.values,
            'nearest_pipeline_distance_m':       all_res['_dist'].values,
            'nearest_pipeline_operator_tier':    all_res['operator_tier'].values,
            'nearest_pipeline_est_diameter_in':  all_res['diameter_in'].values,
            'nearest_tier1_pipeline_distance_m': d_tier1,
            'nearest_other_pipeline_distance_m': d_other,
        }))

    out = pd.concat(out_rows, ignore_index=True)
    assert len(out) == len(cands)
    out['pipeline_diameter_estimated'] = True

    out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.2f} MB, {len(out):,} rows)')

    print('\n=== Distance medians (m) ===')
    for c in ['nearest_pipeline_distance_m','nearest_tier1_pipeline_distance_m','nearest_other_pipeline_distance_m']:
        s = out[c].dropna()
        print(f'  {c:<40} median={s.median():>9.0f}  beyond_80km={out[c].isna().sum():,}')
    print('\n=== Nearest operator_tier ===')
    print(out.nearest_pipeline_operator_tier.value_counts(dropna=False).to_string())

    print('\n=== Checks ===')
    checks = {
        'Has all reuse nodes'              : len(out) == len(cands),
        'Unique candidate_ids'             : out.candidate_id.is_unique,
        'pipeline_diameter_estimated True' : out.pipeline_diameter_estimated.all(),
        'nearest_pipeline non-neg or NaN'  : (out.nearest_pipeline_distance_m.fillna(0) >= 0).all(),
        'tier_1 dist non-neg or NaN'       : (out.nearest_tier1_pipeline_distance_m.fillna(0) >= 0).all(),
        'other dist non-neg or NaN'        : (out.nearest_other_pipeline_distance_m.fillna(0) >= 0).all(),
        'tier in {tier_1, other, NaN}'     : set(out.nearest_pipeline_operator_tier.dropna().unique()) <= {'tier_1','other'},
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
