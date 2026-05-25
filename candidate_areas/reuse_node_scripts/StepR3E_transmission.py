"""
StepR3E -- Transmission line distance per reuse node (500/345/230 kV).

Mirror of Step1E_transmission.py for reuse_nodes_clean.parquet.

Output columns identical to Step1E:
  nearest_500kv_distance_m, nearest_345kv_distance_m, nearest_230kv_distance_m,
  crosses_500kv_flag, crosses_345kv_flag, crosses_230kv_flag

Run: python candidate_areas/reuse_node_scripts/StepR3E_transmission.py
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

TX_PATH  = Path('ingestion_scripts/hifld_transmission_lines/hifld_transmission_lines.parquet')
OUT_PATH = out_path('stepR3e_transmission.parquet')

MAX_SEARCH_M = 50_000.0
STATES = ['AZ','CA','NV','TX','VA']

VOLT_CLASSES = {
    '500kv': [500.0, 765.0],
    '345kv': [345.0, 450.0],
    '230kv': [230.0, 220.0, 236.0, 250.0],
}


def vectorized_nearest(cands_gdf, src_gdf):
    if len(src_gdf) == 0:
        return np.full(len(cands_gdf), np.nan)
    result = gpd.sjoin_nearest(
        cands_gdf[['candidate_id','geometry']],
        src_gdf[['geometry']],
        how='left', max_distance=MAX_SEARCH_M, distance_col='_dist',
    )
    result = result.sort_values('_dist').drop_duplicates(subset=['candidate_id'], keep='first')
    result = result.set_index('candidate_id').reindex(cands_gdf['candidate_id'].values)
    return result['_dist'].values


def main():
    print('Loading reuse nodes ...')
    cands = load_reuse_nodes_as_candidates(crs_epsg=5070)
    print(f'  {len(cands):,} reuse nodes')

    print(f'\nLoading transmission lines: {TX_PATH} ...')
    tx = gpd.read_parquet(TX_PATH)
    if tx.crs.to_epsg() != 5070:
        tx = tx.to_crs(5070)
    print(f'  {len(tx):,} transmission line segments')

    tx_500 = tx[tx.voltage_kv.isin(VOLT_CLASSES['500kv'])].reset_index(drop=True)
    tx_345 = tx[tx.voltage_kv.isin(VOLT_CLASSES['345kv'])].reset_index(drop=True)
    tx_230 = tx[tx.voltage_kv.isin(VOLT_CLASSES['230kv'])].reset_index(drop=True)
    print(f'  500kV+: {len(tx_500):,}  345kV: {len(tx_345):,}  230kV: {len(tx_230):,}')

    out_rows = []
    pad_box = 60_000
    for st in STATES:
        sd = cands[cands.state == st].reset_index(drop=True)
        if len(sd) == 0:
            continue
        print(f'\n-- {st} : {len(sd):,} reuse nodes -----------')

        bb = sd.total_bounds
        env = box(bb[0]-pad_box, bb[1]-pad_box, bb[2]+pad_box, bb[3]+pad_box)
        sub_500 = tx_500[tx_500.geometry.intersects(env)].reset_index(drop=True)
        sub_345 = tx_345[tx_345.geometry.intersects(env)].reset_index(drop=True)
        sub_230 = tx_230[tx_230.geometry.intersects(env)].reset_index(drop=True)

        print(f'  500kV ({len(sub_500):,}) ...', flush=True)
        t0 = time.time(); d500 = vectorized_nearest(sd, sub_500); print(f'    {time.time()-t0:.1f}s')
        print(f'  345kV ({len(sub_345):,}) ...', flush=True)
        t0 = time.time(); d345 = vectorized_nearest(sd, sub_345); print(f'    {time.time()-t0:.1f}s')
        print(f'  230kV ({len(sub_230):,}) ...', flush=True)
        t0 = time.time(); d230 = vectorized_nearest(sd, sub_230); print(f'    {time.time()-t0:.1f}s')

        out_rows.append(pd.DataFrame({
            'candidate_id':             sd.candidate_id.values,
            'nearest_500kv_distance_m': d500,
            'nearest_345kv_distance_m': d345,
            'nearest_230kv_distance_m': d230,
        }))

    out = pd.concat(out_rows, ignore_index=True)
    assert len(out) == len(cands)
    out['crosses_500kv_flag'] = (out.nearest_500kv_distance_m == 0)
    out['crosses_345kv_flag'] = (out.nearest_345kv_distance_m == 0)
    out['crosses_230kv_flag'] = (out.nearest_230kv_distance_m == 0)

    out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.2f} MB, {len(out):,} rows)')

    print('\n=== Distance medians (m) ===')
    for c in ['nearest_500kv_distance_m','nearest_345kv_distance_m','nearest_230kv_distance_m']:
        s = out[c].dropna()
        print(f'  {c:<32} median={s.median():>9.0f}  beyond_50km={out[c].isna().sum():,}')
    print('\n=== Crosses flags ===')
    for c in ['crosses_500kv_flag','crosses_345kv_flag','crosses_230kv_flag']:
        n = int(out[c].sum())
        print(f'  {c:<28} {n:>5,} ({100*n/len(out):5.1f}%)')

    print('\n=== Checks ===')
    checks = {
        'Has all reuse nodes'  : len(out) == len(cands),
        'Unique candidate_ids' : out.candidate_id.is_unique,
        'crosses_500 iff dist=0': ((out.nearest_500kv_distance_m == 0) == out.crosses_500kv_flag).all(),
        'crosses_345 iff dist=0': ((out.nearest_345kv_distance_m == 0) == out.crosses_345kv_flag).all(),
        'crosses_230 iff dist=0': ((out.nearest_230kv_distance_m == 0) == out.crosses_230kv_flag).all(),
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
