"""
Step 1E — Transmission line distance per candidate.

For each candidate, compute the distance from the polygon edge to the nearest
transmission line in each of the three voltage classes we care about per the
scoring design (Section 4.2) and Owen Rec #6:

  500 kV   highest score   (685 lines)
  345 kV   strong score    (2,113 lines)
  230 kV   secondary score (5,482 lines)

We also compute a "crosses" flag (distance == 0) per voltage class since the
scoring table treats line-crossing as a separate top-tier band.

Method
------
Vectorized via gpd.sjoin_nearest, per voltage class. max_distance = 50 km
since the scoring band caps at 25 mi (~40 km) for the regional tier. NO
geometry simplification anywhere — full line precision preserved.

Output:
  candidate_areas/enrichment_outputs/step1e_transmission.parquet

Columns per candidate:
  nearest_500kv_distance_m
  nearest_345kv_distance_m
  nearest_230kv_distance_m
  crosses_500kv_flag
  crosses_345kv_flag
  crosses_230kv_flag

Run:
  python candidate_areas/enrichment_scripts/Step1E_transmission.py
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import geopandas as gpd

CAND_PATH = Path('candidate_areas/outputs/candidate_areas.parquet')
TX_PATH   = Path('ingestion_scripts/hifld_transmission_lines/hifld_transmission_lines.parquet')
OUT_PATH  = Path('candidate_areas/enrichment_outputs/step1e_transmission.parquet')

MAX_SEARCH_M = 50_000.0  # 50 km cap — beyond the regional band of 25 mi (~40 km)
STATES = ['AZ','CA','NV','TX','VA']

# Voltage class assignments — match the scoring design (4.2)
# Note transmission file has some 220 kV (81 lines) which we lump with 230 kV
# per HIFLD docs treating them as the same nominal class for system planning.
VOLT_CLASSES = {
    '500kv': [500.0, 765.0],            # 500 kV+, including 765 kV
    '345kv': [345.0, 450.0],            # 345 kV (and rare 450)
    '230kv': [230.0, 220.0, 236.0, 250.0],  # 230 kV class (incl. 220/236/250)
}


def vectorized_nearest(cands_gdf, src_gdf):
    """Same vectorized sjoin_nearest as Step 1D. Returns ndarray of distances."""
    if len(src_gdf) == 0:
        return np.full(len(cands_gdf), np.nan)
    result = gpd.sjoin_nearest(
        cands_gdf[['candidate_id','geometry']],
        src_gdf[['geometry']],
        how='left',
        max_distance=MAX_SEARCH_M,
        distance_col='_dist',
    )
    result = result.sort_values('_dist').drop_duplicates(subset=['candidate_id'], keep='first')
    result = result.set_index('candidate_id').reindex(cands_gdf['candidate_id'].values)
    return result['_dist'].values


def main():
    print(f'Loading candidates: {CAND_PATH} ...')
    cands = gpd.read_parquet(CAND_PATH)
    print(f'  {len(cands):,} candidates in EPSG:{cands.crs.to_epsg()}')

    print(f'\nLoading transmission lines: {TX_PATH} ...')
    tx = gpd.read_parquet(TX_PATH)
    if tx.crs.to_epsg() != 5070:
        tx = tx.to_crs(5070)
    print(f'  {len(tx):,} transmission line segments in EPSG:{tx.crs.to_epsg()}')
    print(f'  Voltage distribution (top 10):')
    print(tx.voltage_kv.value_counts().head(10).to_string())

    # Split by voltage class
    print('\nSplitting by voltage class ...')
    tx_500 = tx[tx.voltage_kv.isin(VOLT_CLASSES['500kv'])].reset_index(drop=True)
    tx_345 = tx[tx.voltage_kv.isin(VOLT_CLASSES['345kv'])].reset_index(drop=True)
    tx_230 = tx[tx.voltage_kv.isin(VOLT_CLASSES['230kv'])].reset_index(drop=True)
    print(f'  500 kV+:  {len(tx_500):,}')
    print(f'  345 kV:   {len(tx_345):,}')
    print(f'  230 kV:   {len(tx_230):,}')

    # Per-state processing (so sjoin doesn't include irrelevant out-of-state lines)
    out_rows = []
    from shapely.geometry import box
    pad_box = 60_000  # 60 km padding > our 50 km cap

    for st in STATES:
        sd = cands[cands.state == st].reset_index(drop=True)
        if len(sd) == 0:
            continue
        print(f'\n-- {st} -----------------------------------------')
        print(f'  {len(sd):,} candidates')

        bb = sd.total_bounds
        env = box(bb[0]-pad_box, bb[1]-pad_box, bb[2]+pad_box, bb[3]+pad_box)
        sub_500 = tx_500[tx_500.geometry.intersects(env)].reset_index(drop=True)
        sub_345 = tx_345[tx_345.geometry.intersects(env)].reset_index(drop=True)
        sub_230 = tx_230[tx_230.geometry.intersects(env)].reset_index(drop=True)

        print(f'  500kV ({len(sub_500):,} lines in envelope) ...', flush=True)
        t0 = time.time()
        d500 = vectorized_nearest(sd, sub_500)
        print(f'    {time.time()-t0:.1f}s')

        print(f'  345kV ({len(sub_345):,} lines in envelope) ...', flush=True)
        t0 = time.time()
        d345 = vectorized_nearest(sd, sub_345)
        print(f'    {time.time()-t0:.1f}s')

        print(f'  230kV ({len(sub_230):,} lines in envelope) ...', flush=True)
        t0 = time.time()
        d230 = vectorized_nearest(sd, sub_230)
        print(f'    {time.time()-t0:.1f}s')

        out_rows.append(pd.DataFrame({
            'candidate_id':            sd.candidate_id.values,
            'nearest_500kv_distance_m': d500,
            'nearest_345kv_distance_m': d345,
            'nearest_230kv_distance_m': d230,
        }))

    print('\nConcatenating per-state results ...')
    out = pd.concat(out_rows, ignore_index=True)
    assert len(out) == len(cands), f'lost candidates: expected {len(cands)}, got {len(out)}'

    # Crosses flag = distance == 0 (line crosses polygon)
    out['crosses_500kv_flag'] = (out.nearest_500kv_distance_m == 0)
    out['crosses_345kv_flag'] = (out.nearest_345kv_distance_m == 0)
    out['crosses_230kv_flag'] = (out.nearest_230kv_distance_m == 0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f'\nSaved: {OUT_PATH} ({size_mb:.2f} MB, {len(out):,} rows)')

    # ---- Distribution summary ----
    print('\n=== Distance medians (m) ===')
    for c in ['nearest_500kv_distance_m','nearest_345kv_distance_m','nearest_230kv_distance_m']:
        s = out[c].dropna()
        n_miss = out[c].isna().sum()
        print(f'  {c:<32} median={s.median():>10.0f}  p10={s.quantile(0.1):>8.0f}  '
              f'p90={s.quantile(0.9):>8.0f}  beyond_50km={n_miss:,}')

    print('\n=== Crosses flag counts ===')
    for c in ['crosses_500kv_flag','crosses_345kv_flag','crosses_230kv_flag']:
        n = out[c].sum()
        print(f'  {c:<28} True: {n:>6,} ({100*n/len(out):5.1f}%)')

    print('\n=== Per-state median distance (m) ===')
    cdf = cands[['candidate_id','state']].merge(out, on='candidate_id')
    print(f'  {"state":<5} {"n":>7} {"d500_p50":>10} {"d345_p50":>10} {"d230_p50":>10}')
    for st in STATES:
        sd = cdf[cdf.state == st]
        if len(sd) == 0:
            continue
        print(f'  {st:<5} {len(sd):>7,} '
              f'{sd.nearest_500kv_distance_m.median():>9.0f}m '
              f'{sd.nearest_345kv_distance_m.median():>9.0f}m '
              f'{sd.nearest_230kv_distance_m.median():>9.0f}m')

    print('\n=== Checks ===')
    checks = {
        'Has all candidates'                  : len(out) == len(cands),
        'Unique candidate_ids'                : out.candidate_id.is_unique,
        '500kV dist non-neg or NaN'           : (out.nearest_500kv_distance_m.fillna(0) >= 0).all(),
        '345kV dist non-neg or NaN'           : (out.nearest_345kv_distance_m.fillna(0) >= 0).all(),
        '230kV dist non-neg or NaN'           : (out.nearest_230kv_distance_m.fillna(0) >= 0).all(),
        'crosses_flag iff dist=0 (500kV)'     : ((out.nearest_500kv_distance_m == 0) == out.crosses_500kv_flag).all(),
        'crosses_flag iff dist=0 (345kV)'     : ((out.nearest_345kv_distance_m == 0) == out.crosses_345kv_flag).all(),
        'crosses_flag iff dist=0 (230kV)'     : ((out.nearest_230kv_distance_m == 0) == out.crosses_230kv_flag).all(),
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
