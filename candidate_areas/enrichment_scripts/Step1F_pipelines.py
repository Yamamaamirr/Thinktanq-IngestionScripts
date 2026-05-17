"""
Step 1F — Pipeline distance + tier + estimated diameter per candidate.

For each candidate, compute the distance to the nearest pipeline in each
operator-tier category, plus the diameter of the nearest pipeline.

Owen May 2 instruction: diameter is labeled as estimated, must not be used
as a hard filter. We expose both the value and the estimated flag.

Method
------
Vectorized via gpd.sjoin_nearest. Two sjoin calls per state — one for
tier_1 (major trunk operators ≥20") and one for "other" (everything else).
No simplification of pipeline geometries.

Our pipelines.parquet has only two operator_tier values:
  tier_1   11,579 segments  (Transco, Tennessee, El Paso, REX, etc.)
  other    21,392 segments  (smaller / intrastate)

The scoring matrix uses three tiers (tier_1 / tier_2 / other) — we map our
data to the scoring engine by diameter:
  tier_1 in scoring = operator_tier=tier_1
  tier_2 in scoring = operator_tier=other AND diameter_in 14-19
  other  in scoring = operator_tier=other AND diameter_in <14
The scoring engine will derive the tier label from nearest_pipeline_operator_tier
and nearest_pipeline_est_diameter_in.

Output:
  candidate_areas/enrichment_outputs/step1f_pipelines.parquet

Columns:
  nearest_pipeline_distance_m         distance to nearest pipeline of ANY tier
  nearest_pipeline_operator_tier      'tier_1' / 'other' of the nearest
  nearest_pipeline_est_diameter_in    estimated diameter of the nearest
  pipeline_diameter_estimated         True always (label flag, Owen May 2)
  nearest_tier1_pipeline_distance_m   distance to nearest tier_1 pipeline
  nearest_other_pipeline_distance_m   distance to nearest non-tier-1 pipeline

Run:
  python candidate_areas/enrichment_scripts/Step1F_pipelines.py
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

CAND_PATH = Path('candidate_areas/outputs/candidate_areas.parquet')
PIPE_PATH = Path('ingestion_scripts/eia_phmsa_pipelines/pipelines.parquet')
OUT_PATH  = Path('candidate_areas/enrichment_outputs/step1f_pipelines.parquet')

MAX_SEARCH_M = 80_000.0   # 80 km cap (50 mi) — beyond the scoring matrix top band
STATES = ['AZ','CA','NV','TX','VA']


def vectorized_nearest_with_attrs(cands_gdf, src_gdf, keep_cols=None):
    """sjoin_nearest returning distance + selected attributes from the source."""
    if len(src_gdf) == 0:
        return None
    keep = ['geometry'] + (keep_cols or [])
    result = gpd.sjoin_nearest(
        cands_gdf[['candidate_id','geometry']],
        src_gdf[keep],
        how='left',
        max_distance=MAX_SEARCH_M,
        distance_col='_dist',
    )
    result = result.sort_values('_dist').drop_duplicates(subset=['candidate_id'], keep='first')
    result = result.set_index('candidate_id').reindex(cands_gdf['candidate_id'].values)
    return result


def main():
    print(f'Loading candidates: {CAND_PATH} ...')
    cands = gpd.read_parquet(CAND_PATH)
    print(f'  {len(cands):,} candidates in EPSG:{cands.crs.to_epsg()}')

    print(f'\nLoading pipelines: {PIPE_PATH} ...')
    pipe = gpd.read_parquet(PIPE_PATH)
    if pipe.crs.to_epsg() != 5070:
        pipe = pipe.to_crs(5070)
    print(f'  {len(pipe):,} pipeline segments')
    print(f'  operator_tier counts: {pipe.operator_tier.value_counts().to_dict()}')
    print(f'  diameter_estimated all True: {pipe.diameter_estimated.all()}')

    pipe_tier1 = pipe[pipe.operator_tier == 'tier_1'].reset_index(drop=True)
    pipe_other = pipe[pipe.operator_tier != 'tier_1'].reset_index(drop=True)

    out_rows = []
    pad_box = 90_000  # 90 km > our 80 km cap

    for st in STATES:
        sd = cands[cands.state == st].reset_index(drop=True)
        if len(sd) == 0:
            continue
        print(f'\n-- {st} -----------------------------------------')
        print(f'  {len(sd):,} candidates')

        bb = sd.total_bounds
        env = box(bb[0]-pad_box, bb[1]-pad_box, bb[2]+pad_box, bb[3]+pad_box)
        st_t1 = pipe_tier1[pipe_tier1.geometry.intersects(env)].reset_index(drop=True)
        st_ot = pipe_other[pipe_other.geometry.intersects(env)].reset_index(drop=True)

        # Nearest of ANY tier — joined to keep attrs
        print(f'  Nearest pipeline ({len(st_t1)+len(st_ot):,} all-tier lines in envelope) ...', flush=True)
        t0 = time.time()
        st_all = pd.concat([st_t1, st_ot], ignore_index=True)
        st_all_gdf = gpd.GeoDataFrame(st_all, crs=pipe.crs)
        all_res = vectorized_nearest_with_attrs(sd, st_all_gdf, keep_cols=['operator_tier','diameter_in'])
        print(f'    {time.time()-t0:.1f}s')

        # Nearest tier_1 only (distance only)
        print(f'  Nearest tier_1 pipeline ({len(st_t1):,} in envelope) ...', flush=True)
        t0 = time.time()
        t1_res = vectorized_nearest_with_attrs(sd, st_t1, keep_cols=[])
        d_tier1 = t1_res['_dist'].values if t1_res is not None else np.full(len(sd), np.nan)
        print(f'    {time.time()-t0:.1f}s')

        # Nearest non-tier_1 only
        print(f'  Nearest other pipeline ({len(st_ot):,} in envelope) ...', flush=True)
        t0 = time.time()
        ot_res = vectorized_nearest_with_attrs(sd, st_ot, keep_cols=[])
        d_other = ot_res['_dist'].values if ot_res is not None else np.full(len(sd), np.nan)
        print(f'    {time.time()-t0:.1f}s')

        out_rows.append(pd.DataFrame({
            'candidate_id': sd.candidate_id.values,
            'nearest_pipeline_distance_m':      all_res['_dist'].values,
            'nearest_pipeline_operator_tier':   all_res['operator_tier'].values,
            'nearest_pipeline_est_diameter_in': all_res['diameter_in'].values,
            'nearest_tier1_pipeline_distance_m':d_tier1,
            'nearest_other_pipeline_distance_m':d_other,
        }))

    print('\nConcatenating per-state results ...')
    out = pd.concat(out_rows, ignore_index=True)
    assert len(out) == len(cands), f'lost candidates: expected {len(cands)}, got {len(out)}'
    out['pipeline_diameter_estimated'] = True

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f'\nSaved: {OUT_PATH} ({size_mb:.2f} MB, {len(out):,} rows)')

    # ---- Distribution summary ----
    print('\n=== Distance medians (m) ===')
    for c in ['nearest_pipeline_distance_m','nearest_tier1_pipeline_distance_m','nearest_other_pipeline_distance_m']:
        s = out[c].dropna()
        n_miss = out[c].isna().sum()
        print(f'  {c:<38} median={s.median():>10.0f}  p90={s.quantile(0.9):>10.0f}  beyond_80km={n_miss:,}')

    print('\n=== Nearest operator_tier distribution ===')
    print(out.nearest_pipeline_operator_tier.value_counts(dropna=False).to_string())

    print('\n=== Nearest diameter distribution ===')
    s = out.nearest_pipeline_est_diameter_in.dropna()
    print(f'  median={s.median()}, mode={s.mode().iloc[0]}, p90={s.quantile(0.9)}')

    print('\n=== Checks ===')
    checks = {
        'Has all candidates'                : len(out) == len(cands),
        'Unique candidate_ids'              : out.candidate_id.is_unique,
        'pipeline_diameter_estimated True'  : out.pipeline_diameter_estimated.all(),
        'nearest_pipeline_distance non-neg' : (out.nearest_pipeline_distance_m.fillna(0) >= 0).all(),
        'nearest_tier1 dist non-neg'        : (out.nearest_tier1_pipeline_distance_m.fillna(0) >= 0).all(),
        'nearest_other dist non-neg'        : (out.nearest_other_pipeline_distance_m.fillna(0) >= 0).all(),
        'tier in {tier_1, other, nan}'      : set(out.nearest_pipeline_operator_tier.dropna().unique()) <= {'tier_1','other'},
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
