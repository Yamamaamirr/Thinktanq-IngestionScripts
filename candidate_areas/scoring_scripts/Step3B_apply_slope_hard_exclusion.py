"""
Step 3B — Apply PDF Rule 13 slope hard exclusion (slope_max > 15%) and re-export.

PDF Rule 13 (Stage 7): if slope_max_pct > 15: hard_exclusion_flag = True

This step corrects an earlier deviation where Step 2A used slope_MEAN > 15
instead of slope_MAX > 15 for the kill gate. Cross-validated against:
  - Geography: violators concentrate in known mountain regions (Blue Ridge,
    Sierra Nevada, Great Basin, Mogollon Rim, TX Hill Country)
  - Land cover: 79% forest/shrub (consistent with hilly terrain)
  - Slope distribution: mean slope_max 19.6%, p99 34.7%
  - slope_tier: all 9,730 violators were already in 'penalized' tier

Reads:
  candidate_areas/outputs/candidate_areas_enriched.parquet  (source, unchanged)

Writes (overwrites):
  candidate_areas/outputs/candidates_final.parquet
  candidate_areas/outputs/candidates_final.csv
  candidate_areas/outputs/candidates_final.fgb

Excluded rows are marked candidate_status='excluded' and candidate_status_reason
='slope_max_gt_15_pct_pdf_rule_13' in the source enriched parquet as well, so
the audit trail is preserved.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')
CSV_OUT       = Path('candidate_areas/outputs/candidates_final.csv')
PARQUET_OUT   = Path('candidate_areas/outputs/candidates_final.parquet')
FGB_OUT       = Path('candidate_areas/outputs/candidates_final.fgb')

SLOPE_HARD_THRESHOLD = 15.0  # PDF Rule 13


def _json_encode(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return json.dumps(x.tolist())
    if isinstance(x, (list, tuple, dict)):
        return json.dumps(list(x) if isinstance(x, tuple) else x)
    return x


def main():
    print('=' * 72)
    print(' Step 3B: applying PDF Rule 13 slope_max > 15% hard exclusion')
    print('=' * 72)

    print(f'\nLoading enriched parquet: {ENRICHED_PATH} ...')
    gdf = gpd.read_parquet(ENRICHED_PATH)
    n_total = len(gdf)
    print(f'  {n_total:,} rows x {len(gdf.columns)} cols, CRS={gdf.crs.to_epsg()}')

    # Identify violators - PDF Rule 13 is a HARD exclusion regardless of current status,
    # so we exclude all slope_max > 15 candidates including those currently manual_review.
    violators_mask = (gdf['slope_max_pct'] > SLOPE_HARD_THRESHOLD) & (gdf['candidate_status'] != 'excluded')
    n_excluded = int(violators_mask.sum())
    print(f'\n  slope_max > {SLOPE_HARD_THRESHOLD}% (any non-excluded status): {n_excluded:,} candidates flagged')

    # Preserve original status reason in a separate column for audit, then re-mark
    prior_reason_col = gdf.loc[violators_mask, 'candidate_status_reason']
    prior_status_col = gdf.loc[violators_mask, 'candidate_status']
    gdf.loc[violators_mask, 'candidate_status'] = 'excluded'
    new_reason = prior_reason_col.fillna('').astype(str).apply(
        lambda x: 'slope_max_gt_15_pct_pdf_rule_13' + (f' (prior_reason: {x})' if x else '')
    )
    gdf.loc[violators_mask, 'candidate_status_reason'] = new_reason

    # Save updated source
    print(f'  Updating source enriched parquet (audit trail) ...')
    gdf.to_parquet(ENRICHED_PATH, index=False)

    # Filter to pass candidates only for final exports
    pass_gdf = gdf[gdf['candidate_status'] != 'excluded'].copy()
    n_pass = len(pass_gdf)
    print(f'  Pass candidates after slope exclusion: {n_pass:,} ({100*n_pass/n_total:.1f}% of original)')

    # Sort by composite_score desc
    pass_gdf = pass_gdf.sort_values('composite_score', ascending=False).reset_index(drop=True)

    # ---- GeoParquet ----
    print(f'\nWriting GeoParquet: {PARQUET_OUT} ...')
    pass_gdf.to_parquet(PARQUET_OUT, index=False)
    print(f'  Saved ({PARQUET_OUT.stat().st_size/1e6:.1f} MB)')

    # ---- CSV ----
    print(f'\nWriting CSV: {CSV_OUT} ...')
    csv_df = pass_gdf.copy()
    csv_df['geometry_wkt'] = csv_df.geometry.to_wkt()
    csv_df = csv_df.drop(columns=['geometry'])
    for col in csv_df.columns:
        sample = csv_df[col].dropna().head(1)
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict, tuple, np.ndarray)):
            csv_df[col] = csv_df[col].apply(_json_encode)
    csv_df.to_csv(CSV_OUT, index=False)
    print(f'  Saved ({CSV_OUT.stat().st_size/1e6:.1f} MB)')

    # ---- FlatGeobuf ----
    print(f'\nWriting FlatGeobuf: {FGB_OUT} ...')
    fgb_df = pass_gdf.copy()
    for col in fgb_df.columns:
        if col == 'geometry':
            continue
        sample = fgb_df[col].dropna().head(1)
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict, tuple, np.ndarray)):
            fgb_df[col] = fgb_df[col].apply(_json_encode)
    if FGB_OUT.exists():
        FGB_OUT.unlink()
    fgb_df.to_file(FGB_OUT, driver='FlatGeobuf')
    print(f'  Saved ({FGB_OUT.stat().st_size/1e6:.1f} MB)')

    # ---- Verification ----
    print(f'\n{"=" * 72}\n VERIFICATION\n{"=" * 72}')

    pq_check = gpd.read_parquet(PARQUET_OUT)
    csv_check = pd.read_csv(CSV_OUT, low_memory=False)
    fgb_check = gpd.read_file(FGB_OUT)

    checks = {
        'parquet row count matches pass set': len(pq_check) == n_pass,
        'csv row count matches pass set'    : len(csv_check) == n_pass,
        'fgb row count matches pass set'    : len(fgb_check) == n_pass,
        'no slope_max > 15 in parquet'      : (pq_check.slope_max_pct > 15).sum() == 0,
        'no slope_max > 15 in csv'          : (csv_check.slope_max_pct > 15).sum() == 0,
        'no slope_max > 15 in fgb'          : (fgb_check.slope_max_pct > 15).sum() == 0,
        'all candidate_status == pass or manual_review':
            set(pq_check.candidate_status.unique()) <= {'pass', 'manual_review'},
        'parquet sorted by composite desc'  :
            (pq_check.composite_score.diff().dropna() <= 0.0001).all(),
        'csv sorted by composite desc'      :
            (csv_check.composite_score.diff().dropna() <= 0.0001).all(),
        'parquet CRS = 5070'                : pq_check.crs.to_epsg() == 5070,
        'fgb CRS = 5070'                    : fgb_check.crs.to_epsg() == 5070,
        'no duplicate candidate_ids'        : pq_check.candidate_id.is_unique,
    }
    fails = 0
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            fails += 1

    # Per-state count breakdown
    print(f'\n  Per-state pass counts (after exclusion):')
    for st in ['AZ','CA','NV','TX','VA']:
        n_st = (pq_check.state == st).sum()
        n_st_orig = ((gdf.state == st)).sum()
        n_excl = ((gdf.state == st) & (gdf.candidate_status == 'excluded')).sum()
        print(f'    {st}: {n_st:>6,}  (excluded {n_excl:>5,} of {n_st_orig:>6,} original)')

    # Action band breakdown
    print(f'\n  Action band counts (after exclusion):')
    print(pq_check.recommended_action.value_counts().to_string().replace('\n', '\n    '))

    print(f'\n  Composite score distribution (after exclusion):')
    for p in [50, 75, 90, 95, 99, 100]:
        print(f'    p{p:>3}: composite={np.percentile(pq_check.composite_score, p):.2f}')
    print(f'    min={pq_check.composite_score.min():.2f}, max={pq_check.composite_score.max():.2f}, '
          f'mean={pq_check.composite_score.mean():.2f}')

    print(f'\n{"=" * 72}')
    if fails == 0:
        print(' ALL CHECKS PASSED')
    else:
        print(f' {fails} CHECKS FAILED')
    print(f'{"=" * 72}')
    print(f'\nFinal deliverables:')
    print(f'  {PARQUET_OUT}  ({PARQUET_OUT.stat().st_size/1e6:.1f} MB, {n_pass:,} rows)')
    print(f'  {CSV_OUT}      ({CSV_OUT.stat().st_size/1e6:.1f} MB, {n_pass:,} rows)')
    print(f'  {FGB_OUT}      ({FGB_OUT.stat().st_size/1e6:.1f} MB, {n_pass:,} rows)')


if __name__ == '__main__':
    main()
