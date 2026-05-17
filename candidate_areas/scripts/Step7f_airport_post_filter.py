"""
Step 7f - Post-filter: drop candidates intersecting airport runway protection zones.

PDF Rule 10 hard-exclusion. Approximates runway protection zones by buffering
airport points (large 5km, medium 3km, small 1.5km).

Reads:
  candidate_areas/outputs/candidates_final.parquet  (current deliverable)
  ingestion_scripts/ourairports/airports_us_buffered.parquet

Writes:
  candidate_areas/outputs/candidate_areas_enriched.parquet  (marks excluded)
  candidate_areas/outputs/candidates_final.parquet  (filtered)
  candidate_areas/outputs/candidates_final.csv
  candidate_areas/outputs/candidates_final.fgb (best effort, write to *_NEW.fgb if locked)
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

CAND_FINAL    = Path('candidate_areas/outputs/candidates_final.parquet')
CSV_OUT       = Path('candidate_areas/outputs/candidates_final.csv')
FGB_OUT       = Path('candidate_areas/outputs/candidates_final.fgb')
ENRICHED_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')
AIRPORTS      = Path('ingestion_scripts/ourairports/airports_us_buffered.parquet')


def _je(x):
    if x is None: return None
    if isinstance(x, np.ndarray): return json.dumps(x.tolist())
    if isinstance(x, (list, tuple, dict)): return json.dumps(list(x) if isinstance(x, tuple) else x)
    return x


def main():
    print(f'Loading current final parquet: {CAND_FINAL}')
    cands = gpd.read_parquet(CAND_FINAL)
    n_before = len(cands)
    print(f'  {n_before:,} candidates')

    print(f'\nLoading airport buffers: {AIRPORTS}')
    airports = gpd.read_parquet(AIRPORTS)
    if airports.crs != cands.crs:
        airports = airports.to_crs(cands.crs)
    print(f'  {len(airports):,} airport buffers, total {airports.geometry.area.sum()/1e6:,.0f} km^2')

    print('\nSpatial join: candidates intersects airport buffers ...')
    t0 = time.time()
    inter = gpd.sjoin(
        cands[['candidate_id','geometry']],
        airports[['type','geometry']],
        how='inner',
        predicate='intersects',
    )
    excluded_ids = set(inter.candidate_id.unique())
    print(f'  {time.time()-t0:.1f}s, {len(excluded_ids):,} candidates intersect airport buffers')

    # Breakdown by airport type
    by_type = inter.groupby('type')['candidate_id'].nunique()
    print(f'  By airport type:')
    print(by_type.to_string())

    # Update enriched source: mark these as excluded
    print(f'\nUpdating {ENRICHED_PATH} to mark airport exclusions ...')
    enriched = gpd.read_parquet(ENRICHED_PATH)
    mask = enriched.candidate_id.isin(excluded_ids) & (enriched.candidate_status != 'excluded')
    print(f'  {mask.sum():,} candidates being marked as excluded (airport)')
    enriched.loc[mask, 'candidate_status'] = 'excluded'
    enriched.loc[mask, 'candidate_status_reason'] = 'airport_runway_protection_zone_pdf_rule_10'
    enriched.to_parquet(ENRICHED_PATH, index=False)

    # Filter the final file
    keep = cands[~cands.candidate_id.isin(excluded_ids)].copy()
    n_after = len(keep)
    print(f'\nDropping {n_before - n_after:,} candidates')
    print(f'Remaining: {n_after:,}')
    print(f'  By state (lost / kept):')
    for st in ['AZ','CA','NV','TX','VA']:
        n_b = (cands.state == st).sum()
        n_a = (keep.state == st).sum()
        print(f'    {st}: lost {n_b-n_a:>5,}  kept {n_a:>6,}  ({100*(n_b-n_a)/max(n_b,1):.1f}%)')

    # Re-sort
    keep = keep.sort_values('composite_score', ascending=False).reset_index(drop=True)

    # ---- Write parquet ----
    print(f'\nWriting GeoParquet: {CAND_FINAL}')
    keep.to_parquet(CAND_FINAL, index=False)
    print(f'  Saved ({CAND_FINAL.stat().st_size/1e6:.1f} MB)')

    # ---- Write CSV ----
    print(f'\nWriting CSV: {CSV_OUT}')
    csv_df = keep.copy()
    csv_df['geometry_wkt'] = csv_df.geometry.to_wkt()
    csv_df = csv_df.drop(columns=['geometry'])
    for col in csv_df.columns:
        sample = csv_df[col].dropna().head(1)
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict, tuple, np.ndarray)):
            csv_df[col] = csv_df[col].apply(_je)
    csv_df.to_csv(CSV_OUT, index=False)
    print(f'  Saved ({CSV_OUT.stat().st_size/1e6:.1f} MB)')

    # ---- Write FGB ----
    fgb_df = keep.copy()
    for col in fgb_df.columns:
        if col == 'geometry': continue
        sample = fgb_df[col].dropna().head(1)
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict, tuple, np.ndarray)):
            fgb_df[col] = fgb_df[col].apply(_je)

    fgb_target = FGB_OUT
    try:
        if fgb_target.exists():
            fgb_target.unlink()
        fgb_df.to_file(fgb_target, driver='FlatGeobuf')
        print(f'\nWrote FGB: {fgb_target} ({fgb_target.stat().st_size/1e6:.1f} MB)')
    except (PermissionError, OSError):
        fgb_target = FGB_OUT.with_name('candidates_final_NEW.fgb')
        if fgb_target.exists():
            fgb_target.unlink()
        fgb_df.to_file(fgb_target, driver='FlatGeobuf')
        print(f'\nFGB locked, wrote to {fgb_target} ({fgb_target.stat().st_size/1e6:.1f} MB)')
        print('  User: close any program holding candidates_final.fgb, then rename _NEW over it.')


if __name__ == '__main__':
    main()
