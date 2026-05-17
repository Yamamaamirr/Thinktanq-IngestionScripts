"""
Final end-to-end verification of all Phase 3 deliverables.

Compares: source enriched parquet → final parquet → final CSV
Checks: row count, column completeness, ID set parity, sort order, top row,
        numerical integrity on critical columns, geometry roundtrip,
        list/dict columns serialization, per-state counts, composite math,
        nulls in critical fields.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt

SRC_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')
GPQ_PATH = Path('candidate_areas/outputs/candidates_final.parquet')
CSV_PATH = Path('candidate_areas/outputs/candidates_final.csv')


def main():
    print('=' * 72)
    print(' END-TO-END VERIFICATION: source enriched -> final parquet -> final CSV')
    print('=' * 72)

    print('\nLoading sources ...')
    src = gpd.read_parquet(SRC_PATH)
    gpq = gpd.read_parquet(GPQ_PATH)
    print(f'  source enriched: {len(src):,} rows x {len(src.columns)} cols')
    print(f'  final parquet:   {len(gpq):,} rows x {len(gpq.columns)} cols')
    print('  loading full CSV (may take ~30s) ...')
    csv = pd.read_csv(CSV_PATH)
    print(f'  final CSV:       {len(csv):,} rows x {len(csv.columns)} cols')

    # 1. Row count
    print('\n=== 1. Row count integrity ===')
    print(f'  src vs gpq:  {len(src) == len(gpq)}')
    print(f'  src vs csv:  {len(src) == len(csv)}')

    # 2. Column completeness
    print('\n=== 2. Column completeness ===')
    src_cols = set(src.columns)
    gpq_cols = set(gpq.columns)
    csv_cols = set(csv.columns)
    print(f'  src=gpq column set: {src_cols == gpq_cols}')
    expected_csv = (src_cols - {'geometry'}) | {'geometry_wkt'}
    print(f'  csv matches expected (geometry -> geometry_wkt): {csv_cols == expected_csv}')
    extra = csv_cols - expected_csv
    missing = expected_csv - csv_cols
    if extra:   print(f'    extra in CSV: {extra}')
    if missing: print(f'    missing in CSV: {missing}')

    # 3. ID set parity
    print('\n=== 3. candidate_id set parity ===')
    src_ids = set(src.candidate_id)
    gpq_ids = set(gpq.candidate_id)
    csv_ids = set(csv.candidate_id)
    print(f'  src vs gpq: {gpq_ids == src_ids}')
    print(f'  src vs csv: {csv_ids == src_ids}')
    print(f'  all unique in csv: {csv.candidate_id.is_unique}')

    # 4. Sort order
    print('\n=== 4. Sort order (composite_score descending) ===')
    print(f'  gpq sorted desc: {(gpq.composite_score.diff().dropna() <= 0.0001).all()}')
    print(f'  csv sorted desc: {(csv.composite_score.diff().dropna() <= 0.0001).all()}')

    # 5. Top row identity
    print('\n=== 5. Top row identity ===')
    src_sorted = src.sort_values('composite_score', ascending=False).reset_index(drop=True)
    src_top = src_sorted.iloc[0]
    gpq_top = gpq.iloc[0]
    csv_top = csv.iloc[0]
    print(f'  src top cand_id: {src_top.candidate_id}')
    print(f'  gpq top cand_id matches: {src_top.candidate_id == gpq_top.candidate_id}')
    print(f'  csv top cand_id matches: {src_top.candidate_id == csv_top.candidate_id}')

    # 6. Critical numerical value integrity
    print('\n=== 6. Critical column value integrity (gpq vs src) ===')
    critical = [
        'composite_score','utility_score','buildability_score',
        'supporting_infra_score','dev_risk_score','area_acres',
        'building_footprint_pct','slope_mean_pct','slope_max_pct',
        'seismic_hazard_pga','buildable_area_ratio',
    ]
    src_ind = src_sorted.set_index('candidate_id')
    gpq_ind = gpq.set_index('candidate_id').loc[src_ind.index]
    for c in critical:
        if c not in src_ind.columns or c not in gpq_ind.columns:
            continue
        a = src_ind[c].fillna(-9999.0).astype(float).values
        b = gpq_ind[c].fillna(-9999.0).astype(float).values
        print(f'  {c:<30} max abs diff: {np.abs(a-b).max():.6e}')

    # 7. CSV value integrity
    print('\n=== 7. Critical column value integrity (csv vs src) ===')
    csv_ind = csv.set_index('candidate_id').loc[src_ind.index]
    for c in critical:
        if c not in src_ind.columns or c not in csv_ind.columns:
            continue
        a = src_ind[c].fillna(-9999.0).astype(float).values
        b = csv_ind[c].fillna(-9999.0).astype(float).values
        print(f'  {c:<30} max abs diff: {np.abs(a-b).max():.6e}')

    # 8. Geometry integrity
    print('\n=== 8. Geometry integrity ===')
    src_g = src.sort_values('candidate_id').reset_index(drop=True)
    gpq_g = gpq.sort_values('candidate_id').reset_index(drop=True)
    print(f'  src vs gpq geom equals (1mm tol): {src_g.geometry.geom_equals_exact(gpq_g.geometry, tolerance=0.001).all()}')
    print(f'  src CRS: {src.crs.to_epsg()}  gpq CRS: {gpq.crs.to_epsg()}')
    print(f'  csv geometry_wkt parseable (sample 6 rows):')
    for idx in [0, 1, 100, 1000, 50000, 95268]:
        try:
            g = wkt.loads(csv.iloc[idx]['geometry_wkt'])
            print(f'    row {idx}: ok, valid={g.is_valid}, type={g.geom_type}')
        except Exception as e:
            print(f'    row {idx}: ERROR {e}')

    # 9. List column integrity
    print('\n=== 9. List-column integrity (top_reason_codes) ===')
    src_codes_top = list(src_sorted.iloc[0]['top_reason_codes'])
    gpq_codes_top = list(gpq.iloc[0]['top_reason_codes'])
    print(f'  src top row codes: {src_codes_top}')
    print(f'  gpq top row codes: {gpq_codes_top}')
    print(f'  gpq matches: {src_codes_top == gpq_codes_top}')
    csv_codes_str = csv.iloc[0]['top_reason_codes']
    csv_codes_top = json.loads(csv_codes_str)
    print(f'  csv top row codes: {csv_codes_top}')
    print(f'  csv matches: {csv_codes_top == src_codes_top}')

    # Sample 100 rows: verify list columns end-to-end
    n_mismatch = 0
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(src_sorted), size=100, replace=False)
    for i in sample_idx:
        src_l = list(src_sorted.iloc[i]['top_reason_codes'])
        cid = src_sorted.iloc[i]['candidate_id']
        gpq_l = list(gpq.set_index('candidate_id').loc[cid]['top_reason_codes'])
        csv_l = json.loads(csv.set_index('candidate_id').loc[cid]['top_reason_codes'])
        if src_l != gpq_l or src_l != csv_l:
            n_mismatch += 1
    print(f'  Random 100-row reason_codes mismatches: {n_mismatch}')

    # 10. Per-state counts
    print('\n=== 10. Per-state count integrity ===')
    print(f'  {"state":<5} {"src":>7} {"gpq":>7} {"csv":>7}  match')
    for st in ['AZ','CA','NV','TX','VA']:
        n_src = int((src.state == st).sum())
        n_gpq = int((gpq.state == st).sum())
        n_csv = int((csv.state == st).sum())
        ok = (n_src == n_gpq == n_csv)
        print(f'  {st:<5} {n_src:>7,} {n_gpq:>7,} {n_csv:>7,}  {"OK" if ok else "MISMATCH"}')

    # 11. Composite formula re-derivation
    print('\n=== 11. Composite formula re-derivation ===')
    expected = (
        0.40 * gpq.utility_score
      + 0.20 * gpq.buildability_score
      + 0.15 * gpq.supporting_infra_score
      + 0.15 * gpq.dev_risk_score
    ) / 0.90
    diff = (expected - gpq.composite_score).abs()
    print(f'  max |expected - actual|: {diff.max():.8f}')
    print(f'  all within 0.001:        {(diff < 0.001).all()}')

    # 12. Nulls in critical fields
    print('\n=== 12. Null check on critical scoring columns ===')
    must_non_null = [
        'candidate_id','state','area_acres','composite_score','confidence',
        'recommended_action','actionability_status','top_reason_codes',
        'utility_score','buildability_score','supporting_infra_score','dev_risk_score',
        'candidate_status','run_id','run_date','scoring_model_version',
    ]
    fails = 0
    for c in must_non_null:
        n_null = gpq[c].isna().sum()
        if n_null == 0:
            print(f'  {c:<30} OK (0 nulls)')
        else:
            print(f'  {c:<30} {n_null} NULLS')
            fails += 1
    print(f'\n  Critical null failures: {fails}')

    # 13. Recommended_action / actionability_status set check
    print('\n=== 13. Action field value-set check ===')
    rec_valid = {'Ignore','Monitor','Manual Review','Parcel Pull','Utility Desk Check',
                 'Ownership Review','Reuse Diligence','Shortlist'}
    act_valid = {'do_not_pitch','internal_diligence_only','apn_owner_pull_required',
                 'broker_verify_required','nda_teaser_possible','buyer_ready_with_caveats'}
    rec_set = set(gpq.recommended_action.unique())
    act_set = set(gpq.actionability_status.unique())
    print(f'  recommended_action values: {rec_set} subset OK: {rec_set <= rec_valid}')
    print(f'  actionability values:      {act_set} subset OK: {act_set <= act_valid}')

    # 14. CSV roundtrip integrity for the SECOND row (not just top)
    print('\n=== 14. Spot-check row #50000 across all three ===')
    src_50k = src_sorted.iloc[50000]
    cid = src_50k.candidate_id
    gpq_50k = gpq.set_index('candidate_id').loc[cid]
    csv_50k = csv.set_index('candidate_id').loc[cid]
    print(f'  cand_id: {cid[:12]}..')
    print(f'  src composite: {src_50k.composite_score:.4f}')
    print(f'  gpq composite: {gpq_50k.composite_score:.4f}')
    print(f'  csv composite: {csv_50k.composite_score:.4f}')
    print(f'  src state: {src_50k.state}')
    print(f'  csv state: {csv_50k.state}')
    print(f'  src recommended_action: {src_50k.recommended_action}')
    print(f'  csv recommended_action: {csv_50k.recommended_action}')

    print('\nVERIFICATION COMPLETE.')


if __name__ == '__main__':
    main()
