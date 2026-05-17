"""
Step 2G — Add Phase 2 placeholder columns + dataset versioning per Owen Rec #30.

Phase 2 placeholder columns (all null in this run, per scoring_design Sec 6):
  parcel_count, owner_count, largest_owner_acres, largest_owner_pct_of_candidate,
  assessed_value_total, assessed_value_per_acre, last_sale_date, last_sale_price,
  land_use_code, zoning_code, road_frontage_flag, legal_access_flag,
  site_control_score, economic_proxy_score,
  serving_utility, utility_territory_known, nearest_load_serving_node,
  utility_service_feasibility_score, utility_review_required,
  communications_route_distance, communications_provider_count, communications_access_score,
  water_capacity_known, water_capacity_review_required,
  jurisdiction_review_required, local_policy_notes,
  manual_imagery_review_status, manual_imagery_review_notes

Dataset versioning (Rec #30):
  run_id, run_date, scoring_model_version, exclusion_model_version,
  cdl_year, padus_version, fema_nfhl_date, nwi_date,
  transmission_dataset_version, queue_dataset_date, dem_dataset_version

Reads / Writes:
  candidate_areas/outputs/candidate_areas_enriched.parquet

Run:
  python candidate_areas/scoring_scripts/Step2G_phase2_placeholders_versioning.py
"""

from pathlib import Path
import uuid
from datetime import date
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')

PHASE2_PLACEHOLDER_COLS = [
    # Parcel/owner block
    'parcel_count', 'owner_count', 'largest_owner_acres', 'largest_owner_pct_of_candidate',
    'assessed_value_total', 'assessed_value_per_acre',
    'last_sale_date', 'last_sale_price',
    'land_use_code', 'zoning_code', 'road_frontage_flag', 'legal_access_flag',
    # Scoring placeholders
    'site_control_score', 'economic_proxy_score',
    # Utility feasibility
    'serving_utility', 'utility_territory_known', 'nearest_load_serving_node',
    'utility_service_feasibility_score', 'utility_review_required',
    # Communications
    'communications_route_distance', 'communications_provider_count', 'communications_access_score',
    # Water capacity
    'water_capacity_known', 'water_capacity_review_required',
    # Jurisdiction
    'jurisdiction_review_required', 'local_policy_notes',
    # Manual QA
    'manual_imagery_review_status', 'manual_imagery_review_notes',
]

VERSION_FIELDS = {
    'run_id':                       str(uuid.uuid4()),
    'run_date':                     date.today().isoformat(),
    'scoring_model_version':        '1.0.0-phase1',
    'exclusion_model_version':      '1.0',
    'cdl_year':                     2025,
    'padus_version':                '4.1',
    'fema_nfhl_date':               '2025',          # snapshot date for exclusion_fema
    'nwi_date':                     '2025',          # snapshot date for NWI exclusion files
    'transmission_dataset_version': 'HIFLD-2024-09',
    'queue_dataset_date':           '2026-05',
    'dem_dataset_version':          'USGS-3DEP-1arcsec',
}


def main():
    print(f'Loading enriched candidates: {ENRICHED_PATH} ...')
    df = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(df):,} candidates, {len(df.columns)} columns before')

    # ---- Phase 2 placeholders ----
    print(f'\nAdding {len(PHASE2_PLACEHOLDER_COLS)} Phase 2 placeholder columns (null) ...')
    added = 0
    for col in PHASE2_PLACEHOLDER_COLS:
        if col not in df.columns:
            df[col] = None
            added += 1
    # parcel_owner_module_status — per Owen May 11 it's an explicit string, not null
    if 'parcel_owner_module_status' not in df.columns:
        df['parcel_owner_module_status'] = 'not_built'
        added += 1
    print(f'  Added {added} columns')

    # ---- Dataset versioning ----
    print(f'\nAdding {len(VERSION_FIELDS)} dataset versioning fields per Owen Rec #30 ...')
    for col, val in VERSION_FIELDS.items():
        df[col] = val

    df.to_parquet(ENRICHED_PATH, index=False)
    size_mb = ENRICHED_PATH.stat().st_size / 1e6
    print(f'\nUpdated: {ENRICHED_PATH} ({size_mb:.1f} MB, {len(df.columns)} columns)')

    print('\n=== Versioning sample ===')
    for col in VERSION_FIELDS:
        print(f'  {col:<30} = {df[col].iloc[0]}')

    print('\n=== Phase 2 placeholders: null verification ===')
    for col in PHASE2_PLACEHOLDER_COLS:
        n_non_null = df[col].notna().sum()
        if n_non_null > 0:
            print(f'  {col}: {n_non_null:,} non-null (expected 0)')
    print('  All Phase 2 placeholder columns are null as expected.')

    print('\n=== Checks ===')
    checks = {
        'Has rows'                                 : len(df) > 0,
        'All placeholder cols added'               : all(c in df.columns for c in PHASE2_PLACEHOLDER_COLS),
        'parcel_owner_module_status is not_built'  : (df['parcel_owner_module_status'] == 'not_built').all(),
        'run_id consistent'                        : df.run_id.nunique() == 1,
        'run_date consistent'                      : df.run_date.nunique() == 1,
        'scoring_model_version present'            : df.scoring_model_version.notna().all(),
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
