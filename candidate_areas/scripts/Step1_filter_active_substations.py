"""
Step 1 — Filter ISO queue anchor stats to active substations.

Reads:  ingestion_scripts/gridstatus_iso_queues/anchor_queue_stats.parquet
Writes: ingestion_scripts/gridstatus_iso_queues/anchor_queue_stats_active.parquet
        ingestion_scripts/gridstatus_iso_queues/anchor_queue_stats_active_target.parquet

Run from project root:  python candidate_areas/Step1_filter_active_substations.py

KEEP-ZONE FILTER LOGIC (compound OR):
  1. total_active_capacity_mw >= 25         — tiered MW floor (Owen rec #2)
                                              <25 MW ignored for Phase 1
                                              25-50 weak, 50-100 moderate,
                                              100-200 strong, 200+ very strong
  2. anchor_state == 'NV'
     AND active_project_count > 0           — NV rescue (NV Energy hides substation names,
                                              so NV anchor coverage is thin — include any
                                              NV substation with confirmed queue activity)
  3. anchor_voltage_kv >= 345
     AND active_project_count > 0           — backbone transmission nodes with any queue
                                              interest, regardless of MW threshold

NOTE — keep-zone rebuild required:
  This filter change expands the qualifying substation set. Steps 2-4 (buffers +
  keep-zone union) and the exclusion subtraction chain must be re-run to reflect
  the new set. The current keep_zones_minus_radar_padus.parquet was built from
  the old 200 MW filter.
"""

import geopandas as gpd

TARGET_STATES = ['CA', 'TX', 'AZ', 'NV', 'VA']

SRC  = 'ingestion_scripts/gridstatus_iso_queues/anchor_queue_stats.parquet'
OUT_NATIONAL = 'ingestion_scripts/gridstatus_iso_queues/anchor_queue_stats_active.parquet'
OUT_TARGET   = 'ingestion_scripts/gridstatus_iso_queues/anchor_queue_stats_active_target.parquet'

print('Loading anchor_queue_stats.parquet ...')
df = gpd.read_parquet(SRC)
print(f'  Full dataset: {len(df)} rows')

mask = df['total_active_capacity_mw'] >= 25   # floor: <25 MW ignored
mask |= (df['anchor_state'] == 'NV') & (df['active_project_count'] > 0)
mask |= (df['anchor_voltage_kv'] >= 345) & (df['active_project_count'] > 0)

national = df[mask].copy()
target   = national[national['anchor_state'].isin(TARGET_STATES)].copy()

national.to_parquet(OUT_NATIONAL, index=False)
target.to_parquet(OUT_TARGET, index=False)

print(f'\nSaved: {OUT_NATIONAL}')
print(f'  Rows: {len(national)}  CRS: {national.crs.to_epsg()}')

print(f'\nSaved: {OUT_TARGET}')
print(f'  Rows: {len(target)}  CRS: {target.crs.to_epsg()}')

print('\nPer target state:')
for state in TARGET_STATES:
    s = target[target['anchor_state'] == state]
    hv   = (s['anchor_voltage_kv'] >= 345).sum()
    le18 = (s['activation_band'] == 'le_18').sum()
    ia   = (s['mw_commitment_letter'] == True).sum()
    print(f'  {state}: {len(s):>3} total | {hv:>3} 345kV+ | {le18:>3} le_18 | {ia:>3} signed IA')

print('\nTiered MW breakdown (target states):')
import pandas as pd
bins   = [0, 25, 50, 100, 200, float('inf')]
labels = ['<25 (excluded)', '25-50 weak', '50-100 moderate', '100-200 strong', '200+ very strong']
target['_mw_tier'] = pd.cut(target['total_active_capacity_mw'], bins=bins, labels=labels, right=True)
print(target['_mw_tier'].value_counts().sort_index().to_string())

print('\nVerification checks:')
full_ts  = df[df['anchor_state'].isin(TARGET_STATES)]
dropped  = full_ts[~full_ts['anchor_id'].isin(target['anchor_id'])]
checks = {
    '345kV+ with active projects still dropped': (
        (dropped['anchor_voltage_kv'] >= 345) & (dropped['active_project_count'] > 0)
    ).sum(),
    'NV with active projects still dropped': (
        (dropped['anchor_state'] == 'NV') & (dropped['active_project_count'] > 0)
    ).sum(),
    'Substations with >=25 MW still dropped': (
        dropped['total_active_capacity_mw'] >= 25
    ).sum(),
}
all_pass = True
for label, count in checks.items():
    status = 'PASS' if count == 0 else 'FAIL'
    if count != 0:
        all_pass = False
    print(f'  [{status}] {label}: {count}')

print('\nAll checks passed.' if all_pass else '\nWARNING: one or more checks failed.')
