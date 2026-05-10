"""
Step 2 — Draw 25 km proximity search zone buffers around qualifying substations.

Reads:  ingestion_scripts/gridstatus_iso_queues/anchor_queue_stats_active_target.parquet
Writes: ingestion_scripts/gridstatus_iso_queues/substation_buffers_25km.parquet

Run from project root:  python candidate_areas/Step2_buffer_substations_25km.py

These buffers identify land near meaningful infrastructure signals.
They are a screening layer — proximity to a qualifying substation is an early
signal, not a confirmation of service feasibility or interconnection availability.

Buffer radius: 25 km (25,000 m)
Projection:    EPSG:5070 (NAD83 / Conus Albers) — used for buffering and saved as-is.
               Albers equal-area ensures accurate metric distances across all target states.
               All downstream spatial operations (intersection, difference) run in 5070.

Output geometry: one Polygon per substation (individual, not dissolved).
Dissolving happens in Step4_build_keep_zones.py when substation and transmission
buffers are unioned into the final keep-zone.

Expected area per buffer: ~1,963 km2 (pi * 25^2). Slight deviation at margins is
normal Albers distortion and is negligible at this scale.

Route complexity placeholders (Owen Rec #5):
  route_complexity_score  FLOAT  null in Phase 1; filled manually for shortlisted candidates.
  route_complexity_notes  TEXT   null in Phase 1; analyst notes on routing barriers.

  Phase 1 trigger: any candidate with a high composite score but located more than
  5-10 km from the nearest infrastructure node should be flagged for manual review
  of routing barriers (dense development, rivers, highways, steep terrain, protected land).
  Automation of this field is medium-to-high complexity and is deferred post-Phase 1.
"""

import geopandas as gpd
import numpy as np

BUFFER_M = 25_000

SRC = 'ingestion_scripts/gridstatus_iso_queues/anchor_queue_stats_active_target.parquet'
OUT = 'ingestion_scripts/gridstatus_iso_queues/substation_buffers_25km.parquet'

KEEP_COLS = [
    'anchor_id', 'anchor_name', 'anchor_state', 'anchor_voltage_kv',
    'iso_region', 'total_active_capacity_mw', 'total_suspended_capacity_mw',
    'active_project_count', 'suspended_project_count',
    'activation_band', 'mw_commitment_letter', 'match_confidence',
    'total_energized_capacity_mw', 'recent_withdrawal_count_12mo',
    'queue_mw_tier', 'queue_status_best', 'queue_status_score', 'queue_signal_score',
    'line_poi_fraction',
    'route_complexity_score', 'route_complexity_notes',  # Phase 1: null; filled manually for shortlisted sites
    'geometry',
]

TARGET_STATES = ['CA', 'TX', 'AZ', 'NV', 'VA']

print(f'Loading {SRC} ...')
df = gpd.read_parquet(SRC)
print(f'  Input: {len(df)} rows, CRS: {df.crs.to_epsg()}')

print(f'Projecting to EPSG:5070 and buffering {BUFFER_M/1000:.0f} km ...')
df_proj = df.to_crs(epsg=5070)
df_proj['geometry'] = df_proj.geometry.buffer(BUFFER_M)

# Phase 1 placeholders — null; filled manually for shortlisted candidates (Owen Rec #5)
df_proj['route_complexity_score'] = np.nan
df_proj['route_complexity_notes'] = None

buffers = df_proj[KEEP_COLS]
buffers.to_parquet(OUT, index=False)

print(f'\nSaved: {OUT}')
print(f'  Rows:  {len(buffers)}')
print(f'  CRS:   {buffers.crs.to_epsg()}')
print(f'  Geoms: {buffers.geometry.geom_type.value_counts().to_dict()}')

print('\nPer state area summary:')
THEORETICAL_KM2 = 3.14159 * 25**2
for state in TARGET_STATES:
    s = buffers[buffers['anchor_state'] == state]
    raw_km2      = s.geometry.area.sum() / 1e6
    dissolved_km2 = s.dissolve().geometry.area.sum() / 1e6
    overlap_pct  = 100 * (1 - dissolved_km2 / raw_km2)
    print(f'  {state}: {len(s):>3} buffers | {raw_km2:>8,.0f} km2 raw | '
          f'{dissolved_km2:>8,.0f} km2 dissolved | {overlap_pct:.0f}% overlap')

print('\nSpot-check individual buffer areas (expect ~1,963 km2 each):')
spot = ['Bitter Creek Substation', 'Mohave Substation', 'Brunswick Heritage Substation']
all_pass = True
for name in spot:
    row = buffers[buffers['anchor_name'] == name]
    if len(row):
        area = row.iloc[0].geometry.area / 1e6
        deviation = abs(area - THEORETICAL_KM2) / THEORETICAL_KM2 * 100
        status = 'PASS' if deviation < 1.0 else 'FAIL'
        if status == 'FAIL':
            all_pass = False
        print(f'  [{status}] {name}: {area:.0f} km2 ({deviation:.2f}% from theoretical)')

print('\nAll checks passed.' if all_pass else '\nWARNING: one or more checks failed.')
