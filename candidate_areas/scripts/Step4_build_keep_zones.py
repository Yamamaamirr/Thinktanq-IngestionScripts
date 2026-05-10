"""
Step 4 — Union substation and transmission buffers into a single keep-zone layer.

Reads:  ingestion_scripts/gridstatus_iso_queues/substation_buffers_25km.parquet
        ingestion_scripts/hifld_transmission_lines/transmission_buffers_345kv_plus.parquet
Writes: candidate_areas/outputs/keep_zones.parquet

Run from project root:  python candidate_areas/Step4_build_keep_zones.py

What this produces:
  A single dissolved MultiPolygon covering every area worth looking at from a
  grid perspective. Anything outside this geometry is ignored in all downstream
  steps — no CDL processing, no scoring, nothing.

  The result is one row with a MultiPolygon containing disconnected geographic
  islands across CA, TX, AZ, NV, VA (e.g. the NV cluster is spatially separate
  from the TX cluster).

Why dissolve:
  The raw inputs are 1,419 overlapping individual shapes (711 substation circles +
  708 primary-tier transmission corridors). Without dissolving, a CDL patch inside
  two overlapping buffers would get matched twice. Dissolving collapses all overlaps
  into one clean geometry where every point is counted once.

Why only 345kV+ transmission here (Owen Rec #6):
  The 230kV corridor file (transmission_buffers_230kv.parquet) is NOT included in
  the keep-zone union. 230kV is a secondary scoring signal — it tells us a site has
  some transmission proximity, but it is not strong enough to anchor the search zone
  on its own. Including it in the keep-zone would expand the search area without a
  meaningful quality gate.

  230kV corridors are used in Step9 scoring (infrastructure_proximity_score) after
  a site is already inside the primary keep-zone. They do not drive inclusion.

CRS: EPSG:5070 (NAD83 / Conus Albers) throughout — same as input files.
     All downstream overlay operations (intersection, difference) run in 5070.
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path

SUB_SRC = Path('ingestion_scripts/gridstatus_iso_queues/substation_buffers_25km.parquet')
TXL_SRC = Path('ingestion_scripts/hifld_transmission_lines/transmission_buffers_345kv_plus.parquet')
OUT     = Path('candidate_areas/outputs/keep_zones.parquet')

# ── Load ──────────────────────────────────────────────────────────────────────

print('Loading substation buffers ...')
subs = gpd.read_parquet(SUB_SRC)
print(f'  {len(subs):,} rows  CRS: EPSG:{subs.crs.to_epsg()}')

print('Loading transmission buffers (345kV+ primary tier only — 230kV excluded from keep-zone) ...')
txln = gpd.read_parquet(TXL_SRC)
print(f'  {len(txln):,} rows  CRS: EPSG:{txln.crs.to_epsg()}')

assert subs.crs == txln.crs, 'CRS mismatch between input files'

# ── Union and dissolve ────────────────────────────────────────────────────────

print('Combining and dissolving ...')
combined = pd.concat([
    subs[['geometry']],
    txln[['geometry']],
], ignore_index=True)

dissolved = combined.dissolve()
dissolved = dissolved.reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────

dissolved.to_parquet(OUT, index=False)

# ── Verify ────────────────────────────────────────────────────────────────────

check = gpd.read_parquet(OUT)
geom  = check.geometry.iloc[0]

total_area_km2 = geom.area / 1e6
sub_count = len(geom.geoms) if geom.geom_type == 'MultiPolygon' else 1

print(f'\nSaved: {OUT}')
print(f'  Rows          : {len(check)} (one dissolved multipolygon)')
print(f'  CRS           : EPSG:{check.crs.to_epsg()}')
print(f'  Geometry type : {geom.geom_type}')
print(f'  Sub-polygons  : {sub_count} disconnected geographic islands')
print(f'  Total area    : {total_area_km2:,.0f} km2')
print()

raw_area_km2 = (subs.geometry.area.sum() + txln.geometry.area.sum()) / 1e6
overlap_km2  = raw_area_km2 - total_area_km2
print(f'Area accounting:')
print(f'  Substation buffers raw  : {subs.geometry.area.sum()/1e6:>10,.0f} km2')
print(f'  Transmission buffers raw: {txln.geometry.area.sum()/1e6:>10,.0f} km2')
print(f'  Combined raw (with dups): {raw_area_km2:>10,.0f} km2')
print(f'  After dissolve (unique) : {total_area_km2:>10,.0f} km2')
print(f'  Overlap removed         : {overlap_km2:>10,.0f} km2 ({100*overlap_km2/raw_area_km2:.0f}%)')
print()

print('Checks:')
checks = {
    'Single row output'        : len(check) == 1,
    'CRS is EPSG:5070'         : check.crs.to_epsg() == 5070,
    'Geometry is MultiPolygon' : geom.geom_type == 'MultiPolygon',
    'Area > 500,000 km2'       : total_area_km2 > 500_000,
    'No null geometry'         : check.geometry.isna().sum() == 0,
    'Geometry is valid'        : check.geometry.is_valid.all(),
}
all_pass = True
for label, result in checks.items():
    status = 'PASS' if result else 'FAIL'
    if not result:
        all_pass = False
    print(f'  [{status}] {label}')

print('\nAll checks passed.' if all_pass else '\nWARNING: one or more checks failed.')
