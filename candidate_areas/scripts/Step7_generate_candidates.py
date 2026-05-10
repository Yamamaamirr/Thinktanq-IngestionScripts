"""
Step 7 - Generate candidate areas: combine 5-state CDL candidates, add county.

1. Load all 5 state CDL candidate parquets (AZ/CA/NV/TX/VA)
2. Combine into single GeoDataFrame
3. Re-filter >= 50 acres (safeguard after combine)
4. Spatial join to Census TIGER state boundaries to add state_code
5. Spatial join to Census TIGER county boundaries to add county name + FIPS
6. Save candidate_areas.parquet + candidate_areas.fgb

Inputs:
  candidate_areas/outputs/states/{state}/cdl_candidates_{state}.parquet  x5
  ingestion_scripts/census_tiger/state_boundaries.parquet
  ingestion_scripts/census_tiger/county_boundaries.parquet  (if exists)

Outputs:
  candidate_areas/outputs/candidate_areas.parquet
  candidate_areas/outputs/candidate_areas.fgb

Run: python candidate_areas/Step7_generate_candidates.py
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
from datetime import date

STATE_FILES = {
    'AZ': Path('candidate_areas/outputs/states/az/cdl_candidates_az.parquet'),
    'CA': Path('candidate_areas/outputs/states/ca/cdl_candidates_ca.parquet'),
    'NV': Path('candidate_areas/outputs/states/nv/cdl_candidates_nv.parquet'),
    'TX': Path('candidate_areas/outputs/states/tx/cdl_candidates_tx.parquet'),
    'VA': Path('candidate_areas/outputs/states/va/cdl_candidates_va.parquet'),
}

STATES_PATH   = Path('ingestion_scripts/census_tiger/state_boundaries.parquet')
OUT_PARQUET   = Path('candidate_areas/outputs/candidate_areas.parquet')
OUT_FGB       = Path('candidate_areas/outputs/candidate_areas.fgb')

MIN_ACRES = 50

# ---- Step 1: Load and combine all states ------------------------------------

print('Loading state CDL candidate files ...')
dfs = []
for state, path in STATE_FILES.items():
    if not path.exists():
        print(f'  WARNING: {path} not found — skipping {state}')
        continue
    gdf = gpd.read_parquet(path)
    gdf['state'] = state  # ensure state column is set
    print(f'  {state}: {len(gdf):,} candidates | {gdf.area_acres.sum():,.0f} acres')
    dfs.append(gdf)
    del gdf

print('\nCombining ...')
combined = gpd.GeoDataFrame(pd.concat(dfs, ignore_index=True), crs='EPSG:5070')
del dfs
print(f'  Combined: {len(combined):,} candidates | {combined.area_acres.sum():,.0f} acres')

# ---- Step 2: Re-filter >= 50 acres ------------------------------------------

before = len(combined)
combined = combined[combined['area_acres'] >= MIN_ACRES].copy()
print(f'  After re-filter: {before:,} -> {len(combined):,} candidates')

# ---- Step 3: Add county via spatial join ------------------------------------

print('\nLoading Census TIGER boundaries ...')
states_gdf = gpd.read_parquet(STATES_PATH).to_crs('EPSG:5070')

# Try to load county boundaries
county_path = Path('ingestion_scripts/census_tiger/county_boundaries.parquet')
if county_path.exists():
    print('  Loading county boundaries ...')
    counties_gdf = gpd.read_parquet(county_path).to_crs('EPSG:5070')
    has_counties = True
    print(f'  {len(counties_gdf):,} counties loaded')
else:
    print('  county_boundaries.parquet not found — will use state join only')
    has_counties = False

# Use centroid for spatial join (faster than polygon join)
print('\nBuilding centroid GeoDataFrame for spatial join ...')
centroids = gpd.GeoDataFrame(
    combined[['candidate_id']],
    geometry=combined.geometry.centroid,
    crs='EPSG:5070'
)

# State join
print('Joining to state boundaries ...')
state_join = gpd.sjoin(
    centroids, states_gdf[['state_code', 'geometry']],
    how='left', predicate='within'
)
state_join = state_join[['candidate_id', 'state_code']].rename(
    columns={'state_code': 'state_from_join'})
combined = combined.merge(state_join, on='candidate_id', how='left')
# Use state_from_join to fill any missing state values
if 'state_from_join' in combined.columns:
    combined['state'] = combined['state'].fillna(combined['state_from_join'])
    combined.drop(columns=['state_from_join'], inplace=True)

# County join
if has_county := has_counties:
    print('Joining to county boundaries ...')
    # Determine county name and FIPS columns
    county_cols = list(counties_gdf.columns)
    print(f'  County columns available: {county_cols}')

    name_col  = next((c for c in county_cols if 'name' in c.lower()), None)
    fips_col  = next((c for c in county_cols
                      if any(x in c.lower() for x in ['geoid','fips','county_fp','countyfp'])), None)

    join_cols = ['geometry']
    if name_col:  join_cols.append(name_col)
    if fips_col:  join_cols.append(fips_col)

    county_join = gpd.sjoin(
        centroids, counties_gdf[join_cols],
        how='left', predicate='within'
    )
    county_join = county_join[['candidate_id'] +
                               ([name_col] if name_col else []) +
                               ([fips_col] if fips_col else [])]
    if name_col:
        county_join = county_join.rename(columns={name_col: 'county_name'})
    if fips_col:
        county_join = county_join.rename(columns={fips_col: 'county_fips'})

    combined = combined.merge(county_join, on='candidate_id', how='left')
    print(f'  County join done')
else:
    combined['county_name'] = None
    combined['county_fips'] = None

del centroids

# ---- Step 4: Final clean-up -------------------------------------------------

# Ensure snapshot_date is set
if 'snapshot_date' not in combined.columns:
    combined['snapshot_date'] = date.today().isoformat()

# Owen Rec #5 — route complexity placeholders (null until manually reviewed)
combined['route_complexity_score'] = None  # 0-100 when filled; low=simple, high=barrier-heavy
combined['route_complexity_notes'] = None  # analyst notes on barriers (rivers, terrain, etc.)

# Owen Rec #11 — CDL non-ag classes (shrub/forest/barren) are less precise than ag classes.
# NLCD cross-check fields: null for Phase 1, populated in future NLCD pass.
combined['nlcd_class']                = None  # NLCD class code when cross-checked
combined['nlcd_label']                = None  # NLCD human-readable label
combined['landcover_confidence_score'] = None  # 0-1; lower for shrub/forest, higher for cropland

# Drop label_id (internal raster artifact, not needed downstream)
if 'label_id' in combined.columns:
    combined.drop(columns=['label_id'], inplace=True)

combined = combined[combined.geometry.is_valid & ~combined.geometry.is_empty].copy()

# ---- Summary ----------------------------------------------------------------

total_area = combined.area_acres.sum()
print(f'\n=== RESULTS ===')
print(f'  Total candidates: {len(combined):,}')
print(f'  Total area:       {total_area:,.0f} acres ({total_area/640:,.0f} sq miles)')
print(f'  CRS:              EPSG:{combined.crs.to_epsg()}')
print(f'  Columns:          {list(combined.columns)}')

print(f'\n  By state:')
for state, grp in combined.groupby('state'):
    print(f'    {state}: {len(grp):>6,} candidates | {grp.area_acres.sum():>14,.0f} acres')

print(f'\n  By CDL group:')
for g, grp in combined.groupby('cdl_group'):
    print(f'    {g:<15} {len(grp):>6,} candidates | {grp.area_acres.sum():>14,.0f} acres')

# ---- Save -------------------------------------------------------------------

print(f'\nSaving parquet ...')
combined.to_parquet(OUT_PARQUET, index=False)
print(f'  Saved: {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e6:.1f} MB)')

print(f'Writing FlatGeobuf ...')
combined.to_file(str(OUT_FGB), driver='FlatGeobuf')
print(f'  Saved: {OUT_FGB} ({OUT_FGB.stat().st_size/1e6:.1f} MB)')

# ---- Checks -----------------------------------------------------------------

print('\n=== CHECKS ===')
checks = {
    'Has candidates'     : len(combined) > 0,
    'CRS EPSG:5070'      : combined.crs.to_epsg() == 5070,
    'No null geometry'   : combined.geometry.isna().sum() == 0,
    'No empty geometry'  : (~combined.geometry.is_empty).all(),
    'All >= 50 acres'    : (combined['area_acres'] >= 50).all(),
    'Has state column'   : 'state' in combined.columns,
    'Has cdl_group'      : 'cdl_group' in combined.columns,
    'Has candidate_id'   : 'candidate_id' in combined.columns,
    'Has centroid_lat'   : 'centroid_lat' in combined.columns,
    'All 5 states'       : combined['state'].nunique() == 5,
    'Parquet saved'      : OUT_PARQUET.exists(),
    'FGB saved'          : OUT_FGB.exists(),
}
all_pass = True
for lbl, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
    if not ok:
        all_pass = False

print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
print('\nNext: Step 8 ingest_reuse_nodes.py, then Step 9 score_and_export.py')
