"""
StepR6_finalize_reuse_table -- Backfill derivable + run-metadata fields on
the standalone reuse_nodes_enriched.parquet so it carries the same schema
as the unified deliverable.

Fields populated:
  area_m2          from geometry.area (EPSG:5070 metric)
  centroid_lon     geometry centroid reprojected to EPSG:4326
  centroid_lat     same
  county_fips      spatial-join to TIGER county boundaries (GEOID)
  county_name      backfill any remaining nulls from the same join

  run_id                  new UUID per run
  run_date                today's date (ISO)
  snapshot_date           today's date (ISO)
  scoring_model_version   '1.0.0-phase1-reuse'
  exclusion_model_version copied from greenfield run-metadata
  padus_version           same
  fema_nfhl_date          same
  nwi_date                same
  transmission_dataset_version same
  queue_dataset_date      same
  dem_dataset_version     same

Fields intentionally NOT populated:
  cdl_year, pixel_count -- not applicable to reuse nodes

Reads / Writes:
  candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet  (in-place)
  reads metadata defaults from candidate_areas/outputs/candidates_final.parquet

Run: python candidate_areas/reuse_node_scripts/StepR6_finalize_reuse_table.py
"""
from pathlib import Path
from datetime import date
import uuid
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')
GREENFIELD_PATH = Path('candidate_areas/outputs/candidates_final.parquet')
COUNTY_PATH = Path('ingestion_scripts/census_tiger/county_boundaries.parquet')

SHARED_METADATA = [
    'exclusion_model_version',
    'padus_version',
    'fema_nfhl_date',
    'nwi_date',
    'transmission_dataset_version',
    'queue_dataset_date',
    'dem_dataset_version',
]


def main():
    print(f'Loading: {ENRICHED_PATH} ...')
    g = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(g):,} rows, {len(g.columns)} columns, CRS={g.crs.to_epsg()}')

    # ---- Derivable from geometry ----
    print('\nComputing area_m2 + centroid_lon/lat from geometry ...')
    g_4326_cents = g.geometry.centroid.to_crs(4326)
    g['area_m2']      = g.geometry.area
    g['centroid_lon'] = g_4326_cents.x.values
    g['centroid_lat'] = g_4326_cents.y.values

    # ---- county_fips via spatial join (also backfills any null county_name) ----
    if COUNTY_PATH.exists():
        print(f'Spatial join for county_fips: {COUNTY_PATH} ...')
        county = gpd.read_parquet(COUNTY_PATH)
        if county.crs.to_epsg() != 4326:
            county = county.to_crs(4326)
        cents_gdf = gpd.GeoDataFrame(
            {'_idx': range(len(g))},
            geometry=g_4326_cents.values, crs='EPSG:4326',
        )
        joined = gpd.sjoin(cents_gdf, county[['GEOID','NAME','geometry']],
                           how='left', predicate='within')
        joined = joined.drop_duplicates(subset='_idx', keep='first')
        joined = joined.set_index('_idx').reindex(range(len(g)))
        g['county_fips'] = joined['GEOID'].values
        if 'county_name' in g.columns:
            missing_name = g.county_name.isna()
            g.loc[missing_name, 'county_name'] = joined.loc[missing_name.values, 'NAME'].values
        else:
            g['county_name'] = joined['NAME'].values
    else:
        print(f'  WARN: {COUNTY_PATH} not found; county_fips not populated')

    # ---- Run metadata ----
    print('\nPopulating run metadata ...')
    g['run_id']                = str(uuid.uuid4())
    g['run_date']              = date.today().isoformat()
    g['snapshot_date']         = date.today().isoformat()
    g['scoring_model_version'] = '1.0.0-phase1-reuse'

    if GREENFIELD_PATH.exists():
        gf = pd.read_parquet(GREENFIELD_PATH, columns=SHARED_METADATA)
        for col in SHARED_METADATA:
            if col in gf.columns:
                val = gf[col].dropna().head(1)
                v = val.iloc[0] if len(val) else None
                g[col] = v
                print(f'  {col:<32} = {v}  (from greenfield)')
    else:
        print(f'  WARN: {GREENFIELD_PATH} not found; shared metadata not populated')

    # ---- Save back ----
    g.to_parquet(ENRICHED_PATH, index=False)
    print(f'\nUpdated: {ENRICHED_PATH} ({ENRICHED_PATH.stat().st_size/1e6:.1f} MB, {len(g.columns)} columns)')

    # ---- Summary ----
    print('\n=== Fill verification ===')
    fields = ['area_m2','centroid_lon','centroid_lat','county_fips','county_name',
              'run_id','run_date','snapshot_date','scoring_model_version',
              'exclusion_model_version','padus_version','fema_nfhl_date',
              'nwi_date','transmission_dataset_version','queue_dataset_date',
              'dem_dataset_version']
    for c in fields:
        if c in g.columns:
            n = int(g[c].notna().sum())
            sample = g[c].dropna().head(1).tolist()
            print(f'  {c:<32} nonnull={n}/{len(g)}  sample={sample}')


if __name__ == '__main__':
    main()
