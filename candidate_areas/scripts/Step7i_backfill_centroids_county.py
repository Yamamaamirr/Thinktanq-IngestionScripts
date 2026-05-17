"""
Step 7i - Backfill centroid_lon/lat for fragments (set null by Step7d) and
fix any null county_name via spatial join with county boundaries.

Reads:
  candidate_areas/outputs/candidates_final.parquet
  ingestion_scripts/census_tiger/county_boundaries.parquet

Writes (overwrites in place):
  candidate_areas/outputs/candidates_final.parquet
  candidate_areas/outputs/candidates_final.csv
  candidate_areas/outputs/candidates_final.fgb
  candidate_areas/outputs/candidate_areas_enriched.parquet (also patches)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

CAND_FINAL    = Path('candidate_areas/outputs/candidates_final.parquet')
CSV_OUT       = Path('candidate_areas/outputs/candidates_final.csv')
FGB_OUT       = Path('candidate_areas/outputs/candidates_final.fgb')
ENRICHED_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')
COUNTY_PATH   = Path('ingestion_scripts/census_tiger/county_boundaries.parquet')


def _je(x):
    if x is None: return None
    if isinstance(x, np.ndarray): return json.dumps(x.tolist())
    if isinstance(x, (list, tuple, dict)): return json.dumps(list(x) if isinstance(x, tuple) else x)
    return x


def main():
    print(f'Loading {CAND_FINAL} ...')
    g = gpd.read_parquet(CAND_FINAL)
    print(f'  {len(g):,} candidates')

    # ---- A. Compute centroids (EPSG:4326 lon/lat) ----
    n_lon_null_before = g.centroid_lon.isna().sum()
    n_lat_null_before = g.centroid_lat.isna().sum()
    print(f'\nA. Centroid backfill — null before: lon={n_lon_null_before:,}, lat={n_lat_null_before:,}')
    centroids_5070 = g.geometry.centroid
    centroids_4326 = gpd.GeoSeries(centroids_5070, crs=g.crs).to_crs(4326)
    g['centroid_lon'] = centroids_4326.x.values
    g['centroid_lat'] = centroids_4326.y.values
    print(f'  lon/lat fully populated: lon_null={g.centroid_lon.isna().sum()}, lat_null={g.centroid_lat.isna().sum()}')

    # ---- B. Fix any null county_name ----
    n_county_null_before = g.county_name.isna().sum()
    print(f'\nB. county_name nulls before: {n_county_null_before}')
    if n_county_null_before > 0 and COUNTY_PATH.exists():
        county = gpd.read_parquet(COUNTY_PATH)
        if county.crs != g.crs:
            county = county.to_crs(g.crs)
        # Find county-name column - common Census attribute is NAME or NAMELSAD
        county_name_col = None
        for c in ['NAME','county_name','NAMELSAD','name']:
            if c in county.columns:
                county_name_col = c
                break
        if county_name_col:
            print(f'  Using county column: {county_name_col}')
            null_rows = g[g.county_name.isna()].copy()
            null_rows['_geom'] = null_rows.geometry.centroid
            null_pts = gpd.GeoDataFrame(
                null_rows[['candidate_id']],
                geometry=null_rows['_geom'],
                crs=g.crs,
            )
            joined = gpd.sjoin(null_pts, county[[county_name_col,'geometry']],
                               how='left', predicate='within')
            joined = joined.set_index('candidate_id')[county_name_col]
            mask = g.candidate_id.isin(joined.index)
            g.loc[mask, 'county_name'] = g.loc[mask, 'candidate_id'].map(joined).values
        print(f'  county_name nulls after: {g.county_name.isna().sum()}')

    # ---- Write outputs ----
    print(f'\nWriting parquet ...')
    g.to_parquet(CAND_FINAL, index=False)
    print(f'  saved ({CAND_FINAL.stat().st_size/1e6:.1f} MB)')

    print('Writing CSV ...')
    csv_df = g.copy()
    csv_df['geometry_wkt'] = csv_df.geometry.to_wkt()
    csv_df = csv_df.drop(columns=['geometry'])
    for col in csv_df.columns:
        sample = csv_df[col].dropna().head(1)
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict, tuple, np.ndarray)):
            csv_df[col] = csv_df[col].apply(_je)
    csv_df.to_csv(CSV_OUT, index=False)
    print(f'  saved ({CSV_OUT.stat().st_size/1e6:.1f} MB)')

    print('Writing FGB ...')
    fgb_df = g.copy()
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
        print(f'  saved ({fgb_target.stat().st_size/1e6:.1f} MB)')
    except (PermissionError, OSError):
        fgb_target = FGB_OUT.with_name('candidates_final_NEW.fgb')
        if fgb_target.exists():
            fgb_target.unlink()
        fgb_df.to_file(fgb_target, driver='FlatGeobuf')
        print(f'  FGB locked, wrote to {fgb_target}')

    # ---- Also update enriched source ----
    print('\nUpdating enriched source ...')
    enriched = gpd.read_parquet(ENRICHED_PATH)
    lookup_lon = g.set_index('candidate_id').centroid_lon
    lookup_lat = g.set_index('candidate_id').centroid_lat
    lookup_county = g.set_index('candidate_id').county_name
    mask = enriched.candidate_id.isin(g.candidate_id)
    enriched.loc[mask, 'centroid_lon'] = enriched.loc[mask, 'candidate_id'].map(lookup_lon).values
    enriched.loc[mask, 'centroid_lat'] = enriched.loc[mask, 'candidate_id'].map(lookup_lat).values
    enriched.loc[mask, 'county_name'] = enriched.loc[mask, 'candidate_id'].map(lookup_county).values
    enriched.to_parquet(ENRICHED_PATH, index=False)
    print(f'  patched {mask.sum():,} rows')


if __name__ == '__main__':
    main()
