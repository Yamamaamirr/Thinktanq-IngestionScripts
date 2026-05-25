"""
EIA Form 860 reuse-node ingestion -- retired and retiring coal/gas plants.

Owen's Infrastructure Reuse Nodes layer (May 1): "retired / retiring coal and
gas plants from EIA 860 / 860M". These are redevelopment candidates -- sites
where the grid interconnection already physically exists.

Source: ingestion_scripts/eia_form860/cache/out_eia__yearly_generators.parquet
        (PUDL cleaned EIA 860, already downloaded -- no network needed)

This script re-cuts that cache specifically for reuse nodes. It differs from
eia_form860_retired.parquet in four ways:
  1. Carries the actual retirement date (generator_retirement_date /
     planned_generator_retirement_date), not just the report year.
  2. Aggregates generator rows to PLANT level -- one row per site.
  3. Filters to coal + gas only (Owen's spec). Oil/waste/solar/wind/hydro
     retired gensets are dropped -- they are not meaningful reuse sites.
  4. Splits into 'retired' (already down) and 'retiring' (planned future date).

A 10 MW plant-total floor drops trivial gensets while keeping every real plant.

Output: ingestion_scripts/eia_form860/eia_reuse_nodes.parquet (EPSG:4326 points)
  Columns:
    plant_id, plant_name, state_abbr, reuse_status (retired|retiring),
    reuse_asset_type (coal_plant|gas_plant), retired_capacity_mw,
    retirement_year, n_generators, dominant_technology, geometry

Run: python ingestion_scripts/eia_form860/ingest_eia_reuse_nodes.py
"""
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

CACHE = Path('ingestion_scripts/eia_form860/cache/out_eia__yearly_generators.parquet')
OUT_PATH = Path('ingestion_scripts/eia_form860/eia_reuse_nodes.parquet')

TARGET_STATES = {'AZ', 'CA', 'NV', 'TX', 'VA'}
REUSE_FUELS = {'coal': 'coal_plant', 'gas': 'gas_plant'}
MIN_PLANT_MW = 10.0   # drop trivial gensets; keep every real plant


def main():
    print(f'Loading PUDL cache: {CACHE} ...')
    df = pd.read_parquet(CACHE)
    print(f'  {len(df):,} generator-year rows')

    # Keep the most recent report year only
    df['report_year'] = pd.to_datetime(df['report_date']).dt.year
    latest = df['report_year'].max()
    df = df[df['report_year'] == latest].copy()
    print(f'  Latest report year {latest}: {len(df):,} generator rows')

    # Target states + coal/gas only
    df = df[df['state'].isin(TARGET_STATES)].copy()
    df = df[df['fuel_type_code_pudl'].isin(REUSE_FUELS.keys())].copy()
    print(f'  Coal/gas generators in 5 target states: {len(df):,}')

    # Classify reuse_status per generator
    today = date.today()

    def _gen_status(r):
        if r['operational_status'] == 'retired':
            return 'retired'
        # 'retiring' = still existing but has a planned retirement date
        prd = r.get('planned_generator_retirement_date')
        if pd.notna(prd) and r['operational_status'] == 'existing':
            return 'retiring'
        return 'other'

    df['reuse_status'] = df.apply(_gen_status, axis=1)
    df = df[df['reuse_status'].isin(['retired', 'retiring'])].copy()
    print(f'  Retired or retiring coal/gas generators: {len(df):,}')
    print(f'    retired:  {(df.reuse_status == "retired").sum():,}')
    print(f'    retiring: {(df.reuse_status == "retiring").sum():,}')

    if len(df) == 0:
        print('No reuse-node generators found. Aborting.')
        return

    # Retirement year per generator (actual date for retired, planned for retiring)
    def _retire_year(r):
        if r['reuse_status'] == 'retired':
            d = r.get('generator_retirement_date')
        else:
            d = r.get('planned_generator_retirement_date')
        if pd.notna(d):
            return pd.to_datetime(d).year
        return None

    df['retire_year'] = df.apply(_retire_year, axis=1)

    # ---- Aggregate generator rows -> plant-level sites ----
    print('\nAggregating generators to plant level ...')
    rows = []
    for (plant_id, status), grp in df.groupby(['plant_id_eia', 'reuse_status'], observed=True):
        total_mw = grp['capacity_mw'].sum()
        if total_mw < MIN_PLANT_MW:
            continue
        # Dominant fuel by capacity
        fuel_by_mw = grp.groupby('fuel_type_code_pudl', observed=True)['capacity_mw'].sum()
        dominant_fuel = fuel_by_mw.idxmax()
        # Dominant technology
        tech_by_mw = grp.groupby('technology_description', observed=True)['capacity_mw'].sum()
        dominant_tech = tech_by_mw.idxmax() if len(tech_by_mw) else None
        # Retirement year: latest generator to retire = when the whole site went down
        years = grp['retire_year'].dropna()
        retire_year = int(years.max()) if len(years) else None
        first = grp.iloc[0]
        rows.append({
            'plant_id':            str(plant_id),
            'plant_name':          first['plant_name_eia'],
            'state_abbr':          first['state'],
            'reuse_status':        status,
            'reuse_asset_type':    REUSE_FUELS[dominant_fuel],
            'retired_capacity_mw': float(total_mw),
            'retirement_year':     retire_year,
            'n_generators':        int(len(grp)),
            'dominant_technology': dominant_tech,
            'latitude':            first['latitude'],
            'longitude':           first['longitude'],
        })

    out = pd.DataFrame(rows)
    print(f'  {len(out):,} plant-level reuse-node sites (>= {MIN_PLANT_MW} MW)')

    # Drop rows with no coordinates
    n_no_geom = out['latitude'].isna().sum()
    if n_no_geom:
        print(f'  Dropping {n_no_geom} sites with no coordinates')
        out = out[out['latitude'].notna()].copy()

    # Build GeoDataFrame
    gdf = gpd.GeoDataFrame(
        out,
        geometry=[Point(xy) for xy in zip(out['longitude'], out['latitude'])],
        crs='EPSG:4326',
    ).drop(columns=['latitude', 'longitude'])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.0f} KB, {len(gdf):,} sites)')

    # ---- Summary ----
    print('\n=== Reuse-node sites by state x status ===')
    print(gdf.groupby(['state_abbr', 'reuse_status']).size().unstack(fill_value=0).to_string())
    print('\n=== By asset type ===')
    print(gdf.reuse_asset_type.value_counts().to_string())
    print('\n=== Capacity distribution (MW) ===')
    print(f'  min={gdf.retired_capacity_mw.min():.1f}, median={gdf.retired_capacity_mw.median():.1f}, '
          f'max={gdf.retired_capacity_mw.max():.1f}')
    print('\n=== Retirement year range ===')
    yrs = gdf.retirement_year.dropna()
    if len(yrs):
        print(f'  {int(yrs.min())} - {int(yrs.max())}, {gdf.retirement_year.isna().sum()} sites with no year')
    print('\n=== Largest 10 reuse-node sites ===')
    top = gdf.nlargest(10, 'retired_capacity_mw')
    for _, r in top.iterrows():
        yr = int(r.retirement_year) if pd.notna(r.retirement_year) else '----'
        print(f'  {r.plant_name[:32]:<32} {r.state_abbr}  {r.retired_capacity_mw:>7.0f} MW  '
              f'{r.reuse_asset_type:<10} {r.reuse_status:<8} {yr}')

    # ---- Checks ----
    print('\n=== Checks ===')
    checks = {
        'Has rows'              : len(gdf) > 0,
        'plant_id unique x status': not gdf.duplicated(['plant_id','reuse_status']).any(),
        'All in target states'  : set(gdf.state_abbr.unique()) <= TARGET_STATES,
        'reuse_status valid'     : set(gdf.reuse_status.unique()) <= {'retired','retiring'},
        'reuse_asset_type valid' : set(gdf.reuse_asset_type.unique()) <= {'coal_plant','gas_plant'},
        'capacity >= floor'      : (gdf.retired_capacity_mw >= MIN_PLANT_MW).all(),
        'geometry non-null'      : gdf.geometry.notna().all(),
        'CRS 4326'               : gdf.crs.to_epsg() == 4326,
    }
    ok = True
    for lbl, passed in checks.items():
        print(f'  [{"PASS" if passed else "FAIL"}] {lbl}')
        if not passed:
            ok = False
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
