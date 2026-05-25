"""
Nuclear reuse-node ingestion -- operating and retired reactor sites.

Owen's Infrastructure Reuse Nodes layer (May 1): "operating and retired
nuclear-adjacent sites from NRC/EIA". Retired nuclear sites are strong
redevelopment candidates -- very high-voltage interconnections, large land
footprints, cooling water rights, existing transmission.

Source: ingestion_scripts/eia_form860/cache/out_eia__yearly_generators.parquet
        (PUDL cleaned EIA Form 860, already downloaded -- no network needed)

Source choice (Phase 1)
-----------------------
Owen named NRC as a nuclear source, but NRC publishes only HTML tables and PDF
maps -- no machine-readable download or API, and the tables carry no
coordinates. EIA Form 860 carries the same reactor plants in clean tabular form
WITH coordinates, capacity, and operating/retired status. For Phase 1 the
nuclear layer is built from EIA 860 alone.

Known Phase 1 limitation: EIA 860's modern reporting window does not include
reactors decommissioned long ago (e.g. Rancho Seco 1989, Humboldt Bay 1976).
Those can be added in a follow-up pass from NRC decommissioning records if the
pre-2001 decommissioned universe is wanted. Recently retired reactors such as
San Onofre ARE present.

Output: ingestion_scripts/nrc_nuclear/nuclear_sites.parquet (EPSG:4326 points)
  Columns:
    site_name, state_abbr, reuse_status (operating|retired),
    reuse_asset_type ('nuclear'), capacity_mw, retirement_year, source, geometry

Run: python ingestion_scripts/nrc_nuclear/ingest_nuclear_sites.py
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

CACHE = Path('ingestion_scripts/eia_form860/cache/out_eia__yearly_generators.parquet')
OUT_PATH = Path('ingestion_scripts/nrc_nuclear/nuclear_sites.parquet')

TARGET_STATES = {'AZ', 'CA', 'NV', 'TX', 'VA'}


def main():
    print(f'Loading PUDL cache: {CACHE} ...')
    df = pd.read_parquet(CACHE)
    df['report_year'] = pd.to_datetime(df['report_date']).dt.year
    df = df[df['report_year'] == df['report_year'].max()].copy()

    # Nuclear generators in target states
    nuc = df[(df['fuel_type_code_pudl'] == 'nuclear')
             & (df['state'].isin(TARGET_STATES))].copy()
    print(f'  {len(nuc):,} nuclear generators in 5 target states')

    # Aggregate generators -> plant-level sites
    rows = []
    for plant_id, grp in nuc.groupby('plant_id_eia', observed=True):
        first = grp.iloc[0]
        statuses = set(grp['operational_status'])
        # A site counts as 'retired' only if every reactor unit is retired
        is_retired = statuses == {'retired'}
        retire_year = None
        if is_retired:
            yrs = pd.to_datetime(grp['generator_retirement_date'],
                                 errors='coerce').dt.year.dropna()
            retire_year = int(yrs.max()) if len(yrs) else None
        rows.append({
            'site_name':        first['plant_name_eia'],
            'state_abbr':       first['state'],
            'reuse_status':     'retired' if is_retired else 'operating',
            'reuse_asset_type': 'nuclear',
            'capacity_mw':      float(grp['capacity_mw'].sum()),
            'retirement_year':  retire_year,
            'source':           'EIA-860',
            'latitude':         first['latitude'],
            'longitude':        first['longitude'],
        })
    print(f'  {len(rows)} nuclear plant sites (aggregated from generators)')

    out = pd.DataFrame(rows)
    n_no_geom = out['latitude'].isna().sum()
    if n_no_geom:
        print(f'  WARNING: {n_no_geom} sites with no coordinates -- dropping')
        out = out[out['latitude'].notna()].copy()

    gdf = gpd.GeoDataFrame(
        out,
        geometry=[Point(xy) for xy in zip(out['longitude'], out['latitude'])],
        crs='EPSG:4326',
    ).drop(columns=['latitude', 'longitude'])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({len(gdf)} sites)')

    print('\n=== Nuclear sites ===')
    for _, r in gdf.sort_values(['state_abbr', 'site_name']).iterrows():
        yr = int(r.retirement_year) if pd.notna(r.retirement_year) else '----'
        print(f'  {r.site_name[:38]:<38} {r.state_abbr}  {r.capacity_mw:>6.0f} MW  '
              f'{r.reuse_status:<10} retired_year={yr}')

    print('\n=== Summary ===')
    print(gdf.groupby(['state_abbr', 'reuse_status']).size().unstack(fill_value=0).to_string())

    print('\n=== Checks ===')
    checks = {
        'Has rows'             : len(gdf) > 0,
        'All in target states' : set(gdf.state_abbr.unique()) <= TARGET_STATES,
        'reuse_status valid'   : set(gdf.reuse_status.unique()) <= {'operating', 'retired'},
        'reuse_asset_type all nuclear': (gdf.reuse_asset_type == 'nuclear').all(),
        'geometry non-null'    : gdf.geometry.notna().all(),
        'CRS 4326'             : gdf.crs.to_epsg() == 4326,
        'capacity positive'    : (gdf.capacity_mw > 0).all(),
        'San Onofre present (retired SONGS)': gdf.site_name.str.contains('Onofre', case=False).any(),
    }
    ok = True
    for lbl, passed in checks.items():
        print(f'  [{"PASS" if passed else "FAIL"}] {lbl}')
        if not passed:
            ok = False
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
