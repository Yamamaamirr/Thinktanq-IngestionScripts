"""
StepR1 -- Load all four reuse-node sources and normalise into a unified schema.

Reads:
  ingestion_scripts/eia_form860/eia_reuse_nodes.parquet         (242 EIA coal/gas)
  ingestion_scripts/nrc_nuclear/nuclear_sites.parquet           (7 nuclear)
  ingestion_scripts/epa_repowering/epa_repowering_sites.parquet (18,398 EPA)
  ingestion_scripts/osm_industrial/osm_industrial_sites.parquet (OSM polygons)

Writes:
  candidate_areas/reuse_node_outputs/reuse_nodes_raw.parquet

Unified schema:
  site_id              str   prefixed unique id (EIA-12345, NRC-Onofre, EPA-CR-X, OSM-way/123)
  site_name            str
  state_abbr           str   AZ|CA|NV|TX|VA
  county_name          str   joined from Census county boundaries
  source               str   EIA-860 | EIA-860-nuclear | EPA-RE-Powering | OpenStreetMap
  reuse_asset_type     str   coal_plant | gas_plant | nuclear | brownfield_land |
                              landfill | abandoned_mine | contaminated_land |
                              industrial_land | quarry_mine | industrial_works | power_plant
  reuse_status         str   operating | retired | retiring | active_industrial
  capacity_mw          float retired_capacity_mw from EIA/Nuclear (null elsewhere)
  acreage              float EPA-supplied or computed from polygon (null where unknown)
  retirement_year      int   actual or planned retirement (null for operating/industrial)
  dominant_technology  str   EIA-only (Combined Cycle, Conventional Steam Coal, etc.)
  epa_program          str   EPA-only (SUPERFUND, BROWNFIELDS, etc.)
  geometry_raw         geom  original geometry as-shipped (point for 3 sources, polygon for OSM)

This script does NOT apply the OSM-spatial-search footprint logic -- that is
StepR2's job. R1 just lands every source row into one table with consistent
columns + a single CRS (EPSG:4326).
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

EIA_PATH    = Path('ingestion_scripts/eia_form860/eia_reuse_nodes.parquet')
NUC_PATH    = Path('ingestion_scripts/nrc_nuclear/nuclear_sites.parquet')
EPA_PATH    = Path('ingestion_scripts/epa_repowering/epa_repowering_sites.parquet')
OSM_PATH    = Path('ingestion_scripts/osm_industrial/osm_industrial_sites.parquet')
COUNTY_PATH = Path('ingestion_scripts/census_tiger/county_boundaries.parquet')
OUT_PATH    = Path('candidate_areas/reuse_node_outputs/reuse_nodes_raw.parquet')

ACRES_PER_M2 = 0.000247105


def _load_eia():
    g = gpd.read_parquet(EIA_PATH)
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    # plant_id alone is not unique because a single plant can have both a
    # retired and a retiring portion (partial retirement). Include status.
    out = gpd.GeoDataFrame({
        'site_id':            'EIA-' + g['plant_id'].astype(str) + '-' + g['reuse_status'].astype(str),
        'site_name':          g['plant_name'],
        'state_abbr':         g['state_abbr'],
        'county_name':        None,
        'source':             'EIA-860',
        'reuse_asset_type':   g['reuse_asset_type'],
        'reuse_status':       g['reuse_status'],
        'capacity_mw':        g['retired_capacity_mw'].astype(float),
        'acreage':            pd.NA,
        'retirement_year':    g['retirement_year'],
        'dominant_technology': g.get('dominant_technology'),
        'epa_program':        None,
        'geometry':           g.geometry.values,
    }, geometry='geometry', crs='EPSG:4326')
    return out


def _load_nuclear():
    g = gpd.read_parquet(NUC_PATH)
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    out = gpd.GeoDataFrame({
        'site_id':            'NRC-' + g['site_name'].str.replace(r'\W+', '-', regex=True).str.strip('-'),
        'site_name':          g['site_name'],
        'state_abbr':         g['state_abbr'],
        'county_name':        None,
        'source':             'EIA-860-nuclear',
        'reuse_asset_type':   g['reuse_asset_type'],
        'reuse_status':       g['reuse_status'],
        'capacity_mw':        g['capacity_mw'].astype(float),
        'acreage':            pd.NA,
        'retirement_year':    g['retirement_year'],
        'dominant_technology': None,
        'epa_program':        None,
        'geometry':           g.geometry.values,
    }, geometry='geometry', crs='EPSG:4326')
    return out


def _load_epa():
    g = gpd.read_parquet(EPA_PATH)
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    # Map EPA's reuse_asset_type to our unified vocab (already aligned mostly):
    #   contaminated_land -> contaminated_land
    #   landfill          -> landfill
    #   abandoned_mine    -> abandoned_mine
    out = gpd.GeoDataFrame({
        'site_id':            'EPA-' + g['site_id'].astype(str),
        'site_name':          g['site_name'],
        'state_abbr':         g['state_abbr'],
        'county_name':        g.get('county'),
        'source':             'EPA-RE-Powering',
        'reuse_asset_type':   g['reuse_asset_type'],
        'reuse_status':       'active_industrial',   # EPA sites aren't power plants, no retirement concept
        'capacity_mw':        pd.NA,
        'acreage':            pd.to_numeric(g['acreage'], errors='coerce'),
        'retirement_year':    pd.NA,
        'dominant_technology': None,
        'epa_program':        g.get('epa_program'),
        'geometry':           g.geometry.values,
    }, geometry='geometry', crs='EPSG:4326')
    return out


def _load_osm():
    if not OSM_PATH.exists():
        print(f'  OSM file not yet generated at {OSM_PATH}; skipping OSM source')
        return None
    g = gpd.read_parquet(OSM_PATH)
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    out = gpd.GeoDataFrame({
        'site_id':            'OSM-' + g['osm_id'].astype(str),
        'site_name':          g['site_name'],
        'state_abbr':         g['state_abbr'],
        'county_name':        None,
        'source':             'OpenStreetMap',
        'reuse_asset_type':   g['reuse_asset_type'],
        'reuse_status':       'active_industrial',
        'capacity_mw':        pd.NA,
        'acreage':            g['area_acres'].astype(float),
        'retirement_year':    pd.NA,
        'dominant_technology': None,
        'epa_program':        None,
        'geometry':           g.geometry.values,
    }, geometry='geometry', crs='EPSG:4326')
    return out


def _fill_county_names(gdf):
    """Spatial-join centroid against Census county boundaries for any row missing county_name."""
    if not COUNTY_PATH.exists():
        print('  county_boundaries.parquet not found, skipping county fill')
        return gdf
    county = gpd.read_parquet(COUNTY_PATH)
    if county.crs is None or county.crs.to_epsg() != 4326:
        county = county.to_crs(4326)
    name_col = next((c for c in ['NAME', 'NAMELSAD', 'county_name', 'name'] if c in county.columns), None)
    if name_col is None:
        print('  no county-name column in county_boundaries.parquet, skipping')
        return gdf
    needs = gdf['county_name'].isna()
    if not needs.any():
        return gdf
    print(f'  filling county_name for {int(needs.sum()):,} rows via centroid spatial-join ...')
    # Use projected CRS for accurate centroids
    cents = gdf.loc[needs, ['site_id', 'geometry']].copy()
    cents_5070 = cents.to_crs(5070)
    cents_5070['geometry'] = cents_5070.geometry.centroid
    pts = cents_5070.to_crs(4326)
    joined = gpd.sjoin(pts, county[[name_col, 'geometry']], how='left', predicate='within')
    # Dedupe in case a centroid lands on a county boundary -> multiple matches
    joined = joined.drop_duplicates(subset='site_id', keep='first')
    lookup = joined.set_index('site_id')[name_col]
    gdf.loc[needs, 'county_name'] = gdf.loc[needs, 'site_id'].map(lookup).values
    return gdf


def main():
    print('Loading sources ...')
    frames = []
    for label, loader in [('EIA coal/gas', _load_eia),
                          ('Nuclear', _load_nuclear),
                          ('EPA RE-Powering', _load_epa),
                          ('OSM industrial', _load_osm)]:
        g = loader()
        if g is None:
            continue
        print(f'  {label}: {len(g):,} rows')
        frames.append(g)

    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                           geometry='geometry', crs='EPSG:4326')
    print(f'\nCombined: {len(out):,} reuse-node rows from {len(frames)} sources')

    # Drop rows without geometry (shouldn't happen but defensive)
    bad_geom = out.geometry.isna() | out.geometry.is_empty
    if bad_geom.any():
        print(f'  dropping {int(bad_geom.sum())} rows with bad geometry')
        out = out[~bad_geom].copy()

    # Fill missing county_name
    out = _fill_county_names(out)

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.0f} KB, {len(out):,} rows)')

    # ---- Summary ----
    print('\n=== Rows by source ===')
    print(out.source.value_counts().to_string())
    print('\n=== Rows by reuse_asset_type ===')
    print(out.reuse_asset_type.value_counts().to_string())
    print('\n=== Rows by reuse_status ===')
    print(out.reuse_status.value_counts(dropna=False).to_string())
    print('\n=== Per-state x source ===')
    print(out.groupby(['state_abbr', 'source']).size().unstack(fill_value=0).to_string())
    print('\n=== Coverage of optional fields ===')
    for col in ['capacity_mw', 'acreage', 'retirement_year', 'dominant_technology',
                'epa_program', 'county_name']:
        n_set = out[col].notna().sum()
        print(f'  {col:<22} populated: {n_set:>6,} / {len(out):>6,} ({100*n_set/len(out):.1f}%)')

    # Geometry type breakdown
    print('\n=== Geometry types by source ===')
    out['_gtype'] = out.geometry.type
    print(out.groupby(['source', '_gtype']).size().unstack(fill_value=0).to_string())

    # ---- Checks ----
    print('\n=== Checks ===')
    valid_states = {'AZ','CA','NV','TX','VA'}
    valid_status = {'operating', 'retired', 'retiring', 'active_industrial'}
    valid_types  = {'coal_plant','gas_plant','nuclear','brownfield_land','landfill',
                    'abandoned_mine','contaminated_land','industrial_land',
                    'quarry_mine','industrial_works','power_plant'}
    checks = {
        'Has rows'                 : len(out) > 0,
        'site_id unique'           : out.site_id.is_unique,
        'state_abbr in target set' : set(out.state_abbr.dropna().unique()) <= valid_states,
        'reuse_status valid set'   : set(out.reuse_status.dropna().unique()) <= valid_status,
        'reuse_asset_type valid'   : set(out.reuse_asset_type.dropna().unique()) <= valid_types,
        'CRS 4326'                 : out.crs.to_epsg() == 4326,
        'geometry valid'           : out.geometry.is_valid.all(),
        'no empty geometries'      : (~out.geometry.is_empty).all(),
    }
    ok = True
    for lbl, passed in checks.items():
        print(f'  [{"PASS" if passed else "FAIL"}] {lbl}')
        if not passed:
            ok = False
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
