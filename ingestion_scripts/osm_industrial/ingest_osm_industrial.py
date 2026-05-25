"""
OSM industrial + power-plant reuse-node ingestion.

Owen's Infrastructure Reuse Nodes layer (May 1) lists a 4th type that the
EIA / nuclear / EPA sources do not cover: "large industrial sites with power,
rail, gas, water, or substation adjacency". OpenStreetMap is the only free,
reproducible national source that carries these AS POLYGONS -- which also
gives real footprints (the EIA/EPA sources are points only).

Tags pulled (per state, via Overpass through osmnx):
  landuse=industrial   -- industrial land parcels / districts
  landuse=brownfield   -- brownfield land
  landuse=landfill     -- landfill sites
  landuse=quarry       -- quarries / open-pit mines
  power=plant          -- power plant site footprints
  man_made=works       -- industrial works / processing facilities

The expanded tag set (per GIS expert recommendation) catches the inconsistent
ways OSM contributors tag industrial sites -- many real reuse candidates fall
under brownfield / landfill / quarry / man_made=works rather than the strict
landuse=industrial tag.

Output: ingestion_scripts/osm_industrial/osm_industrial_sites.parquet (EPSG:4326)
  Columns:
    osm_id, site_name, state_abbr, reuse_asset_type
      (industrial_land | power_plant | brownfield_land | landfill |
       quarry_mine | industrial_works), area_acres, source, geometry (polygon)

A 50-acre floor (matches the candidate-pipeline minimum) drops trivial
parcels and keeps genuinely large sites.

Run: python ingestion_scripts/osm_industrial/ingest_osm_industrial.py
"""
from pathlib import Path
import warnings
import pandas as pd
import geopandas as gpd
import osmnx as ox

warnings.filterwarnings('ignore')

OUT_PATH = Path('ingestion_scripts/osm_industrial/osm_industrial_sites.parquet')
CACHE_DIR = Path('ingestion_scripts/osm_industrial/cache')

STATE_PLACES = {
    'AZ': 'Arizona, USA',
    'CA': 'California, USA',
    'NV': 'Nevada, USA',
    'TX': 'Texas, USA',
    'VA': 'Virginia, USA',
}
TAGS = {
    'landuse':   ['industrial', 'brownfield', 'landfill', 'quarry'],
    'power':     'plant',
    'man_made':  'works',
}
MIN_ACRES = 50.0
MAX_ACRES = 5000.0   # match the candidate-pipeline site-scale cap
ACRES_PER_M2 = 0.000247105


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Point osmnx cache here so re-runs skip the Overpass fetch
    ox.settings.cache_folder = str(CACHE_DIR)
    ox.settings.use_cache = True
    ox.settings.overpass_rate_limit = True

    frames = []
    for state, place in STATE_PLACES.items():
        print(f'\n-- {state} ({place}) --------------------------------')
        try:
            gdf = ox.features_from_place(place, tags=TAGS)
        except Exception as e:
            print(f'  ERROR fetching {state}: {type(e).__name__}: {e}')
            continue
        print(f'  raw OSM features: {len(gdf):,}')

        # Keep polygon / multipolygon only (drop points and lines)
        gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
        print(f'  polygon features: {len(gdf):,}')
        if len(gdf) == 0:
            continue

        # Classify asset type from whichever tag is set on the feature.
        # Priority: power_plant > quarry_mine > landfill > brownfield_land >
        # industrial_works > industrial_land
        def _classify(r):
            if r.get('power') == 'plant':
                return 'power_plant'
            lu = r.get('landuse')
            if lu == 'quarry':
                return 'quarry_mine'
            if lu == 'landfill':
                return 'landfill'
            if lu == 'brownfield':
                return 'brownfield_land'
            if r.get('man_made') == 'works':
                return 'industrial_works'
            if lu == 'industrial':
                return 'industrial_land'
            return 'industrial_land'   # fallback
        # Ensure tag columns exist (a state may have 0 features for a tag)
        for col in ('power', 'landuse', 'man_made'):
            if col not in gdf.columns:
                gdf[col] = None
        gdf['reuse_asset_type'] = gdf.apply(_classify, axis=1)

        # Name
        name = gdf['name'] if 'name' in gdf.columns else pd.Series(None, index=gdf.index)

        # OSM id from the (element_type, osmid) index
        osm_ids = [f'{et}/{oid}' for et, oid in gdf.index]

        sub = gpd.GeoDataFrame({
            'osm_id':           osm_ids,
            'site_name':        name.values,
            'state_abbr':       state,
            'reuse_asset_type': gdf['reuse_asset_type'].values,
            'geometry':         gdf.geometry.values,
        }, geometry='geometry', crs=gdf.crs)
        frames.append(sub)

    if not frames:
        print('\nNo OSM features fetched. Aborting.')
        return

    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                           geometry='geometry', crs='EPSG:4326')
    print(f'\nCombined polygon features: {len(out):,}')

    # Compute area in EPSG:5070, apply size band
    out_5070 = out.to_crs(5070)
    out['area_acres'] = out_5070.geometry.area * ACRES_PER_M2
    before = len(out)
    out = out[(out['area_acres'] >= MIN_ACRES) & (out['area_acres'] <= MAX_ACRES)].copy()
    print(f'  After {MIN_ACRES:.0f}-{MAX_ACRES:.0f} ac band: {len(out):,} '
          f'(dropped {before - len(out):,})')

    # Deduplicate, two passes:
    #   (1) same osm_id returned by multiple state queries (borderline features)
    #   (2) same physical feature returned as both way and relation (different
    #       osm_ids but identical centroid + asset type)
    before = len(out)
    out = out.drop_duplicates(subset='osm_id', keep='first')
    cent = out.to_crs(5070).geometry.centroid
    out['_dupe_key'] = list(zip(cent.x.round().astype(int),
                                cent.y.round().astype(int),
                                out['reuse_asset_type']))
    out = out.drop_duplicates(subset='_dupe_key', keep='first').drop(columns='_dupe_key')
    print(f'  After dedup: {len(out):,} (dropped {before - len(out):,})')

    out['source'] = 'OpenStreetMap'
    out = out.sort_values('area_acres', ascending=False).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.0f} KB, {len(out):,} sites)')

    # ---- Summary ----
    print('\n=== Sites by state x asset type ===')
    print(out.groupby(['state_abbr', 'reuse_asset_type']).size().unstack(fill_value=0).to_string())
    print('\n=== Acreage distribution ===')
    print(f'  min={out.area_acres.min():.0f}, median={out.area_acres.median():.0f}, '
          f'p90={out.area_acres.quantile(0.9):.0f}, max={out.area_acres.max():,.0f}')
    print('\n=== Largest 10 sites ===')
    for _, r in out.head(10).iterrows():
        nm = str(r.site_name)[:38] if pd.notna(r.site_name) else '(unnamed)'
        print(f'  {nm:<40} {r.state_abbr}  {r.area_acres:>9,.0f} ac  {r.reuse_asset_type}')

    # ---- Checks ----
    print('\n=== Checks ===')
    checks = {
        'Has rows'              : len(out) > 0,
        'All in target states'  : set(out.state_abbr.unique()) <= set(STATE_PLACES),
        'reuse_asset_type valid': set(out.reuse_asset_type.unique()) <= {
            'industrial_land', 'power_plant', 'brownfield_land',
            'landfill', 'quarry_mine', 'industrial_works'},
        'all polygons'          : out.geometry.type.isin(['Polygon', 'MultiPolygon']).all(),
        'area in [floor, cap]'  : ((out.area_acres >= MIN_ACRES) & (out.area_acres <= MAX_ACRES)).all(),
        'osm_id unique'         : out.osm_id.is_unique,
        'CRS 4326'              : out.crs.to_epsg() == 4326,
        'geometry valid'        : out.geometry.is_valid.all(),
    }
    ok = True
    for lbl, passed in checks.items():
        print(f'  [{"PASS" if passed else "FAIL"}] {lbl}')
        if not passed:
            ok = False
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
