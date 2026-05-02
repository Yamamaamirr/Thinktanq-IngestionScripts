"""Census TIGER/Line ingestion: primary roads + municipal boundaries.

Two datasets from the same Census TIGER source, both stored locally for
spatial joins against parcel polygons:

  Part 1: Primary roads (national)  -- interstates + US highways
  Part 2: Municipal boundaries      -- target states (CA, TX, AZ, NV, VA)

Source year: TIGER 2025 (most recent vintage at time of build, refreshed
annually by Census). Verified live as of build date.

Roads:
  Endpoint: https://www2.census.gov/geo/tiger/TIGER2025/PRIMARYROADS/
            tl_2025_us_primaryroads.zip   (~38 MB, ~13K road segments)
  RTTYP codes filtered to:
    I = Interstate Highway       -> is_interstate = True
    U = US Highway               -> is_us_highway = True
    S/C/M = state/county/common  -> not in the national primaryroads file;
            those live in per-state ROADS files (multi-GB total). Skipped
            until needed -- interstates are the main scoring driver per
            PRD section 8.5.

  MultiLineString features get exploded to LineStrings so PostGIS load
  matches GEOGRAPHY(LINESTRING, 4326) schema. Same pattern as pipelines.

Municipal boundaries:
  Endpoint: https://www2.census.gov/geo/tiger/TIGER2025/PLACE/
            tl_2025_{state_fips}_place.zip   (one file per state)
  No national file exists -- must download per state.
  Currently fetches the 5 initial target states. Add more FIPS codes to
  TARGET_STATE_FIPS to expand.

  PLACE = incorporated places only. Unincorporated areas are in COUSUB
  (county subdivisions); skip for now since PRD's SOI proxy uses
  incorporated municipalities.

Output: two parquet files in this directory:
  primary_roads.parquet
  municipalities.parquet
"""
import io
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError

TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER2025"
ROADS_URL = f"{TIGER_BASE}/PRIMARYROADS/tl_2025_us_primaryroads.zip"
PLACE_URL_TMPL = f"{TIGER_BASE}/PLACE/tl_2025_{{fips}}_place.zip"

TARGET_STATE_FIPS = {
    'CA': '06',
    'TX': '48',
    'AZ': '04',
    'NV': '32',
    'VA': '51',
}

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
ROADS_PARQUET = Path(__file__).parent / 'primary_roads.parquet'
MUNI_PARQUET = Path(__file__).parent / 'municipalities.parquet'

RETRY_DELAYS = [5, 15, 45, 120]
NETWORK_ERRORS = (ConnectionError, Timeout, ChunkedEncodingError, ProtocolError)


def _download_with_retry(url, dest_path, label):
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / 1e6
        print(f"  [{label}] cached ({size_mb:.1f} MB) at {dest_path.name}")
        return
    last_exc = None
    for attempt, delay in enumerate(RETRY_DELAYS, 1):
        try:
            print(f"  [{label}] downloading from {url} ...")
            r = requests.get(url, timeout=180, stream=True)
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            size_mb = dest_path.stat().st_size / 1e6
            print(f"  [{label}] downloaded {size_mb:.1f} MB -> {dest_path.name}")
            return
        except NETWORK_ERRORS as e:
            last_exc = e
            print(f"  [{label}] network error attempt {attempt} ({type(e).__name__}) -- retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"{label}: max retries exceeded ({last_exc!r})")


def _read_shapefile_from_zip(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        shp_name = next((n for n in zf.namelist() if n.endswith('.shp')), None)
        if shp_name is None:
            raise RuntimeError(f"No .shp in {zip_path}")
    return gpd.read_file(f"zip://{zip_path}!{shp_name}")


# ── Part 1: Primary roads ────────────────────────────────────────────────────

def process_roads():
    print("\n" + "=" * 70)
    print("Part 1: Primary roads (national)")
    print("=" * 70)
    zip_path = CACHE_DIR / 'tl_2025_us_primaryroads.zip'
    _download_with_retry(ROADS_URL, zip_path, 'roads')

    gdf = _read_shapefile_from_zip(zip_path)
    print(f"  raw rows: {len(gdf):,}")
    print(f"  columns: {list(gdf.columns)}")
    print(f"  CRS: {gdf.crs}")

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf = gdf[gdf['RTTYP'].isin(['I', 'U'])].copy()
    print(f"  after RTTYP filter (I + U): {len(gdf):,}")

    gdf['is_interstate'] = gdf['RTTYP'] == 'I'
    gdf['is_us_highway'] = gdf['RTTYP'] == 'U'
    gdf = gdf.rename(columns={
        'LINEARID': 'source_id',
        'FULLNAME': 'full_name',
        'RTTYP': 'route_type',
    })

    keep_cols = ['source_id', 'full_name', 'route_type',
                 'is_interstate', 'is_us_highway', 'geometry']
    gdf = gdf[keep_cols]

    # Explode MultiLineString into LineStrings (PostGIS schema requires LINESTRING).
    n_before = len(gdf)
    n_multi = (gdf.geom_type == 'MultiLineString').sum()
    print(f"\n  MultiLineString features before explode: {n_multi}")
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    print(f"  after explode: {len(gdf):,} rows (+{len(gdf)-n_before} component linestrings)")
    non_ls = gdf[gdf.geom_type != 'LineString']
    assert len(non_ls) == 0, f"Non-LineString geometries remain: {non_ls.geom_type.value_counts().to_dict()}"
    print("  geometry assertion passed: all features are LineString.")

    print(f"\n  route_type breakdown:\n{gdf['route_type'].value_counts().to_string()}")
    print(f"\n  is_interstate / is_us_highway counts:")
    print(f"    interstates: {gdf['is_interstate'].sum():,}")
    print(f"    US highways: {gdf['is_us_highway'].sum():,}")
    print(f"\n  sample full_name values (top 10 most-frequent):")
    print(gdf['full_name'].value_counts(dropna=False).head(10).to_string())
    print(f"\n  null counts:\n{gdf[['source_id','full_name','route_type']].isnull().sum().to_string()}")

    gdf.to_parquet(ROADS_PARQUET, index=False)
    print(f"\n  wrote {ROADS_PARQUET} ({ROADS_PARQUET.stat().st_size/1e6:.1f} MB)")
    return gdf


# ── Part 2: Municipal boundaries ─────────────────────────────────────────────

def process_municipalities():
    print("\n" + "=" * 70)
    print(f"Part 2: Municipal boundaries ({len(TARGET_STATE_FIPS)} states)")
    print("=" * 70)
    frames = []
    for state, fips in TARGET_STATE_FIPS.items():
        url = PLACE_URL_TMPL.format(fips=fips)
        zip_path = CACHE_DIR / f'tl_2025_{fips}_place.zip'
        _download_with_retry(url, zip_path, f'place {state}')
        gdf = _read_shapefile_from_zip(zip_path)
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        frames.append(gdf)

    print()
    combined = pd.concat(frames, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry='geometry', crs='EPSG:4326')
    print(f"  combined raw rows: {len(combined):,}")
    print(f"  columns: {list(combined.columns)}")

    combined = combined.rename(columns={
        'PLACEFP': 'place_fips',
        'STATEFP': 'state_fips',
        'NAME': 'name',
        'NAMELSAD': 'name_lsad',
        'LSAD': 'lsad',
        'FUNCSTAT': 'funcstat',
    })
    keep_cols = ['name', 'name_lsad', 'state_fips', 'place_fips', 'lsad', 'funcstat', 'geometry']
    keep_cols = [c for c in keep_cols if c in combined.columns]
    combined = combined[keep_cols]
    combined['source_id'] = combined['state_fips'] + combined['place_fips']

    # is_incorporated: only municipalities with actual governing authority should
    # trigger the SOI proxy. CDPs (LSAD 57) are statistical entities with no
    # government, no zoning, no annexation power -- exclude from SOI but keep
    # in the table so nearest_municipality / distance queries can still use them.
    #
    # Primary signal: FUNCSTAT A/B/C = active government (A=primary, B=consolidated
    # with another govt, C=consolidated city-county). This catches edge cases like
    # Carson City NV (consolidated city-county, LSAD=00 but FUNCSTAT=C).
    # LSAD fallback covers any active places Census hasn't assigned a standard code.
    # If geographic expansion adds NJ/AK/PA/etc., extend INCORPORATED_LSAD with
    # 21 (borough), 37 (municipality), 39/45 (town variants), 49 (consolidated city).
    INCORPORATED_LSAD = {'25', '43', '47'}  # city / town / village
    ACTIVE_FUNCSTAT = {'A', 'B', 'C'}       # active government
    combined['is_incorporated'] = (
        combined['lsad'].isin(INCORPORATED_LSAD)
        | combined['funcstat'].isin(ACTIVE_FUNCSTAT)
    )

    # Explode any MultiPolygons -> Polygons (some PLACE entries are split,
    # e.g. coastal cities with islands or annexations).
    n_before = len(combined)
    n_multi = (combined.geom_type == 'MultiPolygon').sum()
    print(f"\n  MultiPolygon features before explode: {n_multi}")
    combined = combined.explode(index_parts=False).reset_index(drop=True)
    print(f"  after explode: {len(combined):,} rows (+{len(combined)-n_before} component polygons)")
    non_poly = combined[combined.geom_type != 'Polygon']
    assert len(non_poly) == 0, f"Non-Polygon geometries remain: {non_poly.geom_type.value_counts().to_dict()}"
    print("  geometry assertion passed: all features are Polygon.")

    fips_to_state = {v: k for k, v in TARGET_STATE_FIPS.items()}
    print(f"\n  per-state row counts:")
    for fips, n in combined['state_fips'].value_counts().items():
        print(f"    {fips_to_state.get(fips, fips):<3} ({fips}): {n:,}")
    print(f"\n  LSAD x is_incorporated breakdown:")
    print(combined.groupby(['lsad', 'is_incorporated']).size().to_string())
    print(f"\n  is_incorporated totals:")
    print(combined['is_incorporated'].value_counts(dropna=False).to_string())

    # Spot-check known cases.
    print(f"\n  spot-check known incorporated (should be True):")
    for state_fips, name in [('06','Folsom'), ('06','Los Angeles'), ('32','Reno'),
                             ('48','Houston'), ('04','Phoenix'), ('51','Richmond')]:
        match = combined[(combined['state_fips']==state_fips) & (combined['name']==name)]
        if len(match):
            r = match.iloc[0]
            mark = 'OK' if r['is_incorporated'] else 'MISS'
            print(f"    [{mark}] {fips_to_state[state_fips]} {name:<15} LSAD={r['lsad']} is_incorporated={r['is_incorporated']}")
        else:
            print(f"    [N/A] {fips_to_state[state_fips]} {name} not found")

    print(f"\n  spot-check known CDPs / unincorporated (should be False):")
    for state_fips, name in [('06','Stanford'), ('06','Marin City'), ('48','The Woodlands'),
                             ('51','Centreville'), ('04','Sun City')]:
        match = combined[(combined['state_fips']==state_fips) & (combined['name']==name)]
        if len(match):
            r = match.iloc[0]
            mark = 'OK' if not r['is_incorporated'] else 'MISS'
            print(f"    [{mark}] {fips_to_state[state_fips]} {name:<15} LSAD={r['lsad']} is_incorporated={r['is_incorporated']}")
        else:
            print(f"    [N/A] {fips_to_state[state_fips]} {name} not found")

    combined.to_parquet(MUNI_PARQUET, index=False)
    print(f"\n  wrote {MUNI_PARQUET} ({MUNI_PARQUET.stat().st_size/1e6:.1f} MB)")
    return combined


if __name__ == '__main__':
    roads = process_roads()
    munis = process_municipalities()
    print("\n" + "=" * 70)
    print("DONE")
    print(f"  primary_roads.parquet:  {len(roads):,} LineStrings")
    print(f"  municipalities.parquet: {len(munis):,} Polygons")
    print("=" * 70)
