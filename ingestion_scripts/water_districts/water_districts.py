"""EPA Community Water Service Area Boundaries ingestion.

Source: EPA's "Community Water Service Area Boundaries" hosted feature service
        services.arcgis.com/cJ9YHowT8TU7DUyn/.../Water_System_Boundaries
        Owner: EPA Office of Research and Development.
        Coverage: ~44,656 Community Water System (CWS) service area polygons
                  with real boundary geometry — national scope.

What this dataset IS:
  Community Water System (CWS) service areas only -- the systems that provide
  water for residential consumption to year-round populations. Each polygon is
  PWSID-linked back to EPA SDWIS for population/connection counts.

What this dataset IS NOT:
  - Does NOT include Non-Transient or Transient Non-Community Water Systems
    (schools, factories, campgrounds with their own wells). Those exist in
    SDWIS but have no published service-area polygons.
  - Does NOT include private wells / individual rural service. A parcel
    outside all polygons is genuinely outside any community water system.

PRD limitations / known caveats (documented for downstream consumers):

  Limitation 1 -- 'constrained' classification requires analyst input.
  The PRD's `constrained` water-availability class needs jurisdiction-level
  active water-use restrictions, which come from Owen's analyst reference
  tables (not yet built). Until then, this script can only distinguish:
      'abundant'      -- inside a CWS service area polygon
      'unavailable'   -- outside all polygons (set by enrichment, not here)
      'unknown'       -- system too small to serve a data center
                         (pop_served < 100, e.g. mobile-home parks, HOAs)
  When the analyst restriction tables come online, the enrichment job
  reclassifies in-CWS parcels as 'constrained' if the system's primacy_agency
  + jurisdiction is in the restriction list.

  Limitation 2 -- Rural agricultural parcels frequently score 'unavailable'.
  This is correct, not a data gap. Most 50-500 acre ag parcels in the
  western US are outside municipal/CWS boundaries because they rely on
  private wells. The PRD scoring penalises this for large parcels in arid
  states (zeroes Dimension 4) -- expected behavior.

  Limitation 3 -- Verification quality varies.
  ~70% of polygons are 'Modeled' (Random Forest from EPA Office of Research
  and Development), the rest are 'Confirmed' (provided directly by states or
  utilities). For high-stakes site decisions, prefer Confirmed polygons --
  the verification_status column is preserved for analyst filtering.

Output schema (matches water_districts table; PostGIS auto-generates district_id):
  pwsid             TEXT          EPA Public Water System ID
  district_name     TEXT          PWS_Name
  state_abbr        TEXT          2-letter state code (Primacy_Agency)
  service_area_type TEXT          Municipality / Census Place / etc.
  pop_served        INTEGER       Population_Served_Count
  service_connections INTEGER     Service_Connections_Count
  verification      TEXT          Modeled / Confirmed / etc.
  data_provider     TEXT          original data provider (state agency or EPA)
  availability      TEXT          abundant (always for CWS records here)
  has_geometry      BOOLEAN       True (every record in this dataset has geom)
  geometry          Polygon or MultiPolygon (4326) -- kept intact, NOT exploded.
                    A water utility serving disconnected service islands is one
                    operational entity (same PWSID/operator/population) and
                    splitting risks duplicate matches in spatial joins.
                    PostGIS table column should be GEOGRAPHY(MULTIPOLYGON, 4326).
  ingested_at       TIMESTAMPTZ
"""
import io
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError

BASE_URL = (
    'https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/'
    'Water_System_Boundaries/FeatureServer/0/query'
)
PAGE_SIZE = 1000  # smaller pages -- EPA service 504s on 2000-row pages near the tail

# Census cartographic state boundaries (1:500K, ~3 MB). Used to recover the
# real 2-letter state code for tribal/territorial water systems whose
# Primacy_Agency is an EPA region number (1-10), not a state -- the EPA holds
# primacy on tribal land. The PWSID prefix is also the region number, so it
# can't disambiguate (Region 9 alone spans CA/AZ/NV/HI).
STATE_BOUNDARIES_URL = (
    'https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_state_500k.zip'
)
STATE_BOUNDARIES_CACHE = Path(__file__).parent / 'cb_2024_us_state_500k.parquet'

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = Path(__file__).parent / 'raw_water_districts.parquet'
OUT_PARQUET = Path(__file__).parent / 'water_districts.parquet'

OUTFIELDS = ','.join([
    'PWSID', 'PWS_Name', 'Primacy_Agency', 'Service_Area_Type',
    'Population_Served_Count', 'Service_Connections_Count',
    'Symbology_Field', 'Original_Data_Provider',
])

RETRY_DELAYS = [5, 10, 20, 30, 60, 120, 180]
NETWORK_ERRORS = (ConnectionError, Timeout, ChunkedEncodingError, ProtocolError,
                  requests.exceptions.JSONDecodeError)
RETRY_HTTP_STATUSES = {429, 500, 502, 503, 504}  # transient server errors


def _fetch_page(offset):
    params = {
        'where': '1=1',
        'outFields': OUTFIELDS,
        'resultOffset': offset,
        'resultRecordCount': PAGE_SIZE,
        'f': 'geojson',
    }
    last_exc = None
    for delay in RETRY_DELAYS:
        try:
            r = requests.get(BASE_URL, params=params, timeout=120)
            if r.status_code in RETRY_HTTP_STATUSES:
                last_exc = RuntimeError(f"HTTP {r.status_code}")
                print(f"  Server error HTTP {r.status_code} -- retry in {delay}s")
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.json().get('features', [])
        except NETWORK_ERRORS as e:
            last_exc = e
            print(f"  Network error: {type(e).__name__} -- retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"Max retries exceeded ({last_exc!r})")


def _load_state_boundaries():
    """Fetch & cache Census state cartographic boundaries (1:500K)."""
    if STATE_BOUNDARIES_CACHE.exists():
        return gpd.read_parquet(STATE_BOUNDARIES_CACHE)
    print(f"Fetching Census state boundaries: {STATE_BOUNDARIES_URL}")
    last_exc = None
    for delay in RETRY_DELAYS:
        try:
            r = requests.get(STATE_BOUNDARIES_URL, timeout=120)
            if r.status_code in RETRY_HTTP_STATUSES:
                last_exc = RuntimeError(f"HTTP {r.status_code}")
                print(f"  Server error HTTP {r.status_code} -- retry in {delay}s")
                time.sleep(delay)
                continue
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                tmp_dir = STATE_BOUNDARIES_CACHE.parent / '_state_boundary_tmp'
                tmp_dir.mkdir(exist_ok=True)
                zf.extractall(tmp_dir)
                shp = next(tmp_dir.glob('*.shp'))
                states = gpd.read_file(shp)[['STUSPS', 'geometry']]
            states = states.to_crs(epsg=4326)
            states.to_parquet(STATE_BOUNDARIES_CACHE)
            return states
        except NETWORK_ERRORS as e:
            last_exc = e
            print(f"  Network error: {type(e).__name__} -- retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"Failed to fetch state boundaries: {last_exc!r}")


# ── Fetch ────────────────────────────────────────────────────────────────────

if CACHE_FILE.exists():
    print(f"Loading cache: {CACHE_FILE.name}")
    gdf = gpd.read_parquet(CACHE_FILE)
else:
    print("Fetching from EPA Community Water Service Area Boundaries...")
    # Per-page parquet caching: each successful page is written to cache/page_NNNN.parquet
    # so a mid-fetch crash doesn't lose progress. Resume picks up from highest page number.
    existing = sorted(CACHE_DIR.glob('page_*.parquet'))
    if existing:
        last_page = int(existing[-1].stem.split('_')[1])
        offset = (last_page + 1) * PAGE_SIZE
        print(f"  resuming from offset {offset:,} ({len(existing)} pages cached)")
    else:
        last_page = -1
        offset = 0

    page_index = last_page + 1
    while True:
        feats = _fetch_page(offset)
        if not feats:
            break
        page_path = CACHE_DIR / f'page_{page_index:04d}.parquet'
        gpd.GeoDataFrame.from_features(feats, crs='EPSG:4326').to_parquet(page_path)
        offset += len(feats)
        page_index += 1
        print(f"  fetched {offset:,} features (saved page {page_index - 1})")
        if len(feats) < PAGE_SIZE:
            break

    # Merge all pages into single cache parquet
    print("  merging page files...")
    pages = [gpd.read_parquet(p) for p in sorted(CACHE_DIR.glob('page_*.parquet'))]
    gdf = pd.concat(pages, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326')
    gdf.to_parquet(CACHE_FILE)
    print(f"  cached {len(gdf):,} rows -> {CACHE_FILE.name}")

print(f"\nRaw rows: {len(gdf):,}")
print(f"Columns: {list(gdf.columns)}")

# ── Process ──────────────────────────────────────────────────────────────────

n_before_geom = len(gdf)
gdf = gdf.dropna(subset='geometry')
print(f"After geometry filter: {len(gdf):,} (dropped {n_before_geom - len(gdf)})")

if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)

# Rename to canonical schema names.
gdf = gdf.rename(columns={
    'PWSID':                     'pwsid',
    'PWS_Name':                  'district_name',
    'Primacy_Agency':            'state_abbr',
    'Service_Area_Type':         'service_area_type',
    'Population_Served_Count':   'pop_served',
    'Service_Connections_Count': 'service_connections',
    'Symbology_Field':           'verification',
    'Original_Data_Provider':    'data_provider',
})

# Coerce numeric fields.
for col in ('pop_served', 'service_connections'):
    if col in gdf.columns:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce').astype('Int64')

# Resolve tribal/territorial state codes. Primacy_Agency is the EPA region
# number (1-10) for tribal systems instead of a 2-letter state code, since
# EPA -- not the state -- holds primacy on tribal land. Recover the actual
# state by spatial-joining the polygon centroid against US state boundaries.
numeric_state_mask = gdf['state_abbr'].astype(str).str.match(r'^\d+$', na=False)
n_numeric = int(numeric_state_mask.sum())
if n_numeric:
    print(f"\nResolving {n_numeric} tribal/territorial rows to actual states...")
    states = _load_state_boundaries()
    # representative_point() is guaranteed inside the polygon, while centroid
    # can fall outside concave shapes (e.g. coastal rancherias whose centroid
    # lands offshore).
    rep_pts = gdf.loc[numeric_state_mask].geometry.representative_point().values
    cents = gpd.GeoDataFrame(
        {'pwsid': gdf.loc[numeric_state_mask, 'pwsid'].values},
        geometry=rep_pts,
        crs='EPSG:4326',
    )
    joined = gpd.sjoin(cents, states, how='left', predicate='within')
    resolved = joined.drop_duplicates('pwsid').set_index('pwsid')['STUSPS']
    new_codes = gdf.loc[numeric_state_mask, 'pwsid'].map(resolved)
    gdf.loc[numeric_state_mask, 'state_abbr'] = new_codes.fillna(
        gdf.loc[numeric_state_mask, 'state_abbr']
    )
    still_numeric = int(
        gdf['state_abbr'].astype(str).str.match(r'^\d+$', na=False).sum())
    print(f"  resolved {n_numeric - still_numeric}/{n_numeric}; "
          f"unresolved {still_numeric} (likely offshore/territorial)")

# Drop rows without a district_name -- can't be useful downstream.
n_before_name = len(gdf)
gdf = gdf[gdf['district_name'].notna()]
print(f"After district_name filter: {len(gdf):,} (dropped {n_before_name - len(gdf)})")

# Availability classification. All polygons in this dataset are CWS, so
# every record starts as 'abundant'. Owen's analyst tables will later
# downgrade specific systems/jurisdictions to 'constrained'. Setting any
# pop_served < 100 to 'unknown' as a flag for manual review (very small
# systems may not actually have capacity for a data center).
gdf['availability'] = 'abundant'
gdf.loc[gdf['pop_served'].fillna(0) < 100, 'availability'] = 'unknown'
gdf['has_geometry'] = True

# Reorder columns.
keep_cols = ['pwsid', 'district_name', 'state_abbr', 'service_area_type',
             'pop_served', 'service_connections', 'verification',
             'data_provider', 'availability', 'has_geometry', 'geometry']
gdf = gdf[keep_cols].copy()

# Keep MultiPolygons intact -- a water utility serving multiple disconnected
# service islands is still ONE operational entity (same PWSID, operator,
# population total). Splitting into separate rows would fragment that single
# utility into multiple records all carrying the same PWSID, which is
# misleading and risks duplicate matches in spatial joins.
# The PostGIS table column for water_districts is therefore GEOGRAPHY(MULTIPOLYGON, 4326).
n_multi = (gdf.geom_type == 'MultiPolygon').sum()
n_single = (gdf.geom_type == 'Polygon').sum()
print(f"\nGeometry mix kept intact: {n_single:,} Polygon + {n_multi:,} MultiPolygon")
allowed = {'Polygon', 'MultiPolygon'}
non_poly = gdf[~gdf.geom_type.isin(allowed)]
assert len(non_poly) == 0, f"Non-(Multi)Polygon geometries remain: {non_poly.geom_type.value_counts().to_dict()}"
print("Geometry assertion passed: all features are Polygon or MultiPolygon.")

invalid = (~gdf.geometry.is_valid).sum()
print(f"Invalid geometries: {invalid}")
if invalid:
    # EPA polygons occasionally have self-intersections / topology issues that
    # cause shapely contains() to throw GEOSException. Repair via make_valid().
    print(f"  repairing {invalid} invalid geometries with make_valid()...")
    from shapely.validation import make_valid
    mask = ~gdf.geometry.is_valid
    gdf.loc[mask, 'geometry'] = gdf.loc[mask, 'geometry'].apply(make_valid)
    still_invalid = (~gdf.geometry.is_valid).sum()
    print(f"  invalid after repair: {still_invalid}")

gdf['ingested_at'] = pd.Timestamp.now(tz='UTC')

# ── Validation summary ───────────────────────────────────────────────────────

assert 30_000 <= gdf['pwsid'].nunique() <= 60_000, \
    f"Unexpected unique PWSID count: {gdf['pwsid'].nunique()}"

print(f"\nDistinct PWSIDs: {gdf['pwsid'].nunique():,}")
print(f"\nState coverage (top 15 by polygon count):")
state_counts = gdf['state_abbr'].value_counts().head(15)
print(state_counts.to_string())
print(f"\nTotal states with at least 1 polygon: {gdf['state_abbr'].nunique()}")

print(f"\nAvailability classification:")
print(gdf['availability'].value_counts(dropna=False).to_string())

print(f"\nVerification (Modeled vs Confirmed):")
print(gdf['verification'].value_counts(dropna=False).head(8).to_string())

print(f"\nService_area_type breakdown (top 8):")
print(gdf['service_area_type'].value_counts(dropna=False).head(8).to_string())

print(f"\nNull counts:")
print(gdf.drop(columns='geometry').isnull().sum().to_string())

print(f"\nPopulation served stats (excluding nulls):")
pop = gdf['pop_served'].dropna()
print(f"  count={len(pop):,}  median={int(pop.median())}  "
      f"mean={int(pop.mean())}  max={int(pop.max())}")

# ── Spot checks ──────────────────────────────────────────────────────────────

print("\nSpot checks: which water district contains each known location?")
print("-" * 95)
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore', message='Geometry is in a geographic CRS')

URBAN_INSIDE = {
    "New York City NY":  (-74.0060, 40.7128),
    "Los Angeles CA":   (-118.2437, 34.0522),
    "Chicago IL":        (-87.6298, 41.8781),
    "Houston TX":        (-95.3698, 29.7604),
    "Phoenix AZ":       (-112.0740, 33.4484),
    "Las Vegas NV":     (-115.1398, 36.1699),
    "Richmond VA":       (-77.4360, 37.5407),
    "Atlanta GA":        (-84.3880, 33.7490),
    "Denver CO":        (-104.9903, 39.7392),
    "Seattle WA":       (-122.3321, 47.6062),
    "San Francisco CA": (-122.4194, 37.7749),
    "Dallas TX":         (-96.7970, 32.7767),
    "Boston MA":         (-71.0589, 42.3601),
}
RURAL_OUTSIDE = {
    "Rural West Texas":     (-101.9000, 33.5000),
    "Rural Nevada desert":  (-116.5000, 38.5000),
    "Rural Kansas":          (-98.5000, 38.5000),
    "Rural Iowa":            (-93.6000, 42.0000),
    "Rural Wyoming":        (-107.0000, 43.0000),
}

inside_ok, inside_total = 0, len(URBAN_INSIDE)
for label, (lon, lat) in URBAN_INSIDE.items():
    pt = Point(lon, lat)
    hits = gdf[gdf.geometry.contains(pt)]
    if len(hits) > 0:
        inside_ok += 1
        # Pick the best by population if overlap
        best = hits.sort_values('pop_served', ascending=False, na_position='last').iloc[0]
        overlap = f' (+{len(hits)-1})' if len(hits) > 1 else ''
        print(f"  OK INSIDE   {label:<24} -> {str(best['district_name'])[:38]:<40} "
              f"pop={best['pop_served']}  {best['verification']}{overlap}")
    else:
        print(f"  !! OUTSIDE  {label:<24} -> NO district contains point")

for label, (lon, lat) in RURAL_OUTSIDE.items():
    pt = Point(lon, lat)
    hits = gdf[gdf.geometry.contains(pt)]
    if len(hits) == 0:
        print(f"  OK OUTSIDE  {label:<24} -> no district (expected)")
    else:
        best = hits.iloc[0]
        print(f"  ?? INSIDE   {label:<24} -> unexpectedly found {str(best['district_name'])[:38]}")

print(f"\n  Urban-inside hit rate: {inside_ok}/{inside_total} ({inside_ok/inside_total*100:.0f}%)")

# ── Output ───────────────────────────────────────────────────────────────────

gdf.to_parquet(OUT_PARQUET, index=False)
print(f"\nWrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e6:.1f} MB)")
sample = gdf.head(20).copy()
sample.drop(columns='geometry').to_csv(Path(__file__).parent / 'sample_rows.csv', index=False)
print("Exported: sample_rows.csv")
