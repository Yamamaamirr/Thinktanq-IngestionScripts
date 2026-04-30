"""EIA / HIFLD Electric Retail Service Territories ingestion.

Fetches all US electric retail service territories from the HIFLD-hosted
ArcGIS REST FeatureServer and produces a parquet ready for PostGIS load
into the utility_territories table.

The endpoint your PRD brief lists (Hp6G80Pky0om7QvQ/Electric_Retail_...)
returns "Invalid URL". The active hosted dataset is at services5/HDRa0B57OVrv2E1q
where it shows up as "U.S. Electric Retail Service Territories" (NREL
DistributedWind upload, derived from the canonical EIA/HIFLD source). Note
that this dataset includes a handful of Canadian utilities — we filter to
COUNTRY='USA' to match the PRD scope.

Source schema (uppercase HIFLD-style fields):
  ID            - EIA utility ID  -> source_id
  NAME          - utility name     -> utility_name
  TYPE          - utility type     -> utility_type (INVESTOR OWNED / COOPERATIVE /
                                       MUNICIPAL / FEDERAL / POLITICAL SUBDIVISION /
                                       NOT AVAILABLE)
  STATE         - 2-letter state   -> state_abbr
  CNTRL_AREA    - ISO/RTO control area, e.g. MISO/PJM. Hugely useful for
                  cross-referencing with iso_queues table -- kept as
                  control_area.
  HOLDING_CO    - parent company    -> holding_company (a single utility
                                       holding company often owns multiple
                                       service-territory subsidiaries)
  WEBSITE       - utility website   -> website (audit trail / analyst follow-up)
  geometry      - polygon

Output schema:
  source_id        TEXT (NOT NULL)
  utility_name     TEXT (NOT NULL)
  utility_type     TEXT
  state_abbr       TEXT
  control_area     TEXT
  holding_company  TEXT
  website          TEXT
  geometry         Polygon, EPSG:4326
  ingested_at      TIMESTAMPTZ

How this feeds enrichment (IMPORTANT - read before writing the join query):

  Territories overlap heavily. A typical urban parcel has 3-10 polygons
  containing it: Federal wholesale marketers (WAPA, BPA, TVA), state/
  political-subdivision utilities, the actual investor-owned retailer,
  city munis, plus rural cooperatives whose polygons extend into metro
  areas, plus noise polygons from MultiPolygon explode.

  Validation across 39 metros showed neither "smallest area wins" nor
  "investor-owned first" alone is correct. The right rule is CUSTOMER COUNT:
  the polygon with the most active retail customers wins, after demoting
  Federal/Wholesale wholesalers.

  Production join SQL:

  WITH ranked AS (
    SELECT
      utility_name, source_id,
      CASE
        WHEN utility_type IN ('FEDERAL', 'WHOLESALE POWER MARKETER') THEN 99
        WHEN utility_type IN ('MUNICIPAL', 'POLITICAL SUBDIVISION')
             AND COALESCE(customers, 0) >= 30000               THEN 1
        ELSE 2
      END AS retail_tier,
      COALESCE(customers, 0) AS customers_for_sort,
      ST_Area(geometry::geometry) AS area
    FROM utility_territories
    WHERE ST_Within(ST_Point($lon, $lat)::geography::geometry, geometry::geometry)
  )
  SELECT utility_name, source_id FROM ranked
  ORDER BY retail_tier ASC,
           customers_for_sort DESC,    -- pick the utility with the most customers
           area ASC                    -- tiebreak: smallest area = most specific
  LIMIT 1;

  Why customers DESC works: in Austin, Austin Energy = 488k customers
  while overlapping rural cooperatives have <100k each. In Charlotte, Duke
  = 2.6M while overlapping town munis have <6k each. The actual retail
  provider almost always has 5x-100x more customers than overlapping noise.

  Caveat: the HIFLD `CUSTOMERS` field is sparse for some Texas utilities
  (Oncor, CenterPoint show -999999). Those rows fall to the area tiebreak.
  In practice this means downtown Houston / Dallas may resolve to a
  cooperative or muni — flag for analyst override (see OPEN_QUESTIONS).

Same paginated-fetch pattern as HIFLD substations and transmission lines.
"""
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError

BASE_URL = (
    'https://services5.arcgis.com/HDRa0B57OVrv2E1q/arcgis/rest/services/'
    'Electric_Retail_Service_Territories/FeatureServer/0/query'
)
PAGE_SIZE = 2000

CACHE_FILE = Path(__file__).parent / 'raw_utility_territories.parquet'
OUT_PARQUET = Path(__file__).parent / 'utility_territories.parquet'

SENTINEL = -999999

RETRY_DELAYS = [5, 10, 20, 30, 60]
NETWORK_ERRORS = (ConnectionError, Timeout, ChunkedEncodingError, ProtocolError)


def _fetch_page(offset):
    params = {
        'where': "COUNTRY='USA'",
        # CUSTOMERS and RETAIL_MWH are essential for the join tiebreaker --
        # see docstring "ranked-priority join" section.
        'outFields': 'ID,NAME,TYPE,STATE,CNTRL_AREA,HOLDING_CO,WEBSITE,CUSTOMERS,RETAIL_MWH',
        'resultOffset': offset,
        'resultRecordCount': PAGE_SIZE,
        'f': 'geojson',
    }
    last_exc = None
    for delay in RETRY_DELAYS:
        try:
            r = requests.get(BASE_URL, params=params, timeout=120)
            r.raise_for_status()
            return r.json().get('features', [])
        except NETWORK_ERRORS as e:
            last_exc = e
            print(f"  Network error: {type(e).__name__} -- retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"Max retries exceeded ({last_exc!r})")


# ── Fetch ────────────────────────────────────────────────────────────────────

if CACHE_FILE.exists():
    print(f"Loading cache: {CACHE_FILE.name}")
    gdf = gpd.read_parquet(CACHE_FILE)
else:
    print("Fetching from EIA/HIFLD ArcGIS service...")
    all_features = []
    offset = 0
    while True:
        feats = _fetch_page(offset)
        if not feats:
            break
        all_features.extend(feats)
        offset += len(feats)
        print(f"  fetched {offset:,} features...")
        if len(feats) < PAGE_SIZE:
            break
    gdf = gpd.GeoDataFrame.from_features(all_features, crs='EPSG:4326')
    gdf.to_parquet(CACHE_FILE)
    print(f"  cached {len(gdf):,} rows -> {CACHE_FILE.name}")

print(f"\nRaw rows: {len(gdf):,}")
print(f"Columns: {list(gdf.columns)}")

# ── Process ──────────────────────────────────────────────────────────────────

# Drop rows without geometry.
n_before_geom = len(gdf)
gdf = gdf.dropna(subset='geometry')
print(f"After geometry filter: {len(gdf):,} (dropped {n_before_geom - len(gdf)})")

if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)

# Clean sentinel placeholders.
text_cols = ['ID', 'NAME', 'TYPE', 'STATE', 'CNTRL_AREA', 'HOLDING_CO', 'WEBSITE']
for col in text_cols:
    if col in gdf.columns:
        gdf[col] = gdf[col].replace('NOT AVAILABLE', np.nan)
# Numeric sentinel = -999999 -> NaN so downstream sort treats them as missing
for col in ['CUSTOMERS', 'RETAIL_MWH']:
    if col in gdf.columns:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
        gdf.loc[gdf[col] <= 0, col] = np.nan

# Rename to PRD-canonical names.
gdf = gdf.rename(columns={
    'ID':         'source_id',
    'NAME':       'utility_name',
    'TYPE':       'utility_type',
    'STATE':      'state_abbr',
    'CNTRL_AREA': 'control_area',
    'HOLDING_CO': 'holding_company',
    'WEBSITE':    'website',
    'CUSTOMERS':  'customers',
    'RETAIL_MWH': 'retail_mwh',
})

keep_cols = ['source_id', 'utility_name', 'utility_type', 'state_abbr',
             'control_area', 'holding_company', 'website',
             'customers', 'retail_mwh', 'geometry']
gdf = gdf[keep_cols].copy()

# Drop rows with no utility_name -- can't be useful downstream.
n_before_name = len(gdf)
gdf = gdf[gdf['utility_name'].notna()]
print(f"After utility_name filter: {len(gdf):,} (dropped {n_before_name - len(gdf)})")

# Explode MultiPolygon -> Polygon (PRD schema requires GEOGRAPHY(POLYGON, 4326)).
n_before = len(gdf)
n_multi = (gdf.geom_type == 'MultiPolygon').sum()
print(f"\nMultiPolygon features before explode: {n_multi}")
gdf = gdf.explode(index_parts=False).reset_index(drop=True)
print(f"After explode: {len(gdf):,} rows (+{len(gdf) - n_before} component polygons)")
non_poly = gdf[gdf.geom_type != 'Polygon']
assert len(non_poly) == 0, f"Non-Polygon geometries remain: {non_poly.geom_type.value_counts().to_dict()}"
print("Geometry assertion passed: all features are Polygon.")

invalid = (~gdf.geometry.is_valid).sum()
print(f"Invalid geometries: {invalid}")

gdf['ingested_at'] = pd.Timestamp.utcnow()

# ── Validation summary ───────────────────────────────────────────────────────

assert 2500 <= gdf['source_id'].nunique() <= 4000, \
    f"Unexpected unique territory count: {gdf['source_id'].nunique()}"

print(f"\nDistinct utilities (source_id): {gdf['source_id'].nunique():,}")
print(f"\nutility_type breakdown:")
print(gdf['utility_type'].value_counts(dropna=False).head(10).to_string())
print(f"\ncontrol_area breakdown (top 10):")
print(gdf['control_area'].value_counts(dropna=False).head(10).to_string())
print(f"\nstate_abbr breakdown (top 10):")
print(gdf['state_abbr'].value_counts(dropna=False).head(10).to_string())

print(f"\nNull counts:")
print(gdf.drop(columns='geometry').isnull().sum().to_string())

# ── Spot checks ──────────────────────────────────────────────────────────────

print("\nSpot-check: utility lookup using the production join rule above")
print("-" * 95)
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore', message='Geometry is in a geographic CRS')

# Replicates the production SQL ordering exactly.
# Tier 1: MUNICIPAL / POLITICAL SUBDIVISION with >=30k customers — real city utilities
#         with legally exclusive territories. The 30k floor filters out water/irrigation
#         districts and tribal utilities that share the HIFLD type codes but have
#         HIFLD polygon geometries that bleed far outside their actual service areas.
# Tier 2: INVESTOR OWNED, COOPERATIVE, STATE, and small munis (<30k customers).
# Tier 99: FEDERAL / WHOLESALE — demoted, never the retail provider.
MUNI_FLOOR = 30_000

gdf_metric = gdf.to_crs(epsg=3857)

def _tier(utype, customers):
    if utype in ('FEDERAL', 'WHOLESALE POWER MARKETER'):
        return 99
    if utype in ('MUNICIPAL', 'POLITICAL SUBDIVISION') and (customers or 0) >= MUNI_FLOOR:
        return 1
    return 2

def lookup_utility(lon, lat):
    pt = Point(lon, lat)
    hits = gdf[gdf.geometry.contains(pt)].copy()
    if len(hits) == 0:
        return None, None, 0
    hits['_tier'] = hits.apply(lambda r: _tier(r['utility_type'], r['customers']), axis=1)
    hits['_cust'] = hits['customers'].fillna(0)
    hits['_area'] = gdf_metric.loc[hits.index].geometry.area
    best = hits.sort_values(['_tier', '_cust', '_area'],
                            ascending=[True, False, True]).iloc[0]
    return best['utility_name'], best['utility_type'], len(hits)

spot_checks = [
    ("San Francisco CA (PG&E)",            -122.4194, 37.7749),
    ("Sacramento CA (SMUD)",               -121.4944, 38.5816),
    ("LA downtown (LADWP)",                -118.2437, 34.0522),
    ("Las Vegas NV (NV Energy)",           -115.1398, 36.1699),
    ("Phoenix AZ (APS)",                   -112.0740, 33.4484),
    ("Richmond VA (Dominion)",              -77.4360, 37.5407),
    ("Austin TX (Austin Energy)",           -97.7431, 30.2672),
    ("San Antonio TX (CPS Energy)",         -98.4936, 29.4241),
    ("Memphis TN (MLGW)",                   -90.0490, 35.1495),
    ("Charlotte NC (Duke Energy)",          -80.8431, 35.2271),
    ("Seattle WA (City Light)",            -122.3321, 47.6062),
    ("NYC Manhattan (Con Ed)",              -73.9857, 40.7484),
    ("Dallas TX (Oncor -- known weak)",     -96.7970, 32.7767),
    ("Houston TX (CenterPoint -- weak)",    -95.3698, 29.7604),
]
for label, lon, lat in spot_checks:
    name, utype, count = lookup_utility(lon, lat)
    n = (str(name) if name else 'NONE')[:40]
    overlap = f" (+{count-1})" if count > 1 else ""
    print(f"  {label:<40} -> {n:<42} {utype}{overlap}")

# ── Output ───────────────────────────────────────────────────────────────────

gdf.to_parquet(OUT_PARQUET, index=False)
print(f"\nWrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e6:.1f} MB)")
sample = gdf.head(20).copy()
sample.drop(columns='geometry').to_csv('sample_rows.csv', index=False)
print("Exported: sample_rows.csv")
