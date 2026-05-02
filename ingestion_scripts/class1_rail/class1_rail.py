"""Class I railroad network ingestion.

Source: Bureau of Transportation Statistics (BTS) National Transportation Atlas
        Database -- "North American Rail Network Lines" hosted feature service.
        services.arcgis.com/xOi1kZaI0eWDREZv/.../NTAD_North_American_Rail_Network_Lines
        Owner: USDOT/BTS, sourced from FRA. Annual refresh per PRD section 10.2.
        Coverage: 302,734 rail line segments national (US + adjacent CA/MX),
                  filtered here to the seven Class I freight railroads.

What "Class I railroad" means:
  Federal Surface Transportation Board classification: 2024 threshold is
  ~$1B annual operating revenue. Seven networks qualify -- the only ones
  relevant for heavy-haul industrial / data-center site selection per PRD
  section 4.5 + scoring dimension 6 (Strategic):
    BNSF      BNSF Railway
    UP        Union Pacific
    CSX       CSX Transportation             (encoded as 'CSXT' upstream)
    NS        Norfolk Southern
    CN        Canadian National
    CPKC      Canadian Pacific Kansas City   (encoded as 'CPRS' upstream;
                                              KCS-merged-into-CP, 2023)
  Class II (Regional) and Class III (Short-line) are NOT included -- they
  exist for ~600 short lines in this dataset but the PRD scope is Class I
  only.

Encoding gotcha:
  BTS uses railroad operating codes, not the marketing names in the PRD.
  CSX is 'CSXT', CPKC is 'CPRS', and a residual 25 features are tagged
  'CPR' (pre-merger Canadian Pacific). KCS is fully merged -- 0 records
  remain under that code. Filter must use these source codes; output
  rebrands them to the PRD names (CSXT->CSX, CPRS/CPR->CPKC).

Joint ownership:
  ~2,143 segments have a Class I in RROWNER2 or RROWNER3 (joint trackage,
  e.g. BNSF/UP shared mainline through Texas). We INCLUDE these and tag
  them with the first Class I owner encountered, since for proximity
  scoring the question is "is Class I rail physically here?" not "who
  holds title."

Trackage rights (TRKRGHTS1-9) are EXCLUDED:
  These are operating-only agreements (~14k features where a Class I runs
  on track owned by a non-Class-I). The physical rail is the same line --
  including them would inflate counts. Owner is what matters for the
  PRD's rail_class field.

Output schema (matches rail_lines table; PostGIS auto-generates rail_id):
  source_id         TEXT          FRAARCID (BTS feature identifier)
  railroad          TEXT          BNSF / UP / CSX / NS / CN / CPKC
  rail_class        TEXT          always 'class_1' for this dataset
  track_type        TEXT          main / yard / siding / industrial / other
                                  (mapped from NET code)
  state_abbr        TEXT          STATEAB (segment's home state)
  is_stracnet       BOOLEAN       Strategic Rail Corridor Network flag
                                  (DoD-designated military-priority routes)
  n_tracks          INTEGER       parallel track count (1 = single-track,
                                  2+ = double/triple)
  miles             DOUBLE        segment length in miles (BTS computed)
  geometry          LineString (4326) -- exploded from MultiLineString.
                    PostGIS table column: GEOGRAPHY(LINESTRING, 4326).
                    Loader needs no further ST_Dump.
  ingested_at       TIMESTAMPTZ
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

BASE_URL = (
    'https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/'
    'NTAD_North_American_Rail_Network_Lines/FeatureServer/0/query'
)
PAGE_SIZE = 1000

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = Path(__file__).parent / 'raw_class1_rail.parquet'
OUT_PARQUET = Path(__file__).parent / 'class1_rail.parquet'

# BTS source-code -> PRD name. Order matters for the OR-filter pass.
CLASS_1_RAW_OWNERS = ['BNSF', 'UP', 'CSXT', 'NS', 'CN', 'CPRS', 'CPR', 'KCS']
RAILROAD_NORMALIZE = {
    'BNSF': 'BNSF',
    'UP':   'UP',
    'CSXT': 'CSX',
    'NS':   'NS',
    'CN':   'CN',
    'CPRS': 'CPKC',
    'CPR':  'CPKC',  # legacy, pre-2023 merger
    'KCS':  'CPKC',  # merged, 2023
}

# NET code -> track_type label (BTS internal classification)
NET_TO_TRACK_TYPE = {
    'M': 'main',         # main line
    'I': 'main',         # interconnector / interchange
    'A': 'main',         # active main, secondary spelling
    'Y': 'yard',
    'S': 'siding',
    'X': 'industrial',
    'O': 'other',
    'F': 'ferry',
    'R': 'abandoned',    # retired
    'T': 'tunnel',
}

OUTFIELDS = ','.join([
    'FRAARCID', 'RROWNER1', 'RROWNER2', 'RROWNER3',
    'NET', 'STATEAB', 'STRACNET', 'TRACKS', 'MILES',
])

# WHERE clause: any of the three owner slots is a Class I source code.
_owner_list = ','.join(f"'{o}'" for o in CLASS_1_RAW_OWNERS)
WHERE_CLAUSE = (
    f"RROWNER1 IN ({_owner_list}) OR "
    f"RROWNER2 IN ({_owner_list}) OR "
    f"RROWNER3 IN ({_owner_list})"
)

RETRY_DELAYS = [5, 10, 20, 30, 60, 120, 180]
NETWORK_ERRORS = (ConnectionError, Timeout, ChunkedEncodingError, ProtocolError,
                  requests.exceptions.JSONDecodeError)
RETRY_HTTP_STATUSES = {429, 500, 502, 503, 504}


def _fetch_page(offset):
    params = {
        'where': WHERE_CLAUSE,
        'outFields': OUTFIELDS,
        'resultOffset': offset,
        'resultRecordCount': PAGE_SIZE,
        'orderByFields': 'FRAARCID',  # stable pagination
        'f': 'geojson',
    }
    last_exc = None
    for delay in RETRY_DELAYS:
        try:
            r = requests.get(BASE_URL, params=params, timeout=180)
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


# ── Fetch ────────────────────────────────────────────────────────────────────

if CACHE_FILE.exists():
    print(f"Loading cache: {CACHE_FILE.name}")
    gdf = gpd.read_parquet(CACHE_FILE)
else:
    print("Fetching Class I rail segments from BTS NTAD...")
    print(f"  filter: {WHERE_CLAUSE}")
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
gdf = gdf[~gdf.geometry.is_empty]
print(f"After geometry filter: {len(gdf):,} (dropped {n_before_geom - len(gdf)})")

if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)

# Resolve the Class I owner per row. Prefer RROWNER1; fall back to 2 then 3.
def _pick_railroad(row):
    for col in ('RROWNER1', 'RROWNER2', 'RROWNER3'):
        v = row.get(col)
        if v in RAILROAD_NORMALIZE:
            return RAILROAD_NORMALIZE[v]
    return None

gdf['railroad'] = gdf.apply(_pick_railroad, axis=1)
n_before_owner = len(gdf)
gdf = gdf[gdf['railroad'].notna()]
print(f"After Class I owner filter: {len(gdf):,} "
      f"(dropped {n_before_owner - len(gdf)} server-side false positives)")

gdf['source_id']    = gdf['FRAARCID'].astype('Int64').astype(str)
gdf['rail_class']   = 'class_1'
gdf['track_type']   = gdf['NET'].map(NET_TO_TRACK_TYPE).fillna('other')
gdf['state_abbr']   = gdf['STATEAB']
gdf['is_stracnet']  = gdf['STRACNET'].isin(['C', 'S'])
# BTS uses 0 to mean "unknown track count" — convert to NULL.
n_tracks_raw = pd.to_numeric(gdf['TRACKS'], errors='coerce').astype('Int64')
gdf['n_tracks'] = n_tracks_raw.where(n_tracks_raw > 0, other=pd.NA)
gdf['miles']        = pd.to_numeric(gdf['MILES'], errors='coerce')

# Filter to US-only segments. Source covers North America (CN/CPKC include
# Canadian segments; BNSF/CPKC include a handful of Mexican segments).
US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC','PR','VI','GU','AS','MP',
}
n_before_us = len(gdf)
gdf = gdf[gdf['state_abbr'].isin(US_STATES)]
print(f"After US-only filter: {len(gdf):,} (dropped {n_before_us - len(gdf):,} non-US segments)")

# Drop abandoned and ferry segments — not active infrastructure for site scoring.
n_before_active = len(gdf)
gdf = gdf[~gdf['track_type'].isin(['abandoned', 'ferry'])]
print(f"After active-only filter: {len(gdf):,} (dropped {n_before_active - len(gdf):,} abandoned/ferry)")

# Explode any MultiLineString into individual LineStrings so the parquet
# matches the rail_lines table (GEOGRAPHY(LINESTRING, 4326)). BTS data is
# almost entirely single LineStrings already; this is defensive.
n_multi_before = (gdf.geom_type == 'MultiLineString').sum()
if n_multi_before:
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    print(f"Exploded {n_multi_before:,} MultiLineStrings to LineStrings; "
          f"now {len(gdf):,} rows")

allowed = {'LineString'}
non_line = gdf[~gdf.geom_type.isin(allowed)]
assert len(non_line) == 0, \
    f"Non-LineString geometries remain: {non_line.geom_type.value_counts().to_dict()}"
print("Geometry assertion passed: all features are LineString.")

invalid = (~gdf.geometry.is_valid).sum()
print(f"Invalid geometries: {invalid}")
if invalid:
    from shapely.validation import make_valid
    mask = ~gdf.geometry.is_valid
    gdf.loc[mask, 'geometry'] = gdf.loc[mask, 'geometry'].apply(make_valid)
    print(f"  invalid after repair: {(~gdf.geometry.is_valid).sum()}")

keep_cols = ['source_id', 'railroad', 'rail_class', 'track_type', 'state_abbr',
             'is_stracnet', 'n_tracks', 'miles', 'geometry']
gdf = gdf[keep_cols].copy()
gdf['ingested_at'] = pd.Timestamp.utcnow()

# ── Validation ───────────────────────────────────────────────────────────────

assert len(gdf) > 50_000, f"Suspiciously low feature count: {len(gdf):,}"
assert gdf['railroad'].nunique() >= 5, \
    f"Missing expected Class I railroads: {gdf['railroad'].unique().tolist()}"
for rr in ('BNSF', 'UP', 'CSX', 'NS'):  # the four largest -- must all appear
    assert (gdf['railroad'] == rr).any(), f"Expected railroad '{rr}' not found"

print(f"\nTotal Class I rail segments: {len(gdf):,}")
print(f"\nBy railroad operator:")
print(gdf['railroad'].value_counts().to_string())
print(f"\nBy track type:")
print(gdf['track_type'].value_counts(dropna=False).to_string())
print(f"\nState coverage (top 15):")
print(gdf['state_abbr'].value_counts(dropna=False).head(15).to_string())
print(f"Total states+territories with Class I rail: {gdf['state_abbr'].nunique()}")

# Route miles via Albers for an honest length calc.
proj = gdf.to_crs(epsg=5070)
total_km = proj.geometry.length.sum() / 1000
total_mi = total_km * 0.621371
print(f"\nTotal rail length: ~{total_km:,.0f} km (~{total_mi:,.0f} miles)")
print(f"  (US Class I network is ~140k route miles per AAR; this dataset is "
      f"track-miles incl. parallel tracks, so a higher figure is expected)")

print(f"\nNull counts:")
print(gdf.drop(columns='geometry').isnull().sum().to_string())

# ── Spot checks ──────────────────────────────────────────────────────────────

print("\nSpot checks: distance to nearest Class I rail")
print("-" * 78)
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore', message='Geometry is in a geographic CRS')

# Use Albers-projected geometry for accurate metric distance.
proj_lines = gdf.set_geometry(proj.geometry).set_crs(epsg=5070)
sindex = proj_lines.sindex

SPOT_CHECKS = {
    "Chicago IL (major rail hub)":      (-87.6298, 41.8781),
    "Kansas City MO (BNSF/UP hub)":     (-94.5786, 39.0997),
    "Memphis TN (CN/CPKC hub)":         (-90.0490, 35.1495),
    "Amarillo TX (BNSF corridor)":     (-101.8313, 35.2220),
    "Bakersfield CA (UP corridor)":    (-119.0187, 35.3733),
    "Atlanta GA (NS hub)":              (-84.3880, 33.7490),
    "Jacksonville FL (CSX HQ)":         (-81.6557, 30.3322),
    "Rural Wyoming (UP mainline)":     (-107.0000, 43.0000),
    # Should be far from Class I rail
    "Rural Nevada desert":             (-116.5000, 38.5000),
    "Florida Keys":                    ( -81.0000, 24.6000),
    "Big Bend ranch TX":               (-103.8000, 29.5000),
}

# Project test points to Albers as well.
test_pts = gpd.GeoSeries(
    [Point(lon, lat) for (lon, lat) in SPOT_CHECKS.values()],
    crs='EPSG:4326',
).to_crs(epsg=5070)

for (label, _), pt in zip(SPOT_CHECKS.items(), test_pts):
    # Search nearest using a small expanding bbox query for speed.
    for radius_m in (5_000, 25_000, 100_000, 500_000, 2_000_000):
        bbox = (pt.x - radius_m, pt.y - radius_m, pt.x + radius_m, pt.y + radius_m)
        cands = list(sindex.intersection(bbox))
        if cands:
            break
    if not cands:
        print(f"  {label:<35} no rail within 2,000 km (??)")
        continue
    sub = proj_lines.iloc[cands]
    distances = sub.geometry.distance(pt)
    nearest = sub.iloc[distances.values.argmin()]
    nd_m = float(distances.min())
    nd_mi = nd_m / 1609.344
    print(f"  {label:<35} -> {nearest['railroad']:<5} "
          f"{nd_mi:6.2f} mi  ({nearest['track_type']:<10}  "
          f"{nearest['state_abbr']})")

# ── Output ───────────────────────────────────────────────────────────────────

gdf.to_parquet(OUT_PARQUET, index=False)
print(f"\nWrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e6:.1f} MB)")
sample = gdf.head(20).copy()
sample.drop(columns='geometry').to_csv(
    Path(__file__).parent / 'sample_rows.csv', index=False)
print("Exported: sample_rows.csv")
