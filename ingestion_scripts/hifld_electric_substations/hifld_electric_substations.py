"""Electric substation ingestion — layered OSM / HIFLD approach.

Sources in priority order:
  1. OpenStreetMap Overpass API — power=substation, voltage >= 69 kV; all 50 states;
     near-realtime, community-maintained, more current than HIFLD by 4+ years.
  2. HIFLD (national fallback) — covers substations OSM hasn't mapped yet, and
     fills the California gap where HIFLD has slightly better coverage than OSM.

Spatial deduplication (DEDUP_RADIUS_M = 500 m, EPSG:5070 Conus Albers):
  Before adding HIFLD records, drop any HIFLD point within 500 m of an OSM point.
  Same physical substation in both sources → keep the OSM version (more current).

State assignment for OSM:
  state_abbr is set from the state ISO code used in each Overpass query — no spatial
  join needed. OSM addr:state tag is present on < 2% of US substations.

Voltage → class_rating (PRD class_weight_mapping):
  >= 345 kV  →  class_1    regional backbone (345/500/765 kV)
  230–344 kV →  class_2    sub-regional / metro feeds
  69–229 kV  →  class_3    lower transmission / sub-transmission
  < 69 kV    →  sub_class  distribution
  unknown    →  NULL       scoring engine falls back to tx-line proximity

Output schema (PostGIS GEOMETRY(POINT, 4326)):
  name              TEXT
  state_abbr        TEXT
  substation_type   TEXT        'transmission'/'distribution'/'tap'/'unknown'
  max_volt          NUMERIC     kV, nullable
  min_volt          NUMERIC     kV, nullable (HIFLD records only)
  max_volt_inferred BOOLEAN     True if HIFLD inferred voltage (HIFLD only)
  lines             INTEGER     connected tx lines (HIFLD only)
  class_rating      TEXT        class_1/class_2/class_3/sub_class/NULL
  volt_source       TEXT        'osm'/'reported'/'hifld_inferred'/'tap'/NULL
  data_source       TEXT        'osm'/'hifld'
  geometry          POINT(4326)
  ingested_at       TIMESTAMPTZ
"""
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout

HERE         = Path(__file__).parent
OSM_CACHE   = HERE / 'cache_osm'       # one .parquet per state inside this dir
HIFLD_CACHE = HERE / 'raw_substations.parquet'  # existing cache from previous runs
OUT_PARQUET = HERE / 'hifld_electric_substations.parquet'

DEDUP_RADIUS_M = 500
OVERPASS_URL   = 'https://overpass-api.de/api/interpreter'
OVERPASS_UA    = 'ThinqTank-SubstationIngestion/1.0'
HIFLD_URL      = (
    'https://services6.arcgis.com/OO2s4OoyCZkYJ6oE/arcgis/rest/services/'
    'Substations/FeatureServer/0/query'
)
HIFLD_SENTINEL = -999999.0

US_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_voltage_kv(raw) -> float | None:
    """Parse OSM voltage tag to kV float.

    Handles: '138000' (volts), '69000;138000' (multi-voltage), '138 kV', '138'.
    Returns the maximum when semicolon-joined.
    """
    if not raw or pd.isna(raw):
        return None
    values = []
    for part in str(raw).split(';'):
        cleaned = part.lower().replace('kv', '').replace('k', '').replace(' ', '')
        try:
            v = float(cleaned)
            if v >= 1000:
                v /= 1000
            values.append(v)
        except ValueError:
            continue
    return max(values) if values else None


def _volt_to_class(kv) -> str | None:
    if kv is None or pd.isna(kv):
        return None
    if kv >= 345: return 'class_1'
    if kv >= 230: return 'class_2'
    if kv >= 69:  return 'class_3'
    return 'sub_class'


def _dedup(high: gpd.GeoDataFrame, low: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return rows of `low` not within DEDUP_RADIUS_M of any row in `high`."""
    if high.empty or low.empty:
        return low
    high_proj = high.to_crs(epsg=5070)
    low_proj  = low.to_crs(epsg=5070)
    high_buf  = high_proj.copy()
    high_buf['geometry'] = high_buf.geometry.buffer(DEDUP_RADIUS_M)
    joined = gpd.sjoin(low_proj, high_buf[['geometry']], how='left', predicate='within')
    unique = joined[~joined.index.duplicated(keep='first')]
    return low.loc[unique[unique['index_right'].isna()].index].copy()


def _overpass_post(query: str, timeout_s: int = 95) -> list:
    """POST to Overpass with retry. Returns list of elements."""
    data = {'data': query}
    for attempt, delay in enumerate([0, 15, 30, 60]):
        if delay:
            print(f"    retry in {delay}s...")
            time.sleep(delay)
        try:
            r = requests.post(
                OVERPASS_URL,
                data=data,
                headers={'User-Agent': OVERPASS_UA},
                timeout=timeout_s,
            )
            r.raise_for_status()
            return r.json().get('elements', [])
        except (ConnectionError, Timeout, ChunkedEncodingError) as e:
            print(f"    network error: {type(e).__name__}")
        except requests.HTTPError as e:
            print(f"    HTTP {e.response.status_code}")
    raise RuntimeError(f"Overpass: max retries exceeded for query")


# ── Layer 1: OpenStreetMap (all 50 states, cached per state) ─────────────────

def _fetch_osm_state(state: str) -> gpd.GeoDataFrame:
    """Fetch OSM substations for one state. Returns GDF with >=69kV records."""
    cache_path = OSM_CACHE / f'{state}.parquet'
    if cache_path.exists():
        return gpd.read_parquet(cache_path)

    query = (
        f'[out:json][timeout:90];'
        f'area["ISO3166-2"="US-{state}"]->.s;'
        f'('
        f'  node["power"="substation"]["voltage"](area.s);'
        f'  way["power"="substation"]["voltage"](area.s);'
        f'  relation["power"="substation"]["voltage"](area.s);'
        f');'
        f'out center tags;'
    )
    elements = _overpass_post(query)

    rows = []
    for el in elements:
        tags = el.get('tags', {})
        kv = _parse_voltage_kv(tags.get('voltage'))
        if kv is None or kv < 69:
            continue
        if el['type'] == 'node':
            lat, lon = el['lat'], el['lon']
        elif 'center' in el:
            lat, lon = el['center']['lat'], el['center']['lon']
        else:
            continue
        rows.append({
            'name':            tags.get('name') or tags.get('ref') or '',
            'state_abbr':      state,
            'substation_type': tags.get('substation', 'unknown'),
            'max_volt':        kv,
            'min_volt':        np.nan,
            'max_volt_inferred': pd.NA,
            'lines':           np.nan,
            'volt_source':     'osm',
            'lat': lat, 'lon': lon,
        })

    if rows:
        df = pd.DataFrame(rows)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df['lon'], df['lat']),
            crs='EPSG:4326',
        ).drop(columns=['lat', 'lon'])
    else:
        gdf = gpd.GeoDataFrame(
            columns=['name', 'state_abbr', 'substation_type', 'max_volt',
                     'min_volt', 'max_volt_inferred', 'lines', 'volt_source', 'geometry'],
            crs='EPSG:4326',
        )

    gdf.to_parquet(cache_path)
    return gdf


def _fetch_osm_all() -> gpd.GeoDataFrame:
    OSM_CACHE.mkdir(exist_ok=True)
    cached = [s for s in US_STATES if (OSM_CACHE / f'{s}.parquet').exists()]
    needed = [s for s in US_STATES if s not in cached]

    if cached:
        print(f"  OSM: {len(cached)} states already cached, fetching {len(needed)} remaining")
    else:
        print(f"  OSM: fetching all {len(US_STATES)} states from Overpass API")

    gdfs = []
    # Load cached states
    for s in cached:
        gdfs.append(gpd.read_parquet(OSM_CACHE / f'{s}.parquet'))

    # Fetch uncached states
    for i, s in enumerate(needed):
        print(f"  OSM [{i+1}/{len(needed)}] {s}...", end=' ', flush=True)
        gdf_s = _fetch_osm_state(s)
        print(f"{len(gdf_s):,} substations")
        gdfs.append(gdf_s)
        if i < len(needed) - 1:
            time.sleep(1)  # be polite to Overpass

    result = gpd.GeoDataFrame(
        pd.concat([g for g in gdfs if not g.empty], ignore_index=True),
        crs='EPSG:4326',
    )
    result['data_source'] = 'osm'
    return result


# ── Layer 2: HIFLD (national fallback) ───────────────────────────────────────

def _fetch_hifld() -> gpd.GeoDataFrame:
    if HIFLD_CACHE.exists():
        print("  HIFLD: loading from cache (raw_substations.parquet)...")
        raw = gpd.read_parquet(HIFLD_CACHE)
    else:
        print("  HIFLD: fetching from ArcGIS REST (no cache found)...")
        all_features, offset = [], 0
        while True:
            r = requests.get(
                HIFLD_URL,
                params={'where': '1=1', 'outFields': '*',
                        'resultOffset': offset, 'resultRecordCount': 2000, 'f': 'geojson'},
                timeout=30,
            )
            r.raise_for_status()
            feats = r.json().get('features', [])
            if not feats:
                break
            all_features.extend(feats)
            offset += len(feats)
            print(f"    HIFLD fetched {offset:,}...")
            if len(feats) < 2000:
                break
        raw = gpd.GeoDataFrame.from_features(all_features, crs='EPSG:4326')
        raw.to_parquet(HIFLD_CACHE)

    print(f"  HIFLD: {len(raw):,} total rows in cache")

    for col in ('MAX_VOLT', 'MIN_VOLT', 'LINES'):
        if col in raw.columns:
            raw[col] = raw[col].replace(HIFLD_SENTINEL, np.nan)

    gdf = raw[raw['STATUS'] == 'IN SERVICE'].dropna(subset='geometry').copy()
    gdf['SOURCEDATE'] = pd.to_datetime(gdf.get('SOURCEDATE'), unit='ms', errors='coerce')
    gdf['VAL_DATE']   = pd.to_datetime(gdf.get('VAL_DATE'),   unit='ms', errors='coerce')

    def _volt_source(row):
        if row.get('TYPE') == 'TAP': return 'tap'
        if pd.notna(row.get('MAX_VOLT')):
            return 'hifld_inferred' if row.get('MAX_INFER') == 'Y' else 'reported'
        return None

    result = gpd.GeoDataFrame({
        'name':             gdf.get('NAME', pd.Series(dtype=str)).fillna(''),
        'state_abbr':       gdf.get('STATE', pd.Series(dtype=str)).fillna(''),
        'substation_type':  gdf.get('TYPE',  pd.Series(dtype=str)).fillna('unknown'),
        'max_volt':         gdf.get('MAX_VOLT'),
        'min_volt':         gdf.get('MIN_VOLT'),
        'max_volt_inferred': (gdf.get('MAX_INFER') == 'Y'),
        'lines':            gdf.get('LINES'),
        'volt_source':      gdf.apply(_volt_source, axis=1),
        'data_source':      'hifld',
        'geometry':         gdf['geometry'],
    }, crs='EPSG:4326')

    print(f"  HIFLD: {len(result):,} IN SERVICE substations")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

print("=== Electric Substation Ingestion (layered OSM / HIFLD) ===\n")

# ── Layer 1: OSM ──────────────────────────────────────────────────────────────
print("Layer 1: OpenStreetMap (all 50 states)")
gdf_osm = _fetch_osm_all()
print(f"  Total OSM: {len(gdf_osm):,} substations >= 69 kV\n")

# ── Layer 2: HIFLD (national fallback) ───────────────────────────────────────
print("Layer 2: HIFLD (national fallback)")
gdf_hifld = _fetch_hifld()
gdf_hifld_new = _dedup(gdf_osm, gdf_hifld)
n_hifld_drop = len(gdf_hifld) - len(gdf_hifld_new)
print(f"  {len(gdf_hifld):,} HIFLD total -> {len(gdf_hifld_new):,} after dedup "
      f"({n_hifld_drop} within {DEDUP_RADIUS_M}m of OSM)\n")

# ── Merge ─────────────────────────────────────────────────────────────────────
SCHEMA = ['name', 'state_abbr', 'substation_type', 'max_volt', 'min_volt',
          'max_volt_inferred', 'lines', 'volt_source', 'data_source', 'geometry']

all_layers = [gdf_osm, gdf_hifld_new]
aligned = []
for g in all_layers:
    if g.empty:
        continue
    g = g.copy()
    for col in SCHEMA:
        if col not in g.columns:
            g[col] = np.nan
    aligned.append(g[SCHEMA])

gdf = gpd.GeoDataFrame(pd.concat(aligned, ignore_index=True), crs='EPSG:4326')

# Cast max_volt_inferred to nullable boolean — OSM rows have pd.NA, HIFLD rows
# have True/False; concat produces object dtype without this explicit cast.
gdf['max_volt_inferred'] = gdf['max_volt_inferred'].astype('boolean')

# ── Class rating ──────────────────────────────────────────────────────────────
gdf['class_rating'] = gdf.apply(
    lambda r: 'class_3' if r['substation_type'] == 'TAP'
              else _volt_to_class(r['max_volt']),
    axis=1,
)

gdf['ingested_at'] = pd.Timestamp.now(tz='UTC')

# ── Validation ────────────────────────────────────────────────────────────────
assert len(gdf) > 50_000, f"Suspiciously low count: {len(gdf):,}"
assert gdf.geom_type.eq('Point').all(), "Non-Point geometry found"
assert gdf.geometry.is_valid.all(), "Invalid geometries"
valid_classes = {'class_1', 'class_2', 'class_3', 'sub_class'}
assert set(gdf['class_rating'].dropna().unique()) <= valid_classes

# ── Summary ───────────────────────────────────────────────────────────────────
hifld_baseline = len(gdf_hifld)  # IN SERVICE count before dedup

print("=== Final dataset ===")
print(f"Total substations:  {len(gdf):,}")
print(f"HIFLD baseline was: {hifld_baseline:,}  (delta: {len(gdf) - hifld_baseline:+,})")

print(f"\nData source breakdown:")
print(gdf['data_source'].value_counts().to_string())

print(f"\nClass rating breakdown:")
print(gdf['class_rating'].value_counts(dropna=False).to_string())

null_cr = gdf['class_rating'].isna().sum()
print(f"\nNULL class_rating: {null_cr:,} ({100*null_cr/len(gdf):.1f}%)")

print(f"\nTarget state breakdown:")
for state in ['CA', 'TX', 'AZ', 'NV', 'VA']:
    st = gdf[gdf['state_abbr'] == state]
    cr = st['class_rating'].value_counts(dropna=False)
    print(f"  {state}: {len(st):,} | "
          f"class_1={cr.get('class_1', 0)}  class_2={cr.get('class_2', 0)}  "
          f"class_3={cr.get('class_3', 0)}  sub_class={cr.get('sub_class', 0)}  "
          f"NULL={st['class_rating'].isna().sum()}")

print(f"\nAll-state breakdown (OSM-sourced records):")
state_counts = gdf[gdf['data_source'] == 'osm']['state_abbr'].value_counts().sort_index()
print(state_counts.to_string())

# ── Output ────────────────────────────────────────────────────────────────────
gdf.to_parquet(OUT_PARQUET, index=False)
print(f"\nWrote {OUT_PARQUET.name} ({OUT_PARQUET.stat().st_size / 1e6:.1f} MB)")

sample = gdf.head(20).copy()
sample.drop(columns='geometry').to_csv(HERE / 'sample_rows.csv', index=False)
sample.to_file(HERE / 'sample_rows.geojson', driver='GeoJSON')
print("Exported: sample_rows.csv and sample_rows.geojson")
