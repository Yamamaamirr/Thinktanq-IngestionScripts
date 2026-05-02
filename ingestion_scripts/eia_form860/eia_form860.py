"""EIA Form 860 generator capacity ingestion via PUDL.

Source: PUDL (Public Utility Data Liberation, catalyst.coop) -- the cleaned,
        normalized release of EIA Form 860 + 923 + FERC + EPA data.
        Public S3: pudl.catalyst.coop, no auth required.
        Stable channel ('stable/') is versioned (v2026.4.0 = 2026-04-09).
        Annual refresh per PRD section 10.2 + EIA 860's annual cycle.

What this dataset IS:
  Every utility-scale (>= 1 MW) electric generator in the US, as reported on
  EIA Form 860. PUDL's `out_eia__yearly_generators` is the analyst-ready join
  of generators + plants + utilities, with location, capacity, fuel, and
  operational status pre-resolved. Multi-year history 2001 to current --
  this script keeps only the most recent report year.

What it IS NOT:
  - No transmission / distribution data (those live in HIFLD, separate script)
  - No queue / planned-but-not-built data (GridStatus ISO queues, separate)
  - No load / demand data (FERC 714, sister script ferc714/ferc714.py)
  - Not behind-the-meter / distributed generation < 1 MW

What feeds the capacity_signal (PRD section 4.1):
  capacity_signal = generation_capacity (this script -- existing only)
                  - queued_MW          (gridstatus_iso_queues script)
                  + load_headroom      (ferc714 script)
  At enrichment time the engine sums capacity_mw within a 50-mile (80,467 m)
  buffer of each substation to compute generation_capacity_per_substation.

Outputs:
  eia_form860.parquet          -- existing/operational generators (capacity scoring)
  eia_form860_retired.parquet  -- retired generators (Owen's Infrastructure
                                  Reuse Nodes layer: retired coal/gas/nuclear sites)

Output schema -- existing parquet (matches generators table):
  plant_id           TEXT     plant_id_eia (cast to str for PG portability)
  generator_id       TEXT     generator_id (within plant)
  plant_name         TEXT     plant_name_eia
  utility_id         TEXT     utility_id_eia
  utility_name       TEXT     utility_name_eia (analyst convenience)
  state_abbr         CHAR(2)  state
  capacity_mw        NUMERIC  nameplate capacity in MW
  fuel_type          TEXT     fuel_type_code_pudl
                              (coal/gas/nuclear/wind/solar/hydro/oil/other)
  technology         TEXT     technology_description (e.g.
                              'Natural Gas Fired Combined Cycle')
  ba_code            TEXT     balancing_authority_code_eia
                              (joins to FERC 714 + utility_territories)
                              NOTE: AK/HI have null ba_code -- expected,
                              neither state is part of FERC interconnection.
  geometry           Point(4326)  -- from latitude, longitude
                              PostGIS column: GEOGRAPHY(POINT, 4326)
  report_year        INTEGER  derived from report_date
  ingested_at        TIMESTAMPTZ

Output schema -- retired parquet (same as above plus):
  operational_status TEXT     'retired' or 'retiring' -- kept for downstream
                              filtering (e.g. retirement year, fuel reuse priority)
"""
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError

PUDL_BASE = 'https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/stable/'
SOURCE_FILE = 'out_eia__yearly_generators.parquet'

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
RAW_FILE = CACHE_DIR / SOURCE_FILE
OUT_PARQUET = Path(__file__).parent / 'eia_form860.parquet'
OUT_RETIRED = Path(__file__).parent / 'eia_form860_retired.parquet'

KEEP_COLS = [
    'plant_id_eia', 'generator_id',
    'plant_name_eia',
    'utility_id_eia', 'utility_name_eia',
    'state',
    'capacity_mw',
    'fuel_type_code_pudl',
    'technology_description',
    'balancing_authority_code_eia',
    'operational_status',
    'latitude', 'longitude',
    'report_date',
]

RETRY_DELAYS = [5, 10, 20, 30, 60, 120, 180]
NETWORK_ERRORS = (ConnectionError, Timeout, ChunkedEncodingError, ProtocolError)
RETRY_HTTP_STATUSES = {429, 500, 502, 503, 504}


def _download(url, dest):
    """Stream-download with retry on transient errors."""
    last_exc = None
    for delay in RETRY_DELAYS:
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                if r.status_code in RETRY_HTTP_STATUSES:
                    last_exc = RuntimeError(f"HTTP {r.status_code}")
                    print(f"  Server error HTTP {r.status_code} -- retry in {delay}s")
                    time.sleep(delay)
                    continue
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + '.part')
                with open(tmp, 'wb') as f:
                    for chunk in r.iter_content(8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.rename(dest)
                return
        except NETWORK_ERRORS as e:
            last_exc = e
            print(f"  Network error: {type(e).__name__} -- retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"Max retries exceeded ({last_exc!r})")


# ── Fetch ────────────────────────────────────────────────────────────────────

if not RAW_FILE.exists():
    url = PUDL_BASE + SOURCE_FILE
    print(f"Downloading {SOURCE_FILE} from PUDL stable...")
    _download(url, RAW_FILE)
print(f"Loading raw: {RAW_FILE.name} "
      f"({RAW_FILE.stat().st_size/1e6:.1f} MB)")
df = pd.read_parquet(RAW_FILE, columns=KEEP_COLS)
print(f"Raw rows: {len(df):,}")

# ── Process ──────────────────────────────────────────────────────────────────

# PUDL stores multi-year history. Keep most recent reported year only.
df['report_year'] = pd.to_datetime(df['report_date']).dt.year
latest_year = int(df['report_year'].max())
n_before_year = len(df)
df = df[df['report_year'] == latest_year]
print(f"After year filter ({latest_year}): {len(df):,} "
      f"(dropped {n_before_year - len(df):,} historical)")

# Split now, before any row-dropping, so retired generators are captured.
# existing -> capacity scoring (Dimension 2 / capacity_signal)
# retired/retiring -> Infrastructure Reuse Nodes layer
df_existing = df[df['operational_status'] == 'existing'].copy()
df_retired = df[df['operational_status'].isin(['retired', 'retiring'])].copy()
print(f"Status split: {len(df_existing):,} existing, "
      f"{len(df_retired):,} retired/retiring, "
      f"{len(df) - len(df_existing) - len(df_retired):,} other (proposed/etc.)")


def _build_gdf(frame, label, include_status=False):
    """Shared pipeline: drop nulls, rename, geometry, return GeoDataFrame."""
    n_before = len(frame)
    frame = frame.dropna(subset=['latitude', 'longitude', 'capacity_mw'])
    frame = frame[frame['capacity_mw'] > 0]
    print(f"[{label}] After location+capacity filter: {len(frame):,} "
          f"(dropped {n_before - len(frame):,})")

    frame = frame.rename(columns={
        'plant_id_eia':                 'plant_id',
        'plant_name_eia':               'plant_name',
        'utility_id_eia':               'utility_id',
        'utility_name_eia':             'utility_name',
        'state':                        'state_abbr',
        'fuel_type_code_pudl':          'fuel_type',
        'technology_description':       'technology',
        'balancing_authority_code_eia': 'ba_code',
    })

    frame['plant_id'] = frame['plant_id'].astype('Int64').astype(str)
    frame['utility_id'] = frame['utility_id'].astype('Int64').astype(str)

    # ~8 plants straddle state borders; PUDL reports no state for them.
    # These are border facilities (e.g. MN/WI, VT/NH) where EIA has no
    # single-state assignment. Drop rather than guess -- <0.05% of rows.
    n_before_state = len(frame)
    frame = frame.dropna(subset=['state_abbr'])
    dropped = n_before_state - len(frame)
    if dropped:
        print(f"[{label}] Dropped {dropped} null state_abbr "
              f"(border-straddle plants without EIA state assignment)")

    gdf = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame['longitude'], frame['latitude']),
        crs='EPSG:4326',
    )

    keep_out = ['plant_id', 'generator_id', 'plant_name', 'utility_id',
                'utility_name', 'state_abbr', 'capacity_mw', 'fuel_type',
                'technology', 'ba_code', 'geometry', 'report_year']
    if include_status:
        keep_out.append('operational_status')
    gdf = gdf[keep_out].copy()
    gdf['ingested_at'] = pd.Timestamp.utcnow()
    return gdf


gdf = _build_gdf(df_existing, 'existing')
gdf_retired = _build_gdf(df_retired, 'retired', include_status=True)

# ── Validation (existing) ────────────────────────────────────────────────────

assert len(gdf) > 5_000, f"Suspiciously low generator count: {len(gdf):,}"
total_gw = gdf['capacity_mw'].sum() / 1000
assert 800 < total_gw < 1500, \
    f"Total capacity {total_gw:.0f} GW outside expected 800-1500 range"

print(f"\n=== Existing generators (capacity scoring) ===")
print(f"Report year: {latest_year}")
print(f"Total operating generators: {len(gdf):,}")
print(f"Total nameplate capacity: {total_gw:,.0f} GW "
      f"({gdf['capacity_mw'].sum():,.0f} MW)")

print(f"\nBy fuel type (capacity GW):")
fuel_gw = gdf.groupby('fuel_type', observed=True)['capacity_mw'].sum() / 1000
print(fuel_gw.sort_values(ascending=False).round(1).to_string())

print(f"\nState coverage (top 15 by GW):")
state_gw = ((gdf.groupby('state_abbr', observed=True)['capacity_mw'].sum() / 1000)
            .sort_values(ascending=False).round(1))
print(state_gw.head(15).to_string())
print(f"Total states covered: {gdf['state_abbr'].nunique()}")

print(f"\nNull counts (key fields):")
for c in ['plant_id', 'capacity_mw', 'state_abbr', 'fuel_type', 'ba_code']:
    n = gdf[c].isnull().sum()
    note = ''
    if c == 'ba_code' and n > 0:
        ak_hi = gdf[gdf['state_abbr'].isin(['AK', 'HI'])][c].isnull().sum()
        note = f'  (AK/HI not in FERC interconnection -- {ak_hi} expected null)'
    print(f"  {c:<14} {n:>5}{note}")

# ── Spot checks (existing) ───────────────────────────────────────────────────

print("\nSpot checks (known large plants):")
print("-" * 70)
SPOT_PLANTS = {
    "Palo Verde Nuclear (AZ)":     ['palo verde'],
    "Robert W Scherer (GA)":       ['scherer'],
    "Grand Coulee Hydro (WA)":     ['grand coulee'],
    "West County Energy (FL)":     ['west county'],
    "Comanche Peak Nuclear (TX)":  ['comanche peak'],
}
for label, name_parts in SPOT_PLANTS.items():
    mask = gdf['plant_name'].str.lower().fillna('').str.contains(name_parts[0])
    sub = gdf[mask]
    if len(sub):
        total = sub['capacity_mw'].sum()
        n_gen = len(sub)
        plant = sub['plant_name'].iloc[0]
        print(f"  OK  {label:<32} -> {plant[:36]:<38} "
              f"{n_gen} gen, {total:,.0f} MW")
    else:
        print(f"  ?   {label:<32} -> not found")

print(f"\nTarget-state capacity:")
for s in ('CA', 'TX', 'AZ', 'NV', 'VA'):
    sub = gdf[gdf['state_abbr'] == s]
    print(f"  {s}: {len(sub):>5} generators, {sub['capacity_mw'].sum()/1000:>6.1f} GW")

# ── Retired summary ──────────────────────────────────────────────────────────

print(f"\n=== Retired generators (Infrastructure Reuse Nodes layer) ===")
print(f"Total retired/retiring: {len(gdf_retired):,} generators")
retired_gw = gdf_retired['capacity_mw'].sum() / 1000
print(f"Total retired capacity: {retired_gw:,.0f} GW")
print(f"\nBy fuel type (retired GW -- reuse potential):")
ret_fuel = (gdf_retired.groupby('fuel_type', observed=True)['capacity_mw'].sum() / 1000)
print(ret_fuel.sort_values(ascending=False).round(1).to_string())
print(f"\nTop 10 retired plants by capacity:")
ret_top = (gdf_retired.groupby(['plant_name', 'state_abbr', 'fuel_type'], observed=True)
           ['capacity_mw'].sum().sort_values(ascending=False).head(10))
print(ret_top.round(0).to_string())

# ── Output ───────────────────────────────────────────────────────────────────

gdf.to_parquet(OUT_PARQUET, index=False)
print(f"\nWrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e6:.1f} MB)")

gdf_retired.to_parquet(OUT_RETIRED, index=False)
print(f"Wrote {OUT_RETIRED} ({OUT_RETIRED.stat().st_size/1e6:.1f} MB)")

sample = gdf.head(20).copy()
sample.drop(columns='geometry').to_csv(
    Path(__file__).parent / 'sample_rows.csv', index=False)
print("Exported: sample_rows.csv")
