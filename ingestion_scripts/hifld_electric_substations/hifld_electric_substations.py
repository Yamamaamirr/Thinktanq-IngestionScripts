"""HIFLD Electric Power Substations ingestion.

Source: HIFLD (Homeland Infrastructure Foundation-Level Data), DHS/CISA.
        Live ArcGIS REST service — paginated fetch, cached to parquet.

What this dataset IS:
  Every electric power substation in the US at 69 kV and above, as reported
  by utilities to HIFLD. Substations with MAX_VOLT < 69 kV may be present
  but coverage is not guaranteed.

What it IS NOT:
  - No transmission lines (see hifld_transmission_lines script)
  - No generation / capacity data (see eia_form860 script)

Voltage → class_rating mapping (based on MAX_VOLT, matches PRD class_weight_mapping):
  >= 345 kV  →  class_1   (345/500/765kV — regional backbone and above)
  230–287 kV →  class_2   (sub-regional / metro feeds)
  69–161 kV  →  class_3   (lower transmission / sub-transmission)
  < 69 kV    →  sub_class (distribution voltage)
  TAP        →  class_3   (tap points forced to class_3 regardless of voltage)
  no voltage →  NULL      (18.5% of IN SERVICE non-TAP records; utility did not file voltage)

MAX_INFER flag: HIFLD field indicating HOW the MAX_VOLT value was obtained.
  'Y' = HIFLD inferred the voltage from connected transmission line filings.
  'N' = Voltage was directly reported by the utility.
  This is a data-quality flag on MAX_VOLT, NOT a separate fallback voltage value.
  When MAX_VOLT is null, MAX_INFER is still Y/N but there is no voltage to use.

volt_source in output:
  'reported'       — MAX_VOLT present, utility directly filed it (MAX_INFER='N')
  'hifld_inferred' — MAX_VOLT present, HIFLD derived it from connected lines (MAX_INFER='Y')
  'tap'            — TYPE=TAP, forced class_3
  NULL             — no voltage data; scoring engine must fall back to transmission
                     line proximity for parcels near these substations

Data profile (as of last ingestion):
  72,364 IN SERVICE substations
  class_1: ~1,904 (345 kV and above — includes 345/500/765 kV)
  class_2: ~3,821 (230–287 kV)
  class_3: ~51,104 (69–161 kV or TAP)
  sub_class: ~2,134 (< 69 kV)
  NULL:    ~13,401 (18.5% of IN SERVICE — no recovery path within HIFLD)

Output schema (PostGIS GEOMETRY(POINT, 4326)):
  name                TEXT
  city                TEXT
  state_abbr          TEXT
  county              TEXT
  substation_type     TEXT
  max_volt            NUMERIC   highest voltage at substation (kV), nullable
  min_volt            NUMERIC   lowest voltage at substation (kV), nullable
  max_volt_inferred   BOOLEAN   True if HIFLD inferred max_volt (vs utility-reported)
  lines               INTEGER   number of connected transmission lines
  class_rating        TEXT      class_1/class_2/class_3/sub_class/NULL
  volt_source         TEXT      'reported'/'hifld_inferred'/'tap'/NULL
  source_date         DATE
  val_date            DATE
  geometry            POINT(4326)
  ingested_at         TIMESTAMPTZ
"""
import requests
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path

BASE_URL = (
    "https://services6.arcgis.com/OO2s4OoyCZkYJ6oE/arcgis/rest/services/"
    "Substations/FeatureServer/0/query"
)
CACHE_FILE = Path(__file__).parent / 'raw_substations.parquet'
OUT_PARQUET = Path(__file__).parent / 'hifld_electric_substations.parquet'

KEEP_COLS = ['NAME', 'CITY', 'STATE', 'COUNTY', 'TYPE', 'STATUS',
             'MAX_VOLT', 'MIN_VOLT', 'MAX_INFER', 'LINES',
             'SOURCEDATE', 'VAL_DATE', 'geometry']

SENTINEL = -999999.0


# ── Fetch (cached) ────────────────────────────────────────────────────────────

if CACHE_FILE.exists():
    print("Loading from cache...")
    gdf = gpd.read_parquet(CACHE_FILE)
else:
    all_features = []
    offset = 0
    while True:
        params = {
            'where': '1=1',
            'outFields': '*',
            'resultOffset': offset,
            'resultRecordCount': 2000,
            'f': 'geojson',
        }
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        features = r.json().get('features', [])
        if not features:
            break
        all_features.extend(features)
        offset += 2000
        print(f"Fetched {offset} records...")

    gdf = gpd.GeoDataFrame.from_features(all_features, crs='EPSG:4326')
    gdf.to_parquet(CACHE_FILE)
    print(f"Cached {len(gdf)} rows to {CACHE_FILE}")

print(f"Raw rows: {len(gdf):,}")

# ── Select columns and clean sentinels ───────────────────────────────────────

gdf = gdf[KEEP_COLS].copy()
gdf = gdf.dropna(subset='geometry')

for col in ('MAX_VOLT', 'MIN_VOLT', 'LINES'):
    gdf[col] = gdf[col].replace(SENTINEL, np.nan)

gdf['SOURCEDATE'] = pd.to_datetime(gdf['SOURCEDATE'], unit='ms', errors='coerce')
gdf['VAL_DATE']   = pd.to_datetime(gdf['VAL_DATE'],   unit='ms', errors='coerce')

# ── Filter ────────────────────────────────────────────────────────────────────

gdf = gdf[gdf['STATUS'] == 'IN SERVICE'].copy()
print(f"After IN SERVICE filter: {len(gdf):,}")

# ── Class rating ──────────────────────────────────────────────────────────────
# MAX_INFER is a Y/N flag on MAX_VOLT (was it inferred or directly reported?).
# It is NOT a fallback voltage value. When MAX_VOLT is null, class_rating is NULL.

def _voltage_to_class(v):
    if v >= 345: return 'class_1'
    if v >= 230: return 'class_2'
    if v >= 69:  return 'class_3'
    return 'sub_class'

def assign_class(row):
    if row['TYPE'] == 'TAP':
        return 'class_3'
    if pd.notna(row['MAX_VOLT']):
        return _voltage_to_class(row['MAX_VOLT'])
    return pd.NA

def assign_volt_source(row):
    if row['TYPE'] == 'TAP':
        return 'tap'
    if pd.notna(row['MAX_VOLT']):
        return 'hifld_inferred' if row['MAX_INFER'] == 'Y' else 'reported'
    return pd.NA

gdf['class_rating'] = gdf.apply(assign_class, axis=1)
gdf['volt_source']  = gdf.apply(assign_volt_source, axis=1)

# ── Rename to canonical schema ────────────────────────────────────────────────

gdf = gdf.rename(columns={
    'NAME':       'name',
    'CITY':       'city',
    'STATE':      'state_abbr',
    'COUNTY':     'county',
    'TYPE':       'substation_type',
    'MAX_VOLT':   'max_volt',
    'MIN_VOLT':   'min_volt',
    'LINES':      'lines',
    'SOURCEDATE': 'source_date',
    'VAL_DATE':   'val_date',
})
gdf['max_volt_inferred'] = gdf['MAX_INFER'] == 'Y'
gdf = gdf.drop(columns=['STATUS', 'MAX_INFER'])
gdf['ingested_at'] = pd.Timestamp.utcnow()

# ── Validation ────────────────────────────────────────────────────────────────

assert len(gdf) > 50_000, f"Suspiciously low substation count: {len(gdf):,}"
assert gdf.geom_type.eq('Point').all(), "Non-Point geometry found"
assert gdf.geometry.is_valid.all(), "Invalid geometries detected"
valid_classes = {'class_1', 'class_2', 'class_3', 'sub_class'}
actual_classes = set(gdf['class_rating'].dropna().unique())
assert actual_classes <= valid_classes, f"Unexpected class values: {actual_classes - valid_classes}"

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\nFinal CRS: {gdf.crs}")
print(f"Total IN SERVICE substations: {len(gdf):,}")

print(f"\nClass rating breakdown:")
print(gdf['class_rating'].value_counts(dropna=False).to_string())

print(f"\nVoltage source breakdown:")
print(gdf['volt_source'].value_counts(dropna=False).to_string())

null_cr = gdf['class_rating'].isna().sum()
print(f"\nNULL class_rating (no voltage data): {null_cr:,} ({100*null_cr/len(gdf):.1f}%)")
print("  These have no recovery path within HIFLD.")
print("  Scoring engine must fall back to transmission line proximity.")

print(f"\nTarget state breakdown (class_1/2 = high-value, class_3/sub_class/NULL = low-value):")
for state in ['CA', 'TX', 'AZ', 'NV', 'VA']:
    st = gdf[gdf['state_abbr'] == state]
    cr = st['class_rating'].value_counts(dropna=False)
    print(f"  {state}: {len(st):,} total | "
          f"class_1={cr.get('class_1',0)}  class_2={cr.get('class_2',0)}  "
          f"class_3={cr.get('class_3',0)}  sub_class={cr.get('sub_class',0)}  "
          f"NULL={st['class_rating'].isna().sum()}")

# ── Output ────────────────────────────────────────────────────────────────────

gdf.to_parquet(OUT_PARQUET, index=False)
print(f"\nWrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size / 1e6:.1f} MB)")
