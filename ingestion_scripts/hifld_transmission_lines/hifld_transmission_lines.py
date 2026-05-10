"""HIFLD Electric Power Transmission Lines ingestion.

Source: HIFLD (Homeland Infrastructure Foundation-Level Data), DHS/CISA.
        Pre-downloaded shapefile: US_Electric_Power_Transmission_Lines_*.zip
        Stored locally — no URL fetch needed (shapefile ships with this repo).

What this dataset IS:
  Every electric transmission line segment in the US >= 100 kV, as mapped by
  HIFLD from utility filings and aerial imagery. This script keeps all IN
  SERVICE lines at 100 kV and above, classified into four tiers per PRD v2.1.

What it IS NOT:
  - No distribution lines (< 100 kV)
  - No substations (see hifld_electric_substations script)
  - No generation / capacity data (see eia_form860 script)

Voltage → class_rating mapping (matches PRD v2.1 class_weight_mapping exactly):
  735 kV+ / DC  →  class_1    weight 500  (regional backbone)
  500 kV        →  class_1    weight 500  (regional backbone)
  345 kV        →  class_2    weight 345  (high-capacity sub-regional)
  220–287 kV    →  class_3    weight 230  (sub-regional / metro feeds)
  100–161 kV    →  sub_class  weight 138  (local transmission, includes 138kV)

Note for keep-zone buffering: only class_1 and class_2 (345kV+) are buffered
in build_keep_zones.py. class_3 and sub_class are retained in this parquet
for scoring use only (infrastructure_proximity_score).

Output schema (PostGIS GEOMETRY(LINESTRING, 4326)):
  voltage_kv      NUMERIC   actual kV (imputed from volt_class where null)
  volt_class      TEXT      HIFLD voltage class label
  owner           TEXT      utility / owner name
  substation_from TEXT      upstream substation name
  substation_to   TEXT      downstream substation name
  class_rating    TEXT      class_1 / class_2 / class_3 / sub_class
  geometry        LINESTRING(4326)
  ingested_at     TIMESTAMPTZ
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path

SOURCE_ZIP = Path(__file__).parent / 'US_Electric_Power_Transmission_Lines_-8378795084787321001.zip'
OUT_PARQUET = Path(__file__).parent / 'hifld_transmission_lines.parquet'

KEEP_COLS = ['VOLTAGE', 'VOLT_CLASS', 'OWNER', 'STATUS', 'SUB_1', 'SUB_2', 'geometry']

VOLT_CLASS_LOWER = {
    '100-161':       100,
    '220-287':       220,
    '345':           345,
    '500':           500,
    'DC':            500,
    '735 AND ABOVE': 735,
}

SCORING_CLASSES = {'100-161', '220-287', '345', '500', '735 AND ABOVE', 'DC'}

CLASS_RATING_MAP = {
    '100-161':       'sub_class',  # 138kV local transmission
    '220-287':       'class_3',    # sub-regional / metro feeds
    '345':           'class_2',    # high-capacity sub-regional
    '500':           'class_1',    # regional backbone
    '735 AND ABOVE': 'class_1',    # regional backbone
    'DC':            'class_1',    # regional backbone (HVDC)
}

# ── Load ─────────────────────────────────────────────────────────────────────

gdf = gpd.read_file(SOURCE_ZIP)
print(f"Raw rows: {len(gdf):,}")

gdf = gdf[KEEP_COLS].copy()

# ── Voltage imputation ────────────────────────────────────────────────────────

gdf['VOLTAGE'] = gdf['VOLTAGE'].replace(-999999.0, np.nan)
null_mask = gdf['VOLTAGE'].isna()
gdf.loc[null_mask, 'VOLTAGE'] = gdf.loc[null_mask, 'VOLT_CLASS'].map(VOLT_CLASS_LOWER)
print(f"Imputed voltage from VOLT_CLASS for {null_mask.sum():,} rows")

# ── Filter ────────────────────────────────────────────────────────────────────

gdf = gdf[gdf['STATUS'] == 'IN SERVICE']
print(f"After IN SERVICE filter: {len(gdf):,}")

gdf = gdf[gdf['VOLT_CLASS'].isin(SCORING_CLASSES)]
print(f"After voltage class filter (100 kV+): {len(gdf):,}")

# ── Geometry ──────────────────────────────────────────────────────────────────

gdf = gdf.to_crs(epsg=4326)

# Explode MultiLineString -> LineString for uniform PostGIS column type.
# The 8 source MultiLineStrings represent lines with physical gaps (river
# crossings, underground sections). Exploding adds ~10-15 rows but ensures
# GEOMETRY(LINESTRING, 4326) typing on load with no INSERT errors.
n_before = len(gdf)
gdf = gdf.explode(index_parts=False).reset_index(drop=True)
print(f"After explode: {len(gdf):,} (+{len(gdf) - n_before} from MultiLineString splits)")

# ── Rename to canonical schema ────────────────────────────────────────────────

gdf = gdf.rename(columns={
    'VOLTAGE':   'voltage_kv',
    'VOLT_CLASS': 'volt_class',
    'OWNER':     'owner',
    'SUB_1':     'substation_from',
    'SUB_2':     'substation_to',
})
gdf['class_rating'] = gdf['volt_class'].map(CLASS_RATING_MAP)
gdf = gdf.drop(columns=['STATUS'])
gdf['ingested_at'] = pd.Timestamp.utcnow()

# ── Validation ────────────────────────────────────────────────────────────────

assert len(gdf) > 15_000, f"Suspiciously low segment count: {len(gdf):,}"
assert gdf.geom_type.eq('LineString').all(), "Non-LineString geometry found after explode"
assert gdf.geometry.is_valid.all(), "Invalid geometries detected"
assert gdf['class_rating'].notna().all(), "Unmapped volt_class found"

print(f"\nFinal CRS: {gdf.crs}")
print(f"Total segments: {len(gdf):,}")

print(f"\nVolt class breakdown:")
print(gdf['volt_class'].value_counts().to_string())

print(f"\nClass rating breakdown:")
print(gdf['class_rating'].value_counts().to_string())

print(f"\nNull voltage after imputation: {gdf['voltage_kv'].isna().sum()}")

print(f"\nTop 15 owners by segment count:")
print(gdf['owner'].value_counts().head(15).to_string())

# ── Output ────────────────────────────────────────────────────────────────────

gdf.to_parquet(OUT_PARQUET, index=False)
print(f"\nWrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size / 1e6:.1f} MB)")
