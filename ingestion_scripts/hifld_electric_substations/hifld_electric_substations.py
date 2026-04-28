import requests
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path

BASE_URL = (
    "https://services6.arcgis.com/OO2s4OoyCZkYJ6oE/arcgis/rest/services/"
    "Substations/FeatureServer/0/query"
)
CACHE_FILE = Path("raw_substations.parquet")

KEEP_COLS = ['NAME', 'CITY', 'STATE', 'COUNTY', 'TYPE', 'STATUS',
             'MAX_VOLT', 'MIN_VOLT', 'LINES', 'MAX_INFER', 'MIN_INFER',
             'SOURCEDATE', 'VAL_DATE', 'geometry']

SENTINEL = -999999.0

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

print(f"Raw rows: {len(gdf)}")

gdf = gdf[KEEP_COLS].copy()

gdf = gdf.dropna(subset='geometry')

for col in ('MAX_VOLT', 'MIN_VOLT', 'LINES'):
    gdf[col] = gdf[col].replace(SENTINEL, np.nan)

gdf['SOURCEDATE'] = pd.to_datetime(gdf['SOURCEDATE'], unit='ms', errors='coerce')
gdf['VAL_DATE']   = pd.to_datetime(gdf['VAL_DATE'],   unit='ms', errors='coerce')

gdf = gdf[gdf['STATUS'] == 'IN SERVICE']
print(f"After IN SERVICE filter: {len(gdf)}")

def assign_class_rating(row):
    if row['TYPE'] == 'TAP':
        return 'class_3'
    v = row['MAX_VOLT']
    if pd.isna(v):
        return pd.NA
    if v >= 735:
        return 'class_0'
    if v >= 345:
        return 'class_1'
    if v >= 220:
        return 'class_2'
    return 'class_3'

gdf['class_rating'] = gdf.apply(assign_class_rating, axis=1)

print(f"\nclass_rating breakdown:\n{gdf['class_rating'].value_counts(dropna=False)}")
print(f"\nNull counts:\n{gdf.isnull().sum()}")

sample = gdf.head(20).copy()
sample.drop(columns='geometry').to_csv('sample_rows.csv', index=False)
sample.to_file('sample_rows.geojson', driver='GeoJSON')
print("\nExported: sample_rows.csv and sample_rows.geojson")
