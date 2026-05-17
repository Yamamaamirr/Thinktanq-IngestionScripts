"""
Ingest OurAirports public airport data and build runway protection buffers.

Source: https://ourairports.com/data/airports.csv  (public domain, ~80k airports global)

Output:
  ingestion_scripts/ourairports/airports_us_buffered.parquet
    columns: airport_id, name, iata_code, type, runway_buffer_geom (EPSG:5070)

Buffer sizes per PDF Rule 10 (runway protection zone approximation):
  large_airport  : 5 km
  medium_airport : 3 km
  small_airport  : 1.5 km
  heliport / seaplane / closed : skip

Run:
  python ingestion_scripts/ourairports/ingest_airports.py
"""
from pathlib import Path
import requests
import io
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

URL = 'https://ourairports.com/data/airports.csv'
OUT_PATH = Path('ingestion_scripts/ourairports/airports_us_buffered.parquet')
STATES = {'AZ','CA','NV','TX','VA'}

# Buffer per type, in metres
BUFFER_M = {
    'large_airport':  5000.0,
    'medium_airport': 3000.0,
    'small_airport':  1500.0,
}


def main():
    print(f'Downloading {URL} ...')
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    print(f'  {len(df):,} airports global')

    # Filter to US + 5 target states + non-closed
    us = df[(df.iso_country == 'US')].copy()
    print(f'  {len(us):,} US airports')

    # The OurAirports iso_region for US looks like 'US-TX'
    us['state_code'] = us['iso_region'].str.replace('US-', '', regex=False)
    target = us[us.state_code.isin(STATES) & us.type.isin(BUFFER_M.keys())].copy()
    print(f'  {len(target):,} airports in target states (large/medium/small only)')
    print(target.type.value_counts().to_string())

    # Make geometry & reproject to EPSG:5070
    target['geometry'] = target.apply(
        lambda r: Point(r['longitude_deg'], r['latitude_deg']), axis=1
    )
    g = gpd.GeoDataFrame(target, geometry='geometry', crs='EPSG:4326').to_crs(5070)

    # Apply per-type buffer
    g['buffer_m'] = g['type'].map(BUFFER_M)
    g['geometry'] = g.apply(lambda r: r.geometry.buffer(r['buffer_m']), axis=1)

    out_cols = ['ident','name','iata_code','type','state_code','buffer_m','geometry']
    out = g[out_cols].rename(columns={'ident':'airport_id'})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH}')
    print(f'  By state:')
    print(out.state_code.value_counts().to_string())
    print(f'  Total buffered area: {out.geometry.area.sum()/1e6:,.0f} km^2')


if __name__ == '__main__':
    main()
