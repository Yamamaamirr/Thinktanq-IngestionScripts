"""Build a supplemental substation dataset to fill HIFLD gaps.

Two sources:
  1. EIA Form 860 plant locations — covers power plant interconnection substations
     (nuclear plants, large gas/coal) that HIFLD classifies as generators not
     substations. Calvert Cliffs, Braidwood, Sioux etc. live here.

  2. OpenStreetMap via Overpass API — covers new high-voltage substations built
     after HIFLD's snapshot (post-2022 ERCOT renewables buildout, MISO expansion).
     Queried for states with the largest unmatched MW gaps.

Output:
  supplemental_substations.parquet
  Schema matches what anchor_queue_enrichment.py expects:
    name        TEXT   — substation/plant name
    state_abbr  TEXT   — US state abbreviation
    geometry    POINT  — EPSG:4326
    class_rating TEXT  — 'supplement_eia' or 'supplement_osm'
    source      TEXT   — 'eia860' or 'osm'
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.wkb import loads as wkb_loads

BASE = Path(__file__).parent
EIA_PARQUET = Path(r'C:\thinqtank\ingestion_scripts\eia_form860\eia_form860.parquet')
OUT_PARQUET  = BASE / 'supplemental_substations.parquet'

# States where unmatched active MW is largest — targeted OSM scrape.
# TX: new ERCOT renewable substations; IL/MD/WV/VA/MO/OH: PJM+MISO gaps
OSM_TARGET_STATES = {
    'TX': (25.8, -106.7, 36.6,  -93.5),   # all of Texas
    'IL': (36.9, -91.5,  42.5,  -87.0),
    'MD': (37.9, -79.5,  39.7,  -74.9),
    'WV': (37.2, -82.7,  40.6,  -77.7),
    'VA': (36.5, -83.7,  39.5,  -75.2),
    'MO': (36.0, -95.8,  40.6,  -89.1),
    'OH': (38.4, -84.8,  41.9,  -80.5),
    'ND': (45.9, -104.1, 49.0,  -96.5),
    'IA': (40.4, -96.6,  43.5,  -90.1),
    'MI': (41.7, -90.4,  48.3,  -82.4),
    'IN': (37.8, -88.1,  41.8,  -84.7),
    'AR': (33.0, -94.6,  36.5,  -89.6),
    'LA': (28.9, -94.0,  33.0,  -88.8),
    'AZ': (31.3, -114.8, 37.0, -109.0),
    'CA': (32.5, -124.5, 42.0, -114.1),
    'NV': (35.0, -120.0, 42.0, -114.0),
}

# OSM voltage values (in volts) for HV transmission substations.
OSM_VOLTAGE_REGEX = r'^(115000|138000|161000|230000|345000|500000|765000)$'

OVERPASS_URL = 'https://overpass-api.de/api/interpreter'


def _overpass_query(bbox, voltage_regex):
    """Fetch HV substations from OSM within bounding box (S,W,N,E)."""
    s, w, n, e = bbox
    q = (
        f'[out:json][timeout:90];'
        f'('
        f'node["power"="substation"]["voltage"~"{voltage_regex}"]({s},{w},{n},{e});'
        f'way["power"="substation"]["voltage"~"{voltage_regex}"]({s},{w},{n},{e});'
        f'relation["power"="substation"]["voltage"~"{voltage_regex}"]({s},{w},{n},{e});'
        f');out center;'
    )
    encoded = urllib.parse.urlencode({'data': q}).encode('utf-8')
    req = urllib.request.Request(
        OVERPASS_URL, data=encoded,
        headers={
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'thinqtank-substation-research/1.0 (internal data pipeline)',
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def fetch_osm_substations():
    rows = []
    for state, bbox in OSM_TARGET_STATES.items():
        print(f'  OSM {state} ...', end=' ', flush=True)
        try:
            result = _overpass_query(bbox, OSM_VOLTAGE_REGEX)
            elements = result.get('elements', [])
            for el in elements:
                tags = el.get('tags', {})
                name = tags.get('name') or tags.get('operator') or tags.get('ref')
                if not name:
                    continue
                if el['type'] == 'node':
                    lat, lon = el['lat'], el['lon']
                else:
                    center = el.get('center', {})
                    lat, lon = center.get('lat'), center.get('lon')
                if lat is None or lon is None:
                    continue
                rows.append({
                    'name': name,
                    'state_abbr': state,
                    'lat': lat, 'lon': lon,
                    'source': 'osm',
                    'class_rating': 'supplement_osm',
                })
            print(f'{len(elements)} elements, {sum(1 for r in rows if r["state_abbr"]==state)} named')
        except Exception as exc:
            print(f'ERROR: {exc}')
        time.sleep(1)   # polite rate limit
    return rows


def fetch_eia_substations():
    """Use EIA Form 860 plant locations as virtual substations.

    Only large plants (total plant capacity >= 200 MW) or nuclear plants are
    included. Small solar/wind/battery sites are excluded: they share names
    with counties and suburbs, causing false matches to transmission substations
    in the ISO queue. Large thermal plants and nuclear plants have dedicated
    transmission-level switchyards that genuinely share the plant name.
    """
    eia = pd.read_parquet(EIA_PARQUET)
    plant_cap = (
        eia.groupby('plant_id', observed=True)['capacity_mw']
        .sum()
        .reset_index()
        .rename(columns={'capacity_mw': 'total_cap_mw'})
    )
    eia = eia.merge(plant_cap, on='plant_id')
    is_nuclear = eia['fuel_type'] == 'nuclear'
    is_large   = eia['total_cap_mw'] >= 200
    eia = eia[is_nuclear | is_large]

    # Keep one row per plant (deduplicate generators at the same plant).
    plants = (
        eia.dropna(subset=['geometry'])
        .drop_duplicates(subset=['plant_id'])
        .copy()
    )

    rows = []
    for _, row in plants.iterrows():
        try:
            geom = wkb_loads(row['geometry'])
        except Exception:
            continue
        rows.append({
            'name': row['plant_name'],
            'state_abbr': row['state_abbr'],
            'lat': geom.y,
            'lon': geom.x,
            'source': 'eia860',
            'class_rating': 'supplement_eia',
        })
    return rows


def main():
    print('=== Building supplemental substations ===\n')

    print('1. EIA Form 860 plant locations...')
    eia_rows = fetch_eia_substations()
    print(f'   {len(eia_rows):,} plants loaded\n')

    print('2. OpenStreetMap HV substations (Overpass API)...')
    osm_rows = fetch_osm_substations()
    print(f'\n   {len(osm_rows):,} OSM substations with names\n')

    all_rows = eia_rows + osm_rows
    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=['name', 'state_abbr', 'lat', 'lon'])
    df['geometry'] = [Point(r.lon, r.lat) for r in df.itertuples()]

    gdf = gpd.GeoDataFrame(df[['name', 'state_abbr', 'geometry', 'class_rating', 'source']],
                           geometry='geometry', crs='EPSG:4326')

    # Remove duplicates by name+state (OSM may overlap with EIA).
    gdf['_key'] = gdf['name'].str.upper().str.strip() + '|' + gdf['state_abbr']
    gdf = gdf.drop_duplicates(subset=['_key']).drop(columns=['_key'])

    gdf.to_parquet(OUT_PARQUET, index=False)
    print(f'Wrote {len(gdf):,} supplemental substations -> {OUT_PARQUET}')
    print(f'\nBreakdown by source:')
    print(gdf['source'].value_counts().to_string())
    print(f'\nBreakdown by state (OSM only):')
    print(gdf[gdf['source']=='osm']['state_abbr'].value_counts().to_string())


if __name__ == '__main__':
    main()
