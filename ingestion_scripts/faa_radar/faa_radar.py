"""FAA radar exclusion zones ingestion (NEXRAD WSR-88D).

Fetches the official NEXRAD WSR-88D station list from the NOAA Weather API,
computes a metric-accurate 3-mile buffer polygon around each station, and
produces a parquet ready for PostGIS load into the radar_exclusion_zones table.

Source: https://api.weather.gov/radar/stations
We filter to stationType == 'WSR-88D' (the standard NEXRAD network, ~160
stations nationally). TDWR (Terminal Doppler Weather Radar at airports) and
Profiler stations have different exclusion rules per FAA OE and are excluded
from this dataset per PRD section 7 disqualifier #7.

Buffer computation:
  Each 3-mile buffer is computed in an azimuthal-equidistant projection
  centred on the station so that 4828m east, 4828m north, 4828m southwest
  all yield the same true distance. A naive degree-based buffer like
  point.buffer(0.04) produces an ellipse that's 1.4x oversized at high
  latitudes -- not safe for a hard disqualifier.

Disqualifier semantics:
  Unlike disqualifier #6 (drought) which needs analyst reference tables,
  disqualifier #7 (radar EMI) is self-contained. A parcel centroid within
  any of these buffers => parcel disqualified, no further checks.

Output schema (matches radar_exclusion_zones, minus zone_id which Postgres
generates via gen_random_uuid()):
  station_id       TEXT             4-letter station ID, e.g. KBMX
  station_name     TEXT
  station_lon      NUMERIC
  station_lat      NUMERIC
  buffer_radius_m  NUMERIC          4828.0 (3 miles)
  geometry         Polygon (4326)   pre-computed buffer
  fetched_at       TIMESTAMPTZ
"""
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from pyproj import Transformer
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from shapely.geometry import Point
from shapely.ops import transform
from urllib3.exceptions import ProtocolError

NEXRAD_URL = 'https://api.weather.gov/radar/stations'
BUFFER_RADIUS_M = 4828.0  # exactly 3 miles (4828.032 metres rounded)

OUT_PARQUET = Path(__file__).parent / 'radar_exclusion_zones.parquet'

RETRY_DELAYS = [5, 15, 45, 120]
NETWORK_ERRORS = (ConnectionError, Timeout, ChunkedEncodingError, ProtocolError)


def _fetch_with_retry(url, timeout=60):
    last_exc = None
    headers = {'User-Agent': 'thinqtank-radar-ingest/1.0 (contact: ops)'}
    for attempt, delay in enumerate(RETRY_DELAYS, 1):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            return r.json()
        except NETWORK_ERRORS as e:
            last_exc = e
            print(f"  network error attempt {attempt} ({type(e).__name__}) -- retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"NOAA radar fetch: max retries exceeded ({last_exc!r})")


def _compute_buffer_polygon(lon, lat, radius_m=BUFFER_RADIUS_M):
    """Metric-accurate buffer via local azimuthal equidistant projection."""
    aeqd = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
    to_metric = Transformer.from_crs('EPSG:4326', aeqd, always_xy=True).transform
    to_wgs84 = Transformer.from_crs(aeqd, 'EPSG:4326', always_xy=True).transform
    pt_m = transform(to_metric, Point(lon, lat))
    buf_m = pt_m.buffer(radius_m)
    return transform(to_wgs84, buf_m)


def main():
    print("=" * 70)
    print("FAA radar exclusion zones (NEXRAD WSR-88D) ingestion")
    print("=" * 70)
    print(f"  fetching {NEXRAD_URL} ...")
    data = _fetch_with_retry(NEXRAD_URL)
    features = data.get('features', [])
    print(f"  total features in response: {len(features)}")

    # Filter to WSR-88D only.
    wsr = [f for f in features if f['properties'].get('stationType') == 'WSR-88D']
    print(f"  WSR-88D stations: {len(wsr)}  (TDWR/Profiler excluded)")
    assert 150 <= len(wsr) <= 180, \
        f"Unexpected station count: {len(wsr)} -- expected ~160 WSR-88D stations"

    print(f"\n  computing 3-mile buffers in metric projection...")
    fetched_at = pd.Timestamp.utcnow()
    records = []
    for f in wsr:
        props = f['properties']
        coords = f['geometry']['coordinates']
        lon, lat = coords[0], coords[1]
        records.append({
            'station_id':      props['id'],
            'station_name':    props['name'],
            'station_lon':     lon,
            'station_lat':     lat,
            'buffer_radius_m': BUFFER_RADIUS_M,
            'geometry':        _compute_buffer_polygon(lon, lat, BUFFER_RADIUS_M),
            'fetched_at':      fetched_at,
        })

    gdf = gpd.GeoDataFrame(records, geometry='geometry', crs='EPSG:4326')

    # Validation.
    assert (gdf.geom_type == 'Polygon').all(), \
        f"Non-polygon geometries: {gdf.geom_type.value_counts().to_dict()}"
    invalid = (~gdf.geometry.is_valid).sum()
    assert invalid == 0, f"{invalid} invalid geometries"
    print(f"  geometry assertions passed: {len(gdf)} valid Polygon buffers")

    # Sanity-check buffer area. A 3-mile circle has area pi * r^2
    # = pi * 4828^2 m^2 = 73.23 km^2.
    # Convert from degree^2 to km^2 using local scaling per centroid latitude.
    import math
    expected_km2 = math.pi * (BUFFER_RADIUS_M / 1000.0) ** 2
    areas_km2 = []
    for _, r in gdf.iterrows():
        lat = r['station_lat']
        deg2_to_km2 = (111.32 * math.cos(math.radians(lat))) * 110.57
        areas_km2.append(r.geometry.area * deg2_to_km2)
    areas_km2 = pd.Series(areas_km2)
    print(f"  buffer area: min={areas_km2.min():.1f} km2  max={areas_km2.max():.1f} km2  "
          f"mean={areas_km2.mean():.1f} km2  expected={expected_km2:.1f} km2")
    assert abs(areas_km2.mean() - expected_km2) < 2.0, \
        f"Mean buffer area {areas_km2.mean():.1f} km2 deviates >2 from expected {expected_km2:.1f}"
    print(f"  buffer-area sanity check passed (within 2 km2 of expected)")

    print(f"\n  station coverage by approximate region:")
    print(f"    AK (lat>=50, lon<-130):  {((gdf['station_lat']>=50) & (gdf['station_lon']<-130)).sum()}")
    print(f"    HI (lat<23, lon<-154):   {((gdf['station_lat']<23)  & (gdf['station_lon']<-154)).sum()}")
    print(f"    PR/VI (lon>-70):         {(gdf['station_lon']>-70).sum()}")
    print(f"    CONUS rest:              {len(gdf) - ((gdf['station_lat']>=50) & (gdf['station_lon']<-130)).sum() - ((gdf['station_lat']<23) & (gdf['station_lon']<-154)).sum() - (gdf['station_lon']>-70).sum()}")

    print(f"\n  fetched_at: {fetched_at.isoformat()}")
    gdf.to_parquet(OUT_PARQUET, index=False)
    print(f"  wrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e3:.0f} KB)")

    # ── Spot checks ──────────────────────────────────────────────────────────
    print(f"\n  spot-check: known points inside or outside radar buffers")
    INSIDE = {
        "KBMX Birmingham AL (at station)":  (-86.7700, 33.1723),
        "KLOT Chicago IL (at station)":     (-88.0844, 41.6044),
        "KBRO Brownsville TX (at station)": (-97.4189, 25.9160),
    }
    OUTSIDE = {
        "Rural Kansas":         (-98.5000, 38.5000),
        "Imperial Valley CA":   (-115.5697, 32.8371),
        "West Texas plains":    (-101.9000, 33.5000),
    }
    for label, (lon, lat) in INSIDE.items():
        pt = Point(lon, lat)
        hits = gdf[gdf.geometry.contains(pt)]
        verdict = 'OK INSIDE ' if len(hits) > 0 else '!! MISS   '
        station = hits.iloc[0]['station_id'] if len(hits) > 0 else '-'
        print(f"    {verdict} {label:<40} -> station={station}")
    for label, (lon, lat) in OUTSIDE.items():
        pt = Point(lon, lat)
        hits = gdf[gdf.geometry.contains(pt)]
        verdict = 'OK OUTSIDE' if len(hits) == 0 else '!! INSIDE '
        station = hits.iloc[0]['station_id'] if len(hits) > 0 else '-'
        print(f"    {verdict} {label:<40} -> station={station}")

    return gdf


if __name__ == '__main__':
    main()
