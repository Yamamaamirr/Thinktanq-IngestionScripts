"""USGS 3DEP slope/elevation enrichment.

Given a parcel polygon, sample USGS 3DEP elevation values on a grid covering
the parcel, compute slope per cell from elevation gradients, and return:
  - avg_slope_pct      mean slope across in-parcel cells
  - slope_pct_p70      70th percentile slope (drives the PRD hard disqualifier)
  - max_slope_pct      maximum slope across in-parcel cells
  - slope_band         Flat / Gentle / Hilly / Very_Hilly per PRD section 8.3
  - disqualified       True if slope_pct_p70 > 8% (per PRD section 7 item 3,
                       subject to analyst geotechnical-mitigation review)

No local raster is stored -- every parcel is resolved on demand, same pattern
as fema_floodhazard.py / usfws_wetlands.py / usda_ssurgo.py.

Endpoint quirks:
  * The getSamples endpoint returns one sample per matched raster per input
    point (1m / 10m / 30m mosaics overlap), so 3 input points can yield 9
    output samples. Use returnFirstValueOnly=true to get a single value per
    point at the highest-available resolution.
  * Sample 'value' field arrives as a string -- coerce to float.
  * Sample 'locationId' indexes back into the input point order.
  * Practical batch limit observed: 500 points in one request; ~6s response.
    For a typical parcel this is well under the limit.

Slope calculation:
  Generate a regular lat/lon grid covering the parcel bounding box. Query all
  grid points (sampling outside the polygon is cheap; filtering happens after).
  Reshape into a 2D array, compute gradients in metres (longitude spacing
  varies with latitude, so x and y spacings differ). Mask the slope grid by
  the parcel polygon and aggregate only over in-parcel cells.
"""
import json
import math
import time

import numpy as np
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from shapely.geometry import Point, Polygon
from urllib3.exceptions import ProtocolError

URL = ("https://elevation.nationalmap.gov/arcgis/rest/services/"
       "3DEPElevation/ImageServer/getSamples")

DISQUALIFIER_P70_THRESHOLD = 8.0  # PRD section 7 item 3

DEFAULT_GRID_SPACING_M = 100.0
MAX_POINTS_PER_REQUEST = 400      # below the 500 ceiling, leaves headroom

RETRY_DELAYS = [2, 5, 15, 45, 120]
NETWORK_ERRORS = (ConnectionError, Timeout, ChunkedEncodingError, ProtocolError)
RETRY_HTTP_STATUSES = {500, 502, 503, 504}  # transient server-side issues


def _request_with_retry(points, timeout=60):
    params = {
        'geometry': json.dumps({
            'points': points,
            'spatialReference': {'wkid': 4326},
        }),
        'geometryType': 'esriGeometryMultipoint',
        'returnFirstValueOnly': 'true',
        'interpolation': 'RSP_BilinearInterpolation',
        'f': 'json',
    }
    last_exc = None
    for attempt, delay in enumerate(RETRY_DELAYS, 1):
        try:
            r = requests.post(URL, data=params, timeout=timeout)
            if r.status_code in RETRY_HTTP_STATUSES:
                last_exc = RuntimeError(f"HTTP {r.status_code}")
                print(f"  Server error (attempt {attempt}): HTTP {r.status_code} -- retry in {delay}s")
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.json().get('samples', [])
        except NETWORK_ERRORS as e:
            last_exc = e
            print(f"  Network error (attempt {attempt}): {type(e).__name__} -- retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"3DEP getSamples: max retries exceeded ({last_exc!r})")


def _fetch_elevations(points):
    """Query elevations for a list of [lon, lat] points. Returns parallel list of floats."""
    out = [None] * len(points)
    for i in range(0, len(points), MAX_POINTS_PER_REQUEST):
        batch = points[i:i + MAX_POINTS_PER_REQUEST]
        samples = _request_with_retry(batch)
        for s in samples:
            loc_id = s.get('locationId')
            val = s.get('value')
            if loc_id is None or val is None:
                continue
            try:
                out[i + loc_id] = float(val)
            except (TypeError, ValueError):
                pass
    return out


def _assign_band(avg_slope_pct):
    if avg_slope_pct is None:
        return 'Flat'  # no data -> assume favorable
    if avg_slope_pct < 2.0:
        return 'Flat'
    if avg_slope_pct < 5.0:
        return 'Gentle'
    if avg_slope_pct < 8.0:
        return 'Hilly'
    return 'Very_Hilly'


def lookup_parcel_slope(rings, input_srid=4326, grid_spacing_m=DEFAULT_GRID_SPACING_M):
    """Return (disqualified, avg_slope_pct, slope_pct_p70, max_slope_pct, slope_band).

    rings           Esri-style polygon rings: [[[lon, lat], ...]].
    input_srid      EPSG code; only 4326 supported (3DEP requires WGS84).
    grid_spacing_m  Spacing between sample points in metres. Default 100m.
                    Smaller spacing = more API points = higher precision.
    """
    if input_srid != 4326:
        raise NotImplementedError("3DEP getSamples requires EPSG:4326.")

    outer = rings[0]
    holes = rings[1:] if len(rings) > 1 else None
    parcel = Polygon(outer, holes)
    if parcel.area <= 0:
        return False, 0.0, 0.0, 0.0, 'Flat'

    centroid = parcel.centroid
    lat_rad = math.radians(centroid.y)
    spacing_lat_deg = grid_spacing_m / 111_000.0
    spacing_lon_deg = grid_spacing_m / (111_000.0 * max(0.1, math.cos(lat_rad)))
    spacing_y_m = grid_spacing_m
    spacing_x_m = grid_spacing_m  # by construction the grid is uniform in metres

    minx, miny, maxx, maxy = parcel.bounds
    # Pad by half a cell so the grid covers the bounds with at least 2 cells.
    xs = np.arange(minx - spacing_lon_deg / 2, maxx + spacing_lon_deg, spacing_lon_deg)
    ys = np.arange(miny - spacing_lat_deg / 2, maxy + spacing_lat_deg, spacing_lat_deg)
    if len(xs) < 2:
        xs = np.array([minx, maxx])
    if len(ys) < 2:
        ys = np.array([miny, maxy])

    lon_grid, lat_grid = np.meshgrid(xs, ys)
    flat_points = [[float(lon), float(lat)]
                   for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())]

    elev_flat = _fetch_elevations(flat_points)
    elev_arr = np.array([(e if e is not None else np.nan) for e in elev_flat],
                        dtype=float).reshape(lon_grid.shape)

    if np.isnan(elev_arr).all():
        return False, 0.0, 0.0, 0.0, 'Flat'

    # Compute slope (% = rise/run * 100) via gradients in metres.
    # nan-safe: gradients of nan stay nan; later we drop them in stats.
    dz_dx = np.gradient(elev_arr, spacing_x_m, axis=1)
    dz_dy = np.gradient(elev_arr, spacing_y_m, axis=0)
    slope_pct_grid = np.sqrt(dz_dx ** 2 + dz_dy ** 2) * 100.0

    # Mask: keep only cells whose centre falls inside the parcel polygon.
    in_parcel = np.zeros_like(slope_pct_grid, dtype=bool)
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            if parcel.contains(Point(lon_grid[i, j], lat_grid[i, j])):
                in_parcel[i, j] = True
    in_parcel_slopes = slope_pct_grid[in_parcel & ~np.isnan(slope_pct_grid)]

    if len(in_parcel_slopes) == 0:
        # Fall back to whole-grid slopes (very small parcels with no interior cells).
        in_parcel_slopes = slope_pct_grid[~np.isnan(slope_pct_grid)]

    if len(in_parcel_slopes) == 0:
        return False, 0.0, 0.0, 0.0, 'Flat'

    avg_slope = float(np.mean(in_parcel_slopes))
    p70_slope = float(np.percentile(in_parcel_slopes, 70))
    max_slope = float(np.max(in_parcel_slopes))
    band = _assign_band(avg_slope)
    disqualified = p70_slope > DISQUALIFIER_P70_THRESHOLD
    return disqualified, avg_slope, p70_slope, max_slope, band


if __name__ == '__main__':
    def parcel_square(lon, lat, side_m=450):
        dy = (side_m / 2) / 111_000.0
        dx = (side_m / 2) / (111_000.0 * max(0.1, math.cos(math.radians(lat))))
        return [[
            [lon - dx, lat - dy], [lon + dx, lat - dy],
            [lon + dx, lat + dy], [lon - dx, lat + dy],
            [lon - dx, lat - dy],
        ]]

    samples = {
        # Flat agricultural land
        "Imperial Valley CA farmland":  (-115.5697, 32.8371),
        "Lubbock TX High Plains":       (-101.9000, 33.5000),
        "Iowa corn belt":                (-93.6000, 42.0000),
        # Gentle rolling
        "VA Piedmont upland":            (-78.8000, 37.1000),
        "Willamette Valley OR":         (-123.0000, 44.5000),
        # Hilly
        "Appalachian foothills NC":      (-82.5000, 35.5000),
        "Sierra Nevada foothills CA":   (-120.5000, 38.5000),
        # Mountainous
        "Rocky Mountains CO":           (-105.5000, 39.5000),
        "Cascades WA":                  (-121.5000, 47.5000),
    }

    print(f"{'location':<32} {'verdict':<5} {'band':<11} {'avg%':>6} {'p70%':>6} {'max%':>6}")
    print("-" * 75)
    for name, (lon, lat) in samples.items():
        rings = parcel_square(lon, lat)
        disq, avg, p70, mx, band = lookup_parcel_slope(rings)
        verdict = "DROP" if disq else "keep"
        print(f"{name:<32} {verdict:<5} {band:<11} {avg:>5.2f}  {p70:>5.2f}  {mx:>5.2f}")
