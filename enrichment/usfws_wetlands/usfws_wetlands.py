"""USFWS National Wetlands Inventory enrichment.

Given a parcel polygon, ask the USFWS NWI service for intersecting wetland
polygons, then compute:
  1. Fraction of the parcel covered by any wetland polygon
  2. Whether the parcel triggers the WOTUS hard disqualifier
     (forested wetlands present OR jurisdictional coverage > 10%)
  3. A scoring band for the parcel's wetland exposure

No local wetland dataset is stored — every parcel is resolved on demand,
same pattern as fema_floodhazard.py.

Field-name quirk: the underlying service joins Wetlands against a codes
table, so outFields must use the table-qualified names (Wetlands.ATTRIBUTE,
Wetlands.WETLAND_TYPE). Requesting unprefixed field names silently returns
zero features.

Coverage percentage is computed in EPSG:4326 planar area. Because the
intersection and the parcel are in the same local latitude band, the ratio
is accurate to well under 1% for realistic (50+ acre) parcels.
"""
import json
import time

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

NWI_URL = (
    "https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/"
    "rest/services/Wetlands/MapServer/0/query"
)

FIELD_ATTRIBUTE = 'Wetlands.ATTRIBUTE'
FIELD_WETLAND_TYPE = 'Wetlands.WETLAND_TYPE'

# Cowardin-code prefixes.
# PFO = palustrine forested — always a hard disqualifier per PRD.
# PEM, PSS, PAB, E* = jurisdictional wetland classes; disqualifying only if
# combined parcel coverage exceeds COVERAGE_THRESHOLD_PCT.
FORESTED_PREFIXES = ('PFO',)
JURISDICTIONAL_PREFIXES = ('PFO', 'PEM', 'PSS', 'PAB', 'E')

COVERAGE_THRESHOLD_PCT = 10.0

MAX_RECORD_COUNT = 1500  # service maxRecordCount; dense swamps may approach
RETRY_DELAYS = [2, 5, 15, 45, 120, 300]
NETWORK_ERRORS = (ConnectionError, Timeout, ChunkedEncodingError, ProtocolError)


def _request_with_retry(params):
    last_exc = None
    for attempt, delay in enumerate(RETRY_DELAYS, 1):
        try:
            r = requests.get(NWI_URL, params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except NETWORK_ERRORS as e:
            last_exc = e
            print(f"  Network error (attempt {attempt}): {type(e).__name__} — retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"NWI API: max retries exceeded ({last_exc!r})")


def _classify_band(coverage_pct, attributes):
    if not attributes:
        return 'none'
    has_forested = any(a.startswith(FORESTED_PREFIXES) for a in attributes)
    has_juris = any(a.startswith(JURISDICTIONAL_PREFIXES) for a in attributes)
    if has_forested or (has_juris and coverage_pct > COVERAGE_THRESHOLD_PCT):
        return 'jurisdictional_permit_required'
    if has_juris:
        return 'non_jurisdictional'
    return 'hydric_only'


def lookup_parcel_wetlands(rings, input_srid=4326, buffer_m=50):
    """Return (disqualified, coverage_pct, wetland_band, wetland_types).

    rings           Esri-style polygon rings: [[[lon, lat], ...]]. Outer ring
                    first; inner rings (holes), if any, follow.
    input_srid      EPSG code of the rings' CRS. Default 4326 (WGS84 lon/lat).
    buffer_m        Server-side buffer in metres applied before the intersects
                    check. Default 50m covers minor geometric noise without
                    false-dragging distant wetlands into the result.

    disqualified    True if forested wetlands are present OR jurisdictional
                    coverage exceeds 10% of the parcel.
    coverage_pct    Float 0.0-100.0, fraction of parcel covered by any
                    wetland polygon (union of all returned features).
    wetland_band    One of 'none', 'hydric_only', 'non_jurisdictional',
                    'jurisdictional_permit_required' — feeds the scoring
                    multiplier per PRD section 8.3.
    wetland_types   Sorted list of distinct WETLAND_TYPE strings returned,
                    kept for audit trail.
    """
    params = {
        'geometry': json.dumps({
            'rings': rings,
            'spatialReference': {'wkid': input_srid},
        }),
        'geometryType': 'esriGeometryPolygon',
        'inSR': input_srid,
        'spatialRel': 'esriSpatialRelIntersects',
        'outFields': f'{FIELD_ATTRIBUTE},{FIELD_WETLAND_TYPE}',
        'returnGeometry': 'true',
        'outSR': input_srid,
        'f': 'geojson',
        'resultRecordCount': MAX_RECORD_COUNT,
    }
    if buffer_m:
        params['distance'] = buffer_m
        params['units'] = 'esriSRUnit_Meter'

    data = _request_with_retry(params)
    features = data.get('features', [])
    if not features:
        return False, 0.0, 'none', []

    attributes = sorted({
        p.get(FIELD_ATTRIBUTE) for p in (f.get('properties') or {} for f in features)
        if p.get(FIELD_ATTRIBUTE)
    })
    wetland_types = sorted({
        p.get(FIELD_WETLAND_TYPE) for p in (f.get('properties') or {} for f in features)
        if p.get(FIELD_WETLAND_TYPE)
    })

    outer = rings[0]
    holes = rings[1:] if len(rings) > 1 else None
    parcel_geom = Polygon(outer, holes)
    if parcel_geom.area <= 0:
        return False, 0.0, 'none', wetland_types

    wetland_geoms = [shape(f['geometry']) for f in features if f.get('geometry')]
    if not wetland_geoms:
        return False, 0.0, 'none', wetland_types

    wetland_union = unary_union(wetland_geoms)
    intersection = parcel_geom.intersection(wetland_union)
    coverage_pct = min(100.0, (intersection.area / parcel_geom.area) * 100.0)

    band = _classify_band(coverage_pct, attributes)
    disqualified = band == 'jurisdictional_permit_required'

    return disqualified, coverage_pct, band, wetland_types


if __name__ == '__main__':
    import math

    def parcel_square(lon, lat, side_m=450):
        """~50-acre (450m x 450m) square parcel in EPSG:4326."""
        dy = (side_m / 2) / 111_000.0
        dx = (side_m / 2) / (111_000.0 * max(0.1, math.cos(math.radians(lat))))
        return [[
            [lon - dx, lat - dy], [lon + dx, lat - dy],
            [lon + dx, lat + dy], [lon - dx, lat + dy],
            [lon - dx, lat - dy],
        ]]

    samples = {
        "Okefenokee Swamp GA":         (-82.4000, 30.7500),
        "Everglades FL":               (-80.8500, 25.4500),
        "Great Dismal Swamp VA/NC":    (-76.5000, 36.6000),
        "Prairie Pothole ND":          (-98.9200, 47.2700),
        "Minto Flats AK":             (-148.8000, 64.8500),
        "Lubbock TX upland (control)":(-101.9000, 33.5000),
        "Phoenix AZ desert (control)":(-112.0740, 33.4480),
        "VA Piedmont upland":          (-78.8000, 37.1000),
    }

    print(f"{'verdict':<5} | {'band':<32} | {'cov%':>6} | {'types':<70} | location")
    print("-" * 140)
    for name, (lon, lat) in samples.items():
        disq, pct, band, types = lookup_parcel_wetlands(parcel_square(lon, lat))
        verdict = "DROP" if disq else "keep"
        types_s = ", ".join(t.replace("Freshwater ", "").replace(" Wetland", "") for t in types) or "-"
        if len(types_s) > 68:
            types_s = types_s[:65] + "..."
        print(f"{verdict:<5} | {band:<32} | {pct:>5.1f}% | {types_s:<70} | {name}")
