"""FEMA flood zone enrichment.

Given a parcel polygon, ask the FEMA ArcGIS FeatureServer whether it intersects
any disqualifying Special Flood Hazard Area (A, AE, AO, AH, V, VE). No local
flood zone dataset is stored — every parcel is resolved on demand.

Known limitation: the Esri "Reduced Set" mirror has spotty coverage on narrow
coastal barrier islands (parts of the Outer Banks, Jersey Shore, some Gulf
islands). Not a concern for this project because qualifying parcels are 50+
acre inland agricultural/undeveloped land — barrier islands never reach Stage 1.

Each lookup also returns a `source` tag so downstream analytics can tell the
two non-disqualifying cases apart:
  sfha_confirmed    — a disqualifying SFHA polygon was found (parcel drops)
  zone_x_confirmed  — FEMA affirmatively said "not in SFHA" (e.g. Zone X)
  no_data_returned  — no polygon returned, likely an unmapped rural county
                      (Nevada/Arizona/Montana/Wyoming rangeland, etc.)
Persist this as a column alongside the zone(s) so the analyst team can decide
whether any no_data_returned parcels in their pipeline need manual review.
"""
import json
import time

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError

FEMA_URL = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
    "USA_Flood_Hazard_Reduced_Set_gdb/FeatureServer/0/query"
)

DISQUALIFYING_ZONES = {'A', 'AE', 'AO', 'AH', 'V', 'VE'}
RETRY_DELAYS = [2, 5, 15, 45, 120, 300]
NETWORK_ERRORS = (ConnectionError, Timeout, ChunkedEncodingError, ProtocolError)


def _request_with_retry(params):
    last_exc = None
    for attempt, delay in enumerate(RETRY_DELAYS, 1):
        try:
            r = requests.get(FEMA_URL, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except NETWORK_ERRORS as e:
            last_exc = e
            print(f"  Network error (attempt {attempt}): {type(e).__name__} — retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"FEMA API: max retries exceeded ({last_exc!r})")


def lookup_parcel_flood_zone(rings, input_srid=4326, buffer_m=100):
    """Return (disqualified, zones, source) for a parcel polygon.

    rings        Esri-style polygon rings: [[[lon, lat], [lon, lat], ...]].
                 Outer ring first; first and last coord must match.
    input_srid   EPSG code of the parcel CRS. Default 4326 (WGS84 lon/lat).
    buffer_m     Expand the parcel by this many metres before the intersects
                 check. Compensates for generalized polygons in the Esri
                 "Reduced Set" mirror that can leave small coverage gaps along
                 narrow zones (barrier islands, streams). Set 0 to disable.

    disqualified True if any intersecting zone is in the SFHA disqualifier set.
    zones        List of intersecting FLD_ZONE strings, kept for audit/logging.
    source       One of 'sfha_confirmed', 'zone_x_confirmed', 'no_data_returned'
                 — persist alongside zones so analysts can separate true
                 non-SFHA results from unmapped-county results.
    """
    params = {
        'geometry': json.dumps({
            'rings': rings,
            'spatialReference': {'wkid': input_srid},
        }),
        'geometryType': 'esriGeometryPolygon',
        'inSR': input_srid,
        'spatialRel': 'esriSpatialRelIntersects',
        'outFields': 'FLD_ZONE',
        'returnGeometry': 'false',
        'f': 'json',
    }
    if buffer_m:
        params['distance'] = buffer_m
        params['units'] = 'esriSRUnit_Meter'
    data = _request_with_retry(params)
    zones = sorted({f['attributes'].get('FLD_ZONE') for f in data.get('features', [])} - {None})
    disqualified = any(z in DISQUALIFYING_ZONES for z in zones)
    if disqualified:
        source = 'sfha_confirmed'
    elif zones:
        source = 'zone_x_confirmed'
    else:
        source = 'no_data_returned'
    return disqualified, zones, source


if __name__ == '__main__':
    samples = {
        "Houston Buffalo Bayou": [[
            [-95.3820, 29.7610], [-95.3810, 29.7610],
            [-95.3810, 29.7620], [-95.3820, 29.7620],
            [-95.3820, 29.7610],
        ]],
        "NOLA Lower 9th Ward": [[
            [-89.9760, 29.9670], [-89.9750, 29.9670],
            [-89.9750, 29.9680], [-89.9760, 29.9680],
            [-89.9760, 29.9670],
        ]],
        "Denver downtown": [[
            [-104.9905, 39.7390], [-104.9895, 39.7390],
            [-104.9895, 39.7400], [-104.9905, 39.7400],
            [-104.9905, 39.7390],
        ]],
    }
    for name, rings in samples.items():
        disq, zones, source = lookup_parcel_flood_zone(rings)
        verdict = "DROP" if disq else "keep"
        print(f"{verdict:5} | {name:25} | {source:18} | zones={zones}")
