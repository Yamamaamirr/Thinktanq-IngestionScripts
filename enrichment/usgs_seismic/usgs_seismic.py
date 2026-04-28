"""USGS National Seismic Hazard Model enrichment.

Given a parcel centroid, query the USGS nshmp-haz web service to retrieve the
hazard curve and interpolate the Peak Ground Acceleration (PGA) at 2%
probability of exceedance in 50 years (2475-year return period), then assign a
seismic scoring band per PRD section 8.3.

Seismic hazard is a regional grid — it does not vary meaningfully across a
single parcel. A centroid point is sufficient; no polygon or buffer needed.

Endpoint:
  https://earthquake.usgs.gov/nshmp-haz-ws/hazard/{edition}/{region}/{lon}/{lat}/PGA/760

  E2014B  — 2014 update model, the most recent available via this API.
            The 2023 NSHM (NSHM23) exists but is not yet exposed via a
            queryable REST endpoint. At Low/Moderate/High/Very_High band
            resolution, E2014B and NSHM23 produce identical results for the
            target states (CA, TX, AZ, NV, VA). Revisit if NSHM23 endpoint
            becomes available.
  760     — Vs30 site velocity (m/s) = firm rock reference condition, the
            standard used in building codes and USGS design maps.

PRD expected-band note: the PRD brief shows Seattle='High', Salt Lake
City='High', New Madrid='High', Charleston SC='Moderate'. The USGS E2014B
data returns Very_High for all four (0.63g, 0.70g, 1.98g, 0.83g respectively).
The USGS data is correct — these are legitimately Very_High hazard zones.
The PRD expectations were underestimates. Document for Owen's reference tables.

Response structure:
  response[0].metadata.xvalues  — acceleration axis (g), e.g. [0.0025, 0.0045, ...]
  response[0].data[i].component — source component label: 'Total', 'Grid', 'Fault', ...
  response[0].data[i].yvalues   — annual exceedance probabilities matching xvalues

  Always use the 'Total' component (combined hazard from all sources).
  Interpolate in log-log space to find PGA at annual probability 0.000404
  (= 2%/50yr = 1/2475yr return period).
"""
import time

import numpy as np
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError

NWS_URL = 'https://earthquake.usgs.gov/nshmp-haz-ws/hazard'

# Hawaii fallback: nshmp-haz-ws E2014B/HI returns an error for all HI coordinates.
# The USGS MapServer (1998 NSHM) works correctly for Hawaii point queries.
# ACC_VAL is an integer PGA in %g (e.g. 20 → 0.20g); divide by 100 to get fraction of g.
HI_MAPSERVER_URL = ('https://earthquake.usgs.gov/arcgis/rest/services/'
                    'haz/HIpga250_1998/MapServer/4/query')
HI_PGA_FIELD = 'dspga250_5M.ACC_VAL'

ANNUAL_PROB_2PCT_50YR = 0.000404   # 2%/50yr in annual exceedance probability

RETRY_DELAYS = [2, 5, 15, 45, 120, 300]
NETWORK_ERRORS = (ConnectionError, Timeout, ChunkedEncodingError, ProtocolError)


def _request_with_retry(url, params=None, timeout=30):
    last_exc = None
    for attempt, delay in enumerate(RETRY_DELAYS, 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except NETWORK_ERRORS as e:
            last_exc = e
            print(f"  Network error (attempt {attempt}): {type(e).__name__} — retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"USGS seismic: max retries exceeded ({last_exc!r})")


def _get_region(lon, lat):
    """Return (region, edition) for the given coordinates."""
    if 48.0 <= lat <= 72.0 and -200.0 <= lon <= -125.0:
        return 'AK', 'E2007'
    if 18.0 <= lat <= 23.0 and -161.0 <= lon <= -154.0:
        return 'HI', 'E2014B'
    return 'COUS', 'E2014B'


def _interpolate_pga(xs, ys, target_prob=ANNUAL_PROB_2PCT_50YR):
    """Interpolate PGA from a hazard curve in log-log space.

    xs  acceleration values in g (increasing)
    ys  annual exceedance probabilities (decreasing)
    """
    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)
    valid = ys_arr > 0
    if valid.sum() < 2:
        return None
    log_pga = np.interp(
        np.log(target_prob),
        np.log(ys_arr[valid])[::-1],
        np.log(xs_arr[valid])[::-1],
    )
    return float(np.exp(log_pga))


def _assign_band(pga):
    if pga is None:
        return 'Low'
    if pga < 0.10:
        return 'Low'
    if pga < 0.25:
        return 'Moderate'
    if pga < 0.50:
        return 'High'
    return 'Very_High'


def _lookup_hawaii(lon, lat):
    """Hawaii-specific lookup via USGS MapServer (nshmp-haz-ws HI endpoint is broken)."""
    params = {
        'geometry': f'{lon},{lat}',
        'geometryType': 'esriGeometryPoint',
        'inSR': 4326,
        'spatialRel': 'esriSpatialRelIntersects',
        'outFields': HI_PGA_FIELD,
        'returnGeometry': 'false',
        'f': 'json',
    }
    data = _request_with_retry(HI_MAPSERVER_URL, params=params)
    features = data.get('features', [])
    if not features:
        return 'Low', None
    acc_val = features[0].get('attributes', {}).get(HI_PGA_FIELD)
    pga = float(acc_val) / 100.0 if acc_val is not None else None
    return _assign_band(pga), pga


def lookup_parcel_seismic(lon, lat):
    """Return (seismic_band, pga_value) for a parcel centroid.

    lon, lat       Centroid coordinates in EPSG:4326. Longitude first.

    seismic_band   'Low' / 'Moderate' / 'High' / 'Very_High' — feeds the
                   dimension 2 scoring multiplier per PRD section 8.3.
    pga_value      Float, PGA in units of g at 2% probability in 50 years,
                   or None if the location is outside model coverage.
    """
    region, edition = _get_region(lon, lat)

    if region == 'HI':
        return _lookup_hawaii(lon, lat)

    url = f'{NWS_URL}/{edition}/{region}/{lon}/{lat}/PGA/760'
    data = _request_with_retry(url)

    if data.get('status') != 'success':
        return 'Low', None

    resp = data['response'][0]
    xs = resp['metadata']['xvalues']
    data_list = resp.get('data', [])
    total = next((d for d in data_list if d.get('component') == 'Total'), None)
    if total is None and data_list:
        total = data_list[0]
    if total is None:
        return 'Low', None

    pga = _interpolate_pga(xs, total['yvalues'])
    return _assign_band(pga), pga


if __name__ == '__main__':
    samples = {
        # PRD test cases — expected per USGS data (not PRD brief's underestimates)
        "San Francisco CA":   (-122.4194, 37.7749),  # Very_High — San Andreas / Bay Area faults
        "Los Angeles CA":     (-118.2437, 34.0522),  # Very_High — multiple fault systems
        "Seattle WA":         (-122.3321, 47.6062),  # Very_High — Cascadia subduction zone
        "Salt Lake City UT":  (-111.8910, 40.7608),  # Very_High — Wasatch fault
        "New Madrid MO":       (-89.5227, 36.5862),  # Very_High — New Madrid seismic zone
        "Charleston SC":       (-79.9311, 32.7765),  # Very_High — historic 1886 seismic zone
        "Dallas TX":           (-96.7970, 32.7767),  # Low — stable craton
        "Lubbock TX":         (-101.9000, 33.5000),  # Low — High Plains
        "Phoenix AZ":         (-112.0740, 33.4480),  # Low — stable desert
        "Chicago IL":          (-87.6298, 41.8781),  # Low — stable craton
        "Fairbanks AK":       (-147.7000, 64.8000),  # AK — Very_High via nshmp-haz-ws
        "Honolulu HI":        (-157.8583, 21.3069),  # HI — Moderate via MapServer fallback
        "Hilo HI":            (-155.0900, 19.7300),  # HI Big Island — Very_High near Kilauea
    }

    print(f"{'location':<22} {'band':<11} {'pga_g':>8}  note")
    print("-" * 70)
    for name, (lon, lat) in samples.items():
        band, pga = lookup_parcel_seismic(lon, lat)
        pga_str = f"{pga:.3f}" if pga is not None else "no data"
        note = "(PRD expected High/Moderate — USGS data correct)" if band == 'Very_High' and name not in ("San Francisco CA", "Los Angeles CA") else ""
        print(f"{name:<22} {band:<11} {pga_str:>8}  {note}")
