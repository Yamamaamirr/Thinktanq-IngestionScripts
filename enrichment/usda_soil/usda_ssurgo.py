"""USDA SSURGO soil enrichment.

Given a parcel polygon, ask the USDA NRCS Soil Data Access (SDA) endpoint for
the soil map units intersecting the parcel, then derive:
  1. Soil bearing band (Good / Moderate / Poor) — feeds dimension 2 multiplier
  2. Whether the parcel triggers the expansive-soil-majority hard disqualifier
     (>50% of parcel covered by expansive clay or peat per PRD section 7)

No local SSURGO mirror is stored — every parcel is resolved on demand, same
pattern as fema_floodhazard.py and usfws_wetlands.py.

Endpoint quirks worth knowing:
  * URL must be /Tabular/post.rest with capital T.
  * Format must be JSON+COLUMNNAME (singular). Format=JSON+COLUMNNAMES is
    silently rejected with an unhelpful XML error.
  * shrinkswellpotential does NOT exist on the component table (despite some
    docs suggesting it does). The real signal is chorizon.lep_r — Linear
    Extensibility Percent. NRCS thresholds: <3 = low, 3-6 = moderate,
    6-9 = high, >=9 = very high. >=6 is treated as expansive here.
  * DECLARE statements aren't allowed; geometry must be inlined as
    geometry::STGeomFromText(...).
  * Peat is detected via chorizon.om_r (organic matter %) >=20 in the top
    60cm OR component drainage class = 'Very poorly drained'.

Coverage % is computed in EPSG:4326 planar area. Same caveat as the wetlands
script — the ratio is exact for any realistic parcel size in the US since
intersection and parcel share the same local latitude band.
"""
import json
import time

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError

SDA_URL = 'https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest'

# NRCS LEP threshold — Linear Extensibility Percent at which a soil is
# considered to have high or very high shrink-swell potential.
LEP_EXPANSIVE_THRESHOLD = 6.0
LEP_LOW_THRESHOLD = 3.0          # below this = "low" shrink-swell
PEAT_OM_PCT_THRESHOLD = 20.0     # organic matter % indicating peat/muck
COVERAGE_DISQ_THRESHOLD = 50.0   # % of parcel disqualifying-soil coverage

RETRY_DELAYS = [2, 5, 15, 45, 120, 300]
NETWORK_ERRORS = (ConnectionError, Timeout, ChunkedEncodingError, ProtocolError)


def _post_with_retry(query, timeout=120):
    last_exc = None
    payload = {'query': query, 'format': 'JSON+COLUMNNAME'}
    for attempt, delay in enumerate(RETRY_DELAYS, 1):
        try:
            r = requests.post(SDA_URL, json=payload, timeout=timeout)
            if r.status_code != 200:
                raise RuntimeError(f"SDA HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        except NETWORK_ERRORS as e:
            last_exc = e
            print(f"  Network error (attempt {attempt}): {type(e).__name__} — retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"SDA: max retries exceeded ({last_exc!r})")


def _rings_to_wkt(rings):
    """Convert Esri-style rings to a WKT POLYGON string for SDA.

    WKT polygon syntax: POLYGON((x y, x y, ...), (hole), ...). Each ring is
    wrapped in its own parens, then the whole list is wrapped in an outer
    set of parens — single rings still need both.
    """
    ring_strs = [
        "(" + ", ".join(f"{lon} {lat}" for lon, lat in ring) + ")"
        for ring in rings
    ]
    return "POLYGON(" + ", ".join(ring_strs) + ")"


def _classify_bearing_band(component_signals):
    """Pick a single bearing band based on the dominant-component signal mix.

    component_signals is a list of dicts: {pct_of_parcel, drainage, lep, om_pct}.
    """
    if not component_signals:
        return 'Good'  # no SSURGO data — assume favorable rather than block

    # Weighted by parcel coverage of the component
    total_pct = sum(c['pct_of_parcel'] for c in component_signals)
    if total_pct == 0:
        return 'Good'

    poor_pct = sum(c['pct_of_parcel'] for c in component_signals if _is_poor(c))
    good_pct = sum(c['pct_of_parcel'] for c in component_signals if _is_good(c))

    if poor_pct >= total_pct * 0.5:
        return 'Poor'
    if good_pct >= total_pct * 0.7:
        return 'Good'
    return 'Moderate'


def _is_poor(c):
    if c.get('lep') is not None and c['lep'] >= LEP_EXPANSIVE_THRESHOLD:
        return True
    if c.get('om_pct') is not None and c['om_pct'] >= PEAT_OM_PCT_THRESHOLD:
        return True
    if c.get('drainage') in ('Poorly drained', 'Very poorly drained'):
        return True
    return False


GOOD_DRAINAGE = ('Well drained', 'Excessively drained', 'Somewhat excessively drained')


def _is_good(c):
    if c.get('drainage') not in GOOD_DRAINAGE:
        return False
    if c.get('lep') is not None and c['lep'] >= LEP_LOW_THRESHOLD:
        return False
    if c.get('om_pct') is not None and c['om_pct'] >= PEAT_OM_PCT_THRESHOLD:
        return False
    return True


def _is_expansive_or_peat(c):
    if c.get('lep') is not None and c['lep'] >= LEP_EXPANSIVE_THRESHOLD:
        return True
    if c.get('om_pct') is not None and c['om_pct'] >= PEAT_OM_PCT_THRESHOLD:
        return True
    return False


def lookup_parcel_soil(rings, input_srid=4326):
    """Return (disqualified, soil_bearing_band, drainage_class, shrink_swell).

    rings              Esri-style polygon rings: [[[lon, lat], ...]].
    input_srid         EPSG code of the rings' CRS. Default 4326. SDA's spatial
                       function requires WGS84, so other SRIDs aren't supported
                       here yet — caller must pre-project if needed.

    disqualified       True if expansive-clay / peat soil covers >50% of parcel.
    soil_bearing_band  'Good' / 'Moderate' / 'Poor' — feeds the dimension 2
                       scoring multiplier.
    drainage_class     Drainage class of the dominant component, kept for audit.
    shrink_swell       Categorical band derived from LEP — 'Low' / 'Moderate' /
                       'High' / 'Very High' / None (no horizon data).
    """
    if input_srid != 4326:
        raise NotImplementedError("SDA spatial functions require WGS84 (EPSG:4326).")

    wkt = _rings_to_wkt(rings)
    geom_expr = f"geometry::STGeomFromText('{wkt}', 4326)"

    # One-shot query: per-mukey overlap area + dominant component + LEP + OM%.
    # We scope chorizon to the top 152cm (~5ft) — engineering bearing depth.
    query = f"""
SELECT
    areas.mukey,
    areas.overlap_deg2,
    areas.parcel_deg2,
    c.compname,
    c.comppct_r,
    c.drainagecl,
    c.compkind,
    c.taxorder,
    (SELECT MAX(ch.lep_r) FROM chorizon ch
     WHERE ch.cokey = c.cokey AND ch.hzdept_r < 152 AND ch.lep_r IS NOT NULL) AS max_lep,
    (SELECT MAX(ch.om_r) FROM chorizon ch
     WHERE ch.cokey = c.cokey AND ch.hzdept_r < 60  AND ch.om_r  IS NOT NULL) AS max_om_pct
FROM (
    SELECT mp.mukey,
           SUM(mp.mupolygongeo.STIntersection({geom_expr}).STArea()) AS overlap_deg2,
           {geom_expr}.STArea() AS parcel_deg2
    FROM mupolygon mp
    WHERE mp.mupolygongeo.STIntersects({geom_expr}) = 1
    GROUP BY mp.mukey
) areas
INNER JOIN component c ON c.mukey = areas.mukey AND c.majcompflag = 'Yes'
ORDER BY areas.overlap_deg2 DESC
"""
    data = _post_with_retry(query)
    rows = data.get('Table', [])
    if not rows or len(rows) < 2:
        return False, 'Good', None, None

    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}

    component_signals = []
    parcel_deg2 = None
    for row in rows[1:]:
        overlap = float(row[idx['overlap_deg2']])
        parcel = float(row[idx['parcel_deg2']])
        parcel_deg2 = parcel
        if parcel <= 0:
            continue
        mukey_pct_of_parcel = (overlap / parcel) * 100.0
        comppct = float(row[idx['comppct_r']]) if row[idx['comppct_r']] is not None else 100.0
        component_signals.append({
            'compname': row[idx['compname']],
            'pct_of_parcel': mukey_pct_of_parcel * (comppct / 100.0),
            'drainage': row[idx['drainagecl']],
            'lep': float(row[idx['max_lep']]) if row[idx['max_lep']] is not None else None,
            'om_pct': float(row[idx['max_om_pct']]) if row[idx['max_om_pct']] is not None else None,
            'taxorder': row[idx['taxorder']],
        })

    band = _classify_bearing_band(component_signals)

    # Hard disqualifier: >50% of parcel covered by expansive or peat soil.
    bad_pct = sum(c['pct_of_parcel'] for c in component_signals if _is_expansive_or_peat(c))
    disqualified = bad_pct > COVERAGE_DISQ_THRESHOLD

    dominant = max(component_signals, key=lambda c: c['pct_of_parcel']) if component_signals else {}
    drainage_class = dominant.get('drainage')

    lep = dominant.get('lep')
    if lep is None:
        shrink_swell = None
    elif lep < LEP_LOW_THRESHOLD:
        shrink_swell = 'Low'
    elif lep < LEP_EXPANSIVE_THRESHOLD:
        shrink_swell = 'Moderate'
    elif lep < 9:
        shrink_swell = 'High'
    else:
        shrink_swell = 'Very High'

    return disqualified, band, drainage_class, shrink_swell


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
        "TX Waxahachie Blackland (Vertisol)": (-96.8400, 32.3900),  # DROP/Poor
        "Okefenokee GA peat":                 ( -82.4000, 30.7500),  # DROP/Poor
        "NE Sandhills":                       (-101.5000, 41.8000),  # keep/Good
        "VA Piedmont upland":                 ( -78.8000, 37.1000),  # keep/Good
        "CA Tulare Co ag":                    (-119.3500, 36.2000),  # keep/Moderate
        "Lubbock TX plains ag":               (-101.7500, 33.6000),  # keep/Moderate
    }

    print(f"{'verdict':<5} | {'band':<10} | {'shrink-swell':<12} | {'drainage':<28} | location")
    print("-" * 100)
    for name, (lon, lat) in samples.items():
        disq, band, drainage, ss = lookup_parcel_soil(parcel_square(lon, lat))
        verdict = "DROP" if disq else "keep"
        print(f"{verdict:<5} | {band:<10} | {str(ss):<12} | {str(drainage):<28} | {name}")
