"""NOAA / USDA US Drought Monitor (USDM) ingestion.

Downloads the current weekly drought-classification shapefile from UNL's
Drought Monitor service and produces a parquet ready for PostGIS load into
the drought_zones table.

Source: https://droughtmonitor.unl.edu/data/shapefiles_m/USDM_current_M.zip
This URL always points to the most recent Thursday release -- no date parsing
needed. The DM field is an integer 0-4 for the standard USDM categories;
areas without a polygon are implicitly outside any drought zone.

  0 = D0  Abnormally Dry
  1 = D1  Moderate Drought
  2 = D2  Severe Drought
  3 = D3  Extreme Drought
  4 = D4  Exceptional Drought  <- triggers the preliminary disqualifier flag

Important — this script handles only the NOAA half of PRD section 7
disqualifier #6 (`d4_drought_water_restricted_no_closed_loop`). The full
disqualifier requires two additional conditions from Owen's analyst
reference tables that are not yet built:
  - water restriction status for the jurisdiction
  - closed-loop cooling commitment / recycled-water MOU flag
Until those reference tables exist, drought_disqualified=True is a
preliminary warning, NOT a confirmed disqualification. The scoring engine
must check all three conditions before excluding a parcel.

Output schema (matches drought_zones table, minus zone_id which Postgres
generates at INSERT via gen_random_uuid()):
  dm_level    INTEGER          0..4
  dm_label    TEXT             D0/D1/D2/D3/D4
  is_d4       BOOLEAN          convenience flag for the disqualifier index
  geometry    Polygon (4326)   exploded from MultiPolygon if necessary
  fetched_at  TIMESTAMPTZ      when this snapshot was downloaded
"""
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError

DROUGHT_URL = 'https://droughtmonitor.unl.edu/data/shapefiles_m/USDM_current_M.zip'

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
CACHE_ZIP = CACHE_DIR / 'USDM_current_M.zip'
OUT_PARQUET = Path(__file__).parent / 'drought_zones.parquet'

DM_LABELS = {0: 'D0', 1: 'D1', 2: 'D2', 3: 'D3', 4: 'D4'}

RETRY_DELAYS = [5, 15, 45, 120]
NETWORK_ERRORS = (ConnectionError, Timeout, ChunkedEncodingError, ProtocolError)


CACHE_TTL_DAYS = 7  # USDM releases every Thursday; expire cache after 7 days


def _download_with_retry(url, dest_path, label):
    # Expire stale cache — USDM is a weekly snapshot, so cached data older than
    # 7 days is out of date and must be refreshed.
    if dest_path.exists():
        age_days = (time.time() - dest_path.stat().st_mtime) / 86400
        if age_days <= CACHE_TTL_DAYS:
            size_mb = dest_path.stat().st_size / 1e6
            print(f"  [{label}] cached ({size_mb:.1f} MB, {age_days:.1f}d old) at {dest_path.name}")
            return
        print(f"  [{label}] cache is {age_days:.1f} days old (> {CACHE_TTL_DAYS}d TTL) -- refreshing")
        dest_path.unlink()

    last_exc = None
    tmp = dest_path.with_suffix(dest_path.suffix + '.part')
    for attempt, delay in enumerate(RETRY_DELAYS, 1):
        try:
            print(f"  [{label}] downloading from {url} ...")
            r = requests.get(url, timeout=180, stream=True)
            r.raise_for_status()
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            tmp.rename(dest_path)
            size_mb = dest_path.stat().st_size / 1e6
            print(f"  [{label}] downloaded {size_mb:.1f} MB -> {dest_path.name}")
            return
        except NETWORK_ERRORS as e:
            last_exc = e
            if tmp.exists():
                tmp.unlink()
            print(f"  [{label}] network error attempt {attempt} ({type(e).__name__}) -- retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"{label}: max retries exceeded ({last_exc!r})")


def _read_shapefile_from_zip(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        shp_name = next((n for n in zf.namelist() if n.endswith('.shp')), None)
        if shp_name is None:
            raise RuntimeError(f"No .shp file in {zip_path}")
    return gpd.read_file(f"zip://{zip_path}!{shp_name}")


def main():
    print("=" * 70)
    print("NOAA / USDA US Drought Monitor ingestion")
    print("=" * 70)
    _download_with_retry(DROUGHT_URL, CACHE_ZIP, 'usdm')

    gdf = _read_shapefile_from_zip(CACHE_ZIP)
    print(f"\n  raw rows: {len(gdf):,}")
    print(f"  columns: {list(gdf.columns)}")
    print(f"  CRS: {gdf.crs}")

    assert 'DM' in gdf.columns, "Missing DM field in USDM shapefile"
    assert gdf['DM'].between(0, 4).all(), \
        f"DM values out of expected 0-4 range: {sorted(gdf['DM'].unique())}"
    assert len(gdf) > 0, "Empty USDM shapefile"

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf = gdf[['DM', 'geometry']].rename(columns={'DM': 'dm_level'}).copy()
    gdf['dm_label'] = gdf['dm_level'].map(DM_LABELS)
    gdf['is_d4'] = gdf['dm_level'] == 4

    # Explode MultiPolygons to Polygons (PostGIS schema requires GEOGRAPHY(POLYGON)).
    n_before = len(gdf)
    n_multi = (gdf.geom_type == 'MultiPolygon').sum()
    print(f"\n  MultiPolygon features before explode: {n_multi}")
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    print(f"  after explode: {len(gdf):,} rows (+{len(gdf)-n_before} component polygons)")
    non_poly = gdf[gdf.geom_type != 'Polygon']
    assert len(non_poly) == 0, \
        f"Non-Polygon geometries remain: {non_poly.geom_type.value_counts().to_dict()}"
    print("  geometry assertion passed: all features are Polygon.")

    invalid = (~gdf.geometry.is_valid).sum()
    print(f"  invalid geometries: {invalid}")

    fetched_at = pd.Timestamp.utcnow()
    gdf['fetched_at'] = fetched_at

    print(f"\n  DM level distribution:")
    counts = gdf['dm_level'].value_counts().sort_index()
    for lvl, n in counts.items():
        print(f"    {DM_LABELS[lvl]} (level {lvl}): {n:,} polygons")
    print(f"\n  D4 polygons: {gdf['is_d4'].sum():,}")
    print(f"  fetched_at:  {fetched_at.isoformat()}")

    gdf.to_parquet(OUT_PARQUET, index=False)
    print(f"\n  wrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e6:.1f} MB)")

    # ── Spot checks: lookup DM at known points ───────────────────────────────
    print(f"\n  spot-check: drought level at well-known locations:")
    from shapely.geometry import Point
    test_points = {
        "Central Valley CA":  (-120.0, 36.5),
        "West Texas":         (-101.9, 33.5),
        "Kansas wheat belt":   (-98.0, 38.5),
        "Atlanta GA":          (-84.4, 33.7),
        "Pacific NW (WA)":    (-122.0, 47.5),
    }
    # Use the unexploded geometries (a parcel either contains a point or doesn't)
    for label, (lon, lat) in test_points.items():
        pt = Point(lon, lat)
        hits = gdf[gdf.geometry.contains(pt)]
        if len(hits) == 0:
            print(f"    {label:<22} ({lon:>7.2f}, {lat:>5.2f}) -> None (no drought polygon)")
        else:
            worst = hits.sort_values('dm_level', ascending=False).iloc[0]
            print(f"    {label:<22} ({lon:>7.2f}, {lat:>5.2f}) -> {worst['dm_label']} "
                  f"(level {worst['dm_level']}, {len(hits)} overlapping polygons)")

    return gdf


if __name__ == '__main__':
    main()
