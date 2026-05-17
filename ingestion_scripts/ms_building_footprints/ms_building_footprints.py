"""
Microsoft Building Footprints ingestion (per state).

The Microsoft Azure blob URL is no longer public. Source files must be
downloaded manually from https://github.com/microsoft/USBuildingFootprints
and placed under ingestion_scripts/ms_building_footprints/data/.

Expected layout (either form works):
  data/{State}.geojson                  (flat file)
  data/{State}.geojson/{State}.geojson  (nested, as produced by some downloaders)

Output (per state):
  data/ms_buildings_{state}.parquet  (EPSG:5070, geometry only)

Usage:
  python ingestion_scripts/ms_building_footprints/ms_building_footprints.py --state NV
  python ingestion_scripts/ms_building_footprints/ms_building_footprints.py --state ALL
"""

import argparse
import gc
from pathlib import Path
import geopandas as gpd

STATE_NAMES = {
    'AZ': 'Arizona',
    'CA': 'California',
    'NV': 'Nevada',
    'TX': 'Texas',
    'VA': 'Virginia',
}

DATA_DIR = Path('ingestion_scripts/ms_building_footprints/data')
DATA_DIR.mkdir(parents=True, exist_ok=True)


def find_geojson(state_abbr):
    """Locate the source GeoJSON for a state, handling flat and nested layouts."""
    name = STATE_NAMES[state_abbr]
    flat = DATA_DIR / f'{name}.geojson'

    if flat.is_file():
        return flat
    if flat.is_dir():
        nested = flat / f'{name}.geojson'
        if nested.is_file():
            return nested
        # fallback: any .geojson inside the folder
        candidates = list(flat.glob('*.geojson'))
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(
            f'Found folder {flat} but could not identify a unique .geojson inside '
            f'(found: {[c.name for c in candidates]}).'
        )
    raise FileNotFoundError(
        f'Source not found for {state_abbr}. Expected {flat} (file or directory). '
        f'Download from https://github.com/microsoft/USBuildingFootprints and place it there.'
    )


def convert_to_parquet(geojson_path, state_abbr):
    """Convert GeoJSON to parquet in EPSG:5070, geometry only."""
    out_path = DATA_DIR / f'ms_buildings_{state_abbr.lower()}.parquet'
    size_mb = geojson_path.stat().st_size / 1e6
    print(f'  Source: {geojson_path} ({size_mb:,.0f} MB)')

    print(f'  Reading GeoJSON ...', flush=True)
    gdf = gpd.read_file(geojson_path)
    print(f'    {len(gdf):,} building polygons, CRS: {gdf.crs}')

    if gdf.crs is None:
        gdf.set_crs('EPSG:4326', inplace=True)
    if gdf.crs.to_epsg() != 5070:
        print(f'    Reprojecting to EPSG:5070 ...', flush=True)
        gdf = gdf.to_crs('EPSG:5070')

    gdf = gdf[['geometry']].copy()

    print(f'  Writing parquet ...', flush=True)
    gdf.to_parquet(out_path, index=False)
    out_mb = out_path.stat().st_size / 1e6
    print(f'  Saved: {out_path.name} ({out_mb:,.0f} MB)')

    del gdf
    gc.collect()
    return out_path


def process_state(state_abbr):
    print(f'\n{"="*55}')
    print(f' {state_abbr} ({STATE_NAMES[state_abbr]})')
    print(f'{"="*55}')

    out_path = DATA_DIR / f'ms_buildings_{state_abbr.lower()}.parquet'
    if out_path.exists():
        print(f'  Already done: {out_path.name} ({out_path.stat().st_size/1e6:.0f} MB) - skipping')
        return

    try:
        geojson_path = find_geojson(state_abbr)
    except FileNotFoundError as e:
        print(f'  SKIP: {e}')
        return

    convert_to_parquet(geojson_path, state_abbr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True,
                        choices=list(STATE_NAMES.keys()) + ['ALL'])
    args = parser.parse_args()

    states = list(STATE_NAMES.keys()) if args.state == 'ALL' else [args.state]
    for s in states:
        process_state(s)

    print(f'\nDone. Outputs in {DATA_DIR}')


if __name__ == '__main__':
    main()
