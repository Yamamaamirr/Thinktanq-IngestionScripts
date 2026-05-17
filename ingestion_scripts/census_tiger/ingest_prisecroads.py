"""
Download TIGER Primary+Secondary Roads for the 5 target states.

PRISECROADS = Interstates + US Highways + State Highways
(does NOT include residential streets, county roads, dirt roads)

Source: TIGER 2025 per-state files
  https://www2.census.gov/geo/tiger/TIGER2025/PRISECROADS/tl_2025_{fips}_prisecroads.zip

Output: ingestion_scripts/census_tiger/prisecroads_5states.parquet (EPSG:4326)
"""
import io, time, zipfile
from pathlib import Path
import geopandas as gpd
import pandas as pd
import requests

TIGER_BASE = 'https://www2.census.gov/geo/tiger/TIGER2025/PRISECROADS'
TARGET_STATES = {'CA': '06', 'TX': '48', 'AZ': '04', 'NV': '32', 'VA': '51'}
CACHE = Path('ingestion_scripts/census_tiger/cache')
OUT_PATH = Path('ingestion_scripts/census_tiger/prisecroads_5states.parquet')


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for state, fips in TARGET_STATES.items():
        url = f'{TIGER_BASE}/tl_2025_{fips}_prisecroads.zip'
        cache_path = CACHE / f'tl_2025_{fips}_prisecroads.zip'
        if not cache_path.exists():
            print(f'  [{state}] downloading {url} ...')
            r = requests.get(url, timeout=180, stream=True)
            r.raise_for_status()
            with open(cache_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            print(f'    -> {cache_path.stat().st_size/1e6:.1f} MB')
        else:
            print(f'  [{state}] cached ({cache_path.stat().st_size/1e6:.1f} MB)')

        # Read shapefile from zip
        with zipfile.ZipFile(cache_path) as zf:
            shp_name = next(n for n in zf.namelist() if n.endswith('.shp'))
            extract_dir = CACHE / f'{state}_extract'
            extract_dir.mkdir(exist_ok=True)
            zf.extractall(extract_dir)

        shp_path = list(extract_dir.glob('*.shp'))[0]
        g = gpd.read_file(shp_path)
        g['state'] = state
        print(f'    {len(g):,} road segments')
        frames.append(g[['LINEARID','FULLNAME','RTTYP','MTFCC','state','geometry']])

    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry='geometry', crs='EPSG:4326')
    print(f'\nTotal: {len(out):,} road segments across 5 states')
    print(f'RTTYP distribution:')
    print(out.RTTYP.value_counts().to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
