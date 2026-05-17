"""
Step 3A (FGB-only) — Export candidates_final.fgb from the already-written GeoParquet.

Avoids re-running CSV export (which is slow). Reads candidates_final.parquet and
writes candidates_final.fgb alongside it.
"""
import json
from pathlib import Path
import numpy as np
import geopandas as gpd

PARQUET_IN = Path('candidate_areas/outputs/candidates_final.parquet')
FGB_OUT    = Path('candidate_areas/outputs/candidates_final.fgb')


def _json_encode(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return json.dumps(x.tolist())
    if isinstance(x, (list, tuple, dict)):
        return json.dumps(list(x) if isinstance(x, tuple) else x)
    return x


def main():
    print(f'Loading {PARQUET_IN} ...')
    gdf = gpd.read_parquet(PARQUET_IN)
    print(f'  {len(gdf):,} rows x {len(gdf.columns)} cols, CRS={gdf.crs.to_epsg()}')

    # Serialize list/dict/ndarray columns — FGB schema can't represent them natively
    print('Encoding list-type columns to JSON strings ...')
    for col in gdf.columns:
        if col == 'geometry':
            continue
        sample = gdf[col].dropna().head(1)
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict, tuple, np.ndarray)):
            print(f'  json-encoding: {col}')
            gdf[col] = gdf[col].apply(_json_encode)

    if FGB_OUT.exists():
        FGB_OUT.unlink()
    print(f'\nWriting FlatGeobuf: {FGB_OUT} ...')
    gdf.to_file(FGB_OUT, driver='FlatGeobuf')
    sz_mb = FGB_OUT.stat().st_size / 1e6
    print(f'  Saved ({sz_mb:.1f} MB)')

    # Verification
    print('\n=== Verification (re-read FGB) ===')
    fgb = gpd.read_file(FGB_OUT)
    print(f'  rows={len(fgb):,}  cols={len(fgb.columns)}  CRS={fgb.crs.to_epsg()}')
    checks = {
        'Row count matches parquet'   : len(fgb) == len(gdf),
        'Column count matches parquet': len(fgb.columns) == len(gdf.columns),
        'CRS preserved (EPSG:5070)'   : fgb.crs.to_epsg() == 5070,
        'All geometries valid'        : fgb.geometry.is_valid.all(),
        'candidate_id unique'         : fgb.candidate_id.is_unique,
        'composite_score in [0,100]'  : (fgb.composite_score >= 0).all() and (fgb.composite_score <= 100.001).all(),
    }
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')

    # Compare top-row by candidate_id
    src_top_id = gdf.sort_values('composite_score', ascending=False).iloc[0].candidate_id
    fgb_top_id = fgb.sort_values('composite_score', ascending=False).iloc[0].candidate_id
    print(f'  Top candidate_id: parquet={src_top_id[:12]}.. fgb={fgb_top_id[:12]}..  match={src_top_id == fgb_top_id}')


if __name__ == '__main__':
    main()
