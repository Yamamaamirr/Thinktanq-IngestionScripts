"""
Step 3A — Export final deliverables: CSV + GeoParquet + FlatGeobuf.

Reads:
  candidate_areas/outputs/candidate_areas_enriched.parquet  (95,269 rows x 152 cols)

Writes:
  candidate_areas/outputs/candidates_final.csv      (full 152 cols, geometry as WKT)
  candidate_areas/outputs/candidates_final.parquet  (full 152 cols, GeoParquet w/ geometry)
  candidate_areas/outputs/candidates_final.fgb      (full 152 cols, FlatGeobuf — single-file, QGIS/ArcGIS native)

All three sorted by composite_score descending.

Run:
  python candidate_areas/scoring_scripts/Step3A_export_csv_geoparquet.py
"""

from pathlib import Path
import json
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')
CSV_OUT       = Path('candidate_areas/outputs/candidates_final.csv')
PARQUET_OUT   = Path('candidate_areas/outputs/candidates_final.parquet')
FGB_OUT       = Path('candidate_areas/outputs/candidates_final.fgb')


def main():
    print(f'Loading enriched parquet: {ENRICHED_PATH} ...')
    gdf = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(gdf):,} rows x {len(gdf.columns)} cols, CRS={gdf.crs.to_epsg()}')

    # Sort by composite_score desc so highest-ranked surface at the top
    print('Sorting by composite_score descending ...')
    gdf = gdf.sort_values('composite_score', ascending=False).reset_index(drop=True)

    # ---- GeoParquet (the primary geo deliverable) ----
    print(f'\nWriting GeoParquet: {PARQUET_OUT} ...')
    gdf.to_parquet(PARQUET_OUT, index=False)
    sz_mb = PARQUET_OUT.stat().st_size / 1e6
    print(f'  Saved ({sz_mb:.1f} MB)')

    # ---- CSV (full schema, geometry serialized as WKT) ----
    print(f'\nWriting CSV: {CSV_OUT} ...')
    import numpy as np
    csv_df = gdf.copy()
    csv_df['geometry_wkt'] = csv_df.geometry.to_wkt()
    csv_df = csv_df.drop(columns=['geometry'])
    # Serialize list/dict/ndarray columns to JSON strings so the CSV is parseable
    def _json_encode(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return json.dumps(x.tolist())
        if isinstance(x, (list, tuple, dict)):
            return json.dumps(list(x) if isinstance(x, tuple) else x)
        return x
    for col in csv_df.columns:
        sample = csv_df[col].dropna().head(1)
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict, tuple, np.ndarray)):
            csv_df[col] = csv_df[col].apply(_json_encode)
    csv_df.to_csv(CSV_OUT, index=False)
    sz_mb = CSV_OUT.stat().st_size / 1e6
    print(f'  Saved ({sz_mb:.1f} MB)')

    # ---- FlatGeobuf (single-file geo format, opens directly in QGIS/ArcGIS) ----
    print(f'\nWriting FlatGeobuf: {FGB_OUT} ...')
    fgb_df = gdf.copy()
    # FGB doesn't support list/dict/ndarray columns — serialize to JSON strings (same as CSV)
    for col in fgb_df.columns:
        if col == 'geometry':
            continue
        sample = fgb_df[col].dropna().head(1)
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict, tuple, np.ndarray)):
            fgb_df[col] = fgb_df[col].apply(_json_encode)
    if FGB_OUT.exists():
        FGB_OUT.unlink()
    fgb_df.to_file(FGB_OUT, driver='FlatGeobuf')
    sz_mb = FGB_OUT.stat().st_size / 1e6
    print(f'  Saved ({sz_mb:.1f} MB)')

    # ---- Verification ----
    print('\n=== Verification ===')
    # Re-read both and confirm round-trip integrity
    print('  Re-reading GeoParquet ...')
    gp_check = gpd.read_parquet(PARQUET_OUT)
    print(f'    rows={len(gp_check):,}  cols={len(gp_check.columns)}  CRS={gp_check.crs.to_epsg()}')

    print('  Re-reading CSV (first 100 rows) ...')
    csv_check = pd.read_csv(CSV_OUT, nrows=100)
    print(f'    rows={len(csv_check):,}  cols={len(csv_check.columns)}')

    print('  Top row in CSV matches top row in source:')
    src_top = gdf.iloc[0]
    csv_top = csv_check.iloc[0]
    matches = (
        src_top.candidate_id == csv_top.candidate_id and
        abs(src_top.composite_score - csv_top.composite_score) < 0.0001
    )
    print(f'    candidate_id match: {src_top.candidate_id == csv_top.candidate_id}')
    print(f'    composite_score: src={src_top.composite_score:.4f} csv={csv_top.composite_score:.4f}')
    print(f'    overall: {matches}')

    print('  GeoParquet integrity:')
    pq_checks = {
        'Has all candidates'    : len(gp_check) == len(gdf),
        'Same column count'     : len(gp_check.columns) == len(gdf.columns),
        'CRS preserved'         : gp_check.crs == gdf.crs,
        'Geometry valid'        : gp_check.geometry.is_valid.all(),
        'composite_score range' : (gp_check.composite_score >= 0).all() and (gp_check.composite_score <= 100.001).all(),
        'Top row composite match': abs(gp_check.iloc[0].composite_score - gdf.iloc[0].composite_score) < 0.0001,
        'No duplicate cand_ids' : gp_check.candidate_id.is_unique,
    }
    for lbl, ok in pq_checks.items():
        print(f'    [{"PASS" if ok else "FAIL"}] {lbl}')

    print('  Re-reading FlatGeobuf ...')
    fgb_check = gpd.read_file(FGB_OUT)
    print(f'    rows={len(fgb_check):,}  cols={len(fgb_check.columns)}  CRS={fgb_check.crs.to_epsg()}')
    fgb_checks = {
        'Has all candidates'    : len(fgb_check) == len(gdf),
        'CRS preserved'         : fgb_check.crs == gdf.crs,
        'Geometry valid'        : fgb_check.geometry.is_valid.all(),
        'No duplicate cand_ids' : fgb_check.candidate_id.is_unique,
        'Top composite match'   : abs(fgb_check.sort_values('composite_score', ascending=False).iloc[0].composite_score - gdf.iloc[0].composite_score) < 0.0001,
    }
    for lbl, ok in fgb_checks.items():
        print(f'    [{"PASS" if ok else "FAIL"}] {lbl}')

    print('\nExports complete.')
    print(f'\nFinal deliverables:')
    print(f'  {PARQUET_OUT}  ({PARQUET_OUT.stat().st_size/1e6:.1f} MB)')
    print(f'  {CSV_OUT}      ({CSV_OUT.stat().st_size/1e6:.1f} MB)')
    print(f'  {FGB_OUT}      ({FGB_OUT.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
