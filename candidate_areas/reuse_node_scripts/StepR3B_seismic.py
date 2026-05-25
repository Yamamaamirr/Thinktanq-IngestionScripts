"""
StepR3B -- Seismic hazard per reuse node (point-in-polygon on NSHM23).

Mirror of candidate_areas/enrichment_scripts/Step1B_seismic.py, but reads
reuse_nodes_clean.parquet and writes to reuse_node_enrichment_outputs/.

Output columns identical to Step1B (so downstream scoring can be reused):
  seismic_hazard_pga, seismic_hazard_tier, seismic_polygon_pga_range,
  seismic_valley_response

Run:
  python candidate_areas/reuse_node_scripts/StepR3B_seismic.py
"""
from pathlib import Path
import sys
import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _r3_helpers import load_reuse_nodes_as_candidates, out_path

NSHM_PATH = Path('ingestion_scripts/usgs_seismic/seismic_hazard_polygons_nshm23.parquet')
OUT_PATH  = out_path('stepR3b_seismic.parquet')


def assign_band(pga):
    if pd.isna(pga):
        return None
    if pga < 0.10: return 'very_low'
    if pga < 0.25: return 'low'
    if pga < 0.50: return 'moderate'
    if pga < 1.0:  return 'high'
    return 'very_high'


def main():
    print('Loading reuse nodes (as candidates) ...')
    cands = load_reuse_nodes_as_candidates(crs_epsg=5070)
    print(f'  {len(cands):,} reuse nodes in EPSG:{cands.crs.to_epsg()}')

    print('Computing centroids in EPSG:4326 ...')
    centroids_5070 = cands.geometry.centroid
    centroids_4326 = gpd.GeoSeries(centroids_5070, crs=cands.crs).to_crs(4326)
    pts = gpd.GeoDataFrame(
        {'candidate_id': cands.candidate_id.values},
        geometry=centroids_4326,
        crs='EPSG:4326',
    )

    print(f'Loading NSHM23 polygons: {NSHM_PATH} ...')
    nshm = gpd.read_parquet(NSHM_PATH)
    print(f'  {len(nshm):,} polygons in EPSG:{nshm.crs.to_epsg()}')

    print('Running spatial join (point-in-polygon) ...')
    keep = ['pga_mid', 'seismic_band', 'pga_range_label', 'valley_response', 'geometry']
    joined = gpd.sjoin(pts, nshm[keep], how='left', predicate='within')

    n_null = joined.seismic_band.isna().sum()
    print(f'  Reuse nodes with no polygon containing centroid: {n_null:,}')
    if n_null > 0:
        print(f'  Running sjoin_nearest fallback for {n_null} points ...')
        null_pts = pts.loc[joined[joined.seismic_band.isna()].index]
        fallback = gpd.sjoin_nearest(null_pts, nshm[keep], how='left', max_distance=None)
        fallback = fallback[~fallback.index.duplicated(keep='first')]
        for col in ['pga_mid', 'seismic_band', 'pga_range_label', 'valley_response']:
            joined.loc[fallback.index, col] = fallback[col].values

    joined['seismic_hazard_pga']  = joined['pga_mid']
    joined['seismic_hazard_tier'] = joined['pga_mid'].apply(assign_band)

    out = pd.DataFrame({
        'candidate_id':              joined.candidate_id.values,
        'seismic_hazard_pga':        joined['seismic_hazard_pga'].values,
        'seismic_hazard_tier':       joined['seismic_hazard_tier'].values,
        'seismic_polygon_pga_range': joined['pga_range_label'].values,
        'seismic_valley_response':   joined['valley_response'].values,
    })

    out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.2f} MB, {len(out):,} rows)')

    print('\n=== Seismic tier distribution ===')
    print(out.seismic_hazard_tier.value_counts().reindex(
        ['very_low','low','moderate','high','very_high']).to_string())

    print('\n=== Per-state seismic distribution ===')
    cdf = cands[['candidate_id','state']].merge(out, on='candidate_id')
    for st in ['TX','VA','AZ','NV','CA']:
        sd = cdf[cdf.state == st]
        bc = sd.seismic_hazard_tier.value_counts().to_dict()
        print(f'  {st}: n={len(sd):>5,}  vlow={bc.get("very_low",0):>4}  '
              f'low={bc.get("low",0):>4}  mod={bc.get("moderate",0):>4}  '
              f'high={bc.get("high",0):>4}  vhi={bc.get("very_high",0):>4}')

    print('\n=== Checks ===')
    checks = {
        'Has all reuse nodes'  : len(out) == len(cands),
        'Unique candidate_ids' : out.candidate_id.is_unique,
        'No null PGA'          : out.seismic_hazard_pga.notna().all(),
        'No null tier'         : out.seismic_hazard_tier.notna().all(),
        'Tier in valid set'    : set(out.seismic_hazard_tier.unique()) <= {'very_low','low','moderate','high','very_high'},
        'PGA non-negative'     : (out.seismic_hazard_pga >= 0).all(),
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
