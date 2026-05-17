"""
Step 1B — Seismic hazard per candidate (point-in-polygon on NSHM23).

Reads:
  candidate_areas/outputs/candidate_areas.parquet      (95,269 candidate polygons in EPSG:5070)
  ingestion_scripts/usgs_seismic/seismic_hazard_polygons_nshm23.parquet
    (4,170 polygon bands in EPSG:4326 covering all 5 states)

For each candidate centroid we find the NSHM23 polygon that contains it
and copy that polygon's mid PGA value plus the 5-band classification
(very_low / low / moderate / high / very_high per scoring_design.md 5.2).

If a candidate centroid happens to fall on a polygon boundary or just
outside any polygon (rare — verified earlier that every candidate has
coverage), we fall back to sjoin_nearest.

Output:
  candidate_areas/enrichment_outputs/step1b_seismic.parquet

Columns added per candidate:
  seismic_hazard_pga        PGA in g (from polygon mid)
  seismic_hazard_tier       very_low / low / moderate / high / very_high
  seismic_polygon_pga_range range string for audit, e.g. "0.83 - 0.84"
  seismic_valley_response   True if candidate falls in a valley-response polygon

Run:
  python candidate_areas/enrichment_scripts/Step1B_seismic.py
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

CAND_PATH = Path('candidate_areas/outputs/candidate_areas.parquet')
NSHM_PATH = Path('ingestion_scripts/usgs_seismic/seismic_hazard_polygons_nshm23.parquet')
OUT_PATH  = Path('candidate_areas/enrichment_outputs/step1b_seismic.parquet')


def assign_band(pga):
    """Per scoring_design.md 5.2."""
    if pd.isna(pga):
        return None
    if pga < 0.10: return 'very_low'
    if pga < 0.25: return 'low'
    if pga < 0.50: return 'moderate'
    if pga < 1.0:  return 'high'
    return 'very_high'


def main():
    print(f'Loading candidates: {CAND_PATH} ...')
    cands = gpd.read_parquet(CAND_PATH)
    print(f'  {len(cands):,} candidates in EPSG:{cands.crs.to_epsg()}')

    # We need centroids in EPSG:4326 to match the NSHM23 polygons
    # Centroid is computed in projected CRS then reprojected to 4326 for accuracy
    print('Computing centroids in EPSG:4326 ...')
    centroids_5070 = cands.geometry.centroid
    centroids_4326 = gpd.GeoSeries(centroids_5070, crs=cands.crs).to_crs(4326)
    pts = gpd.GeoDataFrame(
        {'candidate_id': cands.candidate_id.values},
        geometry=centroids_4326,
        crs='EPSG:4326',
    )
    print(f'  {len(pts):,} centroid points')

    print(f'Loading NSHM23 polygons: {NSHM_PATH} ...')
    nshm = gpd.read_parquet(NSHM_PATH)
    print(f'  {len(nshm):,} polygons in EPSG:{nshm.crs.to_epsg()}')

    # Spatial join: each point → containing polygon
    print('Running spatial join (point-in-polygon, sjoin within) ...')
    keep = ['pga_mid', 'seismic_band', 'pga_range_label', 'valley_response', 'geometry']
    joined = gpd.sjoin(pts, nshm[keep], how='left', predicate='within')

    n_null = joined.seismic_band.isna().sum()
    print(f'  Candidates with no polygon containing centroid: {n_null:,}')

    if n_null > 0:
        # Fallback: nearest polygon for any points that fell outside
        print(f'  Running sjoin_nearest fallback for {n_null} points ...')
        null_pts = pts.loc[joined[joined.seismic_band.isna()].index]
        fallback = gpd.sjoin_nearest(
            null_pts, nshm[keep],
            how='left', max_distance=None,
        )
        # Take the first match per source point (sjoin_nearest can return ties)
        fallback = fallback[~fallback.index.duplicated(keep='first')]
        for col in ['pga_mid', 'seismic_band', 'pga_range_label', 'valley_response']:
            joined.loc[fallback.index, col] = fallback[col].values

    # Re-band locally to make sure our 5-band scheme is applied (NSHM file
    # had its own seismic_band column; we keep it but re-derive to be safe)
    joined['seismic_hazard_pga']  = joined['pga_mid']
    joined['seismic_hazard_tier'] = joined['pga_mid'].apply(assign_band)

    out = pd.DataFrame({
        'candidate_id':             joined.candidate_id.values,
        'seismic_hazard_pga':       joined['seismic_hazard_pga'].values,
        'seismic_hazard_tier':      joined['seismic_hazard_tier'].values,
        'seismic_polygon_pga_range': joined['pga_range_label'].values,
        'seismic_valley_response':   joined['valley_response'].values,
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f'\nSaved: {OUT_PATH} ({size_mb:.2f} MB, {len(out):,} rows)')

    # ---- Distribution summary ----
    print('\n=== Seismic tier distribution overall ===')
    print(out.seismic_hazard_tier.value_counts().reindex(
        ['very_low','low','moderate','high','very_high']).to_string())

    print('\n=== Per-state seismic distribution ===')
    cdf = cands[['candidate_id','state']].merge(out, on='candidate_id')
    for st in ['TX','VA','AZ','NV','CA']:
        sd = cdf[cdf.state == st]
        bc = sd.seismic_hazard_tier.value_counts().to_dict()
        print(f'  {st}: n={len(sd):>6,}  '
              f'vlow={bc.get("very_low",0):>5}  '
              f'low={bc.get("low",0):>5}  '
              f'mod={bc.get("moderate",0):>5}  '
              f'high={bc.get("high",0):>5}  '
              f'vhi={bc.get("very_high",0):>5}')

    print('\n=== Checks ===')
    checks = {
        'Has all candidates'    : len(out) == len(cands),
        'Unique candidate_ids'  : out.candidate_id.is_unique,
        'No null PGA'           : out.seismic_hazard_pga.notna().all(),
        'No null tier'          : out.seismic_hazard_tier.notna().all(),
        'Tier in 5 valid values': set(out.seismic_hazard_tier.unique()) <= {'very_low','low','moderate','high','very_high'},
        'PGA non-negative'      : (out.seismic_hazard_pga >= 0).all(),
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False

    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
