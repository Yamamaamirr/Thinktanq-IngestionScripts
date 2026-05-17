"""
Step 7g - Road clip exclusion: subtract major road buffers from candidates.

Uses TIGER 2025 PRISECROADS (Interstate + US Highway + State Highway + major
named roads), buffered 25m, subtracted from candidate geometries. Candidates
whose net area drops below 50 acres are dropped.

FAST APPROACH: per-candidate difference with only intersecting roads.
Avoids expensive global unary_union.

Reads:
  candidate_areas/outputs/candidates_final.parquet
  ingestion_scripts/census_tiger/prisecroads_5states.parquet

Writes:
  candidate_areas/outputs/candidate_areas_enriched.parquet  (mark excluded)
  candidate_areas/outputs/candidates_final.parquet  (clipped)
  candidate_areas/outputs/candidates_final.csv
  candidate_areas/outputs/candidates_final.fgb (best effort)
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

CAND_FINAL    = Path('candidate_areas/outputs/candidates_final.parquet')
CSV_OUT       = Path('candidate_areas/outputs/candidates_final.csv')
FGB_OUT       = Path('candidate_areas/outputs/candidates_final.fgb')
ENRICHED_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')
ROADS_PATH    = Path('ingestion_scripts/census_tiger/prisecroads_5states.parquet')

ROAD_BUFFER_M = 25.0
MIN_ACRES = 50.0
ACRES_PER_M2 = 0.000247105


def _je(x):
    if x is None: return None
    if isinstance(x, np.ndarray): return json.dumps(x.tolist())
    if isinstance(x, (list, tuple, dict)): return json.dumps(list(x) if isinstance(x, tuple) else x)
    return x


def main():
    print(f'Loading {CAND_FINAL} ...')
    cands = gpd.read_parquet(CAND_FINAL)
    n_before = len(cands)
    print(f'  {n_before:,} candidates')

    print(f'\nLoading {ROADS_PATH} ...')
    roads = gpd.read_parquet(ROADS_PATH)
    if roads.crs != cands.crs:
        roads = roads.to_crs(cands.crs)
    # Buffer each road segment individually (vectorized)
    print(f'Buffering all {len(roads):,} road segments {ROAD_BUFFER_M}m ...')
    t0 = time.time()
    roads['geometry'] = roads.geometry.buffer(ROAD_BUFFER_M)
    print(f'  done ({time.time()-t0:.1f}s)')

    # Spatial join: find which roads intersect each candidate
    print(f'\nSpatial join (candidate x road buffer intersect) ...')
    t0 = time.time()
    join = gpd.sjoin(
        cands[['candidate_id','geometry']],
        roads[['geometry']].rename_geometry('road_geom') if False else roads[['geometry']],
        how='inner',
        predicate='intersects',
    )
    print(f'  {time.time()-t0:.1f}s, {len(join):,} candidate-road intersections '
          f'({join.candidate_id.nunique():,} unique candidates affected)')

    # Per-candidate: collect intersecting road indices
    print('\nBuilding per-candidate road lists ...')
    t0 = time.time()
    cand_to_roads = join.groupby('candidate_id')['index_right'].apply(list).to_dict()
    print(f'  done ({time.time()-t0:.1f}s)')

    # Per-candidate difference
    print('\nPer-candidate clip + recompute area ...')
    t0 = time.time()
    affected_ids = set(cand_to_roads.keys())
    new_geoms = {}
    new_areas_ac = {}
    n_done = 0
    cands_indexed = cands.set_index('candidate_id')
    for cid, road_idxs in cand_to_roads.items():
        candidate_geom = cands_indexed.loc[cid, 'geometry']
        # Build union of only intersecting road buffers (small set, fast)
        road_geoms = roads.geometry.iloc[road_idxs].values
        # For very large sets, unary_union; for small sets, just .difference repeatedly is OK
        if len(road_geoms) == 1:
            road_union = road_geoms[0]
        else:
            road_union = unary_union(list(road_geoms))
        clipped = candidate_geom.difference(road_union)
        new_geoms[cid] = clipped
        new_areas_ac[cid] = clipped.area * ACRES_PER_M2
        n_done += 1
        if n_done % 5000 == 0:
            print(f'  {n_done:,}/{len(cand_to_roads):,} ({time.time()-t0:.1f}s)')
    print(f'  done ({time.time()-t0:.1f}s)')

    # Update geometries
    print('\nUpdating geometries and areas ...')
    out = cands.copy()
    out['geometry'] = out.apply(
        lambda r: new_geoms.get(r.candidate_id, r.geometry), axis=1
    )
    out['area_m2'] = out.geometry.area
    out['area_acres'] = out['area_m2'] * ACRES_PER_M2

    # Drop candidates below 50 ac
    n_too_small = (out.area_acres < MIN_ACRES).sum()
    print(f'\nDropping {n_too_small:,} candidates now < {MIN_ACRES} ac after clip')
    keep = out[out.area_acres >= MIN_ACRES].copy().sort_values('composite_score', ascending=False).reset_index(drop=True)
    n_after = len(keep)
    print(f'  before={n_before:,}, after={n_after:,}, total dropped={n_before-n_after:,}')

    print(f'  By state:')
    for st in ['AZ','CA','NV','TX','VA']:
        n_b = (cands.state == st).sum()
        n_a = (keep.state == st).sum()
        print(f'    {st}: {n_b:>6,} -> {n_a:>6,}  (lost {n_b-n_a:>5,}, {100*(n_b-n_a)/max(n_b,1):.1f}%)')

    # Update enriched
    print(f'\nUpdating enriched source ...')
    enriched = gpd.read_parquet(ENRICHED_PATH)
    dropped_ids = set(cands.candidate_id) - set(keep.candidate_id)
    mask = enriched.candidate_id.isin(dropped_ids) & (enriched.candidate_status != 'excluded')
    enriched.loc[mask, 'candidate_status'] = 'excluded'
    enriched.loc[mask, 'candidate_status_reason'] = 'road_clip_residual_lt_50ac_pdf_rule_10'
    print(f'  Marked {mask.sum():,} as excluded')
    # Also update geometry for kept candidates
    geom_lookup = keep.set_index('candidate_id').geometry
    upd = enriched.candidate_id.isin(keep.candidate_id)
    new_g = enriched.loc[upd, 'candidate_id'].map(geom_lookup)
    enriched.loc[upd, 'geometry'] = new_g.values
    print(f'  Updated geometry for {upd.sum():,} clipped survivors')
    enriched.to_parquet(ENRICHED_PATH, index=False)

    # Write outputs
    print(f'\nWriting GeoParquet: {CAND_FINAL}')
    keep.to_parquet(CAND_FINAL, index=False)
    print(f'  Saved ({CAND_FINAL.stat().st_size/1e6:.1f} MB)')

    print(f'\nWriting CSV: {CSV_OUT}')
    csv_df = keep.copy()
    csv_df['geometry_wkt'] = csv_df.geometry.to_wkt()
    csv_df = csv_df.drop(columns=['geometry'])
    for col in csv_df.columns:
        sample = csv_df[col].dropna().head(1)
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict, tuple, np.ndarray)):
            csv_df[col] = csv_df[col].apply(_je)
    csv_df.to_csv(CSV_OUT, index=False)
    print(f'  Saved ({CSV_OUT.stat().st_size/1e6:.1f} MB)')

    # FGB
    fgb_df = keep.copy()
    for col in fgb_df.columns:
        if col == 'geometry': continue
        sample = fgb_df[col].dropna().head(1)
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict, tuple, np.ndarray)):
            fgb_df[col] = fgb_df[col].apply(_je)
    fgb_target = FGB_OUT
    try:
        if fgb_target.exists():
            fgb_target.unlink()
        fgb_df.to_file(fgb_target, driver='FlatGeobuf')
        print(f'\nWrote FGB: {fgb_target} ({fgb_target.stat().st_size/1e6:.1f} MB)')
    except (PermissionError, OSError):
        fgb_target = FGB_OUT.with_name('candidates_final_NEW.fgb')
        if fgb_target.exists():
            fgb_target.unlink()
        fgb_df.to_file(fgb_target, driver='FlatGeobuf')
        print(f'\nFGB locked, wrote to {fgb_target} ({fgb_target.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
