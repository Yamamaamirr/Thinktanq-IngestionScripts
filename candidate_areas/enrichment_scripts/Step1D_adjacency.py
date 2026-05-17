"""
Step 1D — Adjacency distances per candidate (vectorized).

Reads:
  candidate_areas/outputs/candidate_areas.parquet      (95,269 candidate polygons EPSG:5070)
  ingestion_scripts/protected_areas_USA/pad_us_all.parquet  (PAD-US all GAP statuses)
  candidate_areas/exclusion_layers/exclusion_fema.parquet   (Floodway + AE/VE/A zones)
  candidate_areas/exclusion_layers/exclusion_nwi_{state}.parquet (NWI per state)
  ingestion_scripts/faa_radar/radar_exclusion_zones.parquet (159 radar stations)

Algorithm
---------
Vectorized via gpd.sjoin_nearest, one (candidate-set × source-set) call per
state per layer. No Python loop, no per-candidate STRtree calls. Geometries
are NOT simplified — full polygon edge fidelity is preserved.

Bounds:
  * max_distance = 10_000 m  (10 km).  Every scoring band tops out below
    8 km, so cutting search beyond 10 km changes no tier assignment.
    Candidates beyond 10 km from a given layer get a NaN distance, which
    the scoring engine interprets as "no concern from this layer".

  * Radar uses scipy cKDTree on station POINTS (the layer's natural form)
    rather than against its 3-mile buffer polygons. KDTree on points is
    mathematically identical to STRtree on points and far faster.

  * State-by-state processing keeps each sjoin bounded so even TX × 992k
    wetlands stays in memory.

Output:
  candidate_areas/enrichment_outputs/step1d_adjacency.parquet

Columns added per candidate:
  nearest_padus_distance_m         meters, NaN if > 10 km
  near_padus_flag                  True if < 500 m
  nearest_wetland_distance_m       meters, NaN if > 10 km
  near_wetland_flag                True if < 500 m
  nearest_floodway_distance_m      meters, NaN if > 10 km
  adjacent_floodway_flag           True if < 500 m
  nearest_fema_ae_distance_m       meters, NaN if > 10 km
  fema_ae_overlap_flag             True if distance == 0
  fema_ae_adjacent_flag            True if < 500 m
  nearest_radar_distance_m         meters, NaN if > 10 km
  radar_distance_miles             miles
  radar_review_flag                True if < 3 mi (Owen Rec #9)

Run:
  python candidate_areas/enrichment_scripts/Step1D_adjacency.py
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

CAND_PATH    = Path('candidate_areas/outputs/candidate_areas.parquet')
PADUS_PATH   = Path('ingestion_scripts/protected_areas_USA/pad_us_all.parquet')
NWI_DIR      = Path('candidate_areas/exclusion_layers')
FEMA_PATH    = Path('candidate_areas/exclusion_layers/exclusion_fema.parquet')
RADAR_PATH   = Path('ingestion_scripts/faa_radar/radar_exclusion_zones.parquet')
OUT_PATH     = Path('candidate_areas/enrichment_outputs/step1d_adjacency.parquet')

STATES = ['AZ','CA','NV','TX','VA']

MAX_SEARCH_M     = 10_000.0
NEAR_500_M       = 500.0
RADAR_NEAR_MI    = 3.0
RADAR_NEAR_M     = RADAR_NEAR_MI * 1609.34


def vectorized_nearest_distance(cands_gdf, src_gdf, label):
    """
    Vectorized nearest-distance lookup using gpd.sjoin_nearest.
    Returns an ndarray of distances aligned to cands_gdf.index (NaN where no
    match within MAX_SEARCH_M). Geometries used at full precision.
    """
    if len(src_gdf) == 0:
        return np.full(len(cands_gdf), np.nan)

    # sjoin_nearest can return multiple rows per left if ties — take just one
    # row per left index (the first match has min distance per docs).
    result = gpd.sjoin_nearest(
        cands_gdf[['candidate_id','geometry']],
        src_gdf[['geometry']],
        how='left',
        max_distance=MAX_SEARCH_M,
        distance_col='_dist',
    )
    # Deduplicate on candidate_id keeping the row with min distance
    result = result.sort_values('_dist').drop_duplicates(subset=['candidate_id'], keep='first')
    # Reindex to cands_gdf's order
    result = result.set_index('candidate_id').reindex(cands_gdf['candidate_id'].values)
    return result['_dist'].values


def main():
    print(f'Loading candidates: {CAND_PATH} ...')
    cands = gpd.read_parquet(CAND_PATH)
    cand_crs = cands.crs.to_epsg()
    print(f'  {len(cands):,} candidates in EPSG:{cand_crs}')
    assert cand_crs == 5070, "Expected candidates in EPSG:5070"

    # ---- Reference layers (reprojected to EPSG:5070 once) ----
    print(f'\nLoading PAD-US: {PADUS_PATH} ...')
    padus = gpd.read_parquet(PADUS_PATH)
    if padus.crs.to_epsg() != 5070:
        padus = padus.to_crs(5070)
    print(f'  {len(padus):,} PAD-US polygons (all GAP statuses, full geometry)')

    print(f'\nLoading FEMA: {FEMA_PATH} ...')
    fema = gpd.read_parquet(FEMA_PATH)
    if fema.crs.to_epsg() != 5070:
        fema = fema.to_crs(5070)
    floodway_mask = fema['zone_subty'].isin([
        'Floodway','Administrative Floodway','Riverine Floodway Shown in Coastal Zone'
    ])
    ae_mask = fema['fld_zone'].isin(['AE','VE','A'])
    floodway = fema[floodway_mask].reset_index(drop=True)
    fema_ae  = fema[ae_mask].reset_index(drop=True)
    print(f'  Floodway: {len(floodway):,}  |  AE/VE/A: {len(fema_ae):,}')

    print(f'\nLoading FAA radar: {RADAR_PATH} ...')
    radar = gpd.read_parquet(RADAR_PATH)
    if radar.crs.to_epsg() != 5070:
        radar = radar.to_crs(5070)
    # Radar will use cKDTree on station points reprojected to EPSG:5070 metres
    radar_pts_4326 = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(radar.station_lon, radar.station_lat),
        crs='EPSG:4326',
    )
    radar_pts_5070 = radar_pts_4326.to_crs(5070)
    radar_xy = np.column_stack([
        radar_pts_5070.geometry.x.values,
        radar_pts_5070.geometry.y.values,
    ])
    radar_tree = cKDTree(radar_xy)
    print(f'  {len(radar):,} radar stations (cKDTree built)')

    print('\nLoading NWI wetlands per state ...')
    nwi_by_state = {}
    for st in STATES:
        for p in (NWI_DIR / f'exclusion_nwi_{st.lower()}.parquet',
                  NWI_DIR / f'exclusion_nwi_{st.lower()}_test.parquet'):
            if p.exists():
                w = gpd.read_parquet(p)
                if w.crs.to_epsg() != 5070:
                    w = w.to_crs(5070)
                nwi_by_state[st] = w
                print(f'  {st}: {len(w):,} wetland polygons')
                break

    # ---- Per-state nearest-distance computation ----
    out_rows = []
    for st in STATES:
        sd = cands[cands.state == st].reset_index(drop=True)
        if len(sd) == 0:
            continue
        print(f'\n-- {st} -----------------------------------------')
        print(f'  {len(sd):,} candidates')

        # Candidate centroids in EPSG:5070 metres for the radar KDTree
        cand_xy = np.column_stack([
            sd.geometry.centroid.x.values,
            sd.geometry.centroid.y.values,
        ])

        # PAD-US: keep state-specific subset for sjoin
        st_padus = padus[padus.state == st].reset_index(drop=True)
        print(f'  PAD-US ({len(st_padus):,} polys) ...', flush=True)
        t0 = time.time()
        d_padus = vectorized_nearest_distance(sd, st_padus, 'padus')
        print(f'    {time.time()-t0:.1f}s')

        # Wetlands
        st_wet = nwi_by_state.get(st)
        if st_wet is not None and len(st_wet) > 0:
            print(f'  Wetlands ({len(st_wet):,} polys) ...', flush=True)
            t0 = time.time()
            d_wet = vectorized_nearest_distance(sd, st_wet, 'wetland')
            print(f'    {time.time()-t0:.1f}s')
        else:
            d_wet = np.full(len(sd), np.nan)

        # Floodway (national-level, but state bbox prefilter helps)
        # We pre-clip to a generous state envelope to keep sjoin_nearest tight
        st_bbox = sd.total_bounds  # minx,miny,maxx,maxy
        pad_box = 12_000  # 12 km padding > our 10 km cap
        bb = (st_bbox[0]-pad_box, st_bbox[1]-pad_box,
              st_bbox[2]+pad_box, st_bbox[3]+pad_box)
        from shapely.geometry import box
        env = box(*bb)
        st_fw = floodway[floodway.geometry.intersects(env)]
        print(f'  Floodway ({len(st_fw):,} polys in state envelope) ...', flush=True)
        t0 = time.time()
        d_fw = vectorized_nearest_distance(sd, st_fw, 'floodway')
        print(f'    {time.time()-t0:.1f}s')

        st_ae = fema_ae[fema_ae.geometry.intersects(env)]
        print(f'  FEMA AE ({len(st_ae):,} polys in state envelope) ...', flush=True)
        t0 = time.time()
        d_ae = vectorized_nearest_distance(sd, st_ae, 'fema_ae')
        print(f'    {time.time()-t0:.1f}s')

        # Radar via KDTree on centroid points (then cap at MAX_SEARCH_M)
        print(f'  Radar (159 stations, KDTree) ...', flush=True)
        t0 = time.time()
        dr, _ = radar_tree.query(cand_xy, k=1)
        dr = np.where(dr > MAX_SEARCH_M, np.nan, dr)
        print(f'    {time.time()-t0:.1f}s')

        out_rows.append(pd.DataFrame({
            'candidate_id':                 sd.candidate_id.values,
            'nearest_padus_distance_m':     d_padus,
            'nearest_wetland_distance_m':   d_wet,
            'nearest_floodway_distance_m':  d_fw,
            'nearest_fema_ae_distance_m':   d_ae,
            'nearest_radar_distance_m':     dr,
        }))

    print('\nConcatenating per-state results ...')
    out = pd.concat(out_rows, ignore_index=True)
    assert len(out) == len(cands), f'lost candidates: expected {len(cands)}, got {len(out)}'

    # Derived flags
    out['near_padus_flag']        = out.nearest_padus_distance_m.fillna(1e9) < NEAR_500_M
    out['near_wetland_flag']      = out.nearest_wetland_distance_m.fillna(1e9) < NEAR_500_M
    out['adjacent_floodway_flag'] = out.nearest_floodway_distance_m.fillna(1e9) < NEAR_500_M
    out['fema_ae_overlap_flag']   = out.nearest_fema_ae_distance_m.fillna(1e9) <= 0
    out['fema_ae_adjacent_flag']  = out.nearest_fema_ae_distance_m.fillna(1e9) < NEAR_500_M
    out['radar_distance_miles']   = out.nearest_radar_distance_m / 1609.34
    out['radar_review_flag']      = out.nearest_radar_distance_m.fillna(1e9) < RADAR_NEAR_M

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f'\nSaved: {OUT_PATH} ({size_mb:.2f} MB, {len(out):,} rows)')

    # ---- Distribution summary ----
    print('\n=== Distance medians (m) ===')
    for c in ['nearest_padus_distance_m','nearest_wetland_distance_m',
              'nearest_floodway_distance_m','nearest_fema_ae_distance_m',
              'nearest_radar_distance_m']:
        s = out[c].dropna()
        n_miss = out[c].isna().sum()
        if len(s):
            print(f'  {c:<33} median={s.median():>10.0f}  p10={s.quantile(0.1):>8.0f}  '
                  f'p90={s.quantile(0.9):>8.0f}  beyond_10km={n_miss:,}')

    print('\n=== Flag counts ===')
    for c in ['near_padus_flag','near_wetland_flag','adjacent_floodway_flag',
              'fema_ae_overlap_flag','fema_ae_adjacent_flag','radar_review_flag']:
        n = out[c].sum()
        print(f'  {c:<28} True: {n:>6,} ({100*n/len(out):5.1f}%)')

    print('\n=== Checks ===')
    checks = {
        'Has all candidates'                : len(out) == len(cands),
        'Unique candidate_ids'              : out.candidate_id.is_unique,
        'PAD-US dist non-neg or NaN'        : (out.nearest_padus_distance_m.fillna(0) >= 0).all(),
        'Wetland dist non-neg or NaN'       : (out.nearest_wetland_distance_m.fillna(0) >= 0).all(),
        'Floodway dist non-neg or NaN'      : (out.nearest_floodway_distance_m.fillna(0) >= 0).all(),
        'FEMA AE dist non-neg or NaN'       : (out.nearest_fema_ae_distance_m.fillna(0) >= 0).all(),
        'Radar dist non-neg or NaN'         : (out.nearest_radar_distance_m.fillna(0) >= 0).all(),
        'flag consistency: overlap→adjacent': ((~out.fema_ae_overlap_flag) | out.fema_ae_adjacent_flag).all(),
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
