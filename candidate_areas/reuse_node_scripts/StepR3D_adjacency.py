"""
StepR3D -- Adjacency distances per reuse node (PAD-US, NWI, floodway, FEMA AE, radar).

Mirror of Step1D_adjacency.py for reuse_nodes_clean.parquet. Identical
algorithm, columns, and thresholds; only swaps input/output paths and the
PAD-US state filter (since our helper renames state_abbr -> state).

Output columns (identical to Step1D):
  nearest_padus_distance_m, near_padus_flag,
  nearest_wetland_distance_m, near_wetland_flag,
  nearest_floodway_distance_m, adjacent_floodway_flag,
  nearest_fema_ae_distance_m, fema_ae_overlap_flag, fema_ae_adjacent_flag,
  nearest_radar_distance_m, radar_distance_miles, radar_review_flag

Run: python candidate_areas/reuse_node_scripts/StepR3D_adjacency.py
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent))
from _r3_helpers import load_reuse_nodes_as_candidates, out_path

PADUS_PATH = Path('ingestion_scripts/protected_areas_USA/pad_us_all.parquet')
NWI_DIR    = Path('candidate_areas/exclusion_layers')
FEMA_PATH  = Path('candidate_areas/exclusion_layers/exclusion_fema.parquet')
RADAR_PATH = Path('ingestion_scripts/faa_radar/radar_exclusion_zones.parquet')
OUT_PATH   = out_path('stepR3d_adjacency.parquet')

STATES = ['AZ','CA','NV','TX','VA']
MAX_SEARCH_M  = 10_000.0
NEAR_500_M    = 500.0
RADAR_NEAR_MI = 3.0
RADAR_NEAR_M  = RADAR_NEAR_MI * 1609.34


def vectorized_nearest_distance(cands_gdf, src_gdf):
    if len(src_gdf) == 0:
        return np.full(len(cands_gdf), np.nan)
    result = gpd.sjoin_nearest(
        cands_gdf[['candidate_id','geometry']],
        src_gdf[['geometry']],
        how='left', max_distance=MAX_SEARCH_M, distance_col='_dist',
    )
    result = result.sort_values('_dist').drop_duplicates(subset=['candidate_id'], keep='first')
    result = result.set_index('candidate_id').reindex(cands_gdf['candidate_id'].values)
    return result['_dist'].values


def main():
    print('Loading reuse nodes ...')
    cands = load_reuse_nodes_as_candidates(crs_epsg=5070)
    print(f'  {len(cands):,} reuse nodes')

    print(f'\nLoading PAD-US: {PADUS_PATH} ...')
    padus = gpd.read_parquet(PADUS_PATH)
    if padus.crs.to_epsg() != 5070:
        padus = padus.to_crs(5070)
    print(f'  {len(padus):,} PAD-US polygons')

    print(f'\nLoading FEMA: {FEMA_PATH} ...')
    fema = gpd.read_parquet(FEMA_PATH)
    if fema.crs.to_epsg() != 5070:
        fema = fema.to_crs(5070)
    floodway = fema[fema['zone_subty'].isin([
        'Floodway','Administrative Floodway','Riverine Floodway Shown in Coastal Zone'
    ])].reset_index(drop=True)
    fema_ae = fema[fema['fld_zone'].isin(['AE','VE','A'])].reset_index(drop=True)
    print(f'  Floodway: {len(floodway):,}  |  AE/VE/A: {len(fema_ae):,}')

    print(f'\nLoading FAA radar: {RADAR_PATH} ...')
    radar = gpd.read_parquet(RADAR_PATH)
    radar_pts_4326 = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(radar.station_lon, radar.station_lat),
        crs='EPSG:4326',
    )
    radar_pts_5070 = radar_pts_4326.to_crs(5070)
    radar_xy = np.column_stack([
        radar_pts_5070.geometry.x.values, radar_pts_5070.geometry.y.values
    ])
    radar_tree = cKDTree(radar_xy)
    print(f'  {len(radar):,} radar stations')

    print('\nLoading NWI per state ...')
    nwi_by_state = {}
    for st in STATES:
        for p in (NWI_DIR / f'exclusion_nwi_{st.lower()}.parquet',
                  NWI_DIR / f'exclusion_nwi_{st.lower()}_test.parquet'):
            if p.exists():
                w = gpd.read_parquet(p)
                if w.crs.to_epsg() != 5070:
                    w = w.to_crs(5070)
                nwi_by_state[st] = w
                print(f'  {st}: {len(w):,} wetlands')
                break

    out_rows = []
    for st in STATES:
        sd = cands[cands.state == st].reset_index(drop=True)
        if len(sd) == 0:
            continue
        print(f'\n-- {st} : {len(sd):,} reuse nodes -----------')

        cand_xy = np.column_stack([
            sd.geometry.centroid.x.values, sd.geometry.centroid.y.values
        ])

        st_padus = padus[padus.state == st].reset_index(drop=True)
        print(f'  PAD-US ({len(st_padus):,}) ...', flush=True)
        t0 = time.time()
        d_padus = vectorized_nearest_distance(sd, st_padus)
        print(f'    {time.time()-t0:.1f}s')

        st_wet = nwi_by_state.get(st)
        if st_wet is not None and len(st_wet):
            print(f'  Wetlands ({len(st_wet):,}) ...', flush=True)
            t0 = time.time()
            d_wet = vectorized_nearest_distance(sd, st_wet)
            print(f'    {time.time()-t0:.1f}s')
        else:
            d_wet = np.full(len(sd), np.nan)

        st_bbox = sd.total_bounds
        pad_box = 12_000
        env = box(st_bbox[0]-pad_box, st_bbox[1]-pad_box,
                  st_bbox[2]+pad_box, st_bbox[3]+pad_box)
        st_fw = floodway[floodway.geometry.intersects(env)]
        print(f'  Floodway ({len(st_fw):,}) ...', flush=True)
        t0 = time.time()
        d_fw = vectorized_nearest_distance(sd, st_fw)
        print(f'    {time.time()-t0:.1f}s')

        st_ae = fema_ae[fema_ae.geometry.intersects(env)]
        print(f'  FEMA AE ({len(st_ae):,}) ...', flush=True)
        t0 = time.time()
        d_ae = vectorized_nearest_distance(sd, st_ae)
        print(f'    {time.time()-t0:.1f}s')

        print(f'  Radar (KDTree) ...', flush=True)
        t0 = time.time()
        dr, _ = radar_tree.query(cand_xy, k=1)
        dr = np.where(dr > MAX_SEARCH_M, np.nan, dr)
        print(f'    {time.time()-t0:.1f}s')

        out_rows.append(pd.DataFrame({
            'candidate_id':                sd.candidate_id.values,
            'nearest_padus_distance_m':    d_padus,
            'nearest_wetland_distance_m':  d_wet,
            'nearest_floodway_distance_m': d_fw,
            'nearest_fema_ae_distance_m':  d_ae,
            'nearest_radar_distance_m':    dr,
        }))

    out = pd.concat(out_rows, ignore_index=True)
    assert len(out) == len(cands), f'lost rows: expected {len(cands)}, got {len(out)}'

    out['near_padus_flag']        = out.nearest_padus_distance_m.fillna(1e9) < NEAR_500_M
    out['near_wetland_flag']      = out.nearest_wetland_distance_m.fillna(1e9) < NEAR_500_M
    out['adjacent_floodway_flag'] = out.nearest_floodway_distance_m.fillna(1e9) < NEAR_500_M
    out['fema_ae_overlap_flag']   = out.nearest_fema_ae_distance_m.fillna(1e9) <= 0
    out['fema_ae_adjacent_flag']  = out.nearest_fema_ae_distance_m.fillna(1e9) < NEAR_500_M
    out['radar_distance_miles']   = out.nearest_radar_distance_m / 1609.34
    out['radar_review_flag']      = out.nearest_radar_distance_m.fillna(1e9) < RADAR_NEAR_M

    out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.2f} MB, {len(out):,} rows)')

    print('\n=== Flag counts ===')
    for c in ['near_padus_flag','near_wetland_flag','adjacent_floodway_flag',
              'fema_ae_overlap_flag','fema_ae_adjacent_flag','radar_review_flag']:
        n = int(out[c].sum())
        print(f'  {c:<28} {n:>5,} ({100*n/len(out):5.1f}%)')

    print('\n=== Checks ===')
    checks = {
        'Has all reuse nodes'         : len(out) == len(cands),
        'Unique candidate_ids'        : out.candidate_id.is_unique,
        'PAD-US dist non-neg or NaN'  : (out.nearest_padus_distance_m.fillna(0) >= 0).all(),
        'Wetland dist non-neg or NaN' : (out.nearest_wetland_distance_m.fillna(0) >= 0).all(),
        'Floodway dist non-neg or NaN': (out.nearest_floodway_distance_m.fillna(0) >= 0).all(),
        'FEMA AE dist non-neg or NaN' : (out.nearest_fema_ae_distance_m.fillna(0) >= 0).all(),
        'Radar dist non-neg or NaN'   : (out.nearest_radar_distance_m.fillna(0) >= 0).all(),
        'overlap implies adjacent'    : ((~out.fema_ae_overlap_flag) | out.fema_ae_adjacent_flag).all(),
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
