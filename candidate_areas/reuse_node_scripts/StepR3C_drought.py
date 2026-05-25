"""
StepR3C -- Drought level per reuse node (point-in-polygon on NOAA D0-D4).

Mirror of Step1C_drought.py for reuse_nodes_clean.parquet.

Output columns identical to Step1C:
  drought_level (None|D0..D4), drought_label

Run: python candidate_areas/reuse_node_scripts/StepR3C_drought.py
"""
from pathlib import Path
import sys
import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _r3_helpers import load_reuse_nodes_as_candidates, out_path

DROUGHT_PATH = Path('ingestion_scripts/noaa_drought/drought_zones.parquet')
OUT_PATH     = out_path('stepR3c_drought.parquet')

DM_LEVEL_LABELS = {
    0: ('D0', 'abnormally_dry'),
    1: ('D1', 'moderate_drought'),
    2: ('D2', 'severe_drought'),
    3: ('D3', 'extreme_drought'),
    4: ('D4', 'exceptional_drought'),
}


def main():
    print('Loading reuse nodes ...')
    cands = load_reuse_nodes_as_candidates(crs_epsg=5070)
    print(f'  {len(cands):,} reuse nodes')

    centroids_5070 = cands.geometry.centroid
    centroids_4326 = gpd.GeoSeries(centroids_5070, crs=cands.crs).to_crs(4326)
    pts = gpd.GeoDataFrame(
        {'candidate_id': cands.candidate_id.values, '_idx': range(len(cands))},
        geometry=centroids_4326, crs='EPSG:4326',
    )

    print(f'Loading drought zones: {DROUGHT_PATH} ...')
    dr = gpd.read_parquet(DROUGHT_PATH)
    print(f'  {len(dr):,} drought polygons')

    print('Running spatial join ...')
    joined = gpd.sjoin(pts, dr[['dm_level', 'geometry']], how='left', predicate='within')

    grouped = joined.groupby('_idx')['dm_level'].max()
    max_levels = grouped.reindex(range(len(cands))).reset_index()
    max_levels.columns = ['_idx', 'dm_level']

    out = pd.DataFrame({
        'candidate_id': cands.candidate_id.values,
        'dm_level':     max_levels.dm_level.values,
    })
    out['drought_level'] = out.dm_level.apply(lambda v: None if pd.isna(v) else DM_LEVEL_LABELS[int(v)][0])
    out['drought_label'] = out.dm_level.apply(lambda v: 'no_drought' if pd.isna(v) else DM_LEVEL_LABELS[int(v)][1])
    out = out.drop(columns=['dm_level'])

    out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.2f} MB, {len(out):,} rows)')

    print('\n=== Drought distribution ===')
    print(out.drought_label.value_counts().reindex(
        ['no_drought','abnormally_dry','moderate_drought','severe_drought','extreme_drought','exceptional_drought']
    ).to_string())

    valid_labels = {'no_drought','abnormally_dry','moderate_drought','severe_drought','extreme_drought','exceptional_drought'}
    valid_levels = {None,'D0','D1','D2','D3','D4'}
    print('\n=== Checks ===')
    checks = {
        'Has all reuse nodes'      : len(out) == len(cands),
        'Unique candidate_ids'     : out.candidate_id.is_unique,
        'drought_label never null' : out.drought_label.notna().all(),
        'drought_label valid set'  : set(out.drought_label.unique()) <= valid_labels,
        'drought_level valid set'  : set(out.drought_level.unique()) <= valid_levels,
        'level=None iff no_drought': ((out.drought_level.isna()) == (out.drought_label == 'no_drought')).all(),
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
