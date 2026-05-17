"""
Step 1C — Drought level per candidate (point-in-polygon on NOAA D0-D4 zones).

Reads:
  candidate_areas/outputs/candidate_areas.parquet
  ingestion_scripts/noaa_drought/drought_zones.parquet
    columns: dm_level (0=D0, 1=D1, 2=D2, 3=D3, 4=D4), dm_label, is_d4, geometry

For each candidate centroid, find the highest dm_level polygon that
contains it. "None" if the centroid falls outside every drought polygon
(meaning the area is currently not in drought).

Drought zones overlap by design (D0 includes everything that's at least
abnormally dry, D1 includes D2+D3+D4, etc.). We take the maximum
dm_level among polygons containing the point so a candidate that sits
inside both a D0 polygon and a D3 polygon gets labeled D3.

Output:
  candidate_areas/enrichment_outputs/step1c_drought.parquet

Columns added per candidate:
  drought_level  None / D0 / D1 / D2 / D3 / D4
  drought_label  no_drought / abnormally_dry / moderate_drought / severe_drought / extreme_drought / exceptional_drought

Run:
  python candidate_areas/enrichment_scripts/Step1C_drought.py
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd

CAND_PATH    = Path('candidate_areas/outputs/candidate_areas.parquet')
DROUGHT_PATH = Path('ingestion_scripts/noaa_drought/drought_zones.parquet')
OUT_PATH     = Path('candidate_areas/enrichment_outputs/step1c_drought.parquet')

# Drought Monitor numeric → label and our scoring label
DM_LEVEL_LABELS = {
    0: ('D0', 'abnormally_dry'),
    1: ('D1', 'moderate_drought'),
    2: ('D2', 'severe_drought'),
    3: ('D3', 'extreme_drought'),
    4: ('D4', 'exceptional_drought'),
}


def main():
    print(f'Loading candidates: {CAND_PATH} ...')
    cands = gpd.read_parquet(CAND_PATH)
    print(f'  {len(cands):,} candidates in EPSG:{cands.crs.to_epsg()}')

    print('Computing centroids in EPSG:4326 ...')
    centroids_5070 = cands.geometry.centroid
    centroids_4326 = gpd.GeoSeries(centroids_5070, crs=cands.crs).to_crs(4326)
    pts = gpd.GeoDataFrame(
        {'candidate_id': cands.candidate_id.values, '_idx': range(len(cands))},
        geometry=centroids_4326,
        crs='EPSG:4326',
    )

    print(f'Loading drought zones: {DROUGHT_PATH} ...')
    dr = gpd.read_parquet(DROUGHT_PATH)
    print(f'  {len(dr):,} drought polygons in EPSG:{dr.crs.to_epsg()}')
    print(f'  dm_level distribution: {dr.dm_level.value_counts().sort_index().to_dict()}')

    # Spatial join: each point may match 0..N polygons (zones overlap)
    print('Running spatial join (point-in-polygon, sjoin within) ...')
    joined = gpd.sjoin(pts, dr[['dm_level', 'geometry']], how='left', predicate='within')

    # For each candidate, take the MAX dm_level among matching polygons
    print('Reducing to max drought level per candidate ...')
    grouped = joined.groupby('_idx')['dm_level'].max()
    # Some candidates have no matching polygons → NaN, which means no drought
    max_levels = grouped.reindex(range(len(cands))).reset_index()
    max_levels.columns = ['_idx', 'dm_level']

    # Build the output
    out = pd.DataFrame({
        'candidate_id': cands.candidate_id.values,
        'dm_level':     max_levels.dm_level.values,
    })

    def to_level(v):
        if pd.isna(v):
            return None
        return DM_LEVEL_LABELS[int(v)][0]

    def to_label(v):
        if pd.isna(v):
            return 'no_drought'
        return DM_LEVEL_LABELS[int(v)][1]

    out['drought_level'] = out.dm_level.apply(to_level)
    out['drought_label'] = out.dm_level.apply(to_label)
    out = out.drop(columns=['dm_level'])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f'\nSaved: {OUT_PATH} ({size_mb:.2f} MB, {len(out):,} rows)')

    # ---- Distribution ----
    print('\n=== Drought distribution overall ===')
    print(out.drought_label.value_counts().reindex(
        ['no_drought','abnormally_dry','moderate_drought','severe_drought','extreme_drought','exceptional_drought']
    ).to_string())

    print('\n=== Per-state drought distribution ===')
    cdf = cands[['candidate_id','state']].merge(out, on='candidate_id')
    for st in ['TX','VA','AZ','NV','CA']:
        sd = cdf[cdf.state == st]
        c = sd.drought_label.value_counts()
        print(f'  {st}: n={len(sd):>6,}  '
              + ' '.join(f'{lbl[:6]}={c.get(lbl,0):>5}' for lbl in
                         ['no_drought','abnormally_dry','moderate_drought',
                          'severe_drought','extreme_drought','exceptional_drought']))

    # ---- Checks ----
    print('\n=== Checks ===')
    valid_labels = {'no_drought','abnormally_dry','moderate_drought','severe_drought','extreme_drought','exceptional_drought'}
    valid_levels = {None,'D0','D1','D2','D3','D4'}
    checks = {
        'Has all candidates'        : len(out) == len(cands),
        'Unique candidate_ids'      : out.candidate_id.is_unique,
        'drought_label never null'  : out.drought_label.notna().all(),
        'drought_label valid set'   : set(out.drought_label.unique()) <= valid_labels,
        'drought_level valid set'   : set(out.drought_level.unique()) <= valid_levels,
        'level=None iff label=no_drought': ((out.drought_level.isna()) == (out.drought_label == 'no_drought')).all(),
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False

    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
