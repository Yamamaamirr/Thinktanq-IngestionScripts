"""
StepR5D -- Supporting infrastructure subscore for reuse nodes (15% of composite).

Mirror of Step2D_subscore3_supporting_infra.py. Identical scoring bands
since the underlying distance fields are identical schema between greenfield
and reuse-node enrichment.

Adds:
  transmission_score, pipeline_score, rail_score, water_score,
  supporting_infra_score

Run: python candidate_areas/reuse_node_scripts/StepR5D_subscore_supporting_infra.py
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')
MI_TO_M = 1609.34


def distance_to_score_transmission(dist_m, voltage_class):
    if pd.isna(dist_m):
        return 0.0
    if voltage_class == '500kv':
        scores = [100.0, 90.0, 75.0, 55.0, 35.0, 15.0, 0.0]
    elif voltage_class == '345kv':
        scores = [90.0, 80.0, 65.0, 45.0, 25.0, 10.0, 0.0]
    else:
        scores = [90.0, 80.0, 65.0, 45.0, 25.0, 10.0, 0.0]
    if dist_m == 0:           return scores[0]
    if dist_m <= 1*MI_TO_M:   return scores[1]
    if dist_m <= 3*MI_TO_M:   return scores[2]
    if dist_m <= 5*MI_TO_M:   return scores[3]
    if dist_m <= 10*MI_TO_M:  return scores[4]
    if dist_m <= 25*MI_TO_M:  return scores[5]
    return scores[6]


def transmission_score_row(row):
    s500 = distance_to_score_transmission(row.get('nearest_500kv_distance_m'), '500kv')
    s345 = distance_to_score_transmission(row.get('nearest_345kv_distance_m'), '345kv')
    s230 = 0.5 * distance_to_score_transmission(row.get('nearest_230kv_distance_m'), '230kv')
    return max(s500, s345, s230)


def pipeline_tier_score(diameter_in, operator_tier):
    if operator_tier == 'tier_1':
        return 'tier_1'
    if pd.isna(diameter_in):
        return 'other'
    if diameter_in >= 20:
        return 'tier_1_by_diameter'
    if diameter_in >= 14:
        return 'tier_2'
    return 'other'


def pipeline_dist_score(dist_m, tier):
    if pd.isna(dist_m):
        return 0.0
    is_tier1 = tier in ('tier_1', 'tier_1_by_diameter')
    is_tier2 = tier == 'tier_2'
    if dist_m <= 5*MI_TO_M:   return 100.0 if is_tier1 else (75.0 if is_tier2 else 50.0)
    if dist_m <= 10*MI_TO_M:  return 70.0  if is_tier1 else (50.0 if is_tier2 else 30.0)
    if dist_m <= 25*MI_TO_M:  return 40.0  if is_tier1 else (25.0 if is_tier2 else 15.0)
    if dist_m <= 50*MI_TO_M:  return 15.0  if is_tier1 else (10.0 if is_tier2 else  5.0)
    return 0.0


def rail_score_row(row):
    d = row.get('nearest_class1_rail_distance_m')
    if pd.isna(d):
        return 5.0
    if d <= 1*MI_TO_M:   base = 100.0
    elif d <= 3*MI_TO_M: base =  80.0
    elif d <= 5*MI_TO_M: base =  60.0
    elif d <= 10*MI_TO_M:base =  40.0
    elif d <= 25*MI_TO_M:base =  20.0
    else:                base =   5.0
    if row.get('nearest_rail_is_stracnet', False):
        base += 10.0
    n_tracks = row.get('nearest_rail_n_tracks')
    if pd.notna(n_tracks) and n_tracks >= 2:
        base += 5.0
    return min(100.0, base)


def water_score_row(row):
    inside = row.get('within_water_service_area', False)
    if inside:
        base = 100.0
    else:
        d = row.get('nearest_water_service_distance_m')
        if pd.isna(d):       base = 5.0
        elif d <= 1*MI_TO_M:  base = 75.0
        elif d <= 5*MI_TO_M:  base = 50.0
        elif d <= 15*MI_TO_M: base = 25.0
        else:                 base = 5.0
    pop = row.get('nearest_water_service_pop_served', None)
    if pd.notna(pop) and pop > 50_000:    base += 10.0
    elif pd.notna(pop) and pop > 10_000:  base += 5.0
    return min(100.0, base)


def main():
    print(f'Loading: {ENRICHED_PATH} ...')
    df = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(df):,} rows, {len(df.columns)} columns')

    print('\nComputing transmission_score ...')
    df['transmission_score'] = df.apply(transmission_score_row, axis=1)

    print('Computing pipeline_score ...')
    df['_pipe_tier_label'] = df.apply(
        lambda r: pipeline_tier_score(r.get('nearest_pipeline_est_diameter_in'),
                                       r.get('nearest_pipeline_operator_tier')),
        axis=1,
    )
    df['pipeline_score'] = df.apply(
        lambda r: pipeline_dist_score(r.get('nearest_pipeline_distance_m'),
                                       r['_pipe_tier_label']),
        axis=1,
    )

    print('Computing rail_score ...')
    df['rail_score'] = df.apply(rail_score_row, axis=1)

    print('Computing water_score ...')
    df['water_score'] = df.apply(water_score_row, axis=1)

    print('\nComposing supporting_infra_score ...')
    df['supporting_infra_score'] = (
        0.35 * df['transmission_score']
      + 0.25 * df['pipeline_score']
      + 0.20 * df['rail_score']
      + 0.20 * df['water_score']
    )
    df = df.drop(columns=['_pipe_tier_label'])

    df.to_parquet(ENRICHED_PATH, index=False)
    print(f'\nUpdated: {ENRICHED_PATH} ({ENRICHED_PATH.stat().st_size/1e6:.1f} MB)')

    print('\n=== Component medians ===')
    for c in ['transmission_score','pipeline_score','rail_score','water_score','supporting_infra_score']:
        s = df[c]
        print(f'  {c:<28} median={s.median():>6.2f}  p10={s.quantile(0.1):>6.2f}  p90={s.quantile(0.9):>6.2f}')

    print('\n=== supporting_infra_score bands ===')
    bands = pd.cut(df.supporting_infra_score, [-0.01, 20, 40, 60, 80, 100.01],
                   labels=['<20','20-40','40-60','60-80','80-100'])
    print(bands.value_counts().reindex(['<20','20-40','40-60','60-80','80-100']).to_string())

    print('\n=== Checks ===')
    checks = {
        'Has rows'                       : len(df) > 0,
        'supporting_infra in [0,100]'    : ((df.supporting_infra_score >= 0) & (df.supporting_infra_score <= 100.001)).all(),
        'supporting_infra never null'    : df.supporting_infra_score.notna().all(),
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
