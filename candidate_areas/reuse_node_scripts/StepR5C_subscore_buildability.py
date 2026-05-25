"""
StepR5C -- Buildability subscore for reuse nodes (20% of composite).

Mirror of Step2C_subscore2_buildability.py with reuse-node adaptations:

  land_cover_score:
    Greenfield uses CDL group (grassland 90 / shrub_barren 80 / cropland 65 /
    forest 45). Reuse nodes have NO cdl_group -- they ARE industrial land by
    definition. We assign a fixed 85.0 -- equivalent to "developable
    industrial-zoned land," sitting between greenfield grassland (90) and
    shrub_barren (80). This is intentionally favourable: the whole point of
    a reuse node is that the land is already cleared and zoned.

  slope_tier_score:
    Real per-reuse-node slope sampling now exists (StepR3A consumes
    slope_per_reuse_node.parquet). For any row where slope is still
    unknown (sampling missed the polygon), we fill with the neutral
    80.0 -- same value greenfield assigns to 'acceptable' slope.

  constraint_score:
    Greenfield stacks FEMA AE penalty + building_footprint penalty.
    Reuse nodes have no building_footprint_pct (the polygon IS the building
    footprint by definition). We keep the FEMA AE penalty only.

Adds:
  land_cover_score, constraint_score, buildability_score,
  buildability_review_required

Run: python candidate_areas/reuse_node_scripts/StepR5C_subscore_buildability.py
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')

LAND_COVER_SCORE_DEFAULT = 85.0   # reuse-node industrial land
SLOPE_TIER_SCORE_DEFAULT = 80.0   # neutral "acceptable" until slope sampled


def constraint_score_row(row):
    score = 100.0
    if row.get('fema_ae_overlap_flag', False):
        score -= 30.0
    elif row.get('fema_ae_adjacent_flag', False):
        score -= 10.0
    return max(0.0, score)


def main():
    print(f'Loading: {ENRICHED_PATH} ...')
    df = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(df):,} rows, {len(df.columns)} columns')

    print('\nComputing land_cover_score (fixed at 85.0 for industrial reuse land) ...')
    df['land_cover_score'] = LAND_COVER_SCORE_DEFAULT

    print('Filling slope_tier_score (deferred -> neutral 80.0) ...')
    df['slope_tier_score'] = df['slope_tier_score'].fillna(SLOPE_TIER_SCORE_DEFAULT)

    print('Computing constraint_score (FEMA AE only) ...')
    df['constraint_score'] = df.apply(constraint_score_row, axis=1)
    print(f'  Distribution: min={df.constraint_score.min():.1f}, median={df.constraint_score.median():.1f}, '
          f'p10={df.constraint_score.quantile(0.1):.1f}, max={df.constraint_score.max():.1f}')

    print('\nComputing buildability_score ...')
    df['buildability_score'] = (
        0.35 * df['land_cover_score']
      + 0.25 * df['acreage_tier_score']
      + 0.25 * df['slope_tier_score']
      + 0.15 * df['constraint_score']
    )

    df['buildability_review_required'] = (
        df['slope_review_flag'].fillna(False)
        | df['fema_ae_adjacent_flag'].fillna(False)
    )

    df.to_parquet(ENRICHED_PATH, index=False)
    print(f'\nUpdated: {ENRICHED_PATH} ({ENRICHED_PATH.stat().st_size/1e6:.1f} MB)')

    s = df.buildability_score
    print(f'\n=== buildability_score: min={s.min():.2f}, p10={s.quantile(0.1):.2f}, '
          f'median={s.median():.2f}, p90={s.quantile(0.9):.2f}, max={s.max():.2f}')

    print('\n=== buildability_score bands ===')
    bands = pd.cut(s, [-0.01, 40, 55, 70, 85, 100.01],
                   labels=['<40','40-55','55-70','70-85','85-100'])
    print(bands.value_counts().reindex(['<40','40-55','55-70','70-85','85-100']).to_string())

    print('\n=== Per-state median ===')
    for st in ['AZ','CA','NV','TX','VA']:
        sd = df[df.state == st]
        print(f'  {st}: n={len(sd):>5,}  median={sd.buildability_score.median():>6.2f}  '
              f'p90={sd.buildability_score.quantile(0.9):>6.2f}  '
              f'review_req={int(sd.buildability_review_required.sum()):>4,}')

    print('\n=== Checks ===')
    checks = {
        'Has rows'                          : len(df) > 0,
        'buildability_score in [0,100]'     : ((df.buildability_score >= 0) & (df.buildability_score <= 100.001)).all(),
        'buildability_score never null'     : df.buildability_score.notna().all(),
        'constraint_score in [0,100]'       : ((df.constraint_score >= 0) & (df.constraint_score <= 100)).all(),
        'land_cover_score all 85'           : (df.land_cover_score == 85.0).all(),
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
