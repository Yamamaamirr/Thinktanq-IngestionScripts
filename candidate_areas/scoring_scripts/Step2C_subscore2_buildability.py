"""
Step 2C — Subscore 2: Physical Buildability (20% of composite).

Implements scoring_design.md Section 3.

  buildability_score = 0.35 * land_cover_score
                     + 0.25 * acreage_tier_score
                     + 0.25 * slope_tier_score
                     + 0.15 * constraint_score

All four inputs come from Step 1 enrichment (already in the parquet).
Constraint score stacks FEMA AE penalty + building footprint penalty
from a baseline of 100, floored at 0.

Reads / Writes:
  candidate_areas/outputs/candidate_areas_enriched.parquet

Adds:
  land_cover_score          (component, 0-100)
  constraint_score          (component, 0-100)
  buildability_score        (final, 0-100)
  buildability_review_required (bool — any review_flag is True)

Run:
  python candidate_areas/scoring_scripts/Step2C_subscore2_buildability.py
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')

LAND_COVER_SCORE = {
    'grassland'    : 90.0,
    'shrub_barren' : 80.0,
    'cropland'     : 65.0,
    'forest'       : 45.0,
}


def constraint_score_row(row):
    """Stack FEMA AE + building footprint penalties from 100, floor at 0."""
    score = 100.0
    # FEMA AE penalty (per scoring_design 3.5)
    if row.get('fema_ae_overlap_flag', False):
        score -= 30.0
    elif row.get('fema_ae_adjacent_flag', False):
        score -= 10.0
    # Building footprint penalty (banded)
    bf = row.get('building_footprint_pct', 0) or 0
    if   bf < 0.25:  pass               # no penalty
    elif bf < 1.0:   score -= 10.0
    elif bf < 5.0:   score -= 40.0
    elif bf < 15.0:  score -= 90.0
    else:            score -= 120.0     # caps at 0 via floor
    return max(0.0, score)


def main():
    print(f'Loading enriched candidates: {ENRICHED_PATH} ...')
    df = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(df):,} candidates, {len(df.columns)} columns')

    # 1) Land cover score from CDL group
    print('\nComputing land_cover_score ...')
    df['land_cover_score'] = df['cdl_group'].map(LAND_COVER_SCORE).fillna(50.0)
    print(f'  Distribution: {df.land_cover_score.value_counts().sort_index().to_dict()}')

    # 2) Acreage tier score is already in the parquet (acreage_tier_score from 1A)
    # 3) Slope tier score is already in the parquet (slope_tier_score from 1A)
    # 4) Constraint score
    print('\nComputing constraint_score ...')
    df['constraint_score'] = df.apply(constraint_score_row, axis=1)
    print(f'  Distribution: min={df.constraint_score.min():.1f}, '
          f'median={df.constraint_score.median():.1f}, '
          f'p10={df.constraint_score.quantile(0.1):.1f}, '
          f'max={df.constraint_score.max():.1f}')

    # 5) Compose buildability_score
    print('\nComputing buildability_score ...')
    df['buildability_score'] = (
        0.35 * df['land_cover_score']
      + 0.25 * df['acreage_tier_score']
      + 0.25 * df['slope_tier_score']
      + 0.15 * df['constraint_score']
    )

    # 6) Composite review flag
    df['buildability_review_required'] = (
        df['slope_review_flag'].fillna(False)
        | df['fema_ae_adjacent_flag'].fillna(False)
        | (df['building_footprint_pct'].fillna(0) >= 1.0)
    )

    # Save
    df.to_parquet(ENRICHED_PATH, index=False)
    size_mb = ENRICHED_PATH.stat().st_size / 1e6
    print(f'\nUpdated: {ENRICHED_PATH} ({size_mb:.1f} MB)')

    # --- Summary ---
    print('\n=== buildability_score distribution ===')
    s = df.buildability_score
    print(f'  min={s.min():.2f}, p10={s.quantile(0.1):.2f}, median={s.median():.2f}, '
          f'p90={s.quantile(0.9):.2f}, max={s.max():.2f}')

    bands = pd.cut(s, [-0.01, 40, 55, 70, 85, 100.01],
                   labels=['<40','40-55','55-70','70-85','85-100'])
    print('\n=== buildability_score bands ===')
    print(bands.value_counts().reindex(['<40','40-55','55-70','70-85','85-100']).to_string())

    print('\n=== Per-state median buildability_score ===')
    for st in ['AZ','CA','NV','TX','VA']:
        sd = df[df.state == st]
        print(f'  {st}: n={len(sd):>6,}  median={sd.buildability_score.median():>6.2f}  '
              f'p90={sd.buildability_score.quantile(0.9):>6.2f}  '
              f'review_req={sd.buildability_review_required.sum():>6,}')

    # Component breakdown verification
    print('\n=== Component score medians (per state) ===')
    print(f'  {"state":<5} {"land_cover":>11} {"acreage":>10} {"slope":>10} {"constraint":>11}')
    for st in ['AZ','CA','NV','TX','VA']:
        sd = df[df.state == st]
        print(f'  {st:<5} {sd.land_cover_score.median():>11.1f} {sd.acreage_tier_score.median():>10.1f} '
              f'{sd.slope_tier_score.median():>10.1f} {sd.constraint_score.median():>11.1f}')

    print('\n=== Checks ===')
    checks = {
        'Has rows'                        : len(df) > 0,
        'buildability_score in [0, 100]'  : ((df.buildability_score >= 0) & (df.buildability_score <= 100.001)).all(),
        'buildability_score never null'   : df.buildability_score.notna().all(),
        'land_cover_score in valid set'   : set(df.land_cover_score.unique()) <= {45.0, 50.0, 65.0, 80.0, 90.0},
        'constraint_score in [0, 100]'    : ((df.constraint_score >= 0) & (df.constraint_score <= 100)).all(),
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
