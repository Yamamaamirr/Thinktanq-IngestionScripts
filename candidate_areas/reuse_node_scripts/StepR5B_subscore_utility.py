"""
StepR5B -- Utility subscore for reuse nodes (25% of composite, recalibrated).

Mirror of Step2B_subscore1_utility.py adapted for reuse nodes, with one
calibration change vs greenfield:

  PER-ANCHOR CAP. The greenfield formula sums anchor_contributions
  per candidate and caps the SUM at 100. Reuse nodes by definition sit
  near grid infrastructure, so a single strong anchor (500 kV substation
  with executed-IA queue) can hit ~120 by itself and pin utility_score
  at 100. ~78% of reuse nodes ended up pinned at 100 with the original
  formula, killing all discrimination among "good" sites.

  We cap PER-ANCHOR contribution at 30. Now a site needs 4+ good anchors
  in range to max out, which restores variance:
    1 anchor in range  -> max util ~= 30
    2 anchors          -> max util ~= 60
    3 anchors          -> max util ~= 90
    4+ anchors         -> capped at 100

  This converts "any decent anchor" into "anchor density," which is the
  real signal: a site surrounded by multiple substations + queue activity
  is genuinely better than a site near one strong substation.

Adds:
  utility_score (float, 0-100)
  utility_module_status (complete / zone_fallback / failed)

Run: python candidate_areas/reuse_node_scripts/StepR5B_subscore_utility.py
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')
PAIRS_PATH    = Path('candidate_areas/reuse_node_enrichment_outputs/stepR3i_utility_anchor_pairs.parquet')

UTILITY_SCORE_CAP   = 100.0
PER_ANCHOR_CAP      = 30.0


def main():
    print(f'Loading: {ENRICHED_PATH} ...')
    df = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(df):,} rows, {len(df.columns)} columns')

    print(f'\nLoading anchor pairs: {PAIRS_PATH} ...')
    pairs = pd.read_parquet(PAIRS_PATH)
    print(f'  {len(pairs):,} pairs')
    print(f'  Per-anchor contribution stats BEFORE cap: '
          f'median={pairs.anchor_contribution.median():.2f}, '
          f'p90={pairs.anchor_contribution.quantile(0.9):.2f}, '
          f'max={pairs.anchor_contribution.max():.2f}')

    print(f'\nApplying per-anchor cap of {PER_ANCHOR_CAP} ...')
    pairs = pairs.copy()
    pairs['anchor_contribution_capped'] = pairs.anchor_contribution.clip(upper=PER_ANCHOR_CAP)
    n_capped = int((pairs.anchor_contribution > PER_ANCHOR_CAP).sum())
    print(f'  {n_capped:,} of {len(pairs):,} pairs were trimmed ({100*n_capped/len(pairs):.1f}%)')

    agg = pairs.groupby('candidate_id', as_index=False).agg(
        utility_score_raw=('anchor_contribution_capped', 'sum'),
    )
    agg['utility_score'] = agg['utility_score_raw'].clip(upper=UTILITY_SCORE_CAP)

    # Drop any prior utility_score / utility_module_status columns to avoid suffix collisions
    df = df.drop(columns=[c for c in ['utility_score','utility_module_status'] if c in df.columns])
    df = df.merge(agg[['candidate_id', 'utility_score']], on='candidate_id', how='left')
    df['utility_score'] = df['utility_score'].fillna(0.0)

    def module_status(row):
        if row['num_anchors_in_range'] > 0:
            return 'complete'
        if row.get('zone_fallback_used', False):
            return 'zone_fallback'
        return 'failed'
    df['utility_module_status'] = df.apply(module_status, axis=1)

    df.to_parquet(ENRICHED_PATH, index=False)
    print(f'\nUpdated: {ENRICHED_PATH} ({ENRICHED_PATH.stat().st_size/1e6:.1f} MB)')

    s = df.utility_score
    print(f'\n=== utility_score: min={s.min():.2f}, p10={s.quantile(0.1):.2f}, '
          f'median={s.median():.2f}, p90={s.quantile(0.9):.2f}, max={s.max():.2f}')

    print('\n=== utility_module_status ===')
    print(df.utility_module_status.value_counts().to_string())

    print('\n=== Top 10 utility_score reuse nodes ===')
    top = df.sort_values('utility_score', ascending=False).head(10)
    cols = ['candidate_id','source','state','utility_score','num_anchors_in_range',
            'primary_anchor_name','primary_anchor_distance_m','primary_anchor_voltage_kv']
    print(top[cols].to_string(index=False))

    print('\n=== Checks ===')
    checks = {
        'Has rows'                       : len(df) > 0,
        'utility_score in [0,100]'       : ((df.utility_score >= 0) & (df.utility_score <= 100.001)).all(),
        'utility_score never null'       : df.utility_score.notna().all(),
        'utility_module_status valid'    : set(df.utility_module_status.unique()) <= {'complete','zone_fallback','failed'},
        'zone_fallback iff anchors=0'    : ((df.utility_module_status == 'zone_fallback') == (df.num_anchors_in_range == 0)).all(),
        'utility=0 when no anchors'      : (df[df.num_anchors_in_range == 0].utility_score == 0).all(),
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
