"""
Step 2B — Subscore 1: Utility Infrastructure (40% of composite).

Implements scoring_design.md Section 2. For each candidate, sum the
contributions of every anchor within 50 km (already pre-computed per-pair
in Step 1I), cap at 100, and store as utility_score.

Per-anchor contribution formula (already computed in Step 1I pairs file):
  anchor_contribution = base_score                  # 0.6*queue_mw_tier + 0.4*status
                      × voltage_multiplier         # 1.20 / 1.10 / 1.05 / 1.00
                      × distance_decay_weight      # banded 1.00/0.67/0.33/0.10
                      × match_confidence_weight    # 1.00/0.70/0.50 (×0.80 if line POI heavy)

Aggregation per candidate:
  utility_score = min(100, sum(anchor_contribution for each anchor in range))

Zone fallback (Step 1I): if candidate had no named anchors in 50 km,
zone_fallback_used=True. For Phase 1 we treat these candidates with a
floor utility_score of 0 — the scoring engine flags them with a
'utility_module_status' field. (The PRD's zone-level synthetic anchor
math is deferred to a follow-up pass; for Friday we just expose the gap
so analysts know the candidate has no measured power signal.)

Reads:
  candidate_areas/outputs/candidate_areas_enriched.parquet
  candidate_areas/enrichment_outputs/step1i_utility_anchor_pairs.parquet

Writes:
  candidate_areas/outputs/candidate_areas_enriched.parquet
    Adds: utility_score (float, 0-100)
          utility_module_status (string: complete / zone_fallback / failed)

Run:
  python candidate_areas/scoring_scripts/Step2B_subscore1_utility.py
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')
PAIRS_PATH    = Path('candidate_areas/enrichment_outputs/step1i_utility_anchor_pairs.parquet')

UTILITY_SCORE_CAP = 100.0


def main():
    print(f'Loading enriched candidates: {ENRICHED_PATH} ...')
    df = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(df):,} candidates, {len(df.columns)} columns')

    print(f'\nLoading anchor pairs: {PAIRS_PATH} ...')
    pairs = pd.read_parquet(PAIRS_PATH)
    print(f'  {len(pairs):,} (candidate, anchor) pairs')

    # Aggregate: sum anchor_contribution per candidate, cap at 100
    print('\nAggregating utility_score per candidate ...')
    agg = pairs.groupby('candidate_id', as_index=False).agg(
        utility_score_raw=('anchor_contribution', 'sum'),
        anchors_count=('anchor_id', 'count'),
    )
    agg['utility_score'] = agg['utility_score_raw'].clip(upper=UTILITY_SCORE_CAP)

    # Merge into the enriched df. Candidates with no anchor pairs get utility_score = 0
    df = df.merge(
        agg[['candidate_id', 'utility_score']],
        on='candidate_id', how='left'
    )
    df['utility_score'] = df['utility_score'].fillna(0.0)

    # utility_module_status per scoring_design 2.9
    def module_status(row):
        if row['num_anchors_in_range'] > 0:
            return 'complete'
        if row.get('zone_fallback_used', False):
            return 'zone_fallback'
        return 'failed'
    df['utility_module_status'] = df.apply(module_status, axis=1)

    # Save
    df.to_parquet(ENRICHED_PATH, index=False)
    size_mb = ENRICHED_PATH.stat().st_size / 1e6
    print(f'\nUpdated: {ENRICHED_PATH} ({size_mb:.1f} MB)')

    # --- Summary ---
    print('\n=== utility_score distribution ===')
    s = df.utility_score
    print(f'  min={s.min():.2f}, p10={s.quantile(0.1):.2f}, median={s.median():.2f}, '
          f'p90={s.quantile(0.9):.2f}, max={s.max():.2f}')

    # Distribution bands
    print('\n=== utility_score bands ===')
    bands = pd.cut(s, [-0.01, 10, 25, 50, 75, 100.01],
                   labels=['<10','10-25','25-50','50-75','75-100'])
    print(bands.value_counts().reindex(['<10','10-25','25-50','50-75','75-100']).to_string())

    print('\n=== utility_module_status ===')
    print(df.utility_module_status.value_counts().to_string())

    print('\n=== Per-state median utility_score ===')
    for st in ['AZ','CA','NV','TX','VA']:
        sd = df[df.state == st]
        print(f'  {st}: n={len(sd):>6,}  median={sd.utility_score.median():>6.2f}  '
              f'p90={sd.utility_score.quantile(0.9):>6.2f}  '
              f'zone_fallback={(sd.utility_module_status=="zone_fallback").sum():>5,}')

    # Spot-checks
    print('\n=== Top 10 utility_score candidates (sanity check) ===')
    top = df.sort_values('utility_score', ascending=False).head(10)
    cols = ['candidate_id','state','county_name','area_acres',
            'utility_score','num_anchors_in_range','primary_anchor_name',
            'primary_anchor_distance_m','primary_anchor_voltage_kv',
            'primary_anchor_queue_mw_tier','ia_executed_nearby']
    print(top[cols].to_string(index=False))

    # Checks
    print('\n=== Checks ===')
    checks = {
        'Has rows'                        : len(df) > 0,
        'utility_score in [0, 100]'       : ((df.utility_score >= 0) & (df.utility_score <= 100.001)).all(),
        'utility_score never null'        : df.utility_score.notna().all(),
        'utility_module_status valid set' : set(df.utility_module_status.unique()) <= {'complete','zone_fallback','failed'},
        'zone_fallback iff anchors=0'     : ((df.utility_module_status == 'zone_fallback') == (df.num_anchors_in_range == 0)).all(),
        'utility_score=0 when no anchors' : (df[df.num_anchors_in_range == 0].utility_score == 0).all(),
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
