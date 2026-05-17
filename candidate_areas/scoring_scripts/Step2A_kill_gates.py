"""
Step 2A — Kill gates (candidate_status filter before composite scoring).

Per Owen Rec #17 in his Phase 1 review. The kill gates run BEFORE composite
scoring so structurally-bad candidates don't get a misleading score.

Reads:
  candidate_areas/outputs/candidate_areas_enriched.parquet  (95,269 rows, 84 cols)

Writes:
  candidate_areas/outputs/candidate_areas_enriched.parquet
    Adds two columns: candidate_status, candidate_status_reason

Gates per Owen Rec #17
----------------------
The PRD's gates assume some conditions may STILL be present in the candidate
set even after upstream exclusions. In our pipeline almost all of those gates
were enforced upstream — every candidate already cleared:
  * hard exclusions (slope >15%, floodway, PAD-US GAP 1/2, open water,
    wetland-heavy via >10% coverage threshold)
  * minimum acreage 50+

So most of these gates will yield 0 hits for us. That's expected.

What we DO catch here:
  1. Building footprint > 5% of candidate area — these were a known
     post-exclusion outlier (the 20 worst cases per Step A's verification).
     Per the user's call: SCORE them low rather than exclude — but flag
     for manual_review so analysts know to look.

  2. Slope mean > 15% — these slipped past the >15% hard exclusion due to
     Step5e's polygon simplification. 4 candidates (per Step1A verification).
     manual_review.

  3. Very small NET area (after exclusions) when original area was large —
     i.e., buildable_area_ratio < 0.25 means the candidate is a tiny sliver
     of what it might have been. 3 candidates per Step1J. manual_review.

  4. ~Phase 2 only: reuse-node environmental uncertainty — N/A in Phase 1
     since we deferred reuse nodes per user decision.

Status values
-------------
  pass           — fully clears all gates, will be scored normally
  excluded       — would only be set for hard reasons we caught here
                   (currently we have none; reserved for the schema)
  manual_review  — composite_score WILL still be computed and the row
                   appears in output, but with this flag so an analyst
                   visually checks before action

Run:
  python candidate_areas/scoring_scripts/Step2A_kill_gates.py
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')


def main():
    print(f'Loading enriched candidates: {ENRICHED_PATH} ...')
    df = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(df):,} candidates, {len(df.columns)} columns')

    # --- Gate computation ---
    print('\nComputing kill gates ...')

    # Gate 1: building footprint > 5%
    g1_mask = df['building_footprint_pct'].fillna(0) > 5.0
    print(f'  Gate 1 (building_footprint > 5%):      {g1_mask.sum():,}')

    # Gate 2: slope_mean > 15% — the WHOLE polygon is steep, not just a corner.
    # Cases where only slope_max > 15% are already penalized via slope_tier
    # in Step 1A; they don't need additional manual_review. We only flag
    # candidates where the mean slope itself exceeds the threshold.
    g2_mask = df['slope_mean_pct'].fillna(0) > 15.0
    print(f'  Gate 2 (slope_mean > 15%):             {g2_mask.sum():,}')

    # Gate 3: heavily clipped (buildable_area_ratio < 0.25)
    g3_mask = df['buildable_area_ratio'].fillna(1.0) < 0.25
    print(f'  Gate 3 (buildable_ratio < 0.25):       {g3_mask.sum():,}')

    # Combined manual_review mask (any of g1/g2/g3)
    review_mask = g1_mask | g2_mask | g3_mask
    print(f'  Union (any manual_review trigger):     {review_mask.sum():,}')

    # Compute status + reason string per candidate
    def status_for_row(g1, g2, g3):
        if g1 or g2 or g3:
            reasons = []
            if g1: reasons.append('building_footprint_>5pct')
            if g2: reasons.append('slope_>15pct')
            if g3: reasons.append('buildable_ratio_<25pct')
            return 'manual_review', ';'.join(reasons)
        return 'pass', None

    statuses = [
        status_for_row(g1, g2, g3)
        for g1, g2, g3 in zip(g1_mask.values, g2_mask.values, g3_mask.values)
    ]
    df['candidate_status']        = [s[0] for s in statuses]
    df['candidate_status_reason'] = [s[1] for s in statuses]

    # --- Save back ---
    df.to_parquet(ENRICHED_PATH, index=False)
    print(f'\nUpdated: {ENRICHED_PATH}')
    size_mb = ENRICHED_PATH.stat().st_size / 1e6
    print(f'  {size_mb:.1f} MB')

    # --- Summary ---
    print('\n=== candidate_status distribution ===')
    print(df.candidate_status.value_counts().to_string())

    print('\n=== reasons among manual_review ===')
    print(df.candidate_status_reason.value_counts().head(10).to_string())

    print('\n=== Per-state breakdown ===')
    for st in sorted(df.state.unique()):
        sd = df[df.state == st]
        n_review = (sd.candidate_status == 'manual_review').sum()
        print(f'  {st}: total={len(sd):>6,}  pass={(sd.candidate_status=="pass").sum():>6,}  '
              f'manual_review={n_review:>5,} ({100*n_review/len(sd):.2f}%)')

    print('\n=== Checks ===')
    checks = {
        'Has rows'                        : len(df) > 0,
        'candidate_status never null'     : df.candidate_status.notna().all(),
        'status in {pass,excluded,manual_review}': set(df.candidate_status.unique()) <= {'pass','excluded','manual_review'},
        'reason matches status'           : (
            (df.candidate_status == 'pass').eq(df.candidate_status_reason.isna())
        ).all(),
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
