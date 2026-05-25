"""
StepR5A -- Kill gates for reuse nodes (candidate_status + reason).

Mirror of Step2A_kill_gates.py adapted for reuse-node attributes.

Greenfield used three gates (building_footprint > 5%, slope_mean > 15%,
buildable_ratio < 0.25). For reuse nodes:
  * building_footprint_pct does not exist (reuse-node polygons ARE the
    industrial footprint -- the whole polygon is built up by definition).
  * slope_mean_pct is deferred (StepR3A docstring); skip the gate. Reuse
    nodes are already industrially developed land and are nearly always
    flat -- this gate would be near-zero hits anyway.
  * buildable_area_ratio < 0.25 is applicable: it flags reuse-node polygons
    that have a very low fill ratio of their convex hull (irregular polygon
    shapes that may have lost most area to internal voids).

One additional reuse-node-specific gate:
  * geometry_confidence == 'LOW' -- ALREADY filtered upstream in R2b, so
    this gate will be 0 hits. Kept as a defensive check.

Status values: pass / excluded / manual_review (same vocabulary as Step2A).

Run: python candidate_areas/reuse_node_scripts/StepR5A_kill_gates.py
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')


def main():
    print(f'Loading enriched reuse nodes: {ENRICHED_PATH} ...')
    df = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(df):,} rows, {len(df.columns)} columns')

    print('\nComputing kill gates ...')

    # Gate 1: heavily clipped polygons -- buildable_ratio < 0.25
    g1_mask = df['buildable_area_ratio'].fillna(1.0) < 0.25
    print(f'  Gate 1 (buildable_ratio < 0.25):  {int(g1_mask.sum()):,}')

    # Gate 2: defensive -- any LOW confidence row that survived R2b
    g2_mask = df['geometry_confidence'] == 'LOW'
    print(f'  Gate 2 (geometry_confidence=LOW): {int(g2_mask.sum()):,}')

    review_mask = g1_mask | g2_mask
    print(f'  Union (any manual_review trigger): {int(review_mask.sum()):,}')

    def status_for_row(g1, g2):
        if g1 or g2:
            reasons = []
            if g1: reasons.append('buildable_ratio_<25pct')
            if g2: reasons.append('geometry_confidence_LOW')
            return 'manual_review', ';'.join(reasons)
        return 'pass', None

    statuses = [
        status_for_row(g1, g2)
        for g1, g2 in zip(g1_mask.values, g2_mask.values)
    ]
    df['candidate_status']        = [s[0] for s in statuses]
    df['candidate_status_reason'] = [s[1] for s in statuses]

    df.to_parquet(ENRICHED_PATH, index=False)
    print(f'\nUpdated: {ENRICHED_PATH} ({ENRICHED_PATH.stat().st_size/1e6:.1f} MB)')

    print('\n=== candidate_status distribution ===')
    print(df.candidate_status.value_counts().to_string())

    print('\n=== Per-state breakdown ===')
    for st in sorted(df.state.unique()):
        sd = df[df.state == st]
        n_review = int((sd.candidate_status == 'manual_review').sum())
        print(f'  {st}: total={len(sd):>5,}  pass={(sd.candidate_status=="pass").sum():>5,}  '
              f'manual_review={n_review:>4,} ({100*n_review/len(sd):.2f}%)')

    print('\n=== Checks ===')
    checks = {
        'Has rows'                            : len(df) > 0,
        'candidate_status never null'         : df.candidate_status.notna().all(),
        'status in {pass,excluded,manual_review}': set(df.candidate_status.unique()) <= {'pass','excluded','manual_review'},
        'reason matches status'               : (
            (df.candidate_status == 'pass').eq(df.candidate_status_reason.isna())
        ).all(),
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
