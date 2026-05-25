"""
StepR5E -- Development risk subscore for reuse nodes (15% of composite).

Recalibrated after first-pass cross-check showed reuse_environmental had
no influence when buried inside dev_risk at 0.12 weight (composite delta
of only 3.85 pts for contaminated sites). Two changes:

  1) dev_risk_score reverts to the greenfield 6-component formula -- same
     weights as Step2E (seismic 0.25, drought 0.20, radar/padus/wetland
     0.15 each, floodway 0.10). Identical to greenfield.

  2) reuse_environmental_score is computed as a standalone column and
     promoted to its own 20% slot in the composite (see StepR5F). The
     contamination penalty is also escalated so it actually moves the
     needle:

       known_contamination_flag       -45  (was -25)
       legacy_asset_risk_flag         -20  (was -15)
       environmental_review_required  -10  (unchanged)
       decommissioning_status_known   +10  (unchanged)

     Floored at 0, ceiling at 100, starting from 100. A site with
     contamination + legacy + env-review will land at 25, which (at
     20% composite weight) pulls a 90-utility site down to ~70 -- the
     intended outcome per Owen's spec.

Run: python candidate_areas/reuse_node_scripts/StepR5E_subscore_dev_risk.py
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')

SEISMIC_TIER_SCORE = {
    'very_low': 100.0, 'low': 80.0, 'moderate': 55.0,
    'high': 30.0, 'very_high': 10.0,
}
DROUGHT_LABEL_SCORE = {
    'no_drought': 100.0, 'abnormally_dry': 85.0, 'moderate_drought': 70.0,
    'severe_drought': 50.0, 'extreme_drought': 30.0, 'exceptional_drought': 10.0,
}


def radar_score(dist_m):
    if pd.isna(dist_m):
        return 100.0
    miles = dist_m / 1609.34
    if miles > 10: return 100.0
    if miles >  5: return  85.0
    if miles >  3: return  60.0
    return            30.0


def adjacency_score(d_m, near_thresh_m, close_thresh_m, scores):
    if pd.isna(d_m):
        return scores[0]
    if d_m > 8047:           return scores[0]
    if d_m >= near_thresh_m: return scores[1]
    if d_m >= close_thresh_m: return scores[2]
    return scores[3]


def padus_score(d_m):   return adjacency_score(d_m, 500, 100, [100, 85, 60, 30])
def wetland_score(d_m): return adjacency_score(d_m, 500, 100, [100, 85, 60, 35])


def floodway_score(d_m):
    if pd.isna(d_m):  return 100.0
    if d_m > 1609.34: return 100.0
    if d_m >= 500:    return  80.0
    if d_m >= 100:    return  50.0
    return                    25.0


def reuse_environmental_score(row):
    score = 100.0
    if row.get('known_contamination_flag', False):     score -= 45.0
    if row.get('legacy_asset_risk_flag', False):       score -= 20.0
    if row.get('environmental_review_required', False): score -= 10.0
    if row.get('decommissioning_status_known', False): score += 10.0
    return max(0.0, min(100.0, score))


def main():
    print(f'Loading: {ENRICHED_PATH} ...')
    df = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(df):,} rows, {len(df.columns)} columns')

    print('\nComputing component scores ...')
    df['seismic_score']  = df['seismic_hazard_tier'].map(SEISMIC_TIER_SCORE).fillna(80.0)
    df['drought_score']  = df['drought_label'].map(DROUGHT_LABEL_SCORE).fillna(100.0)
    df['radar_score']    = df['nearest_radar_distance_m'].apply(radar_score)
    df['padus_score']    = df['nearest_padus_distance_m'].apply(padus_score)
    df['wetland_score']  = df['nearest_wetland_distance_m'].apply(wetland_score)
    df['floodway_score'] = df['nearest_floodway_distance_m'].apply(floodway_score)
    df['reuse_environmental_score'] = df.apply(reuse_environmental_score, axis=1)

    print('Composing dev_risk_score (greenfield-equivalent 6-component) ...')
    df['dev_risk_score'] = (
        0.25 * df['seismic_score']
      + 0.20 * df['drought_score']
      + 0.15 * df['radar_score']
      + 0.15 * df['padus_score']
      + 0.15 * df['wetland_score']
      + 0.10 * df['floodway_score']
    )

    df.to_parquet(ENRICHED_PATH, index=False)
    print(f'\nUpdated: {ENRICHED_PATH} ({ENRICHED_PATH.stat().st_size/1e6:.1f} MB)')

    print('\n=== Component medians ===')
    for c in ['seismic_score','drought_score','radar_score','padus_score',
              'wetland_score','floodway_score','reuse_environmental_score','dev_risk_score']:
        s = df[c]
        print(f'  {c:<28} median={s.median():>6.2f}  p10={s.quantile(0.1):>6.2f}  p90={s.quantile(0.9):>6.2f}')

    print('\n=== dev_risk_score bands ===')
    bands = pd.cut(df.dev_risk_score, [-0.01, 40, 55, 70, 85, 100.01],
                   labels=['<40','40-55','55-70','70-85','85-100'])
    print(bands.value_counts().reindex(['<40','40-55','55-70','70-85','85-100']).to_string())

    print('\n=== Checks ===')
    checks = {
        'Has rows'                  : len(df) > 0,
        'dev_risk in [0,100]'       : ((df.dev_risk_score >= 0) & (df.dev_risk_score <= 100.001)).all(),
        'dev_risk never null'       : df.dev_risk_score.notna().all(),
        'reuse_env in [0,100]'      : ((df.reuse_environmental_score >= 0) & (df.reuse_environmental_score <= 100)).all(),
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
