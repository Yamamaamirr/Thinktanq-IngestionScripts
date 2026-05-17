"""
Step 2E — Subscore 4: Development Risk (15% of composite).

Implements scoring_design.md Section 5.

  dev_risk_score = 0.25 * seismic_score
                 + 0.20 * drought_score
                 + 0.15 * radar_score
                 + 0.15 * padus_score
                 + 0.15 * wetland_score
                 + 0.10 * floodway_score

All components return 0-100 where 100 = no risk.

Reads / Writes:
  candidate_areas/outputs/candidate_areas_enriched.parquet

Adds:
  seismic_score, drought_score, radar_score, padus_score, wetland_score,
  floodway_score, dev_risk_score

Run:
  python candidate_areas/scoring_scripts/Step2E_subscore4_dev_risk.py
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')

SEISMIC_TIER_SCORE = {
    'very_low':  100.0,
    'low':        80.0,
    'moderate':   55.0,
    'high':       30.0,
    'very_high':  10.0,
}

DROUGHT_LABEL_SCORE = {
    'no_drought':          100.0,
    'abnormally_dry':       85.0,
    'moderate_drought':     70.0,
    'severe_drought':       50.0,
    'extreme_drought':      30.0,
    'exceptional_drought':  10.0,
}


def radar_score(dist_m):
    """Per scoring_design 5.4. radar_distance_miles already in df."""
    if pd.isna(dist_m):
        return 100.0
    miles = dist_m / 1609.34
    if miles > 10: return 100.0
    if miles >  5: return  85.0
    if miles >  3: return  60.0
    return            30.0


def adjacency_score(d_m, near_thresh_m, close_thresh_m, scores):
    """
    Generic adjacency scorer:
      d > 5 mi (8047 m) → scores[0]
      500 m–5 mi        → scores[1]
      100–500 m         → scores[2]
      < 100 m           → scores[3]
    Caller passes near_thresh_m (500 typically) and close_thresh_m (100).
    """
    if pd.isna(d_m):
        return scores[0]  # treat NaN as "very far" — beyond max_search anyway
    if d_m > 8047:        return scores[0]
    if d_m >= near_thresh_m: return scores[1]
    if d_m >= close_thresh_m: return scores[2]
    return scores[3]


def padus_score(d_m):    return adjacency_score(d_m, 500, 100, [100, 85, 60, 30])
def wetland_score(d_m):  return adjacency_score(d_m, 500, 100, [100, 85, 60, 35])


def floodway_score(d_m):
    """Per scoring_design 5.7. Different thresholds (1 mi top, 500 m, 100 m)."""
    if pd.isna(d_m):  return 100.0
    if d_m > 1609.34: return 100.0
    if d_m >= 500:    return  80.0
    if d_m >= 100:    return  50.0
    return                    25.0


def main():
    print(f'Loading enriched candidates: {ENRICHED_PATH} ...')
    df = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(df):,} candidates, {len(df.columns)} columns')

    # ---- Components ----
    print('\nComputing seismic_score ...')
    df['seismic_score']  = df['seismic_hazard_tier'].map(SEISMIC_TIER_SCORE).fillna(80.0)

    print('Computing drought_score ...')
    df['drought_score']  = df['drought_label'].map(DROUGHT_LABEL_SCORE).fillna(100.0)

    print('Computing radar_score ...')
    df['radar_score']    = df['nearest_radar_distance_m'].apply(radar_score)

    print('Computing padus_score ...')
    df['padus_score']    = df['nearest_padus_distance_m'].apply(padus_score)

    print('Computing wetland_score ...')
    df['wetland_score']  = df['nearest_wetland_distance_m'].apply(wetland_score)

    print('Computing floodway_score ...')
    df['floodway_score'] = df['nearest_floodway_distance_m'].apply(floodway_score)

    # ---- Composite ----
    print('Composing dev_risk_score ...')
    df['dev_risk_score'] = (
        0.25 * df['seismic_score']
      + 0.20 * df['drought_score']
      + 0.15 * df['radar_score']
      + 0.15 * df['padus_score']
      + 0.15 * df['wetland_score']
      + 0.10 * df['floodway_score']
    )

    df.to_parquet(ENRICHED_PATH, index=False)
    size_mb = ENRICHED_PATH.stat().st_size / 1e6
    print(f'\nUpdated: {ENRICHED_PATH} ({size_mb:.1f} MB)')

    # ---- Summary ----
    print('\n=== Subscore 4 component medians ===')
    for c in ['seismic_score','drought_score','radar_score','padus_score','wetland_score','floodway_score','dev_risk_score']:
        s = df[c]
        print(f'  {c:<20} median={s.median():>6.2f}  p10={s.quantile(0.1):>6.2f}  p90={s.quantile(0.9):>6.2f}')

    print('\n=== dev_risk_score bands ===')
    bands = pd.cut(df.dev_risk_score, [-0.01, 40, 55, 70, 85, 100.01],
                   labels=['<40','40-55','55-70','70-85','85-100'])
    print(bands.value_counts().reindex(['<40','40-55','55-70','70-85','85-100']).to_string())

    print('\n=== Per-state median dev_risk_score ===')
    for st in ['AZ','CA','NV','TX','VA']:
        sd = df[df.state == st]
        print(f'  {st}: n={len(sd):>6,}  median={sd.dev_risk_score.median():>6.2f}  p90={sd.dev_risk_score.quantile(0.9):>6.2f}')

    print('\n=== Checks ===')
    checks = {
        'Has rows'                   : len(df) > 0,
        'dev_risk in [0, 100]'       : ((df.dev_risk_score >= 0) & (df.dev_risk_score <= 100.001)).all(),
        'dev_risk never null'        : df.dev_risk_score.notna().all(),
        'seismic_score never null'   : df.seismic_score.notna().all(),
        'drought_score never null'   : df.drought_score.notna().all(),
        'wetland_score never null'   : df.wetland_score.notna().all(),
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
