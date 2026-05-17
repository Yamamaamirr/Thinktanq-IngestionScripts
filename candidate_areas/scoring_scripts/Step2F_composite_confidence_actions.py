"""
Step 2F — Composite score + Confidence + Two action fields + Reason codes.

Implements scoring_design.md Sections 1.2, 1.3, 1.5, 1.4.

Composite per 1.2:
  composite_score = (0.40*util + 0.20*build + 0.15*supp + 0.15*risk) / 0.90
  → renormalized over the 90% observable weight (site_control 10% is null in Phase 1)

Confidence per 1.3 (worst-of-three):
  1) Data coverage tier: based on len(missing_modules)
  2) Anchor confidence tier: from primary_anchor_match_confidence
  3) Signal corroboration tier: count of subscores >= 50

recommended_action per Owen Rec #29 (1.5 in scoring_design):
  Ignore / Monitor / Manual Review / Parcel Pull / Utility Desk Check /
  Ownership Review / Reuse Diligence / Shortlist

actionability_status per Owen May 11 (separate field):
  do_not_pitch / internal_diligence_only / apn_owner_pull_required /
  broker_verify_required / nda_teaser_possible / buyer_ready_with_caveats

top_reason_codes (1.4): 3-5 codes per row, controlled vocabulary, ranked by
absolute contribution + padded with up to 2 uncertain codes.

Reads / Writes:
  candidate_areas/outputs/candidate_areas_enriched.parquet

Run:
  python candidate_areas/scoring_scripts/Step2F_composite_confidence_actions.py
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')

# -------- Composite weights (active subscores; site_control deferred) --------
W_UTIL = 0.40
W_BUILD = 0.20
W_SUPP = 0.15
W_RISK = 0.15
W_TOTAL_ACTIVE = W_UTIL + W_BUILD + W_SUPP + W_RISK   # 0.90


def confidence_tier(row):
    """Worst-of-three per scoring_design 1.3."""
    # Dimension 1: data coverage
    missing = row.get('missing_modules', []) or []
    if isinstance(missing, str):
        try:
            import json
            missing = json.loads(missing)
        except Exception:
            missing = []
    n_missing = len(missing)
    cov = 'high' if n_missing == 0 else ('medium' if n_missing == 1 else 'low')

    # Dimension 2: anchor confidence
    mc = row.get('primary_anchor_match_confidence')
    if   mc == 'high':   anchor = 'high'
    elif mc == 'medium': anchor = 'medium'
    elif mc is None or pd.isna(mc): anchor = 'low'  # zone fallback / no anchors
    else: anchor = 'low'

    # Dimension 3: signal corroboration — count subscores >= 50
    above = sum(
        (row.get(c) or 0) >= 50
        for c in ['utility_score','buildability_score','supporting_infra_score','dev_risk_score']
    )
    if   above >= 3: corr = 'high'
    elif above == 2: corr = 'medium'
    else:            corr = 'low'

    # Take worst
    order = {'high': 2, 'medium': 1, 'low': 0}
    levels = [cov, anchor, corr]
    worst = min(levels, key=lambda x: order[x])
    return worst


def recommended_action(row):
    """Per Owen Rec #29 + May 11 Shortlist activation criteria."""
    if row.get('candidate_status') == 'manual_review':
        return 'Manual Review'
    if row.get('candidate_status') == 'excluded':
        return 'Ignore'

    score = row['composite_score']
    # Shortlist: top tier — strong all-around candidates ready for diligence prioritization
    if (score >= 90
        and 500 <= (row.get('area_acres') or 0) <= 5000
        and not row.get('slope_review_flag', False)
        and not row.get('oversized_flag', False)
        and row.get('confidence') in ('medium', 'high')
        and (row.get('num_anchors_in_range') or 0) >= 3):
        return 'Shortlist'
    if score >= 85 and row.get('utility_module_status') == 'zone_fallback':
        return 'Utility Desk Check'
    if score >= 75 and pd.isna(row.get('parcel_count')):
        return 'Parcel Pull'
    if score >= 65:
        return 'Monitor'
    return 'Ignore'


def actionability_status(row):
    """Per Owen May 11. Most Phase 1 rows land at apn_owner_pull_required."""
    if row.get('candidate_status') == 'manual_review':
        return 'internal_diligence_only'
    if row.get('candidate_status') == 'excluded':
        return 'do_not_pitch'
    # Phase 1: we lack parcel data, so default to apn_owner_pull_required
    return 'apn_owner_pull_required'


REASON_RULES = [
    # (code, sign, predicate, contribution_weight)
    # POSITIVE — weights bumped so row-specific features (slope/pipeline-close/etc.)
    # have a real chance of surfacing alongside generic queue/IA codes
    ('strong_queue_signal',         '+', lambda r: r.get('primary_anchor_queue_mw_tier') in ('strong','very_strong'), 3),
    ('interconnection_agreement_executed', '+', lambda r: r.get('ia_executed_nearby') == True, 3),
    ('500kv_anchor_in_range',       '+', lambda r: (r.get('nearest_500kv_distance_m') or 1e9) <= 5*1609.34, 3),
    ('345kv_anchor_in_range',       '+', lambda r: (r.get('nearest_345kv_distance_m') or 1e9) <= 5*1609.34, 2),
    ('tier1_pipeline_within_5mi',   '+', lambda r: (r.get('nearest_tier1_pipeline_distance_m') or 1e9) <= 5*1609.34, 2),
    ('class1_rail_within_3mi',      '+', lambda r: (r.get('nearest_class1_rail_distance_m') or 1e9) <= 3*1609.34, 2),
    ('ideal_slope',                 '+', lambda r: r.get('slope_tier') == 'ideal', 2),
    ('low_seismic_risk',            '+', lambda r: r.get('seismic_hazard_tier') in ('very_low','low'), 2),
    ('within_water_service_area',   '+', lambda r: r.get('within_water_service_area') == True, 2),
    ('large_fallow_candidate',      '+', lambda r: r.get('cdl_group')=='cropland' and (r.get('area_acres') or 0) >= 500, 2),
    ('strategic_scale_candidate',   '+', lambda r: r.get('acreage_tier') == 'strategic_scale', 2),
    ('building_density_low',        '+', lambda r: (r.get('building_footprint_pct') or 0) < 0.25, 1),

    # NEGATIVE
    ('building_density_high',       '-', lambda r: (r.get('building_footprint_pct') or 0) >= 1.0, 3),
    ('no_queue_activity',           '-', lambda r: r.get('primary_anchor_queue_mw_tier') in ('negligible',) or r.get('num_anchors_in_range')==0, 3),
    ('no_executed_ia_in_range',     '-', lambda r: r.get('ia_executed_nearby') == False and r.get('num_anchors_in_range',0) > 0, 2),
    ('high_seismic_zone',           '-', lambda r: r.get('seismic_hazard_tier') in ('high','very_high'), 2),
    ('floodway_adjacent_500m',      '-', lambda r: r.get('adjacent_floodway_flag') == True, 2),
    ('fema_ae_overlap',             '-', lambda r: r.get('fema_ae_overlap_flag') == True, 2),
    ('elevated_slope_band',         '-', lambda r: r.get('slope_tier') == 'penalized', 2),
    ('drought_tier_high',           '-', lambda r: r.get('drought_label') in ('severe_drought','extreme_drought','exceptional_drought'), 1),
    ('padus_adjacent_500m',         '-', lambda r: r.get('near_padus_flag') == True, 1),
    ('wetland_adjacent_500m',       '-', lambda r: r.get('near_wetland_flag') == True, 1),
    ('forest_land',                 '-', lambda r: r.get('cdl_group') == 'forest', 1),
    ('small_candidate',             '-', lambda r: r.get('acreage_tier') == 'small', 1),
    ('radar_review_flag',           '-', lambda r: r.get('radar_review_flag') == True, 1),

    # UNCERTAIN — only row-specific (removed always-fire clutter pipeline_diameter_estimated + owner_data_missing)
    ('queue_anchor_zone_fallback',  '?', lambda r: r.get('utility_module_status') == 'zone_fallback', 1),
    ('oversized_polygon',           '?', lambda r: r.get('size_class') == 'region', 1),
    ('allocation_risk_possible',    '?', lambda r: r.get('allocation_or_competitive_heat_flag') == True, 1),
]


def top_reason_codes(row):
    """Per Owen May 11: 3-5 codes per row mixing positives, negatives, uncertain.

    Allocation: 3 positives + 2 negatives + 0-1 uncertain, capped at 5.
    Uncertain slot only fills when row-specific uncertain codes fire
    (zone_fallback / oversized / allocation_risk).
    """
    fired = []
    for code, sign, pred, weight in REASON_RULES:
        try:
            if pred(row):
                fired.append((code, sign, weight))
        except Exception:
            continue
    positives = sorted([f for f in fired if f[1] == '+'], key=lambda x: -x[2])
    negatives = sorted([f for f in fired if f[1] == '-'], key=lambda x: -x[2])
    uncertain = sorted([f for f in fired if f[1] == '?'], key=lambda x: -x[2])

    codes = []
    # 3 positives surfaces row-specific strengths (queue/IA/345kV/pipeline/slope/etc.)
    codes.extend([c[0] for c in positives[:3]])
    # 2 negatives surfaces the row's risks
    codes.extend([c[0] for c in negatives[:2]])
    # Drop earliest positive to make room for uncertain if any fired (keeps cap at 5)
    if uncertain and len(codes) >= 5:
        codes = codes[:4] + [uncertain[0][0]]
    elif uncertain:
        codes.append(uncertain[0][0])

    # If under 3 (rare candidate with very few signals), fill from remaining
    if len(codes) < 3:
        seen = set(codes)
        for c in positives[3:] + negatives[2:] + uncertain[1:]:
            if c[0] not in seen:
                codes.append(c[0])
                seen.add(c[0])
                if len(codes) >= 3:
                    break
    return codes[:5]


def main():
    print(f'Loading enriched candidates: {ENRICHED_PATH} ...')
    df = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(df):,} candidates, {len(df.columns)} columns')

    # ---- Composite score (renormalize over the 90% active weight) ----
    print('\nComputing composite_score ...')
    df['composite_score'] = (
        W_UTIL  * df['utility_score']
      + W_BUILD * df['buildability_score']
      + W_SUPP  * df['supporting_infra_score']
      + W_RISK  * df['dev_risk_score']
    ) / W_TOTAL_ACTIVE

    df['score_v1_observable'] = True

    # ---- Per-row data_coverage_pct & missing_modules (Task #9, Owen May 11) ----
    # Site_control (10%) is always missing in Phase 1. The other 90% is allocated
    # equally across 9 active modules, each worth 10 points. We deduct 10 points
    # per module that had no data for this candidate.
    print('Computing per-row data_coverage_pct ...')
    modules_present = {
        'utility':     df['primary_anchor_name'].notna() & (df['utility_module_status'] != 'failed'),
        'land_cover':  df['cdl_group'].notna(),
        'slope':       df['slope_mean_pct'].notna(),
        'seismic':     df['seismic_hazard_pga'].notna(),
        'drought':     df['drought_level'].notna(),
        'transmission': (df['nearest_500kv_distance_m'].notna() | df['nearest_345kv_distance_m'].notna() | df['nearest_230kv_distance_m'].notna()),
        'pipeline':    df['nearest_pipeline_distance_m'].notna(),
        'rail':        df['nearest_class1_rail_distance_m'].notna(),
        'water':       df['nearest_water_service_distance_m'].notna(),
    }
    # Each module = 10 percentage points; total max = 90 (site_control 10 always missing)
    cov_pct = sum(m.astype(int) * 10 for m in modules_present.values())
    df['data_coverage_pct'] = cov_pct.astype(float)

    # missing_modules: list per row
    print('Computing per-row missing_modules ...')
    def _missing_for_row(idx):
        miss = ['site_control']  # always
        for mod_name, mod_mask in modules_present.items():
            if not mod_mask.iloc[idx]:
                miss.append(mod_name)
        return miss
    df['missing_modules'] = [_missing_for_row(i) for i in range(len(df))]

    # ---- Confidence ----
    print('Computing confidence ...')
    df['confidence'] = df.apply(confidence_tier, axis=1)

    # ---- recommended_action ----
    print('Computing recommended_action ...')
    df['recommended_action'] = df.apply(recommended_action, axis=1)

    # ---- actionability_status ----
    print('Computing actionability_status ...')
    df['actionability_status'] = df.apply(actionability_status, axis=1)

    # ---- top_reason_codes ----
    print('Computing top_reason_codes ...')
    df['top_reason_codes'] = df.apply(top_reason_codes, axis=1)

    # ---- Save ----
    df.to_parquet(ENRICHED_PATH, index=False)
    size_mb = ENRICHED_PATH.stat().st_size / 1e6
    print(f'\nUpdated: {ENRICHED_PATH} ({size_mb:.1f} MB)')

    # ---- Summary ----
    print('\n=== composite_score distribution ===')
    s = df.composite_score
    print(f'  min={s.min():.2f}, p10={s.quantile(0.1):.2f}, median={s.median():.2f}, '
          f'p90={s.quantile(0.9):.2f}, max={s.max():.2f}')

    bands = pd.cut(s, [-0.01, 40, 55, 70, 85, 100.01],
                   labels=['<40','40-55','55-70','70-85','85-100'])
    print('\n=== composite_score bands ===')
    print(bands.value_counts().reindex(['<40','40-55','55-70','70-85','85-100']).to_string())

    print('\n=== Per-state median composite ===')
    for st in ['AZ','CA','NV','TX','VA']:
        sd = df[df.state == st]
        print(f'  {st}: n={len(sd):>6,}  median={sd.composite_score.median():>6.2f}  '
              f'p90={sd.composite_score.quantile(0.9):>6.2f}')

    print('\n=== confidence ===')
    print(df.confidence.value_counts().to_string())

    print('\n=== recommended_action ===')
    print(df.recommended_action.value_counts().to_string())

    print('\n=== actionability_status ===')
    print(df.actionability_status.value_counts().to_string())

    print('\n=== Top reason code frequency (top 15) ===')
    code_counts = pd.Series(
        [c for codes in df.top_reason_codes for c in codes]
    ).value_counts().head(15)
    print(code_counts.to_string())

    print('\n=== Top 10 by composite_score (sanity check) ===')
    top = df.sort_values('composite_score', ascending=False).head(10)
    for _, r in top.iterrows():
        print(f'  {r.candidate_id[:8]}.. {r.state} {r.county_name:<15} {r.area_acres:>6.0f}ac  '
              f'composite={r.composite_score:.1f} | util={r.utility_score:.0f} build={r.buildability_score:.0f} '
              f'supp={r.supporting_infra_score:.0f} risk={r.dev_risk_score:.0f}  '
              f'conf={r.confidence}  action={r.recommended_action}')

    # ---- Checks ----
    print('\n=== Checks ===')
    checks = {
        'Has rows'                          : len(df) > 0,
        'composite in [0, 100]'             : ((df.composite_score >= 0) & (df.composite_score <= 100.001)).all(),
        'composite never null'              : df.composite_score.notna().all(),
        'confidence valid set'              : set(df.confidence.unique()) <= {'high','medium','low'},
        'recommended_action valid set'      : set(df.recommended_action.unique()) <= {
            'Ignore','Monitor','Manual Review','Parcel Pull','Utility Desk Check',
            'Ownership Review','Reuse Diligence','Shortlist'
        },
        'actionability_status valid set'    : set(df.actionability_status.unique()) <= {
            'do_not_pitch','internal_diligence_only','apn_owner_pull_required',
            'broker_verify_required','nda_teaser_possible','buyer_ready_with_caveats'
        },
        'top_reason_codes never empty'      : df.top_reason_codes.apply(len).min() > 0,
        'all reason code lists are list'    : df.top_reason_codes.apply(lambda x: isinstance(x, list)).all(),
        'manual_review → Manual Review'     : (df[df.candidate_status=='manual_review'].recommended_action == 'Manual Review').all() if (df.candidate_status=='manual_review').any() else True,
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
