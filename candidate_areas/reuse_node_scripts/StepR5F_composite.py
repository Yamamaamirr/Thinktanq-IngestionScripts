"""
StepR5F -- Composite score + confidence + recommended_action + reason codes
            for reuse nodes.

Mirror of Step2F_composite_confidence_actions.py with reuse-node adaptations:

  * Composite weights identical to greenfield (40/20/15/15, renormalized over
    0.90 active weight since site_control is still Phase-2).
  * data_coverage_pct accounts for reuse-node modules. cdl_group and
    slope are not applicable -- we credit those module slots automatically
    so reuse nodes are not penalized for greenfield-only enrichments they
    structurally lack.
  * Reuse-specific reason codes added (reuse_high_contamination_risk,
    legacy_asset_decom_risk, decommissioning_timeline_known, reuse_node_with_
    grid_anchor, etc.).
  * recommended_action: 'Reuse Diligence' fires when reuse-specific risk
    flags are set and score is moderate or higher.

Reads / Writes:
  candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet

Run: python candidate_areas/reuse_node_scripts/StepR5F_composite.py
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')

# Recalibrated weights (was 40/20/15/15 = 0.90, copied from greenfield).
# First-pass cross-check on reuse nodes showed:
#   - utility dominated (corr 0.99 with composite) because reuse polygons
#     all sit near grid anchors and pinned at 100 (~78% of rows)
#   - buildability was near-constant (std 4.2) because land_cover and
#     slope are fixed defaults for reuse nodes
#   - reuse_environmental was buried inside dev_risk at 0.12 * 0.15 = 1.8%
#     of composite, so contamination penalty barely moved the needle
# New weights below restore variance and give contamination real influence.
# Sum is still 0.90 so the renormalization denominator is unchanged.
W_UTIL      = 0.25   # was 0.40
W_BUILD     = 0.15   # was 0.20
W_SUPP      = 0.15
W_RISK      = 0.15
W_REUSE_ENV = 0.20   # new -- standalone (was 0.12 * 0.15 inside dev_risk)
W_TOTAL_ACTIVE = W_UTIL + W_BUILD + W_SUPP + W_RISK + W_REUSE_ENV  # 0.90


def confidence_tier(row):
    """Worst-of-three. Same dimensions as greenfield."""
    missing = row.get('missing_modules', []) or []
    if isinstance(missing, str):
        try:
            import json
            missing = json.loads(missing)
        except Exception:
            missing = []
    n_missing = len(missing)
    cov = 'high' if n_missing == 0 else ('medium' if n_missing == 1 else 'low')

    mc = row.get('primary_anchor_match_confidence')
    if   mc == 'high':   anchor = 'high'
    elif mc == 'medium': anchor = 'medium'
    elif mc is None or pd.isna(mc): anchor = 'low'
    else: anchor = 'low'

    # 5 subscores now (reuse_environmental_score promoted to standalone).
    # Rescale tier thresholds: >=4 of 5 = high, 3 = medium, <=2 = low.
    above = sum(
        (row.get(c) or 0) >= 50
        for c in ['utility_score','buildability_score','supporting_infra_score',
                  'dev_risk_score','reuse_environmental_score']
    )
    if   above >= 4: corr = 'high'
    elif above == 3: corr = 'medium'
    else:            corr = 'low'

    order = {'high': 2, 'medium': 1, 'low': 0}
    return min([cov, anchor, corr], key=lambda x: order[x])


def recommended_action(row):
    if row.get('candidate_status') == 'manual_review':
        return 'Manual Review'
    if row.get('candidate_status') == 'excluded':
        return 'Ignore'

    score = row['composite_score']

    # Reuse-specific: high contamination / legacy asset risk -> diligence required
    if (row.get('known_contamination_flag', False) or row.get('legacy_asset_risk_flag', False)) and score >= 60:
        return 'Reuse Diligence'

    # Shortlist (same threshold as greenfield, adapted check set)
    if (score >= 90
        and 500 <= (row.get('area_acres') or 0) <= 5000
        and not row.get('oversized_flag', False)
        and row.get('confidence') in ('medium', 'high')
        and (row.get('num_anchors_in_range') or 0) >= 3):
        return 'Shortlist'
    if score >= 85 and row.get('utility_module_status') == 'zone_fallback':
        return 'Utility Desk Check'
    if score >= 75:
        return 'Parcel Pull'   # always pulled for reuse since EPA records lack APN
    if score >= 65:
        return 'Monitor'
    return 'Ignore'


def actionability_status(row):
    if row.get('candidate_status') == 'manual_review':
        return 'internal_diligence_only'
    if row.get('candidate_status') == 'excluded':
        return 'do_not_pitch'
    return 'apn_owner_pull_required'


REASON_RULES = [
    # POSITIVE
    ('strong_queue_signal',           '+', lambda r: r.get('primary_anchor_queue_mw_tier') in ('strong','very_strong'), 3),
    ('interconnection_agreement_executed', '+', lambda r: r.get('ia_executed_nearby') == True, 3),
    ('500kv_anchor_in_range',         '+', lambda r: (r.get('nearest_500kv_distance_m') or 1e9) <= 5*1609.34, 3),
    ('345kv_anchor_in_range',         '+', lambda r: (r.get('nearest_345kv_distance_m') or 1e9) <= 5*1609.34, 2),
    ('tier1_pipeline_within_5mi',     '+', lambda r: (r.get('nearest_tier1_pipeline_distance_m') or 1e9) <= 5*1609.34, 2),
    ('class1_rail_within_3mi',        '+', lambda r: (r.get('nearest_class1_rail_distance_m') or 1e9) <= 3*1609.34, 2),
    ('low_seismic_risk',              '+', lambda r: r.get('seismic_hazard_tier') in ('very_low','low'), 2),
    ('within_water_service_area',     '+', lambda r: r.get('within_water_service_area') == True, 2),
    ('strategic_scale_candidate',     '+', lambda r: r.get('acreage_tier') == 'strategic_scale', 2),
    # Reuse-specific positives
    ('high_confidence_real_footprint','+', lambda r: r.get('geometry_source') == 'OSM_POLYGON', 2),
    ('decommissioning_timeline_known','+', lambda r: r.get('decommissioning_status_known') == True, 2),
    ('reuse_node_with_grid_anchor',   '+', lambda r: r.get('source') in ('EIA-860','EIA-860-nuclear') and (r.get('num_anchors_in_range') or 0) >= 3, 2),

    # NEGATIVE
    ('no_queue_activity',             '-', lambda r: r.get('primary_anchor_queue_mw_tier') in ('negligible',) or r.get('num_anchors_in_range') == 0, 3),
    ('no_executed_ia_in_range',       '-', lambda r: r.get('ia_executed_nearby') == False and (r.get('num_anchors_in_range') or 0) > 0, 2),
    ('high_seismic_zone',             '-', lambda r: r.get('seismic_hazard_tier') in ('high','very_high'), 2),
    ('floodway_adjacent_500m',        '-', lambda r: r.get('adjacent_floodway_flag') == True, 2),
    ('fema_ae_overlap',               '-', lambda r: r.get('fema_ae_overlap_flag') == True, 2),
    ('drought_tier_high',             '-', lambda r: r.get('drought_label') in ('severe_drought','extreme_drought','exceptional_drought'), 1),
    ('padus_adjacent_500m',           '-', lambda r: r.get('near_padus_flag') == True, 1),
    ('wetland_adjacent_500m',         '-', lambda r: r.get('near_wetland_flag') == True, 1),
    ('radar_review_flag',             '-', lambda r: r.get('radar_review_flag') == True, 1),
    # Reuse-specific negatives
    ('reuse_high_contamination_risk', '-', lambda r: r.get('known_contamination_flag') == True, 3),
    ('legacy_asset_decom_risk',       '-', lambda r: r.get('legacy_asset_risk_flag') == True, 2),
    ('environmental_review_required', '-', lambda r: r.get('environmental_review_required') == True, 1),

    # UNCERTAIN
    ('queue_anchor_zone_fallback',    '?', lambda r: r.get('utility_module_status') == 'zone_fallback', 1),
    ('oversized_polygon',             '?', lambda r: r.get('size_class') == 'region', 1),
    ('allocation_risk_possible',      '?', lambda r: r.get('allocation_or_competitive_heat_flag') == True, 1),
    ('estimated_footprint_buffered',  '?', lambda r: r.get('geometry_source') in ('BUFFER_FROM_ACREAGE','BUFFER_FROM_CAPACITY'), 1),
    ('multi_parcel_aggregated',       '?', lambda r: (r.get('aliased_site_count') or 1) >= 5, 1),
]


def top_reason_codes(row):
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
    codes.extend([c[0] for c in positives[:3]])
    codes.extend([c[0] for c in negatives[:2]])
    if uncertain and len(codes) >= 5:
        codes = codes[:4] + [uncertain[0][0]]
    elif uncertain:
        codes.append(uncertain[0][0])

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
    print(f'Loading: {ENRICHED_PATH} ...')
    df = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(df):,} rows, {len(df.columns)} columns')

    print('\nComputing composite_score (5-component recalibrated) ...')
    df['composite_score'] = (
        W_UTIL      * df['utility_score']
      + W_BUILD     * df['buildability_score']
      + W_SUPP      * df['supporting_infra_score']
      + W_RISK      * df['dev_risk_score']
      + W_REUSE_ENV * df['reuse_environmental_score']
    ) / W_TOTAL_ACTIVE

    df['score_v1_observable'] = True

    print('Computing per-row data_coverage_pct ...')
    # 9 modules at 10 pp each; reuse nodes get automatic credit for
    # land_cover and slope (greenfield-only enrichments they structurally
    # don't carry), and lose site_control like greenfield.
    modules_present = {
        'utility':      df['primary_anchor_name'].notna() & (df['utility_module_status'] != 'failed'),
        'land_cover':   pd.Series([True] * len(df), index=df.index),  # auto-credit
        'slope':        pd.Series([True] * len(df), index=df.index),  # auto-credit (deferred)
        'seismic':      df['seismic_hazard_pga'].notna(),
        'drought':      df['drought_label'].notna(),
        'transmission': (df['nearest_500kv_distance_m'].notna() | df['nearest_345kv_distance_m'].notna() | df['nearest_230kv_distance_m'].notna()),
        'pipeline':     df['nearest_pipeline_distance_m'].notna(),
        'rail':         df['nearest_class1_rail_distance_m'].notna(),
        'water':        df['nearest_water_service_distance_m'].notna(),
    }
    cov_pct = sum(m.astype(int) * 10 for m in modules_present.values())
    df['data_coverage_pct'] = cov_pct.astype(float)

    print('Computing per-row missing_modules ...')
    def _missing_for_row(idx):
        miss = ['site_control']
        for mod_name, mod_mask in modules_present.items():
            if not mod_mask.iloc[idx]:
                miss.append(mod_name)
        return miss
    df['missing_modules'] = [_missing_for_row(i) for i in range(len(df))]

    print('Computing confidence ...')
    df['confidence'] = df.apply(confidence_tier, axis=1)

    print('Computing recommended_action ...')
    df['recommended_action'] = df.apply(recommended_action, axis=1)

    print('Computing actionability_status ...')
    df['actionability_status'] = df.apply(actionability_status, axis=1)

    print('Computing top_reason_codes ...')
    df['top_reason_codes'] = df.apply(top_reason_codes, axis=1)

    df.to_parquet(ENRICHED_PATH, index=False)
    print(f'\nUpdated: {ENRICHED_PATH} ({ENRICHED_PATH.stat().st_size/1e6:.1f} MB)')

    s = df.composite_score
    print(f'\n=== composite_score: min={s.min():.2f}, p10={s.quantile(0.1):.2f}, '
          f'median={s.median():.2f}, p90={s.quantile(0.9):.2f}, max={s.max():.2f}')

    print('\n=== composite_score bands ===')
    bands = pd.cut(s, [-0.01, 40, 55, 70, 85, 100.01],
                   labels=['<40','40-55','55-70','70-85','85-100'])
    print(bands.value_counts().reindex(['<40','40-55','55-70','70-85','85-100']).to_string())

    print('\n=== Per-state median composite ===')
    for st in ['AZ','CA','NV','TX','VA']:
        sd = df[df.state == st]
        print(f'  {st}: n={len(sd):>5,}  median={sd.composite_score.median():>6.2f}  p90={sd.composite_score.quantile(0.9):>6.2f}')

    print('\n=== Per-source median composite ===')
    for src in df.source.unique():
        sd = df[df.source == src]
        print(f'  {src:<22} n={len(sd):>5,}  median={sd.composite_score.median():>6.2f}  p90={sd.composite_score.quantile(0.9):>6.2f}')

    print('\n=== confidence ===')
    print(df.confidence.value_counts().to_string())

    print('\n=== recommended_action ===')
    print(df.recommended_action.value_counts().to_string())

    print('\n=== Top reason code frequency (top 20) ===')
    code_counts = pd.Series(
        [c for codes in df.top_reason_codes for c in codes]
    ).value_counts().head(20)
    print(code_counts.to_string())

    print('\n=== Top 10 reuse nodes by composite_score ===')
    top = df.sort_values('composite_score', ascending=False).head(10)
    for _, r in top.iterrows():
        nm = (str(r.site_name)[:24] if pd.notna(r.site_name) else '(unnamed)')
        print(f'  {r.candidate_id[:22]:<22} {r.state} {nm:<24} {r.area_acres:>5.0f}ac '
              f'comp={r.composite_score:>5.1f} util={r.utility_score:>4.0f} '
              f'build={r.buildability_score:>4.0f} supp={r.supporting_infra_score:>4.0f} '
              f'risk={r.dev_risk_score:>4.0f} conf={r.confidence} | {r.recommended_action}')

    print('\n=== Checks ===')
    checks = {
        'Has rows'                       : len(df) > 0,
        'composite in [0,100]'           : ((df.composite_score >= 0) & (df.composite_score <= 100.001)).all(),
        'composite never null'           : df.composite_score.notna().all(),
        'confidence valid set'           : set(df.confidence.unique()) <= {'high','medium','low'},
        'recommended_action valid set'   : set(df.recommended_action.unique()) <= {
            'Ignore','Monitor','Manual Review','Parcel Pull','Utility Desk Check',
            'Ownership Review','Reuse Diligence','Shortlist'
        },
        'top_reason_codes never empty'   : df.top_reason_codes.apply(len).min() > 0,
        'manual_review -> Manual Review' : (df[df.candidate_status == 'manual_review'].recommended_action == 'Manual Review').all() if (df.candidate_status == 'manual_review').any() else True,
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
