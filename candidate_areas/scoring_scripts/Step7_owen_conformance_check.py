"""
Step 7 - Owen Conformance & Geographic Sanity Check.

Verifies the final deliverable against every field Owen asked for across
his entire conversation (May 1-14) plus the PDF (May 6, 30 rules), AND
runs geographic sanity checks on every row.

Owen's asks (consolidated):
  Funnel (May 1):
    1. Keep-zones around infrastructure
    2. Vacant/large land detection
    3. Hard exclusions: wetlands, protected, flood, dense buildings, water, radar/airport
    4. CDL + MS Building Footprints
    5. Ranked candidate polygons

  PDF (May 6) - 30 rules:
    - Tiered queue scoring (not 200 MW threshold)
    - Queue status quality
    - 25km substation buffer = search zone
    - 230 kV as secondary signal
    - Hard exclusions vs review flags split
    - Radar as review flag
    - 8% slope review + 15% hard exclusion
    - Acreage tiers
    - Candidate type
    - Kill gates
    - Utility Infrastructure Signal rename
    - Action labels (8 values)
    - Dataset versioning

  Fields he cares about (May 11):
    state, county, acreage, buildable_acres, centroid, candidate_type,
    hard_exclusion flags, building_footprint_pct, slope/flood/wetland/protected_area flags,
    nearest 345kV/500kV, ISO queue MW/status, fiber/gas/water/road proximity,
    parcel/owner/assessed-value placeholders, composite_score, recommended_action,
    top_reason_codes

  May 13 tweaks:
    Hot-node vs actionable-site split:
      node_activity_score, queue_mw_nearby, ia_executed_nearby, activation_band,
      recent_withdrawals_nearby, allocation_or_competitive_heat_flag
    Distance decay/bands + anchor transparency:
      primary_anchor_name/distance/voltage/queue_mw_tier/queue_status_score/
      activation_band/match_confidence/distance_band/zone_fallback_used
    Buildable acreage:
      original_area_acres, net_buildable_area_acres, buildable_area_ratio
    Score coverage:
      score_v1_observable, data_coverage_pct, missing_modules, parcel_owner_module_status
    Actionability:
      actionability_status (6 values)
    Reason codes with negatives + uncertainties

  Geographic sanity:
    - centroid_lon/lat in state bounding box
    - area_acres matches geometry.area
    - state matches centroid
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

PARQUET = Path('candidate_areas/outputs/candidates_final.parquet')

# State bounding boxes (lon_min, lat_min, lon_max, lat_max) - generous coverage
STATE_BBOX = {
    'TX': (-106.7, 25.5, -93.4, 36.7),
    'VA': (-83.7, 36.5, -75.1, 39.5),
    'CA': (-124.7, 32.3, -114.0, 42.1),
    'AZ': (-114.9, 31.2, -109.0, 37.0),
    'NV': (-120.1, 34.9, -114.0, 42.1),
}

ACRES_PER_M2 = 0.000247105


def main():
    print(f'Loading {PARQUET}\n')
    g = gpd.read_parquet(PARQUET)
    n = len(g)
    print(f'Rows: {n:,}')
    print(f'Columns: {len(g.columns)}\n')

    # ==================================================================
    # PART A: Owen's required fields - presence + non-null where expected
    # ==================================================================
    print('=' * 78)
    print('PART A: Owen-required fields presence check')
    print('=' * 78)

    required_fields = {
        # Funnel basics (May 1)
        'state':            'non_null',
        'county_name':      'non_null',
        'area_acres':       'non_null',
        'net_buildable_area_acres': 'non_null',
        'centroid_lon':     'non_null',
        'centroid_lat':     'non_null',
        'candidate_type':   'non_null',
        'cdl_group':        'non_null',
        'building_footprint_pct': 'non_null',

        # Slope/flood/wetland/protected (adjacency form per Yamama May 13 to Owen)
        'slope_mean_pct':   'non_null',
        'slope_max_pct':    'non_null',
        'slope_review_flag':'non_null',
        'near_padus_flag':  'non_null',
        'near_wetland_flag':'non_null',
        'adjacent_floodway_flag': 'non_null',
        'fema_ae_overlap_flag': 'non_null',
        'fema_ae_adjacent_flag':'non_null',
        'radar_review_flag':'non_null',

        # Nearest 345kV / 500kV / 230kV transmission (May 11)
        'nearest_500kv_distance_m': 'numeric_or_null',
        'nearest_345kv_distance_m': 'numeric_or_null',
        'nearest_230kv_distance_m': 'numeric_or_null',

        # ISO queue / utility anchor (May 11, 13)
        'primary_anchor_name':     'mostly_non_null',
        'primary_anchor_distance_m':'mostly_non_null',
        'primary_anchor_voltage_kv':'mostly_non_null',
        'primary_anchor_queue_mw_tier':'mostly_non_null',
        'primary_anchor_queue_status_score':'mostly_non_null',
        'primary_anchor_activation_band':'mostly_non_null',
        'primary_anchor_match_confidence':'mostly_non_null',
        'primary_anchor_distance_band':'mostly_non_null',
        'num_anchors_in_range':    'non_null',
        'zone_fallback_used':      'non_null',
        'node_activity_score':     'non_null',
        'queue_mw_nearby':         'non_null',
        'ia_executed_nearby':      'non_null',
        'recent_withdrawals_nearby':'non_null',
        'allocation_or_competitive_heat_flag': 'non_null',

        # Supporting infra (May 11)
        'nearest_pipeline_distance_m':       'numeric_or_null',
        'nearest_pipeline_operator_tier':    'numeric_or_null',
        'nearest_pipeline_est_diameter_in':  'numeric_or_null',
        'pipeline_diameter_estimated':       'non_null',
        'nearest_class1_rail_distance_m':    'numeric_or_null',
        'nearest_rail_is_stracnet':          'numeric_or_null',
        'within_water_service_area':         'non_null',
        'nearest_water_service_distance_m':  'non_null',

        # Buildable acreage tracking (May 13)
        'original_area_acres':       'non_null',
        'buildable_area_ratio':      'non_null',

        # Scoring (May 11/13)
        'utility_score':         'non_null',
        'buildability_score':    'non_null',
        'supporting_infra_score':'non_null',
        'dev_risk_score':        'non_null',
        'composite_score':       'non_null',
        'score_v1_observable':   'non_null',
        'data_coverage_pct':     'non_null',
        'missing_modules':       'non_null',
        'confidence':            'non_null',
        'recommended_action':    'non_null',
        'actionability_status':  'non_null',
        'top_reason_codes':      'non_null',

        # Phase 2 placeholders (expected null) - PDF Rules 21/22/23/24/25/26/28
        'parcel_count':          'expected_null',
        'owner_count':           'expected_null',
        'assessed_value_per_acre':'expected_null',
        'land_use_code':         'expected_null',
        'zoning_code':           'expected_null',
        'road_frontage_flag':    'expected_null',
        'legal_access_flag':     'expected_null',
        'site_control_score':    'expected_null',
        'economic_proxy_score':  'expected_null',
        'serving_utility':       'expected_null',
        'utility_review_required':'expected_null',
        'communications_route_distance':'expected_null',
        'water_capacity_known':  'expected_null',
        'jurisdiction_review_required':'expected_null',
        'manual_imagery_review_status':'expected_null',
        'parcel_owner_module_status':'non_null',  # set to 'not_built'

        # Versioning (PDF Rule 30)
        'run_id':'non_null',
        'run_date':'non_null',
        'scoring_model_version':'non_null',
        'exclusion_model_version':'non_null',
        'cdl_year':'non_null',
        'padus_version':'non_null',
        'fema_nfhl_date':'non_null',
        'nwi_date':'non_null',
        'transmission_dataset_version':'non_null',
        'queue_dataset_date':'non_null',
        'dem_dataset_version':'non_null',
    }

    n_missing = 0
    n_unexpected_null = 0
    n_ok = 0
    for col, expected in required_fields.items():
        if col not in g.columns:
            print(f'  [MISSING] {col}')
            n_missing += 1
            continue
        null_count = g[col].isna().sum()
        null_pct = 100 * null_count / n
        if expected == 'non_null':
            if null_count > 0:
                print(f'  [WARN]    {col} has {null_count:,} nulls ({null_pct:.1f}%)')
                n_unexpected_null += 1
            else:
                n_ok += 1
        elif expected == 'mostly_non_null':
            if null_pct > 20:
                print(f'  [WARN]    {col} has {null_pct:.1f}% null (expected <20)')
                n_unexpected_null += 1
            else:
                n_ok += 1
        elif expected == 'numeric_or_null':
            n_ok += 1
        elif expected == 'expected_null':
            if null_count < n:
                print(f'  [WARN]    {col} expected 100% null but has {n - null_count:,} non-null')
                n_unexpected_null += 1
            else:
                n_ok += 1

    print(f'\n  OK fields: {n_ok}/{len(required_fields)}')
    print(f'  Missing columns: {n_missing}')
    print(f'  Unexpected null/non-null: {n_unexpected_null}')

    # ==================================================================
    # PART B: Geographic sanity
    # ==================================================================
    print('\n' + '=' * 78)
    print('PART B: Geographic sanity')
    print('=' * 78)

    # B1: centroid within stated state bbox
    print('\n--- B1: centroid in state bbox ---')
    bad_centroids = 0
    for st, (xmin, ymin, xmax, ymax) in STATE_BBOX.items():
        sub = g[g.state == st]
        out_of_box = sub[
            (sub.centroid_lon < xmin) | (sub.centroid_lon > xmax)
            | (sub.centroid_lat < ymin) | (sub.centroid_lat > ymax)
        ]
        bad_centroids += len(out_of_box)
        print(f'  {st}: {len(sub):,} candidates, {len(out_of_box):,} centroids outside bbox')
        if len(out_of_box) > 0 and len(out_of_box) <= 5:
            for _, r in out_of_box.iterrows():
                print(f'    -> {r.candidate_id[:8]}.. lon={r.centroid_lon:.3f} lat={r.centroid_lat:.3f} county={r.county_name}')
    print(f'  Total bad centroids: {bad_centroids}')

    # B2: area_acres vs geometry.area (sanity within 0.1%)
    print('\n--- B2: area_acres matches geometry.area ---')
    geom_acres = g.geometry.area * ACRES_PER_M2
    diff = (geom_acres - g.area_acres).abs() / g.area_acres
    bad_area = (diff > 0.001).sum()
    print(f'  Rows where |area_acres - geometry.area| / area_acres > 0.1%: {bad_area:,}')
    if bad_area > 0:
        worst = g.iloc[diff.argsort()[::-1].iloc[:5]]
        for _, r in worst.iterrows():
            g_acres = r.geometry.area * ACRES_PER_M2
            print(f'    {r.candidate_id[:8]}.. stored={r.area_acres:.1f} geom={g_acres:.1f} diff={(r.area_acres-g_acres):.1f}')

    # B3: all area_acres >= 50 (PDF Rule 14 minimum)
    print('\n--- B3: area_acres >= 50 (PDF Rule 14 minimum) ---')
    below_min = (g.area_acres < 50).sum()
    print(f'  Below 50 ac: {below_min:,}')

    # B4: no slope_max > 15 (PDF Rule 13 hard exclusion)
    print('\n--- B4: slope_max_pct <= 15 (PDF Rule 13 hard exclusion) ---')
    high_slope = (g.slope_max_pct > 15).sum()
    print(f'  slope_max > 15%: {high_slope:,}')

    # B5: county_name not empty
    print('\n--- B5: county_name populated ---')
    bad_county = ((g.county_name.isna()) | (g.county_name.str.strip() == '')).sum()
    print(f'  Missing county_name: {bad_county:,}')

    # B6: net_buildable_area_acres <= original_area_acres
    print('\n--- B6: net_buildable <= original_area_acres ---')
    bad_net = (g.net_buildable_area_acres > g.original_area_acres * 1.001).sum()
    print(f'  net > original: {bad_net:,}')

    # B7: buildable_area_ratio in [0, 1]
    print('\n--- B7: buildable_area_ratio in [0, 1] ---')
    bad_ratio = ((g.buildable_area_ratio < 0) | (g.buildable_area_ratio > 1.001)).sum()
    print(f'  ratio out of [0,1]: {bad_ratio:,}')

    # B8: composite_score in [0, 100]
    print('\n--- B8: composite_score in [0, 100] ---')
    bad_comp = ((g.composite_score < 0) | (g.composite_score > 100.001)).sum()
    print(f'  composite out of [0,100]: {bad_comp:,}')

    # ==================================================================
    # PART C: Scoring formula reproduces (May 13 spec)
    # ==================================================================
    print('\n' + '=' * 78)
    print('PART C: Scoring formula integrity')
    print('=' * 78)

    expected_comp = (0.40 * g.utility_score + 0.20 * g.buildability_score
                     + 0.15 * g.supporting_infra_score + 0.15 * g.dev_risk_score) / 0.90
    err = (expected_comp - g.composite_score).abs()
    print(f'  Composite formula error: max={err.max():.6f}, all <0.01: {(err < 0.01).all()}')

    # ==================================================================
    # PART D: Vocabulary compliance (PDF Rule 29 + May 11)
    # ==================================================================
    print('\n' + '=' * 78)
    print('PART D: Vocabulary compliance')
    print('=' * 78)

    pdf29_actions = {'Ignore','Monitor','Manual Review','Parcel Pull',
                     'Utility Desk Check','Ownership Review','Reuse Diligence','Shortlist'}
    actions_in_data = set(g.recommended_action.unique())
    print(f'  recommended_action subset of PDF Rule 29 set: {actions_in_data <= pdf29_actions}')
    print(f'    Used: {sorted(actions_in_data)}')

    may11_act = {'do_not_pitch','internal_diligence_only','apn_owner_pull_required',
                 'broker_verify_required','nda_teaser_possible','buyer_ready_with_caveats'}
    act_in_data = set(g.actionability_status.unique())
    print(f'  actionability_status subset of May-11 set: {act_in_data <= may11_act}')
    print(f'    Used: {sorted(act_in_data)}')

    candidate_types = set(g.candidate_type.unique())
    print(f'  candidate_type in {{greenfield, reuse_node, hybrid}}: '
          f'{candidate_types <= {"greenfield","reuse_node","hybrid"}}')
    print(f'    Used: {sorted(candidate_types)}')

    confidence_vals = set(g.confidence.unique())
    print(f'  confidence in {{low, medium, high}}: '
          f'{confidence_vals <= {"low","medium","high"}}')
    print(f'    Used: {sorted(confidence_vals)}')

    # ==================================================================
    # PART E: Anchor-level transparency (May 13 specific ask)
    # ==================================================================
    print('\n' + '=' * 78)
    print('PART E: Anchor-level transparency (May 13 ask)')
    print('=' * 78)
    anchor_cols = ['primary_anchor_name','primary_anchor_distance_m','primary_anchor_voltage_kv',
                   'primary_anchor_queue_mw_tier','primary_anchor_queue_status_score',
                   'primary_anchor_activation_band','primary_anchor_match_confidence',
                   'primary_anchor_distance_band','zone_fallback_used']
    print(f'  All 9 anchor-level columns present: {all(c in g.columns for c in anchor_cols)}')

    # ==================================================================
    # PART F: Hot-node vs actionable split (May 13 specific ask)
    # ==================================================================
    print('\n' + '=' * 78)
    print('PART F: Hot-node vs actionable-site split (May 13 ask)')
    print('=' * 78)
    hot_node_cols = ['node_activity_score','queue_mw_nearby','ia_executed_nearby',
                     'primary_anchor_activation_band','recent_withdrawals_nearby',
                     'allocation_or_competitive_heat_flag']
    for c in hot_node_cols:
        present = c in g.columns
        print(f'  [{"OK" if present else "MISSING"}] {c}')

    # ==================================================================
    # PART G: Final headline numbers
    # ==================================================================
    print('\n' + '=' * 78)
    print('PART G: Headline numbers')
    print('=' * 78)
    print(f'  Total candidates: {n:,}')
    print(f'  Total acreage: {g.area_acres.sum():,.0f}')
    print(f'  By state: {dict(g.state.value_counts())}')
    print(f'  By recommended_action:')
    for k, v in g.recommended_action.value_counts().items():
        print(f'    {k}: {v:,}')
    print(f'  Composite: min={g.composite_score.min():.1f}, p50={g.composite_score.median():.1f}, p90={g.composite_score.quantile(0.9):.1f}, max={g.composite_score.max():.1f}')


if __name__ == '__main__':
    main()
