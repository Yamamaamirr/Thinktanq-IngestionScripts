"""
Step 6 — Final deep verification before implementation tasks.

Exhaustive cross-checks across:
  A) Deliverable consistency (parquet/csv/fgb)
  B) Cross-attribute consistency (flag<->distance, tier<->score, etc.)
  C) Score integrity (subscores, composite formula, derivable ranges)
  D) Outlier hunting (impossible combinations, hidden masking)
  E) PDF rule compliance (final pass)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

PARQUET = Path('candidate_areas/outputs/candidates_final.parquet')
CSV     = Path('candidate_areas/outputs/candidates_final.csv')
FGB     = Path('candidate_areas/outputs/candidates_final.fgb')


class Checker:
    def __init__(self):
        self.results = []  # (section, label, pass/fail, details)

    def check(self, section, label, ok, details=''):
        self.results.append((section, label, bool(ok), details))
        sym = 'PASS' if ok else 'FAIL'
        det = f'  ({details})' if details and not ok else ''
        print(f'  [{sym}] {label}{det}')

    def section(self, name):
        print(f'\n{"=" * 78}\n{name}\n{"=" * 78}')

    def summary(self):
        total = len(self.results)
        passed = sum(1 for _, _, ok, _ in self.results if ok)
        failed = total - passed
        print(f'\n\n{"#" * 78}\n FINAL SUMMARY\n{"#" * 78}')
        print(f'  Total checks: {total}')
        print(f'  Passed:       {passed}')
        print(f'  Failed:       {failed}')
        if failed:
            print('\n  Failed checks:')
            for sec, lbl, ok, det in self.results:
                if not ok:
                    print(f'    [{sec}] {lbl}: {det}')


def main():
    c = Checker()

    print(f'Loading parquet: {PARQUET}')
    g = gpd.read_parquet(PARQUET)
    n = len(g)
    print(f'  {n:,} rows x {len(g.columns)} cols\n')
    print(f'Loading CSV (~30s) ...')
    csv_df = pd.read_csv(CSV, low_memory=False)
    print(f'  {len(csv_df):,} rows')
    print(f'Loading FGB ...')
    fgb_df = gpd.read_file(FGB)
    print(f'  {len(fgb_df):,} rows')

    # ===================================================================
    # SECTION A: Deliverable consistency
    # ===================================================================
    c.section('SECTION A: Deliverable consistency (parquet vs csv vs fgb)')
    c.check('A', 'parquet rows == csv rows', len(g) == len(csv_df),
            f'parquet={len(g)}, csv={len(csv_df)}')
    c.check('A', 'parquet rows == fgb rows', len(g) == len(fgb_df),
            f'parquet={len(g)}, fgb={len(fgb_df)}')
    c.check('A', 'parquet candidate_id set == csv', set(g.candidate_id) == set(csv_df.candidate_id))
    c.check('A', 'parquet candidate_id set == fgb', set(g.candidate_id) == set(fgb_df.candidate_id))
    c.check('A', 'all candidate_ids unique in parquet', g.candidate_id.is_unique)
    c.check('A', 'all candidate_ids unique in csv', csv_df.candidate_id.is_unique)
    c.check('A', 'all candidate_ids unique in fgb', fgb_df.candidate_id.is_unique)
    c.check('A', 'parquet sorted by composite desc',
            (g.composite_score.diff().dropna() <= 0.0001).all())
    c.check('A', 'csv sorted by composite desc',
            (csv_df.composite_score.diff().dropna() <= 0.0001).all())
    c.check('A', 'parquet CRS = EPSG:5070', g.crs.to_epsg() == 5070)
    c.check('A', 'fgb CRS = EPSG:5070', fgb_df.crs.to_epsg() == 5070)
    c.check('A', 'all parquet geometries valid', g.geometry.is_valid.all())
    c.check('A', 'no empty parquet geometries', (~g.geometry.is_empty).all())

    # ===================================================================
    # SECTION B: Cross-attribute consistency
    # ===================================================================
    c.section('SECTION B: Cross-attribute consistency')

    # Flags should match distance thresholds (500m / 3 mi etc.)
    c.check('B', 'near_padus_flag == (dist < 500)',
            (g['near_padus_flag'] == (g['nearest_padus_distance_m'].fillna(1e9) < 500)).all())
    c.check('B', 'near_wetland_flag == (dist < 500)',
            (g['near_wetland_flag'] == (g['nearest_wetland_distance_m'].fillna(1e9) < 500)).all())
    c.check('B', 'adjacent_floodway_flag == (dist < 500)',
            (g['adjacent_floodway_flag'] == (g['nearest_floodway_distance_m'].fillna(1e9) < 500)).all())
    c.check('B', 'fema_ae_overlap_flag == (dist == 0)',
            (g['fema_ae_overlap_flag'] == (g['nearest_fema_ae_distance_m'].fillna(1e9) <= 0)).all())
    c.check('B', 'fema_ae_adjacent_flag == (dist < 500)',
            (g['fema_ae_adjacent_flag'] == (g['nearest_fema_ae_distance_m'].fillna(1e9) < 500)).all())
    c.check('B', 'radar_review_flag == (dist_miles < 3)',
            (g['radar_review_flag'] == (g['radar_distance_miles'].fillna(1e9) < 3)).all())
    c.check('B', 'oversized_flag == (size_class == region)',
            (g['oversized_flag'] == (g['size_class'] == 'region')).all())
    c.check('B', 'within_water_service_area == (dist == 0)',
            (g['within_water_service_area'] == (g['nearest_water_service_distance_m'] == 0)).all())

    # Tier-to-score mappings
    slope_map = g.groupby('slope_tier')['slope_tier_score'].unique()
    c.check('B', 'slope_tier=ideal => score=100',
            all(slope_map.get('ideal', [100]) == [100.0]) if 'ideal' in slope_map else True)
    c.check('B', 'slope_tier=acceptable => score=80',
            all(slope_map.get('acceptable', [80]) == [80.0]) if 'acceptable' in slope_map else True)
    c.check('B', 'slope_tier=penalized => score=50',
            all(slope_map.get('penalized', [50]) == [50.0]) if 'penalized' in slope_map else True)

    acre_map = g.groupby('acreage_tier')['acreage_tier_score'].unique()
    c.check('B', 'acreage_tier=small => score=50',
            all(acre_map.get('small', [50]) == [50.0]))
    c.check('B', 'acreage_tier=moderate => score=70',
            all(acre_map.get('moderate', [70]) == [70.0]))
    c.check('B', 'acreage_tier=large => score=85',
            all(acre_map.get('large', [85]) == [85.0]))
    c.check('B', 'acreage_tier=very_large => score=95',
            all(acre_map.get('very_large', [95]) == [95.0]))
    c.check('B', 'acreage_tier=strategic_scale => score=100',
            all(acre_map.get('strategic_scale', [100]) == [100.0]))

    # Acreage tier vs area_acres
    seg_check = (
        ((g.acreage_tier == 'small') & (g.area_acres >= 50) & (g.area_acres < 100)) |
        ((g.acreage_tier == 'moderate') & (g.area_acres >= 100) & (g.area_acres < 250)) |
        ((g.acreage_tier == 'large') & (g.area_acres >= 250) & (g.area_acres < 500)) |
        ((g.acreage_tier == 'very_large') & (g.area_acres >= 500) & (g.area_acres < 1000)) |
        ((g.acreage_tier == 'strategic_scale') & (g.area_acres >= 1000))
    )
    c.check('B', 'acreage_tier matches area_acres band', seg_check.all(),
            f'{(~seg_check).sum()} mismatches')

    # Seismic tier vs PGA
    seismic_check = (
        ((g.seismic_hazard_tier == 'very_low') & (g.seismic_hazard_pga < 0.10)) |
        ((g.seismic_hazard_tier == 'low')      & (g.seismic_hazard_pga >= 0.10) & (g.seismic_hazard_pga < 0.25)) |
        ((g.seismic_hazard_tier == 'moderate') & (g.seismic_hazard_pga >= 0.25) & (g.seismic_hazard_pga < 0.50)) |
        ((g.seismic_hazard_tier == 'high')     & (g.seismic_hazard_pga >= 0.50) & (g.seismic_hazard_pga < 1.00)) |
        ((g.seismic_hazard_tier == 'very_high')& (g.seismic_hazard_pga >= 1.00))
    )
    c.check('B', 'seismic_tier matches PGA band', seismic_check.all(),
            f'{(~seismic_check).sum()} mismatches')

    # size_class vs area_acres
    size_check = (
        ((g.size_class == 'site')    & (g.area_acres < 500)) |
        ((g.size_class == 'campus')  & (g.area_acres >= 500) & (g.area_acres < 5000)) |
        ((g.size_class == 'region')  & (g.area_acres >= 5000))
    )
    c.check('B', 'size_class matches area_acres band', size_check.all(),
            f'{(~size_check).sum()} mismatches')

    # ===================================================================
    # SECTION C: Score integrity
    # ===================================================================
    c.section('SECTION C: Score integrity')

    # composite formula
    expected = (0.40*g.utility_score + 0.20*g.buildability_score
                + 0.15*g.supporting_infra_score + 0.15*g.dev_risk_score) / 0.90
    err = (expected - g.composite_score).abs()
    c.check('C', 'composite formula reproduces exactly',
            (err < 0.01).all(), f'max err = {err.max():.6f}')

    # All scores in [0, 100]
    for sc in ['composite_score','utility_score','buildability_score','supporting_infra_score',
               'dev_risk_score','land_cover_score','constraint_score','slope_tier_score',
               'transmission_score','pipeline_score','rail_score','water_score',
               'seismic_score','drought_score','radar_score','padus_score','wetland_score','floodway_score',
               'acreage_tier_score']:
        if sc in g.columns:
            mn = g[sc].min()
            mx = g[sc].max()
            c.check('C', f'{sc} in [0, 100]',
                    (mn >= 0) and (mx <= 100.001), f'min={mn}, max={mx}')

    # supporting_infra_score = 0.35*trans + 0.25*pipe + 0.20*rail + 0.20*water  (Step2D)
    s_expected = (0.35*g.transmission_score + 0.25*g.pipeline_score
                  + 0.20*g.rail_score + 0.20*g.water_score)
    s_err = (s_expected - g.supporting_infra_score).abs()
    c.check('C', 'supporting_infra = 0.35*T + 0.25*P + 0.20*R + 0.20*W',
            (s_err < 0.01).all(), f'max err = {s_err.max():.6f}')

    # dev_risk = 0.25*seismic + 0.20*drought + 0.15*radar + 0.15*padus + 0.15*wet + 0.10*flood  (Step2E)
    d_expected = (0.25*g.seismic_score + 0.20*g.drought_score + 0.15*g.radar_score
                  + 0.15*g.padus_score + 0.15*g.wetland_score + 0.10*g.floodway_score)
    d_err = (d_expected - g.dev_risk_score).abs()
    c.check('C', 'dev_risk = 0.25*S + 0.20*D + 0.15*R + 0.15*P + 0.15*W + 0.10*F',
            (d_err < 0.01).all(), f'max err = {d_err.max():.6f}')

    # score_v1_observable should match composite_score (since site_control is deferred)
    if 'score_v1_observable' in g.columns:
        # score_v1_observable is a boolean in this schema (T = score is observable-only, no parcel data)
        if g.score_v1_observable.dtype == bool:
            c.check('C', 'score_v1_observable=True for all rows (Phase 1)',
                    g.score_v1_observable.all())

    # No null in critical score columns
    for col in ['composite_score','utility_score','buildability_score','supporting_infra_score','dev_risk_score']:
        c.check('C', f'{col} has no nulls', g[col].notna().all())

    # ===================================================================
    # SECTION D: Outlier hunting
    # ===================================================================
    c.section('SECTION D: Outlier hunting')

    # Slope hard exclusion should be 0
    c.check('D', 'no slope_max > 15 anywhere',
            (g.slope_max_pct > 15).sum() == 0, f'{(g.slope_max_pct > 15).sum()} rows still present')

    # Candidate_status only pass or manual_review
    c.check('D', 'candidate_status in {pass, manual_review}',
            set(g.candidate_status.unique()) <= {'pass', 'manual_review'})

    # No "pass" rows with hard-exclusion violations
    pass_g = g[g.candidate_status == 'pass']
    c.check('D', 'no pass rows with slope_max > 15', (pass_g.slope_max_pct > 15).sum() == 0)
    c.check('D', 'no pass rows with building_footprint > 5%',
            (pass_g.building_footprint_pct > 5).sum() == 0)
    c.check('D', 'no pass rows with buildable_area_ratio < 0.25',
            (pass_g.buildable_area_ratio.fillna(1.0) < 0.25).sum() == 0)
    c.check('D', 'no pass rows with slope_mean > 15',
            (pass_g.slope_mean_pct > 15).sum() == 0)

    # area_acres minimum (PDF rule)
    c.check('D', 'all area_acres >= 50', (g.area_acres >= 50).all(),
            f'{(g.area_acres < 50).sum()} below minimum')

    # No rows where utility_score is high but no anchors
    weird_util = g[(g.utility_score > 50) & (g.num_anchors_in_range == 0)]
    c.check('D', 'no high utility_score with zero anchors', len(weird_util) == 0,
            f'{len(weird_util)} mismatches')

    # Composite > 80 with multiple bad flags is suspicious
    bad_flags = (g.adjacent_floodway_flag.astype(int) + g.fema_ae_overlap_flag.astype(int)
                 + g.near_wetland_flag.astype(int) + g.radar_review_flag.astype(int)
                 + g.slope_review_flag.astype(int))
    high_with_3_flags = g[(g.composite_score > 80) & (bad_flags >= 3)]
    c.check('D', 'no composite>80 with 3+ adversity flags',
            len(high_with_3_flags) == 0,
            f'{len(high_with_3_flags)} suspect rows (will surface, not fail)')

    # Confidence consistency: zone_fallback should usually -> low/medium, not high
    zf = g[g.zone_fallback_used == True]
    zf_high = (zf.confidence == 'high').sum() if len(zf) else 0
    c.check('D', 'zone_fallback candidates not labeled high confidence',
            zf_high == 0, f'{zf_high} fallback rows tagged high')

    # ===================================================================
    # SECTION E: PDF rule compliance final pass
    # ===================================================================
    c.section('SECTION E: PDF rule compliance')

    valid_actions = {'Ignore','Monitor','Manual Review','Parcel Pull',
                     'Utility Desk Check','Ownership Review','Reuse Diligence','Shortlist'}
    c.check('E', 'recommended_action subset of PDF Rule 29 vocab',
            set(g.recommended_action.unique()) <= valid_actions)

    valid_act = {'do_not_pitch','internal_diligence_only','apn_owner_pull_required',
                 'broker_verify_required','nda_teaser_possible','buyer_ready_with_caveats'}
    c.check('E', 'actionability_status subset of May-11 vocab',
            set(g.actionability_status.unique()) <= valid_act)

    # Action rule reproducibility (PDF Rule 29 + Shortlist band)
    # manual_review -> Manual Review
    # composite>=90 + 500-5000 ac + no slope flag + medium/high conf + >=3 anchors -> Shortlist
    # composite>=75 -> Parcel Pull (or above)
    # 65<=composite<75 -> Monitor
    # composite<65 -> Ignore
    shortlist_mask = (
        (g.composite_score >= 90)
        & (g.area_acres >= 500) & (g.area_acres <= 5000)
        & (~g.slope_review_flag.fillna(False).astype(bool))
        & (~g.oversized_flag.fillna(False).astype(bool))
        & (g.confidence.isin(['medium', 'high']))
        & (g.num_anchors_in_range.fillna(0) >= 3)
    )
    expected_action = np.where(
        g.candidate_status == 'manual_review', 'Manual Review',
        np.where(shortlist_mask, 'Shortlist',
            np.where(g.composite_score >= 75, 'Parcel Pull',
                     np.where(g.composite_score >= 65, 'Monitor', 'Ignore')))
    )
    action_match = (expected_action == g.recommended_action.values).sum()
    c.check('E', 'PDF Rule 29 + Shortlist action assignment correct',
            action_match == n, f'{n - action_match} mismatches')

    # Audit-trail fields populated
    for col in ['run_id','run_date','scoring_model_version','exclusion_model_version',
                'cdl_year','padus_version','fema_nfhl_date','nwi_date',
                'transmission_dataset_version','queue_dataset_date','dem_dataset_version']:
        c.check('E', f'{col} populated (PDF Rule 30)', g[col].notna().all())

    # Phase 2 placeholders all null (as committed to Owen)
    for col in ['parcel_count','owner_count','assessed_value_per_acre','land_use_code',
                'serving_utility','communications_route_distance','water_capacity_known',
                'jurisdiction_review_required','manual_imagery_review_status',
                'site_control_score','economic_proxy_score']:
        c.check('E', f'{col} all null (Phase 2 placeholder)',
                g[col].isna().all(), f'{g[col].notna().sum()} non-null')

    c.summary()


if __name__ == '__main__':
    main()
