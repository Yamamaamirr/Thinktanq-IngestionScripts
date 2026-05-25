"""
StepR5_verify -- Deep verification of the fully-scored reuse-node table.

Checks:
  A. Schema & identity
  B. All score columns present, never null, in [0,100]
  C. composite_score matches the weighted formula
  D. buildability_score matches its formula
  E. dev_risk_score matches the reuse-weighted formula
  F. R4 risk-field cross-checks against source/asset_type
  G. recommended_action logic gates fire correctly
  H. top_reason_codes integrity (length, type)
  I. Reason-code firing sanity (only when corresponding flag is True)
  J. Score distribution sanity (no degenerate ranges)
  K. Confidence-tier field sanity
  L. Spot-check 5 random rows end-to-end

Run: python candidate_areas/reuse_node_scripts/StepR5_verify.py
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')


def main():
    g = gpd.read_parquet(PATH)
    print(f'Loaded: {len(g):,} rows, {len(g.columns)} columns\n')

    ok = True
    def chk(label, passed, detail=''):
        nonlocal ok
        sym = 'PASS' if passed else 'FAIL'
        msg = f'  [{sym}] {label}'
        if detail and not passed:
            msg += f'  ({detail})'
        print(msg)
        if not passed:
            ok = False

    print('=== A. Schema & identity ===')
    chk('row count = 6,631', len(g) == 6631)
    chk('candidate_id unique', g.candidate_id.is_unique)
    chk('geometry all polygon/multipolygon',
        g.geometry.type.isin(['Polygon','MultiPolygon']).all())
    chk('all geometries valid', g.geometry.is_valid.all())
    chk('CRS = EPSG:5070', g.crs.to_epsg() == 5070)

    # Note: land_cover_score is fixed 85.0; slope_tier_score may include real
    # sampled rows now (was constant 80.0 when slope was deferred). Both checks
    # below validate ranges, not constancy.
    print('\n=== B. Score columns present & in [0,100] ===')
    score_cols = ['utility_score','buildability_score','supporting_infra_score',
                  'dev_risk_score','composite_score',
                  'land_cover_score','constraint_score',
                  'transmission_score','pipeline_score','rail_score','water_score',
                  'seismic_score','drought_score','radar_score','padus_score',
                  'wetland_score','floodway_score','reuse_environmental_score']
    for c in score_cols:
        chk(f'{c} present', c in g.columns)
        if c in g.columns:
            chk(f'{c} never null', g[c].notna().all())
            chk(f'{c} in [0,100]',
                ((g[c] >= 0) & (g[c] <= 100.001)).all(),
                f'min={g[c].min():.2f}, max={g[c].max():.2f}')

    print('\n=== C. Composite arithmetic check (recalibrated 25/15/15/15/20) ===')
    W_U, W_B, W_S, W_R, W_RE = 0.25, 0.15, 0.15, 0.15, 0.20
    WT = W_U + W_B + W_S + W_R + W_RE
    expected = (W_U*g.utility_score + W_B*g.buildability_score
                + W_S*g.supporting_infra_score + W_R*g.dev_risk_score
                + W_RE*g.reuse_environmental_score) / WT
    diff = (g.composite_score - expected).abs()
    chk(f'composite_score matches recalibrated formula (max diff {diff.max():.5f})',
        (diff < 0.01).all())

    print('\n=== D. Buildability arithmetic check ===')
    expected_b = (0.35*g.land_cover_score + 0.25*g.acreage_tier_score
                  + 0.25*g.slope_tier_score + 0.15*g.constraint_score)
    diff_b = (g.buildability_score - expected_b).abs()
    chk(f'buildability_score matches formula (max diff {diff_b.max():.5f})',
        (diff_b < 0.01).all())

    print('\n=== E. Dev-risk arithmetic check (greenfield-equivalent 6-component) ===')
    expected_dr = (0.25*g.seismic_score + 0.20*g.drought_score
                   + 0.15*g.radar_score + 0.15*g.padus_score
                   + 0.15*g.wetland_score + 0.10*g.floodway_score)
    diff_dr = (g.dev_risk_score - expected_dr).abs()
    chk(f'dev_risk_score matches 6-component formula (max diff {diff_dr.max():.5f})',
        (diff_dr < 0.01).all())

    print('\n=== F. R4 risk-field cross-checks ===')
    chk('All EPA-RE-Powering have environmental_review_required',
        g[g.source=='EPA-RE-Powering'].environmental_review_required.all())
    chk('All OSM landfill have environmental_review_required',
        g[(g.source=='OpenStreetMap') & (g.reuse_asset_type=='landfill')].environmental_review_required.all())
    chk('All OSM quarry have environmental_review_required',
        g[(g.source=='OpenStreetMap') & (g.reuse_asset_type=='quarry_mine')].environmental_review_required.all())
    chk('All abandoned_mine have legacy_asset_risk_flag',
        g[g.reuse_asset_type=='abandoned_mine'].legacy_asset_risk_flag.all())
    chk('decommissioning_status_known only on EIA',
        (g[g.decommissioning_status_known].source.isin(['EIA-860','EIA-860-nuclear'])).all())

    print('\n=== G. recommended_action logic checks ===')
    chk('manual_review status -> Manual Review action',
        (g[g.candidate_status=='manual_review'].recommended_action == 'Manual Review').all())
    shortlist = g[g.recommended_action == 'Shortlist']
    chk(f'All Shortlist have composite >= 90 (n={len(shortlist)})',
        (shortlist.composite_score >= 90).all())
    chk('All Shortlist have 500-5000 ac',
        ((shortlist.area_acres >= 500) & (shortlist.area_acres <= 5000)).all())
    chk('All Shortlist have num_anchors_in_range >= 3',
        (shortlist.num_anchors_in_range >= 3).all())
    chk('All Shortlist have confidence in {medium, high}',
        shortlist.confidence.isin(['medium','high']).all())
    reuse_dil = g[g.recommended_action == 'Reuse Diligence']
    chk(f'All Reuse Diligence have contamination or legacy flag (n={len(reuse_dil)})',
        (reuse_dil.known_contamination_flag | reuse_dil.legacy_asset_risk_flag).all())
    chk('All Reuse Diligence have composite >= 60',
        (reuse_dil.composite_score >= 60).all())
    ig = g[g.recommended_action == 'Ignore']
    chk(f'All Ignore have composite < 65 OR excluded (n={len(ig)})',
        ((ig.composite_score < 65) | (ig.candidate_status == 'excluded')).all())

    print('\n=== H. top_reason_codes integrity ===')
    chk('All rows have non-empty top_reason_codes',
        (g.top_reason_codes.apply(len) > 0).all())
    # After parquet round-trip pandas list columns come back as numpy arrays.
    # The contents are identical -- check iterability + string elements.
    import numpy as np
    chk('All top_reason_codes are iterable list-like',
        g.top_reason_codes.apply(lambda x: isinstance(x, (list, np.ndarray))).all())
    chk('All top_reason_codes contain strings',
        g.top_reason_codes.apply(lambda x: all(isinstance(c, str) for c in x)).all())
    chk('top_reason_codes capped at 5',
        (g.top_reason_codes.apply(len) <= 5).all())

    print('\n=== I. Reason-code firing sanity ===')
    has_contam = g.top_reason_codes.apply(lambda L: 'reuse_high_contamination_risk' in L)
    chk('reuse_high_contamination_risk code only when known_contamination_flag True',
        g[has_contam].known_contamination_flag.all())
    has_decomm = g.top_reason_codes.apply(lambda L: 'decommissioning_timeline_known' in L)
    chk('decommissioning_timeline_known code only when decom flag True',
        g[has_decomm].decommissioning_status_known.all())
    has_real = g.top_reason_codes.apply(lambda L: 'high_confidence_real_footprint' in L)
    chk('high_confidence_real_footprint code only when OSM_POLYGON',
        (g[has_real].geometry_source == 'OSM_POLYGON').all())

    print('\n=== J. Distribution sanity ===')
    s = g.composite_score
    chk(f'composite range > 30 pts wide (max-min={s.max()-s.min():.1f})',
        (s.max() - s.min()) > 30)
    chk(f'p90-p10 spread > 15 pts (got {s.quantile(0.9)-s.quantile(0.1):.1f})',
        (s.quantile(0.9) - s.quantile(0.1)) > 15)
    nuc_med = g[g.source=='EIA-860-nuclear'].composite_score.median()
    overall_med = g.composite_score.median()
    chk(f'Nuclear median ({nuc_med:.2f}) > overall median ({overall_med:.2f})',
        nuc_med > overall_med)
    zf = g[g.utility_module_status == 'zone_fallback']
    chk(f'All zone_fallback have utility_score == 0 (n={len(zf)})',
        (zf.utility_score == 0).all())

    print('\n=== K. Confidence field sanity ===')
    chk('confidence in {high, medium, low}',
        set(g.confidence.unique()) <= {'high','medium','low'})
    high_corr = ((g.utility_score >= 50) & (g.buildability_score >= 50)
                 & (g.supporting_infra_score >= 50) & (g.dev_risk_score >= 50))
    print(f'  rows with all 4 subscores >= 50: {high_corr.sum():,}')
    print(f'  confidence breakdown among those: {g[high_corr].confidence.value_counts().to_dict()}')

    print('\n=== L. Spot-check 5 random rows ===')
    sample = g.sample(5, random_state=42)
    for _, r in sample.iterrows():
        nm = str(r.site_name)[:24] if pd.notna(r.site_name) else '(unnamed)'
        print(f'  {r.candidate_id[:22]:<22} {r.state} {nm:<24} {r.area_acres:>5.0f}ac '
              f'src={r.source[:6]:<6} type={r.reuse_asset_type[:16]:<16}')
        print(f'     scores: util={r.utility_score:.0f} build={r.buildability_score:.0f} '
              f'supp={r.supporting_infra_score:.0f} risk={r.dev_risk_score:.0f} '
              f'comp={r.composite_score:.1f} -> {r.recommended_action}')
        flags = []
        if r.environmental_review_required: flags.append('env_review')
        if r.legacy_asset_risk_flag:        flags.append('legacy_risk')
        if r.known_contamination_flag:      flags.append('contam')
        if r.decommissioning_status_known:  flags.append('decom_known')
        print(f'     R4 flags: {flags if flags else "none"}')
        print(f'     codes:    {r.top_reason_codes}')

    print('\n' + '=' * 60)
    print('OVERALL:', 'ALL VERIFICATIONS PASSED' if ok else 'SOME CHECKS FAILED')
    print('=' * 60)


if __name__ == '__main__':
    main()
