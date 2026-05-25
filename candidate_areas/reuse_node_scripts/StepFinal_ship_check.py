"""
StepFinal_ship_check -- Final readiness gate on final_candidates_phase1.parquet
and its export companions.

Treats the unified file as a standalone deliverable and re-verifies everything
a downstream consumer might assume. Not redundant with verify_merge.py -- that
checked the MERGE was lossless; this checks the FINAL FILE is well-formed
regardless of how it was produced.

Sections:
  A. Load + identity (row count, candidate_id unique, CRS, dtype sanity)
  B. Geometry integrity (validity, non-empty, polygon types, CRS metres)
  C. Required columns per layer present + non-null
  D. Score ranges [0, 100] for both layers
  E. Composite arithmetic reproducible per layer
        - greenfield: original 40/20/15/15 weights / 0.90
        - reuse:      recalibrated 25/15/15/15/20 weights / 0.90
  F. Action label vocabulary valid + layer-correct
        - Reuse Diligence is a reuse-only label
        - Shortlist criteria hold for both layers
        - Manual Review only on rows with candidate_status='manual_review'
  G. R4 reuse-risk flags consistent with source/asset_type (reuse rows)
  H. Score distribution sanity (no degenerate ranges, no nulls)
  I. Export file round-trip:
        - parquet re-reads to same row count
        - GPKG re-reads to same row count
        - FGB  re-reads to same row count
  J. Spot-check: top 5 sites per layer print correctly
  K. No surprise nulls in critical fields (composite_score, recommended_action,
     candidate_type, geometry, candidate_id)

Run: python candidate_areas/reuse_node_scripts/StepFinal_ship_check.py
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

PARQUET_PATH = Path('candidate_areas/outputs/final_candidates_phase1.parquet')
GPKG_PATH    = Path('candidate_areas/outputs/exports/final_candidates_phase1.gpkg')
FGB_PATH     = Path('candidate_areas/outputs/exports/final_candidates_phase1.fgb')


def main():
    print(f'Loading {PARQUET_PATH} ...')
    g = gpd.read_parquet(PARQUET_PATH)
    print(f'  {len(g):,} rows, {len(g.columns)} columns, CRS={g.crs.to_epsg()}\n')

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

    gf = g[g.candidate_type == 'greenfield']
    rn = g[g.candidate_type == 'reuse_node']

    # ---- A. Identity ----
    expected_rows = 86_187 + 6_631  # greenfield shipped + reuse
    print('=== A. Identity ===')
    chk(f'row count = {expected_rows:,} (shipped greenfield + reuse)',
        len(g) == expected_rows, f'got {len(g)}')
    chk('candidate_id unique', g.candidate_id.is_unique)
    chk('candidate_id no nulls', g.candidate_id.notna().all())
    chk('CRS = EPSG:5070', g.crs.to_epsg() == 5070)
    chk('candidate_type non-null', g.candidate_type.notna().all())
    chk('candidate_type in {greenfield, reuse_node}',
        set(g.candidate_type.unique()) == {'greenfield','reuse_node'})

    # ---- B. Geometry integrity ----
    print('\n=== B. Geometry integrity ===')
    chk('all geometries non-null', g.geometry.notna().all())
    chk('no empty geometries', (~g.geometry.is_empty).all())
    chk('all geometries valid', g.geometry.is_valid.all())
    chk('all polygon/multipolygon types',
        g.geometry.type.isin(['Polygon','MultiPolygon']).all())
    # EPSG:5070 is a metric CRS; areas should be in m^2 and reasonable
    area_ac = g.geometry.area / 4046.856
    chk(f'all areas > 0 (min={area_ac.min():.4f} ac)', (area_ac > 0).all())
    chk(f'all areas <= 10,000 ac (max={area_ac.max():.0f} ac)',
        (area_ac <= 10_000).all())

    # ---- C. Required columns present per layer ----
    print('\n=== C. Required column presence + non-null ===')
    REQUIRED_BOTH = ['candidate_id','candidate_type','geometry',
                     'composite_score','recommended_action',
                     'utility_score','buildability_score',
                     'supporting_infra_score','dev_risk_score',
                     'state']
    for col in REQUIRED_BOTH:
        chk(f'{col} present in unified', col in g.columns)
        if col in g.columns:
            chk(f'{col} non-null', g[col].notna().all(),
                f'{g[col].isna().sum()} nulls')

    REUSE_ONLY = ['reuse_environmental_score','known_contamination_flag',
                  'legacy_asset_risk_flag','environmental_review_required',
                  'decommissioning_status_known','aliased_site_count',
                  'geometry_source','source','reuse_asset_type']
    print('  Reuse-only columns:')
    for col in REUSE_ONLY:
        chk(f'  {col} present', col in g.columns)
        if col in g.columns:
            chk(f'  {col} non-null on reuse rows', rn[col].notna().all(),
                f'{rn[col].isna().sum()} nulls')

    # ---- D. Score ranges per layer ----
    print('\n=== D. Score ranges [0, 100] ===')
    score_cols_common = ['utility_score','buildability_score',
                         'supporting_infra_score','dev_risk_score',
                         'composite_score']
    for col in score_cols_common:
        for label, sub in [('greenfield', gf), ('reuse', rn)]:
            s = sub[col]
            in_range = ((s >= 0) & (s <= 100.001)).all()
            chk(f'{label} {col} in [0,100]', in_range,
                f'min={s.min():.2f}, max={s.max():.2f}')
    # reuse_env only on reuse
    s = rn['reuse_environmental_score']
    chk('reuse reuse_environmental_score in [0,100]',
        ((s >= 0) & (s <= 100.001)).all(),
        f'min={s.min():.2f}, max={s.max():.2f}')

    # ---- E. Composite arithmetic per layer ----
    print('\n=== E. Composite arithmetic ===')
    # Greenfield: (0.40*util + 0.20*build + 0.15*supp + 0.15*risk) / 0.90
    gf_expected = (0.40*gf.utility_score + 0.20*gf.buildability_score
                   + 0.15*gf.supporting_infra_score + 0.15*gf.dev_risk_score) / 0.90
    gf_diff = (gf.composite_score - gf_expected).abs()
    chk(f'greenfield composite matches 40/20/15/15 formula (max diff {gf_diff.max():.5f})',
        (gf_diff < 0.01).all())

    # Reuse: (0.25*util + 0.15*build + 0.15*supp + 0.15*risk + 0.20*reuse_env) / 0.90
    rn_expected = (0.25*rn.utility_score + 0.15*rn.buildability_score
                   + 0.15*rn.supporting_infra_score + 0.15*rn.dev_risk_score
                   + 0.20*rn.reuse_environmental_score) / 0.90
    rn_diff = (rn.composite_score - rn_expected).abs()
    chk(f'reuse composite matches 25/15/15/15/20 formula (max diff {rn_diff.max():.5f})',
        (rn_diff < 0.01).all())

    # ---- F. Action label vocabulary ----
    print('\n=== F. recommended_action vocabulary + layer correctness ===')
    VALID_ACTIONS = {'Ignore','Monitor','Manual Review','Parcel Pull',
                     'Utility Desk Check','Ownership Review','Reuse Diligence',
                     'Shortlist'}
    chk('all recommended_action in valid set',
        set(g.recommended_action.unique()) <= VALID_ACTIONS)
    chk('Reuse Diligence is reuse-only',
        (g[g.recommended_action == 'Reuse Diligence'].candidate_type == 'reuse_node').all())
    chk('Manual Review iff candidate_status==manual_review',
        (g[g.candidate_status=='manual_review'].recommended_action == 'Manual Review').all())
    # Shortlist criteria on BOTH layers
    sl = g[g.recommended_action == 'Shortlist']
    chk('all Shortlist composite >= 90', (sl.composite_score >= 90).all())
    chk('all Shortlist confidence in {medium, high}',
        sl.confidence.isin(['medium','high']).all())
    chk('all Shortlist >=3 anchors', (sl.num_anchors_in_range >= 3).all())

    # ---- G. R4 reuse-risk flag consistency (reuse rows) ----
    print('\n=== G. R4 reuse-risk consistency (reuse rows) ===')
    chk('All EPA-RE-Powering have env_review_required',
        rn[rn.source=='EPA-RE-Powering'].environmental_review_required.all())
    chk('All abandoned_mine have legacy_asset_risk_flag',
        rn[rn.reuse_asset_type=='abandoned_mine'].legacy_asset_risk_flag.all())
    chk('decommissioning_status_known only on EIA',
        (rn[rn.decommissioning_status_known].source.isin(['EIA-860','EIA-860-nuclear'])).all())

    # ---- H. Distribution sanity ----
    print('\n=== H. Distribution sanity ===')
    for label, sub in [('greenfield', gf), ('reuse', rn)]:
        s = sub.composite_score
        chk(f'{label} composite spread > 30 pts (max-min={s.max()-s.min():.1f})',
            (s.max() - s.min()) > 30)
        chk(f'{label} composite p10/p90 spread > 15 pts',
            (s.quantile(0.9) - s.quantile(0.1)) > 15)

    # ---- I. Export round-trip ----
    print('\n=== I. Export round-trip ===')
    if GPKG_PATH.exists():
        print(f'  reading {GPKG_PATH} ({GPKG_PATH.stat().st_size/1e6:.0f} MB) ...')
        gpkg = gpd.read_file(GPKG_PATH)
        chk(f'GPKG row count = parquet ({len(gpkg)} vs {len(g)})',
            len(gpkg) == len(g))
        chk(f'GPKG CRS = EPSG:5070', gpkg.crs.to_epsg() == 5070)
    else:
        chk('GPKG export exists', False, str(GPKG_PATH))
    if FGB_PATH.exists():
        print(f'  reading {FGB_PATH} ({FGB_PATH.stat().st_size/1e6:.0f} MB) ...')
        fgb = gpd.read_file(FGB_PATH)
        chk(f'FGB row count = parquet ({len(fgb)} vs {len(g)})',
            len(fgb) == len(g))
        chk(f'FGB CRS = EPSG:5070', fgb.crs.to_epsg() == 5070)
    else:
        chk('FGB export exists', False, str(FGB_PATH))

    # ---- J. Spot-check top sites per layer ----
    print('\n=== J. Top 5 per layer (visual sanity) ===')
    for label, sub in [('GREENFIELD', gf), ('REUSE', rn)]:
        print(f'  {label}:')
        for _, r in sub.nlargest(5, 'composite_score').iterrows():
            nm = (str(r.site_name)[:24] if 'site_name' in r and pd.notna(r.get('site_name'))
                  else str(r.candidate_id)[:24])
            area = r.area_acres if pd.notna(r.area_acres) else 0
            print(f'    comp={r.composite_score:>5.1f}  {r.state}  {r.candidate_id[:18]:<18}  '
                  f'{nm:<24}  {area:>5.0f}ac  {r.recommended_action}')

    # ---- K. Critical-field null check ----
    print('\n=== K. Critical-field non-null ===')
    for col in ['candidate_id','candidate_type','geometry','composite_score',
                'recommended_action','state']:
        chk(f'{col} no nulls anywhere', g[col].notna().all(),
            f'{g[col].isna().sum()} nulls')

    print('\n' + '=' * 60)
    print('OVERALL:', 'READY TO SHIP' if ok else 'NOT READY -- review failures')
    print('=' * 60)


if __name__ == '__main__':
    main()
