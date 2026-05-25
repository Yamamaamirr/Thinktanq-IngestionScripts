"""
StepFinal_verify_merge -- Deep cross-check that final_candidates_phase1.parquet
matches both source layers byte-for-byte where it should, and that nothing
got silently lost or scrambled by the merge.

Checks:
  A. Row count is exactly greenfield + reuse
  B. candidate_id unique across the union
  C. candidate_type splits match source row counts
  D. Every greenfield candidate_id from the input appears in the unified
  E. Every reuse site_id from the input appears in the unified
  F. Spot-check: 100 random greenfield rows have identical geometry and
     identical composite_score to their source-file counterparts
  G. Spot-check: 100 random reuse rows likewise
  H. Column union: every column from either input parquet exists in unified
  I. Reuse-only columns are NULL on greenfield rows
  J. Greenfield-only columns are NULL on reuse rows
  K. Both layers share the same CRS in the unified file
  L. recommended_action / composite_score distributions per layer match the
     standalone files

Run: python candidate_areas/reuse_node_scripts/StepFinal_verify_merge.py
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

GF_PATH  = Path('candidate_areas/outputs/candidates_final.parquet')
RN_PATH  = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')
UNI_PATH = Path('candidate_areas/outputs/final_candidates_phase1.parquet')


def main():
    print('Loading three layers ...')
    gf  = gpd.read_parquet(GF_PATH)
    rn  = gpd.read_parquet(RN_PATH)
    uni = gpd.read_parquet(UNI_PATH)
    print(f'  greenfield: {len(gf):>7,} rows, {len(gf.columns)} cols')
    print(f'  reuse:      {len(rn):>7,} rows, {len(rn.columns)} cols')
    print(f'  unified:    {len(uni):>7,} rows, {len(uni.columns)} cols\n')

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

    print('=== A. Row count integrity ===')
    chk(f'unified rows = greenfield + reuse ({len(gf)} + {len(rn)} = {len(gf)+len(rn)})',
        len(uni) == len(gf) + len(rn))

    print('\n=== B. ID uniqueness across the union ===')
    chk('candidate_id unique across unified', uni.candidate_id.is_unique)

    print('\n=== C. candidate_type splits ===')
    n_gf_uni = int((uni.candidate_type == 'greenfield').sum())
    n_rn_uni = int((uni.candidate_type == 'reuse_node').sum())
    chk(f'greenfield rows in unified = {len(gf)}', n_gf_uni == len(gf),
        f'got {n_gf_uni}')
    chk(f'reuse rows in unified = {len(rn)}', n_rn_uni == len(rn),
        f'got {n_rn_uni}')

    print('\n=== D. All greenfield candidate_ids present in unified ===')
    gf_uni_ids = set(uni[uni.candidate_type == 'greenfield'].candidate_id)
    gf_src_ids = set(gf.candidate_id)
    missing_gf = gf_src_ids - gf_uni_ids
    extra_gf   = gf_uni_ids - gf_src_ids
    chk(f'no greenfield IDs missing (lost: {len(missing_gf)})', len(missing_gf) == 0)
    chk(f'no extra greenfield IDs    (extra: {len(extra_gf)})',   len(extra_gf) == 0)

    print('\n=== E. All reuse site_ids present in unified ===')
    rn_uni_ids = set(uni[uni.candidate_type == 'reuse_node'].candidate_id)
    rn_src_ids = set(rn.candidate_id)
    missing_rn = rn_src_ids - rn_uni_ids
    extra_rn   = rn_uni_ids - rn_src_ids
    chk(f'no reuse IDs missing (lost: {len(missing_rn)})', len(missing_rn) == 0)
    chk(f'no extra reuse IDs    (extra: {len(extra_rn)})',   len(extra_rn) == 0)

    print('\n=== F. Spot-check 100 greenfield rows: geometry + composite preserved ===')
    sample_gf = gf.sample(100, random_state=42)
    uni_gf = uni[uni.candidate_type == 'greenfield'].set_index('candidate_id')
    bad_geom = 0
    bad_score = 0
    for _, src_row in sample_gf.iterrows():
        if src_row.candidate_id not in uni_gf.index:
            bad_geom += 1; bad_score += 1; continue
        uni_row = uni_gf.loc[src_row.candidate_id]
        if not src_row.geometry.equals_exact(uni_row.geometry, 0.001):
            bad_geom += 1
        if abs(src_row.composite_score - uni_row.composite_score) > 0.001:
            bad_score += 1
    chk(f'100/100 greenfield geometries match', bad_geom == 0, f'{bad_geom} mismatched')
    chk(f'100/100 greenfield composite_scores match', bad_score == 0, f'{bad_score} mismatched')

    print('\n=== G. Spot-check 100 reuse rows: geometry + composite preserved ===')
    sample_rn = rn.sample(100, random_state=42)
    uni_rn = uni[uni.candidate_type == 'reuse_node'].set_index('candidate_id')
    bad_geom_rn = 0
    bad_score_rn = 0
    for _, src_row in sample_rn.iterrows():
        if src_row.candidate_id not in uni_rn.index:
            bad_geom_rn += 1; bad_score_rn += 1; continue
        uni_row = uni_rn.loc[src_row.candidate_id]
        if not src_row.geometry.equals_exact(uni_row.geometry, 0.001):
            bad_geom_rn += 1
        if abs(src_row.composite_score - uni_row.composite_score) > 0.001:
            bad_score_rn += 1
    chk(f'100/100 reuse geometries match', bad_geom_rn == 0, f'{bad_geom_rn} mismatched')
    chk(f'100/100 reuse composite_scores match', bad_score_rn == 0, f'{bad_score_rn} mismatched')

    print('\n=== H. Column union completeness ===')
    gf_cols = set(gf.columns)
    rn_cols = set(rn.columns)
    uni_cols = set(uni.columns)
    missing_from_uni = (gf_cols | rn_cols) - uni_cols
    chk(f'unified has every input column (missing: {sorted(missing_from_uni)[:5]})',
        len(missing_from_uni) == 0)

    print('\n=== I. Reuse-only columns are NULL on greenfield rows ===')
    reuse_only_cols = list(rn_cols - gf_cols)
    print(f'  {len(reuse_only_cols)} reuse-only columns to check')
    gf_block = uni[uni.candidate_type == 'greenfield']
    nonnull_violations = []
    for col in reuse_only_cols:
        if col in gf_block.columns:
            n_nonnull = int(gf_block[col].notna().sum())
            if n_nonnull > 0:
                nonnull_violations.append(f'{col}={n_nonnull}')
    chk(f'all {len(reuse_only_cols)} reuse-only cols null on greenfield rows',
        len(nonnull_violations) == 0,
        f'violations: {nonnull_violations[:3]}')

    print('\n=== J. Greenfield-only columns are NULL on reuse rows ===')
    # Some columns from greenfield's input parquet are INTENTIONALLY populated
    # on reuse rows during merge (candidate_type + run-metadata columns that
    # reflect upstream dataset versions shared by both pipelines). Exclude
    # them from the "must be null" check.
    INTENTIONAL_SHARED = {
        'candidate_type',
        'run_id', 'run_date', 'snapshot_date',
        'scoring_model_version', 'exclusion_model_version',
        'padus_version', 'fema_nfhl_date', 'nwi_date',
        'transmission_dataset_version', 'queue_dataset_date',
        'dem_dataset_version',
        # Derivable from geometry, backfilled on reuse rows during merge
        'area_m2', 'centroid_lon', 'centroid_lat',
        'county_fips',  # spatial-joined on reuse rows during merge
    }
    gf_only_cols = list(gf_cols - rn_cols - INTENTIONAL_SHARED)
    print(f'  {len(gf_only_cols)} greenfield-only columns to check '
          f'(excluding {len(INTENTIONAL_SHARED)} intentionally shared)')
    rn_block = uni[uni.candidate_type == 'reuse_node']
    nonnull_violations_gf = []
    for col in gf_only_cols:
        if col in rn_block.columns:
            n_nonnull = int(rn_block[col].notna().sum())
            if n_nonnull > 0:
                nonnull_violations_gf.append(f'{col}={n_nonnull}')
    chk(f'all {len(gf_only_cols)} greenfield-only cols null on reuse rows',
        len(nonnull_violations_gf) == 0,
        f'violations: {nonnull_violations_gf[:3]}')

    print('\n=== K. CRS preserved ===')
    chk(f'unified CRS = EPSG:5070', uni.crs.to_epsg() == 5070)
    chk(f'greenfield CRS = EPSG:5070', gf.crs.to_epsg() == 5070)
    chk(f'reuse CRS = EPSG:5070', rn.crs.to_epsg() == 5070)

    print('\n=== L1. Column order preserved (greenfield slice) ===')
    # The greenfield rows in the unified file must list greenfield's original
    # columns in the SAME order as candidates_final.parquet. (Reuse-only
    # columns may be interleaved but greenfield columns themselves should
    # appear in their original relative order.)
    uni_cols = list(uni.columns)
    gf_input_cols = list(gf.columns)
    # Find the positions of greenfield columns in the unified column list and
    # verify they are in the same relative order.
    gf_positions = [uni_cols.index(c) for c in gf_input_cols if c in uni_cols]
    in_order = gf_positions == sorted(gf_positions)
    chk('greenfield columns preserved in original relative order in unified',
        in_order, 'reordering detected')

    print('\n=== L2. Byte-identical values for greenfield slice ===')
    # For 200 random greenfield rows, compare EVERY shared column value to the
    # source candidates_final.parquet. Mismatches mean the merge corrupted data.
    sample_ids = list(gf.sample(min(200, len(gf)), random_state=7).candidate_id)
    gf_src = gf.set_index('candidate_id')
    uni_gf_idx = uni[uni.candidate_type == 'greenfield'].set_index('candidate_id')
    bad_cols = {}
    for cid in sample_ids:
        if cid not in uni_gf_idx.index:
            continue
        src_row = gf_src.loc[cid]
        uni_row = uni_gf_idx.loc[cid]
        for col in gf_input_cols:
            if col in ('geometry', 'candidate_id'):
                continue  # geometry covered by spot-check F; candidate_id is the index
            sv, uv = src_row[col], uni_row[col]
            # List/array columns (top_reason_codes, missing_modules): compare as lists
            if hasattr(sv, '__len__') and not isinstance(sv, str):
                try:
                    if list(sv) != list(uv):
                        bad_cols[col] = bad_cols.get(col, 0) + 1
                except Exception:
                    bad_cols[col] = bad_cols.get(col, 0) + 1
                continue
            # Scalar NaN-aware compare
            try:
                sv_na = pd.isna(sv)
                uv_na = pd.isna(uv)
                if sv_na and uv_na:
                    continue
                if sv_na != uv_na:
                    bad_cols[col] = bad_cols.get(col, 0) + 1
                    continue
                if sv != uv:
                    bad_cols[col] = bad_cols.get(col, 0) + 1
            except Exception:
                bad_cols[col] = bad_cols.get(col, 0) + 1
    chk(f'200 sampled greenfield rows: all {len(gf_input_cols)-1} non-geom cols byte-identical',
        len(bad_cols) == 0, f'mismatches in: {dict(list(bad_cols.items())[:5])}')

    print('\n=== L3. Per-layer score distributions match source ===')
    gf_uni_med  = uni[uni.candidate_type == 'greenfield'].composite_score.median()
    gf_src_med  = gf.composite_score.median()
    rn_uni_med  = uni[uni.candidate_type == 'reuse_node'].composite_score.median()
    rn_src_med  = rn.composite_score.median()
    chk(f'greenfield median composite ({gf_uni_med:.2f}) == source ({gf_src_med:.2f})',
        abs(gf_uni_med - gf_src_med) < 0.01)
    chk(f'reuse median composite ({rn_uni_med:.2f}) == source ({rn_src_med:.2f})',
        abs(rn_uni_med - rn_src_med) < 0.01)

    # Also check recommended_action counts match
    print('\n  recommended_action by candidate_type:')
    print('  ' + uni.groupby(['candidate_type','recommended_action']).size().unstack(fill_value=0).to_string().replace('\n','\n  '))

    src_action_gf = gf.recommended_action.value_counts().to_dict()
    src_action_rn = rn.recommended_action.value_counts().to_dict()
    uni_action_gf = uni[uni.candidate_type == 'greenfield'].recommended_action.value_counts().to_dict()
    uni_action_rn = uni[uni.candidate_type == 'reuse_node'].recommended_action.value_counts().to_dict()
    chk('greenfield recommended_action distribution preserved', src_action_gf == uni_action_gf)
    chk('reuse recommended_action distribution preserved',      src_action_rn == uni_action_rn)

    print('\n' + '=' * 60)
    print('OVERALL:', 'ALL MERGE CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    print('=' * 60)


if __name__ == '__main__':
    main()
