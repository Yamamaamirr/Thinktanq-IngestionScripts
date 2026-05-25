"""
StepR3A -- Slope + acreage tier + size class per reuse node.

Mirror of Step1A_slope_acreage_size_class.py for reuse_nodes_clean.parquet.

Reads:
  candidate_areas/reuse_node_outputs/reuse_nodes_clean.parquet  (via helper)
  ingestion_scripts/usgs_slope/slope_per_reuse_node.parquet      (real slope)

Writes:
  candidate_areas/reuse_node_enrichment_outputs/stepR3a_slope_acreage_size.parquet

Slope sampling now comes from the reuse-node-specific USGS 3DEP ingest
(ingest_slope_per_reuse_node.py). If that file is missing, slope fields
fall back to 'unknown' so the downstream scoring engine continues to work.

Output columns identical to Step1A:
  slope_mean_pct, slope_max_pct,
  slope_tier (ideal/acceptable/penalized/unknown), slope_tier_score, slope_review_flag,
  acreage_tier (small/moderate/large/very_large/strategic_scale), acreage_tier_score,
  size_class (site/campus/region), oversized_flag

Run: python candidate_areas/reuse_node_scripts/StepR3A_slope_acreage_size_class.py
"""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _r3_helpers import load_reuse_nodes_as_candidates, out_path

SLOPE_PATH = Path('ingestion_scripts/usgs_slope/slope_per_reuse_node.parquet')
OUT_PATH   = out_path('stepR3a_slope_acreage_size.parquet')


def slope_tier(mean_pct, max_pct):
    """Identical to Step1A.slope_tier()."""
    if pd.isna(mean_pct) or pd.isna(max_pct):
        return 'unknown', None, False
    if mean_pct <= 5.0 and max_pct <= 5.0:
        return 'ideal', 100.0, False
    if mean_pct <= 8.0 and max_pct <= 8.0:
        return 'acceptable', 80.0, False
    return 'penalized', 50.0, True


def acreage_tier(acres):
    if acres < 100:  return 'small',           50.0
    if acres < 250:  return 'moderate',        70.0
    if acres < 500:  return 'large',           85.0
    if acres < 1000: return 'very_large',      95.0
    return                'strategic_scale',  100.0


def size_class(acres):
    if acres < 500:  return 'site',   False
    if acres < 5000: return 'campus', False
    return                'region',  True


def main():
    print('Loading reuse nodes ...')
    cands = load_reuse_nodes_as_candidates(crs_epsg=5070)
    print(f'  {len(cands):,} reuse nodes')

    df = cands[['candidate_id', 'area_acres']].copy()

    acr = df.area_acres.apply(acreage_tier)
    df['acreage_tier']       = [t[0] for t in acr]
    df['acreage_tier_score'] = [t[1] for t in acr]

    sc = df.area_acres.apply(size_class)
    df['size_class']     = [t[0] for t in sc]
    df['oversized_flag'] = [t[1] for t in sc]

    # Slope: join in real per-reuse-node sampling if available
    if SLOPE_PATH.exists():
        print(f'Loading slope: {SLOPE_PATH} ...')
        slope = pd.read_parquet(SLOPE_PATH)
        # slope file is keyed on site_id; our df.candidate_id == site_id via the helper
        slope = slope.rename(columns={'site_id': 'candidate_id'})
        print(f'  {len(slope):,} slope rows; {slope.slope_mean_pct.notna().sum():,} with data')
        df = df.merge(slope[['candidate_id','slope_mean_pct','slope_max_pct']],
                      on='candidate_id', how='left')
        tier_results = df.apply(
            lambda r: slope_tier(r.slope_mean_pct, r.slope_max_pct), axis=1
        )
        df['slope_tier']        = [t[0] for t in tier_results]
        df['slope_tier_score']  = [t[1] for t in tier_results]
        df['slope_review_flag'] = [t[2] for t in tier_results]
    else:
        print(f'WARN: {SLOPE_PATH} not found; slope fields will be null/unknown')
        df['slope_mean_pct']    = pd.NA
        df['slope_max_pct']     = pd.NA
        df['slope_tier']        = 'unknown'
        df['slope_tier_score']  = pd.NA
        df['slope_review_flag'] = False

    out = df[[
        'candidate_id',
        'slope_mean_pct', 'slope_max_pct',
        'slope_tier', 'slope_tier_score', 'slope_review_flag',
        'acreage_tier', 'acreage_tier_score',
        'size_class', 'oversized_flag',
    ]]

    out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.2f} MB, {len(out):,} rows)')

    print('\n=== Slope tier distribution ===')
    print(out.slope_tier.value_counts().to_string())
    n_slope = out.slope_mean_pct.notna().sum()
    print(f'  rows with real slope data: {n_slope:,} / {len(out):,} ({100*n_slope/len(out):.1f}%)')
    print('\n=== Acreage tier distribution ===')
    print(out.acreage_tier.value_counts().reindex(
        ['small','moderate','large','very_large','strategic_scale']).to_string())
    print('\n=== Size class distribution ===')
    print(out.size_class.value_counts().to_string())
    print(f'  oversized_flag True: {int(out.oversized_flag.sum()):,}')

    valid_slope_tiers = {'ideal','acceptable','penalized','unknown'}
    print('\n=== Checks ===')
    checks = {
        'Has all reuse nodes'              : len(out) == len(cands),
        'Unique candidate_ids'             : out.candidate_id.is_unique,
        'acreage_tier valid set'           : set(out.acreage_tier.unique()) <= {'small','moderate','large','very_large','strategic_scale'},
        'acreage_score valid set'          : set(out.acreage_tier_score.unique()) <= {50.0,70.0,85.0,95.0,100.0},
        'size_class in {site,campus,region}': set(out.size_class.unique()) <= {'site','campus','region'},
        'oversized iff region'             : (out.oversized_flag == (out.size_class == 'region')).all(),
        'slope_tier valid set'             : set(out.slope_tier.unique()) <= valid_slope_tiers,
        'slope_tier_score in valid set'    : set(out.slope_tier_score.dropna().unique()) <= {50.0, 80.0, 100.0},
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
