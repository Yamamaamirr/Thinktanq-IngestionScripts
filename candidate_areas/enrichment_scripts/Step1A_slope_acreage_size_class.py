"""
Step 1A — Simple per-candidate transforms (no spatial joins).

Reads:
  candidate_areas/outputs/candidate_areas.parquet
  ingestion_scripts/usgs_slope/slope_per_candidate.parquet

Computes and writes:
  candidate_areas/enrichment_outputs/step1a_slope_acreage_size.parquet

Columns added per candidate:
  slope_mean_pct      (from slope ingest, copied for downstream convenience)
  slope_max_pct
  slope_tier          ideal / acceptable / penalized   (per scoring_design.md 3.4)
  slope_tier_score    100 / 80 / 50
  slope_review_flag   True when slope mean or max is in the 8-15% band

  acreage_tier        small / moderate / large / very_large / strategic_scale  (Owen Rec #14)
  acreage_tier_score  50 / 70 / 85 / 95 / 100

  size_class          site / campus / region (Yamama → Owen, May 14 oversize discussion)
                      site:   < 500 ac     (real site-scale, the bulk of the dataset)
                      campus: 500-5,000 ac
                      region: > 5,000 ac   (aggregate land, needs subdivision in Phase 2)
                      oversized_flag set True when size_class == 'region'

Run:
  python candidate_areas/enrichment_scripts/Step1A_slope_acreage_size_class.py
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

CAND_PATH  = Path('candidate_areas/outputs/candidate_areas.parquet')
SLOPE_PATH = Path('ingestion_scripts/usgs_slope/slope_per_candidate.parquet')
OUT_PATH   = Path('candidate_areas/enrichment_outputs/step1a_slope_acreage_size.parquet')


def slope_tier(mean_pct, max_pct):
    """
    Per scoring_design.md 3.4. Anything >15% folds into 'penalized' tier
    because Step5e simplification left some leftover steep edge cases in
    the candidate set (the >15% hard exclusion used a simplified slope
    polygon mask, while our new ingest measures actual slope per pixel).
    """
    if pd.isna(mean_pct) or pd.isna(max_pct):
        return 'unknown', None, False

    if mean_pct <= 5.0 and max_pct <= 5.0:
        return 'ideal', 100.0, False
    if mean_pct <= 8.0 and max_pct <= 8.0:
        return 'acceptable', 80.0, False
    # 8% < mean or max ≤ 15%  → penalized + review
    # also catches the >15% edge cases (folded into worst tier we score)
    return 'penalized', 50.0, True


def acreage_tier(acres):
    if acres < 100:    return 'small',           50.0
    if acres < 250:    return 'moderate',        70.0
    if acres < 500:    return 'large',           85.0
    if acres < 1000:   return 'very_large',      95.0
    return                  'strategic_scale',   100.0


def size_class(acres):
    if acres < 500:    return 'site',    False
    if acres < 5000:   return 'campus',  False
    return                  'region',    True


def main():
    print(f'Loading candidates: {CAND_PATH} ...')
    cands = gpd.read_parquet(CAND_PATH)
    print(f'  {len(cands):,} candidates')

    print(f'Loading slope: {SLOPE_PATH} ...')
    slope = pd.read_parquet(SLOPE_PATH)
    print(f'  {len(slope):,} slope rows')
    print(f'  Nulls in slope_mean_pct: {slope.slope_mean_pct.isna().sum()}')

    # Join
    print('Joining slope into candidates ...')
    df = cands[['candidate_id', 'area_acres']].merge(
        slope[['candidate_id', 'slope_mean_pct', 'slope_max_pct']],
        on='candidate_id', how='left'
    )
    print(f'  joined: {len(df):,}')

    # Compute slope tier
    print('Computing slope tier ...')
    tier_results = df.apply(lambda r: slope_tier(r.slope_mean_pct, r.slope_max_pct), axis=1)
    df['slope_tier']        = [t[0] for t in tier_results]
    df['slope_tier_score']  = [t[1] for t in tier_results]
    df['slope_review_flag'] = [t[2] for t in tier_results]

    # Compute acreage tier
    print('Computing acreage tier ...')
    acr_results = df.area_acres.apply(acreage_tier)
    df['acreage_tier']        = [t[0] for t in acr_results]
    df['acreage_tier_score']  = [t[1] for t in acr_results]

    # Compute size_class
    print('Computing size_class ...')
    sc_results = df.area_acres.apply(size_class)
    df['size_class']     = [t[0] for t in sc_results]
    df['oversized_flag'] = [t[1] for t in sc_results]

    # Drop helper columns; keep only what we produced + the join key
    out = df[[
        'candidate_id',
        'slope_mean_pct', 'slope_max_pct',
        'slope_tier', 'slope_tier_score', 'slope_review_flag',
        'acreage_tier', 'acreage_tier_score',
        'size_class', 'oversized_flag',
    ]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f'\nSaved: {OUT_PATH} ({size_mb:.2f} MB, {len(out):,} rows)')

    # ---- Distribution summaries (verification) ----
    print('\n=== Slope tier distribution ===')
    print(out.slope_tier.value_counts().to_string())
    print(f'  review_flag True: {out.slope_review_flag.sum():,}')

    print('\n=== Acreage tier distribution ===')
    print(out.acreage_tier.value_counts().reindex(
        ['small','moderate','large','very_large','strategic_scale']).to_string())

    print('\n=== Size class distribution ===')
    print(out.size_class.value_counts().to_string())
    print(f'  oversized_flag True: {out.oversized_flag.sum():,}')

    # ---- Checks ----
    print('\n=== Checks ===')
    checks = {
        'Has all candidates'                 : len(out) == len(cands),
        'Unique candidate_ids'               : out.candidate_id.is_unique,
        'slope_tier never null'              : out.slope_tier.notna().all(),
        'slope_tier_score in {50,80,100,nan}': set(out.slope_tier_score.dropna().unique()) <= {50.0, 80.0, 100.0},
        'acreage_tier in 5 valid values'     : set(out.acreage_tier.unique()) <= {'small','moderate','large','very_large','strategic_scale'},
        'acreage_score in {50,70,85,95,100}' : set(out.acreage_tier_score.unique()) <= {50.0, 70.0, 85.0, 95.0, 100.0},
        'size_class in {site,campus,region}' : set(out.size_class.unique()) == {'site', 'campus', 'region'},
        'oversized matches region size_class': (out.oversized_flag == (out.size_class == 'region')).all(),
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False

    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
