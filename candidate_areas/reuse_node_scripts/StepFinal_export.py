"""
StepFinal_export -- Export reuse-node and unified Phase 1 outputs in
                    formats Owen / analysts can consume directly.

Writes (under candidate_areas/reuse_node_outputs/exports/):
  reuse_nodes_enriched.gpkg              -- GeoPackage of all 6.6k reuse nodes
  reuse_nodes_enriched.fgb               -- FlatGeobuf (streaming-friendly)
  reuse_nodes_enriched.csv               -- CSV (no geometry) of all 6.6k
  reuse_nodes_actionable.gpkg            -- Filtered subset (~775 sites)
  reuse_nodes_actionable.fgb             -- Same subset as FGB
  reuse_nodes_actionable.csv             -- Same subset as CSV

And under candidate_areas/outputs/exports/:
  final_candidates_phase1.gpkg           -- Unified greenfield + reuse layer
  final_candidates_phase1.fgb            -- Unified as FlatGeobuf

Run: python candidate_areas/reuse_node_scripts/StepFinal_export.py
"""
from pathlib import Path
import json
import pandas as pd
import geopandas as gpd

REUSE_PATH   = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')
UNIFIED_PATH = Path('candidate_areas/outputs/final_candidates_phase1.parquet')
REUSE_EXP_DIR = Path('candidate_areas/reuse_node_outputs/exports')
UNI_EXP_DIR   = Path('candidate_areas/outputs/exports')


def _stringify_list_cols(df):
    """GeoPackage / CSV don't support list columns -- JSON-encode them."""
    df = df.copy()
    for col in df.columns:
        if col == 'geometry':
            continue
        sample = df[col].dropna().head(1)
        if len(sample) and isinstance(sample.iloc[0], (list, tuple)) or \
           (len(sample) and hasattr(sample.iloc[0], '__len__') and not isinstance(sample.iloc[0], str)):
            try:
                df[col] = df[col].apply(
                    lambda v: json.dumps(list(v)) if v is not None and hasattr(v, '__iter__') and not isinstance(v, str) else v
                )
            except Exception:
                df[col] = df[col].astype(str)
    return df


def main():
    REUSE_EXP_DIR.mkdir(parents=True, exist_ok=True)
    UNI_EXP_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Loading reuse nodes: {REUSE_PATH} ...')
    rn = gpd.read_parquet(REUSE_PATH)
    print(f'  {len(rn):,} rows, {len(rn.columns)} columns')

    rn_str = _stringify_list_cols(rn)

    # ---- Full GeoPackage ----
    fp_gpkg = REUSE_EXP_DIR / 'reuse_nodes_enriched.gpkg'
    print(f'\nWriting GeoPackage: {fp_gpkg} ...')
    rn_str.to_file(fp_gpkg, driver='GPKG', layer='reuse_nodes_enriched')
    print(f'  {fp_gpkg.stat().st_size/1e6:.1f} MB')

    # ---- Full FlatGeobuf ----
    fp_fgb = REUSE_EXP_DIR / 'reuse_nodes_enriched.fgb'
    print(f'\nWriting FlatGeobuf: {fp_fgb} ...')
    rn_str.to_file(fp_fgb, driver='FlatGeobuf')
    print(f'  {fp_fgb.stat().st_size/1e6:.1f} MB')

    # ---- Full CSV (no geometry, with WKT centroid for reference) ----
    fp_csv = REUSE_EXP_DIR / 'reuse_nodes_enriched.csv'
    print(f'\nWriting CSV: {fp_csv} ...')
    cents = rn.geometry.centroid.to_crs(4326)
    rn_csv = pd.DataFrame(rn_str.drop(columns=['geometry']))
    rn_csv['centroid_lon'] = cents.x.values
    rn_csv['centroid_lat'] = cents.y.values
    rn_csv.to_csv(fp_csv, index=False)
    print(f'  {fp_csv.stat().st_size/1e6:.1f} MB')

    # ---- Actionable subset ----
    # Per audit:
    #   Shortlist (33)                                           -- top picks
    #   Reuse Diligence on retired EIA/Nuclear (~211)            -- decom sites
    #   Parcel Pull on EPA brownfields (531)                     -- EPA candidates
    is_shortlist = rn.recommended_action == 'Shortlist'
    is_retired_eia_diligence = (
        (rn.recommended_action == 'Reuse Diligence')
        & rn.source.isin(['EIA-860', 'EIA-860-nuclear'])
    )
    is_epa_parcel_pull = (
        (rn.recommended_action == 'Parcel Pull')
        & (rn.source == 'EPA-RE-Powering')
    )
    actionable_mask = is_shortlist | is_retired_eia_diligence | is_epa_parcel_pull
    actionable = rn_str[actionable_mask].copy()
    print(f'\nActionable subset: {len(actionable):,} rows')
    print(f'  Shortlist:                       {int(is_shortlist.sum())}')
    print(f'  Retired EIA Reuse Diligence:     {int(is_retired_eia_diligence.sum())}')
    print(f'  EPA Parcel Pull:                 {int(is_epa_parcel_pull.sum())}')

    fp_act = REUSE_EXP_DIR / 'reuse_nodes_actionable.gpkg'
    print(f'Writing actionable GeoPackage: {fp_act} ...')
    actionable.to_file(fp_act, driver='GPKG', layer='reuse_nodes_actionable')
    print(f'  {fp_act.stat().st_size/1e6:.1f} MB')

    fp_act_fgb = REUSE_EXP_DIR / 'reuse_nodes_actionable.fgb'
    print(f'Writing actionable FlatGeobuf: {fp_act_fgb} ...')
    actionable.to_file(fp_act_fgb, driver='FlatGeobuf')
    print(f'  {fp_act_fgb.stat().st_size/1e6:.1f} MB')

    fp_act_csv = REUSE_EXP_DIR / 'reuse_nodes_actionable.csv'
    act_csv = pd.DataFrame(actionable.drop(columns=['geometry']))
    act_cents = rn.loc[actionable_mask, 'geometry'].centroid.to_crs(4326)
    act_csv['centroid_lon'] = act_cents.x.values
    act_csv['centroid_lat'] = act_cents.y.values
    act_csv.to_csv(fp_act_csv, index=False)
    print(f'  CSV: {fp_act_csv.stat().st_size/1e6:.1f} MB')

    # ---- Unified table GeoPackage (greenfield + reuse) ----
    if UNIFIED_PATH.exists():
        print(f'\nLoading unified: {UNIFIED_PATH} ...')
        uni = gpd.read_parquet(UNIFIED_PATH)
        print(f'  {len(uni):,} rows, {len(uni.columns)} columns')
        uni_str = _stringify_list_cols(uni)
        fp_uni = UNI_EXP_DIR / 'final_candidates_phase1.gpkg'
        print(f'Writing unified GeoPackage: {fp_uni} ...')
        uni_str.to_file(fp_uni, driver='GPKG', layer='final_candidates_phase1')
        print(f'  {fp_uni.stat().st_size/1e6:.1f} MB')

        fp_uni_fgb = UNI_EXP_DIR / 'final_candidates_phase1.fgb'
        print(f'Writing unified FlatGeobuf: {fp_uni_fgb} ...')
        uni_str.to_file(fp_uni_fgb, driver='FlatGeobuf')
        print(f'  {fp_uni_fgb.stat().st_size/1e6:.1f} MB')
    else:
        print(f'\nWARN: {UNIFIED_PATH} not found; skipping unified export. '
              f'Run StepFinal_merge_greenfield_and_reuse.py first.')

    print('\nDone.')


if __name__ == '__main__':
    main()
