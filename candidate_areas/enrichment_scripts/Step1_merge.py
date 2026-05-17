"""
Step 1 — Merge all enrichment substeps into candidate_areas_enriched.parquet.

Reads:
  candidate_areas/outputs/candidate_areas.parquet  (95,269 base rows + 21 cols)
  candidate_areas/enrichment_outputs/step1{a..j}*.parquet (10 substep files)

Joins each on candidate_id and writes a unified enriched dataset. Pairs
file (step1i_utility_anchor_pairs.parquet) is NOT merged — it stays as a
companion table that the scoring engine joins on-demand.

Output:
  candidate_areas/outputs/candidate_areas_enriched.parquet

Verifies:
  - Row count preserved (95,269)
  - candidate_id still unique
  - Original columns unchanged (byte-identical to source)
  - All expected new columns present
  - Per-column null rate (most should be ~0; some may have known nulls)

Run:
  python candidate_areas/enrichment_scripts/Step1_merge.py
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd

CAND_PATH = Path('candidate_areas/outputs/candidate_areas.parquet')
ENRICH_DIR = Path('candidate_areas/enrichment_outputs')
OUT_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')

# Order matters only for log readability; merging is by candidate_id key.
SUBSTEPS = [
    ('1A', 'step1a_slope_acreage_size.parquet'),
    ('1B', 'step1b_seismic.parquet'),
    ('1C', 'step1c_drought.parquet'),
    ('1D', 'step1d_adjacency.parquet'),
    ('1E', 'step1e_transmission.parquet'),
    ('1F', 'step1f_pipelines.parquet'),
    ('1G', 'step1g_rail.parquet'),
    ('1H', 'step1h_water.parquet'),
    ('1I', 'step1i_utility_summary.parquet'),
    ('1J', 'step1j_acreage_breakdown.parquet'),
]


def main():
    print(f'Loading base candidates: {CAND_PATH} ...')
    cands = gpd.read_parquet(CAND_PATH)
    base_cols = list(cands.columns)
    print(f'  {len(cands):,} rows, {len(base_cols)} base columns')
    print(f'  base columns: {base_cols}')

    enriched = cands.copy()

    for label, fname in SUBSTEPS:
        fp = ENRICH_DIR / fname
        if not fp.exists():
            raise FileNotFoundError(f'Missing substep {label}: {fp}')
        sub = pd.read_parquet(fp)
        # Strip geometry if any (it's only on base; substeps are tabular)
        if 'geometry' in sub.columns:
            sub = sub.drop(columns=['geometry'])
        new_cols = [c for c in sub.columns if c != 'candidate_id']
        # Detect any collision with existing columns (would be a bug)
        collisions = [c for c in new_cols if c in enriched.columns]
        if collisions:
            raise ValueError(f'Substep {label} has column collision with existing: {collisions}')
        print(f'  {label}: {fname:<40} +{len(new_cols):>3} cols ({len(sub):,} rows)')
        enriched = enriched.merge(sub, on='candidate_id', how='left')
        assert len(enriched) == len(cands), \
            f'Substep {label} broke row count: {len(enriched)} != {len(cands)}'

    print(f'\nFinal: {len(enriched):,} rows, {len(enriched.columns)} columns '
          f'(+{len(enriched.columns) - len(base_cols)} added)')

    # ---- Save ----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f'\nSaved: {OUT_PATH} ({size_mb:.1f} MB)')

    # ---- Verification ----
    print('\n=== Integrity ===')
    print(f'  Row count preserved:       {len(enriched) == len(cands)}')
    print(f'  Unique candidate_id:       {enriched.candidate_id.is_unique}')
    print(f'  CRS unchanged:             {enriched.crs == cands.crs}')
    # Byte-check on original columns
    original_match = all(
        (enriched[c].fillna("__NA__") == cands[c].fillna("__NA__")).all()
        if c != 'geometry' else
        enriched.geometry.geom_equals_exact(cands.geometry, 0.001).all()
        for c in base_cols
    )
    print(f'  Original cols unchanged:   {original_match}')

    # ---- Per-column null rate ----
    print('\n=== Null rate per added column (top 20) ===')
    added = [c for c in enriched.columns if c not in base_cols]
    nulls = enriched[added].isna().sum().sort_values(ascending=False)
    if len(nulls) > 0:
        for c, n in nulls.head(20).items():
            pct = 100 * n / len(enriched)
            print(f'  {c:<40} {n:>6,} nulls ({pct:5.1f}%)')

    # ---- Column groups ----
    groups = {
        '1A slope/acreage/size': ['slope_mean_pct','slope_max_pct','slope_tier','slope_tier_score','slope_review_flag','acreage_tier','acreage_tier_score','size_class','oversized_flag'],
        '1B seismic'            : ['seismic_hazard_pga','seismic_hazard_tier','seismic_polygon_pga_range','seismic_valley_response'],
        '1C drought'            : ['drought_level','drought_label'],
        '1D adjacency'          : ['nearest_padus_distance_m','near_padus_flag','nearest_wetland_distance_m','near_wetland_flag','nearest_floodway_distance_m','adjacent_floodway_flag','nearest_fema_ae_distance_m','fema_ae_overlap_flag','fema_ae_adjacent_flag','nearest_radar_distance_m','radar_distance_miles','radar_review_flag'],
        '1E transmission'       : ['nearest_500kv_distance_m','nearest_345kv_distance_m','nearest_230kv_distance_m','crosses_500kv_flag','crosses_345kv_flag','crosses_230kv_flag'],
        '1F pipelines'          : ['nearest_pipeline_distance_m','nearest_pipeline_operator_tier','nearest_pipeline_est_diameter_in','nearest_tier1_pipeline_distance_m','nearest_other_pipeline_distance_m','pipeline_diameter_estimated'],
        '1G rail'               : ['nearest_class1_rail_distance_m','nearest_rail_is_stracnet','nearest_rail_n_tracks'],
        '1H water'              : ['within_water_service_area','nearest_water_service_distance_m','nearest_water_service_pop_served'],
        '1I utility'            : ['num_anchors_in_range','zone_fallback_used','primary_anchor_name','primary_anchor_distance_m','primary_anchor_voltage_kv','primary_anchor_queue_mw_tier','primary_anchor_queue_status_score','primary_anchor_activation_band','primary_anchor_match_confidence','primary_anchor_distance_band','node_activity_score','queue_mw_nearby','ia_executed_nearby','recent_withdrawals_nearby','allocation_or_competitive_heat_flag'],
        '1J acreage'            : ['original_area_acres','net_buildable_area_acres','buildable_area_ratio'],
    }
    print('\n=== Column-group presence ===')
    for grp, cols in groups.items():
        missing = [c for c in cols if c not in enriched.columns]
        status = 'OK' if not missing else f'MISSING: {missing}'
        print(f'  {grp:<25}  {len(cols)} cols  {status}')


if __name__ == '__main__':
    main()
