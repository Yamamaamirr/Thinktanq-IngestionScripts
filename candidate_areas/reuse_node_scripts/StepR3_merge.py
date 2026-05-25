"""
StepR3 -- Merge all R3 enrichment substeps into reuse_nodes_enriched.parquet.

Mirror of candidate_areas/enrichment_scripts/Step1_merge.py.

Reads:
  candidate_areas/reuse_node_outputs/reuse_nodes_clean.parquet  (base 6,631 rows)
  candidate_areas/reuse_node_enrichment_outputs/stepR3{a..j}*.parquet

Writes:
  candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet

Join key: candidate_id (which is reuse_nodes_clean's site_id renamed via
_r3_helpers.load_reuse_nodes_as_candidates -- see helper docstring).

Verifies:
  - Row count preserved
  - candidate_id (= site_id) still unique
  - All original columns preserved
  - Expected enrichment column groups all present
  - Per-column null rates

Run: python candidate_areas/reuse_node_scripts/StepR3_merge.py
"""
from pathlib import Path
import sys
import pandas as pd
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent))
from _r3_helpers import load_reuse_nodes_as_candidates

ENRICH_DIR = Path('candidate_areas/reuse_node_enrichment_outputs')
OUT_PATH   = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')

SUBSTEPS = [
    ('R3A', 'stepR3a_slope_acreage_size.parquet'),
    ('R3B', 'stepR3b_seismic.parquet'),
    ('R3C', 'stepR3c_drought.parquet'),
    ('R3D', 'stepR3d_adjacency.parquet'),
    ('R3E', 'stepR3e_transmission.parquet'),
    ('R3F', 'stepR3f_pipelines.parquet'),
    ('R3G', 'stepR3g_rail.parquet'),
    ('R3H', 'stepR3h_water.parquet'),
    ('R3I', 'stepR3i_utility_summary.parquet'),
    ('R3J', 'stepR3j_acreage_breakdown.parquet'),
]


def main():
    print('Loading base reuse nodes (as candidates) ...')
    base = load_reuse_nodes_as_candidates(crs_epsg=5070)
    base_cols = list(base.columns)
    print(f'  {len(base):,} rows, {len(base_cols)} base columns')

    enriched = base.copy()

    for label, fname in SUBSTEPS:
        fp = ENRICH_DIR / fname
        if not fp.exists():
            raise FileNotFoundError(f'Missing substep {label}: {fp}')
        sub = pd.read_parquet(fp)
        if 'geometry' in sub.columns:
            sub = sub.drop(columns=['geometry'])
        new_cols = [c for c in sub.columns if c != 'candidate_id']
        collisions = [c for c in new_cols if c in enriched.columns]
        if collisions:
            raise ValueError(f'Substep {label} column collision: {collisions}')
        print(f'  {label}: {fname:<42} +{len(new_cols):>3} cols ({len(sub):,} rows)')
        enriched = enriched.merge(sub, on='candidate_id', how='left')
        assert len(enriched) == len(base), \
            f'Substep {label} broke row count: {len(enriched)} != {len(base)}'

    print(f'\nFinal: {len(enriched):,} rows, {len(enriched.columns)} columns '
          f'(+{len(enriched.columns) - len(base_cols)} added)')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.1f} MB)')

    print('\n=== Integrity ===')
    print(f'  Row count preserved:   {len(enriched) == len(base)}')
    print(f'  Unique candidate_id:   {enriched.candidate_id.is_unique}')
    print(f'  CRS unchanged:         {enriched.crs == base.crs}')

    print('\n=== Null rate per added column (top 20) ===')
    added = [c for c in enriched.columns if c not in base_cols]
    nulls = enriched[added].isna().sum().sort_values(ascending=False)
    for c, n in nulls.head(20).items():
        pct = 100 * n / len(enriched)
        print(f'  {c:<42} {n:>6,} nulls ({pct:5.1f}%)')

    # Expected column groups -- mirror of Step1_merge.py
    groups = {
        'R3A slope/acreage/size': ['slope_mean_pct','slope_max_pct','slope_tier','slope_tier_score','slope_review_flag','acreage_tier','acreage_tier_score','size_class','oversized_flag'],
        'R3B seismic'           : ['seismic_hazard_pga','seismic_hazard_tier','seismic_polygon_pga_range','seismic_valley_response'],
        'R3C drought'           : ['drought_level','drought_label'],
        'R3D adjacency'         : ['nearest_padus_distance_m','near_padus_flag','nearest_wetland_distance_m','near_wetland_flag','nearest_floodway_distance_m','adjacent_floodway_flag','nearest_fema_ae_distance_m','fema_ae_overlap_flag','fema_ae_adjacent_flag','nearest_radar_distance_m','radar_distance_miles','radar_review_flag'],
        'R3E transmission'      : ['nearest_500kv_distance_m','nearest_345kv_distance_m','nearest_230kv_distance_m','crosses_500kv_flag','crosses_345kv_flag','crosses_230kv_flag'],
        'R3F pipelines'         : ['nearest_pipeline_distance_m','nearest_pipeline_operator_tier','nearest_pipeline_est_diameter_in','nearest_tier1_pipeline_distance_m','nearest_other_pipeline_distance_m','pipeline_diameter_estimated'],
        'R3G rail'              : ['nearest_class1_rail_distance_m','nearest_rail_is_stracnet','nearest_rail_n_tracks'],
        'R3H water'             : ['within_water_service_area','nearest_water_service_distance_m','nearest_water_service_pop_served'],
        'R3I utility'           : ['num_anchors_in_range','zone_fallback_used','primary_anchor_name','primary_anchor_distance_m','primary_anchor_voltage_kv','primary_anchor_queue_mw_tier','primary_anchor_queue_status_score','primary_anchor_activation_band','primary_anchor_match_confidence','primary_anchor_distance_band','node_activity_score','queue_mw_nearby','ia_executed_nearby','recent_withdrawals_nearby','allocation_or_competitive_heat_flag'],
        'R3J acreage'           : ['original_area_acres','net_buildable_area_acres','buildable_area_ratio'],
    }
    print('\n=== Column-group presence ===')
    for grp, cols in groups.items():
        missing = [c for c in cols if c not in enriched.columns]
        status = 'OK' if not missing else f'MISSING: {missing}'
        print(f'  {grp:<25}  {len(cols):>2} cols  {status}')


if __name__ == '__main__':
    main()
