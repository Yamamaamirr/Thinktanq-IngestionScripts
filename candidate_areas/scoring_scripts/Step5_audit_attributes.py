"""
Step 5 — Per-attribute audit. For every column in candidates_final.parquet,
report null/zero %, distribution, and flag anything that doesn't match expectations.

Expected nulls:
  - Phase 2 placeholders (parcel_count, owner_count, assessed_value_*, etc.) — should be 100% null
  - Conditional fields (e.g., primary_anchor_* are null if no anchor within 50 km)

Suspect:
  - Required fields with null > 0%
  - Numeric fields where everything is 0 (default fallthrough)
  - Categorical fields with only 1 unique value (not used)
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

PARQUET = Path('candidate_areas/outputs/candidates_final.parquet')

# Expected-null columns (Phase 2 placeholders, per Owen May 11)
EXPECTED_NULL = {
    # Phase 2 parcel module
    'parcel_count','owner_count','largest_owner_acres','largest_owner_pct_of_candidate',
    'assessed_value_total','assessed_value_per_acre','last_sale_date','last_sale_price',
    'land_use_code','zoning_code','road_frontage_flag','legal_access_flag',
    'site_control_score','economic_proxy_score',
    # Phase 2 utility-service module
    'serving_utility','utility_territory_known','nearest_load_serving_node',
    'utility_service_feasibility_score','utility_review_required',
    # Phase 2 comms / water / jurisdiction
    'communications_route_distance','communications_provider_count','communications_access_score',
    'water_capacity_known','water_capacity_review_required',
    'jurisdiction_review_required','local_policy_notes',
    'manual_imagery_review_status','manual_imagery_review_notes',
    # Stage 5 placeholder (PDF rule)
    'route_complexity_score','route_complexity_notes',
    # Stage 6 NLCD placeholder (PDF rule 11 — kept blank, CDL is primary)
    'nlcd_class','nlcd_label','landcover_confidence_score',
}

GROUPS = {
    'Identity & Location': [
        'candidate_id','snapshot_date','candidate_type','state','county_name','county_fips',
        'centroid_lon','centroid_lat',
    ],
    'Land cover (CDL primary, NLCD placeholder)': [
        'cdl_group','cdl_group_label','nlcd_class','nlcd_label','landcover_confidence_score',
    ],
    'Building footprints (MS USBF)': [
        'building_footprint_pct','building_adjacency_m',
    ],
    'Slope (3DEP)': [
        'slope_mean_pct','slope_max_pct','slope_tier','slope_tier_score','slope_review_flag',
    ],
    'Acreage / size': [
        'area_m2','pixel_count','area_acres','original_area_acres','net_buildable_area_acres',
        'buildable_area_ratio','acreage_tier','acreage_tier_score','size_class','oversized_flag',
    ],
    'Seismic (NSHM23)': [
        'seismic_hazard_pga','seismic_hazard_tier','seismic_polygon_pga_range','seismic_valley_response',
    ],
    'Drought (USDM)': [
        'drought_level','drought_label',
    ],
    'Exclusion adjacency': [
        'nearest_padus_distance_m','nearest_wetland_distance_m','nearest_floodway_distance_m',
        'nearest_fema_ae_distance_m','nearest_radar_distance_m',
        'near_padus_flag','near_wetland_flag','adjacent_floodway_flag',
        'fema_ae_overlap_flag','fema_ae_adjacent_flag',
        'radar_distance_miles','radar_review_flag',
    ],
    'Transmission (HIFLD)': [
        'nearest_500kv_distance_m','nearest_345kv_distance_m','nearest_230kv_distance_m',
        'crosses_500kv_flag','crosses_345kv_flag','crosses_230kv_flag',
    ],
    'Pipelines (PHMSA)': [
        'nearest_pipeline_distance_m','nearest_pipeline_operator_tier','nearest_pipeline_est_diameter_in',
        'nearest_tier1_pipeline_distance_m','nearest_other_pipeline_distance_m','pipeline_diameter_estimated',
    ],
    'Rail (Class 1 / STRACNET)': [
        'nearest_class1_rail_distance_m','nearest_rail_is_stracnet','nearest_rail_n_tracks',
    ],
    'Water service': [
        'within_water_service_area','nearest_water_service_distance_m','nearest_water_service_pop_served',
    ],
    'Utility anchors (queue + IA)': [
        'num_anchors_in_range','zone_fallback_used','primary_anchor_name','primary_anchor_distance_m',
        'primary_anchor_voltage_kv','primary_anchor_queue_mw_tier','primary_anchor_queue_status_score',
        'primary_anchor_activation_band','primary_anchor_match_confidence','primary_anchor_distance_band',
        'node_activity_score','queue_mw_nearby','ia_executed_nearby','recent_withdrawals_nearby',
        'allocation_or_competitive_heat_flag',
    ],
    'Status fields': [
        'candidate_status','candidate_status_reason','route_complexity_score','route_complexity_notes',
    ],
    'Sub-scores': [
        'utility_score','utility_module_status','land_cover_score','constraint_score',
        'buildability_score','buildability_review_required',
        'transmission_score','pipeline_score','rail_score','water_score','supporting_infra_score',
        'seismic_score','drought_score','radar_score','padus_score','wetland_score','floodway_score',
        'dev_risk_score',
    ],
    'Composite & meta': [
        'composite_score','score_v1_observable','data_coverage_pct','missing_modules','confidence',
        'recommended_action','actionability_status','top_reason_codes',
    ],
    'Phase 2 parcel placeholders': [
        'parcel_count','owner_count','largest_owner_acres','largest_owner_pct_of_candidate',
        'assessed_value_total','assessed_value_per_acre','last_sale_date','last_sale_price',
        'land_use_code','zoning_code','road_frontage_flag','legal_access_flag',
        'site_control_score','economic_proxy_score','parcel_owner_module_status',
    ],
    'Phase 2 utility/comm/water/jurisdiction placeholders': [
        'serving_utility','utility_territory_known','nearest_load_serving_node',
        'utility_service_feasibility_score','utility_review_required',
        'communications_route_distance','communications_provider_count','communications_access_score',
        'water_capacity_known','water_capacity_review_required',
        'jurisdiction_review_required','local_policy_notes',
        'manual_imagery_review_status','manual_imagery_review_notes',
    ],
    'Audit trail': [
        'run_id','run_date','scoring_model_version','exclusion_model_version',
        'cdl_year','padus_version','fema_nfhl_date','nwi_date',
        'transmission_dataset_version','queue_dataset_date','dem_dataset_version',
    ],
}


def summarize_col(name, s, n_rows):
    """Return a one-liner dict for a column."""
    n_null = int(s.isna().sum())
    pct_null = 100 * n_null / n_rows
    n_zero = int((s == 0).sum()) if pd.api.types.is_numeric_dtype(s) else 0
    pct_zero = 100 * n_zero / n_rows
    dtype = str(s.dtype)

    # Flag logic
    flags = []
    if name in EXPECTED_NULL:
        if pct_null < 99.99:
            flags.append(f'UNEXPECTED: should be 100% null, but only {pct_null:.1f}% null')
    else:
        if pct_null > 50:
            flags.append(f'WARN: {pct_null:.1f}% null')
        if pct_null == 100:
            flags.append('CRITICAL: 100% null')

    # Distribution string
    if s.notna().sum() == 0:
        dist = '<all null>'
    elif pd.api.types.is_numeric_dtype(s):
        vals = s.dropna()
        if vals.dtype == bool:
            dist = f'true={int(vals.sum())}, false={int((~vals).sum())}'
        else:
            dist = (f'min={vals.min():.3g}, p50={vals.median():.3g}, '
                    f'mean={vals.mean():.3g}, max={vals.max():.3g}, '
                    f'zero={pct_zero:.1f}%')
            # Check if everything is the same value
            if vals.nunique() == 1:
                flags.append(f'CRITICAL: only one value ({vals.iloc[0]})')
    else:
        vals = s.dropna()
        # Handle list/ndarray cells (e.g. missing_modules, top_reason_codes)
        sample = vals.iloc[0] if len(vals) else None
        if isinstance(sample, (list, np.ndarray, tuple)):
            lens = vals.apply(lambda x: len(x) if hasattr(x, '__len__') else 0)
            dist = (f'list-type, len min={lens.min()}, median={int(lens.median())}, '
                    f'max={lens.max()}, all-empty={int((lens==0).sum())}')
        else:
            try:
                n_unique = vals.nunique()
                if n_unique <= 6:
                    counts = vals.value_counts().head(6).to_dict()
                    counts_str = ', '.join(f'{k}={v}' for k, v in counts.items())
                    dist = f'{n_unique} unique: {counts_str}'
                else:
                    top3 = vals.value_counts().head(3).to_dict()
                    dist = f'{n_unique} unique, top3: ' + ', '.join(f'{k}={v}' for k, v in top3.items())
                if n_unique == 1:
                    flags.append(f'CRITICAL: only one value ({vals.iloc[0]})')
            except (TypeError, ValueError) as e:
                dist = f'<un-summarizable: {type(sample).__name__}>'

    return {
        'name': name,
        'dtype': dtype,
        'pct_null': pct_null,
        'pct_zero': pct_zero,
        'dist': dist,
        'flags': flags,
    }


def main():
    print(f'Loading {PARQUET} ...')
    g = gpd.read_parquet(PARQUET)
    n = len(g)
    print(f'  {n:,} rows x {len(g.columns)} columns\n')

    # Process each group
    findings = []
    for grp_name, cols in GROUPS.items():
        print(f'\n{"=" * 80}\n{grp_name}\n{"=" * 80}')
        for col in cols:
            if col not in g.columns:
                print(f'  [{col:<42}] MISSING from dataframe')
                findings.append({'name': col, 'flags': ['MISSING column']})
                continue
            r = summarize_col(col, g[col], n)
            flag_str = '  '.join(f'[{f}]' for f in r['flags']) if r['flags'] else ''
            print(f'  {col:<42} null={r["pct_null"]:>5.1f}%  {r["dist"][:90]}')
            if r['flags']:
                for f in r['flags']:
                    print(f'    >> {f}')
            findings.append(r)

    # Final summary
    n_critical = sum(any('CRITICAL' in f for f in d.get('flags', [])) for d in findings)
    n_warn = sum(any('WARN' in f for f in d.get('flags', [])) for d in findings)
    n_unexpected = sum(any('UNEXPECTED' in f for f in d.get('flags', [])) for d in findings)
    n_missing = sum(any('MISSING' in f for f in d.get('flags', [])) for d in findings)

    print(f'\n\n{"#" * 80}\n SUMMARY\n{"#" * 80}')
    print(f'  Columns audited: {len(findings)}')
    print(f'  CRITICAL flags:   {n_critical}')
    print(f'  WARN flags:       {n_warn}')
    print(f'  UNEXPECTED nulls: {n_unexpected}')
    print(f'  MISSING columns:  {n_missing}')

    # Print just the critical/warn flagged columns
    print(f'\n  FLAGGED COLUMNS:')
    for d in findings:
        if d.get('flags'):
            print(f'    {d["name"]:<42}  ' + '  '.join(d['flags']))


if __name__ == '__main__':
    main()
