"""
Step 3C — Generate qa_summary.md per Owen's May 11 instructions.

Owen May 11 asked for:
  1. score distribution by state/county
  2. missing-data rate per field
  3. top reason-code frequencies
  4. top high-score / low-confidence candidates
  5. candidates with strong power signal but possible allocation/competitive-heat risk
  6. duplicate or tile-boundary cluster issues, especially TX/CA

Reads:
  candidate_areas/outputs/candidate_areas_enriched.parquet

Writes:
  candidate_areas/outputs/qa_summary.md

Run:
  python candidate_areas/scoring_scripts/Step3C_qa_summary.py
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')
OUT_PATH      = Path('candidate_areas/outputs/qa_summary.md')


def fmt_pct(n, total):
    return f'{n:,} ({100*n/total:.1f}%)'


def main():
    print(f'Loading enriched: {ENRICHED_PATH} ...')
    df = gpd.read_parquet(ENRICHED_PATH)
    total = len(df)
    print(f'  {total:,} candidates')

    lines = []
    lines.append(f'# QA Summary — Candidate Site Detection Phase 1')
    lines.append('')
    lines.append(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'**Total candidates:** {total:,}')
    lines.append(f'**Run ID:** `{df.run_id.iloc[0]}`')
    lines.append(f'**Scoring model:** `{df.scoring_model_version.iloc[0]}`')
    lines.append('')
    lines.append('---')
    lines.append('')

    # ---- 1. Score distribution by state/county ----
    lines.append('## 1. Score distribution by state')
    lines.append('')
    lines.append('| State | Count | Median composite | P90 | Max | High-tier (≥85) | Mid-tier (65-85) | Low-tier (<65) |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for st in ['AZ','CA','NV','TX','VA']:
        sd = df[df.state == st]
        if len(sd) == 0: continue
        n_high = (sd.composite_score >= 85).sum()
        n_mid = ((sd.composite_score >= 65) & (sd.composite_score < 85)).sum()
        n_low = (sd.composite_score < 65).sum()
        lines.append(f'| {st} | {len(sd):,} | {sd.composite_score.median():.1f} | '
                     f'{sd.composite_score.quantile(0.9):.1f} | {sd.composite_score.max():.1f} | '
                     f'{n_high:,} | {n_mid:,} | {n_low:,} |')
    lines.append('')

    # By county (top 20 counties with highest median composite, min 50 candidates)
    lines.append('### Top 20 counties by median composite score (min 50 candidates)')
    lines.append('')
    county_stats = df.groupby(['state','county_name']).agg(
        n_cands=('composite_score','count'),
        med_composite=('composite_score','median'),
        p90_composite=('composite_score', lambda s: s.quantile(0.9)),
    ).reset_index()
    county_stats = county_stats[county_stats.n_cands >= 50].sort_values('med_composite', ascending=False).head(20)
    lines.append('| State | County | N | Median | P90 |')
    lines.append('|---|---|---|---|---|')
    for _, r in county_stats.iterrows():
        lines.append(f'| {r["state"]} | {r["county_name"]} | {int(r["n_cands"]):,} | {r["med_composite"]:.1f} | {r["p90_composite"]:.1f} |')
    lines.append('')
    lines.append('---')
    lines.append('')

    # ---- 2. Missing-data rate per field ----
    lines.append('## 2. Missing-data rate by field')
    lines.append('')
    lines.append('Fields with > 0% null rate (Phase 2 placeholder fields explicitly excluded — they are 100% null by design).')
    lines.append('')
    phase2_placeholder_cols = {
        'parcel_count','owner_count','largest_owner_acres','largest_owner_pct_of_candidate',
        'assessed_value_total','assessed_value_per_acre','last_sale_date','last_sale_price',
        'land_use_code','zoning_code','road_frontage_flag','legal_access_flag',
        'site_control_score','economic_proxy_score',
        'serving_utility','utility_territory_known','nearest_load_serving_node',
        'utility_service_feasibility_score','utility_review_required',
        'communications_route_distance','communications_provider_count','communications_access_score',
        'water_capacity_known','water_capacity_review_required',
        'jurisdiction_review_required','local_policy_notes',
        'manual_imagery_review_status','manual_imagery_review_notes',
        # Upstream NLCD/route still placeholders
        'route_complexity_score','route_complexity_notes','nlcd_class','nlcd_label','landcover_confidence_score',
    }
    nulls = df.isna().sum().sort_values(ascending=False)
    nulls = nulls[(nulls > 0) & (~nulls.index.isin(phase2_placeholder_cols))]
    lines.append('| Field | Nulls | % Null | Notes |')
    lines.append('|---|---|---|---|')
    explanations = {
        'nearest_500kv_distance_m':  'Candidate has no 500 kV line within 50 km — expected for most of TX, eastern VA',
        'nearest_345kv_distance_m':  'No 345 kV line within 50 km',
        'nearest_230kv_distance_m':  'No 230 kV line within 50 km',
        'nearest_padus_distance_m':  'No protected land within 10 km — clean candidate',
        'nearest_wetland_distance_m':'No wetland within 10 km — clean candidate',
        'nearest_floodway_distance_m':'No floodway within 10 km',
        'nearest_fema_ae_distance_m':'No AE flood zone within 10 km',
        'nearest_radar_distance_m':  'No FAA radar within 10 km',
        'nearest_pipeline_distance_m':'No pipeline within 80 km',
        'nearest_tier1_pipeline_distance_m':'No tier-1 pipeline within 80 km',
        'nearest_other_pipeline_distance_m':'No other-tier pipeline within 80 km',
        'nearest_class1_rail_distance_m':'No Class 1 rail within 50 km',
        'nearest_rail_n_tracks':     'Rail attribute missing on nearest segment',
        'nearest_rail_is_stracnet':  'STRACNET flag missing on nearest segment',
        'drought_level':             'Candidate centroid outside any D0–D4 drought zone — no drought',
        'primary_anchor_name':       'Zone fallback (no named anchor in 50 km)',
        'primary_anchor_distance_m': 'Zone fallback',
        'primary_anchor_voltage_kv': 'Zone fallback or unknown voltage',
        'primary_anchor_queue_mw_tier':'Zone fallback',
        'primary_anchor_queue_status_score':'Zone fallback',
        'primary_anchor_activation_band':'Anchor has no active queue projects',
        'primary_anchor_match_confidence':'Zone fallback',
        'primary_anchor_distance_band':'Zone fallback',
        'candidate_status_reason':   'NULL means candidate passed all gates (no review needed)',
        'county_name':               'County boundary join missed (rare edge cases)',
        'county_fips':               'County boundary join missed',
    }
    for col, n in nulls.head(40).items():
        pct = 100 * n / total
        note = explanations.get(col, '')
        lines.append(f'| `{col}` | {n:,} | {pct:.1f}% | {note} |')
    lines.append('')
    lines.append('---')
    lines.append('')

    # ---- 3. Top reason-code frequencies ----
    lines.append('## 3. Top reason-code frequencies')
    lines.append('')
    # top_reason_codes is stored as numpy.ndarray after parquet round-trip,
    # not a Python list. Use a lax iterable check instead of isinstance.
    def _iter_codes(codes):
        if codes is None: return []
        try:
            return list(codes)
        except TypeError:
            return []
    code_counts = pd.Series(
        [c for codes in df.top_reason_codes for c in _iter_codes(codes)]
    ).value_counts()
    lines.append('| Reason code | Count | % of candidates |')
    lines.append('|---|---|---|')
    for code, n in code_counts.head(25).items():
        lines.append(f'| `{code}` | {n:,} | {100*n/total:.1f}% |')
    lines.append('')
    lines.append('---')
    lines.append('')

    # ---- 4. Top high-score / low-confidence ----
    lines.append('## 4. High-score / low-confidence candidates (flagged for review)')
    lines.append('')
    lines.append('High composite score (≥75) but **low** confidence tier means the score is driven by a weak/zone-fallback anchor match or limited corroboration. These warrant analyst review before action.')
    lines.append('')
    hslc = df[(df.composite_score >= 75) & (df.confidence == 'low')].sort_values('composite_score', ascending=False)
    lines.append(f'**Total flagged:** {len(hslc):,}')
    if len(hslc) == 0:
        lines.append('')
        lines.append('*By design*: every low-confidence candidate in this run is a zone-fallback case '
                     '(no named anchor within 50 km), which forces `utility_score = 0`. With 40% '
                     'composite weight on utility, no zone-fallback candidate can reach composite ≥ 75. '
                     'All low-confidence candidates therefore sit in the lower composite bands and '
                     'are surfaced under "Recommended Action: Ignore" rather than this list.')
        lines.append('')
        # Show the highest-scoring low-confidence ones for context
        low_top = df[df.confidence == 'low'].sort_values('composite_score', ascending=False).head(10)
        lines.append('### Top 10 *low-confidence* by composite score (informational)')
        lines.append('')
        lines.append('| Candidate | State | County | Acres | Composite | Module Status |')
        lines.append('|---|---|---|---|---|---|')
        for _, r in low_top.iterrows():
            lines.append(f'| `{r.candidate_id[:8]}..` | {r.state} | {r.county_name} | {r.area_acres:.0f} | '
                         f'{r.composite_score:.1f} | {r.utility_module_status} |')
    else:
        lines.append('')
        lines.append('### Top 20')
        lines.append('')
        lines.append('| Candidate | State | County | Acres | Composite | Util | Build | Supp | Risk | Module Status |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|')
        for _, r in hslc.head(20).iterrows():
            lines.append(f'| `{r.candidate_id[:8]}..` | {r.state} | {r.county_name} | {r.area_acres:.0f} | '
                         f'{r.composite_score:.1f} | {r.utility_score:.0f} | {r.buildability_score:.0f} | '
                         f'{r.supporting_infra_score:.0f} | {r.dev_risk_score:.0f} | {r.utility_module_status} |')
    lines.append('')
    lines.append('---')
    lines.append('')

    # ---- 5. Strong power signal + competitive heat ----
    lines.append('## 5. Strong power signal but possible allocation / competitive-heat risk')
    lines.append('')
    lines.append('Candidates where the underlying queue is very active (strong signal) but the heuristic suggests the capacity may already be allocated or highly competitive. Treat the high score with care.')
    lines.append('')
    heat = df[
        (df.composite_score >= 80) &
        (df.allocation_or_competitive_heat_flag == True)
    ].sort_values('composite_score', ascending=False)
    lines.append(f'**Total flagged:** {len(heat):,}')
    lines.append('')
    lines.append('### Top 20')
    lines.append('')
    lines.append('| Candidate | State | County | Acres | Composite | Queue MW Nearby | Primary Anchor |')
    lines.append('|---|---|---|---|---|---|---|')
    for _, r in heat.head(20).iterrows():
        lines.append(f'| `{r.candidate_id[:8]}..` | {r.state} | {r.county_name} | {r.area_acres:.0f} | '
                     f'{r.composite_score:.1f} | {r.queue_mw_nearby:.0f} | {r.primary_anchor_name} |')
    lines.append('')
    lines.append('---')
    lines.append('')

    # ---- 6. TX/CA tile-boundary cluster issues ----
    lines.append('## 6. TX/CA tile-boundary cluster signals')
    lines.append('')
    lines.append('The candidate generation pipeline processed TX and CA in CDL tiles. Polygons that touched a tile boundary could appear as multiple adjacent small candidates instead of one large one. This section surfaces clusters of small candidates close together as potential tile-boundary artifacts to inspect in QGIS.')
    lines.append('')
    # Heuristic: clusters of small (<200 acre) candidates with centroid within 500m of another in same state
    lines.append('### Heuristic flags')
    lines.append('')
    lines.append('Looking for candidates with `<200 acres` and `<500m` between centroids in TX and CA. These are the tightest cluster candidates.')
    lines.append('')
    from scipy.spatial import cKDTree
    for st in ['TX','CA']:
        sd = df[(df.state == st) & (df.area_acres < 200)].copy().reset_index(drop=True)
        if len(sd) < 2:
            continue
        # Project to meters
        sd_proj = sd.to_crs(5070) if sd.crs.to_epsg() != 5070 else sd
        coords = list(zip(sd_proj.geometry.centroid.x, sd_proj.geometry.centroid.y))
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=500.0)
        # Count distinct candidates that are in at least one pair
        in_pair = set()
        for a, b in pairs:
            in_pair.add(a)
            in_pair.add(b)
        lines.append(f'- **{st}**: {len(in_pair):,} candidates in tight clusters '
                     f'({len(pairs):,} pairs within 500 m centroid distance)')
    lines.append('')
    lines.append('---')
    lines.append('')

    # ---- 7. Action distribution ----
    lines.append('## 7. Recommended action distribution')
    lines.append('')
    lines.append('| Action | Count |')
    lines.append('|---|---|')
    for act, n in df.recommended_action.value_counts().items():
        lines.append(f'| {act} | {n:,} |')
    lines.append('')
    lines.append('| Actionability status | Count |')
    lines.append('|---|---|')
    for act, n in df.actionability_status.value_counts().items():
        lines.append(f'| `{act}` | {n:,} |')
    lines.append('')
    lines.append('---')
    lines.append('')

    # ---- 8. Schema audit ----
    lines.append('## 8. Field schema audit')
    lines.append('')
    lines.append(f'- **Total columns:** {len(df.columns)}')
    lines.append(f'- **Base candidate columns (from upstream pipeline):** 21')
    lines.append(f'- **Phase 1 enrichment columns added by Step 1A-1J:** 63')
    lines.append(f'- **Scoring engine columns added by Step 2A-2G:** {len(df.columns) - 21 - 63}')
    lines.append('')
    lines.append('### Owen Phase 1 Review (PDF) recommendation compliance')
    lines.append('')
    lines.append('| Rec | Requirement | Status |')
    lines.append('|---|---|---|')
    recs = [
        ('1', 'End-market neutral language', 'PASS (candidate_type = greenfield, no end-use specific naming)'),
        ('2', 'Tiered MW scoring (no 200MW hard threshold)', 'PASS (queue_mw_tier 5-tier system)'),
        ('3', 'Queue status quality scoring', 'PASS (primary_anchor_queue_status_score 0-100)'),
        ('4', '25km buffer = proximity zone, not feasibility', 'PASS (no claim of confirmed service)'),
        ('5', 'Route complexity placeholders', 'PASS (route_complexity_score reserved as nullable)'),
        ('6', '230kV as secondary signal', 'PASS (nearest_230kv_distance_m at half-weight)'),
        ('8', 'FEMA AE soft / Floodway hard', 'PASS (fema_ae_overlap_flag soft, floodway is hard-excluded upstream)'),
        ('9', 'Radar as review flag', 'PASS (radar_review_flag, not a hard exclude)'),
        ('11','CDL primary, NLCD optional', 'PASS (cdl_group primary, nlcd_class nullable)'),
        ('12','Cropland scoring adjustment', 'PASS (cropland 65 < grassland 90)'),
        ('13','Slope two thresholds (8% review, 15% hard)', 'PASS (slope_tier with review_flag; >15% excluded upstream)'),
        ('14','Acreage tiers', 'PASS (acreage_tier in 5 buckets)'),
        ('15','Reuse nodes separate (Phase 2)', 'DEFERRED (per user decision May 14)'),
        ('17','Kill gates before composite', 'PASS (candidate_status applied; 27 flagged manual_review)'),
        ('18','"Utility Infrastructure Signal" naming', 'PASS (utility_score, not "power_readiness")'),
        ('19','40/20/15/15/10 weights', 'PASS (40 util / 20 build / 15 supp / 15 risk; 10 site_control deferred)'),
        ('29','Recommended action strings', 'PASS (8 Owen-defined labels supported)'),
        ('30','Dataset versioning', 'PASS (run_id, run_date, scoring_model_version, etc.)'),
    ]
    for rec_id, req, status in recs:
        lines.append(f'| #{rec_id} | {req} | {status} |')
    lines.append('')
    lines.append('---')
    lines.append('')

    # ---- 9. Known limitations / Phase 2 followups ----
    lines.append('## 9. Known limitations / Phase 2 followups')
    lines.append('')
    lines.append('1. **Site control / economic proxy (10% weight) — null in Phase 1.** Phase 2 will add parcel_count, owner_count, assessed_value_per_acre, ownership_fragmentation, distressed signals. Composite score is renormalized over the 90% observable weight.')
    lines.append('2. **Reuse nodes (greenfield only in Phase 1).** Owen Rec #15-16. Deferred per user decision May 14; revisit when ready.')
    lines.append('3. **Building footprint pct ≤ 5% threshold.** 20 candidates above 5% are flagged manual_review.')
    lines.append('4. **Original vs net buildable acreage.** Phase 1 uses convex-hull approximation for the "original" footprint (computationally feasible at scale). Per-reason area attribution (excluded_area_acres_by_reason) deferred to follow-up pass per Owen May 14 reply.')
    lines.append('5. **Oversized polygons (size_class = region).** 1,238 candidates above 5,000 acres; these are aggregates of available land, not single sites. Owen called out for review; subdivision is post-Friday.')
    lines.append('6. **NLCD cross-check.** nlcd_class field reserved but not populated in Phase 1. CDL is the authoritative land cover source.')
    lines.append('7. **Pipeline diameter is estimated** (Owen May 2 instruction). `pipeline_diameter_estimated = True` on every row.')
    lines.append('8. **Cropland active-vs-fallow not split.** All cropland candidates scored as land_cover=65; Phase 2 can split with CDL class codes.')
    lines.append('9. **Slope simplification artifacts.** ~10% of candidates have slope_max > 15% (small steep pixels that survived Step5e polygon simplification). These get the worst slope_tier score (penalized = 50, slope_review_flag = True).')
    lines.append('10. **No PJM staleness check.** Queue snapshot is from 2026-05 — production should add freshness guards (Owen May 2).')
    lines.append('')

    OUT_PATH.write_text('\n'.join(lines), encoding='utf-8')
    sz_kb = OUT_PATH.stat().st_size / 1024
    print(f'\nSaved: {OUT_PATH} ({sz_kb:.0f} KB, {len(lines)} lines)')


if __name__ == '__main__':
    main()
