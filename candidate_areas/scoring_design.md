# Candidate Site Scoring Engine — Design

**Status:** Design locked, awaiting implementation
**Phase:** 1 (Friday deliverable)
**Scope:** 95,269 candidate polygons across AZ, CA, NV, TX, VA

This document records the design of the scoring engine as agreed before implementation begins. Each section was reviewed and locked in conversation. Code references the section number it implements.

---

## 1. Overall Architecture

### 1.1 Subscore structure

Per Owen Rec #19, four operational subscores plus one placeholder.

| Subscore | Weight | Phase 1 status |
|---|---|---|
| Utility infrastructure | 40% | active |
| Physical buildability | 20% | active |
| Supporting infrastructure proximity | 15% | active |
| Development risk | 15% | active |
| Site control / economic proxy | 10% | null placeholder, Phase 2 |

Each active subscore returns 0 to 100. The site control subscore returns null in Phase 1.

### 1.2 Composite blending

```
composite_score = (0.40*util + 0.20*build + 0.15*supp + 0.15*risk) / 0.90
```

Normalized over the 90% observable weight, scaled to 0 to 100. Phase 1 rows get `score_v1_observable=True` and `data_coverage_pct=90`. Subscores that fail at runtime return null; the composite renormalizes over surviving subscores and writes the failed module name to `missing_modules`.

### 1.3 Confidence tier

Three tiers (high / medium / low) determined by worst-of-three across:

1. **Data coverage** — `missing_modules` count: 0 allows high, 1 caps at medium, 2+ forces low
2. **Anchor confidence** — exact substation match allows high, fuzzy match caps at medium, zone fallback forces low
3. **Signal corroboration** — count of subscores above 50: 3+ allows high, 2 caps at medium, fewer forces low

### 1.4 Reason codes

`top_reason_codes` is a list of 3 to 5 strings per candidate, drawn from a controlled vocabulary. Selection rule: rank all eligible codes by their absolute contribution to the composite score, take top 3, pad with up to 2 uncertain codes if any apply.

Three categories: positive (signals pushed score up), negative (signals pushed it down), uncertain (low-confidence inputs).

### 1.5 Two action fields

**`recommended_action`** — per Owen PDF Rec #29:

| composite_score | Condition | Action |
|---|---|---|
| any | hard_exclusion_flag = True | Ignore |
| any | candidate_type='reuse_node' AND environmental_review_required | Reuse Diligence |
| ≥ 85 | utility_review_required | Utility Desk Check |
| ≥ 75 | parcel_count is null | Parcel Pull |
| ≥ 65 | else | Monitor |
| else | | Ignore |

Standard labels: `Ignore`, `Monitor`, `Manual Review`, `Parcel Pull`, `Utility Desk Check`, `Ownership Review`, `Reuse Diligence`, `Shortlist`.

**`actionability_status`** — per Owen May 11 chat. Separate field reflecting what an analyst can actually do with the row.

Labels: `do_not_pitch`, `internal_diligence_only`, `apn_owner_pull_required`, `broker_verify_required`, `nda_teaser_possible`, `buyer_ready_with_caveats`.

Most Phase 1 rows land at `apn_owner_pull_required` since parcel data is not yet enriched.

### 1.6 Output schema additions

Per-candidate columns added by the scoring stage:

| Column | Type |
|---|---|
| utility_score | float |
| buildability_score | float |
| supporting_infra_score | float |
| dev_risk_score | float |
| site_control_score | float (null Phase 1) |
| composite_score | float |
| score_v1_observable | bool (True for all) |
| data_coverage_pct | float (90 for full Phase 1) |
| missing_modules | list[string] |
| confidence | string (high/medium/low) |
| recommended_action | string |
| actionability_status | string |
| top_reason_codes | list[string] |

Plus per-subscore breakdown columns documented in each subscore section.

---

## 2. Subscore 1 — Utility Infrastructure (40%)

### 2.1 Algorithm

For each candidate, find all substations within 50 km. Each is an "anchor." Compute per-anchor contribution, sum across anchors, cap at 100.

If no named anchors are found, fall back to a zone-level signal at reduced weight. If neither is available, `utility_score = null` and `utility_infrastructure` is added to `missing_modules`.

### 2.2 Per-anchor base score

Weighted blend of queue size and queue status, on a 0-100 scale.

**Queue MW tier** (per Owen Rec #2, pulled from `anchor_queue_stats.queue_mw_tier`):

| Active queue MW | Tier | Score |
|---|---|---|
| 0 | 1 | 0 |
| 0–50 | 2 | 25 |
| 50–200 | 3 | 50 |
| 200–500 | 4 | 75 |
| 500+ | 5 | 100 |

**Queue status quality** (per Owen Rec #3, pulled from `anchor_queue_stats.queue_status_score`, MW-weighted across projects at the anchor):

| Status | Score |
|---|---|
| Executed IA | 100 |
| In Study | 67 |
| Suspended | 37 |
| Withdrawn / terminated | 0 |

```
base_score = 0.6 * queue_mw_tier_score + 0.4 * status_score
```

### 2.3 Voltage multiplier

Per Owen Rec #6, with 230kV as secondary signal at lower weight:

| Anchor voltage | Multiplier |
|---|---|
| 500 kV | 1.20 |
| 345 kV | 1.10 |
| 230 kV | 1.05 |
| < 230 kV | 1.00 |

### 2.4 Activation band weighting

Each project's MW contribution to the anchor's total is weighted before tier classification:

| Months to activation | Weight |
|---|---|
| ≤ 18 | 1.10 |
| 18–36 | 1.00 |
| 36–60 | 0.90 |
| > 60 | 0.75 |

### 2.5 Distance decay (banded per Owen May 11 feedback)

| Candidate-to-anchor distance | Band | Weight |
|---|---|---|
| 0–5 km | strong | 1.00 |
| 5–10 km | moderate | 0.67 |
| 10–25 km | weak | 0.33 |
| 25–50 km | regional | 0.10 |
| > 50 km | not counted | 0 |

### 2.6 Match confidence weight

Per Owen May 2 instruction to distinguish true substations from line-based POIs:

| Match type | Weight |
|---|---|
| Exact name match to HIFLD | 1.00 |
| Fuzzy match (owner+voltage+county) | 0.70 |
| Zone fallback (ERCOT West, NV Energy, etc.) | 0.50 |

### 2.7 Per-anchor contribution

```
anchor_contribution = base_score
                    * voltage_multiplier
                    * distance_decay
                    * match_confidence
```

### 2.8 Aggregation

```
utility_score = min(100, sum(anchor_contribution for anchor in range))
```

The cap handles over-saturation. The primary anchor (top contributor) is exposed in the output for transparency.

### 2.9 Zone fallback

If zero substations within 50 km: use zone-level aggregated stats (from `eia_utility_territories`) as a synthetic anchor at moderate distance band (0.67) and zone_fallback confidence (0.50). `zone_fallback_used = True`.

If neither is available: `utility_score = null`, `utility_module_status = 'failed'`.

### 2.10 Standalone "hot node vs actionable site" fields (Owen May 14)

These are surfaced separately from the composite so the "near a busy node" signal is visible independent of the score:

- `node_activity_score` — same base as utility_score but without distance decay (raw "this region has hot anchors")
- `queue_mw_nearby` — sum of activation-weighted queue MW within 25 km
- `ia_executed_nearby` — True if any anchor within 25 km has an executed IA
- `activation_band_nearby` — most recent activation band among nearby anchors
- `recent_withdrawals_nearby` — count of withdrawals in last 12 months
- `allocation_or_competitive_heat_flag` — heuristic: high MW + low effective headroom relative to voltage class

### 2.11 Output columns

| Column | Type |
|---|---|
| utility_score | float |
| utility_module_status | string (complete / zone_fallback / failed) |
| primary_anchor_name | string |
| primary_anchor_distance_m | float |
| primary_anchor_voltage_kv | int |
| primary_anchor_queue_mw | float |
| primary_anchor_ia_executed | bool |
| primary_anchor_activation_band | string |
| primary_anchor_match_confidence | string |
| num_anchors_in_range | int |
| zone_fallback_used | bool |
| anchor_breakdown | list[dict] (top 3-5) |
| node_activity_score | float |
| queue_mw_nearby | float |
| ia_executed_nearby | bool |
| recent_withdrawals_nearby | int |
| allocation_or_competitive_heat_flag | bool |

### 2.12 Reason codes

**Positive:** `strong_queue_signal`, `interconnection_agreement_executed`, `345kv_anchor_in_range`, `500kv_anchor_in_range`, `recent_activation_band`, `multiple_anchors_in_range`

**Negative:** `no_queue_activity`, `no_executed_ia_in_range`, `distant_anchors_only`

**Uncertain:** `queue_anchor_zone_fallback`, `status_mostly_suspended`, `no_recent_queue_activity`, `allocation_risk_possible`

---

## 3. Subscore 2 — Physical Buildability (20%)

### 3.1 Components and weights

| Component | Weight |
|---|---|
| Land cover class (CDL group, Rec #12) | 35% |
| Acreage tier (Rec #14) | 25% |
| Slope tier (Rec #13) | 25% |
| Constraint penalties (FEMA AE + building footprint) | 15% |

### 3.2 Land cover (35%)

Per Owen Rec #12. Cropland kept merged at score 65 for Phase 1 (active vs fallow split deferred to Phase 2).

| CDL group | Score |
|---|---|
| grassland | 90 |
| shrub_barren | 80 |
| cropland | 65 |
| forest | 45 |

### 3.3 Acreage tier (25%)

Per Owen Rec #14. Step function with plateau at strategic-scale.

| Acreage | Tier | Score |
|---|---|---|
| 50–100 | small | 50 |
| 100–250 | moderate | 70 |
| 250–500 | large | 85 |
| 500–1,000 | very_large | 95 |
| 1,000+ | strategic_scale | 100 |

### 3.4 Slope tier (25%)

Per Owen Rec #13. 15%+ slopes already hard-excluded upstream.

| Condition | Tier | Score | review_flag |
|---|---|---|---|
| slope_mean ≤ 5% AND slope_max ≤ 5% | ideal | 100 | false |
| 5% < slope_mean ≤ 8% OR 5% < slope_max ≤ 8% | acceptable | 80 | false |
| 8% < slope_mean ≤ 15% OR 8% < slope_max ≤ 15% | penalized | 50 | true |

### 3.5 Constraint penalties (15%)

Stack on a baseline of 100, floor at 0.

**FEMA AE (Rec #8):**

| Condition | Penalty |
|---|---|
| AE zone overlaps candidate | −30 |
| AE zone within 500m of edge | −10 |

**Building footprint** (heavy penalty per user direction, no kill gate):

| building_footprint_pct | Penalty |
|---|---|
| 0–0.25 | 0 |
| 0.25–1 | −10 |
| 1–5 | −40 |
| 5–15 | −90 |
| > 15 | −120 |

```
constraint_score = max(0, 100 - sum(penalties))
```

### 3.6 Final formula

```
buildability_score = 0.35*land_cover_score
                   + 0.25*acreage_tier_score
                   + 0.25*slope_tier_score
                   + 0.15*constraint_score
```

### 3.7 Output columns

`buildability_score`, `land_cover_score`, `acreage_tier`, `acreage_tier_score`, `slope_tier`, `slope_tier_score`, `slope_mean_pct`, `slope_max_pct`, `slope_review_flag`, `fema_ae_flag`, `fema_ae_adjacent_flag`, `constraint_score`, `buildability_review_required`

### 3.8 Reason codes

**Positive:** `large_fallow_candidate`, `large_grassland_candidate`, `strategic_scale_candidate`, `ideal_slope`, `building_density_low`, `no_fema_exposure`

**Negative:** `forest_land`, `small_candidate`, `elevated_slope_band`, `fema_ae_overlap`, `building_density_moderate`, `building_density_high`

**Uncertain:** `cropland_active_vs_fallow_unknown`, `wetland_removed_high_share`, `floodway_adjacent_500m`

---

## 4. Subscore 3 — Supporting Infrastructure Proximity (15%)

### 4.1 Components and weights

| Component | Weight |
|---|---|
| Transmission (345/500 kV with 230 kV secondary) | 35% |
| Pipeline (tier + diameter weighted) | 25% |
| Class 1 rail (with STRACNET bonus) | 20% |
| Water service area | 20% |

### 4.2 Transmission (35%)

For 500 kV and 345 kV:

| Distance | 500 kV score | 345 kV score |
|---|---|---|
| Crosses candidate | 100 | 90 |
| 0–1 mi | 90 | 80 |
| 1–3 mi | 75 | 65 |
| 3–5 mi | 55 | 45 |
| 5–10 mi | 35 | 25 |
| 10–25 mi | 15 | 10 |
| > 25 mi | 0 | 0 |

For 230 kV: same bands, multiply final score by 0.5 (Rec #6).

```
transmission_score = max(score_500kv, score_345kv, 0.5 * score_230kv)
```

### 4.3 Pipeline (25%)

Owen-approved tier definitions (May 2). Diameter is estimated, labeled as such per Owen direction.

| Distance | Tier 1 (≥20") | Tier 2 (14–19") | Other (<14") |
|---|---|---|---|
| 0–5 mi | 100 | 75 | 50 |
| 5–10 mi | 70 | 50 | 30 |
| 10–25 mi | 40 | 25 | 15 |
| 25–50 mi | 15 | 10 | 5 |
| > 50 mi | 0 | 0 | 0 |

Take max across pipelines within 50 mi.

### 4.4 Class 1 rail (20%)

| Distance to nearest Class 1 rail | Score |
|---|---|
| Within 1 mi | 100 |
| 1–3 mi | 80 |
| 3–5 mi | 60 |
| 5–10 mi | 40 |
| 10–25 mi | 20 |
| > 25 mi | 5 |

Bonuses (cap at 100): STRACNET segment +10, multi-track (n_tracks ≥ 2) +5.

### 4.5 Water service (20%)

| Water service relationship | Base score |
|---|---|
| Centroid inside service area | 100 |
| 0–1 mi from boundary | 75 |
| 1–5 mi | 50 |
| 5–15 mi | 25 |
| > 15 mi | 5 |

Capacity bonus (cap at 100): pop_served > 50,000 → +10, pop_served 10,000–50,000 → +5.

### 4.6 Final formula

```
supporting_infra_score = 0.35*transmission_score
                       + 0.25*pipeline_score
                       + 0.20*rail_score
                       + 0.20*water_score
```

### 4.7 Output columns

`supporting_infra_score`, `transmission_score`, `pipeline_score`, `rail_score`, `water_score`, `nearest_500kv_distance_m`, `nearest_345kv_distance_m`, `nearest_230kv_distance_m`, `nearest_pipeline_distance_m`, `nearest_pipeline_operator_tier`, `nearest_pipeline_est_diameter_in`, `pipeline_diameter_estimated`, `nearest_class1_rail_distance_m`, `nearest_rail_is_stracnet`, `nearest_rail_n_tracks`, `within_water_service_area`, `nearest_water_service_distance_m`, `nearest_water_service_pop_served`

### 4.8 Reason codes

**Positive:** `500kv_within_3mi`, `345kv_within_5km`, `transmission_crosses_candidate`, `tier1_pipeline_within_5mi`, `large_pipeline_within_10mi`, `class1_rail_within_3mi`, `stracnet_rail_within_5mi`, `within_water_service_area`, `large_water_district_nearby`

**Negative:** `no_hv_transmission_in_range`, `distant_pipeline_only`, `no_class1_rail_in_25mi`, `outside_water_service`

**Uncertain:** `pipeline_diameter_estimated`, `water_capacity_unknown`

---

## 5. Subscore 4 — Development Risk (15%)

### 5.1 Components and weights

| Component | Weight |
|---|---|
| Seismic hazard | 25% |
| Drought tier | 20% |
| Radar review flag | 15% |
| PAD-US adjacency | 15% |
| Wetland adjacency | 15% |
| Floodway adjacency | 10% |

All components return 0–100 where 100 = no risk.

### 5.2 Seismic (25%)

Uses USGS NSHM PGA at 2% probability of exceedance in 50 years.

| PGA value (g) | Tier | Score |
|---|---|---|
| < 0.10 | very_low | 100 |
| 0.10–0.25 | low | 80 |
| 0.25–0.50 | moderate | 55 |
| 0.50–1.0 | high | 30 |
| > 1.0 | very_high | 10 |

### 5.3 Drought (20%)

NOAA Drought Monitor classification.

| Level | Label | Score |
|---|---|---|
| none | no drought | 100 |
| D0 | abnormally_dry | 85 |
| D1 | moderate_drought | 70 |
| D2 | severe_drought | 50 |
| D3 | extreme_drought | 30 |
| D4 | exceptional_drought | 10 |

### 5.4 Radar (15%)

Per Owen Rec #9, downgraded to review flag.

| radar_distance_miles | Score |
|---|---|
| > 10 mi | 100 |
| 5–10 mi | 85 |
| 3–5 mi | 60 |
| 0–3 mi (review flag) | 30 |

### 5.5 PAD-US adjacency (15%)

Protected land already excluded upstream; scoring captures soft proximity risk.

| Nearest PAD-US distance from edge | Score | near_padus_flag |
|---|---|---|
| > 5 mi | 100 | false |
| 500m–5 mi | 85 | false |
| 100m–500m | 60 | true |
| < 100m | 30 | true |

### 5.6 Wetland adjacency (15%)

Same logic as PAD-US.

| Nearest wetland distance from edge | Score | near_wetland_flag |
|---|---|---|
| > 5 mi | 100 | false |
| 500m–5 mi | 85 | false |
| 100m–500m | 60 | true |
| < 100m | 35 | true |

### 5.7 Floodway adjacency (10%)

| Nearest floodway distance from edge | Score | adjacent_floodway_flag |
|---|---|---|
| > 1 mi | 100 | false |
| 500m–1 mi | 80 | false |
| 100m–500m | 50 | true |
| < 100m | 25 | true |

### 5.8 Final formula

```
dev_risk_score = 0.25*seismic_score
               + 0.20*drought_score
               + 0.15*radar_score
               + 0.15*padus_score
               + 0.15*wetland_score
               + 0.10*floodway_score
```

### 5.9 Output columns

`dev_risk_score`, `seismic_score`, `seismic_hazard_pga`, `seismic_hazard_tier`, `drought_score`, `drought_level`, `drought_label`, `radar_score`, `radar_distance_miles`, `radar_review_flag`, `padus_score`, `nearest_padus_distance_m`, `near_padus_flag`, `wetland_score`, `nearest_wetland_distance_m`, `near_wetland_flag`, `floodway_score`, `nearest_floodway_distance_m`, `adjacent_floodway_flag`

### 5.10 Reason codes

**Positive:** `low_seismic_risk`, `no_drought_exposure`, `no_padus_adjacency`, `no_wetland_adjacency`, `no_floodway_adjacency`

**Negative:** `high_seismic_zone`, `drought_tier_high`, `radar_review_flag`, `padus_adjacent_500m`, `wetland_adjacent_500m`, `floodway_adjacent_500m`

**Uncertain:** `seismic_data_missing`, `drought_data_stale`

---

## 6. Phase 2 placeholder columns

These get written as null on every Phase 1 row so the output schema is final.

**Parcel/owner block:** `parcel_count`, `owner_count`, `largest_owner_acres`, `largest_owner_pct_of_candidate`, `assessed_value_total`, `assessed_value_per_acre`, `last_sale_date`, `last_sale_price`, `land_use_code`, `zoning_code`, `road_frontage_flag`, `legal_access_flag`

**Scoring placeholders:** `site_control_score`, `economic_proxy_score`

**Utility feasibility:** `serving_utility`, `utility_territory_known`, `nearest_load_serving_node`, `utility_service_feasibility_score`, `utility_review_required`

**Communications:** `communications_route_distance`, `communications_provider_count`, `communications_access_score`

**Water capacity:** `water_capacity_known`, `water_capacity_review_required`

**Jurisdiction:** `jurisdiction_review_required`, `local_policy_notes`

**Manual QA:** `manual_imagery_review_status`, `manual_imagery_review_notes`

---

## 7. Dataset versioning

Per Owen Rec #30. Eleven fields stamped on every row:

`run_id`, `run_date`, `scoring_model_version`, `exclusion_model_version`, `cdl_year`, `padus_version`, `fema_nfhl_date`, `nwi_date`, `transmission_dataset_version`, `queue_dataset_date`, `dem_dataset_version`

---

## 8. Kill gates (Owen Rec #17)

Per-candidate `candidate_status` computed before composite scoring. Values: `pass`, `excluded`, `manual_review`. Only `pass` and `manual_review` rows enter the composite stage.

Gate conditions:

| Gate | Action |
|---|---|
| Hard exclusion already triggered upstream | excluded |
| Slope max > 15% (should not happen, already excluded) | excluded |
| Protected land conflict (should not happen, already excluded) | excluded |
| Open water / wetland-heavy (already excluded) | excluded |
| No plausible access | manual_review |
| High-risk jurisdictional issue | manual_review |
| Reuse-node environmental uncertainty (Phase 2) | manual_review |

For Phase 1, almost all rows will be `pass` since upstream exclusions handle most gates.

---

## 9. Execution sequence

1. **USGS NSHM seismic ingestion** — fill the only data gap
2. **Pre-scoring enrichment** — compute every per-candidate input the four subscores need
3. **Kill gates layer** — compute `candidate_status`
4. **Subscore implementations** — utility → buildability → supporting infra → dev risk, verified each before moving on
5. **Standalone surfaced fields** — node_activity_score and related (Owen May 14)
6. **Composite, confidence, two action fields, reason codes**
7. **Phase 2 placeholder columns and dataset versioning stamping**
8. **Final exports** — `candidates_final.csv` + `candidates_final.geojson`
9. **QA summary markdown**
10. **Deliver to Owen** — GitHub push + share outputs

---

## 10. What is explicitly deferred

- Reuse nodes ingestion (Owen Phase 1 review Step 8) — user deferred
- Parcel and ownership enrichment (PDF Stage 10, Rec #21-22)
- Economic proxy automation (Rec #22)
- Utility-service feasibility automation (Rec #23, manual for top candidates only)
- Communications/backhaul real data (Rec #24)
- Water capacity confirmation (Rec #25)
- Local jurisdiction automation (Rec #26)
- Manual imagery QA workflow (Rec #28)
- Historical site validation (Rec #27)
- SSURGO civil/site variables (Rec #20)
- Military, airports, tribal lands, mines, wildfire exclusion layers (Rec #10)
- Subdivision of oversized polygons (Option B from oversized polygons message — pending Owen reply)
- `excluded_area_acres_by_reason` per-layer area attribution — post-Friday follow-up pass per Owen approval
