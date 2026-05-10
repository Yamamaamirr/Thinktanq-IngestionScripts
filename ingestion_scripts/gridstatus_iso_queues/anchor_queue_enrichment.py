"""ISO queue -> per-anchor aggregation enrichment.

Produces the anchor_queue_stats table: one row per HIFLD anchor that has any
matched queue activity. Joined later to parcels via spatial nearest-anchor
lookup to populate candidate_activation per the PRD section 5.2 schema.

Inputs:
  cache/{caiso,miso,isone,...}.parquet — raw normalized queue data
  raw_substations.parquet              — HIFLD substations (the anchors)

Output:
  anchor_queue_stats.parquet           — per-anchor aggregations

Schema:
  anchor_id                          TEXT PRIMARY KEY
  anchor_name                        TEXT
  anchor_state                       TEXT
  iso_region                         TEXT
  total_active_capacity_mw           NUMERIC
  total_energized_capacity_mw        NUMERIC
  total_withdrawn_capacity_mw        NUMERIC
  active_project_count               INT
  energized_project_count            INT
  earliest_active_in_service_date    DATE
  estimated_activation_mo            INT       -- months from today to earliest
  activation_band                    TEXT      -- le_18 / 18_36 / 36_60 / gt_60
  recent_withdrawal_count_12mo       INT
  recent_withdrawal_capacity_mw_12mo NUMERIC
  mw_commitment_letter               BOOL      -- True if any active/energized row has ia_executed=True
  best_match_method                  TEXT      -- highest-confidence match used
  match_count                        INT       -- queue rows mapped to this anchor

Why per-anchor and not per-parcel:
  Per the PRD, candidate_activation is per-parcel. But the queue data attaches
  to substations, not parcels. The right architecture is two stages:
    1. (this script) aggregate queue -> anchor_queue_stats
    2. (parcel pipeline) for each parcel, find nearest anchor within Xmi,
       lookup its anchor_queue_stats row, write to candidate_activation.
  Stage 2 lives with the parcel-scoring code, not here, because it depends on
  parcel geometry which doesn't exist in this ingestion folder.
"""
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd

from iso_anchor_match import build_anchor_index, match_all

# PRD section 8.2 activation bands.
def _activation_band(months):
    if pd.isna(months):
        return None
    if months <= 18:
        return 'le_18'
    if months <= 36:
        return '18_36'
    if months <= 60:
        return '36_60'
    return 'gt_60'

# Match-method priority for "best_match_method" — highest confidence first.
METHOD_RANK = {
    'exact': 4,
    'fuzzy_high': 3,
    'fuzzy_high_endpoint': 2,
    'fuzzy_medium': 1,
    'fuzzy_low': 0,
}

SUBS_PARQUET = Path(r'C:\thinqtank\ingestion_scripts\hifld_electric_substations\hifld_electric_substations.parquet')
SUPP_PARQUET = Path(__file__).parent / 'supplemental_substations.parquet'
QUEUE_PARQUET = Path(__file__).parent / 'iso_queues.parquet'
OUT_PARQUET = Path(__file__).parent / 'anchor_queue_stats.parquet'
ZONE_OUT_PARQUET = Path(__file__).parent / 'zone_queue_stats.parquet'

WITHDRAWAL_RECENT_DAYS = 365


def _best_match_method(series):
    methods_present = [m for m in series if m in METHOD_RANK]
    if not methods_present:
        return None
    return max(methods_present, key=METHOD_RANK.get)


def main():
    print("Loading substations (HIFLD + supplemental)...")
    subs = gpd.read_parquet(SUBS_PARQUET)
    # Exclude NULL class_rating (private industrial sites — not grid anchors).
    subs = subs[subs['class_rating'].notna()].copy()
    print(f"  {len(subs):,} HIFLD grid-anchor substations")

    if SUPP_PARQUET.exists():
        supp = gpd.read_parquet(SUPP_PARQUET)
        # Align columns: supplemental has name, state_abbr, geometry, class_rating
        # max_volt is HIFLD-only; supplemental substations get None so anchor_voltage_kv
        # is still populated for HIFLD anchors after the concat.
        supp_aligned = supp[['name', 'state_abbr', 'geometry', 'class_rating']].copy()
        supp_aligned['max_volt'] = None
        hifld_cols = ['name', 'state_abbr', 'geometry', 'class_rating', 'max_volt']
        subs = pd.concat([subs[hifld_cols], supp_aligned[hifld_cols]],
                         ignore_index=True)
        subs = gpd.GeoDataFrame(subs, geometry='geometry', crs='EPSG:4326')
        print(f"  +{len(supp_aligned):,} supplemental (EIA860+OSM) -> {len(subs):,} total")
    else:
        print("  (no supplemental_substations.parquet found, using HIFLD only)")

    anchor_meta = subs[['name', 'state_abbr', 'max_volt']].copy()
    anchor_meta.columns = ['anchor_name', 'anchor_state', 'anchor_voltage_kv']
    anchor_meta['anchor_id'] = anchor_meta.index.astype(str)

    print("\nBuilding state-indexed anchor lookup...")
    anchor_idx = build_anchor_index(subs, name_col='name', state_col='state_abbr',
                                    volt_col='max_volt')
    print(f"  {len(anchor_idx)} states indexed")

    print("\nLoading + matching ISO queues...")
    queues = pd.read_parquet(QUEUE_PARQUET)
    print(f"  {len(queues):,} queue rows loaded")
    matched = match_all(queues, anchor_idx)
    # Preserve load_zone through the match so zone fallback can use it.
    if 'load_zone' not in matched.columns and 'load_zone' in queues.columns:
        matched['load_zone'] = queues['load_zone'].values
    matched_only = matched[matched['anchor_id'].notna()].copy()
    print(f"  {len(matched_only):,} matched to an anchor "
          f"({len(matched_only)/len(matched)*100:.0f}%)")
    if 'poi_type' in matched.columns:
        n_line_poi = (matched['poi_type'] == 'line_poi').sum()
        n_line_matched = (matched_only['poi_type'] == 'line_poi').sum() if 'poi_type' in matched_only.columns else '?'
        print(f"  {n_line_poi:,} line-POI entries detected "
              f"({n_line_matched} matched via endpoint-split)")

    # Coerce dates and capacity for safe aggregation.
    matched_only['in_service_date'] = pd.to_datetime(matched_only['in_service_date'], errors='coerce')
    matched_only['withdrawal_date'] = pd.to_datetime(matched_only['withdrawal_date'], errors='coerce')
    matched_only['capacity_mw'] = pd.to_numeric(matched_only['capacity_mw'], errors='coerce')
    # ia_executed: old caches lack this column — default False so agg never crashes.
    if 'ia_executed' not in matched_only.columns:
        matched_only['ia_executed'] = False
    matched_only['ia_executed'] = matched_only['ia_executed'].fillna(False).astype(bool)
    # poi_type: line-POI rows are transmission-line taps, not substations.
    # Tracked per anchor so analysts can see match composition quality.
    if 'poi_type' not in matched_only.columns:
        matched_only['poi_type'] = 'unknown'
    matched_only['_is_line_poi'] = matched_only['poi_type'] == 'line_poi'

    today = pd.Timestamp(datetime.utcnow().date())
    cutoff_recent = today - timedelta(days=WITHDRAWAL_RECENT_DAYS)

    matched_only['_active'] = matched_only['queue_status'] == 'active'
    matched_only['_energized'] = matched_only['queue_status'] == 'energized'
    matched_only['_withdrawn'] = matched_only['queue_status'] == 'withdrawn'
    matched_only['_suspended'] = matched_only['queue_status'] == 'suspended'
    matched_only['_recent_withdrawal'] = (
        matched_only['_withdrawn']
        & matched_only['withdrawal_date'].notna()
        & (matched_only['withdrawal_date'] >= cutoff_recent)
    )
    # Only consider future in-service dates as a signal of "time to activation".
    # Projects with past in-service dates that are still 'active' are chronically
    # delayed -- treating them as imminent (mo=0) would falsely score them as
    # ready. Better to leave estimated_activation_mo null so they're flagged for
    # analyst attention rather than auto-scored le_18.
    matched_only['_active_future'] = (
        matched_only['_active']
        & matched_only['in_service_date'].notna()
        & (matched_only['in_service_date'] >= today)
    )

    print("\nAggregating per anchor...")
    grouped = matched_only.groupby('anchor_id', dropna=False)

    stats = grouped.agg(
        anchor_name=('anchor_name', 'first'),
        iso_region=('iso_region', lambda s: ','.join(sorted(set(s.dropna())))),
        total_active_capacity_mw=('capacity_mw', lambda s: s[matched_only.loc[s.index, '_active']].sum()),
        total_energized_capacity_mw=('capacity_mw', lambda s: s[matched_only.loc[s.index, '_energized']].sum()),
        total_withdrawn_capacity_mw=('capacity_mw', lambda s: s[matched_only.loc[s.index, '_withdrawn']].sum()),
        total_suspended_capacity_mw=('capacity_mw', lambda s: s[matched_only.loc[s.index, '_suspended']].sum()),
        active_project_count=('_active', 'sum'),
        energized_project_count=('_energized', 'sum'),
        suspended_project_count=('_suspended', 'sum'),
        earliest_active_in_service_date=(
            'in_service_date',
            lambda s: s[matched_only.loc[s.index, '_active_future']].min(),
        ),
        active_overdue_count=(
            '_active',
            lambda s: int((s & ~matched_only.loc[s.index, '_active_future']
                           & matched_only.loc[s.index, 'in_service_date'].notna()).sum()),
        ),
        recent_withdrawal_count_12mo=('_recent_withdrawal', 'sum'),
        recent_withdrawal_capacity_mw_12mo=(
            'capacity_mw',
            lambda s: s[matched_only.loc[s.index, '_recent_withdrawal']].sum(),
        ),
        match_count=('source_id', 'count'),
        best_match_method=('match_method', _best_match_method),
        mw_commitment_letter=(
            'ia_executed',
            lambda s: bool(s[matched_only.loc[s.index, '_active'] | matched_only.loc[s.index, '_energized']].any()),
        ),
        line_poi_count=('_is_line_poi', 'sum'),
    ).reset_index()

    stats = stats.merge(
        anchor_meta[['anchor_id', 'anchor_state', 'anchor_voltage_kv']],
        on='anchor_id', how='left'
    )

    # Attach geometry from substations so the scoring engine can do spatial
    # radius queries directly on anchor_queue_stats (no join back to subs needed).
    subs_geom = subs[['geometry']].copy()
    subs_geom.index = subs_geom.index.astype(str)
    subs_geom.columns = ['geometry']
    stats = stats.merge(subs_geom, left_on='anchor_id', right_index=True, how='left')

    # Compute estimated activation months from earliest active in-service date.
    delta = stats['earliest_active_in_service_date'] - today
    stats['estimated_activation_mo'] = (delta.dt.days / 30.44).round().astype('Int64')
    # All retained dates are >= today by construction, so values are >= 0.
    stats['activation_band'] = stats['estimated_activation_mo'].apply(_activation_band)

    # match_confidence: explicit confidence tier for each anchor.
    _CONF = {'exact': 'high', 'fuzzy_high': 'high',
             'fuzzy_high_endpoint': 'medium', 'fuzzy_medium': 'medium', 'fuzzy_low': 'low'}
    stats['match_confidence'] = stats['best_match_method'].map(_CONF)

    # queue_mw_tier: tiered MW signal strength — used for scoring at Stage 9.
    # <25 MW is noise at this scale; 200+ MW is unambiguous developer convergence.
    # Note: the keep-zone spatial filter stays at >=200 MW (with rescue clauses).
    # Tiering is a scoring signal, not a justification to expand the spatial filter —
    # the net-new substations a 25 MW floor would add are predominantly sub-138 kV
    # distribution nodes that cannot serve large-format infrastructure loads.
    def _mw_tier(mw):
        if pd.isna(mw) or mw < 25:
            return 'negligible'
        if mw < 50:
            return 'weak'
        if mw < 100:
            return 'moderate'
        if mw < 200:
            return 'strong'
        return 'very_strong'

    stats['queue_mw_tier'] = stats['total_active_capacity_mw'].apply(_mw_tier)

    # queue_status_best / queue_status_score: best IA status quality across projects.
    # Four-tier hierarchy per Owen #3:
    #   Executed IA  → 100  (active project with ia_executed=True)
    #   In Study     → 67   (active project, no signed IA; midpoint of Owen's 60-75)
    #   Suspended    → 37   (suspended project; midpoint of Owen's 30-45)
    #   No active    → 0
    # NOTE: suspended scoring requires gridstatus_iso_queues.py STATUS_MAP fix
    # ("suspended" → "suspended" not "withdrawn"). Re-run the full ingestion chain
    # (gridstatus_iso_queues → iso_anchor_match → this script) for suspended
    # records to appear in total_suspended_capacity_mw / suspended_project_count.
    def _status_score(row):
        if row['mw_commitment_letter']:
            return ('executed_ia', 100)
        if row['active_project_count'] > 0:
            return ('in_study', 67)
        if row['suspended_project_count'] > 0:
            return ('suspended', 37)
        return ('no_active', 0)

    status_cols = stats.apply(_status_score, axis=1, result_type='expand')
    stats['queue_status_best'] = status_cols[0]
    stats['queue_status_score'] = status_cols[1]

    # queue_signal_score: composite of MW tier strength and status quality.
    # MW tier mapped to 0-100: negligible=0, weak=25, moderate=50, strong=75, very_strong=100
    # Weighted 60% MW signal + 40% status quality — MW is the primary signal.
    _MW_SCORE = {'negligible': 0, 'weak': 25, 'moderate': 50, 'strong': 75, 'very_strong': 100}
    stats['queue_signal_score'] = (
        stats['queue_mw_tier'].map(_MW_SCORE) * 0.6
        + stats['queue_status_score'] * 0.4
    ).round(1)

    # line_poi_fraction: proportion of matched rows that are line-tap POIs, not substations.
    # A high fraction flags anchors whose MW totals may be inflated by mis-attributed
    # line-segment capacity (analyst signal, not a disqualifier).
    stats['line_poi_fraction'] = (
        stats['line_poi_count'] / stats['match_count'].replace(0, pd.NA)
    ).round(2)

    # Reorder columns to match PRD-style schema (geometry last for parquet compat).
    stats = stats[[
        'anchor_id', 'anchor_name', 'anchor_state', 'anchor_voltage_kv', 'iso_region',
        'total_active_capacity_mw', 'total_energized_capacity_mw',
        'total_withdrawn_capacity_mw', 'total_suspended_capacity_mw',
        'active_project_count', 'energized_project_count', 'suspended_project_count',
        'active_overdue_count',
        'earliest_active_in_service_date', 'estimated_activation_mo', 'activation_band',
        'recent_withdrawal_count_12mo', 'recent_withdrawal_capacity_mw_12mo',
        'mw_commitment_letter',
        'queue_mw_tier', 'queue_status_best', 'queue_status_score', 'queue_signal_score',
        'best_match_method', 'match_confidence', 'match_count',
        'line_poi_count', 'line_poi_fraction',
        'geometry',
    ]]
    stats = gpd.GeoDataFrame(stats, geometry='geometry', crs='EPSG:4326')

    print(f"\n{len(stats):,} anchors have queue activity\n")
    print("activation_band breakdown:")
    print(stats['activation_band'].value_counts(dropna=False).to_string())
    print(f"\nbest_match_method breakdown:")
    print(stats['best_match_method'].value_counts(dropna=False).to_string())
    print(f"\nestimated_activation_mo distribution:")
    print(stats['estimated_activation_mo'].describe().round(1).to_string())
    print(f"\nTop 10 anchors by total_active_capacity_mw:")
    top = stats.nlargest(10, 'total_active_capacity_mw')[
        ['anchor_name', 'anchor_state', 'iso_region', 'total_active_capacity_mw',
         'active_project_count', 'estimated_activation_mo', 'activation_band',
         'recent_withdrawal_count_12mo', 'best_match_method']
    ]
    print(top.to_string(index=False))

    print(f"\nAnchors with mw_commitment_letter=True: {stats['mw_commitment_letter'].sum()}")
    print(f"\nTop 10 anchors by recent withdrawals (negative signal):")
    high_withdraw = stats[stats['recent_withdrawal_count_12mo'] > 0].nlargest(
        10, 'recent_withdrawal_capacity_mw_12mo'
    )[['anchor_name', 'anchor_state', 'iso_region',
       'recent_withdrawal_count_12mo', 'recent_withdrawal_capacity_mw_12mo',
       'best_match_method']]
    print(high_withdraw.to_string(index=False))

    stats.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {OUT_PARQUET}")

    # ── Zone-level fallback (Owen PRD section) ────────────────────────────────
    # For rows that couldn't be matched to a specific substation but have a
    # known load zone (ERCOT CDR Zone, NV Energy territory), aggregate MW by
    # zone. The scoring engine uses these at 50% weight when no substation-
    # level anchor is found within range of a parcel.
    unmatched_zoned = matched[
        matched['anchor_id'].isna()
        & matched['load_zone'].notna()
        & matched['queue_status'].isin(['active', 'energized'])
    ].copy()
    unmatched_zoned['in_service_date'] = pd.to_datetime(
        unmatched_zoned['in_service_date'], errors='coerce'
    )
    unmatched_zoned['_future'] = (
        unmatched_zoned['in_service_date'].notna()
        & (unmatched_zoned['in_service_date'] >= today)
    )

    if len(unmatched_zoned) > 0:
        zone_stats = unmatched_zoned.groupby('load_zone').agg(
            iso_region=('iso_region', lambda s: ','.join(sorted(set(s.dropna())))),
            total_unmatched_active_mw=('capacity_mw', 'sum'),
            active_project_count=('source_id', 'count'),
            earliest_in_service_date=(
                'in_service_date',
                lambda s: s[unmatched_zoned.loc[s.index, '_future']].min(),
            ),
        ).reset_index()
        zone_stats['zone_is_fallback'] = True   # Owen: always flag zone rows as low-confidence
        zone_stats['score_weight'] = 0.5         # half-weight vs substation-matched anchors
        delta = zone_stats['earliest_in_service_date'] - today
        zone_stats['estimated_activation_mo'] = (
            (delta.dt.days / 30.44).round().astype('Int64')
        )
        zone_stats.to_parquet(ZONE_OUT_PARQUET, index=False)
        print(f"\nZone-level fallback stats ({len(zone_stats)} zones):")
        print(zone_stats[['load_zone', 'iso_region', 'total_unmatched_active_mw',
                           'active_project_count', 'score_weight',
                           'estimated_activation_mo']].to_string(index=False))
        print(f"Wrote {ZONE_OUT_PARQUET}")


if __name__ == '__main__':
    main()
