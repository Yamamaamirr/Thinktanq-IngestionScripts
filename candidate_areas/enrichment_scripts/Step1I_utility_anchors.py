"""
Step 1I — Utility anchor lookup per candidate.

For each candidate, find all queue anchors within 50 km of the candidate
polygon edge, compute per-anchor scoring components, surface primary anchor
info + standalone "hot node" fields. The scoring engine (Phase 2) consumes
these and produces utility_score per scoring_design.md Section 2.

Method (corrected)
------------------
1) Build an STRtree over anchor points (each anchor is a POINT in EPSG:5070).
2) For each candidate polygon, query the tree with the polygon's bbox
   EXPANDED by 50 km — returns anchor indices that might be in range.
3) For each (candidate, anchor_idx) pair returned by the bbox prefilter,
   compute exact polygon-to-anchor-point distance.
4) Filter to distance <= 50 km.
5) Aggregate per candidate: primary anchor, anchor list, node_activity fields.

This is mathematically identical to the polygon-buffer + sjoin approach but
much faster because we skip the expensive polygon-buffer step (which on
complex multi-ring TX candidates was the 33-minute bottleneck).

Output (two parquets):
  candidate_areas/enrichment_outputs/step1i_utility_summary.parquet
    one row per candidate with primary anchor info + standalone hot-node fields
  candidate_areas/enrichment_outputs/step1i_utility_anchor_pairs.parquet
    one row per (candidate, contributing anchor) pair

Run:
  python candidate_areas/enrichment_scripts/Step1I_utility_anchors.py
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import STRtree
from shapely.geometry import box

CAND_PATH    = Path('candidate_areas/outputs/candidate_areas.parquet')
ANCHOR_PATH  = Path('ingestion_scripts/gridstatus_iso_queues/anchor_queue_stats.parquet')
OUT_SUMMARY  = Path('candidate_areas/enrichment_outputs/step1i_utility_summary.parquet')
OUT_PAIRS    = Path('candidate_areas/enrichment_outputs/step1i_utility_anchor_pairs.parquet')

MAX_SEARCH_M     = 50_000.0   # 50 km — scoring design Section 2.5
NEARBY_FOR_NODE  = 25_000.0   # 25 km for the standalone node_activity fields
STATES = ['AZ','CA','NV','TX','VA']

QUEUE_MW_TIER_SCORE = {
    'negligible':   0.0,
    'weak':        25.0,
    'moderate':    50.0,
    'strong':      75.0,
    'very_strong':100.0,
}

def voltage_multiplier(kv):
    if pd.isna(kv): return 1.0
    if kv >= 500: return 1.20
    if kv >= 345: return 1.10
    if kv >= 230: return 1.05
    return 1.00

ACTIVATION_WEIGHT = {
    'le_18': 1.10, '18_36': 1.00, '36_60': 0.90, 'gt_60': 0.75, None: 1.00,
}

def distance_band_and_decay(d_m):
    if d_m <= 5000:   return 'strong',   1.00
    if d_m <= 10000:  return 'moderate', 0.67
    if d_m <= 25000:  return 'weak',     0.33
    if d_m <= 50000:  return 'regional', 0.10
    return 'out_of_range', 0.0

def match_confidence_weight(mc, line_poi_frac):
    if mc == 'high':     base = 1.00
    elif mc == 'medium': base = 0.70
    else:                base = 0.50
    if pd.notna(line_poi_frac) and line_poi_frac > 0.5:
        base *= 0.80
    return base


def main():
    print(f'Loading candidates: {CAND_PATH} ...')
    cands = gpd.read_parquet(CAND_PATH)
    print(f'  {len(cands):,} candidates in EPSG:{cands.crs.to_epsg()}')

    print(f'\nLoading anchors: {ANCHOR_PATH} ...')
    anchors = gpd.read_parquet(ANCHOR_PATH)
    if anchors.crs.to_epsg() != 5070:
        anchors = anchors.to_crs(5070)
    print(f'  {len(anchors):,} anchors')

    # Pre-compute per-anchor scoring components
    anchors['queue_mw_tier_score']    = anchors.queue_mw_tier.map(QUEUE_MW_TIER_SCORE).fillna(0.0)
    anchors['voltage_multiplier']     = anchors.anchor_voltage_kv.apply(voltage_multiplier)
    anchors['activation_band_weight'] = anchors.activation_band.map(ACTIVATION_WEIGHT).fillna(1.0)
    # base_score: 0.6 * mw_tier_score + 0.4 * status_score
    anchors['base_score'] = (
        0.6 * anchors['queue_mw_tier_score'] +
        0.4 * anchors['queue_status_score']
    )

    summary_rows = []
    pairs_rows   = []

    for st in STATES:
        sd = cands[cands.state == st].reset_index(drop=True)
        if len(sd) == 0:
            continue
        print(f'\n-- {st} -----------------------------------------')
        print(f'  {len(sd):,} candidates')

        # Restrict anchors to state envelope (with padding > 50 km cap)
        bb = sd.total_bounds
        pad = 60_000
        env = box(bb[0]-pad, bb[1]-pad, bb[2]+pad, bb[3]+pad)
        st_anchors = anchors[anchors.geometry.intersects(env)].reset_index(drop=True)
        print(f'  Anchors in 60 km envelope: {len(st_anchors):,}')

        if len(st_anchors) == 0:
            for cid in sd.candidate_id.values:
                summary_rows.append({
                    'candidate_id': cid,
                    'num_anchors_in_range': 0, 'zone_fallback_used': True,
                    'primary_anchor_name': None, 'primary_anchor_distance_m': None,
                    'primary_anchor_voltage_kv': None,
                    'primary_anchor_queue_mw_tier': None,
                    'primary_anchor_queue_status_score': None,
                    'primary_anchor_activation_band': None,
                    'primary_anchor_match_confidence': None,
                    'primary_anchor_distance_band': None,
                    'node_activity_score': 0.0, 'queue_mw_nearby': 0.0,
                    'ia_executed_nearby': False, 'recent_withdrawals_nearby': 0,
                    'allocation_or_competitive_heat_flag': False,
                })
            continue

        # Build STRtree over anchor POINTS
        anchor_geoms = st_anchors.geometry.values
        tree = STRtree(anchor_geoms)

        # Per-candidate: bbox prefilter (50 km expanded) + exact polygon distance
        print(f'  Per-candidate bbox prefilter + exact distance ...', flush=True)
        t0 = time.time()
        pair_records = []
        log_every = max(1, len(sd) // 10)
        for i, row in enumerate(sd.itertuples(index=False)):
            geom = row.geometry
            cid  = row.candidate_id
            # Expanded bbox: 50 km on each side
            minx, miny, maxx, maxy = geom.bounds
            query_box = box(minx - MAX_SEARCH_M, miny - MAX_SEARCH_M,
                            maxx + MAX_SEARCH_M, maxy + MAX_SEARCH_M)
            cand_idx = tree.query(query_box, predicate='intersects')
            for k in cand_idx:
                d = geom.distance(anchor_geoms[k])
                if d <= MAX_SEARCH_M:
                    pair_records.append((cid, k, d))
            if (i+1) % log_every == 0:
                elapsed = time.time() - t0
                rate = (i+1) / elapsed if elapsed > 0 else 0
                eta = (len(sd) - i - 1) / max(rate, 0.01)
                print(f'    {i+1:,}/{len(sd):,} ({100*(i+1)/len(sd):.0f}%) — '
                      f'{rate:.0f} cand/s, ETA {eta:.0f}s', flush=True)
        print(f'    {time.time()-t0:.1f}s ({len(pair_records):,} pairs)')

        if not pair_records:
            for cid in sd.candidate_id.values:
                summary_rows.append({
                    'candidate_id': cid,
                    'num_anchors_in_range': 0, 'zone_fallback_used': True,
                    'primary_anchor_name': None, 'primary_anchor_distance_m': None,
                    'primary_anchor_voltage_kv': None,
                    'primary_anchor_queue_mw_tier': None,
                    'primary_anchor_queue_status_score': None,
                    'primary_anchor_activation_band': None,
                    'primary_anchor_match_confidence': None,
                    'primary_anchor_distance_band': None,
                    'node_activity_score': 0.0, 'queue_mw_nearby': 0.0,
                    'ia_executed_nearby': False, 'recent_withdrawals_nearby': 0,
                    'allocation_or_competitive_heat_flag': False,
                })
            continue

        # Build a DataFrame of pairs and join in anchor attributes
        pair_df = pd.DataFrame(pair_records, columns=['candidate_id', 'anchor_idx', 'distance_m'])
        anchor_attr_cols = [
            'anchor_id','anchor_name','anchor_voltage_kv','queue_mw_tier',
            'queue_mw_tier_score','queue_status_best','queue_status_score',
            'queue_signal_score','match_confidence','line_poi_fraction',
            'activation_band','activation_band_weight',
            'total_active_capacity_mw','recent_withdrawal_count_12mo',
            'voltage_multiplier','base_score',
        ]
        anchor_attrs = st_anchors[anchor_attr_cols].reset_index(drop=True)
        pair_df = pair_df.merge(anchor_attrs, left_on='anchor_idx', right_index=True, how='left')

        # Compute per-pair scoring components
        band_decay = pair_df['distance_m'].apply(distance_band_and_decay)
        pair_df['distance_band']         = [t[0] for t in band_decay]
        pair_df['distance_decay_weight'] = [t[1] for t in band_decay]
        pair_df['match_confidence_weight'] = pair_df.apply(
            lambda r: match_confidence_weight(r['match_confidence'], r['line_poi_fraction']), axis=1
        )
        pair_df['anchor_contribution'] = (
            pair_df['base_score']
            * pair_df['voltage_multiplier']
            * pair_df['distance_decay_weight']
            * pair_df['match_confidence_weight']
        )

        # Per candidate: primary anchor = max anchor_contribution
        primary = pair_df.sort_values('anchor_contribution', ascending=False).drop_duplicates(subset=['candidate_id'], keep='first')
        primary = primary.set_index('candidate_id')

        # Standalone hot-node fields (within 25 km only)
        nearby = pair_df[pair_df['distance_m'] <= NEARBY_FOR_NODE]
        if len(nearby) > 0:
            nearby_grp = nearby.groupby('candidate_id').agg(
                node_activity_score = ('base_score','sum'),
                queue_mw_nearby     = ('total_active_capacity_mw','sum'),
                ia_executed_nearby  = ('queue_status_best', lambda s: (s == 'executed_ia').any()),
                recent_withdrawals_nearby = ('recent_withdrawal_count_12mo','sum'),
                executed_ia_count   = ('queue_status_best', lambda s: (s == 'executed_ia').sum()),
                total_count         = ('queue_status_best','count'),
            )
            nearby_grp['node_activity_score'] = nearby_grp['node_activity_score'].clip(upper=100)
            ia_frac = (nearby_grp['executed_ia_count'] / nearby_grp['total_count'].clip(lower=1)).fillna(0)
            nearby_grp['allocation_or_competitive_heat_flag'] = (
                (nearby_grp['queue_mw_nearby'] > 500) & (ia_frac < 0.3)
            )
        else:
            nearby_grp = pd.DataFrame()

        anchors_count = pair_df.groupby('candidate_id').size().rename('num_anchors_in_range')

        for cid in sd.candidate_id.values:
            n_in_range = int(anchors_count.get(cid, 0))
            if n_in_range == 0:
                summary_rows.append({
                    'candidate_id': cid,
                    'num_anchors_in_range': 0, 'zone_fallback_used': True,
                    'primary_anchor_name': None, 'primary_anchor_distance_m': None,
                    'primary_anchor_voltage_kv': None,
                    'primary_anchor_queue_mw_tier': None,
                    'primary_anchor_queue_status_score': None,
                    'primary_anchor_activation_band': None,
                    'primary_anchor_match_confidence': None,
                    'primary_anchor_distance_band': None,
                    'node_activity_score': 0.0, 'queue_mw_nearby': 0.0,
                    'ia_executed_nearby': False, 'recent_withdrawals_nearby': 0,
                    'allocation_or_competitive_heat_flag': False,
                })
                continue
            pr = primary.loc[cid]
            ng = nearby_grp.loc[cid] if cid in nearby_grp.index else None
            summary_rows.append({
                'candidate_id': cid,
                'num_anchors_in_range': n_in_range,
                'zone_fallback_used': False,
                'primary_anchor_name':          pr['anchor_name'],
                'primary_anchor_distance_m':    float(pr['distance_m']),
                'primary_anchor_voltage_kv':    float(pr['anchor_voltage_kv']) if pd.notna(pr['anchor_voltage_kv']) else None,
                'primary_anchor_queue_mw_tier': pr['queue_mw_tier'],
                'primary_anchor_queue_status_score': float(pr['queue_status_score']),
                'primary_anchor_activation_band':   pr['activation_band'],
                'primary_anchor_match_confidence':  pr['match_confidence'],
                'primary_anchor_distance_band':     pr['distance_band'],
                'node_activity_score':              float(ng['node_activity_score']) if ng is not None else 0.0,
                'queue_mw_nearby':                  float(ng['queue_mw_nearby']) if ng is not None else 0.0,
                'ia_executed_nearby':               bool(ng['ia_executed_nearby']) if ng is not None else False,
                'recent_withdrawals_nearby':        int(ng['recent_withdrawals_nearby']) if ng is not None else 0,
                'allocation_or_competitive_heat_flag': bool(ng['allocation_or_competitive_heat_flag']) if ng is not None else False,
            })

        # Drop anchor_idx column from pairs output
        st_pairs = pair_df.drop(columns=['anchor_idx'])
        pairs_rows.append(st_pairs)

    print('\nConcatenating results ...')
    summary = pd.DataFrame(summary_rows)
    pairs   = pd.concat(pairs_rows, ignore_index=True) if pairs_rows else pd.DataFrame()
    assert len(summary) == len(cands), f'lost candidates: expected {len(cands)}, got {len(summary)}'

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(OUT_SUMMARY, index=False)
    pairs.to_parquet(OUT_PAIRS, index=False)
    print(f'\nSaved: {OUT_SUMMARY} ({OUT_SUMMARY.stat().st_size/1e6:.2f} MB, {len(summary):,} rows)')
    print(f'Saved: {OUT_PAIRS}   ({OUT_PAIRS.stat().st_size/1e6:.2f} MB, {len(pairs):,} (candidate, anchor) pairs)')

    print('\n=== Anchor coverage ===')
    print(f'Candidates with >= 1 anchor in 50 km: {(summary.num_anchors_in_range > 0).sum():,}')
    print(f'Candidates needing zone fallback:     {summary.zone_fallback_used.sum():,}')
    print(f'num_anchors_in_range: min={summary.num_anchors_in_range.min()}, '
          f'median={summary.num_anchors_in_range.median()}, '
          f'p90={summary.num_anchors_in_range.quantile(0.9):.0f}, '
          f'max={summary.num_anchors_in_range.max()}')

    print('\n=== Primary anchor distance distribution ===')
    s = summary.primary_anchor_distance_m.dropna()
    if len(s):
        print(f'  median={s.median():.0f}m, p10={s.quantile(0.1):.0f}, p90={s.quantile(0.9):.0f}, max={s.max():.0f}')

    print('\n=== Primary anchor distance_band distribution ===')
    print(summary.primary_anchor_distance_band.value_counts(dropna=False).to_string())

    print('\n=== node_activity_score distribution ===')
    print(f'  median={summary.node_activity_score.median():.1f}, '
          f'p90={summary.node_activity_score.quantile(0.9):.1f}, '
          f'max={summary.node_activity_score.max():.1f}')

    print('\n=== Hot-node flags ===')
    print(f'  ia_executed_nearby: {summary.ia_executed_nearby.sum():,} ({100*summary.ia_executed_nearby.mean():.1f}%)')
    print(f'  allocation_or_competitive_heat_flag: {summary.allocation_or_competitive_heat_flag.sum():,} ({100*summary.allocation_or_competitive_heat_flag.mean():.1f}%)')

    print('\n=== Checks ===')
    checks = {
        'Has all candidates'              : len(summary) == len(cands),
        'Unique candidate_ids'            : summary.candidate_id.is_unique,
        'Pairs have valid candidate_ids'  : pairs['candidate_id'].isin(summary.candidate_id).all() if len(pairs) else True,
        'Pair distance within 50 km'      : (pairs['distance_m'] <= MAX_SEARCH_M + 1).all() if len(pairs) else True,
        'Pair distance non-negative'      : (pairs['distance_m'] >= 0).all() if len(pairs) else True,
        'anchor_contribution non-neg'     : (pairs['anchor_contribution'] >= 0).all() if len(pairs) else True,
        'zone_fallback eq anchors=0'      : (summary.zone_fallback_used == (summary.num_anchors_in_range == 0)).all(),
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
