"""
StepR3I -- Utility anchor lookup per reuse node.

Mirror of Step1I_utility_anchors.py for reuse_nodes_clean.parquet.

Outputs two parquets identical in schema to Step1I:
  stepR3i_utility_summary.parquet
  stepR3i_utility_anchor_pairs.parquet

Run: python candidate_areas/reuse_node_scripts/StepR3I_utility_anchors.py
"""
from pathlib import Path
import sys
import time
import pandas as pd
import geopandas as gpd
from shapely import STRtree
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent))
from _r3_helpers import load_reuse_nodes_as_candidates, out_path

ANCHOR_PATH = Path('ingestion_scripts/gridstatus_iso_queues/anchor_queue_stats.parquet')
OUT_SUMMARY = out_path('stepR3i_utility_summary.parquet')
OUT_PAIRS   = out_path('stepR3i_utility_anchor_pairs.parquet')

MAX_SEARCH_M    = 50_000.0
NEARBY_FOR_NODE = 25_000.0
STATES = ['AZ','CA','NV','TX','VA']

QUEUE_MW_TIER_SCORE = {
    'negligible': 0.0, 'weak': 25.0, 'moderate': 50.0,
    'strong': 75.0, 'very_strong': 100.0,
}
ACTIVATION_WEIGHT = {
    'le_18': 1.10, '18_36': 1.00, '36_60': 0.90, 'gt_60': 0.75, None: 1.00,
}


def voltage_multiplier(kv):
    if pd.isna(kv): return 1.0
    if kv >= 500: return 1.20
    if kv >= 345: return 1.10
    if kv >= 230: return 1.05
    return 1.00


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


def _empty_row(cid):
    return {
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
    }


def main():
    print('Loading reuse nodes ...')
    cands = load_reuse_nodes_as_candidates(crs_epsg=5070)
    print(f'  {len(cands):,} reuse nodes')

    print(f'\nLoading anchors: {ANCHOR_PATH} ...')
    anchors = gpd.read_parquet(ANCHOR_PATH)
    if anchors.crs.to_epsg() != 5070:
        anchors = anchors.to_crs(5070)
    print(f'  {len(anchors):,} anchors')

    anchors['queue_mw_tier_score']    = anchors.queue_mw_tier.map(QUEUE_MW_TIER_SCORE).fillna(0.0)
    anchors['voltage_multiplier']     = anchors.anchor_voltage_kv.apply(voltage_multiplier)
    anchors['activation_band_weight'] = anchors.activation_band.map(ACTIVATION_WEIGHT).fillna(1.0)
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
        print(f'\n-- {st} : {len(sd):,} reuse nodes -----------')

        bb = sd.total_bounds
        pad = 60_000
        env = box(bb[0]-pad, bb[1]-pad, bb[2]+pad, bb[3]+pad)
        st_anchors = anchors[anchors.geometry.intersects(env)].reset_index(drop=True)
        print(f'  Anchors in envelope: {len(st_anchors):,}')

        if len(st_anchors) == 0:
            for cid in sd.candidate_id.values:
                summary_rows.append(_empty_row(cid))
            continue

        anchor_geoms = st_anchors.geometry.values
        tree = STRtree(anchor_geoms)

        print(f'  Per-reuse-node bbox prefilter + exact distance ...', flush=True)
        t0 = time.time()
        pair_records = []
        log_every = max(1, len(sd) // 10)
        for i, row in enumerate(sd.itertuples(index=False)):
            geom = row.geometry
            cid  = row.candidate_id
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
                print(f'    {i+1:,}/{len(sd):,} ({100*(i+1)/len(sd):.0f}%) '
                      f'{rate:.0f}/s ETA {eta:.0f}s', flush=True)
        print(f'    {time.time()-t0:.1f}s ({len(pair_records):,} pairs)')

        if not pair_records:
            for cid in sd.candidate_id.values:
                summary_rows.append(_empty_row(cid))
            continue

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

        band_decay = pair_df['distance_m'].apply(distance_band_and_decay)
        pair_df['distance_band']           = [t[0] for t in band_decay]
        pair_df['distance_decay_weight']   = [t[1] for t in band_decay]
        pair_df['match_confidence_weight'] = pair_df.apply(
            lambda r: match_confidence_weight(r['match_confidence'], r['line_poi_fraction']), axis=1
        )
        pair_df['anchor_contribution'] = (
            pair_df['base_score']
            * pair_df['voltage_multiplier']
            * pair_df['distance_decay_weight']
            * pair_df['match_confidence_weight']
        )

        primary = pair_df.sort_values('anchor_contribution', ascending=False).drop_duplicates(subset=['candidate_id'], keep='first')
        primary = primary.set_index('candidate_id')

        nearby = pair_df[pair_df['distance_m'] <= NEARBY_FOR_NODE]
        if len(nearby) > 0:
            nearby_grp = nearby.groupby('candidate_id').agg(
                node_activity_score=('base_score','sum'),
                queue_mw_nearby=('total_active_capacity_mw','sum'),
                ia_executed_nearby=('queue_status_best', lambda s: (s == 'executed_ia').any()),
                recent_withdrawals_nearby=('recent_withdrawal_count_12mo','sum'),
                executed_ia_count=('queue_status_best', lambda s: (s == 'executed_ia').sum()),
                total_count=('queue_status_best','count'),
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
                summary_rows.append(_empty_row(cid))
                continue
            pr = primary.loc[cid]
            ng = nearby_grp.loc[cid] if cid in nearby_grp.index else None
            summary_rows.append({
                'candidate_id': cid,
                'num_anchors_in_range': n_in_range,
                'zone_fallback_used': False,
                'primary_anchor_name':              pr['anchor_name'],
                'primary_anchor_distance_m':        float(pr['distance_m']),
                'primary_anchor_voltage_kv':        float(pr['anchor_voltage_kv']) if pd.notna(pr['anchor_voltage_kv']) else None,
                'primary_anchor_queue_mw_tier':     pr['queue_mw_tier'],
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

        st_pairs = pair_df.drop(columns=['anchor_idx'])
        pairs_rows.append(st_pairs)

    summary = pd.DataFrame(summary_rows)
    pairs = pd.concat(pairs_rows, ignore_index=True) if pairs_rows else pd.DataFrame()
    assert len(summary) == len(cands)

    summary.to_parquet(OUT_SUMMARY, index=False)
    pairs.to_parquet(OUT_PAIRS, index=False)
    print(f'\nSaved: {OUT_SUMMARY} ({OUT_SUMMARY.stat().st_size/1e6:.2f} MB, {len(summary):,} rows)')
    print(f'Saved: {OUT_PAIRS}   ({OUT_PAIRS.stat().st_size/1e6:.2f} MB, {len(pairs):,} pairs)')

    print('\n=== Coverage ===')
    print(f'  Reuse nodes with >=1 anchor in 50 km: {(summary.num_anchors_in_range > 0).sum():,}')
    print(f'  Reuse nodes needing zone fallback:    {int(summary.zone_fallback_used.sum()):,}')
    print(f'  num_anchors_in_range: median={summary.num_anchors_in_range.median()}, '
          f'p90={summary.num_anchors_in_range.quantile(0.9):.0f}, '
          f'max={summary.num_anchors_in_range.max()}')

    print('\n=== Checks ===')
    checks = {
        'Has all reuse nodes'         : len(summary) == len(cands),
        'Unique candidate_ids'        : summary.candidate_id.is_unique,
        'Pair distance within 50 km'  : (pairs['distance_m'] <= MAX_SEARCH_M + 1).all() if len(pairs) else True,
        'Pair distance non-neg'       : (pairs['distance_m'] >= 0).all() if len(pairs) else True,
        'anchor_contribution non-neg' : (pairs['anchor_contribution'] >= 0).all() if len(pairs) else True,
        'zone_fallback eq anchors=0'  : (summary.zone_fallback_used == (summary.num_anchors_in_range == 0)).all(),
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
