"""
StepR2b -- Clean the reuse-node candidate set before enrichment.

Cleanups in order:
  1. Cross-source dedup: same physical site in multiple sources -- keep the
     richest row per 100m centroid cluster, prefer EIA > Nuclear > EPA > OSM.
  2. Drop LOW geometry_confidence (DEFAULT_BUFFER) -- EPA points with no
     acreage info, fake 200-ac envelopes.
  3. Drop OSM operating power plants -- after dedup, any surviving OSM
     power_plant row has no EIA retired/retiring counterpart, so it's
     active and not a reuse candidate.
  4. Drop EPA rows < 50 acres (any asset_type) -- consistent with the
     candidate pipeline's 50-ac minimum; small urban brownfields, single-pit
     VA NONCOAL mines, and small landfills aren't large-format infra sites.
  5. Drop OSM operating renewable-energy facilities (solar/wind farms named
     in OSM as industrial_land) -- these are active power generators, not
     reusable industrial land.
  6. Drop OSM active military / prison / proving-ground sites by name,
     while preserving decommissioned "Old/Former/Closed" counterparts.

Reads:  candidate_areas/reuse_node_outputs/reuse_nodes_with_footprints.parquet
Writes: candidate_areas/reuse_node_outputs/reuse_nodes_clean.parquet
"""
from pathlib import Path
import re
import pandas as pd
import geopandas as gpd

IN_PATH  = Path('candidate_areas/reuse_node_outputs/reuse_nodes_with_footprints.parquet')
OUT_PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_clean.parquet')

# Source priority for dedup -- higher number wins
SOURCE_RANK = {
    'EIA-860': 4,           # has capacity + retirement year
    'EIA-860-nuclear': 3,   # has capacity, mostly operating but valuable
    'EPA-RE-Powering': 2,   # has acreage + EPA program metadata
    'OpenStreetMap': 1,     # has polygon but no domain metadata
}

MIN_ACRES = 50.0

# OSM names that indicate active renewable power generation -- not reuse land.
RENEWABLE_PAT = re.compile(
    r'solar farm|solar facility|solar plant|solar project|solar generating'
    r'|wind farm|wind energy|wind project|photovoltaic',
    re.I,
)

# OSM names that indicate active federal / military / prison / test sites.
# Anything with Old / Former / Closed / Decommissioned / Abandoned in the name
# stays in (those are legitimate reuse candidates).
ACTIVE_DROP_PAT = re.compile(
    r'\bproving ground'                            # auto test tracks
    r'|\btest track\b'
    r'|aircraft boneyard'                          # AFB 309th AMARG
    r'|\bammunition plant\b|\barsenal\b'
    r'|\bprison\b|correctional (complex|facility|institution)'
    r'|\bnaval (base|station|shipyard)\b'
    r'|\bair force base\b|\bafb\b',
    re.I,
)
KEEP_DECOMMISSIONED_PAT = re.compile(
    r'\b(old|former|closed|decommission(ed)?|abandoned|disused|disposal)\b',
    re.I,
)


def _is_active_excluded(name):
    if not isinstance(name, str):
        return False
    if KEEP_DECOMMISSIONED_PAT.search(name):
        return False
    return bool(ACTIVE_DROP_PAT.search(name))


def _is_active_renewable(name):
    return isinstance(name, str) and bool(RENEWABLE_PAT.search(name))


def main():
    print(f'Loading {IN_PATH} ...')
    g = gpd.read_parquet(IN_PATH)
    n0 = len(g)
    print(f'  {n0:,} rows')

    # ---- Step 1: Spatial dedup (100m centroid clusters, intra- and cross-source) ----
    # Collapse every 100m cluster to one winner row. Within a cluster, prefer:
    #   1) highest SOURCE_RANK (EIA > Nuclear > EPA > OSM)
    #   2) longest site_name (most-complete label, e.g. "HAYWARD CANNERY AREA"
    #      over a single tenant alias)
    #   3) lowest site_id (deterministic tiebreak)
    # Preserve every dropped row's site_id on the winner as aliased_site_ids /
    # aliased_site_count so EPA's per-parcel records are not lost.
    print('\nStep 1: Spatial dedup (100m centroid clusters, intra- and cross-source) ...')
    g_5070 = g.to_crs(5070)
    cents = g_5070.geometry.centroid
    g['_cx'] = cents.x.round(-2).astype(int)
    g['_cy'] = cents.y.round(-2).astype(int)
    g['_cluster'] = list(zip(g['_cx'], g['_cy']))
    g['_src_rank'] = g['source'].map(SOURCE_RANK)
    g['_name_len'] = g['site_name'].fillna('').str.len()

    cluster_sizes = g.groupby('_cluster').size()
    multi_clusters = set(cluster_sizes[cluster_sizes > 1].index)
    multi_source = int(g[g['_cluster'].isin(multi_clusters)].groupby('_cluster').source.nunique().gt(1).sum())
    print(f'  100m clusters with >1 row: {len(multi_clusters):,}  (of which cross-source: {multi_source:,})')

    in_multi = g['_cluster'].isin(multi_clusters)
    multi = g[in_multi].sort_values(
        ['_src_rank', '_name_len', 'site_id'],
        ascending=[False, False, True],
    )
    # Build alias map per cluster -- all dropped site_ids
    alias_map = multi.groupby('_cluster')['site_id'].apply(list).to_dict()
    keep_one = multi.drop_duplicates(subset='_cluster', keep='first').copy()
    # The winner's own site_id should appear in its alias list too (intentional --
    # makes joining back to the raw R1 sources straightforward).
    keep_one['aliased_site_ids'] = keep_one['_cluster'].map(alias_map)
    keep_one['aliased_site_count'] = keep_one['aliased_site_ids'].str.len()

    other = g[~in_multi].copy()
    other['aliased_site_ids'] = other['site_id'].apply(lambda s: [s])
    other['aliased_site_count'] = 1

    g = pd.concat([other, keep_one], ignore_index=True)
    g = g.drop(columns=['_cx', '_cy', '_cluster', '_src_rank', '_name_len'])
    n1 = len(g)
    n_collapsed = n0 - n1
    print(f'  rows dropped (collapsed into a parent): {n_collapsed:,}; remaining: {n1:,}')
    print(f'  rows that now carry >1 alias: {int((g.aliased_site_count > 1).sum()):,}')
    print(f'  max alias count on a single row: {int(g.aliased_site_count.max())}')

    # ---- Step 2: Drop LOW geometry_confidence ----
    print('\nStep 2: Drop LOW geometry_confidence (DEFAULT_BUFFER) ...')
    low_mask = g['geometry_confidence'] == 'LOW'
    print(f'  rows dropped: {int(low_mask.sum()):,}')
    g = g[~low_mask].copy()
    n2 = len(g)
    print(f'  remaining: {n2:,}')

    # ---- Step 3: Drop OSM operating power plants ----
    print('\nStep 3: Drop OSM operating power plants (no EIA retired counterpart) ...')
    osm_power_mask = (g['source'] == 'OpenStreetMap') & (g['reuse_asset_type'] == 'power_plant')
    print(f'  rows dropped: {int(osm_power_mask.sum()):,}')
    g = g[~osm_power_mask].copy()
    n3 = len(g)
    print(f'  remaining: {n3:,}')

    # ---- Step 4: Drop EPA rows below 50 ac (any asset_type) ----
    print('\nStep 4: Drop EPA rows below 50 acres (any asset_type) ...')
    small_epa_mask = (
        (g['source'] == 'EPA-RE-Powering')
        & (g['footprint_acres'] < MIN_ACRES)
    )
    print(f'  rows dropped: {int(small_epa_mask.sum()):,}')
    print(f'    by asset_type: {g.loc[small_epa_mask, "reuse_asset_type"].value_counts().to_dict()}')
    g = g[~small_epa_mask].copy()
    n4 = len(g)
    print(f'  remaining: {n4:,}')

    # ---- Step 5: Drop OSM operating renewable energy ----
    print('\nStep 5: Drop OSM operating renewable-energy facilities ...')
    renew_mask = (g['source'] == 'OpenStreetMap') & g['site_name'].apply(_is_active_renewable)
    print(f'  rows dropped: {int(renew_mask.sum()):,}')
    if renew_mask.any():
        for _, r in g[renew_mask].iterrows():
            print(f'    - {r.site_name} ({r.state_abbr}, {r.footprint_acres:.0f} ac)')
    g = g[~renew_mask].copy()
    n5 = len(g)
    print(f'  remaining: {n5:,}')

    # ---- Step 6: Drop OSM active military / prison / proving (keep Old/Former) ----
    print('\nStep 6: Drop OSM active military / prison / proving sites ...')
    active_mask = (g['source'] == 'OpenStreetMap') & g['site_name'].apply(_is_active_excluded)
    print(f'  rows dropped: {int(active_mask.sum()):,}')
    if active_mask.any():
        for _, r in g[active_mask].iterrows():
            print(f'    - {r.site_name} ({r.state_abbr}, {r.footprint_acres:.0f} ac)')
    g = g[~active_mask].copy()
    n6 = len(g)
    print(f'  remaining: {n6:,}')

    # ---- Save ----
    g = g.reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.0f} KB, {len(g):,} rows)')

    # ---- Summary ----
    print('\n=== Funnel ===')
    print(f'  Start:                  {n0:>6,}')
    print(f'  After cross-src dedup:  {n1:>6,}  (-{n0-n1:,})')
    print(f'  After LOW drop:         {n2:>6,}  (-{n1-n2:,})')
    print(f'  After OSM ops-power:    {n3:>6,}  (-{n2-n3:,})')
    print(f'  After EPA <50 ac drop:  {n4:>6,}  (-{n3-n4:,})')
    print(f'  After OSM renewables:   {n5:>6,}  (-{n4-n5:,})')
    print(f'  After OSM active fed:   {n6:>6,}  (-{n5-n6:,})')
    print(f'  Total dropped:          {n0-n6:>6,} ({100*(n0-n6)/n0:.1f}%)')

    print('\n=== Final by source ===')
    print(g.source.value_counts().to_string())
    print('\n=== Final by reuse_asset_type ===')
    print(g.reuse_asset_type.value_counts().to_string())
    print('\n=== Final by geometry_confidence ===')
    print(g.geometry_confidence.value_counts().to_string())
    print('\n=== Final by reuse_status ===')
    print(g.reuse_status.value_counts(dropna=False).to_string())
    print('\n=== Final per-state x source ===')
    print(g.groupby(['state_abbr', 'source']).size().unstack(fill_value=0).to_string())
    print('\n=== Footprint acreage distribution ===')
    a = g.footprint_acres
    print(f'  min={a.min():.0f}, median={a.median():.0f}, '
          f'p90={a.quantile(0.9):.0f}, max={a.max():.0f}')

    # ---- Checks ----
    print('\n=== Checks ===')
    checks = {
        'Has rows'                   : len(g) > 0,
        'site_id unique'             : g.site_id.is_unique,
        'no LOW confidence remaining': not (g.geometry_confidence == 'LOW').any(),
        'no OSM power_plant remaining': not ((g.source == 'OpenStreetMap') & (g.reuse_asset_type == 'power_plant')).any(),
        'no EPA <50ac (any type)'    : not ((g.source == 'EPA-RE-Powering') & (g.footprint_acres < MIN_ACRES)).any(),
        'no OSM renewables remaining': not ((g.source == 'OpenStreetMap') & g.site_name.apply(_is_active_renewable)).any(),
        'no OSM active military/prison/proving': not ((g.source == 'OpenStreetMap') & g.site_name.apply(_is_active_excluded)).any(),
        'CRS 4326'                   : g.crs.to_epsg() == 4326,
        'geometry valid'             : g.geometry.is_valid.all(),
    }
    ok = True
    for lbl, passed in checks.items():
        print(f'  [{"PASS" if passed else "FAIL"}] {lbl}')
        if not passed:
            ok = False
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
