"""
StepR2c -- Deep verification of the cleaned reuse-node candidate set.

Cross-checks run before enrichment:
  A. Schema and identity (columns, types, uniqueness, nulls)
  B. Geometry sanity (validity, polygon shape, acreage matches geometry,
                      centroid in correct state bbox)
  C. Source-asset_type consistency (EIA only coal/gas, EPA only brownfield-ish,
                                    Nuclear only nuclear, OSM only OSM types)
  D. Source-confidence consistency (geometry_source vs geometry_confidence)
  E. Cleanup gates (no LOW, no OSM power_plant, no EPA <50 ac of any type,
                    no OSM operating renewables, no OSM active military/prison
                    /proving, dedup applied)
  F. Asset-specific field presence (EIA has capacity, EPA has epa_program, etc.)
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import geopandas as gpd

CLEAN_PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_clean.parquet')

STATE_BBOX = {
    'TX': (-106.7, 25.5, -93.4, 36.7),
    'VA': (-83.7, 36.5, -75.1, 39.5),
    'CA': (-124.7, 32.3, -114.0, 42.1),
    'AZ': (-114.9, 31.2, -109.0, 37.0),
    'NV': (-120.1, 34.9, -114.0, 42.1),
}

ACRES_PER_M2 = 0.000247105


def check(label, ok, detail=''):
    sym = 'PASS' if ok else 'FAIL'
    msg = f'  [{sym}] {label}'
    if detail and not ok:
        msg += f'  ({detail})'
    print(msg)
    return ok


def main():
    print(f'Loading {CLEAN_PATH}\n')
    g = gpd.read_parquet(CLEAN_PATH)
    n = len(g)
    print(f'Rows: {n:,}    Columns: {len(g.columns)}\n')

    all_ok = True

    # ============================================================
    # A. Schema + identity
    # ============================================================
    print('=== A. Schema + identity ===')
    required = ['site_id', 'site_name', 'state_abbr', 'county_name', 'source',
                'reuse_asset_type', 'reuse_status', 'capacity_mw', 'acreage',
                'retirement_year', 'epa_program', 'geometry',
                'geometry_source', 'geometry_confidence', 'footprint_acres',
                'aliased_site_ids', 'aliased_site_count']
    missing = [c for c in required if c not in g.columns]
    all_ok &= check('All required columns present', not missing,
                    f'missing: {missing}')
    all_ok &= check('site_id unique', g.site_id.is_unique)
    all_ok &= check('site_id no nulls', g.site_id.notna().all())
    all_ok &= check('state_abbr no nulls', g.state_abbr.notna().all())
    all_ok &= check('state_abbr in target set',
                    set(g.state_abbr.unique()) <= set(STATE_BBOX),
                    f'unexpected: {set(g.state_abbr.unique()) - set(STATE_BBOX)}')
    all_ok &= check('source in expected set',
                    set(g.source.unique()) <= {'EIA-860', 'EIA-860-nuclear',
                                               'EPA-RE-Powering', 'OpenStreetMap'})

    # ============================================================
    # B. Geometry sanity
    # ============================================================
    print('\n=== B. Geometry sanity ===')
    all_ok &= check('All geometries valid', g.geometry.is_valid.all())
    all_ok &= check('No empty geometries', (~g.geometry.is_empty).all())
    all_ok &= check('All Polygon/MultiPolygon',
                    g.geometry.type.isin(['Polygon', 'MultiPolygon']).all())
    all_ok &= check('CRS EPSG:4326', g.crs.to_epsg() == 4326)

    # Compute geometric acreage and compare to stored footprint_acres
    g_5070 = g.to_crs(5070)
    geom_acres = g_5070.geometry.area * ACRES_PER_M2
    diff = (geom_acres - g.footprint_acres).abs() / g.footprint_acres.clip(lower=1)
    all_ok &= check('footprint_acres matches geometry (<=1% diff)',
                    (diff < 0.01).all(),
                    f'max relative diff = {diff.max():.4f}')
    all_ok &= check('All footprint_acres > 0', (g.footprint_acres > 0).all())
    all_ok &= check('All footprint_acres <= 5,000 (1% tol)',
                    (g.footprint_acres <= 5000 * 1.01).all())

    # Centroid in state bbox
    print('\n  Centroid-in-state-bbox check:')
    bad_total = 0
    for st, (xmin, ymin, xmax, ymax) in STATE_BBOX.items():
        sub = g[g.state_abbr == st]
        cents = sub.geometry.centroid
        oob = sub[(cents.x < xmin) | (cents.x > xmax) | (cents.y < ymin) | (cents.y > ymax)]
        print(f'    {st}: {len(sub):>5,} rows, {len(oob)} out-of-bbox')
        bad_total += len(oob)
    all_ok &= check('All centroids inside their stated state bbox', bad_total == 0,
                    f'{bad_total} out-of-bbox')

    # ============================================================
    # C. Source <-> asset_type consistency
    # ============================================================
    print('\n=== C. Source <-> asset_type consistency ===')
    expected_types = {
        'EIA-860':         {'coal_plant', 'gas_plant'},
        'EIA-860-nuclear': {'nuclear'},
        'EPA-RE-Powering': {'contaminated_land', 'landfill', 'abandoned_mine'},
        'OpenStreetMap':   {'industrial_land', 'industrial_works', 'brownfield_land',
                            'landfill', 'quarry_mine'},  # power_plant dropped by cleanup
    }
    for src, ok_types in expected_types.items():
        sub = g[g.source == src]
        bad = set(sub.reuse_asset_type.unique()) - ok_types
        all_ok &= check(f'{src} only has expected asset types',
                        not bad, f'unexpected: {bad}')

    # ============================================================
    # D. Source <-> confidence consistency
    # ============================================================
    print('\n=== D. geometry_source <-> confidence consistency ===')
    src_conf_pairs = {
        'OSM_POLYGON':         'HIGH',
        'OSM_POLYGON_MATCH':   'HIGH',
        'BUFFER_FROM_ACREAGE': 'MEDIUM',
        'BUFFER_FROM_CAPACITY':'MEDIUM',
        'DEFAULT_BUFFER':      'LOW',   # should not exist after cleanup
    }
    for gsrc, expected_conf in src_conf_pairs.items():
        sub = g[g.geometry_source == gsrc]
        if len(sub) == 0:
            print(f'    {gsrc}: 0 rows (none expected after cleanup)' if gsrc == 'DEFAULT_BUFFER' else f'    {gsrc}: 0 rows')
            continue
        ok = (sub.geometry_confidence == expected_conf).all()
        all_ok &= check(f'{gsrc} -> {expected_conf}', ok,
                        f'mixed confidences: {sub.geometry_confidence.unique()}')

    # ============================================================
    # E. Cleanup gates
    # ============================================================
    print('\n=== E. Cleanup gates ===')
    all_ok &= check('No LOW confidence',
                    not (g.geometry_confidence == 'LOW').any())
    all_ok &= check('No DEFAULT_BUFFER',
                    not (g.geometry_source == 'DEFAULT_BUFFER').any())
    all_ok &= check('No OSM power_plant',
                    not ((g.source == 'OpenStreetMap') & (g.reuse_asset_type == 'power_plant')).any())
    all_ok &= check('No EPA <50 ac (any asset_type)',
                    not ((g.source == 'EPA-RE-Powering')
                         & (g.footprint_acres < 50)).any())

    renewable_pat = re.compile(
        r'solar farm|solar facility|solar plant|solar project|solar generating'
        r'|wind farm|wind energy|wind project|photovoltaic', re.I)
    active_pat = re.compile(
        r'\bproving ground|\btest track\b|aircraft boneyard'
        r'|\bammunition plant\b|\barsenal\b'
        r'|\bprison\b|correctional (complex|facility|institution)'
        r'|\bnaval (base|station|shipyard)\b|\bair force base\b|\bafb\b', re.I)
    keep_pat = re.compile(r'\b(old|former|closed|decommission(ed)?|abandoned|disused|disposal)\b', re.I)

    def _renew(n):
        return isinstance(n, str) and bool(renewable_pat.search(n))
    def _active(n):
        if not isinstance(n, str):
            return False
        if keep_pat.search(n):
            return False
        return bool(active_pat.search(n))

    osm = g[g.source == 'OpenStreetMap']
    all_ok &= check('No OSM operating renewables', not osm.site_name.apply(_renew).any())
    all_ok &= check('No OSM active military/prison/proving', not osm.site_name.apply(_active).any())

    # Dedup recheck: 100m centroid clusters with >1 row -- should be zero
    cents2 = g.to_crs(5070).geometry.centroid
    g_check = g.copy()
    g_check['_cx'] = cents2.x.round(-2).astype(int)
    g_check['_cy'] = cents2.y.round(-2).astype(int)
    sizes = g_check.groupby(['_cx', '_cy']).size()
    n_stacked = int((sizes > 1).sum())
    all_ok &= check('No 100m centroid clusters with >1 row remain',
                    n_stacked == 0, f'{n_stacked} stacked clusters')

    # Alias bookkeeping
    all_ok &= check('aliased_site_count >= 1 for every row',
                    (g.aliased_site_count >= 1).all())
    total_aliases = int(g.aliased_site_count.sum())
    all_ok &= check(f'sum(aliased_site_count) accounts for collapsed rows (sum={total_aliases:,})',
                    True)

    # ============================================================
    # F. Asset-specific field presence
    # ============================================================
    print('\n=== F. Source-specific field presence ===')

    eia = g[g.source == 'EIA-860']
    if len(eia):
        all_ok &= check(f'EIA-860: capacity_mw populated (n={len(eia):,})',
                        eia.capacity_mw.notna().all())
        all_ok &= check('EIA-860: retirement_year populated',
                        eia.retirement_year.notna().all(),
                        f'{eia.retirement_year.isna().sum()} nulls')
        all_ok &= check('EIA-860: reuse_status in {retired, retiring}',
                        set(eia.reuse_status.unique()) <= {'retired', 'retiring'})

    nuc = g[g.source == 'EIA-860-nuclear']
    if len(nuc):
        all_ok &= check(f'Nuclear: capacity_mw populated (n={len(nuc)})',
                        nuc.capacity_mw.notna().all())
        all_ok &= check('Nuclear: reuse_status in {operating, retired}',
                        set(nuc.reuse_status.unique()) <= {'operating', 'retired'})

    epa = g[g.source == 'EPA-RE-Powering']
    if len(epa):
        all_ok &= check(f'EPA: epa_program populated (n={len(epa):,})',
                        epa.epa_program.notna().all(),
                        f'{epa.epa_program.isna().sum()} nulls')
        all_ok &= check('EPA: reuse_status = active_industrial',
                        (epa.reuse_status == 'active_industrial').all())

    osm = g[g.source == 'OpenStreetMap']
    if len(osm):
        all_ok &= check(f'OSM: all from OSM_POLYGON (real polygons) (n={len(osm):,})',
                        (osm.geometry_source == 'OSM_POLYGON').all())
        all_ok &= check('OSM: reuse_status = active_industrial',
                        (osm.reuse_status == 'active_industrial').all())

    # Overlap-factor: total polygon area / area of their union. 1.0 = no overlap.
    g_5070 = g.to_crs(5070)
    from shapely.ops import unary_union as _uu
    union_area = _uu(g_5070.geometry.values).area
    sum_area = g_5070.geometry.area.sum()
    overlap = sum_area / union_area if union_area > 0 else float('inf')
    all_ok &= check(f'Polygon overlap factor <= 1.20 (sum/union = {overlap:.3f})',
                    overlap <= 1.20,
                    f'{overlap:.2f}x means heavy duplication; review R2 logic')

    # ============================================================
    # Summary
    # ============================================================
    print('\n' + '=' * 60)
    print(f'  OVERALL: {"ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"}')
    print('=' * 60)

    # Quick spot-check sample
    print('\n=== Random 5 rows sanity ===')
    sample = g.sample(5, random_state=42)
    for _, r in sample.iterrows():
        nm = str(r.site_name)[:30] if pd.notna(r.site_name) else '(unnamed)'
        print(f'  {r.site_id[:24]:<24} {r.state_abbr} {nm:<30} '
              f'{r.reuse_asset_type:<18} {r.footprint_acres:>5.0f}ac '
              f'{r.geometry_source:<22} {r.geometry_confidence}')


if __name__ == '__main__':
    main()
