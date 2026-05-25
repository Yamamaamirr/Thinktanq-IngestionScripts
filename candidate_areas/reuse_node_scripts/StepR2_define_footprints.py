"""
StepR2 -- Define a polygon footprint for every reuse node, with provenance.

Footprint hierarchy (revised after QGIS QA showed convex-hull over-grab):
  - Real OSM polygon (HIGH)        -- row already a polygon
  - OSM polygon match (HIGH)       -- point falls INSIDE an OSM industrial poly
                                      -> adopt that exact polygon
  - Acreage buffer (MEDIUM)        -- square envelope sized to EPA acreage
  - Capacity template (MEDIUM)     -- square sized by fuel-type / capacity
  - Default buffer (LOW)           -- 200 ac square; cleaned out in R2b

No more convex-hull of nearby OSM polygons. The earlier 2 km hull approach
generated giant triangles that filled in residential land between separated
industrial parcels AND produced duplicate hulls when multiple EPA points
sat near the same OSM cluster (overlap factor reached ~12x in dense areas
like Dallas). Real adjacent OSM polygons are already in the dataset on
their own rows; we don't wrap them.

Reads:
  candidate_areas/reuse_node_outputs/reuse_nodes_raw.parquet  (R1 output)
  ingestion_scripts/osm_industrial/osm_industrial_sites.parquet

Writes:
  candidate_areas/reuse_node_outputs/reuse_nodes_with_footprints.parquet

Output adds:
  geometry_source       OSM_POLYGON | OSM_POLYGON_MATCH | BUFFER_FROM_ACREAGE
                        | BUFFER_FROM_CAPACITY | DEFAULT_BUFFER
  geometry_confidence   HIGH | MEDIUM | LOW
  footprint_acres       polygon area in acres
  matched_osm_id        the OSM polygon adopted (only for OSM_POLYGON_MATCH)

Run: python candidate_areas/reuse_node_scripts/StepR2_define_footprints.py
"""
from pathlib import Path
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

R1_PATH  = Path('candidate_areas/reuse_node_outputs/reuse_nodes_raw.parquet')
OSM_PATH = Path('ingestion_scripts/osm_industrial/osm_industrial_sites.parquet')
OUT_PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_with_footprints.parquet')

ACRES_PER_M2 = 0.000247105
M2_PER_ACRE  = 4046.856

# Cap any envelope to the candidate-pipeline site-scale limit
MAX_REUSE_ACRES = 5000.0

# Default fallback envelope when nothing else is known
DEFAULT_BUFFER_ACRES = 200.0


# ---- Capacity-based footprint templates (acres) per fuel type / technology --
#  Sourced from GIS-expert rule-of-thumb: simple-cycle gas 10-40, CCGT 40-150,
#  coal 300-2000+, nuclear 500-4000. Defaults are mid-range; coal/nuclear scale
#  with capacity to handle very large or small plants.

def _capacity_template_acres(asset_type, dominant_tech, capacity_mw):
    tech = (dominant_tech or '').lower()
    cap = capacity_mw or 0
    if asset_type == 'nuclear':
        # Nuclear sites: ~800 ac small to ~3000 ac large
        return max(800.0, min(3000.0, 0.5 * cap + 600.0))
    if asset_type == 'coal_plant':
        # Coal: scale from ~300 (small) to ~2000 (large 2000+ MW)
        return max(300.0, min(2000.0, 0.5 * cap + 200.0))
    if asset_type == 'gas_plant':
        if 'combined cycle' in tech or 'steam turbine' in tech:
            # CCGT and older gas steam: 40-150 ac range
            return max(40.0, min(150.0, 0.1 * cap + 40.0))
        # Simple-cycle / combustion turbine / internal combustion: 10-40 ac
        return max(10.0, min(40.0, 0.05 * cap + 10.0))
    return None  # not a power plant -- caller falls through to default


def _build_square_envelope(point_geom_5070, acres):
    """Square envelope centred on point, area = acres."""
    side_m = math.sqrt(acres * M2_PER_ACRE)
    x, y = point_geom_5070.x, point_geom_5070.y
    return box(x - side_m / 2, y - side_m / 2, x + side_m / 2, y + side_m / 2)


def main():
    print(f'Loading R1 output: {R1_PATH} ...')
    raw = gpd.read_parquet(R1_PATH)
    print(f'  {len(raw):,} reuse-node rows')

    # Working CRS for area / distance ops
    raw_5070 = raw.to_crs(5070).reset_index(drop=True)

    print(f'\nLoading OSM polygons for 2 km neighbour search: {OSM_PATH} ...')
    osm = gpd.read_parquet(OSM_PATH)
    if osm.crs is None or osm.crs.to_epsg() != 5070:
        osm = osm.to_crs(5070)
    print(f'  {len(osm):,} OSM polygons')
    osm_sindex = osm.sindex

    # OSM id column for matched_osm_id provenance
    osm_id_col = 'osm_id' if 'osm_id' in osm.columns else osm.columns[0]
    osm_id_vals = osm[osm_id_col].values

    # Output columns we'll fill row by row
    geom_out = []
    src_out = []
    conf_out = []
    matched_id_out = []
    n_already_poly = 0
    n_osm_match = 0
    n_buffer_acres = 0
    n_buffer_capacity = 0
    n_default = 0

    for i, row in raw_5070.iterrows():
        g = row.geometry

        # ---- Case 1: row already has a polygon (OSM source) ----
        if g.geom_type in ('Polygon', 'MultiPolygon'):
            geom_out.append(g)
            src_out.append('OSM_POLYGON')
            conf_out.append('HIGH')
            matched_id_out.append(None)
            n_already_poly += 1
            continue

        # ---- Case 2: point -- adopt the OSM polygon it falls inside ----
        # Strict containment only. No convex hull, no nearby dissolve.
        # If a real OSM industrial polygon contains the EIA/EPA point, that
        # IS the reuse footprint. If not, fall through to acreage / capacity.
        cand_idx = list(osm_sindex.intersection(g.bounds))
        matched_geom = None
        matched_id = None
        if cand_idx:
            nearby = osm.iloc[cand_idx]
            contains = nearby.geometry.contains(g)
            if contains.any():
                hits = nearby[contains]
                # If point sits inside multiple overlapping OSM polys, pick the
                # smallest -- it's the most specific industrial parcel.
                hits = hits.assign(_a=hits.geometry.area).sort_values('_a')
                matched_geom = hits.iloc[0].geometry
                matched_id = osm_id_vals[hits.index[0]] if hasattr(hits.index[0], '__index__') else hits.iloc[0].get(osm_id_col)
        if matched_geom is not None:
            area_ac = matched_geom.area * ACRES_PER_M2
            if area_ac <= MAX_REUSE_ACRES:
                geom_out.append(matched_geom)
                src_out.append('OSM_POLYGON_MATCH')
                conf_out.append('HIGH')
                matched_id_out.append(matched_id)
                n_osm_match += 1
                continue

        # ---- Case 3: point with known acreage (mostly EPA) ----
        if pd.notna(row.acreage) and row.acreage > 0:
            envelope = _build_square_envelope(g, min(row.acreage, MAX_REUSE_ACRES))
            geom_out.append(envelope)
            src_out.append('BUFFER_FROM_ACREAGE')
            conf_out.append('MEDIUM')
            matched_id_out.append(None)
            n_buffer_acres += 1
            continue

        # ---- Case 4: point with known capacity (EIA / nuclear) ----
        tmpl = _capacity_template_acres(row.reuse_asset_type,
                                        row.dominant_technology,
                                        row.capacity_mw)
        if tmpl is not None:
            envelope = _build_square_envelope(g, min(tmpl, MAX_REUSE_ACRES))
            geom_out.append(envelope)
            src_out.append('BUFFER_FROM_CAPACITY')
            conf_out.append('MEDIUM')
            matched_id_out.append(None)
            n_buffer_capacity += 1
            continue

        # ---- Case 5: nothing known -- default envelope ----
        envelope = _build_square_envelope(g, DEFAULT_BUFFER_ACRES)
        geom_out.append(envelope)
        src_out.append('DEFAULT_BUFFER')
        conf_out.append('LOW')
        matched_id_out.append(None)
        n_default += 1

    # Assemble output (still in EPSG:5070 here)
    out = raw_5070.copy()
    out['geometry'] = geom_out
    out['geometry_source'] = src_out
    out['geometry_confidence'] = conf_out
    out['matched_osm_id'] = matched_id_out
    out['footprint_acres'] = out.geometry.area * ACRES_PER_M2

    # Back to EPSG:4326 for consistency with other sources
    out = out.to_crs(4326)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.0f} KB, {len(out):,} rows)')

    # ---- Summary ----
    print('\n=== Footprint resolution by geometry_source ===')
    print(out.geometry_source.value_counts().to_string())
    print('\n=== Confidence breakdown ===')
    print(out.geometry_confidence.value_counts().to_string())
    print('\n=== Geometry_source by original source ===')
    print(out.groupby(['source', 'geometry_source']).size().unstack(fill_value=0).to_string())
    print('\n=== Footprint acreage distribution ===')
    a = out.footprint_acres
    print(f'  min={a.min():.0f}, median={a.median():.0f}, '
          f'p90={a.quantile(0.9):.0f}, max={a.max():.0f}')

    # ---- Checks ----
    print('\n=== Checks ===')
    valid_src = {'OSM_POLYGON', 'OSM_POLYGON_MATCH', 'BUFFER_FROM_ACREAGE',
                 'BUFFER_FROM_CAPACITY', 'DEFAULT_BUFFER'}
    valid_conf = {'HIGH', 'MEDIUM', 'LOW'}
    checks = {
        'Has rows'                   : len(out) > 0,
        'site_id unique'             : out.site_id.is_unique,
        'all geometry are polygons'  : out.geometry.type.isin(['Polygon', 'MultiPolygon']).all(),
        'geometry_source valid'      : set(out.geometry_source.unique()) <= valid_src,
        'geometry_confidence valid'  : set(out.geometry_confidence.unique()) <= valid_conf,
        'footprint_acres > 0'        : (out.footprint_acres > 0).all(),
        'footprint <= MAX_REUSE_ACRES (1% tolerance)': (out.footprint_acres <= MAX_REUSE_ACRES * 1.01).all(),
        'CRS 4326'                   : out.crs.to_epsg() == 4326,
        'geometry valid'             : out.geometry.is_valid.all(),
    }
    ok = True
    for lbl, passed in checks.items():
        print(f'  [{"PASS" if passed else "FAIL"}] {lbl}')
        if not passed:
            ok = False
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
