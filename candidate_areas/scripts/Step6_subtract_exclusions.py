"""
Step 6 - Subtract exclusion mask from keep_zones using buffered fishnet.

Expert GIS fixes applied:
  1. set_precision(1.0) + make_valid on all geometries upfront (once per state)
  2. 500m buffered processing window per cell, clip result back to original cell
  3. Direct shapely .difference() instead of gpd.overlay()
  4. Per-cell mask union (NOT global state union — avoids hanging on large NWI)
  5. buffer(0) fallback for residual TopologyException cases

Inputs:
  candidate_areas/outputs/keep_zones.parquet
  candidate_areas/outputs/exclusion_mask.parquet
  ingestion_scripts/census_tiger/state_boundaries.parquet

Outputs:
  candidate_areas/outputs/keep_zones_final.parquet
  candidate_areas/outputs/keep_zones_final.fgb

Run:  python candidate_areas/Step6_subtract_exclusions.py
"""

import gc
import geopandas as gpd
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from shapely import from_wkb, set_precision
from shapely.geometry import box, GeometryCollection
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid
from pathlib import Path

KEEP_PATH   = Path('candidate_areas/outputs/keep_zones.parquet')
MASK_PATH   = Path('candidate_areas/outputs/exclusion_mask.parquet')
STATES_PATH = Path('ingestion_scripts/census_tiger/state_boundaries.parquet')
OUT_PARQUET = Path('candidate_areas/outputs/keep_zones_final.parquet')
OUT_FGB     = Path('candidate_areas/outputs/keep_zones_final.fgb')

STATES    = ['AZ', 'CA', 'TX', 'NV', 'VA']
CELL_SIZE = 25_000   # 25 km
BUFFER    = 500      # metres — processing window expansion
GRID_SIZE = 1.0      # metres — set_precision snap grid

# ---- Helpers ----------------------------------------------------------------

def snap(geom, grid=GRID_SIZE):
    try:
        return make_valid(set_precision(geom, grid))
    except Exception:
        # Large geometries (e.g. TX NWI unions) can OOM during set_precision.
        # Fall back to make_valid only — topology still stabilised at cell level.
        return make_valid(geom)

def extract_polys(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in ('Polygon', 'MultiPolygon'):
        return geom
    if isinstance(geom, GeometryCollection):
        parts = [g for g in geom.geoms
                 if g.geom_type in ('Polygon', 'MultiPolygon') and not g.is_empty]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return make_valid(unary_union(parts))
    return None

def make_fishnet(xmin, ymin, xmax, ymax, cell_size):
    xs = np.arange(xmin, xmax, cell_size)
    ys = np.arange(ymin, ymax, cell_size)
    return [box(x, y, x + cell_size, y + cell_size)
            for x in xs for y in ys]

# ---- Load inputs ------------------------------------------------------------

print('Loading keep_zones ...')
keep_gdf = gpd.read_parquet(KEEP_PATH)
area_before = keep_gdf.geometry.area.sum() / 1e6
print(f'  1 row | {area_before:,.0f} km2 | CRS: EPSG:{keep_gdf.crs.to_epsg()}')

print('Loading state boundaries ...')
states_gdf = (gpd.read_parquet(STATES_PATH)
              .query('state_code in @STATES')
              .to_crs('EPSG:5070'))

print('Opening exclusion_mask via PyArrow ...')
pf = pq.ParquetFile(MASK_PATH)
print(f'  {pf.metadata.num_rows} rows | {pf.metadata.num_row_groups} row groups')

# ---- Per-state processing ---------------------------------------------------

all_results   = []
total_removed = 0.0

for state in STATES:
    print(f'\n{"="*55}')
    print(f' {state}')
    print(f'{"="*55}')

    state_geom = states_gdf.loc[states_gdf['state_code'] == state, 'geometry'].iloc[0]
    sxmin, symin, sxmax, symax = state_geom.bounds

    # Clip keep_zones to state, apply set_precision upfront
    keep_state = gpd.clip(keep_gdf[['geometry']], state_geom)
    keep_state = keep_state[keep_state.geometry.is_valid &
                            ~keep_state.geometry.is_empty].copy()
    area_state = keep_state.geometry.area.sum() / 1e6
    print(f'  keep_zones: {len(keep_state)} features | {area_state:,.0f} km2')

    if len(keep_state) == 0:
        print(f'  No keep_zones in {state} — skipping.')
        continue

    # Apply set_precision to keep geometry upfront
    keep_geom = snap(unary_union(keep_state.geometry.values))
    keep_geom = extract_polys(keep_geom)
    if keep_geom is None:
        print(f'  keep_geom empty — skipping {state}.')
        continue

    # Stream + snap exclusion_mask rows for this state
    # Keep as list of snapped geometries — NO global union here
    print(f'  Streaming + snapping exclusion_mask for {state} ...')
    state_mask_geoms = []
    for gi in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(gi, columns=['geometry'])
        for ri in range(len(tbl)):
            wkb = tbl.column('geometry')[ri].as_py()
            if wkb is None:
                continue
            geom = from_wkb(wkb)
            del wkb
            gxmin, gymin, gxmax, gymax = geom.bounds
            if gxmax < sxmin or gxmin > sxmax or gymax < symin or gymin > symax:
                del geom
                continue
            clipped = geom.intersection(state_geom)
            del geom
            if clipped is None or clipped.is_empty:
                continue
            snapped = snap(clipped)
            if not snapped.is_empty:
                state_mask_geoms.append(snapped)
        del tbl

    if not state_mask_geoms:
        print(f'  No exclusions in {state} — keeping as-is.')
        all_results.append(keep_state)
        continue

    mask_area_total = sum(g.area for g in state_mask_geoms) / 1e6
    print(f'  {len(state_mask_geoms)} snapped mask rows | ~{mask_area_total:,.0f} km2')

    # Build STRtree for O(log n) mask lookups per cell
    mask_tree = STRtree(state_mask_geoms)
    print(f'  STRtree built over {len(state_mask_geoms)} mask geometries')

    # Build fishnet
    cells = make_fishnet(sxmin, symin, sxmax, symax, CELL_SIZE)
    print(f'  Fishnet: {len(cells)} cells @ {CELL_SIZE/1000:.0f}km | buffer={BUFFER}m')

    cell_results = []
    n_empty  = 0
    n_done   = 0
    n_fixed  = 0
    n_failed = 0

    for ci, cell_box in enumerate(cells):
        expanded = cell_box.buffer(BUFFER)
        ex1, ey1, ex2, ey2 = expanded.bounds
        kx1, ky1, kx2, ky2 = keep_geom.bounds

        if ex2 < kx1 or ex1 > kx2 or ey2 < ky1 or ey1 > ky2:
            n_empty += 1
            continue

        # Clip keep to expanded cell
        keep_local = keep_geom.intersection(expanded)
        if keep_local is None or keep_local.is_empty:
            n_empty += 1
            continue
        keep_local = make_valid(keep_local)

        # Use STRtree to find only nearby mask geoms — O(log n) not O(n)
        mask_local_parts = []
        candidate_indices = mask_tree.query(expanded)
        for idx in candidate_indices:
            mg = state_mask_geoms[idx]
            mc = mg.intersection(expanded)
            if mc and not mc.is_empty:
                mask_local_parts.append(make_valid(mc))

        if not mask_local_parts:
            # No mask in this cell — clip keep back to original cell
            result = extract_polys(make_valid(keep_local.intersection(cell_box)))
            if result and not result.is_empty:
                cell_results.append(result)
            n_done += 1
            continue

        # Union mask parts within cell (cheap — already clipped to 25km cell)
        mask_local = make_valid(unary_union(mask_local_parts))

        # Difference with precision retry + buffer(0) fallback
        result = None
        attempts = [(1.0, False), (5.0, False), (10.0, False),
                    (1.0, True),  (10.0, True)]

        for grid, use_buf0 in attempts:
            try:
                k = make_valid(set_precision(keep_local, grid))
                m = make_valid(set_precision(mask_local, grid))
                if use_buf0:
                    k = make_valid(k.buffer(0))
                    m = make_valid(m.buffer(0))
                diff = make_valid(k.difference(m))
                diff = make_valid(diff.intersection(cell_box))
                result = extract_polys(diff)
                if use_buf0:
                    n_fixed += 1
                    print(f'    [buf0-fix] cell {ci} fixed at {grid}m precision')
                break
            except Exception as e:
                if (grid, use_buf0) == (10.0, True):
                    print(f'    [FAILED]   cell {ci} — {e}')
                    n_failed += 1
                continue

        if result is None:
            result = extract_polys(make_valid(keep_local.intersection(cell_box)))

        if result and not result.is_empty:
            cell_results.append(result)

        n_done += 1
        n_active = len(cells) - n_empty  # approx active count
        pct = n_done / max(n_active, 1) * 100
        if n_done % 50 == 0:
            print(f'    [{pct:5.1f}%] {n_done} cells done | {n_fixed} buf0-fixed | {n_failed} failed')
        if n_done % 100 == 0:
            gc.collect()

    print(f'  Cells: {n_done} processed | {n_empty} empty | '
          f'{n_fixed} buf0-fixed | {n_failed} failed')

    if not cell_results:
        print(f'  WARNING: no results for {state}')
        continue

    # Keep cell results as individual rows — no expensive unary_union needed.
    # Expert rec: drop slivers < 1000 m2 (precision dust from set_precision/clipping).
    del state_mask_geoms
    state_gdf = gpd.GeoDataFrame(
        geometry=[g for g in cell_results if g is not None and not g.is_empty],
        crs='EPSG:5070')
    del cell_results
    state_gdf = state_gdf[
        state_gdf.geometry.is_valid &
        ~state_gdf.geometry.is_empty &
        (state_gdf.geometry.area > 1000)   # drop slivers < 1000 m2
    ].copy()

    area_after  = state_gdf.geometry.area.sum() / 1e6
    removed_km2 = area_state - area_after
    total_removed += removed_km2
    pct = removed_km2 / area_state * 100 if area_state > 0 else 0
    print(f'\n  {state}: input={area_state:,.0f} removed={removed_km2:,.0f} ({pct:.1f}%) '
          f'kept={area_after:,.0f} km2 | {n_failed} failed cells')

    all_results.append(state_gdf)
    del state_gdf, keep_geom

# ---- Assemble + save --------------------------------------------------------

print('\nAssembling keep_zones_final ...')
result_gdf = gpd.GeoDataFrame(
    pd.concat(all_results, ignore_index=True), crs='EPSG:5070')
result_gdf = result_gdf[result_gdf.geometry.is_valid &
                        ~result_gdf.geometry.is_empty].copy()
area_final = result_gdf.geometry.area.sum() / 1e6

print('Saving parquet ...')
result_gdf.to_parquet(OUT_PARQUET, index=False)
print(f'Saved: {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e6:.0f} MB)')

print('Writing FlatGeobuf ...')
result_gdf.to_file(str(OUT_FGB), driver='FlatGeobuf')
print(f'Saved: {OUT_FGB} ({OUT_FGB.stat().st_size/1e6:.0f} MB)')

# ---- Summary + checks -------------------------------------------------------

print(f'\n=== SUBTRACTION SUMMARY ===')
print(f'  Input  (keep_zones):       {area_before:,.0f} km2')
print(f'  Removed:                   {total_removed:,.0f} km2 ({total_removed/area_before*100:.1f}%)')
print(f'  Output (keep_zones_final): {area_final:,.0f} km2')
print(f'  Rows:                      {len(result_gdf)}')

checks = {
    'Has rows'         : len(result_gdf) > 0,
    'CRS EPSG:5070'    : result_gdf.crs.to_epsg() == 5070,
    'No null geometry' : result_gdf.geometry.isna().sum() == 0,
    'No empty geometry': (~result_gdf.geometry.is_empty).all(),
    'Area reduced'     : area_final < area_before,
    'Area > 0'         : area_final > 0,
    'Area > 400k km2'  : area_final > 400_000,
}

print('\n=== CHECKS ===')
all_pass = True
for lbl, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
    if not ok:
        all_pass = False

print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
print('\nVerify keep_zones_final.fgb in QGIS — no fishnet seam artifacts.')
