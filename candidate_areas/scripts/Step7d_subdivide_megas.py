"""
Step 7d - Subdivide mega-candidate polygons (>= 5,000 acres) into ~2,400-acre fragments.

Approach:
  - Load candidate_areas.parquet (95,269 polygons, pre-enrichment)
  - For each polygon >= MEGA_THRESHOLD_ACRES: intersect with a regular grid
    of CELL_SIZE_M x CELL_SIZE_M cells aligned to EPSG:5070 origin
  - Each non-trivial fragment becomes its own candidate with:
      * new uuid4 candidate_id
      * parent_candidate_id = original UUID
      * inherit state, county, snapshot_date, candidate_type, original CDL group, etc.
      * recompute area_m2, area_acres, pixel_count_estimate from the new geometry
  - Drop fragments < MIN_FRAGMENT_ACRES (slivers)
  - Combine with the un-touched polygons (<5,000 ac) into a new candidate_areas.parquet

Output:
  candidate_areas/outputs/candidate_areas.parquet  (overwritten)
  candidate_areas/outputs/candidate_areas_PRE_SUBDIVIDE.parquet  (backup)

Run:
  python candidate_areas/scripts/Step7d_subdivide_megas.py
"""
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

CAND_PATH = Path('candidate_areas/outputs/candidate_areas.parquet')
BACKUP_PATH = Path('candidate_areas/outputs/candidate_areas_PRE_SUBDIVIDE.parquet')

MEGA_THRESHOLD_ACRES = 5_000.0      # subdivide anything >= this
CELL_SIZE_M = 4000.0                 # 4.0 km square => max ~3,953 acres per cell (cleanly under 5,000 ac threshold)
MIN_FRAGMENT_ACRES = 50.0            # drop slivers below this

ACRES_PER_M2 = 0.000247105


def gen_grid_for_bounds(minx, miny, maxx, maxy, cell):
    """Generate a list of grid cell polygons covering the bounds (aligned to grid origin)."""
    # Snap bounds to grid
    gx0 = np.floor(minx / cell) * cell
    gy0 = np.floor(miny / cell) * cell
    gx1 = np.ceil(maxx / cell) * cell
    gy1 = np.ceil(maxy / cell) * cell

    cells = []
    x = gx0
    while x < gx1:
        y = gy0
        while y < gy1:
            cells.append(box(x, y, x + cell, y + cell))
            y += cell
        x += cell
    return cells


def subdivide_polygon(row, cell):
    """Split one polygon into grid-cell fragments. Returns list of new rows (dict)."""
    geom = row.geometry
    minx, miny, maxx, maxy = geom.bounds
    cells = gen_grid_for_bounds(minx, miny, maxx, maxy, cell)

    out = []
    for c in cells:
        if not c.intersects(geom):
            continue
        frag = c.intersection(geom)
        if frag.is_empty:
            continue
        area_m2 = frag.area
        area_ac = area_m2 * ACRES_PER_M2
        if area_ac < MIN_FRAGMENT_ACRES:
            continue
        new = {col: row[col] for col in row.index if col not in ('geometry',)}
        new['candidate_id'] = str(uuid.uuid4())
        new['parent_candidate_id'] = row['candidate_id']
        new['area_m2'] = float(area_m2)
        new['area_acres'] = float(area_ac)
        # Pixel count estimate (assuming source CDL 30m raster)
        new['pixel_count'] = int(round(area_m2 / 900.0))  # 30 x 30
        # Recompute centroid
        cen = frag.centroid
        # Reproject centroid to 4326 for centroid_lon/lat (assuming source is 5070)
        new['centroid_lon'] = None  # will be filled by enrichment merge if needed
        new['centroid_lat'] = None
        new['geometry'] = frag
        out.append(new)
    return out


def main():
    print(f'Loading {CAND_PATH} ...')
    g = gpd.read_parquet(CAND_PATH)
    n_orig = len(g)
    crs = g.crs
    print(f'  {n_orig:,} candidates in EPSG:{crs.to_epsg()}')
    print(f'  total acreage: {g.area_acres.sum():,.0f}')

    # Backup once
    if not BACKUP_PATH.exists():
        print(f'\nBacking up to {BACKUP_PATH} ...')
        g.to_parquet(BACKUP_PATH, index=False)

    # Add parent_candidate_id column to all (null for un-split polygons)
    g['parent_candidate_id'] = None

    # Split mega-polygons
    mega = g[g.area_acres >= MEGA_THRESHOLD_ACRES].copy()
    keep = g[g.area_acres < MEGA_THRESHOLD_ACRES].copy()
    print(f'\nMega-polygons (>= {MEGA_THRESHOLD_ACRES:,.0f} ac): {len(mega):,}')
    print(f'Untouched polygons (< {MEGA_THRESHOLD_ACRES:,.0f} ac): {len(keep):,}')
    print(f'Mega acreage: {mega.area_acres.sum():,.0f}')
    print(f'Untouched acreage: {keep.area_acres.sum():,.0f}')

    print('\nSubdividing mega-polygons ...')
    fragments = []
    n_processed = 0
    for _, row in mega.iterrows():
        frags = subdivide_polygon(row, CELL_SIZE_M)
        fragments.extend(frags)
        n_processed += 1
        if n_processed % 50 == 0:
            print(f'  ... {n_processed}/{len(mega)} mega-polygons -> {len(fragments):,} fragments so far')
    print(f'  done. {len(mega):,} mega-polygons -> {len(fragments):,} fragments')

    if not fragments:
        print('NO FRAGMENTS PRODUCED. Aborting.')
        return

    # Build the fragments GeoDataFrame
    frag_df = gpd.GeoDataFrame(fragments, geometry='geometry', crs=crs)
    print(f'  fragments acreage total: {frag_df.area_acres.sum():,.0f} '
          f'(vs original mega acreage {mega.area_acres.sum():,.0f})')

    # Combine: keep untouched + fragments
    print('\nCombining untouched + fragments ...')
    combined = gpd.GeoDataFrame(
        pd.concat([keep, frag_df], ignore_index=True),
        geometry='geometry', crs=crs,
    )
    print(f'  new total: {len(combined):,} candidates')

    # Write out (overwrites)
    print(f'\nWriting back to {CAND_PATH} ...')
    combined.to_parquet(CAND_PATH, index=False)

    # ----- Verification -----
    print('\n=== VERIFICATION ===')
    g2 = gpd.read_parquet(CAND_PATH)
    print(f'  rows: {len(g2):,}')
    print(f'  any polygons >= {MEGA_THRESHOLD_ACRES:,.0f} ac: '
          f'{(g2.area_acres >= MEGA_THRESHOLD_ACRES).sum()}')
    print(f'  any below {MIN_FRAGMENT_ACRES:.0f} ac: '
          f'{(g2.area_acres < MIN_FRAGMENT_ACRES).sum()}')
    print(f'  total acreage: {g2.area_acres.sum():,.0f} '
          f'(orig {g.area_acres.sum():,.0f}, diff {g2.area_acres.sum() - g.area_acres.sum():+,.0f})')
    print(f'  acreage preservation: {100 * g2.area_acres.sum() / g.area_acres.sum():.2f}%')
    print(f'  candidate_id unique: {g2.candidate_id.is_unique}')
    print(f'  fragments with parent_candidate_id: '
          f'{g2.parent_candidate_id.notna().sum():,}')
    print(f'  unsplit (no parent): {g2.parent_candidate_id.isna().sum():,}')
    print(f'  by state:')
    print(g2.state.value_counts().to_string())


if __name__ == '__main__':
    main()
