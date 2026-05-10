"""
NWI wetland exclusion layer — TX (chunked reader).

TX is 3.2 GB / 1.36M rows — too large to load at once.
Reads in row group chunks, bbox-filters each chunk immediately to
drop wetlands outside keep_zones before accumulating in memory,
then runs precise clip + make_valid on the surviving rows.

Output: candidate_areas/exclusion_layers/exclusion_nwi_tx.parquet
"""

import geopandas as gpd
import pyarrow.parquet as pq
import pandas as pd
from shapely import from_wkb, get_coordinates
from shapely.validation import make_valid
from pathlib import Path

NWI_PATH  = Path('enrichment/usfws_wetlands/data/Tx-wetlands.parquet')
KEEP_PATH = Path('candidate_areas/outputs/keep_zones.parquet')
OUT_PATH  = Path('candidate_areas/exclusion_layers/exclusion_nwi_tx.parquet')
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---- Load keep_zones --------------------------------------------------------

print('Loading keep_zones ...')
keep_gdf  = gpd.read_parquet(KEEP_PATH)
keep_geom = keep_gdf.geometry.iloc[0]
keep_bbox = keep_gdf.total_bounds   # [xmin, ymin, xmax, ymax]
xmin, ymin, xmax, ymax = keep_bbox
print(f'  Area: {keep_geom.area/1e6:,.0f} km2 | CRS: EPSG:{keep_gdf.crs.to_epsg()}')
print(f'  Bbox: ({xmin:.0f}, {ymin:.0f}) to ({xmax:.0f}, {ymax:.0f})')

# ---- Chunked read + bbox filter ---------------------------------------------

pf = pq.ParquetFile(NWI_PATH)
n_groups     = pf.metadata.num_row_groups
total_rows   = pf.metadata.num_rows
print(f'\nTX wetlands: {total_rows:,} rows across {n_groups} row groups')
print(f'Reading chunks and applying bbox filter ...\n')

surviving_chunks = []
total_read       = 0
total_kept       = 0

for i in range(n_groups):
    # Read only the geometry column — skip attribute columns we don't need.
    # to_pandas() fails on WKB binary; to_pylist() returns raw bytes safely.
    table      = pf.read_row_group(i, columns=['geometry'])
    geom_bytes = table.column('geometry').to_pylist()
    del table
    geoms  = from_wkb(geom_bytes)
    del geom_bytes
    chunk  = gpd.GeoDataFrame(geometry=geoms, crs='EPSG:5070')
    del geoms

    n = len(chunk)
    total_read += n

    # Bbox filter — keep_zones spans all 5 states so bbox is very wide;
    # clip immediately per chunk to avoid accumulating 1.36M rows in RAM
    in_bbox = chunk.cx[xmin:xmax, ymin:ymax]
    del chunk

    if len(in_bbox) > 0:
        try:
            clipped_chunk = gpd.clip(in_bbox[['geometry']], keep_geom)
            clipped_chunk = clipped_chunk[~clipped_chunk.geometry.is_empty].copy()
            kept = len(clipped_chunk)
            if kept > 0:
                surviving_chunks.append(clipped_chunk)
        except Exception as e:
            print(f'  Warning: clip failed on row group {i+1}: {e}')
            kept = 0
    else:
        kept = 0

    del in_bbox
    total_kept += kept
    print(f'  Row group {i+1:2d}/{n_groups}: {n:>7,} rows -> {kept:>6,} clipped  '
          f'(cumulative kept: {total_kept:,})')

print(f'\nTotal read: {total_read:,}  |  Total surviving clip: {total_kept:,}')
print(f'Dropped:    {total_read - total_kept:,} rows ({(total_read-total_kept)/total_read*100:.1f}%)')

# ---- Combine surviving chunks -----------------------------------------------

print('\nCombining surviving chunks ...')
combined = gpd.GeoDataFrame(
    pd.concat(surviving_chunks, ignore_index=True),
    crs='EPSG:5070'
)
del surviving_chunks
print(f'  Combined: {len(combined):,} rows')

# ---- Precise clip already done per-chunk; skip re-clip ----------------------

clipped = combined[~combined.geometry.is_empty].copy()
del combined
print(f'  After dedup empty: {len(clipped):,} rows | {clipped.geometry.area.sum()/1e6:,.2f} km2')

# ---- make_valid -------------------------------------------------------------

print('\nmake_valid ...')
invalid_before = (~clipped.geometry.is_valid).sum()
print(f'  Invalid before: {invalid_before}')
clipped['geometry'] = clipped.geometry.apply(make_valid)
clipped = clipped[~clipped.geometry.is_empty].copy()
print(f'  Invalid after:  {(~clipped.geometry.is_valid).sum()}')

verts = sum(len(get_coordinates(g)) for g in clipped.geometry)
print(f'  Total vertices: {verts:,}')

# ---- Checks + save ----------------------------------------------------------

area_km2 = clipped.geometry.area.sum() / 1e6
checks = {
    'Has rows'         : len(clipped) > 0,
    'CRS EPSG:5070'    : clipped.crs.to_epsg() == 5070,
    'No null geoms'    : clipped.geometry.isna().sum() == 0,
    'No empty geoms'   : clipped.geometry.is_empty.sum() == 0,
    'No invalid geoms' : (~clipped.geometry.is_valid).sum() == 0,
    'Area > 0'         : area_km2 > 0,
    'Area < keep_zones': area_km2 < keep_geom.area / 1e6,
}

print(f'\n=== RESULT ===')
print(f'  Rows:  {len(clipped):,}')
print(f'  Area:  {area_km2:,.2f} km2')
print(f'  CRS:   EPSG:{clipped.crs.to_epsg()}')

print('\n=== CHECKS ===')
all_pass = True
for lbl, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
    if not ok:
        all_pass = False

# 992k rows × 154M vertices exceeds PyArrow's contiguous WKB allocation limit.
# Write in 100k-row chunks via ParquetWriter instead.
import io
CHUNK_SIZE = 100_000
print(f'\nSaving in {CHUNK_SIZE:,}-row chunks ...')
writer = None
for start in range(0, len(clipped), CHUNK_SIZE):
    chunk = clipped.iloc[start:start + CHUNK_SIZE].copy()
    buf = io.BytesIO()
    chunk.to_parquet(buf, index=False)
    buf.seek(0)
    tbl = pq.read_table(buf)
    if writer is None:
        writer = pq.ParquetWriter(OUT_PATH, tbl.schema)
    writer.write_table(tbl)
    print(f'  Wrote rows {start:,}–{start+len(chunk)-1:,}')
    del chunk, buf, tbl
if writer:
    writer.close()

print(f'\nSaved: {OUT_PATH}')
print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
