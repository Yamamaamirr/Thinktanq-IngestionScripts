"""
Step 7e - Post-filter: drop candidates intersecting public/military/rec GAP-3/4 land.

PDF Rule 8 hard-exclusion list includes "obvious public/non-developable lands
where ownership or legal use is incompatible". Currently exclusion_padus.parquet
only includes GAP-1/2. This post-filter adds the public/military/rec subset of
GAP-3/4 (city parks, military bases, federal rec areas) by dropping any candidate
whose geometry intersects them.

Approach (cheap, no upstream re-run):
  1. Load pad_us_all.parquet (all GAP statuses)
  2. Filter to public/military/rec GAP-3/4:
       GAP_Sts in ('3','4') AND (
         Mang_Type in ('FED','DOD','MIL','LOC') OR
         Des_Tp in ('LREC','MIL','MPUB','NRA')
       )
  3. Union into a single mask geometry
  4. Drop or mark any candidate whose geometry intersects the mask

Reads / Writes:
  candidate_areas/outputs/candidate_areas.parquet  (overwritten)
  candidate_areas/outputs/candidate_areas_PRE_GAP34_FILTER.parquet  (backup)
"""
import time
from pathlib import Path
import numpy as np
import geopandas as gpd

CAND_PATH    = Path('candidate_areas/outputs/candidate_areas.parquet')
PADUS_PATH   = Path('ingestion_scripts/protected_areas_USA/pad_us_all.parquet')
BACKUP_PATH  = Path('candidate_areas/outputs/candidate_areas_PRE_GAP34_FILTER.parquet')

PUBLIC_MANG = {'FED', 'DOD', 'MIL', 'LOC'}
PUBLIC_DESTP = {'LREC', 'MIL', 'MPUB', 'NRA'}


def main():
    print(f'Loading {CAND_PATH} ...')
    cands = gpd.read_parquet(CAND_PATH)
    n_before = len(cands)
    print(f'  {n_before:,} candidates')

    if not BACKUP_PATH.exists():
        print(f'\nBackup -> {BACKUP_PATH}')
        cands.to_parquet(BACKUP_PATH, index=False)

    print(f'\nLoading PAD-US (all GAP statuses) ...')
    pad = gpd.read_parquet(PADUS_PATH)
    if pad.crs != cands.crs:
        pad = pad.to_crs(cands.crs)
    print(f'  {len(pad):,} polygons')

    # Filter to public/military/rec subset of GAP-3/4
    mask = pad['GAP_Sts'].isin(['3','4']) & (
        pad['Mang_Type'].isin(PUBLIC_MANG) | pad['Des_Tp'].isin(PUBLIC_DESTP)
    )
    public34 = pad[mask].copy()
    print(f'\nPublic/military/rec GAP-3/4 polygons: {len(public34):,}')
    print(f'  By Mang_Type:')
    print(public34.Mang_Type.value_counts().head(10).to_string())
    print(f'  By Des_Tp top 10:')
    print(public34.Des_Tp.value_counts().head(10).to_string())
    print(f'  Total area: {public34.geometry.area.sum()/1e6:,.0f} km^2')

    if len(public34) == 0:
        print('No public GAP-3/4 polygons found. No filtering applied.')
        return

    # Use spatial join to find intersections (vectorized)
    print('\nSpatial join (candidates intersects public GAP-3/4) ...')
    t0 = time.time()
    inter = gpd.sjoin(
        cands[['candidate_id','geometry']],
        public34[['geometry']],
        how='inner',
        predicate='intersects',
    )
    intersecting_ids = set(inter.candidate_id.unique())
    print(f'  {time.time()-t0:.1f}s, {len(intersecting_ids):,} candidates intersect public GAP-3/4 land')

    # Drop them
    keep = cands[~cands.candidate_id.isin(intersecting_ids)].copy()
    n_after = len(keep)
    print(f'\n  Dropping {n_before - n_after:,} candidates')
    print(f'  Remaining: {n_after:,}')
    print(f'  By state (lost / kept / pct_lost):')
    for st in ['AZ','CA','NV','TX','VA']:
        n_b = (cands.state == st).sum()
        n_a = (keep.state == st).sum()
        print(f'    {st}: lost {n_b-n_a:>5,}  kept {n_a:>6,}  ({100*(n_b-n_a)/max(n_b,1):.1f}%)')

    print(f'\nWriting back to {CAND_PATH} ...')
    keep.to_parquet(CAND_PATH, index=False)
    print('Done.')


if __name__ == '__main__':
    main()
