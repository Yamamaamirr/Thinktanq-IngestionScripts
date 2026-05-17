"""
Inherit slope_per_candidate values for newly created fragments from their
parent polygons. Avoids re-running multi-hour 3DEP tile downloads.

Approach:
  1. Load existing slope_per_candidate.parquet (95,269 rows, one per ORIGINAL candidate_id)
  2. Load current candidate_areas.parquet (125,109 rows, post-subdivision + GAP-3/4)
  3. For unsplit candidates (parent_candidate_id is null): use existing slope by candidate_id
  4. For fragments: lookup parent_candidate_id in slope_per_candidate, copy slope_mean & slope_max
  5. Write updated slope_per_candidate.parquet with 125,109 rows

Conservative trade-off:
  Fragments inherit parent's slope_MAX, which may overestimate (steep parent corner
  may not be in this fragment). This means more fragments may be flagged by
  PDF Rule 13 (slope_max > 15 -> hard exclusion) than strictly necessary.
  This is documented in known-issues; full per-fragment 3DEP sampling is a
  Phase 1.5 follow-up.
"""
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import geopandas as gpd

CAND_PATH = Path('candidate_areas/outputs/candidate_areas.parquet')
SLOPE_PATH = Path('ingestion_scripts/usgs_slope/slope_per_candidate.parquet')
SLOPE_BACKUP = Path('ingestion_scripts/usgs_slope/slope_per_candidate_PRE_INHERIT.parquet')


def main():
    print(f'Loading {CAND_PATH} ...')
    cands = gpd.read_parquet(CAND_PATH)
    print(f'  {len(cands):,} candidates')

    print(f'Loading {SLOPE_PATH} ...')
    slope = pd.read_parquet(SLOPE_PATH)
    print(f'  {len(slope):,} slope rows')

    # Backup
    if not SLOPE_BACKUP.exists():
        print(f'Backup -> {SLOPE_BACKUP}')
        slope.to_parquet(SLOPE_BACKUP, index=False)

    # Index slope by candidate_id
    slope_lookup = slope.set_index('candidate_id')[['slope_mean_pct','slope_max_pct','n_pixels']]

    out_rows = []
    n_unsplit_found = 0
    n_unsplit_missing = 0
    n_fragments_inherited = 0
    n_fragments_missing = 0

    for _, row in cands[['candidate_id','parent_candidate_id']].iterrows():
        cid = row['candidate_id']
        pid = row['parent_candidate_id']
        if pid is None or (isinstance(pid, float) and pd.isna(pid)):
            # Unsplit: use existing
            if cid in slope_lookup.index:
                s = slope_lookup.loc[cid]
                out_rows.append({
                    'candidate_id':   cid,
                    'slope_mean_pct': s['slope_mean_pct'],
                    'slope_max_pct':  s['slope_max_pct'],
                    'n_pixels':       s['n_pixels'],
                    'inherited_from': None,
                })
                n_unsplit_found += 1
            else:
                out_rows.append({
                    'candidate_id':   cid,
                    'slope_mean_pct': None,
                    'slope_max_pct':  None,
                    'n_pixels':       0,
                    'inherited_from': None,
                })
                n_unsplit_missing += 1
        else:
            # Fragment: inherit from parent
            if pid in slope_lookup.index:
                s = slope_lookup.loc[pid]
                out_rows.append({
                    'candidate_id':   cid,
                    'slope_mean_pct': s['slope_mean_pct'],
                    'slope_max_pct':  s['slope_max_pct'],
                    'n_pixels':       s['n_pixels'],  # not literal pixel count, just inherited
                    'inherited_from': pid,
                })
                n_fragments_inherited += 1
            else:
                out_rows.append({
                    'candidate_id':   cid,
                    'slope_mean_pct': None,
                    'slope_max_pct':  None,
                    'n_pixels':       0,
                    'inherited_from': None,
                })
                n_fragments_missing += 1

    out = pd.DataFrame(out_rows)
    out['ingested_at'] = datetime.now(timezone.utc)

    print(f'\nResults:')
    print(f'  Unsplit with existing slope: {n_unsplit_found:,}')
    print(f'  Unsplit MISSING slope:       {n_unsplit_missing:,}')
    print(f'  Fragments inherited:         {n_fragments_inherited:,}')
    print(f'  Fragments MISSING:           {n_fragments_missing:,}')
    print(f'  Total rows: {len(out):,}')

    n_with_data = out['slope_mean_pct'].notna().sum()
    print(f'  With slope data: {n_with_data:,} / {len(out):,} ({100*n_with_data/len(out):.1f}%)')

    print(f'\nWriting {SLOPE_PATH} ...')
    out.to_parquet(SLOPE_PATH, index=False)

    # Distribution
    if n_with_data > 0:
        s = out.loc[out['slope_mean_pct'].notna()]
        print(f'\n  slope_mean_pct: min={s.slope_mean_pct.min():.2f}, '
              f'median={s.slope_mean_pct.median():.2f}, '
              f'max={s.slope_mean_pct.max():.2f}')
        print(f'  slope_max_pct:  min={s.slope_max_pct.min():.2f}, '
              f'median={s.slope_max_pct.median():.2f}, '
              f'max={s.slope_max_pct.max():.2f}')


if __name__ == '__main__':
    main()
