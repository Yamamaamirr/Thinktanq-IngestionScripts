"""
Step 1J — Original vs net buildable acreage per candidate.

Per Owen May 11:
  "Preserve original vs. usable acreage. Since wetlands, floodways, PAD-US,
   open water, and steep slopes are being physically subtracted, I still want
   to know how much area was removed. A clean 100-acre candidate is very
   different from a 500-acre polygon clipped down to 100 usable acres."

Implementation note (Phase 1 approximation)
-------------------------------------------
The exact computation Owen described (intersect each candidate against the
union of all exclusion layers and sum the removed area) is prohibitively
expensive at our scale — the exclusion_mask is 96 huge multi-polygons
totaling tens of millions of vertices.

For Phase 1 we use the CONVEX HULL of each candidate as a proxy for the
"original buildable footprint" before exclusion. This captures Owen's
directional signal exactly:

  - A clean rectangular candidate that filled its keep-zone region
    cleanly has hull_area ≈ candidate_area, so ratio ≈ 1.0.
  - A swiss-cheese candidate that survived after many exclusions were
    carved out has hull_area >> candidate_area, so ratio is low (0.3-0.7).

The exact per-reason area attribution (Owen's "excluded_area_acres_by_reason"
field) is deferred to a follow-up pass after the Friday delivery, per Owen's
May 14 reply.

Output:
  candidate_areas/enrichment_outputs/step1j_acreage_breakdown.parquet

Columns:
  original_area_acres
  net_buildable_area_acres
  buildable_area_ratio

Run:
  python candidate_areas/enrichment_scripts/Step1J_acreage_breakdown.py
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import geopandas as gpd

CAND_PATH = Path('candidate_areas/outputs/candidate_areas.parquet')
OUT_PATH  = Path('candidate_areas/enrichment_outputs/step1j_acreage_breakdown.parquet')

ACRES_PER_M2 = 1 / 4046.856


def main():
    print(f'Loading candidates: {CAND_PATH} ...')
    cands = gpd.read_parquet(CAND_PATH)
    print(f'  {len(cands):,} candidates in EPSG:{cands.crs.to_epsg()}')

    print('\nComputing convex hulls and areas (vectorized) ...')
    t0 = time.time()
    # Vectorized: shapely 2.0 applies convex_hull to the whole GeoSeries
    hulls = cands.geometry.convex_hull
    hull_area_m2 = hulls.area.values
    poly_area_m2 = cands.geometry.area.values
    print(f'  done in {time.time()-t0:.1f}s')

    # Sanity: convex hull area must be >= polygon area (geometrically required)
    # Numerical edge cases can make hull a hair smaller, so clamp to >= poly area
    hull_area_m2 = np.maximum(hull_area_m2, poly_area_m2)

    original_acres = hull_area_m2 * ACRES_PER_M2
    net_acres      = poly_area_m2 * ACRES_PER_M2
    ratio          = np.where(original_acres > 0, net_acres / original_acres, 1.0)

    out = pd.DataFrame({
        'candidate_id':              cands.candidate_id.values,
        'original_area_acres':       original_acres,
        'net_buildable_area_acres':  net_acres,
        'buildable_area_ratio':      ratio,
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f'\nSaved: {OUT_PATH} ({size_mb:.2f} MB, {len(out):,} rows)')

    # ---- Distribution ----
    print('\n=== Distribution ===')
    print(f'original_area_acres:    median={out.original_area_acres.median():.0f}, '
          f'p10={out.original_area_acres.quantile(0.1):.0f}, '
          f'p90={out.original_area_acres.quantile(0.9):.0f}, '
          f'max={out.original_area_acres.max():.0f}')
    print(f'net_buildable_area:     median={out.net_buildable_area_acres.median():.0f}, '
          f'p10={out.net_buildable_area_acres.quantile(0.1):.0f}, '
          f'p90={out.net_buildable_area_acres.quantile(0.9):.0f}')
    print(f'buildable_area_ratio:   median={out.buildable_area_ratio.median():.3f}, '
          f'p10={out.buildable_area_ratio.quantile(0.1):.3f}, '
          f'p90={out.buildable_area_ratio.quantile(0.9):.3f}, '
          f'min={out.buildable_area_ratio.min():.3f}')

    print('\n=== Buildable ratio bands ===')
    out['_rb'] = pd.cut(out.buildable_area_ratio,
                        bins=[0, 0.25, 0.5, 0.75, 0.9, 1.001],
                        labels=['<25%','25-50%','50-75%','75-90%','>=90%'])
    print(out['_rb'].value_counts().reindex(['<25%','25-50%','50-75%','75-90%','>=90%']).to_string())
    out = out.drop(columns=['_rb'])

    # Sanity: how does this compare per state?
    print('\n=== Per-state median ratio ===')
    cdf = cands[['candidate_id','state']].merge(out, on='candidate_id')
    for st in ['AZ','CA','NV','TX','VA']:
        sd = cdf[cdf.state == st]
        if len(sd) == 0:
            continue
        print(f'  {st}: n={len(sd):>6,}  median_ratio={sd.buildable_area_ratio.median():.3f}  '
              f'median_orig={sd.original_area_acres.median():>5.0f}ac  '
              f'median_net={sd.net_buildable_area_acres.median():>5.0f}ac')

    print('\n=== Checks ===')
    checks = {
        'Has all candidates'        : len(out) == len(cands),
        'Unique candidate_ids'      : out.candidate_id.is_unique,
        'original >= net'           : (out.original_area_acres >= out.net_buildable_area_acres - 0.01).all(),
        'ratio in [0,1]'            : ((out.buildable_area_ratio >= 0) & (out.buildable_area_ratio <= 1.001)).all(),
        'no nulls'                  : out[['original_area_acres','net_buildable_area_acres','buildable_area_ratio']].notna().all().all(),
        'net matches area_acres'    : ((out.net_buildable_area_acres - cands.area_acres.values).abs() < 0.5).all(),
    }
    all_pass = True
    for lbl, ok in checks.items():
        print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
        if not ok:
            all_pass = False
    print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
