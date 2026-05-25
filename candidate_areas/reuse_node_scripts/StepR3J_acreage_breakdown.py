"""
StepR3J -- Original vs net buildable acreage per reuse node.

Mirror of Step1J_acreage_breakdown.py for reuse_nodes_clean.parquet.

Output columns identical to Step1J:
  original_area_acres, net_buildable_area_acres, buildable_area_ratio

Run: python candidate_areas/reuse_node_scripts/StepR3J_acreage_breakdown.py
"""
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _r3_helpers import load_reuse_nodes_as_candidates, out_path

OUT_PATH = out_path('stepR3j_acreage_breakdown.parquet')
ACRES_PER_M2 = 1 / 4046.856


def main():
    print('Loading reuse nodes ...')
    cands = load_reuse_nodes_as_candidates(crs_epsg=5070)
    print(f'  {len(cands):,} reuse nodes')

    print('\nComputing convex hulls and areas ...')
    t0 = time.time()
    hulls = cands.geometry.convex_hull
    hull_area_m2 = hulls.area.values
    poly_area_m2 = cands.geometry.area.values
    print(f'  done in {time.time()-t0:.1f}s')

    hull_area_m2 = np.maximum(hull_area_m2, poly_area_m2)
    original_acres = hull_area_m2 * ACRES_PER_M2
    net_acres      = poly_area_m2 * ACRES_PER_M2
    ratio = np.where(original_acres > 0, net_acres / original_acres, 1.0)

    out = pd.DataFrame({
        'candidate_id':             cands.candidate_id.values,
        'original_area_acres':      original_acres,
        'net_buildable_area_acres': net_acres,
        'buildable_area_ratio':     ratio,
    })

    out.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1e6:.2f} MB, {len(out):,} rows)')

    print('\n=== Distribution ===')
    print(f'original:  median={out.original_area_acres.median():.0f}  p90={out.original_area_acres.quantile(0.9):.0f}')
    print(f'net:       median={out.net_buildable_area_acres.median():.0f}  p90={out.net_buildable_area_acres.quantile(0.9):.0f}')
    print(f'ratio:     median={out.buildable_area_ratio.median():.3f}  p10={out.buildable_area_ratio.quantile(0.1):.3f}')

    print('\n=== Checks ===')
    checks = {
        'Has all reuse nodes'  : len(out) == len(cands),
        'Unique candidate_ids' : out.candidate_id.is_unique,
        'original >= net'      : (out.original_area_acres >= out.net_buildable_area_acres - 0.01).all(),
        'ratio in [0,1]'       : ((out.buildable_area_ratio >= 0) & (out.buildable_area_ratio <= 1.001)).all(),
        'no nulls'             : out[['original_area_acres','net_buildable_area_acres','buildable_area_ratio']].notna().all().all(),
        'net matches area_acres': ((out.net_buildable_area_acres - cands.area_acres.values).abs() < 0.5).all(),
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
