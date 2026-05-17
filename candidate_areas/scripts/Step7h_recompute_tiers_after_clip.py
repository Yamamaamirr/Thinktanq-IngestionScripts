"""
Step 7h - Recompute acreage_tier / size_class / Shortlist after road clip.

Road clip changed area_acres for 10,947 candidates. Their stored tier/class
values are now stale relative to the new areas. Fix by recomputing.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

CAND_FINAL    = Path('candidate_areas/outputs/candidates_final.parquet')
CSV_OUT       = Path('candidate_areas/outputs/candidates_final.csv')
FGB_OUT       = Path('candidate_areas/outputs/candidates_final.fgb')
ENRICHED_PATH = Path('candidate_areas/outputs/candidate_areas_enriched.parquet')


def _je(x):
    if x is None: return None
    if isinstance(x, np.ndarray): return json.dumps(x.tolist())
    if isinstance(x, (list, tuple, dict)): return json.dumps(list(x) if isinstance(x, tuple) else x)
    return x


def acreage_tier(acres):
    if acres < 100:    return 'small',           50.0
    if acres < 250:    return 'moderate',        70.0
    if acres < 500:    return 'large',           85.0
    if acres < 1000:   return 'very_large',      95.0
    return                  'strategic_scale',   100.0


def size_class(acres):
    if acres < 500:    return 'site',    False
    if acres < 5000:   return 'campus',  False
    return                  'region',    True


def recommended_action(r):
    if r['candidate_status'] == 'manual_review':
        return 'Manual Review'
    if r['candidate_status'] == 'excluded':
        return 'Ignore'
    cs = r['composite_score']
    if (cs >= 90 and 500 <= r['area_acres'] <= 5000 and not r['slope_review_flag']
            and not r['oversized_flag'] and r['confidence'] in ('medium','high')
            and (r['num_anchors_in_range'] or 0) >= 3):
        return 'Shortlist'
    if cs >= 85 and r['utility_module_status'] == 'zone_fallback':
        return 'Utility Desk Check'
    if cs >= 75 and pd.isna(r.get('parcel_count')):
        return 'Parcel Pull'
    if cs >= 65:
        return 'Monitor'
    return 'Ignore'


def main():
    print(f'Loading {CAND_FINAL} ...')
    g = gpd.read_parquet(CAND_FINAL)
    print(f'  {len(g):,} candidates')

    print('\nRecomputing acreage_tier from current area_acres ...')
    tr = g.area_acres.apply(acreage_tier)
    g['acreage_tier']       = [t[0] for t in tr]
    g['acreage_tier_score'] = [t[1] for t in tr]

    print('Recomputing size_class from current area_acres ...')
    sc = g.area_acres.apply(size_class)
    g['size_class']     = [t[0] for t in sc]
    g['oversized_flag'] = [t[1] for t in sc]

    print('Recomputing recommended_action ...')
    g['recommended_action'] = g.apply(recommended_action, axis=1)

    print('\nResults:')
    print('  acreage_tier:')
    print(g.acreage_tier.value_counts().to_string())
    print('\n  recommended_action:')
    print(g.recommended_action.value_counts().to_string())

    print(f'\nWriting parquet ...')
    g.to_parquet(CAND_FINAL, index=False)
    print(f'  saved ({CAND_FINAL.stat().st_size/1e6:.1f} MB)')

    print('Writing CSV ...')
    csv_df = g.copy()
    csv_df['geometry_wkt'] = csv_df.geometry.to_wkt()
    csv_df = csv_df.drop(columns=['geometry'])
    for col in csv_df.columns:
        sample = csv_df[col].dropna().head(1)
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict, tuple, np.ndarray)):
            csv_df[col] = csv_df[col].apply(_je)
    csv_df.to_csv(CSV_OUT, index=False)
    print(f'  saved ({CSV_OUT.stat().st_size/1e6:.1f} MB)')

    print('Writing FGB ...')
    fgb_df = g.copy()
    for col in fgb_df.columns:
        if col == 'geometry': continue
        sample = fgb_df[col].dropna().head(1)
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict, tuple, np.ndarray)):
            fgb_df[col] = fgb_df[col].apply(_je)
    fgb_target = FGB_OUT
    try:
        if fgb_target.exists():
            fgb_target.unlink()
        fgb_df.to_file(fgb_target, driver='FlatGeobuf')
        print(f'  saved ({fgb_target.stat().st_size/1e6:.1f} MB)')
    except (PermissionError, OSError):
        fgb_target = FGB_OUT.with_name('candidates_final_NEW.fgb')
        if fgb_target.exists():
            fgb_target.unlink()
        fgb_df.to_file(fgb_target, driver='FlatGeobuf')
        print(f'  FGB locked, wrote to {fgb_target}')


if __name__ == '__main__':
    main()
