"""
Step 4 — Row-by-row audit of candidates_final.parquet against the agreed Phase 1 rules.

Rules sources:
  - Owen's PDF: Candidate_Site_Detection_Phase1_Review.pdf (May 6 2026)
  - Owen May 11 spec additions (reason codes, banded distance, original vs buildable, etc.)
  - Owen May 14 acknowledgments (size category, oversized flag)

Output: per-chunk audit summary (default chunk_size=500, sorted by composite_score desc).

Usage:
  python Step4_audit_data.py                     # walks chunk 1 only
  python Step4_audit_data.py --chunk 5           # walks chunk 5
  python Step4_audit_data.py --all               # writes audit_report.txt with all chunks
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

PARQUET = Path('candidate_areas/outputs/candidates_final.parquet')

# ----- Approved vocabularies (from PDF + May 11 spec) -----
VALID_RECOMMENDED_ACTION = {
    'Ignore', 'Monitor', 'Manual Review', 'Parcel Pull',
    'Utility Desk Check', 'Ownership Review', 'Reuse Diligence', 'Shortlist',
}
VALID_ACTIONABILITY = {
    'do_not_pitch', 'internal_diligence_only', 'apn_owner_pull_required',
    'broker_verify_required', 'nda_teaser_possible', 'buyer_ready_with_caveats',
}
VALID_CONFIDENCE = {'low', 'medium', 'high'}
VALID_CANDIDATE_TYPE = {'greenfield', 'reuse_node', 'hybrid'}
VALID_CANDIDATE_STATUS = {'pass', 'manual_review', 'excluded'}
VALID_STATES = {'AZ', 'CA', 'NV', 'TX', 'VA'}

# PDF acreage tiers (Rule 14)
ACREAGE_TIERS = [
    (50, 100,   'small_candidate'),
    (100, 250,  'moderate_candidate'),
    (250, 500,  'large_candidate'),
    (500, 1000, 'very_large_candidate'),
    (1000, 1e10,'strategic_scale'),
]

# Owen May 14: size_class buckets we proposed
SIZE_CLASS = [
    (50, 500,    'single_site'),
    (500, 5000,  'campus'),
    (5000, 1e10, 'region'),
]

# PDF Rule 13: slope hard exclusion
HARD_SLOPE_THRESHOLD = 15.0
REVIEW_SLOPE_THRESHOLD = 8.0

# Owen May 14 oversized cutoff (used in the conversation)
OVERSIZED_ACRES = 50_000
MEGA_ACRES = 5_000


def audit_row(r):
    """Returns a list of (category, severity, message) violations for one row."""
    v = []

    # ----- 1. Vocabulary compliance -----
    if r['recommended_action'] not in VALID_RECOMMENDED_ACTION:
        v.append(('vocab', 'HIGH', f'recommended_action={r["recommended_action"]!r} not in approved set'))
    if r['actionability_status'] not in VALID_ACTIONABILITY:
        v.append(('vocab', 'HIGH', f'actionability_status={r["actionability_status"]!r} not in approved set'))
    if r['confidence'] not in VALID_CONFIDENCE:
        v.append(('vocab', 'HIGH', f'confidence={r["confidence"]!r} not in approved set'))
    if r['candidate_type'] not in VALID_CANDIDATE_TYPE:
        v.append(('vocab', 'HIGH', f'candidate_type={r["candidate_type"]!r} not in approved set'))
    if r['candidate_status'] not in VALID_CANDIDATE_STATUS:
        v.append(('vocab', 'HIGH', f'candidate_status={r["candidate_status"]!r} not in approved set'))
    if r['state'] not in VALID_STATES:
        v.append(('vocab', 'HIGH', f'state={r["state"]!r} not in approved set'))

    # ----- 2. Numerical bounds -----
    if not (0 <= r['composite_score'] <= 100.001):
        v.append(('bounds', 'HIGH', f'composite_score={r["composite_score"]} out of [0,100]'))
    for sc in ['utility_score','buildability_score','supporting_infra_score','dev_risk_score']:
        if not (0 <= r[sc] <= 100.001):
            v.append(('bounds', 'HIGH', f'{sc}={r[sc]} out of [0,100]'))
    if r['area_acres'] < 50:
        v.append(('bounds', 'HIGH', f'area_acres={r["area_acres"]:.1f} below 50-acre PDF minimum'))
    if pd.notna(r.get('buildable_area_ratio')) and not (0 <= r['buildable_area_ratio'] <= 1.001):
        v.append(('bounds', 'MED', f'buildable_area_ratio={r["buildable_area_ratio"]} out of [0,1]'))
    if pd.notna(r.get('net_buildable_area_acres')) and pd.notna(r.get('original_area_acres')):
        if r['net_buildable_area_acres'] > r['original_area_acres'] * 1.001:
            v.append(('bounds', 'HIGH', f'net_buildable ({r["net_buildable_area_acres"]:.0f}) > original ({r["original_area_acres"]:.0f})'))

    # ----- 3. Composite formula (PDF Revised Scoring Framework, 90% renormalized) -----
    expected = (0.40 * r['utility_score'] + 0.20 * r['buildability_score']
                + 0.15 * r['supporting_infra_score'] + 0.15 * r['dev_risk_score']) / 0.90
    if abs(expected - r['composite_score']) > 0.01:
        v.append(('math', 'HIGH', f'composite math: expected {expected:.4f}, got {r["composite_score"]:.4f}'))

    # ----- 4. PDF Rule 13: slope hard exclusion -----
    if pd.notna(r.get('slope_max_pct')) and r['slope_max_pct'] > HARD_SLOPE_THRESHOLD:
        if r['candidate_status'] == 'pass':
            v.append(('pdf_rule_13', 'HIGH',
                      f'slope_max_pct={r["slope_max_pct"]:.1f}% exceeds 15% hard exclusion but candidate_status=pass'))
    if pd.notna(r.get('slope_mean_pct')) and r['slope_mean_pct'] > REVIEW_SLOPE_THRESHOLD:
        if not r.get('slope_review_flag', False):
            v.append(('pdf_rule_13', 'MED',
                      f'slope_mean_pct={r["slope_mean_pct"]:.1f}% > 8% but slope_review_flag not set'))

    # ----- 5. PDF Rule 29: recommended_action rules -----
    cs = r['composite_score']
    if r['candidate_status'] == 'pass':
        if cs >= 75 and r['recommended_action'] not in ('Parcel Pull', 'Utility Desk Check', 'Shortlist'):
            v.append(('pdf_rule_29', 'MED',
                      f'composite={cs:.1f} >=75 but recommended_action={r["recommended_action"]!r}'))
        elif 65 <= cs < 75 and r['recommended_action'] != 'Monitor':
            v.append(('pdf_rule_29', 'MED',
                      f'composite={cs:.1f} in [65,75) but recommended_action={r["recommended_action"]!r}'))
        elif cs < 65 and r['recommended_action'] != 'Ignore':
            v.append(('pdf_rule_29', 'MED',
                      f'composite={cs:.1f} < 65 but recommended_action={r["recommended_action"]!r}'))

    # ----- 6. Owen May 11: reason codes should have 3-5 entries -----
    codes = r.get('top_reason_codes')
    if codes is None:
        v.append(('reason_codes', 'HIGH', 'top_reason_codes is null'))
    else:
        n = len(codes) if hasattr(codes, '__len__') else 0
        if n < 3 or n > 5:
            v.append(('reason_codes', 'LOW', f'top_reason_codes has {n} entries (expected 3-5)'))

    # ----- 7. Owen May 14: size/oversized awareness -----
    if r['area_acres'] >= OVERSIZED_ACRES:
        v.append(('size', 'HIGH', f'OVERSIZED: area_acres={r["area_acres"]:,.0f} >= 50,000 (Owen disclosed)'))
    elif r['area_acres'] >= MEGA_ACRES:
        v.append(('size', 'MED', f'MEGA: area_acres={r["area_acres"]:,.0f} >= 5,000 (likely needs subdivision)'))

    # ----- 8. Anchor plausibility -----
    pad = r.get('primary_anchor_distance_m')
    if pd.notna(pad) and pad > 50_000 and r['utility_score'] > 80:
        v.append(('anchor', 'MED',
                  f'primary_anchor at {pad/1000:.1f} km > 50 km but utility_score={r["utility_score"]:.1f}'))

    # ----- 9. PAD-US adjacency consistency -----
    if r.get('near_padus_flag', False):
        codes_list = list(codes) if codes is not None and hasattr(codes, '__iter__') else []
        if not any('padus' in str(c).lower() or 'protect' in str(c).lower() for c in codes_list):
            v.append(('consistency', 'LOW',
                      'near_padus_flag=true but no padus/protected reason code present'))

    # ----- 10. Geometry sanity -----
    geom = r.get('geometry')
    if geom is None or geom.is_empty:
        v.append(('geometry', 'HIGH', 'geometry is null/empty'))
    elif not geom.is_valid:
        v.append(('geometry', 'HIGH', 'geometry is invalid'))
    else:
        minx, miny, maxx, maxy = geom.bounds
        bbox_span_km = max(maxx - minx, maxy - miny) / 1000.0  # EPSG:5070 in meters
        if bbox_span_km > 200:
            v.append(('geometry', 'HIGH',
                      f'bbox span {bbox_span_km:.0f} km — implausibly large for a single site'))
        elif bbox_span_km > 50:
            v.append(('geometry', 'MED',
                      f'bbox span {bbox_span_km:.0f} km — likely spans multiple counties'))

    return v


def chunk_report(df_chunk, chunk_id, total_chunks):
    """Print an audit report for a single chunk of rows."""
    n = len(df_chunk)
    cs = df_chunk['composite_score']
    print(f'\n{"=" * 78}')
    print(f' CHUNK {chunk_id}/{total_chunks}   (rows ranked {df_chunk.index[0]+1:,}-{df_chunk.index[-1]+1:,} '
          f'by composite_score desc)')
    print(f'{"=" * 78}')
    print(f'  Composite range:  {cs.max():.2f} (top) -> {cs.min():.2f} (bottom),  median {cs.median():.2f}')
    print(f'  By state: ' + ', '.join(f'{k}={v}' for k, v in df_chunk['state'].value_counts().items()))
    print(f'  By action: ' + ', '.join(f'{k}={v}' for k, v in df_chunk['recommended_action'].value_counts().items()))
    print(f'  By confidence: ' + ', '.join(f'{k}={v}' for k, v in df_chunk['confidence'].value_counts().items()))
    print(f'  Area_acres: min={df_chunk["area_acres"].min():,.0f}, median={df_chunk["area_acres"].median():,.0f}, '
          f'max={df_chunk["area_acres"].max():,.0f}')

    # Run audit
    all_violations = []
    rows_with_high = 0
    rows_with_med = 0
    rows_with_low = 0
    rows_clean = 0
    for _, row in df_chunk.iterrows():
        vs = audit_row(row)
        if not vs:
            rows_clean += 1
            continue
        has_h = any(s == 'HIGH' for _, s, _ in vs)
        has_m = any(s == 'MED' for _, s, _ in vs)
        has_l = any(s == 'LOW' for _, s, _ in vs)
        if has_h: rows_with_high += 1
        elif has_m: rows_with_med += 1
        elif has_l: rows_with_low += 1
        for cat, sev, msg in vs:
            all_violations.append((row['candidate_id'], row['composite_score'], row['state'], cat, sev, msg))

    print(f'\n  ROW-LEVEL AUDIT:')
    print(f'    Clean (no issues):     {rows_clean:>4} ({100*rows_clean/n:.1f}%)')
    print(f'    HIGH-severity issue:   {rows_with_high:>4} ({100*rows_with_high/n:.1f}%)')
    print(f'    MED-severity issue:    {rows_with_med:>4} ({100*rows_with_med/n:.1f}%)')
    print(f'    LOW-severity issue:    {rows_with_low:>4} ({100*rows_with_low/n:.1f}%)')

    # Violation count by category x severity
    vio_df = pd.DataFrame(all_violations, columns=['cid','cs','state','cat','sev','msg'])
    if len(vio_df):
        print(f'\n  VIOLATIONS BY CATEGORY:')
        pivot = vio_df.groupby(['cat','sev']).size().unstack(fill_value=0)
        for col in ['HIGH','MED','LOW']:
            if col not in pivot.columns:
                pivot[col] = 0
        pivot = pivot[['HIGH','MED','LOW']].sort_values('HIGH', ascending=False)
        for cat, row in pivot.iterrows():
            print(f'    {cat:<20}  HIGH={row["HIGH"]:>4}  MED={row["MED"]:>4}  LOW={row["LOW"]:>4}')

        # Sample 3 worst rows (HIGH-severity, top of chunk)
        high_violations = vio_df[vio_df['sev'] == 'HIGH'].sort_values('cs', ascending=False).head(5)
        if len(high_violations):
            print(f'\n  SAMPLE HIGH-SEVERITY ROWS:')
            seen_cids = set()
            for _, vr in high_violations.iterrows():
                if vr['cid'] in seen_cids:
                    continue
                seen_cids.add(vr['cid'])
                row_msgs = vio_df[vio_df['cid'] == vr['cid']]
                print(f'    cand {vr["cid"][:13]}.. (state={vr["state"]}, composite={vr["cs"]:.2f})')
                for _, m in row_msgs.iterrows():
                    print(f'      [{m["sev"]}] {m["cat"]}: {m["msg"]}')
    else:
        print(f'\n  No violations in this chunk.')

    return {
        'chunk': chunk_id,
        'n_rows': n,
        'clean': rows_clean,
        'high': rows_with_high,
        'med': rows_with_med,
        'low': rows_with_low,
        'violations': all_violations,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', type=int, default=1, help='chunk number to audit (1-based)')
    ap.add_argument('--chunk-size', type=int, default=500)
    ap.add_argument('--all', action='store_true', help='audit all chunks, write summary table at end')
    args = ap.parse_args()

    print(f'Loading {PARQUET} ...')
    g = gpd.read_parquet(PARQUET)
    g = g.sort_values('composite_score', ascending=False).reset_index(drop=True)
    n = len(g)
    print(f'  {n:,} rows loaded, sorted by composite_score desc')
    total_chunks = (n + args.chunk_size - 1) // args.chunk_size
    print(f'  {total_chunks} chunks of {args.chunk_size} rows each')

    if args.all:
        all_summary = []
        for c in range(1, total_chunks + 1):
            start = (c - 1) * args.chunk_size
            end = min(start + args.chunk_size, n)
            chunk = g.iloc[start:end]
            s = chunk_report(chunk, c, total_chunks)
            all_summary.append(s)
        # Final summary
        print(f'\n\n{"#" * 78}\n FULL-FILE SUMMARY\n{"#" * 78}')
        sdf = pd.DataFrame([{k: v for k, v in d.items() if k != 'violations'} for d in all_summary])
        print(sdf.to_string(index=False))
        print(f'\nTotal clean rows: {sdf["clean"].sum():,} / {n:,} ({100*sdf["clean"].sum()/n:.1f}%)')
        print(f'Total HIGH-severity rows: {sdf["high"].sum():,} ({100*sdf["high"].sum()/n:.1f}%)')
        print(f'Total MED-severity rows: {sdf["med"].sum():,} ({100*sdf["med"].sum()/n:.1f}%)')
        print(f'Total LOW-severity rows: {sdf["low"].sum():,} ({100*sdf["low"].sum()/n:.1f}%)')
    else:
        c = args.chunk
        if c < 1 or c > total_chunks:
            print(f'Chunk {c} out of range [1, {total_chunks}]', file=sys.stderr)
            sys.exit(2)
        start = (c - 1) * args.chunk_size
        end = min(start + args.chunk_size, n)
        chunk = g.iloc[start:end]
        chunk_report(chunk, c, total_chunks)


if __name__ == '__main__':
    main()
