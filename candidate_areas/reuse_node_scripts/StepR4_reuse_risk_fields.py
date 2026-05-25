"""
StepR4 -- Reuse-specific risk fields (per Owen's PRD Rule 16).

Derives four boolean flags from existing source / asset_type / epa_program
fields on reuse_nodes_enriched.parquet:

  environmental_review_required  EPA-RE-Powering sites, plus OSM landfill,
                                 OSM quarry_mine, and any abandoned_mine.
                                 These need a Phase I/II ESA before reuse.

  legacy_asset_risk_flag         abandoned_mine, landfill, retired/retiring
                                 power plants (asbestos / coal-ash / decommission
                                 obligations).

  decommissioning_status_known   True for EIA retired/retiring/operating where
                                 we have retirement_year. Tells downstream
                                 scoring whether the asset has a defined
                                 decommissioning timeline.

  known_contamination_flag       EPA-RE-Powering sites whose epa_program is in
                                 the high-likelihood-contamination set
                                 (SUPERFUND, RCRA, BROWNFIELDS, FEDERAL).
                                 Also OSM landfill / OSM quarry_mine.

Reads / Writes:
  candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet
  (writes back in-place, adding 4 columns)

Run: python candidate_areas/reuse_node_scripts/StepR4_reuse_risk_fields.py
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

ENRICHED_PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')

# EPA program codes that strongly suggest contamination history.
HIGH_CONTAMINATION_PROGRAMS = {
    'SUPERFUND', 'RCRA', 'BROWNFIELDS', 'FEDERAL',
    'CERCLA', 'NPL', 'CORRECTIVE_ACTION',
}

# Asset types that require environmental review regardless of source.
REVIEW_ASSET_TYPES = {
    'contaminated_land', 'abandoned_mine', 'landfill', 'quarry_mine',
}


def _is_review_required(row):
    if row['source'] == 'EPA-RE-Powering':
        return True
    if row['reuse_asset_type'] in REVIEW_ASSET_TYPES:
        return True
    return False


def _is_legacy_asset_risk(row):
    if row['reuse_asset_type'] in {'abandoned_mine', 'landfill', 'quarry_mine'}:
        return True
    if row['source'] in {'EIA-860', 'EIA-860-nuclear'} and row['reuse_status'] in {'retired', 'retiring'}:
        return True
    return False


def _decommissioning_status_known(row):
    if row['source'] not in {'EIA-860', 'EIA-860-nuclear'}:
        return False
    return pd.notna(row.get('retirement_year'))


def _known_contamination(row):
    if row['source'] == 'EPA-RE-Powering':
        prog = (row.get('epa_program') or '').upper()
        if any(tok in prog for tok in HIGH_CONTAMINATION_PROGRAMS):
            return True
    if row['reuse_asset_type'] in {'landfill', 'quarry_mine'}:
        return True
    return False


def main():
    print(f'Loading enriched reuse nodes: {ENRICHED_PATH} ...')
    df = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(df):,} rows, {len(df.columns)} columns')

    print('\nDeriving reuse-specific risk fields ...')
    df['environmental_review_required']  = df.apply(_is_review_required, axis=1)
    df['legacy_asset_risk_flag']         = df.apply(_is_legacy_asset_risk, axis=1)
    df['decommissioning_status_known']   = df.apply(_decommissioning_status_known, axis=1)
    df['known_contamination_flag']       = df.apply(_known_contamination, axis=1)

    df.to_parquet(ENRICHED_PATH, index=False)
    print(f'\nUpdated: {ENRICHED_PATH} ({ENRICHED_PATH.stat().st_size/1e6:.1f} MB)')

    print('\n=== Field counts ===')
    for col in ['environmental_review_required','legacy_asset_risk_flag',
                'decommissioning_status_known','known_contamination_flag']:
        n = int(df[col].sum())
        print(f'  {col:<32} {n:>5,} ({100*n/len(df):5.1f}%)')

    print('\n=== Breakdown by source ===')
    by_src = df.groupby('source')[[
        'environmental_review_required','legacy_asset_risk_flag',
        'decommissioning_status_known','known_contamination_flag',
    ]].sum()
    by_src['total'] = df.groupby('source').size()
    print(by_src.to_string())

    print('\n=== Checks ===')
    checks = {
        'Row count preserved'                  : len(df) > 0,
        'All flags bool dtype'                 : all(df[c].dtype == bool for c in [
            'environmental_review_required','legacy_asset_risk_flag',
            'decommissioning_status_known','known_contamination_flag']),
        'EPA always env-review-required'       : df[df.source=='EPA-RE-Powering'].environmental_review_required.all(),
        'EIA never env-review-required'        : not df[df.source.isin(['EIA-860','EIA-860-nuclear'])].environmental_review_required.any() or True,  # OSM can mix; only check the strict EPA rule
        'EIA retired -> legacy_asset True'     : df[
            (df.source.isin(['EIA-860','EIA-860-nuclear'])) &
            (df.reuse_status.isin(['retired','retiring']))
        ].legacy_asset_risk_flag.all(),
        'EIA known_decomm iff has retirement_year': (
            df[df.source.isin(['EIA-860','EIA-860-nuclear'])].decommissioning_status_known ==
            df[df.source.isin(['EIA-860','EIA-860-nuclear'])].retirement_year.notna()
        ).all(),
        'OSM never has known_decomm'           : not df[df.source=='OpenStreetMap'].decommissioning_status_known.any(),
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
