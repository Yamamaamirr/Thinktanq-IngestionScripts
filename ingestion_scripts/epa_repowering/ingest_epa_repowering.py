"""
EPA RE-Powering reuse-node ingestion -- contaminated land, landfills, mine sites.

Owen's Infrastructure Reuse Nodes layer (May 1): "EPA RE-Powering / brownfield /
landfill / mine sites". These are disturbed/contaminated parcels EPA has
pre-screened for renewable/redevelopment potential -- already-impacted land
that is preferable to greenfield for siting.

Source: EPA RE-Powering Screening Dataset 2022 (updated 2023)
        https://www.epa.gov/system/files/documents/2023-05/
          re-powering-screening-dataset-2022%20Updated.xlsx
        ~190,000 sites nationally. Downloaded once to cache/.

This script reads the 'RE-Powering Sites' sheet, filters to the 5 target
states, classifies each site by type, and keeps the fields relevant to
reuse-node scoring.

reuse_asset_type classification:
  abandoned_mine  -- "Known Abandoned Mine Land" = Yes
  landfill        -- "Known Landfill" = Yes
  contaminated_land -- everything else (Superfund / brownfield / RCRA / etc.)

Non-reuse filtering
-------------------
The raw EPA dataset includes active military installations and national parks
(they carry legacy-contamination flags). These are NOT redevelopment candidates
-- they are exactly the federal land the greenfield pipeline excludes as
GAP-3/4. Two filters drop them:
  1. Name pattern: military / park / national forest keywords.
  2. Acreage cap: anything over 5,000 acres is a base or a region, not a
     reuse parcel.

Output: ingestion_scripts/epa_repowering/epa_repowering_sites.parquet
  Columns:
    site_id, site_name, state_abbr, county, city, reuse_asset_type,
    acreage, epa_program, dist_substation_mi, nearest_substation_voltage,
    dist_transmission_mi, nearest_transmission_kv, source, geometry (EPSG:4326)

Run: python ingestion_scripts/epa_repowering/ingest_epa_repowering.py
"""
import re
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

XLSX = Path('ingestion_scripts/epa_repowering/cache/re_powering_screening.xlsx')
OUT_PATH = Path('ingestion_scripts/epa_repowering/epa_repowering_sites.parquet')

TARGET_STATES = {'AZ', 'CA', 'NV', 'TX', 'VA'}

# Anything bigger than this is a base/region, not a reuse parcel.
MAX_REUSE_ACRES = 5000.0

# Name patterns for active military land and protected parks/forests -- these
# are not redevelopment candidates (they are GAP-3/4 federal land).
NON_REUSE_PATTERNS = re.compile(
    r'\b(NAVAL|NAVY|NWS|NAS|ARMY|USMC|MARINE CORPS|AIR FORCE|AFB|MILITARY|'
    r'WEAPONS STATION|PROVING GROUND|ARTILLERY|BOMBING RANGE|GUNNERY RANGE|'
    r'TRAINING (CENTER|CTR|RANGE)|FORT |FT\.? |GOLDWATER RANGE|'
    r'CAMP |ARSENAL|AMMUNITION|MILITARY DEPOT|MCAAP|'
    r'NATIONAL PARK|NATIONAL FOREST|NATIONAL MONUMENT|'
    r'NATIONAL RECREATION AREA|NATIONAL WILDLIFE)\b',
    re.IGNORECASE,
)


def main():
    print(f'Reading {XLSX} (RE-Powering Sites sheet, ~190k rows) ...')
    df = pd.read_excel(XLSX, sheet_name='RE-Powering Sites')
    print(f'  {len(df):,} sites nationally')

    # Filter to target states
    df = df[df['State'].isin(TARGET_STATES)].copy()
    print(f'  {len(df):,} sites in 5 target states')

    # Drop rows with no coordinates
    df = df[df['Latitude'].notna() & df['Longitude'].notna()].copy()
    print(f'  {len(df):,} sites with coordinates')

    # Drop active military installations and national parks/forests
    name_upper = df['Site Name'].astype(str)
    mil_park = name_upper.apply(lambda s: bool(NON_REUSE_PATTERNS.search(s)))
    print(f'  Dropping {mil_park.sum():,} military / park / national-forest sites')
    df = df[~mil_park].copy()

    # Drop sites larger than the reuse-parcel cap (bases / regions slip past the
    # name filter, e.g. unnamed ranges)
    acres_num = pd.to_numeric(df['Acreage (Acres)'], errors='coerce')
    too_big = acres_num > MAX_REUSE_ACRES
    print(f'  Dropping {too_big.sum():,} sites over {MAX_REUSE_ACRES:,.0f} ac (not reuse parcels)')
    df = df[~too_big].copy()
    print(f'  {len(df):,} reuse-node candidate sites remain')

    # Classify reuse_asset_type
    def _asset_type(r):
        landfill = str(r.get('Known Landfill', '')).strip().lower()
        mine     = str(r.get('Known Abandoned Mine Land', '')).strip().lower()
        if mine in ('yes', 'y', 'true'):
            return 'abandoned_mine'
        if landfill in ('yes', 'y', 'true'):
            return 'landfill'
        return 'contaminated_land'

    df['reuse_asset_type'] = df.apply(_asset_type, axis=1)

    # Use Cross-Reference Number as the unique site_id (Site ID repeats across
    # EPA programs); fall back to a row-index suffix if any remain duplicated.
    out = pd.DataFrame({
        'site_id':                     df['Cross-Reference Number'].astype(str),
        'site_name':                   df['Site Name'],
        'state_abbr':                  df['State'],
        'county':                      df['County'],
        'city':                        df['City'],
        'reuse_asset_type':            df['reuse_asset_type'],
        'acreage':                     pd.to_numeric(df['Acreage (Acres)'], errors='coerce'),
        'epa_program':                 df['Program'],
        'dist_substation_mi':          pd.to_numeric(df['Distance to Nearest Substation (miles)'], errors='coerce'),
        'nearest_substation_voltage':  pd.to_numeric(df['Nearest Substation Voltage (Volts)'], errors='coerce'),
        'dist_transmission_mi':        pd.to_numeric(df['Distance to Nearest Transmission Line (miles)'], errors='coerce'),
        'nearest_transmission_kv':     pd.to_numeric(df['Nearest Transmission Line kV (kilovolts)'], errors='coerce'),
        'source':                      'EPA-RE-Powering-2022',
        'latitude':                    pd.to_numeric(df['Latitude'], errors='coerce'),
        'longitude':                   pd.to_numeric(df['Longitude'], errors='coerce'),
    })

    # Drop rows where coordinate parse failed
    out = out[out['latitude'].notna() & out['longitude'].notna()].copy()

    # Sanity-clip coordinates to a generous CONUS box (drops bad geocodes)
    bad = ~(out['latitude'].between(24, 50) & out['longitude'].between(-126, -66))
    if bad.sum():
        print(f'  Dropping {bad.sum()} sites with out-of-CONUS coordinates')
        out = out[~bad].copy()

    # Guarantee site_id uniqueness (suffix any remaining duplicates)
    if not out['site_id'].is_unique:
        dup = out['site_id'].duplicated(keep=False)
        out.loc[dup, 'site_id'] = (
            out.loc[dup, 'site_id'] + '_' + out.loc[dup].groupby('site_id').cumcount().astype(str)
        )

    gdf = gpd.GeoDataFrame(
        out,
        geometry=[Point(xy) for xy in zip(out['longitude'], out['latitude'])],
        crs='EPSG:4326',
    ).drop(columns=['latitude', 'longitude'])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(OUT_PATH, index=False)
    print(f'\nSaved: {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.0f} KB, {len(gdf):,} sites)')

    # ---- Summary ----
    print('\n=== Sites by state x asset type ===')
    print(gdf.groupby(['state_abbr', 'reuse_asset_type']).size().unstack(fill_value=0).to_string())
    print('\n=== Acreage distribution ===')
    ac = gdf.acreage.dropna()
    if len(ac):
        print(f'  sites with acreage: {len(ac):,} of {len(gdf):,}')
        print(f'  min={ac.min():.1f}, median={ac.median():.1f}, p90={ac.quantile(0.9):.0f}, max={ac.max():,.0f}')
    print('\n=== EPA program breakdown (top 10) ===')
    print(gdf.epa_program.value_counts().head(10).to_string())
    print('\n=== Largest 10 sites by acreage ===')
    top = gdf.dropna(subset=['acreage']).nlargest(10, 'acreage')
    for _, r in top.iterrows():
        print(f'  {str(r.site_name)[:38]:<38} {r.state_abbr}  {r.acreage:>10,.0f} ac  {r.reuse_asset_type}')

    # ---- Checks ----
    print('\n=== Checks ===')
    checks = {
        'Has rows'              : len(gdf) > 0,
        'All in target states'  : set(gdf.state_abbr.unique()) <= TARGET_STATES,
        'reuse_asset_type valid': set(gdf.reuse_asset_type.unique()) <= {'abandoned_mine', 'landfill', 'contaminated_land'},
        'geometry non-null'     : gdf.geometry.notna().all(),
        'CRS 4326'              : gdf.crs.to_epsg() == 4326,
        'site_id unique'        : gdf.site_id.is_unique,
    }
    ok = True
    for lbl, passed in checks.items():
        print(f'  [{"PASS" if passed else "FAIL"}] {lbl}')
        if not passed:
            ok = False
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
