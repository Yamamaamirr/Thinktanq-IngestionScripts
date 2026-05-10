"""
PAD-US 4.1 ingestion - GAP Status 1 and 2 only.

Reads per-state GDBs from ingestion_scripts/protected_areas_USA/data/.
Uses the Combined layer (Fee + Easement + Designation + Proclamation).
Filters to GAP_Sts IN ('1', '2') — hard exclusions per PRD.
GAP 3 and 4 are NOT excluded (managed multi-use / no mandate).

Output: ingestion_scripts/protected_areas_USA/pad_us_gap12.parquet (EPSG:5070)

Run:  python ingestion_scripts/protected_areas_USA/ingest_pad_us.py
"""

import geopandas as gpd
from pathlib import Path

DATA_DIR = Path('ingestion_scripts/protected_areas_USA/data')
OUT_PATH = Path('candidate_areas/exclusion_layers/exclusion_padus.parquet')

STATES = {
    'AZ': 'PADUS4_1_State_AZ_GDB_KMZ/PADUS4_1_StateAZ.gdb',
    'CA': 'PADUS4_1_State_CA_GDB_KMZ/PADUS4_1_StateCA.gdb',
    'NV': 'PADUS4_1_State_NV_GDB_KMZ/PADUS4_1_StateNV.gdb',
    'TX': 'PADUS4_1_State_TX_GDB_KMZ/PADUS4_1_StateTX.gdb',
    'VA': 'PADUS4_1_State_VA_GDB_KMZ/PADUS4_1_StateVA.gdb',
}

LAYER_TMPL = 'PADUS4_1Comb_DOD_Trib_NGP_Fee_Desig_Ease_State_{state}'

all_parts = []

for state, rel_path in STATES.items():
    gdb = DATA_DIR / rel_path
    layer = LAYER_TMPL.format(state=state)
    print(f'\n-- {state} ----------------------------------------')
    print(f'  Reading {layer} ...')

    gdf = gpd.read_file(gdb, layer=layer)
    print(f'  Total rows: {len(gdf)} | CRS: {gdf.crs.to_epsg() or "ESRI:102039"}')

    gap12 = gdf[gdf['GAP_Sts'].isin(['1', '2'])][['GAP_Sts', 'd_GAP_Sts', 'Unit_Nm', 'geometry']].copy()
    gap12['state'] = state
    print(f'  GAP 1+2 rows: {len(gap12)} | '
          f'GAP 1: {(gap12["GAP_Sts"]=="1").sum()} | '
          f'GAP 2: {(gap12["GAP_Sts"]=="2").sum()}')

    gap12 = gap12.to_crs('EPSG:5070')
    area_km2 = gap12.geometry.area.sum() / 1e6
    print(f'  Area (EPSG:5070): {area_km2:,.0f} km2')

    all_parts.append(gap12)

print('\nConcatenating all states ...')
result = gpd.GeoDataFrame(
    __import__('pandas').concat(all_parts, ignore_index=True),
    crs='EPSG:5070'
)
result = result[result.geometry.is_valid & ~result.geometry.is_empty].copy()

total_km2 = result.geometry.area.sum() / 1e6
print(f'  Total rows: {len(result)} | Total area: {total_km2:,.0f} km2')

result.to_parquet(OUT_PATH, index=False)
print(f'\nSaved: {OUT_PATH}')

checks = {
    'Has rows'         : len(result) > 0,
    'CRS EPSG:5070'    : result.crs.to_epsg() == 5070,
    'No null geometry' : result.geometry.isna().sum() == 0,
    'No empty geometry': (~result.geometry.is_empty).all(),
    'Only GAP 1+2'     : result['GAP_Sts'].isin(['1', '2']).all(),
    'All 5 states'     : set(result['state'].unique()) == {'AZ', 'CA', 'TX', 'NV', 'VA'},
    'Area > 0'         : total_km2 > 0,
}
all_pass = True
for lbl, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
    if not ok:
        all_pass = False
print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
print('\nOutput: candidate_areas/exclusion_layers/exclusion_padus.parquet
Next: python candidate_areas/Step5f_subtract_padus.py')
