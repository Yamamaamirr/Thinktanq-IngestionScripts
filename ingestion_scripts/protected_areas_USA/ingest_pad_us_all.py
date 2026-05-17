"""
PAD-US 4.1 ingestion - ALL GAP statuses (1, 2, 3, 4) for scoring adjacency.

Sibling of ingest_pad_us.py (which keeps only GAP 1/2 for hard exclusion).
This script reads the same per-state GDBs but keeps every GAP_Sts level so the
scoring engine can measure distance from each candidate to ANY protected land,
not just the strictest-protected subset.

Why we want all levels:
  GAP 1  = managed for biodiversity, disturbance allowed (highest protection)
  GAP 2  = managed for biodiversity, disturbance suppressed
  GAP 3  = managed for multiple uses (national forests, BLM general)
  GAP 4  = no known mandate for biodiversity (private easements, etc.)
  Unknown / blank — also captured as a separate flag

For scoring adjacency, a candidate near GAP 3 or 4 still has development
friction (community attitudes, jurisdiction overlays, etc.). We want all of it.

Output: ingestion_scripts/protected_areas_USA/pad_us_all.parquet (EPSG:5070)

Run:  python ingestion_scripts/protected_areas_USA/ingest_pad_us_all.py
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path

DATA_DIR = Path('ingestion_scripts/protected_areas_USA/data')
OUT_PATH = Path('ingestion_scripts/protected_areas_USA/pad_us_all.parquet')

STATES = {
    'AZ': 'PADUS4_1_State_AZ_GDB_KMZ/PADUS4_1_StateAZ.gdb',
    'CA': 'PADUS4_1_State_CA_GDB_KMZ/PADUS4_1_StateCA.gdb',
    'NV': 'PADUS4_1_State_NV_GDB_KMZ/PADUS4_1_StateNV.gdb',
    'TX': 'PADUS4_1_State_TX_GDB_KMZ/PADUS4_1_StateTX.gdb',
    'VA': 'PADUS4_1_State_VA_GDB_KMZ/PADUS4_1_StateVA.gdb',
}

LAYER_TMPL = 'PADUS4_1Comb_DOD_Trib_NGP_Fee_Desig_Ease_State_{state}'

# Attributes worth keeping for downstream context
KEEP_COLS = [
    'GAP_Sts', 'd_GAP_Sts',     # GAP status code + description
    'Unit_Nm',                   # protected unit name
    'Mang_Type', 'd_Mang_Typ',   # manager type (federal/state/tribal/etc.)
    'Mang_Name', 'd_Mang_Nam',   # managing agency
    'IUCN_Cat', 'd_IUCN_Cat',    # IUCN protected-area category
    'Pub_Access', 'd_Pub_Acc',   # public access classification
    'Des_Tp', 'd_Des_Tp',        # designation type (Wilderness, NF, etc.)
    'geometry',
]

all_parts = []

for state, rel_path in STATES.items():
    gdb = DATA_DIR / rel_path
    layer = LAYER_TMPL.format(state=state)
    print(f'\n-- {state} ----------------------------------------')
    print(f'  Reading {layer} ...')

    gdf = gpd.read_file(gdb, layer=layer)
    print(f'  Total rows: {len(gdf)} | CRS: {gdf.crs.to_epsg() or "ESRI:102039"}')

    available = [c for c in KEEP_COLS if c in gdf.columns]
    missing = [c for c in KEEP_COLS if c not in gdf.columns]
    if missing:
        print(f'  Missing columns (skipped): {missing}')

    kept = gdf[available].copy()
    kept['state'] = state

    gap_counts = kept['GAP_Sts'].value_counts(dropna=False).to_dict()
    print(f'  GAP distribution: {gap_counts}')

    kept = kept.to_crs('EPSG:5070')
    kept = kept[kept.geometry.is_valid & ~kept.geometry.is_empty].copy()
    area_km2 = kept.geometry.area.sum() / 1e6
    print(f'  Area in EPSG:5070: {area_km2:,.0f} km2')

    all_parts.append(kept)

print('\nConcatenating all states ...')
result = gpd.GeoDataFrame(pd.concat(all_parts, ignore_index=True), crs='EPSG:5070')
print(f'  Total rows: {len(result):,}')
print(f'  GAP distribution overall: {result["GAP_Sts"].value_counts(dropna=False).to_dict()}')

total_km2 = result.geometry.area.sum() / 1e6
print(f'  Total area: {total_km2:,.0f} km2')

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUT_PATH, index=False)
size_mb = OUT_PATH.stat().st_size / 1e6
print(f'\nSaved: {OUT_PATH} ({size_mb:.1f} MB)')

# ---- Checks ----
print('\n=== Checks ===')
checks = {
    'Has rows'              : len(result) > 0,
    'CRS EPSG:5070'         : result.crs.to_epsg() == 5070,
    'No null geometry'      : result.geometry.isna().sum() == 0,
    'No empty geometry'     : (~result.geometry.is_empty).all(),
    'All 5 states'          : set(result['state'].unique()) == {'AZ','CA','TX','NV','VA'},
    'Has GAP 1 polygons'    : (result['GAP_Sts'] == '1').sum() > 0,
    'Has GAP 2 polygons'    : (result['GAP_Sts'] == '2').sum() > 0,
    'Has GAP 3 polygons'    : (result['GAP_Sts'] == '3').sum() > 0,
    'Has GAP 4 polygons'    : (result['GAP_Sts'] == '4').sum() > 0,
    'Area > 0'              : total_km2 > 0,
}
all_pass = True
for lbl, ok in checks.items():
    print(f'  [{"PASS" if ok else "FAIL"}] {lbl}')
    if not ok:
        all_pass = False

print('\nAll checks passed.' if all_pass else '\nWARNING: some checks failed.')
print(f'\nNext: scoring engine will use this file for PAD-US adjacency distance.')
