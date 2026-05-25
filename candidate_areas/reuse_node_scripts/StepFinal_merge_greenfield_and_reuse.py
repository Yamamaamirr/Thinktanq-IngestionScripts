"""
StepFinal -- Merge greenfield + reuse-node enriched/scored tables into
            the unified Phase 1 candidates deliverable.

Per Owen's PRD: one final table with candidate_type = 'greenfield' |
'reuse_node', so the scoring engine + downstream tooling can consume
both layers from a single source.

Reads:
  candidate_areas/outputs/candidates_final.parquet           (86,187 greenfield -- the
                                                              SHIPPED Phase 1 deliverable
                                                              after Step3B slope hard-
                                                              exclusion. Use this, NOT
                                                              candidate_areas_enriched
                                                              which still contains the
                                                              ~39k >15% slope rows that
                                                              were filtered out.)
  candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet
                                                            (6.6k reuse, ~115 cols)

Writes:
  candidate_areas/outputs/final_candidates_phase1.parquet
    ~132k rows, schema = union of both columns. Greenfield rows have
    reuse-only columns set to NA, reuse rows have greenfield-only columns
    set to NA. candidate_type distinguishes them everywhere downstream.

Run: python candidate_areas/reuse_node_scripts/StepFinal_merge_greenfield_and_reuse.py
"""
from pathlib import Path
import pandas as pd
import geopandas as gpd

GREENFIELD_PATH = Path('candidate_areas/outputs/candidates_final.parquet')
REUSE_PATH      = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')
OUT_PATH        = Path('candidate_areas/outputs/final_candidates_phase1.parquet')


def main():
    print(f'Loading greenfield: {GREENFIELD_PATH} ...')
    gf = gpd.read_parquet(GREENFIELD_PATH)
    gf_input_order = list(gf.columns)   # snapshot before we mutate it
    print(f'  {len(gf):,} rows, {len(gf.columns)} columns, CRS={gf.crs.to_epsg()}')

    print(f'\nLoading reuse nodes: {REUSE_PATH} ...')
    rn = gpd.read_parquet(REUSE_PATH)
    print(f'  {len(rn):,} rows, {len(rn.columns)} columns, CRS={rn.crs.to_epsg()}')

    # Both files use EPSG:5070; confirm.
    assert gf.crs.to_epsg() == 5070, f'greenfield CRS is {gf.crs.to_epsg()}, expected 5070'
    assert rn.crs.to_epsg() == 5070, f'reuse-node CRS is {rn.crs.to_epsg()}, expected 5070'

    # Greenfield already has candidate_type='greenfield'. Tag reuse-node rows.
    if 'candidate_type' not in rn.columns:
        rn = rn.copy()
        rn['candidate_type'] = 'reuse_node'
    else:
        rn['candidate_type'] = rn['candidate_type'].fillna('reuse_node')

    # Populate run-metadata on reuse rows so they aren't blank in the unified
    # table. Most metadata columns reflect UPSTREAM dataset versions that R3
    # consumed (same NWI / FEMA / PADUS / transmission / queue / DEM parquets
    # the greenfield pipeline used), so they should carry the same values.
    # Only run_id / run_date / scoring_model_version / snapshot_date are
    # reuse-specific. cdl_year stays NULL because reuse doesn't use CDL.
    import uuid
    from datetime import date
    reuse_run_id = str(uuid.uuid4())
    today_iso    = date.today().isoformat()

    # Copy upstream dataset versions from greenfield (first non-null value).
    def _gf_value(col):
        s = gf[col].dropna()
        return s.iloc[0] if len(s) else None

    SHARED_FROM_GF = [
        'exclusion_model_version',  # same exclusion masks reused
        'padus_version',
        'fema_nfhl_date',
        'nwi_date',
        'transmission_dataset_version',
        'queue_dataset_date',
        'dem_dataset_version',
    ]
    REUSE_SPECIFIC = {
        'run_id':                reuse_run_id,
        'run_date':              today_iso,
        'snapshot_date':         today_iso,
        'scoring_model_version': '1.0.0-phase1-reuse',  # recalibrated weights
    }

    print('\nPopulating run-metadata on reuse rows ...')
    for col in SHARED_FROM_GF:
        if col in gf.columns:
            val = _gf_value(col)
            rn[col] = val
            print(f'  {col:<32} <- gf value: {val}')
    for col, val in REUSE_SPECIFIC.items():
        rn[col] = val
        print(f'  {col:<32} = {val}')
    # cdl_year intentionally left NULL on reuse (not applicable)
    print('  cdl_year                         = NULL  (not applicable to reuse nodes)')

    # Derive area_m2 + centroid_lon/lat from geometry (these are computable
    # for every reuse polygon; only pixel_count stays NULL because it's
    # CDL-pixel-specific and reuse polygons aren't CDL-derived).
    print('\nBackfilling derivable fields on reuse rows ...')
    rn_4326_cents = rn.geometry.centroid.to_crs(4326)
    rn['area_m2']      = rn.geometry.area
    rn['centroid_lon'] = rn_4326_cents.x.values
    rn['centroid_lat'] = rn_4326_cents.y.values
    print(f'  area_m2          populated: {int(rn.area_m2.notna().sum()):,}/{len(rn)}')
    print(f'  centroid_lon/lat populated: {int(rn.centroid_lon.notna().sum()):,}/{len(rn)}')

    # Spatial-join county_fips from TIGER county boundaries
    county_path = Path('ingestion_scripts/census_tiger/county_boundaries.parquet')
    if county_path.exists():
        county = gpd.read_parquet(county_path)
        if county.crs.to_epsg() != 4326:
            county = county.to_crs(4326)
        cents_gdf = gpd.GeoDataFrame(
            {'_idx': range(len(rn))},
            geometry=rn_4326_cents.values, crs='EPSG:4326',
        )
        joined = gpd.sjoin(cents_gdf, county[['GEOID','NAME','geometry']],
                           how='left', predicate='within')
        joined = joined.drop_duplicates(subset='_idx', keep='first')
        joined = joined.set_index('_idx').reindex(range(len(rn)))
        rn['county_fips'] = joined['GEOID'].values
        # Also backfill the 2 county_name nulls we observed
        missing_name = rn.county_name.isna()
        rn.loc[missing_name, 'county_name'] = joined.loc[missing_name.values, 'NAME'].values
        print(f'  county_fips      populated: {int(rn.county_fips.notna().sum()):,}/{len(rn)}')
        print(f'  county_name      now non-null: {int(rn.county_name.notna().sum()):,}/{len(rn)}')
    else:
        print(f'  WARN: {county_path} not found; county_fips left NULL')

    # Schema union: pad each side with the other's missing columns.
    gf_cols = set(gf.columns)
    rn_cols = set(rn.columns)
    shared    = gf_cols & rn_cols
    only_gf   = gf_cols - rn_cols
    only_rn   = rn_cols - gf_cols
    print(f'\nSchema:')
    print(f'  shared columns:        {len(shared):>3}')
    print(f'  greenfield-only:       {len(only_gf):>3}')
    print(f'  reuse-only:            {len(only_rn):>3}')

    # Reconcile candidate_id types if needed (both should be str already)
    if gf.candidate_id.dtype != rn.candidate_id.dtype:
        print(f'  WARN: candidate_id dtypes differ '
              f'(gf={gf.candidate_id.dtype}, rn={rn.candidate_id.dtype}); coercing to str')
        gf['candidate_id'] = gf['candidate_id'].astype(str)
        rn['candidate_id'] = rn['candidate_id'].astype(str)

    # Pad each side with the other's missing columns set to NA
    for col in only_rn:
        gf[col] = pd.NA
    for col in only_gf:
        rn[col] = pd.NA

    # ------------------------------------------------------------------
    # Column ordering
    # ------------------------------------------------------------------
    # The greenfield deliverable (candidates_final.fgb) ships with a specific
    # column order Owen already reviewed; we must NOT scramble it. We start
    # with that exact order and then insert reuse-only columns at semantically
    # adjacent positions so:
    #   - reuse identity (source/asset_type/etc) sits next to candidate_type
    #   - reuse footprint metadata sits next to area_acres / parent_candidate_id
    #   - R4 flags sit with the identity block (derived from source/asset_type)
    #   - reuse_environmental_score sits next to the other subscores (dev_risk)
    # Result: composite_score / recommended_action stay where Owen expects.
    INSERT_AFTER = {
        'candidate_type': [
            'site_name', 'source', 'reuse_asset_type', 'reuse_status',
            'capacity_mw', 'acreage', 'retirement_year',
            'dominant_technology', 'epa_program',
            'environmental_review_required', 'legacy_asset_risk_flag',
            'decommissioning_status_known', 'known_contamination_flag',
        ],
        'parent_candidate_id': [
            'geometry_source', 'geometry_confidence', 'matched_osm_id',
            'aliased_site_ids', 'aliased_site_count',
        ],
        'dev_risk_score': [
            'reuse_environmental_score',
        ],
    }

    all_cols = []
    for col in gf_input_order:
        all_cols.append(col)
        if col in INSERT_AFTER:
            for new_col in INSERT_AFTER[col]:
                # Only insert if the reuse-only column actually exists
                if new_col in only_rn:
                    all_cols.append(new_col)

    # Defensive: catch any reuse-only column we forgot to place (should be empty)
    placed = set(all_cols)
    missing = (gf_cols | rn_cols) - placed
    if missing:
        print(f'  WARN: appending {len(missing)} unplaced columns at end: {sorted(missing)}')
        all_cols.extend(sorted(missing))

    gf = gf[all_cols]
    rn = rn[all_cols]

    print('\nConcatenating ...')
    out = pd.concat([gf, rn], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry='geometry', crs=gf.crs)
    print(f'  {len(out):,} rows, {len(out.columns)} columns')

    # ID uniqueness across the union: candidate_id namespaces differ
    # (greenfield uses UUID-style, reuse uses EIA-/EPA-/NRC-/OSM- prefixes).
    # Should not collide.
    dup = int(out.candidate_id.duplicated().sum())
    if dup > 0:
        # If there are dupes, rename reuse-node ids with a prefix to disambiguate
        print(f'  WARN: {dup} duplicate candidate_ids across the union; prefixing reuse')
        is_reuse = out.candidate_type == 'reuse_node'
        out.loc[is_reuse, 'candidate_id'] = 'REUSE-' + out.loc[is_reuse, 'candidate_id']
        dup = int(out.candidate_id.duplicated().sum())
        print(f'  after prefix: {dup} duplicates remain')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f'\nSaved: {OUT_PATH} ({size_mb:.1f} MB)')

    # ---- Summary ----
    print('\n=== Final unified table ===')
    print(f'Total rows:        {len(out):,}')
    print(f'  greenfield:      {(out.candidate_type=="greenfield").sum():>7,}')
    print(f'  reuse_node:      {(out.candidate_type=="reuse_node").sum():>7,}')
    print(f'\nBy state:')
    print(out.groupby(['state','candidate_type']).size().unstack(fill_value=0).to_string()
          if 'state' in out.columns else '  (no state column)')
    print(f'\nBy recommended_action:')
    print(out.groupby(['recommended_action','candidate_type']).size().unstack(fill_value=0).to_string())

    print(f'\nComposite score quantiles per layer:')
    for ct in ['greenfield','reuse_node']:
        s = out[out.candidate_type == ct].composite_score
        print(f'  {ct:<12} p10={s.quantile(0.1):.1f}  median={s.median():.1f}  '
              f'p90={s.quantile(0.9):.1f}  max={s.max():.1f}')

    # ---- Checks ----
    print('\n=== Checks ===')
    checks = {
        'Has rows'                   : len(out) > 0,
        'candidate_id unique'        : out.candidate_id.is_unique,
        'candidate_type non-null'    : out.candidate_type.notna().all(),
        'candidate_type valid set'   : set(out.candidate_type.unique()) == {'greenfield','reuse_node'},
        'CRS = EPSG:5070'            : out.crs.to_epsg() == 5070,
        'composite_score non-null'   : out.composite_score.notna().all(),
        'recommended_action non-null': out.recommended_action.notna().all(),
        'row count = sum of inputs'  : len(out) == len(gf) + len(rn),
    }
    ok = True
    for lbl, p in checks.items():
        print(f'  [{"PASS" if p else "FAIL"}] {lbl}')
        ok = ok and p
    print('\nAll checks passed.' if ok else '\nWARNING: some checks failed.')


if __name__ == '__main__':
    main()
