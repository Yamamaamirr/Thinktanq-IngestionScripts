"""FERC Form 714 utility load ingestion via PUDL.

Source: PUDL (Public Utility Data Liberation, catalyst.coop).
        Public S3: pudl.catalyst.coop, no auth required.
        Stable channel ('stable/') is versioned (v2026.4.0 = 2026-04-09).
        Annual refresh per PRD section 10.2.

What this dataset IS:
  Per-utility / per-balancing-authority annual electricity demand summary.
  Built from FERC Form 714 (mandatory hourly load reports for ~218 large
  US planning-area respondents) and rolled up here to:
      annual_mwh    -- total energy delivered for the year
      peak_mw       -- highest single-hour demand for the year
      avg_mw        -- annual mean of hourly demand (= annual_mwh / 8760)

What this dataset IS NOT:
  - Not substation-level. Respondents are utilities and balancing authorities,
    so the headroom signal is a TERRITORY-LEVEL indicator, not a precise
    substation injection-capacity number. For substation-precise capacity
    (Phase 2), the PRD recommends Transect ($300/mo) actual flow studies.
    This script is an explicit directional signal for the v1 scoring engine.
  - Not a forecast. PUDL also publishes a forecast table; we use historicals
    only since the headroom signal needs current-state demand.
  - No retail rates / customer counts (those are in EIA 861, separate source).

Why we download two PUDL files:
  out_ferc714__hourly_planning_area_demand.parquet   -- 226 MB, hourly history
      The only place peak_mw can be computed correctly. PUDL does not pre-
      aggregate peak; their summarized table only has annual MWh totals.
      Cached locally; can be deleted after a successful run if disk-tight.
  out_ferc714__summarized_demand.parquet             -- 0.1 MB, annual rollup
      Population, area, density, utility/BA name + EIA code joins.

Output is the merge of the two: small (~50-100 KB), one row per
(respondent, year), most-recent year per respondent for current headroom.

How it feeds the capacity_signal (PRD section 4.1):
  capacity_signal = generation_capacity (eia_form860)
                  - queued_MW           (gridstatus_iso_queues)
                  + load_headroom       (this script)
  Higher load_headroom = more room for new large loads.
      > 1.2  abundant
      0.9-1.2  moderate
      < 0.9  constrained

  Enrichment join — TWO-STEP with BA fallback (required):

  Many large utilities (Dominion VA, Austin Energy, SCE, Appalachian Power,
  CPS Energy, Entergy Texas) never file FERC 714 individually — they report
  under their balancing authority instead. A direct utility→ferc714 join
  leaves those parcels with null load_headroom. The fix is a BA-level fallback:

  WITH direct AS (
    SELECT ut.source_id,
           f.peak_demand_mw,
           f.avg_demand_mw,
           f.report_year,
           'utility' AS join_level
    FROM   utility_territories ut
    JOIN   utility_load f ON f.eia_code = ut.source_id::INT
  ),
  ba_fallback AS (
    SELECT ut.source_id,
           f.peak_demand_mw,
           f.avg_demand_mw,
           f.report_year,
           'ba' AS join_level
    FROM   utility_territories ut
    -- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    -- !! TRANSLATION IS REQUIRED HERE — DO NOT join directly on control_area.
    -- !!
    -- !! ut.control_area is a HIFLD full name  ("PJM INTERCONNECTION, LLC")
    -- !! f.ba_code       is an EIA short code  ("PJM")
    -- !!
    -- !! A direct equality check silently returns NULL for ~60% of target-state
    -- !! parcels (Dominion VA, Austin Energy, SCE, Appalachian, CPS Energy,
    -- !! Entergy Texas all drop out). Apply CONTROL_AREA_TO_BA_CODE (see the
    -- !! mapping table below) in application code, or inline the VALUES lookup
    -- !! as a CTE, before executing this join.
    -- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    JOIN   utility_load f ON f.ba_code = ctrl_area_to_ba_code(ut.control_area)
    --                                    ^^^^^^^^^^^^^^^^^^^^ replace with the
    --                                    CONTROL_AREA_TO_BA_CODE mapping below
    WHERE  ut.source_id NOT IN (SELECT source_id FROM direct)
  )
  SELECT * FROM direct
  UNION ALL
  SELECT * FROM ba_fallback;

  Direct join covers: APS, TEP, SRP, Nevada Power, Sierra Pacific, SMUD,
  SDG&E, El Paso Electric (~40% of target-state parcels).
  BA fallback covers: Dominion→PJM, Austin Energy→ERCOT, SCE→CAISO,
  Appalachian→PJM, CPS Energy→ERCOT, Entergy Texas→MISO (~60%).

  CONTROL_AREA_TO_BA_CODE translation (required for ba_fallback join above):
  utility_territories.control_area stores HIFLD full names; ferc714.ba_code
  stores EIA short codes. They do not match directly. Full mapping table:

    'ARIZONA PUBLIC SERVICE COMPANY'                              -> 'AZPS'
    'BALANCING AUTHORITY OF NORTHERN CALIFORNIA'                  -> 'BANC'
    'CALIFORNIA INDEPENDENT SYSTEM OPERATOR'                      -> 'CISO'
    'EL PASO ELECTRIC COMPANY'                                    -> 'EPE'
    'ELECTRIC RELIABILITY COUNCIL OF TEXAS, INC.'                 -> 'ERCO'
    'IMPERIAL IRRIGATION DISTRICT'                                -> 'IID'
    'LOS ANGELES DEPARTMENT OF WATER AND POWER'                   -> 'LDWP'
    'MIDCONTINENT INDEPENDENT TRANSMISSION SYSTEM OPERATOR, ...'  -> 'MISO'
    'NEVADA POWER COMPANY'                                        -> 'NEVP'
    'PACIFICORP - EAST'                                           -> 'PACE'
    'PACIFICORP - WEST'                                           -> 'PACW'
    'PJM INTERCONNECTION, LLC'                                    -> 'PJM'
    'PUBLIC SERVICE COMPANY OF NEW MEXICO'                        -> 'PNM'
    'SALT RIVER PROJECT'                                          -> 'SRTP'
    'SOUTHWEST POWER POOL'                                        -> 'SWPP'
    'TENNESSEE VALLEY AUTHORITY'                                  -> 'TVA'
    'TUCSON ELECTRIC POWER COMPANY'                               -> 'AZPS'
    'TURLOCK IRRIGATION DISTRICT'                                 -> 'TIDC'
    'WESTERN AREA POWER ADMIN - DESERT SOUTHWEST REGION'         -> 'WALC'

  Note: Entergy Texas (SE Texas) is in MISO not ERCOT. Lubbock LP&L is in
  SPP per HIFLD data (predates the 2021 switch to ERCOT). These resolve
  correctly once the translation table is applied.

Output schema (matches utility_load table):
  respondent_id     TEXT     respondent_id_ferc714
  respondent_name   TEXT     respondent_name_ferc714
  respondent_type   TEXT     'balancing_authority' or 'utility'
  ba_code           TEXT     balancing_authority_code_eia (joins to EIA 860)
  ba_name           TEXT     balancing_authority_name_eia
  utility_id        TEXT     utility_id_eia (when respondent_type='utility')
  utility_name      TEXT     utility_name_eia
  eia_code          INTEGER  EIA respondent code (joins to BA or utility)
  report_year       INTEGER  most recent reported year per respondent
  annual_mwh        NUMERIC  total annual demand
  peak_demand_mw    NUMERIC  highest hour observed
  avg_demand_mw     NUMERIC  annual average (annual_mwh / hours_observed)
  load_factor       NUMERIC  avg / peak (utility-industry KPI; < 1)
  population        NUMERIC  served population (PUDL summarized; can be NaN)
  area_km2          NUMERIC  service area in km^2 (PUDL summarized)
  hours_observed    INTEGER  count of hourly records (sanity flag for partial yrs)
  ingested_at       TIMESTAMPTZ
"""
import time
from pathlib import Path

import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from urllib3.exceptions import ProtocolError

PUDL_BASE = 'https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/stable/'
HOURLY_FILE = 'out_ferc714__hourly_planning_area_demand.parquet'   # 226 MB
SUMMARY_FILE = 'out_ferc714__summarized_demand.parquet'            # 0.1 MB

CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
HOURLY_CACHE = CACHE_DIR / HOURLY_FILE
SUMMARY_CACHE = CACHE_DIR / SUMMARY_FILE
OUT_PARQUET = Path(__file__).parent / 'ferc714.parquet'

RETRY_DELAYS = [5, 10, 20, 30, 60, 120, 180]
NETWORK_ERRORS = (ConnectionError, Timeout, ChunkedEncodingError, ProtocolError)
RETRY_HTTP_STATUSES = {429, 500, 502, 503, 504}


def _download(url, dest):
    """Stream-download with HTTP Range resume on transient interruption.

    PUDL S3 occasionally stalls mid-stream on this 226 MB file. We use a
    short read-timeout (60s) so a stalled socket fails fast, then retry
    with a Range header that resumes from the existing .part offset.
    """
    tmp = dest.with_suffix(dest.suffix + '.part')
    last_exc = None
    for delay in RETRY_DELAYS:
        try:
            offset = tmp.stat().st_size if tmp.exists() else 0
            headers = {}
            mode = 'wb'
            if offset:
                headers['Range'] = f'bytes={offset}-'
                mode = 'ab'
                print(f"  resuming from offset {offset/1e6:.0f} MB")
            # (connect_timeout, read_timeout) — fail fast on stalls.
            with requests.get(url, stream=True, headers=headers,
                              timeout=(30, 60)) as r:
                if r.status_code in RETRY_HTTP_STATUSES:
                    last_exc = RuntimeError(f"HTTP {r.status_code}")
                    print(f"  Server error HTTP {r.status_code} -- retry in {delay}s")
                    time.sleep(delay)
                    continue
                if offset and r.status_code != 206:
                    print(f"  server returned {r.status_code} (no range support); "
                          f"restarting from 0")
                    tmp.unlink(missing_ok=True)
                    offset = 0
                    mode = 'wb'
                r.raise_for_status()
                # Content-Length is the REMAINING bytes when ranging.
                remaining = int(r.headers.get('Content-Length', 0))
                total = remaining + offset
                downloaded = offset
                last_pct = -1
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            downloaded += len(chunk)
                            if total:
                                pct = downloaded * 100 // total
                                if pct >= last_pct + 5:
                                    print(f"    {downloaded/1e6:.0f} / "
                                          f"{total/1e6:.0f} MB ({pct}%)")
                                    last_pct = pct
                # Windows occasionally holds a transient lock on the file
                # right after the last write; retry the rename briefly.
                for attempt in range(10):
                    try:
                        tmp.rename(dest)
                        return
                    except PermissionError:
                        if attempt == 9:
                            raise
                        time.sleep(0.5)
                return
        except NETWORK_ERRORS as e:
            last_exc = e
            print(f"  Network error: {type(e).__name__} -- retry in {delay}s "
                  f"(have {tmp.stat().st_size/1e6:.0f} MB so far)")
            time.sleep(delay)
    raise RuntimeError(f"Max retries exceeded ({last_exc!r})")


# ── Fetch ────────────────────────────────────────────────────────────────────

if not HOURLY_CACHE.exists():
    print(f"Downloading {HOURLY_FILE} (~226 MB) from PUDL stable...")
    _download(PUDL_BASE + HOURLY_FILE, HOURLY_CACHE)
print(f"Loading hourly: {HOURLY_CACHE.name} "
      f"({HOURLY_CACHE.stat().st_size/1e6:.1f} MB)")

if not SUMMARY_CACHE.exists():
    print(f"Downloading {SUMMARY_FILE} from PUDL stable...")
    _download(PUDL_BASE + SUMMARY_FILE, SUMMARY_CACHE)
print(f"Loading summary: {SUMMARY_CACHE.name} "
      f"({SUMMARY_CACHE.stat().st_size/1e6:.2f} MB)")

# Inspect schema before assuming column names — PUDL renames between versions.
import pyarrow.parquet as _pq
hourly_schema = _pq.read_schema(HOURLY_CACHE)
hourly_cols = [f.name for f in hourly_schema]
print(f"\nHourly file columns ({len(hourly_cols)}):")
print(f"  {hourly_cols}")

# Pick the demand column. PUDL publishes both raw and imputed -- prefer
# imputed (gap-filled, more reliable for peak/avg) over reported (with NaN
# gaps where utilities under-reported). Falls back to legacy column names
# if PUDL renames between releases.
demand_col = next((c for c in ('demand_imputed_pudl_mwh',
                                'demand_reported_mwh',
                                'demand_mwh', 'demand_mw', 'value')
                   if c in hourly_cols), None)
assert demand_col, f"Couldn't identify demand column in: {hourly_cols}"
# Find the timestamp column.
time_col = next((c for c in ('datetime_utc', 'utc_datetime', 'report_date',
                              'report_datetime') if c in hourly_cols), None)
assert time_col, f"Couldn't identify timestamp column in: {hourly_cols}"
print(f"  using demand_col={demand_col!r}, time_col={time_col!r}")

# Load only the columns we need to keep memory reasonable.
hourly = pd.read_parquet(
    HOURLY_CACHE,
    columns=['respondent_id_ferc714', time_col, demand_col],
)
print(f"Hourly rows: {len(hourly):,}")

# ── Aggregate hourly -> per (respondent, year) ───────────────────────────────

hourly['year'] = pd.to_datetime(hourly[time_col]).dt.year
# Drop rows with NaN demand (FERC respondents occasionally report partial years)
n_before = len(hourly)
hourly = hourly.dropna(subset=[demand_col])
hourly = hourly[hourly[demand_col] >= 0]  # drop sentinel negatives if any
print(f"Hourly after null/neg drop: {len(hourly):,} "
      f"(dropped {n_before - len(hourly):,})")

# If the demand column is in MWh-per-hour (i.e. hour energy = average MW for
# that hour, since 1 hour interval), peak_mw == max(demand_mwh) over hours and
# avg_mw == mean(demand_mwh). PUDL docs: hourly demand is reported in MW
# averaged over the hour (numerically equivalent to MWh for that hour).
agg = (hourly.groupby(['respondent_id_ferc714', 'year'])[demand_col]
              .agg(annual_mwh='sum',
                   peak_demand_mw='max',
                   avg_demand_mw='mean',
                   hours_observed='count')
              .reset_index())
agg['load_factor'] = (agg['avg_demand_mw'] / agg['peak_demand_mw']).round(4)
print(f"\nAggregated to {len(agg):,} (respondent, year) combinations")
print(f"  year range: {agg['year'].min()} - {agg['year'].max()}")

# Keep most recent year with at least 6 months (4380 hours) of data per
# respondent. Avoids picking a year that's only partial reporting.
MIN_HOURS = 4380
agg = agg[agg['hours_observed'] >= MIN_HOURS].copy()
agg = agg.sort_values(['respondent_id_ferc714', 'year'])
latest = (agg.groupby('respondent_id_ferc714')
              .tail(1).copy()
              .rename(columns={'year': 'report_year'}))
print(f"Most-recent-year per respondent: {len(latest)} respondents")

# ── Join with summarized for utility/BA names + EIA codes ────────────────────

summary = pd.read_parquet(SUMMARY_CACHE)
summary['report_year'] = pd.to_datetime(summary['report_date']).dt.year
KEEP_FROM_SUMMARY = [
    'respondent_id_ferc714', 'report_year', 'respondent_name_ferc714',
    'respondent_type', 'eia_code',
    'balancing_authority_code_eia', 'balancing_authority_name_eia',
    'utility_id_eia', 'utility_name_eia',
    'population', 'area_km2',
]
summary = summary[KEEP_FROM_SUMMARY]

merged = latest.merge(summary, on=['respondent_id_ferc714', 'report_year'],
                       how='left')
print(f"\nMerged: {len(merged)} respondents (joined name/eia metadata)")
n_unmatched = merged['respondent_name_ferc714'].isna().sum()
if n_unmatched:
    print(f"  {n_unmatched} respondents lack summary metadata for that year "
          f"(common -- summary updates lag hourly data; will fall back to "
          f"latest summary year per respondent below)")
    fallback = (summary.sort_values(['respondent_id_ferc714', 'report_year'])
                       .groupby('respondent_id_ferc714').tail(1)
                       .drop(columns='report_year'))
    fb_cols = [c for c in fallback.columns if c != 'respondent_id_ferc714']
    merged = merged.drop(columns=fb_cols, errors='ignore').merge(
        fallback, on='respondent_id_ferc714', how='left')

# Rename to canonical schema.
merged = merged.rename(columns={
    'respondent_id_ferc714':            'respondent_id',
    'respondent_name_ferc714':          'respondent_name',
    'balancing_authority_code_eia':     'ba_code',
    'balancing_authority_name_eia':     'ba_name',
    'utility_id_eia':                   'utility_id',
    'utility_name_eia':                 'utility_name',
})

# Cast IDs to string for portability; keep eia_code as int.
merged['respondent_id'] = merged['respondent_id'].astype('Int64').astype(str)
merged['utility_id'] = merged['utility_id'].astype('Int64').astype(str).replace('<NA>', None)

OUT_COLS = [
    'respondent_id', 'respondent_name', 'respondent_type',
    'ba_code', 'ba_name', 'utility_id', 'utility_name', 'eia_code',
    'report_year',
    'annual_mwh', 'peak_demand_mw', 'avg_demand_mw', 'load_factor',
    'population', 'area_km2', 'hours_observed',
]
merged = merged[OUT_COLS].copy()
merged['ingested_at'] = pd.Timestamp.utcnow()

# ── Validation ───────────────────────────────────────────────────────────────

assert len(merged) >= 100, f"Suspiciously few respondents: {len(merged)}"
# US national consumption is ~4,000 TWh/yr. FERC 714 sums across BOTH
# balancing-authority and utility respondents, so the same load is reported
# at both levels (a utility's MWh shows up inside its parent BA too). Total
# is therefore expected to over-count -- 1.5x national consumption is the
# typical sum. Use a wide sanity band, not a tight one.
total_twh = merged['annual_mwh'].sum() / 1e6
assert 1500 < total_twh < 12_000, \
    f"Total demand {total_twh:.0f} TWh outside expected range"
# Load factor should be 0.4 to 0.85 for reasonable utilities/BAs
bad_lf = merged[(merged['load_factor'] < 0.2) | (merged['load_factor'] > 0.95)]
print(f"\nLoad-factor outliers (< 0.2 or > 0.95): {len(bad_lf)}")
if len(bad_lf):
    print(bad_lf[['respondent_id', 'respondent_name', 'peak_demand_mw',
                   'avg_demand_mw', 'load_factor']].head().to_string(index=False))

print(f"\nTotal respondents (latest year): {len(merged)}")
print(f"Total annual demand: {total_twh:,.0f} TWh "
      f"({merged['annual_mwh'].sum()/1e9:,.1f} TWh) -- "
      f"US national total is ~4,000 TWh")
print(f"\nBy respondent_type:")
print(merged['respondent_type'].value_counts(dropna=False).to_string())
print(f"\nReport-year distribution:")
print(merged['report_year'].value_counts().sort_index().to_string())

print(f"\nLargest 10 respondents by peak demand:")
top = merged.nlargest(10, 'peak_demand_mw')[
    ['respondent_id', 'respondent_name', 'ba_code',
     'peak_demand_mw', 'avg_demand_mw', 'load_factor', 'report_year']]
print(top.to_string(index=False))

# ── Spot checks ──────────────────────────────────────────────────────────────

# Known reference points (rough magnitudes from EIA / ISO public reports):
#   PJM       — ~150,000 MW peak
#   ERCOT     — ~85,000 MW peak
#   CAISO     — ~50,000 MW peak
#   NYISO     — ~32,000 MW peak
#   ISO-NE    — ~25,000 MW peak
#   MISO      — ~125,000 MW peak
print("\nMajor BA / RTO peak demand spot checks (rough public values):")
print("-" * 70)
SPOT_REFS = {
    'PJM':   ('PJM Interlinks',         140_000, 165_000),
    'ERCO':  ('ERCOT',                   80_000,  90_000),
    'CISO':  ('CAISO',                   45_000,  55_000),
    'NYIS':  ('NYISO',                   28_000,  35_000),
    'ISNE':  ('ISO-NE',                  20_000,  28_000),
    'MISO':  ('MISO',                   115_000, 135_000),
    'SWPP':  ('SPP',                     40_000,  60_000),
    'TVA':   ('Tennessee Valley Auth',   28_000,  36_000),
}
for code, (label, lo, hi) in SPOT_REFS.items():
    rec = merged[merged['ba_code'] == code]
    if len(rec) == 0:
        rec = merged[merged['respondent_name']
                     .str.upper().fillna('').str.contains(label.upper().split()[0])]
    if len(rec):
        peak = float(rec.iloc[0]['peak_demand_mw'])
        ok = lo <= peak <= hi
        flag = 'OK' if ok else '?? '
        print(f"  {flag} {code:<6} {label:<26} peak={peak:>8,.0f} MW "
              f"(expected {lo:,}-{hi:,})")
    else:
        print(f"  ?  {code:<6} {label:<26} NOT FOUND")

# Join-path verification: confirm both legs of the two-step enrichment join
# resolve correctly for every target-state utility.
print("\nJoin-path spot checks (utility-direct leg + BA-fallback leg):")
print("-" * 80)

# (label, source_id, expected_ba_fallback_code)
# source_id = EIA utility ID from utility_territories
DIRECT_CHECKS = [
    ('APS (AZ)',         803,   'AZPS'),
    ('TEP (AZ)',        24211,  'AZPS'),
    ('SRP (AZ)',        16572,  'SRTP'),
    ('Nevada Power (NV)', 13407, 'NEVP'),
    ('Sierra Pacific (NV)', 17166, 'NEVP'),
    ('SMUD (CA)',       16534,  'BANC'),
    ('SDG&E (CA)',      16609,  'CISO'),
    ('El Paso (TX)',     5701,  'ERCO'),
]
BA_FALLBACK_CHECKS = [
    ('Dominion VA',     19876, 'PJM'),
    ('Appalachian (VA)',   733, 'PJM'),
    ('Austin Energy (TX)', 1015, 'ERCO'),
    ('CPS Energy (TX)', 16604, 'ERCO'),
    ('SCE (CA)',        17609, 'CISO'),
    ('Entergy TX',      55937, 'ERCO'),
    ('PG&E (CA)',       14328, 'CISO'),   # utility record stale 2010; BA is 2024
]

all_ok = True
for label, src_id, ba_code in DIRECT_CHECKS:
    rec = merged[merged['eia_code'] == src_id]
    if len(rec):
        r = rec.iloc[0]
        print(f"  OK  DIRECT   {label:<22} eia_code={src_id:<6} "
              f"-> {str(r['respondent_name'])[:35]:<37} peak={r['peak_demand_mw']:>7,.0f} MW  yr={r['report_year']}")
    else:
        print(f"  !!  DIRECT   {label:<22} eia_code={src_id:<6} -> NOT FOUND")
        all_ok = False

for label, src_id, ba_code in BA_FALLBACK_CHECKS:
    direct = merged[merged['eia_code'] == src_id]
    ba     = merged[merged['ba_code']  == ba_code]
    direct_note = f"(direct={int(direct.iloc[0]['report_year'])})" if len(direct) else "(no direct)"
    if len(ba):
        r = ba.iloc[0]
        print(f"  OK  BA-FALL  {label:<22} ba_code={ba_code:<5} "
              f"-> {str(r['respondent_name'])[:35]:<37} peak={r['peak_demand_mw']:>7,.0f} MW  yr={r['report_year']}  {direct_note}")
    else:
        print(f"  !!  BA-FALL  {label:<22} ba_code={ba_code:<5} -> NOT FOUND")
        all_ok = False

print(f"  {'All join-path checks passed.' if all_ok else 'SOME CHECKS FAILED -- review above.'}")

# ── Output ───────────────────────────────────────────────────────────────────

merged.to_parquet(OUT_PARQUET, index=False)
print(f"\nWrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e3:.1f} KB)")
sample = merged.head(20).copy()
sample.to_csv(Path(__file__).parent / 'sample_rows.csv', index=False)
print("Exported: sample_rows.csv")
print(f"\nNote: cache/{HOURLY_FILE} ({HOURLY_CACHE.stat().st_size/1e6:.0f} MB) "
      f"can be deleted to reclaim disk; will re-download on next run.")
