"""GridStatus ISO interconnection queue ingestion.

Fetches interconnection queue data from all available US ISO/RTO grid operators
via the gridstatus Python library and normalises into a unified iso_queues schema.

Availability at time of writing (gridstatus 0.36.0):
  CAISO   OK  -- 2,278 rows, all key fields present
  MISO    OK  -- 3,771 rows; Project Name is 100% null from their API
  ISONE   OK  -- 1,751 rows; dates arrive as M/D/YYYY strings
  PJM     OK  -- POSTs directly to PJM backend export API (no account needed; subscription key from PJM JS bundle)
  ERCOT   --  -- 403 Forbidden from ERCOT MISAPP; may need access request
  SPP     --  -- connection timeout to opsportal.spp.org (network-dependent)
  NYISO   --  -- DNS resolution failure from some networks; retry from server

Each ISO is cached to its own parquet file under ./cache/ so re-runs during
development skip the network fetch. Delete a cache file to force a re-fetch.

Output: concatenated DataFrame ready for INSERT into iso_queues table.
Column contract:
  iso_region       TEXT    -- CAISO / MISO / ISONE / PJM / etc.
  source_id        TEXT    -- "{ISO}-{Queue ID}" — unique within this dataset
  project_name     TEXT    -- may be null for some ISOs (MISO)
  substation_name  TEXT    -- Interconnection Location; key field for anchor join
  state            TEXT    -- US state abbreviation; used as anchor-join pre-filter
  capacity_mw      NUMERIC
  queue_status     TEXT    -- active / withdrawn / energized / unknown
  queue_date       DATE
  in_service_date  DATE
  withdrawal_date  DATE
  fuel_type        TEXT
  ia_executed      BOOL    -- True if interconnection agreement is executed/signed
  county           TEXT    -- county name (ERCOT and LBNL rows only; null for others)

Substation matching note:
  iso_queues.substation_name is the raw string from the ISO queue.
  It is NOT guaranteed to match anchors.name exactly — join during the
  enrichment step using fuzzy string matching (rapidfuzz or pg_trgm).
  CAISO example: "Birds Landing 230 kV"
  ISONE example: "NGRID KING ST T3"
  MISO example:  "NORTHERN STATES POWER COMPANY" (utility, not substation)
"""
import os
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
OUT_PARQUET = Path(__file__).parent / "iso_queues.parquet"

# Status value normalisation — maps ISO-specific strings to the canonical set.
# Case-insensitive prefix matching is used so partial values also map correctly.
STATUS_MAP = {
    "active":              "active",
    "pending":             "active",
    "in process":          "active",
    "executed":            "active",
    "on hold":             "active",
    "energized":           "energized",
    "in service":          "energized",
    "operational":         "energized",
    "complete":            "energized",
    "withdrawn":           "withdrawn",
    "cancelled":           "withdrawn",
    "suspended":           "suspended",   # weak positive signal (score 30-45); not withdrawn
}


def _norm_status(val):
    if pd.isna(val) or not str(val).strip():
        return "unknown"
    lower = str(val).strip().lower()
    for key, canonical in STATUS_MAP.items():
        if lower.startswith(key):
            return canonical
    return "unknown"


def _to_date(series):
    """Parse a mixed-format date series to tz-naive date (date, not datetime)."""
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


# ── Per-ISO fetch and normalise functions ─────────────────────────────────────

def _fetch_caiso():
    import gridstatus
    iso = gridstatus.CAISO()
    q = iso.get_interconnection_queue()
    return pd.DataFrame({
        "iso_region":      "CAISO",
        "source_id":       "CAISO-" + q["Queue ID"].astype(str),
        "project_name":    q["Project Name"],
        "substation_name": q["Interconnection Location"],
        "state":           q["State"],
        "capacity_mw":     pd.to_numeric(q["Capacity (MW)"], errors="coerce"),
        "queue_status":    q["Status"].apply(_norm_status),
        "queue_date":      _to_date(q["Queue Date"]),
        "in_service_date": _to_date(q["Proposed Completion Date"]),
        "withdrawal_date": _to_date(q["Withdrawn Date"]),
        "fuel_type":       q["Fuel-1"],
        "ia_executed":     q.get("Interconnection Agreement Status", pd.Series(dtype=object, index=q.index)) == "Executed",
        "county":          pd.NA,
        "load_zone":       pd.NA,
    })


def _fetch_miso():
    import gridstatus
    iso = gridstatus.MISO()
    q = iso.get_interconnection_queue()
    # MISO does not expose Project Name — null for all rows.
    # Use inService for in-service date; Proposed Completion Date is 85% null.
    return pd.DataFrame({
        "iso_region":      "MISO",
        "source_id":       "MISO-" + q["Queue ID"].astype(str),
        "project_name":    q.get("Project Name"),
        "substation_name": q["Interconnection Location"],
        "state":           q["State"],
        "capacity_mw":     pd.to_numeric(q["Capacity (MW)"], errors="coerce"),
        "queue_status":    q["Status"].apply(_norm_status),
        "queue_date":      _to_date(q["Queue Date"]),
        "in_service_date": _to_date(q.get("inService", pd.Series(dtype=object, index=q.index))),
        "withdrawal_date": _to_date(q.get("Withdrawn Date", pd.Series(dtype=object, index=q.index))),
        "fuel_type":       q.get("Generation Type"),
        "ia_executed":     q.get("giaToExec", pd.Series(dtype=object, index=q.index)).notna(),
        "county":          pd.NA,
        "load_zone":       pd.NA,
    })


def _fetch_isone():
    import gridstatus
    iso = gridstatus.ISONE()
    q = iso.get_interconnection_queue()
    # ISONE dates arrive as US-format strings (M/D/YYYY).
    # Op Date is the operational/in-service target; Proposed Completion Date
    # is also present but sparser.
    return pd.DataFrame({
        "iso_region":      "ISONE",
        "source_id":       "ISONE-" + q["Queue ID"].astype(str),
        "project_name":    q["Project Name"],
        "substation_name": q["Interconnection Location"],
        "state":           q["State"],
        "capacity_mw":     pd.to_numeric(q["Capacity (MW)"], errors="coerce"),
        "queue_status":    q["Status"].apply(_norm_status),
        "queue_date":      _to_date(q["Queue Date"]),
        "in_service_date": _to_date(q.get("Op Date", q.get("Proposed Completion Date"))),
        "withdrawal_date": _to_date(q["Withdrawn Date"]),
        "fuel_type":       q.get("Generation Type"),
        "ia_executed":     q.get("Interconnection Agreement Status", pd.Series(dtype=object, index=q.index)) == "Executed",
        "county":          pd.NA,
        "load_zone":       pd.NA,
    })


def _fetch_pjm():
    """Fetch PJM queue by POSTing directly to PJM's backend export endpoint.

    PROTOTYPE PATH — ToS status unverified.
    The subscription key below is extracted from PJM's frontend JS bundle (same
    technique gridstatus uses). It works without formal API registration, but
    PJM has not confirmed this use is permitted under their data terms.
    Do not make production reliability depend solely on this path — use it to
    generate pjm.csv manually, then commit the CSV and rely on _fetch_pjm_csv()
    for the automated pipeline. Refresh pjm.csv at least quarterly.

    Endpoint: POST https://services.pjm.com/PJMPlanningApi/api/Queue/ExportToXls
    Returns a binary XLS with columns matching the PJM New Services Queue.
    """
    import io
    import requests as _requests

    print("  WARNING: _fetch_pjm() uses an embedded PJM subscription key (prototype path).")
    print("  ToS not confirmed. Intended for manual CSV refresh only — not automated production.")

    PJM_EXPORT_URL = 'https://services.pjm.com/PJMPlanningApi/api/Queue/ExportToXls'
    # Subscription key from PJM's own frontend JS bundle.
    PJM_SUB_KEY = 'E29477D0-70E0-4825-89B0-43F460BF9AB4'

    r = _requests.post(
        PJM_EXPORT_URL,
        headers={
            'api-subscription-key': PJM_SUB_KEY,
            'Host':   'services.pjm.com',
            'Origin': 'https://www.pjm.com',
            'Referer': 'https://www.pjm.com/',
        },
        timeout=60,
    )
    r.raise_for_status()

    q = pd.read_excel(io.BytesIO(r.content), engine='openpyxl')

    # Capacity: take the lesser of MFO and MW In Service (mirrors gridstatus logic)
    q["_capacity_mw"] = pd.to_numeric(
        q[["MFO", "MW In Service"]].min(axis=1), errors="coerce"
    )

    # IA executed: the IA status column contains "Document Posted" when signed
    ia_col = "Interim/Interconnection Service/Generation Interconnection Agreement Status"
    ia_executed = q[ia_col].astype(str).str.strip().str.lower() == "document posted" \
        if ia_col in q.columns else pd.Series(False, index=q.index)

    return pd.DataFrame({
        "iso_region":      "PJM",
        "source_id":       "PJM-" + q["Project ID"].astype(str),
        "project_name":    q["Name"],              # interconnection point name (Commercial Name is 73% null)
        "substation_name": q["Name"],              # same field — best available substation identifier
        "state":           q["State"],
        "capacity_mw":     q["_capacity_mw"],
        "queue_status":    q["Status"].apply(_norm_status),
        "queue_date":      _to_date(q["Submitted Date"]),
        "in_service_date": _to_date(q["Projected In Service Date"]),
        "withdrawal_date": _to_date(q["Withdrawal Date"]),
        "fuel_type":       q["Fuel"],
        "ia_executed":     ia_executed,
        "county":          q["County"],
        "load_zone":       pd.NA,
    })


ERCOT_FUEL_MAP = {
    'WIN': 'Wind', 'SOL': 'Solar', 'GAS': 'Natural Gas',
    'NUC': 'Nuclear', 'HYD': 'Hydro', 'OTH': 'Other',
    'LIG': 'Lignite', 'COA': 'Coal', 'OIL': 'Oil',
    'BIO': 'Biomass', 'GEO': 'Geothermal',
}
ERCOT_TECH_MAP = {'BA': 'Battery Storage', 'FC': 'Fuel Cell'}


def _parse_ercot_gis(path):
    """Parse ERCOT GIS Report Excel file into the standard iso_queues schema.

    Download from: https://www.ercot.com/misapp/GetReports.do?reportTypeId=15933
    Select the most recent GIS_Report_*.xlsx file.

    Sheet structure (verified against Feb 2026 report):
      'Project Details - Large Gen'  -- active large generators (>10 MW)
      'Project Details - Small Gen'  -- active small generators (<=10 MW)
      'Inactive Projects'            -- withdrawn/cancelled projects
    """
    xl_path = str(path)

    def _read_active(sheet, skiprows):
        df = pd.read_excel(xl_path, sheet_name=sheet, header=None,
                           skiprows=skiprows, engine='openpyxl')
        df = df.dropna(how='all').reset_index(drop=True)
        # col 0=INR, 1=Project Name, 4=POI Location, 5=County,
        # 6=CDR Zone, 7=Projected COD, 8=Fuel, 9=Technology, 10=Capacity MW,
        # 16=IA Signed date
        df.columns = range(len(df.columns))
        # Drop header-continuation rows (non-INR rows at top)
        df = df[df[0].astype(str).str.match(r'^\d{2}INR')]
        fuel = df[8].map(ERCOT_FUEL_MAP)
        # Battery storage: OTH fuel + BA technology
        is_battery = (df[8] == 'OTH') & (df[9].map(ERCOT_TECH_MAP) == 'Battery Storage')
        fuel[is_battery] = 'Battery Storage'
        ia_signed = df[16] if 16 in df.columns else pd.Series(dtype=object, index=df.index)
        ia_executed = ia_signed.apply(
            lambda v: pd.notna(v) and str(v).strip() not in ('', 'nan', 'NaT')
        )
        # CDR Zone (col 6) — ERCOT load zone: North, South, West, Houston, etc.
        load_zone = df[6].astype(str).str.strip().str.title().where(
            df[6].notna() & (df[6].astype(str).str.strip() != ''), pd.NA
        )
        return pd.DataFrame({
            "iso_region":      "ERCOT",
            "source_id":       "ERCOT-" + df[0].astype(str),
            "project_name":    df[1].astype(str),
            "substation_name": df[4].astype(str),
            "state":           "TX",
            "capacity_mw":     pd.to_numeric(df[10], errors="coerce"),
            "queue_status":    "active",
            "queue_date":      pd.NaT,
            "in_service_date": _to_date(df[7]),
            "withdrawal_date": pd.NaT,
            "fuel_type":       fuel,
            "ia_executed":     ia_executed,
            "county":          df[5].where(df[5].notna(), pd.NA),
            "load_zone":       load_zone,
        })

    def _read_inactive():
        df = pd.read_excel(xl_path, sheet_name='Inactive Projects', header=None,
                           skiprows=8, engine='openpyxl')
        df = df.dropna(how='all').reset_index(drop=True)
        df.columns = range(len(df.columns))
        df = df[df[0].astype(str).str.match(r'^\d{2}INR')]
        # col 0=INR, 1=Size, 2=Project Name, 3=Fuel, 4=County, 5=Inactive Date, 6=MW
        fuel = df[3].map(lambda v: ERCOT_FUEL_MAP.get(str(v)[:3].upper(), str(v)))
        return pd.DataFrame({
            "iso_region":      "ERCOT",
            "source_id":       "ERCOT-" + df[0].astype(str),
            "project_name":    df[2].astype(str),
            "substation_name": pd.NA,
            "state":           "TX",
            "capacity_mw":     pd.to_numeric(df[6], errors="coerce"),
            "queue_status":    "withdrawn",
            "queue_date":      pd.NaT,
            "in_service_date": pd.NaT,
            "withdrawal_date": _to_date(df[5]),
            "fuel_type":       fuel,
            "ia_executed":     False,
            "county":          pd.NA,
            "load_zone":       pd.NA,
        })

    large = _read_active('Project Details - Large Gen', skiprows=35)
    small = _read_active('Project Details - Small Gen', skiprows=17)
    inactive = _read_inactive()
    combined = pd.concat([large, small, inactive], ignore_index=True)
    print(f"  ERCOT GIS: {len(large)} large-gen + {len(small)} small-gen active, "
          f"{len(inactive)} inactive = {len(combined)} total")
    return combined


# ERCOT_GIS_PATH: set to the path of a manually downloaded GIS Report xlsx.
# Download from: https://www.ercot.com/misapp/GetReports.do?reportTypeId=15933
# Select the most recent GIS_Report_*.xlsx file.
ERCOT_GIS_PATH = os.environ.get(
    "ERCOT_GIS_PATH",
    r"C:\Users\dell\Downloads\RPT.00015933.0000000000000000.20260402.154703469.GIS_Report_March2026.xlsx",
)


def _fetch_ercot():
    # Try gridstatus live API first; fall back to local GIS Report if blocked.
    try:
        import gridstatus
        iso = gridstatus.Ercot()
        q = iso.get_interconnection_queue()
        return pd.DataFrame({
            "iso_region":      "ERCOT",
            "source_id":       "ERCOT-" + q["Queue ID"].astype(str),
            "project_name":    q.get("Project Name"),
            "substation_name": q.get("Interconnection Location"),
            "state":           q.get("State"),
            "capacity_mw":     pd.to_numeric(q.get("Capacity (MW)"), errors="coerce"),
            "queue_status":    q["Status"].apply(_norm_status),
            "queue_date":      _to_date(q.get("Queue Date")),
            "in_service_date": _to_date(q.get("Proposed Completion Date")),
            "withdrawal_date": _to_date(q.get("Withdrawn Date")),
            "fuel_type":       q.get("Fuel-1", q.get("Generation Type")),
            "ia_executed":     False,
            "county":          pd.NA,
            "load_zone":       pd.NA,
        })
    except Exception as live_err:
        if ERCOT_GIS_PATH and os.path.exists(ERCOT_GIS_PATH):
            print(f"  gridstatus ERCOT failed ({type(live_err).__name__}), "
                  f"falling back to GIS Report: {ERCOT_GIS_PATH}")
            return _parse_ercot_gis(ERCOT_GIS_PATH)
        raise


# LBNL_QUEUE_PATH: set to the path of the LBNL "Queued Up" Excel data file.
# Download "Queued Up 2025 Data File XLSX" from: https://emp.lbl.gov/queues
# Covers non-ISO AZ/NV utilities: APS, SRP, TEP (AZ) and NV Energy (NV).
LBNL_QUEUE_PATH = os.environ.get(
    "LBNL_QUEUE_PATH",
    r"C:\Users\dell\Downloads\LBNL_Ix_Queue_Data_File_thru2024_v2.xlsx",
)

# Maps LBNL entity codes to iso_region values in the iso_queues schema.
# CAISO-region entities (SCE, SDGE, GLW, VEA) are filtered out upstream — they
# are already covered by the live CAISO gridstatus fetch.
_LBNL_ENTITY_TO_REGION = {
    'APS':        'APS',
    'SRP':        'SRP',
    'SRP_ANPP':   'SRP',
    'SRP_Gila':   'SRP',
    'SRP_PV-PC':  'SRP',
    'SRP_SWV':    'SRP',
    'TEP':        'TEP',
    'WAPA-DSW':   'WAPA',
    'WAPA-MPP':   'WAPA',
    'WAPA-RM':    'WAPA',
    'NVE':        'NV_ENERGY',
    'IP':         'IDAHO_POWER',
    'LADWP':      'LADWP',
    'N-C':        'NAVAJO_CRYSTAL',
    'PacifiCorp': 'PACIFICORP',
}

_LBNL_STATUS_MAP = {
    'active':      'active',
    'suspended':   'active',      # studies on hold but still in queue
    'withdrawn':   'withdrawn',
    'operational': 'energized',
    'unknown':     'unknown',
}


def _excel_serial_to_date(series):
    """Convert Excel serial date numbers (stored as floats) to tz-naive timestamps."""
    numeric = pd.to_numeric(series, errors='coerce')
    return pd.to_datetime(numeric, unit='D',
                          origin=pd.Timestamp('1899-12-30'), errors='coerce')


def _parse_lbnl_queue(path):
    """Parse LBNL Queued Up Excel file for AZ/NV non-ISO utility queues.

    Sheet '03. Complete Queue Data' (header at row 1) contains project-level
    interconnection request data for all 7 ISOs plus 49 non-ISO utilities.
    We keep only AZ and NV rows with region != 'CAISO' to avoid double-counting
    projects already present in the CAISO gridstatus fetch.
    """
    df = pd.read_excel(path, sheet_name='03. Complete Queue Data',
                       header=1, engine='openpyxl')
    # Keep AZ/NV/VA only. Exclude regions where we already have live data to avoid
    # double-counting. PJM is included here as a fallback — if _fetch_pjm() succeeds,
    # its rows take precedence; if it fails (network-blocked), LBNL covers VA/PJM.
    LIVE_ISO_REGIONS = {'CAISO', 'MISO', 'ERCOT', 'ISO-NE', 'PJM'}
    df = df[df['state'].isin(['AZ', 'NV', 'VA']) & ~df['region'].isin(LIVE_ISO_REGIONS)].copy()

    df['iso_region']   = df['entity'].map(_LBNL_ENTITY_TO_REGION).fillna(df['entity'])
    df['source_id']    = 'LBNL-' + df['entity'].astype(str) + '-' + df['q_id'].astype(str)
    df['queue_status'] = df['q_status'].map(_LBNL_STATUS_MAP).fillna('unknown')

    # NV Energy rows have no poi_name in LBNL — substation_name stays null.
    # They will be unmatched in anchor_queue_enrichment, which is correct:
    # attributing MW to a county-proxy substation produces a false signal.

    return pd.DataFrame({
        'iso_region':      df['iso_region'],
        'source_id':       df['source_id'],
        'project_name':    df['project_name'],
        'substation_name': df['poi_name'],
        'state':           df['state'],
        'capacity_mw':     pd.to_numeric(df['mw1'], errors='coerce'),
        'queue_status':    df['queue_status'],
        'queue_date':      _excel_serial_to_date(df['q_date']),
        'in_service_date': _excel_serial_to_date(df['prop_date']),
        'withdrawal_date': _excel_serial_to_date(df['wd_date']),
        'fuel_type':       df['type_clean'],
        'ia_executed':     df['IA_status_clean'].eq('IA Executed'),
        'county':          df['county'],
        'load_zone':       df['entity'].map(lambda e: 'NV_ENERGY' if e == 'NVE' else pd.NA),
    })


PJM_CSV_PATH = Path(__file__).parent / "pjm.csv"


def _fetch_pjm_csv():
    """Read PJM queue from pjm.csv in this directory.

    Same schema as _fetch_pjm() XLS download; normalisation logic is identical.
    pjm.csv should be refreshed manually via _fetch_pjm() at least quarterly.
    Provenance (snapshot date, row count, hash) is logged on every load.
    """
    import hashlib
    from datetime import datetime as _dt

    mtime = _dt.fromtimestamp(PJM_CSV_PATH.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
    sha = hashlib.sha256(PJM_CSV_PATH.read_bytes()).hexdigest()[:16]
    q = pd.read_csv(PJM_CSV_PATH, dtype=str, low_memory=False)
    print(f"  PJM provenance: snapshot={mtime}  rows={len(q):,}  sha256[:16]={sha}")

    mfo = pd.to_numeric(q["MFO"], errors="coerce")
    mw_in_svc = pd.to_numeric(q["MW In Service"], errors="coerce")
    capacity = pd.DataFrame({"mfo": mfo, "mw_in_svc": mw_in_svc}).min(axis=1)

    ia_col = "Interim/Interconnection Service/Generation Interconnection Agreement Status"
    ia_executed = q[ia_col].astype(str).str.strip().str.lower() == "document posted" \
        if ia_col in q.columns else pd.Series(False, index=q.index)

    return pd.DataFrame({
        "iso_region":      "PJM",
        "source_id":       "PJM-" + q["Project ID"].astype(str),
        "project_name":    q["Name"],
        "substation_name": q["Name"],
        "state":           q["State"],
        "capacity_mw":     capacity,
        "queue_status":    q["Status"].apply(_norm_status),
        "queue_date":      _to_date(q["Submitted Date"]),
        "in_service_date": _to_date(q["Projected In Service Date"]),
        "withdrawal_date": _to_date(q["Withdrawal Date"]),
        "fuel_type":       q["Fuel"],
        "ia_executed":     ia_executed,
        "county":          q["County"],
        "load_zone":       pd.NA,
    })


def _fetch_spp():
    import gridstatus
    iso = gridstatus.SPP()
    q = iso.get_interconnection_queue()
    return pd.DataFrame({
        "iso_region":      "SPP",
        "source_id":       "SPP-" + q["Queue ID"].astype(str),
        "project_name":    q.get("Project Name"),
        "substation_name": q.get("Interconnection Location"),
        "state":           q.get("State"),
        "capacity_mw":     pd.to_numeric(q.get("Capacity (MW)"), errors="coerce"),
        "queue_status":    q["Status"].apply(_norm_status),
        "queue_date":      _to_date(q.get("Queue Date")),
        "in_service_date": _to_date(q.get("Proposed Completion Date")),
        "withdrawal_date": _to_date(q.get("Withdrawn Date")),
        "fuel_type":       q.get("Fuel-1", q.get("Generation Type")),
        "ia_executed":     False,
        "county":          pd.NA,
        "load_zone":       pd.NA,
    })


def _fetch_nyiso():
    import gridstatus
    iso = gridstatus.NYISO()
    q = iso.get_interconnection_queue()
    return pd.DataFrame({
        "iso_region":      "NYISO",
        "source_id":       "NYISO-" + q["Queue ID"].astype(str),
        "project_name":    q.get("Project Name"),
        "substation_name": q.get("Interconnection Location"),
        "state":           q.get("State"),
        "capacity_mw":     pd.to_numeric(q.get("Capacity (MW)"), errors="coerce"),
        "queue_status":    q["Status"].apply(_norm_status),
        "queue_date":      _to_date(q.get("Queue Date")),
        "in_service_date": _to_date(q.get("Proposed Completion Date")),
        "withdrawal_date": _to_date(q.get("Withdrawn Date")),
        "fuel_type":       q.get("Fuel-1", q.get("Generation Type")),
        "ia_executed":     False,
        "county":          pd.NA,
        "load_zone":       pd.NA,
    })


ISO_FETCHERS = {
    "CAISO": _fetch_caiso,
    "MISO":  _fetch_miso,
    "ISONE": _fetch_isone,
    "PJM":   _fetch_pjm_csv,
    "ERCOT": _fetch_ercot,
    "SPP":   _fetch_spp,
    "NYISO": _fetch_nyiso,
}

OUTPUT_COLS = [
    "iso_region", "source_id", "project_name", "substation_name", "state",
    "capacity_mw", "queue_status", "queue_date", "in_service_date",
    "withdrawal_date", "fuel_type", "ia_executed", "county", "load_zone",
]
FINAL_COLS = OUTPUT_COLS + ["ingested_at"]

# ── Main fetch loop ───────────────────────────────────────────────────────────

frames = []
iso_stats = {}

for iso_name, fetcher in ISO_FETCHERS.items():
    cache_path = CACHE_DIR / f"{iso_name.lower()}.parquet"

    if cache_path.exists():
        print(f"{iso_name}: loading from cache ({cache_path})")
        df = pd.read_parquet(cache_path)
        if 'ia_executed' not in df.columns:
            df['ia_executed'] = False
        if 'county' not in df.columns:
            df['county'] = pd.NA
        if 'load_zone' not in df.columns:
            df['load_zone'] = pd.NA
        frames.append(df)
        iso_stats[iso_name] = {"rows": len(df), "status": "cached"}
        continue

    print(f"{iso_name}: fetching...", end=" ", flush=True)
    try:
        df = fetcher()
        df = df[OUTPUT_COLS].copy()
        df.to_parquet(cache_path, index=False)
        frames.append(df)
        iso_stats[iso_name] = {"rows": len(df), "status": "fetched"}
        print(f"{len(df):,} rows cached to {cache_path}")
    except Exception as e:
        iso_stats[iso_name] = {"rows": 0, "status": f"FAILED: {type(e).__name__}: {e}"}
        print(f"FAILED -- {type(e).__name__}: {e}")

# ── PJM staleness check ───────────────────────────────────────────────────────
# Warn if the pjm.csv snapshot is older than PJM_STALE_DAYS. The embedded-key
# fetch path (_fetch_pjm) is for manual refreshes only — see its ToS warning.
PJM_STALE_DAYS = 30
if PJM_CSV_PATH.exists():
    from datetime import datetime as _dt
    _age = (_dt.now() - _dt.fromtimestamp(PJM_CSV_PATH.stat().st_mtime)).days
    if _age > PJM_STALE_DAYS:
        print(f"\n{'!'*65}")
        print(f"  PJM DATA IS {_age} DAYS OLD (threshold: {PJM_STALE_DAYS} days).")
        print(f"  Refresh pjm.csv by running _fetch_pjm() manually and committing the file.")
        print(f"  Note: _fetch_pjm() uses an embedded key — verify PJM ToS before use.")
        print(f"{'!'*65}\n")

# ── LBNL non-ISO utility queues (AZ/NV supplement) ───────────────────────────

LBNL_CACHE_PATH = CACHE_DIR / "lbnl.parquet"

if LBNL_QUEUE_PATH and os.path.exists(LBNL_QUEUE_PATH):
    print(f"\nLBNL: loading non-ISO utility queues from {LBNL_QUEUE_PATH} ...", end=" ", flush=True)
    try:
        lbnl_df = _parse_lbnl_queue(LBNL_QUEUE_PATH)
        lbnl_df = lbnl_df[OUTPUT_COLS].copy()
        lbnl_df.to_parquet(LBNL_CACHE_PATH, index=False)
        frames.append(lbnl_df)
        n_active = (lbnl_df['queue_status'] == 'active').sum()
        az_active = ((lbnl_df['state'] == 'AZ') & (lbnl_df['queue_status'] == 'active')).sum()
        nv_active = ((lbnl_df['state'] == 'NV') & (lbnl_df['queue_status'] == 'active')).sum()
        va_active = ((lbnl_df['state'] == 'VA') & (lbnl_df['queue_status'] == 'active')).sum()
        print(f"{len(lbnl_df):,} rows ({n_active} active)")
        print(f"  AZ (APS/SRP/TEP): {az_active}  | NV (NV Energy): {nv_active}  "
              f"| VA (PJM): {va_active} active rows")
        iso_stats['LBNL'] = {'rows': len(lbnl_df), 'status': 'loaded'}
    except Exception as e:
        print(f"FAILED -- {type(e).__name__}: {e}")
        iso_stats['LBNL'] = {'rows': 0, 'status': f"FAILED: {type(e).__name__}: {e}"}
elif LBNL_CACHE_PATH.exists():
    print(f"\nLBNL: loading from cache ({LBNL_CACHE_PATH})")
    lbnl_df = pd.read_parquet(LBNL_CACHE_PATH)
    if 'ia_executed' not in lbnl_df.columns:
        lbnl_df['ia_executed'] = False
    if 'county' not in lbnl_df.columns:
        lbnl_df['county'] = pd.NA
    if 'load_zone' not in lbnl_df.columns:
        lbnl_df['load_zone'] = pd.NA
    frames.append(lbnl_df)
    iso_stats['LBNL'] = {'rows': len(lbnl_df), 'status': 'cached'}
else:
    print(f"\nLBNL: LBNL_QUEUE_PATH not set and no cache — AZ/NV non-ISO queues not loaded.")
    print(f"  Set LBNL_QUEUE_PATH to path of 'Queued Up 2025 Data File XLSX'")
    print(f"  Download from: https://emp.lbl.gov/queues")
    iso_stats['LBNL'] = {'rows': 0, 'status': 'skipped (LBNL_QUEUE_PATH not set)'}

# ── Combine and report ────────────────────────────────────────────────────────

print("\n" + "="*65)
print("FETCH SUMMARY")
print("="*65)
for iso_name, stat in iso_stats.items():
    status_str = stat["status"]
    rows_str = f"{stat['rows']:,} rows" if stat["rows"] else ""
    print(f"  {iso_name:<8} {rows_str:<12} {status_str}")

if not frames:
    print("\nNo data fetched successfully — nothing to process.")
else:
    all_queues = pd.concat(frames, ignore_index=True)
    all_queues["ingested_at"] = pd.Timestamp.now(tz='UTC')

    # ISONE multi-point projects: one queue ID spans multiple substations.
    # The gridstatus library returns one row per substation, so source_id is
    # not unique at the row level for ISONE. Append a numeric suffix to make
    # it unique while preserving the original queue ID in the prefix.
    dup_mask = all_queues.duplicated('source_id', keep=False)
    if dup_mask.any():
        cumcount = all_queues.groupby('source_id').cumcount()
        all_queues.loc[dup_mask, 'source_id'] = (
            all_queues.loc[dup_mask, 'source_id'] + '-'
            + cumcount[dup_mask].astype(str)
        )
        n_deduped = dup_mask.sum()
        print(f"  Deduplicated {n_deduped} ISONE multi-point source_ids (suffix appended)")

    # Clamp negative capacity_mw to 0. Negatives are demand-response / load-
    # reduction resources (e.g. ISONE-148 Comerford NH = -17 MW) which reduce
    # load, not add to interconnection queue congestion. Treating them as 0
    # is correct for the queue_mw scoring signal.
    neg_count = (all_queues["capacity_mw"] < 0).sum()
    if neg_count:
        all_queues.loc[all_queues["capacity_mw"] < 0, "capacity_mw"] = 0.0
        print(f"  Clamped {neg_count} negative capacity_mw values to 0 "
              f"(demand-response resources)")

    print(f"\nTotal combined rows: {len(all_queues):,}")
    print(f"\nqueue_status breakdown:")
    print(all_queues["queue_status"].value_counts(dropna=False).to_string())
    print(f"\niso_region breakdown:")
    print(all_queues["iso_region"].value_counts().to_string())
    print(f"\ncapacity_mw stats:")
    print(all_queues["capacity_mw"].describe().round(1).to_string())
    print(f"\nnull counts:")
    nulls = all_queues.isnull().sum()
    for col, n in nulls.items():
        pct = n / len(all_queues) * 100
        print(f"  {col:<20} {n:>5} nulls  ({pct:.0f}%)")
    print(f"\nSample rows (5 active, 5 withdrawn):")
    sample = pd.concat([
        all_queues[all_queues["queue_status"] == "active"].head(5),
        all_queues[all_queues["queue_status"] == "withdrawn"].head(5),
    ])
    print(sample[["iso_region","source_id","substation_name","state","capacity_mw",
                  "queue_status","in_service_date","fuel_type"]].to_string())

    # ── PRD schema / sanity assertions ───────────────────────────────────────
    assert list(all_queues.columns) == FINAL_COLS, \
        f"Column mismatch: {list(all_queues.columns)}"
    assert all_queues["queue_status"].isin(
        {"active", "withdrawn", "energized", "unknown"}
    ).all(), \
        f"Unexpected status values: {all_queues['queue_status'].unique()}"
    active_rows = all_queues[all_queues["queue_status"] == "active"]
    assert len(active_rows) > 0, "No active queue entries — unexpected"
    assert active_rows["capacity_mw"].notna().mean() > 0.5, \
        "More than 50% of active rows have null capacity_mw"
    mw_max = all_queues["capacity_mw"].max()
    assert mw_max < 25_000, f"Implausible capacity_mw max: {mw_max}"
    print(f"\nPRD schema assertions passed ({len(all_queues):,} rows, "
          f"{all_queues['iso_region'].nunique()} ISOs)")

    # ── Target-state coverage summary ────────────────────────────────────────
    print(f"\nTarget-state coverage (active projects only):")
    print(f"  {'State':<6} {'ISO':<8} {'Active rows':>12} {'Total MW':>12}")
    print(f"  {'-'*42}")
    for state in ['CA', 'TX', 'AZ', 'NV', 'VA']:
        state_active = active_rows[active_rows["state"] == state]
        for iso in state_active["iso_region"].unique():
            iso_rows = state_active[state_active["iso_region"] == iso]
            total_mw = iso_rows["capacity_mw"].sum()
            print(f"  {state:<6} {iso:<8} {len(iso_rows):>12,} {total_mw:>12,.0f}")
        if len(state_active) == 0:
            print(f"  {state:<6} {'(none)':>8}  -- ISO not available "
                  f"(see OPEN_QUESTIONS #6)")
    failed = [k for k, v in iso_stats.items() if "FAILED" in v["status"]]
    if failed:
        print(f"\n  Missing ISOs: {', '.join(failed)}")
        print(f"  Impact: TX->ERCOT, VA/NJ/MD->PJM coverage zero until resolved.")

    # ── Write output ─────────────────────────────────────────────────────────
    all_queues[FINAL_COLS].to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {OUT_PARQUET} ({OUT_PARQUET.stat().st_size/1e3:.0f} KB)")

    sample.to_csv(Path(__file__).parent / "sample_rows.csv", index=False)
    print("Exported: sample_rows.csv")
