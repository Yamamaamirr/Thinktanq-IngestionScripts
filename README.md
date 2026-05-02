# thinqtank — Infrastructure Land Screening Engine

Data ingestion and spatial analysis scripts for identifying large, undeveloped land areas
near high-value grid infrastructure across the US. Optimised for industrial and
data-centre site selection.

## How it works

The pipeline runs in two layers:

```
ingestion_scripts/   Download and normalise national datasets into local parquet files.
                     Run once (or on a refresh schedule). Output is used by the scoring engine.

enrichment/          Per-parcel enrichment scripts called at scoring time.
                     Each accepts a parcel geometry and returns a scored result.
```

The output is a ranked candidate table — land areas that score well on infrastructure
proximity, power-readiness, buildability, and low environmental risk.

## Ingestion scripts

| Folder | Source | Output |
|---|---|---|
| `hifld_transmission_lines/` | HIFLD / DHS | 220 kV+ transmission line segments |
| `hifld_electric_substations/` | HIFLD / DHS | Electric substations with voltage and class rating |
| `noaa_drought/` | USDA Drought Monitor (UNL) | Weekly drought classification polygons |
| `faa_radar/` | FAA NASR | Radar exclusion zones around WSR-88D NEXRAD sites |
| `eia_utility_territories/` | EIA | Electric utility service territory polygons |
| `water_districts/` | EPA SDWIS / ECHO | Community water system service areas |
| `census_tiger/` | US Census Bureau | Primary roads and municipality boundaries |
| `class1_rail/` | BTS NTAD | Class I railroad network |
| `eia_phmsa_pipelines/` | EIA / PHMSA | Natural gas interstate and intrastate pipeline network with operator-based capacity classification |
| `eia_form860/` | PUDL / EIA Form 860 | Utility-scale generator inventory including retired and retiring plants |
| `ferc714/` | PUDL / FERC Form 714 | Balancing authority peak demand |
| `gridstatus_iso_queues/` | GridStatus.io / ISO portals | Full ISO interconnection queue pipeline (see below) |

### ISO queue pipeline — `gridstatus_iso_queues/`

The most involved ingestion module. Four scripts work in sequence:

| Script | Role |
|---|---|
| `gridstatus_iso_queues.py` | Downloads raw queue data from CAISO, ERCOT, PJM, MISO, ISONE, and NV Energy. Normalises to a unified schema. |
| `iso_anchor_match.py` | Fuzzy-matches each queue entry's substation name to a real HIFLD location. Distinguishes true substations from line-based POIs (e.g. "A–B 345 kV"). Assigns match confidence: high / medium / low. |
| `anchor_queue_enrichment.py` | Aggregates matched queue rows per substation. Produces `anchor_queue_stats.parquet` (one row per substation with active queue activity) and `zone_queue_stats.parquet` (territory-level fallback for ERCOT zones and NV Energy). |
| `build_supplemental_substations.py` | Builds a supplemental substation layer from EIA 860 and OSM to fill HIFLD gaps, primarily post-2021 Texas substations absent from the frozen HIFLD dataset. |

**Scoring approach:** for each candidate site, all substations within a 50 km radius are
collected and weighted by exponential distance decay (influence halves approximately every
10 km). Weighted active MW is summed across all nearby anchors to produce a
`scoring_mw` signal. When no matched substations are found within range, a zone-level
fallback (ERCOT CDR zone or NV Energy territory) is used instead, flagged as
`zone_is_fallback=True` and kept separate from the spatial anchor signal to prevent
territory-wide MW totals from distorting site-level comparisons.

## Enrichment scripts

| Folder | Source | What it returns |
|---|---|---|
| `fema_flood_zones/` | FEMA NFHL (live API) | Flood zone classification and SFHA disqualifier |
| `usgs_slope/` | USGS 3DEP (live API) | Slope % and terrain band |
| `usgs_seismic/` | USGS NSHM (live API) | PGA and seismic hazard band |
| `usfws_wetlands/` | USFWS NWI (live API) | Wetland coverage and WOTUS disqualifier |
| `usda_soil/` | USDA SSURGO (live API) | Soil bearing band and expansive-clay disqualifier |

## Setup

```bash
pip install -r requirements.txt
```

Run each ingestion script once to generate its local parquet file:

```bash
python ingestion_scripts/hifld_electric_substations/hifld_electric_substations.py
python ingestion_scripts/gridstatus_iso_queues/gridstatus_iso_queues.py
python ingestion_scripts/gridstatus_iso_queues/anchor_queue_enrichment.py
# ... etc
```

Enrichment scripts are imported as modules by the scoring engine — they do not produce parquet files.

## Data notes

- Parquet files, cache directories, and raw shapefiles are excluded via `.gitignore`.
  Run the ingestion scripts locally to regenerate them.
- The NOAA Drought Monitor cache auto-refreshes after 7 days (weekly USDM release cycle).
- FERC 714 downloads a 226 MB hourly file once; subsequent runs use the local cache.
- Pipeline diameters are estimated from operator tier and pipeline classification
  (NPMS access is restricted to government/operator entities). Label as estimated in any output.
- PJM queue data is cached locally. Do not rely on the embedded browser key for production
  calls — use the cache and refresh on a defined schedule.
- Target states: CA, TX, AZ, NV, VA.
