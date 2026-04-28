# thinqtank — Parcel Scoring Engine

Data ingestion and parcel enrichment scripts for industrial / data-center site selection.
Each script is standalone: clone the repo, install dependencies, and run scripts to generate local data.

## Structure

```
ingestion_scripts/   Scripts that download datasets and store them as parquet files.
                     Run once (or on a refresh schedule). Not called per-parcel.

enrichment/          Scripts called per-parcel at scoring time.
                     Each accepts a parcel geometry and returns a result for that parcel.
```

## Ingestion scripts

| Folder | Source | Output |
|---|---|---|
| `hifld_transmission_lines/` | HIFLD / DHS | 220 kV+ transmission line segments |
| `hifld_electric_substations/` | HIFLD / DHS | Electric substations with voltage and class rating |
| `noaa_drought/` | USDA Drought Monitor (UNL) | Weekly drought classification polygons |
| `faa_radar/` | FAA NASR | Radar exclusion zones |
| `eia_utility_territories/` | EIA | Utility service territory polygons |
| `water_districts/` | EPA SDWIS / ECHO | Community water system service areas |
| `census_tiger/` | US Census Bureau | Primary roads and municipality boundaries |
| `class1_rail/` | BTS NTAD | Class I railroad network |
| `gridstatus_iso_queues/` | GridStatus.io | ISO interconnection queue entries |
| `eia_phmsa_pipelines/` | EIA / PHMSA | Natural gas transmission pipelines |
| `eia_form860/` | PUDL / EIA Form 860 | Utility-scale generator capacity |
| `ferc714/` | PUDL / FERC Form 714 | Balancing authority peak demand and headroom |

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

Run each ingestion script once to generate its parquet file:

```bash
python ingestion_scripts/eia_form860/eia_form860.py
python ingestion_scripts/ferc714/ferc714.py
# ... etc
```

Enrichment scripts are imported as modules by the scoring engine — they do not produce parquet files.

## Data notes

- Parquet files, cache directories, and shapefiles are excluded from this repo via `.gitignore`.
  Run the ingestion scripts locally to generate them.
- The NOAA Drought Monitor cache auto-refreshes after 7 days (weekly USDM release cycle).
- FERC 714 downloads a 226 MB hourly file once; subsequent runs use the cache.
- Target states for scoring: CA, TX, AZ, NV, VA.
