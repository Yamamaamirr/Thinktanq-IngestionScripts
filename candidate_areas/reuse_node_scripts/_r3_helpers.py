"""
Shared helper for the StepR3 reuse-node enrichment pipeline.

Loads reuse_nodes_clean.parquet and aliases its columns onto the same field
names that the existing Step1X greenfield enrichment scripts expect:

    site_id          -> candidate_id
    footprint_acres  -> area_acres
    state_abbr       -> state

Also reprojects to EPSG:5070 (the working CRS used by Step1X).

This lets the StepR3X scripts be near-verbatim copies of the corresponding
Step1X scripts -- only IN/OUT paths change, and the join key is the same
'candidate_id' string everywhere downstream.
"""
from pathlib import Path
import geopandas as gpd

REUSE_PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_clean.parquet')
R3_OUT_DIR = Path('candidate_areas/reuse_node_enrichment_outputs')


def load_reuse_nodes_as_candidates(crs_epsg=5070):
    """Return reuse_nodes_clean with greenfield-style column names + projection."""
    g = gpd.read_parquet(REUSE_PATH)
    g = g.rename(columns={
        'site_id':         'candidate_id',
        'footprint_acres': 'area_acres',
        'state_abbr':      'state',
    })
    if g.crs is None or g.crs.to_epsg() != crs_epsg:
        g = g.to_crs(crs_epsg)
    return g


def out_path(filename):
    R3_OUT_DIR.mkdir(parents=True, exist_ok=True)
    return R3_OUT_DIR / filename
