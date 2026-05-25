"""
Generate ReuseNodes_QA_Report.pdf -- focused QA summary of the reuse-node
layer for analyst review.

Outputs:
  candidate_areas/reuse_node_outputs/ReuseNodes_QA_Report.pdf

Run: python candidate_areas/reuse_node_scripts/generate_qa_pdf_reuse_nodes.py
"""
from pathlib import Path
from datetime import date
import pandas as pd
import geopandas as gpd
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

ENRICHED_PATH = Path('candidate_areas/reuse_node_outputs/reuse_nodes_enriched.parquet')
OUT_PDF       = Path('candidate_areas/reuse_node_outputs/ReuseNodes_QA_Report.pdf')

styles = getSampleStyleSheet()
TITLE = ParagraphStyle('Title', parent=styles['Heading1'],
                       fontSize=20, leading=24, textColor=colors.HexColor('#1f3a68'),
                       spaceAfter=4)
META  = ParagraphStyle('Meta',  parent=styles['BodyText'],
                       fontSize=10, leading=14, spaceAfter=2)
H1    = ParagraphStyle('H1',    parent=styles['Heading2'],
                       fontSize=14, leading=18, textColor=colors.HexColor('#1f3a68'),
                       spaceBefore=14, spaceAfter=6)
H2    = ParagraphStyle('H2',    parent=styles['Heading3'],
                       fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=3)
BODY  = ParagraphStyle('Body',  parent=styles['BodyText'],
                       fontSize=10, leading=14, alignment=TA_LEFT, spaceAfter=5)
CELL  = ParagraphStyle('Cell',  parent=styles['BodyText'], fontSize=9, leading=11,
                       alignment=TA_LEFT, spaceAfter=0)
CELL_B = ParagraphStyle('CellB', parent=CELL, fontName='Helvetica-Bold')


def hr():
    return HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc'),
                      spaceBefore=8, spaceAfter=8)


def kv_table(rows, col_widths=(2.6*inch, 4.1*inch)):
    data = [[Paragraph(str(k), CELL_B), Paragraph(str(v), CELL)] for k, v in rows]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 0.25, colors.HexColor('#dddddd')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    return t


def grid_table(headers, rows, col_widths):
    head = [Paragraph(h, CELL_B) for h in headers]
    body = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table([head] + body, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8eef7')),
        ('LINEBELOW', (0,0), (-1,0), 0.5, colors.HexColor('#1f3a68')),
        ('LINEBELOW', (0,1), (-1,-1), 0.25, colors.HexColor('#eeeeee')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    return t


def main():
    print(f'Loading {ENRICHED_PATH} ...')
    g = gpd.read_parquet(ENRICHED_PATH)
    print(f'  {len(g):,} rows')

    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=LETTER,
        leftMargin=0.7*inch, rightMargin=0.7*inch,
        topMargin=0.7*inch, bottomMargin=0.7*inch,
        title='ReuseNodes QA Report',
    )
    flow = []

    # ---- Title ----
    flow.append(Paragraph('Reuse Nodes — QA Report', TITLE))
    flow.append(Paragraph(
        f'Phase 1 candidate-site detection &middot; '
        f'{date.today().isoformat()} &middot; '
        f'{len(g):,} scored reuse-node polygons across CA, TX, AZ, NV, VA', META))
    flow.append(hr())

    # ---- Overview ----
    flow.append(Paragraph('1. Overview', H1))
    flow.append(Paragraph(
        'The reuse-node layer captures the four reuse categories in Owen\'s spec: '
        'retired/retiring coal &amp; gas plants (EIA 860), operating/retired nuclear '
        '(NRC + EIA), EPA RE-Powering brownfields/landfills/mines, and large '
        'industrial-zoned sites with grid/rail/water adjacency (OpenStreetMap '
        '<i>landuse=industrial</i> etc.). Each polygon is enriched with the same '
        '~60 attributes the greenfield pipeline uses (transmission, pipelines, '
        'rail, water, utility queue anchors, seismic, drought, slope, adjacency) '
        'plus 4 reuse-specific risk flags, then scored on a recalibrated 5-component '
        'composite.', BODY))

    src_counts = g.source.value_counts()
    flow.append(kv_table([
        ('Total reuse nodes',          f'{len(g):,}'),
        ('EIA-860 (coal/gas retired/retiring)', f'{int(src_counts.get("EIA-860",0)):,}'),
        ('EIA-860-nuclear',            f'{int(src_counts.get("EIA-860-nuclear",0)):,}'),
        ('EPA-RE-Powering',            f'{int(src_counts.get("EPA-RE-Powering",0)):,}'),
        ('OpenStreetMap industrial',   f'{int(src_counts.get("OpenStreetMap",0)):,}'),
        ('Coverage states',            'CA, TX, AZ, NV, VA'),
        ('CRS',                        f'EPSG:{g.crs.to_epsg()}'),
    ]))
    flow.append(Spacer(1, 6))

    # ---- Footprint resolution ----
    flow.append(Paragraph('2. Footprint resolution &amp; cleanup funnel', H1))
    flow.append(Paragraph(
        'Reuse-node sources ship as points (EIA, EPA, NRC) or as polygons (OSM). '
        'StepR2 derives a real polygon for every row via strict point-in-polygon '
        'containment to OSM industrial polygons. If the point sits inside an OSM '
        'polygon, that polygon is adopted; otherwise we fall through to a square '
        'envelope sized by EPA acreage or fuel-type capacity template. Provenance '
        'is tagged in <i>geometry_source</i>.', BODY))

    gs = g.geometry_source.value_counts()
    flow.append(grid_table(
        ['geometry_source', 'count', 'meaning'],
        [
            ['OSM_POLYGON',         f'{int(gs.get("OSM_POLYGON",0)):,}',         'Real polygon drawn by OSM contributors'],
            ['OSM_POLYGON_MATCH',   f'{int(gs.get("OSM_POLYGON_MATCH",0)):,}',   'EPA/EIA point fell inside an OSM industrial polygon'],
            ['BUFFER_FROM_ACREAGE', f'{int(gs.get("BUFFER_FROM_ACREAGE",0)):,}', 'Square envelope sized to EPA acreage'],
            ['BUFFER_FROM_CAPACITY',f'{int(gs.get("BUFFER_FROM_CAPACITY",0)):,}','Square sized by fuel-type capacity template (EIA)'],
        ],
        col_widths=(1.8*inch, 0.8*inch, 4.1*inch),
    ))
    flow.append(Spacer(1, 8))
    flow.append(Paragraph(
        'StepR2b applies six cleanups: intra-source 100m centroid dedup (with '
        '<i>aliased_site_ids</i> preservation), drop LOW-confidence default '
        'buffers, drop OSM operating power plants, drop EPA &lt;50 ac, drop OSM '
        'operating renewables, drop OSM active military/prison/proving (preserving '
        '"Old/Former" decommissioned counterparts). 24,705 raw rows &rarr; 6,631 '
        'cleaned (73% dropped). Polygon overlap factor: <b>1.05x</b> — sites are '
        'effectively non-overlapping.', BODY))

    # ---- Scoring framework ----
    flow.append(Paragraph('3. Scoring framework', H1))
    # First-pass diagnostic numbers (pre-recalibration) -- preserved as the
    # motivation for the weight changes. Owen has the post-recalibration
    # distribution numbers below.
    flow.append(Paragraph(
        'The composite is renormalised over the 0.90 active weight (the 0.10 '
        'site_control component is Phase 2). Reuse-node weights differ from '
        'greenfield because reuse polygons by definition sit near grid '
        'infrastructure -- with greenfield\'s 40% utility weight + uncapped '
        'per-anchor sum, 78% of reuse rows pinned at utility_score=100 and '
        'the composite collapsed (subscore correlation 0.99 with composite). '
        'After recalibration:', BODY))

    flow.append(grid_table(
        ['component', 'weight', 'change vs greenfield'],
        [
            ['utility_score',             '25%', 'Down from 40% + per-anchor cap of 30'],
            ['buildability_score',        '15%', 'Down from 20% (constant inputs)'],
            ['supporting_infra_score',    '15%', 'Unchanged'],
            ['dev_risk_score',            '15%', 'Unchanged (6-component greenfield formula)'],
            ['reuse_environmental_score', '20%', 'NEW standalone slot (was buried in dev_risk)'],
        ],
        col_widths=(2.2*inch, 0.7*inch, 3.8*inch),
    ))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(
        'Effect: contamination-vs-clean composite delta went from <b>3.85 pts &rarr; '
        '18.26 pts</b>. A contaminated EPA site that previously scored 88 (Shortlist) '
        'now lands at ~82 (Reuse Diligence), correctly routing it for ESA before '
        'action.', BODY))

    # ---- Score distribution ----
    flow.append(Paragraph('4. Score distribution', H1))
    s = g.composite_score
    # Compute live subscore correlations against composite
    corr_pairs = [
        ('utility',         g['utility_score'].corr(g['composite_score'])),
        ('supporting_infra',g['supporting_infra_score'].corr(g['composite_score'])),
        ('reuse_env',       g['reuse_environmental_score'].corr(g['composite_score'])),
        ('buildability',    g['buildability_score'].corr(g['composite_score'])),
        ('dev_risk',        g['dev_risk_score'].corr(g['composite_score'])),
    ]
    corr_str = ' &middot; '.join(f'{name} {val:+.2f}' for name, val in corr_pairs)

    flow.append(kv_table([
        ('p10 / median / p90',        f'{s.quantile(0.1):.1f} / {s.median():.1f} / {s.quantile(0.9):.1f}'),
        ('min / max / std',           f'{s.min():.1f} / {s.max():.1f} / {s.std():.1f}'),
        ('Subscore correlations',     corr_str),
    ]))

    flow.append(Paragraph('Score by source (median):', H2))
    src_med = g.groupby('source').composite_score.median().sort_values(ascending=False)
    flow.append(grid_table(
        ['source', 'n', 'median', 'p90'],
        [[s_, f'{int((g.source==s_).sum()):,}',
          f'{g[g.source==s_].composite_score.median():.1f}',
          f'{g[g.source==s_].composite_score.quantile(0.9):.1f}']
         for s_ in src_med.index],
        col_widths=(2.0*inch, 0.8*inch, 1.0*inch, 1.0*inch),
    ))

    flow.append(Paragraph('Score by reuse_asset_type (median):', H2))
    at_med = g.groupby('reuse_asset_type').composite_score.median().sort_values(ascending=False)
    flow.append(grid_table(
        ['reuse_asset_type', 'n', 'median', 'p90'],
        [[at, f'{int((g.reuse_asset_type==at).sum()):,}',
          f'{g[g.reuse_asset_type==at].composite_score.median():.1f}',
          f'{g[g.reuse_asset_type==at].composite_score.quantile(0.9):.1f}']
         for at in at_med.index],
        col_widths=(2.0*inch, 0.8*inch, 1.0*inch, 1.0*inch),
    ))

    # ---- Recommended action ----
    flow.append(Paragraph('5. Recommended action distribution', H1))
    ac = g.recommended_action.value_counts()
    order = ['Shortlist','Reuse Diligence','Parcel Pull','Monitor','Manual Review','Ignore']
    meanings = {
        'Shortlist':       'Top-tier; ready to pitch / immediate diligence priority',
        'Reuse Diligence': 'Score ≥60 AND contam/legacy flag; ESA required first',
        'Parcel Pull':     'Score ≥75; pull APN + owner before action',
        'Monitor':         'Score 65–74; track but not actionable now',
        'Manual Review':   'Kill-gate triggered; analyst inspection',
        'Ignore':          'Score &lt;65; not viable',
    }
    flow.append(grid_table(
        ['action', 'count', '% of total', 'meaning'],
        [[a, f'{int(ac.get(a,0)):,}', f'{100*ac.get(a,0)/len(g):.1f}%', meanings[a]]
         for a in order],
        col_widths=(1.2*inch, 0.7*inch, 0.8*inch, 4.1*inch),
    ))

    actionable = int(ac.get('Shortlist',0)+ac.get('Reuse Diligence',0)+ac.get('Parcel Pull',0))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(
        f'Actionable pipeline (Shortlist + Reuse Diligence + Parcel Pull): '
        f'<b>{actionable:,} sites</b> ({100*actionable/len(g):.1f}% of dataset).',
        BODY))

    # ---- Top sites ----
    flow.append(Paragraph('6. Top 20 sites by composite score', H1))
    top = g.nlargest(20, 'composite_score')
    rows = []
    for _, r in top.iterrows():
        nm = (str(r.site_name)[:26] if pd.notna(r.site_name) else '(unnamed)')
        rows.append([
            f'{r.composite_score:.1f}',
            r.state,
            r.candidate_id[:18],
            nm,
            f'{r.area_acres:.0f}',
            r.recommended_action,
        ])
    flow.append(grid_table(
        ['score','st','id','site_name','acres','action'],
        rows,
        col_widths=(0.5*inch, 0.3*inch, 1.5*inch, 2.4*inch, 0.6*inch, 1.5*inch),
    ))

    # ---- Caveats ----
    flow.append(Paragraph('7. Known caveats', H1))
    flow.append(Paragraph(
        '<b>Actionable ≠ available.</b> "Parcel Pull" and "Shortlist" mean '
        '<i>technically suitable</i>, not <i>for-sale</i>. Several Shortlist hits '
        'are active facilities (TSMC, Raytheon, PEMEX) that ranked top because '
        'they have everything a new tenant needs nearby; they are co-location '
        'leads, not direct-development sites. Downstream filtering by ownership / '
        'MLS / vacancy data is required.', BODY))
    flow.append(Paragraph(
        '<b>OSM industrial inventory ≠ vacancy.</b> The 4,310 OSM industrial '
        'polygons are tagged as currently-used industrial land. We have no signal '
        'on whether tenants would sell or lease. Treat as a shortlist for site-'
        'suitability scoring, not as a list of available parcels.', BODY))
    pct_pinned = 100 * (g.utility_score == 100).mean()
    median_anchors = int(g.num_anchors_in_range.median())
    dev_corr = g['dev_risk_score'].corr(g['composite_score'])
    flow.append(Paragraph(
        f'<b>utility_score still pins {pct_pinned:.0f}% of rows at 100.</b> '
        f'Even with the per-anchor cap of 30, reuse nodes have a median of '
        f'{median_anchors} anchors in 50 km. The other 75% of the composite '
        f'weight does the discriminating among top-tier sites.', BODY))
    flow.append(Paragraph(
        f'<b>dev_risk_score correlation is {dev_corr:+.2f}.</b> The six greenfield '
        'risk components (seismic, drought, radar, padus, wetland, floodway) '
        'don\'t vary much across this dataset -- most reuse polygons sit in '
        'similar risk profiles. Reuse-specific risk lives in '
        '<i>reuse_environmental_score</i>.', BODY))

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc.build(flow)
    print(f'\nWrote {OUT_PDF} ({OUT_PDF.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
