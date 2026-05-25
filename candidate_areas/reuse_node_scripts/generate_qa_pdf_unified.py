"""
generate_qa_pdf_unified -- QA report for the COMBINED greenfield + reuse
deliverable (final_candidates_phase1.parquet).

Output:
  candidate_areas/outputs/FinalCandidatesPhase1_QA_Report.pdf

Run: python candidate_areas/reuse_node_scripts/generate_qa_pdf_unified.py
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

UNIFIED_PATH = Path('candidate_areas/outputs/final_candidates_phase1.parquet')
OUT_PDF      = Path('candidate_areas/outputs/FinalCandidatesPhase1_QA_Report.pdf')

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
    print(f'Loading {UNIFIED_PATH} ...')
    g = gpd.read_parquet(UNIFIED_PATH)
    print(f'  {len(g):,} rows, {len(g.columns)} columns')

    gf = g[g.candidate_type == 'greenfield']
    rn = g[g.candidate_type == 'reuse_node']

    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=LETTER,
        leftMargin=0.7*inch, rightMargin=0.7*inch,
        topMargin=0.7*inch, bottomMargin=0.7*inch,
        title='Phase 1 Final Candidates QA Report',
    )
    flow = []

    # ---- Title ----
    flow.append(Paragraph('Final Candidates Phase 1 &mdash; QA Report', TITLE))
    flow.append(Paragraph(
        f'Combined greenfield + reuse-node deliverable &middot; '
        f'{date.today().isoformat()} &middot; '
        f'{len(g):,} scored candidates across CA, TX, AZ, NV, VA', META))
    flow.append(hr())

    # ---- Overview ----
    flow.append(Paragraph('1. Overview', H1))
    flow.append(Paragraph(
        'This is the unified Phase 1 deliverable produced by merging the '
        'greenfield candidate pipeline (Step1A-1J + Step2A-2G) with the new '
        'infrastructure reuse-node pipeline (StepR1-R6). Both layers are scored '
        'on a common composite framework (utility / buildability / supporting_infra '
        '/ dev_risk / reuse_environmental) and distinguished by '
        '<i>candidate_type</i>. Greenfield rows preserve the original column order '
        'from <i>candidates_final.parquet</i> byte-identically; reuse-only fields '
        'are interleaved at semantically adjacent positions.', BODY))

    flow.append(kv_table([
        ('Total candidates',         f'{len(g):,}'),
        ('Greenfield (CDL/exclusion-derived polygons)', f'{len(gf):,}'),
        ('Reuse-node (EIA / NRC / EPA / OSM)',          f'{len(rn):,}'),
        ('Columns',                  f'{len(g.columns):,} (153 shared + 19 reuse-only)'),
        ('CRS',                      f'EPSG:{g.crs.to_epsg()}'),
        ('Output file',              str(UNIFIED_PATH)),
    ]))

    # ---- Per-layer summary ----
    flow.append(Paragraph('2. Per-layer comparison', H1))
    src_counts_rn = rn.source.value_counts() if 'source' in rn.columns else pd.Series()

    rows = [
        ['greenfield', f'{len(gf):,}', f'{gf.composite_score.median():.1f}',
         f'{gf.composite_score.quantile(0.9):.1f}', f'{gf.area_acres.median():.0f}',
         'CDL-derived from keep-zones'],
        ['reuse_node', f'{len(rn):,}', f'{rn.composite_score.median():.1f}',
         f'{rn.composite_score.quantile(0.9):.1f}', f'{rn.area_acres.median():.0f}',
         'EIA + NRC + EPA + OSM industrial polygons'],
    ]
    flow.append(grid_table(
        ['candidate_type', 'rows', 'median composite', 'p90 composite', 'median acres', 'description'],
        rows,
        col_widths=(1.0*inch, 0.7*inch, 1.0*inch, 0.9*inch, 0.8*inch, 2.3*inch),
    ))

    # ---- Action distribution across both layers ----
    flow.append(Paragraph('3. Recommended action distribution', H1))
    actions = ['Shortlist','Reuse Diligence','Parcel Pull','Monitor','Manual Review','Ignore']
    rows = []
    for a in actions:
        n_gf = int((gf.recommended_action == a).sum())
        n_rn = int((rn.recommended_action == a).sum())
        rows.append([a, f'{n_gf:,}', f'{n_rn:,}', f'{n_gf + n_rn:,}'])
    rows.append(['TOTAL', f'{len(gf):,}', f'{len(rn):,}', f'{len(g):,}'])
    flow.append(grid_table(
        ['recommended_action', 'greenfield', 'reuse_node', 'combined'],
        rows,
        col_widths=(1.8*inch, 1.4*inch, 1.4*inch, 1.4*inch),
    ))

    actionable_combined = int(g.recommended_action.isin(['Shortlist','Reuse Diligence','Parcel Pull']).sum())
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(
        f'<b>Combined actionable pipeline</b> (Shortlist + Reuse Diligence + Parcel Pull): '
        f'<b>{actionable_combined:,} sites</b> ({100*actionable_combined/len(g):.1f}% of dataset).', BODY))

    flow.append(Paragraph(
        '<i>Reuse Diligence</i> is a reuse-only action label that fires when a '
        'site has known contamination or legacy-asset risk AND composite >= 60. '
        'Greenfield rows never receive this label by design.', BODY))

    # ---- Score distribution ----
    flow.append(Paragraph('4. Composite score distribution', H1))
    rows = []
    for label, sub in [('greenfield', gf), ('reuse_node', rn), ('combined', g)]:
        s = sub.composite_score
        rows.append([label,
                     f'{s.min():.1f}', f'{s.quantile(0.1):.1f}',
                     f'{s.median():.1f}', f'{s.quantile(0.9):.1f}',
                     f'{s.max():.1f}'])
    flow.append(grid_table(
        ['layer', 'min', 'p10', 'median', 'p90', 'max'],
        rows,
        col_widths=(1.2*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch),
    ))
    flow.append(Spacer(1, 6))
    flow.append(Paragraph(
        'Both layers calibrate to similar central tendency (medians within 2.5 pts), '
        'though the underlying mechanics differ: greenfield variance is driven by '
        'land-cover + slope + acreage tier; reuse variance by anchor density '
        '(post per-anchor-cap) + reuse_environmental_score.', BODY))

    # ---- Scoring framework comparison ----
    flow.append(Paragraph('5. Scoring framework per layer', H1))
    flow.append(grid_table(
        ['component', 'greenfield weight', 'reuse weight', 'notes'],
        [
            ['utility_score',              '40%', '25%', 'Reuse capped per-anchor at 30 to spread density signal'],
            ['buildability_score',         '20%', '15%', 'Reuse uses fixed land_cover=85 (industrial)'],
            ['supporting_infra_score',     '15%', '15%', 'Identical formula'],
            ['dev_risk_score',             '15%', '15%', 'Identical (greenfield 6-component)'],
            ['reuse_environmental_score',  '0%',  '20%', 'Reuse-only; contamination -45, legacy -20, decom +10'],
            ['site_control (Phase 2)',     '10%', '10%', 'Deferred -- composite renormalised over 0.90'],
        ],
        col_widths=(2.2*inch, 1.0*inch, 0.9*inch, 2.9*inch),
    ))

    # ---- Top sites per layer ----
    flow.append(Paragraph('6. Top 10 sites per layer', H1))

    flow.append(Paragraph('Greenfield:', H2))
    rows = []
    for _, r in gf.nlargest(10, 'composite_score').iterrows():
        county = str(r.county_name)[:18] if pd.notna(r.county_name) else '-'
        rows.append([
            f'{r.composite_score:.1f}', r.state, county,
            f'{r.area_acres:.0f}', r.recommended_action,
        ])
    flow.append(grid_table(
        ['score','st','county','acres','action'],
        rows,
        col_widths=(0.6*inch, 0.4*inch, 1.7*inch, 0.7*inch, 1.5*inch),
    ))

    flow.append(Paragraph('Reuse-node:', H2))
    rows = []
    for _, r in rn.nlargest(10, 'composite_score').iterrows():
        nm = (str(r.site_name)[:24] if pd.notna(r.site_name) else '(unnamed)')
        rows.append([
            f'{r.composite_score:.1f}', r.state, nm, f'{r.area_acres:.0f}',
            r.recommended_action,
        ])
    flow.append(grid_table(
        ['score','st','site_name','acres','action'],
        rows,
        col_widths=(0.6*inch, 0.4*inch, 2.4*inch, 0.7*inch, 1.5*inch),
    ))

    # ---- Caveats ----
    flow.append(Paragraph('7. Known caveats', H1))
    flow.append(Paragraph(
        '<b>Two ID schemes coexist.</b> Greenfield <i>candidate_id</i> is a UUID '
        '(CDL-derived polygons have no inherent name). Reuse-node <i>candidate_id</i> '
        'preserves the source identifier (e.g. <i>EIA-9-retiring</i>, '
        '<i>EPA-58162</i>, <i>NRC-Palo-Verde</i>, <i>OSM-way/1053101996</i>) so '
        'traceability back to EIA / EPA / NRC / OSM source records is preserved. '
        'Use <i>candidate_type</i> to filter.', BODY))
    flow.append(Paragraph(
        '<b>Actionable means technically suitable, not for-sale.</b> Both layers '
        'identify sites that meet geospatial + infrastructure criteria. Downstream '
        'filtering by ownership / MLS / vacancy data is still required. Several '
        'top-tier reuse Shortlist hits (TSMC, Raytheon, PEMEX) are active facilities '
        '-- treat as co-location leads, not direct-development sites.', BODY))
    flow.append(Paragraph(
        '<b>Some reuse fields intentionally NULL on greenfield rows</b> (and '
        'vice versa). <i>reuse_environmental_score</i>, <i>known_contamination_flag</i>, '
        '<i>aliased_site_count</i>, etc. are reuse-only. <i>cdl_group</i>, '
        '<i>cdl_year</i>, <i>nlcd_class</i>, <i>parcel_count</i>, etc. are '
        'greenfield-only. The schema union holds the superset; per-row filtering '
        'uses <i>candidate_type</i>.', BODY))
    flow.append(Paragraph(
        '<b>For per-layer detail see the standalone QA reports:</b> '
        '<i>Phase1_QA_Report.pdf</i> (greenfield) and '
        '<i>ReuseNodes_QA_Report.pdf</i> (reuse).', BODY))

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc.build(flow)
    print(f'\nWrote {OUT_PDF} ({OUT_PDF.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
