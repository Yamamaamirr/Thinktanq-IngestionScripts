"""
Generate Phase1_QA_Report.pdf in the same style as Phase1_Progress_Report.pdf.
First-person voice ("I" instead of "we").

Outputs:
  candidate_areas/outputs/Phase1_QA_Report.pdf
  candidate_areas/outputs/qa_summary.md  (rewritten to match the PDF content)
"""
from pathlib import Path
from datetime import date
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)

OUT_PDF = Path('candidate_areas/outputs/Phase1_QA_Report.pdf')
OUT_MD  = Path('candidate_areas/outputs/qa_summary.md')

# ============ Styles (match Phase1_Progress_Report look) ============
styles = getSampleStyleSheet()

TITLE   = ParagraphStyle('Title',   parent=styles['Heading1'],
                         fontSize=22, leading=26, textColor=colors.HexColor('#1f3a68'),
                         spaceAfter=4)
META    = ParagraphStyle('Meta',    parent=styles['BodyText'],
                         fontSize=10, leading=14, textColor=colors.black, spaceAfter=2)
H1      = ParagraphStyle('H1',      parent=styles['Heading2'],
                         fontSize=15, leading=20, textColor=colors.HexColor('#1f3a68'),
                         spaceBefore=16, spaceAfter=8)
H2      = ParagraphStyle('H2',      parent=styles['Heading3'],
                         fontSize=12, leading=16, textColor=colors.black,
                         spaceBefore=10, spaceAfter=4)
ITALIC_H= ParagraphStyle('Hi',      parent=styles['Heading4'],
                         fontSize=10.5, leading=14, textColor=colors.black,
                         fontName='Helvetica-Oblique', spaceBefore=6, spaceAfter=2)
BODY    = ParagraphStyle('Body',    parent=styles['BodyText'],
                         fontSize=10, leading=14, alignment=TA_LEFT, spaceAfter=6)
BULLET  = ParagraphStyle('Bullet',  parent=BODY, leftIndent=18, bulletIndent=6)
SMALL   = ParagraphStyle('Small',   parent=BODY, fontSize=9, leading=12)
CODE    = ParagraphStyle('Code',    parent=BODY, fontName='Courier', fontSize=9)


def hr():
    return HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc'),
                      spaceBefore=10, spaceAfter=10)


CELL = ParagraphStyle('Cell', parent=styles['BodyText'],
                      fontSize=9, leading=11, alignment=TA_LEFT, spaceAfter=0)
CELL_BOLD = ParagraphStyle('CellBold', parent=CELL,
                           fontName='Helvetica-Bold', fontSize=9.5)
CELL_CODE = ParagraphStyle('CellCode', parent=CELL,
                           fontName='Courier', fontSize=8.5)


def P(text, code=False, bold=False):
    """Wrap text in a Paragraph so it wraps within table cells."""
    style = CELL_CODE if code else (CELL_BOLD if bold else CELL)
    return Paragraph(str(text), style)


def wrap_row(row, code_cols=None):
    """Wrap each cell in a Paragraph. code_cols indices use monospace."""
    code_cols = code_cols or set()
    return [P(cell, code=(i in code_cols)) for i, cell in enumerate(row)]


def make_table(data, col_widths=None, header=True):
    style_cmds = [
        ('FONT',       (0, 0), (-1, -1), 'Helvetica', 9),
        ('GRID',       (0, 0), (-1, -1), 0.25, colors.HexColor('#bbbbbb')),
        ('VALIGN',     (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING',(0, 0), (-1, -1), 6),
        ('RIGHTPADDING',(0,0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0,0),(-1,-1), 4),
    ]
    if header:
        style_cmds += [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef7')),
            ('FONT',       (0, 0), (-1, 0), 'Helvetica-Bold', 9.5),
        ]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle(style_cmds))
    return t


# =====================================================================
# PDF BUILDING
# =====================================================================
def build_pdf():
    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=LETTER,
        leftMargin=0.85*inch, rightMargin=0.85*inch,
        topMargin=0.75*inch, bottomMargin=0.85*inch,
        title='Candidate Site Detection - Phase 1 QA Report',
        author='Yamama Amir',
    )

    flow = []

    # ============ Title block ============
    flow += [
        Paragraph('Candidate Site Detection &mdash; Phase 1 QA Report', TITLE),
        Paragraph(f'<b>As of:</b> 2026-05-17', META),
        Paragraph('<b>States covered:</b> CA, TX, AZ, NV, VA', META),
        Paragraph('<b>Pipeline version:</b> 1.0.0-phase1', META),
        Paragraph('<b>Total candidates:</b> 86,187', META),
        Paragraph('<b>Status:</b> Phase 1 complete. Ranked candidate output delivered.', META),
        hr(),
    ]

    # ============ Overview ============
    flow += [
        Paragraph('Overview', H1),
        Paragraph(
            "This report covers the Phase 1 ranked candidate output. The pipeline turned 932,456 km&sup2; "
            "of grid-relevant land across CA, TX, AZ, NV, and VA into 86,187 scored candidate polygons, "
            "each with composite score, sub-scores, confidence tier, recommended action, "
            "actionability status, and 3-5 reason codes. Every field in Owen's PDF (May 6) and his "
            "May 11/13 spec adjustments is present in the schema.",
            BODY),
        Paragraph(
            "I shipped the output in three formats - GeoParquet for analyst tooling, CSV for "
            "Excel-friendly review, and FlatGeobuf for QGIS/ArcGIS Pro. All three are byte-equivalent "
            "on content and sorted by composite score descending.",
            BODY),
        hr(),
    ]

    # ============ Files delivered ============
    flow += [
        Paragraph('Files delivered', H1),
        make_table([
            wrap_row(['File', 'Size', 'Purpose']),
            wrap_row(['candidates_final.parquet', '78.3 MB',  'GeoParquet, EPSG:5070, full 153-column schema']),
            wrap_row(['candidates_final.csv',     '287.2 MB', 'CSV with geometry as WKT and list columns as JSON']),
            wrap_row(['candidates_final.fgb',     '304.9 MB', 'FlatGeobuf, opens directly in QGIS / ArcGIS Pro']),
            wrap_row(['Phase1_QA_Report.pdf',     '~30 KB',   'This document']),
        ], col_widths=[2.0*inch, 0.9*inch, 3.5*inch]),
        Spacer(1, 6),
        Paragraph(
            "All three data files contain the same 86,187 rows. The parquet and FGB carry the geometry "
            "natively; the CSV serialises it as WKT. List columns (top_reason_codes, missing_modules) "
            "are JSON-encoded in the CSV for cross-tool compatibility.",
            BODY),
        hr(),
    ]

    # ============ Pipeline stages applied ============
    flow += [
        Paragraph('Pipeline stages applied', H1),
        Paragraph(
            "The pipeline runs as a sequential funnel. Each stage narrows the search area or annotates "
            "what survives. Below is a stage-by-stage rundown of what was actually built and run for "
            "this output.", BODY),

        Paragraph('Stages 1-4 &mdash; Keep-zone construction', H2),
        Paragraph(
            "<b>What:</b> Polygons covering every area worth screening from a grid perspective. "
            "Built from active-queue substations (25 km proximity buffers per PDF Rule 4) and 345 kV+ "
            "transmission corridors (10 km buffers). 230 kV is treated as a scoring signal, not a "
            "spatial anchor.", BODY),
        Paragraph(
            "<b>Result:</b> <font face='Courier'>keep_zones.parquet</font> &mdash; 932,456 km&sup2; "
            "across the five states.", BODY),

        Paragraph('Stage 5 &mdash; Hard exclusions', H2),
        Paragraph(
            "<b>What:</b> Layers that physically disqualify land. Any pixel inside these layers is "
            "removed from candidates entirely. I unioned all exclusion layers into a single mask, "
            "then subtracted from keep-zones in one operation per state-tile to avoid topology issues.", BODY),
        make_table([
            wrap_row(['Layer', 'Source', 'Treatment']),
            wrap_row(['FEMA regulatory floodway',           'FEMA NFHL',                 'Hard exclusion (PDF Rule 8)']),
            wrap_row(['FEMA VE coastal high-risk',          'FEMA NFHL',                 'Hard exclusion']),
            wrap_row(['Open water bodies',                  'Census TIGER',              'Hard exclusion']),
            wrap_row(['Protected areas (PAD-US GAP 1-2)',   'USGS PAD-US 4.1',           'Hard exclusion']),
            wrap_row(['Public GAP-3/4 land (FED, DOD, MIL, LOC; LREC, MIL, MPUB, NRA)', 'USGS PAD-US 4.1', 'Post-filter (added this run)']),
            wrap_row(['Slope > 15%',                        'USGS 3DEP',                 'Hard exclusion (PDF Rule 13)']),
            wrap_row(['NWI wetlands (heavy)',               'USFWS NWI',                 'Hard exclusion']),
            wrap_row(['Microsoft Building Footprints',      'MS USBF',                   'Density flag + manual_review trigger']),
            wrap_row(['Airport runway protection zones',    'OurAirports public CSV',    'Post-filter (large 5km, medium 3km, small 1.5km)']),
            wrap_row(['Major roads (Interstate, US Hwy, State Hwy)','TIGER 2025 PRISECROADS',  'Clip 25m buffer, drop fragments < 50 ac']),
            wrap_row(['FAA NEXRAD radar',                   'FAA',                       'Review flag at 3 mi (PDF Rule 9)']),
        ], col_widths=[2.4*inch, 1.5*inch, 2.5*inch]),
        Spacer(1, 6),
        Paragraph(
            "<i>Why airport and road exclusion were added this run:</i> spot-checks against the early "
            "output revealed candidates overlapping airports and bisected by highways. PDF Rule 10 "
            "lists both as hard-exclusion candidates 'if dataset available' - the datasets exist and "
            "are free, so I wired them in. The road exclusion does proper geometric subtraction "
            "(not just drop-if-intersects), then drops the residuals below the 50-acre minimum.", BODY),

        Paragraph('Stages 6-7 &mdash; Land-cover candidate generation', H2),
        Paragraph(
            "<b>What:</b> Within the buildable envelope, classify land that isn't already developed, "
            "underwater, or wetland-heavy. USDA CDL (30 m raster) is the primary input. CDL is "
            "rasterized onto the buildable mask, then connected-component labelling produces candidate "
            "polygons of contiguous same-class land at &ge; 50 acres (PDF Rule 14 minimum).", BODY),
        Paragraph(
            "<b>Subdivision (Owen May 14 disclosure):</b> the raw output had 1,238 polygons &ge; 5,000 "
            "acres - including a single 12.3 million-acre Texas blob. These aren't sites, they're "
            "regions. I subdivided them via a 4 km &times; 4 km grid aligned to NAD83 Albers, "
            "producing 44,725 site-scale fragments with parent_candidate_id metadata. Acreage "
            "preserved 99.96 percent.", BODY),
        Paragraph(
            "<b>Result:</b> 125,109 candidate polygons after subdivision and GAP-3/4 filtering, "
            "narrowed to 86,187 after slope hard exclusion, airport exclusion, and road clip.", BODY),

        Paragraph('Stage 8 &mdash; Enrichment (Step1A-1J)', H2),
        Paragraph(
            "Every candidate gets 11 enrichment passes that compute the scoring inputs:", BODY),
        make_table([
            wrap_row(['Pass', 'What it computes', 'Source']),
            wrap_row(['1A', 'slope_mean/max, slope_tier, acreage_tier, size_class', 'USGS 3DEP per-candidate']),
            wrap_row(['1B', 'seismic_hazard_pga, tier, valley_response',           'USGS NSHM23']),
            wrap_row(['1C', 'drought_level, drought_label',                       'USDM weekly']),
            wrap_row(['1D', 'PAD-US, wetland, floodway, FEMA-AE, radar adjacency', '5 source layers, sjoin_nearest']),
            wrap_row(['1E', 'nearest 500/345/230 kV distances and crosses flags', 'HIFLD-2024-09']),
            wrap_row(['1F', 'pipeline tier, estimated diameter, distance',       'PHMSA + operator tier table']),
            wrap_row(['1G', 'Class 1 rail distance, STRACNET flag, n_tracks',    'FRA / NRHM']),
            wrap_row(['1H', 'within_water_service_area, distance, pop_served',   'Water district shapefiles']),
            wrap_row(['1I', 'utility anchors with banded distance decay',        'ISO queues + HIFLD substations']),
            wrap_row(['1J', 'original/net buildable acreage and ratio',          'Convex hull proxy']),
            wrap_row(['merge', 'Assemble 152-column enriched table',             'All Step1 outputs']),
        ], col_widths=[0.6*inch, 3.4*inch, 2.4*inch]),
        Spacer(1, 6),
        Paragraph(
            "<i>Why banded distance decay, not flat radius (Owen May 13):</i> a candidate 2 km from a "
            "strong substation shouldn't score the same as one 45 km away. I band anchor contributions "
            "into 0-5 km (strong, decay weight 1.00), 5-10 km (moderate, 0.67), 10-25 km (weak, 0.33), "
            "and 25-50 km (regional context, 0.10). Each anchor in range contributes "
            "<font face='Courier'>base &times; voltage_mult &times; distance_decay &times; "
            "match_confidence</font>, summed and capped at 100.", BODY),

        Paragraph('Stage 9 &mdash; Scoring (Step2A-2G)', H2),
        Paragraph(
            "<b>Kill gates (Step2A):</b> before composite scoring, three rules tag candidates for "
            "manual review - slope_mean &gt; 15 percent, building_footprint &gt; 5 percent, or "
            "buildable_area_ratio &lt; 0.25. PDF Rule 17 says these candidates stay in the file with "
            "<font face='Courier'>candidate_status = manual_review</font> rather than being silently dropped.", BODY),
        Paragraph(
            "<b>Four sub-scores (Step2B-2E):</b>", BODY),
        make_table([
            wrap_row(['Sub-score', 'Weight', 'Inputs']),
            wrap_row(['utility_score',          '40%', 'Anchors within 50 km; queue MW tier &times; status &times; voltage &times; distance decay &times; match confidence']),
            wrap_row(['buildability_score',     '20%', 'land_cover_score + constraint_score + slope_tier_score + building_footprint_pct']),
            wrap_row(['supporting_infra_score', '15%', '0.35 &times; transmission + 0.25 &times; pipeline + 0.20 &times; rail + 0.20 &times; water']),
            wrap_row(['dev_risk_score',         '15%', '0.25 &times; seismic + 0.20 &times; drought + 0.15 &times; radar + 0.15 &times; padus + 0.15 &times; wetland + 0.10 &times; floodway']),
            wrap_row(['site_control_score',     '10%', 'Deferred to Phase 2 (parcel data)']),
        ], col_widths=[1.6*inch, 0.6*inch, 4.2*inch]),
        Spacer(1, 6),
        Paragraph(
            "<b>Composite (Step2F):</b> <font face='Courier'>"
            "composite = (0.40&times;utility + 0.20&times;buildability + 0.15&times;supporting + "
            "0.15&times;dev_risk) / 0.90</font>. The 0.90 divisor renormalises over the 90 percent of "
            "weight that's actually observable in Phase 1, so candidates aren't capped just because "
            "the parcel module isn't built yet.", BODY),
        Paragraph(
            "<i>Confidence (worst-of-three):</i> takes the minimum of (a) data coverage tier from "
            "<font face='Courier'>missing_modules</font> count, (b) primary anchor "
            "<font face='Courier'>match_confidence</font>, and (c) signal corroboration from how many "
            "sub-scores cross 50.", BODY),
        Paragraph(
            "<b>Action labels (PDF Rule 29):</b> Ignore / Monitor / Manual Review / Parcel Pull / "
            "Utility Desk Check / Ownership Review / Reuse Diligence / Shortlist. "
            "<b>Actionability (May 11 vocab):</b> do_not_pitch / internal_diligence_only / "
            "apn_owner_pull_required / broker_verify_required / nda_teaser_possible / "
            "buyer_ready_with_caveats.", BODY),
        Paragraph(
            "<b>Reason codes:</b> 3-5 short codes per row, drawn from a 25-code vocabulary mixing "
            "positives (strong_queue_signal, ideal_slope, tier1_pipeline_within_5mi, 345kv_anchor_in_range, "
            "low_seismic_risk, building_density_low&hellip;), negatives "
            "(wetland_adjacent_500m, elevated_slope_band, drought_tier_high, floodway_adjacent_500m, "
            "high_seismic_zone, padus_adjacent_500m&hellip;), and uncertain "
            "(allocation_risk_possible, queue_anchor_zone_fallback&hellip;). I removed the "
            "always-fire clutter codes (pipeline_diameter_estimated, owner_data_missing) so the "
            "codes actually distinguish rows from each other.", BODY),
        hr(),

        # ============ Final distribution ============
        Paragraph('Final distribution', H1),

        Paragraph('By state', H2),
        make_table([
            wrap_row(['State', 'Candidates', 'Share', 'Notes']),
            wrap_row(['TX',    '60,806', '70.6%', 'Flat terrain + ERCOT queue density']),
            wrap_row(['VA',    '16,100', '18.7%', 'Eastern farmland, near PJM grid']),
            wrap_row(['CA',    '6,603',  '7.7%',  'Coastal + Sierra carved out by exclusions']),
            wrap_row(['AZ',    '2,004',  '2.3%',  'Mostly federal land (BLM, parks)']),
            wrap_row(['NV',    '674',    '0.8%',  'Almost all mountainous or federal']),
        ], col_widths=[0.5*inch, 1.0*inch, 0.7*inch, 4.2*inch]),
        Spacer(1, 6),
        Paragraph(
            "Texas dominates not because of a TX bias in the pipeline, but because TX has the largest "
            "contiguous flat shrubland and grassland, the most active ERCOT interconnection queue, and "
            "the fewest federal-land carve-outs. NV's count is small because the Great Basin Range "
            "fails the 15 percent slope rule and the federal-land filter at the same time.",
            BODY),

        Paragraph('By recommended_action', H2),
        make_table([
            wrap_row(['Action', 'Count', 'Cutoff / meaning']),
            wrap_row(['Shortlist',     '569',    'composite &ge; 90 AND 500-5,000 ac AND clean slope AND &ge; 3 anchors AND medium/high confidence']),
            wrap_row(['Parcel Pull',   '46,510', 'composite &ge; 75 (parcel data missing &mdash; enrichment recommended)']),
            wrap_row(['Monitor',       '10,746', '65 &le; composite &lt; 75 (interesting but not actionable yet)']),
            wrap_row(['Manual Review', '231',    'Kill gate flagged (slope, footprint, or ratio) per PDF Rule 17']),
            wrap_row(['Ignore',        '28,131', 'composite &lt; 65 (do not pursue)']),
        ], col_widths=[1.1*inch, 0.7*inch, 4.6*inch]),

        Paragraph('By confidence (worst-of-three)', H2),
        make_table([
            ['Confidence', 'Count', 'Share'],
            ['medium', '67,929', '78.8%'],
            ['low',    '18,258', '21.2%'],
        ], col_widths=[1.1*inch, 1.0*inch, 0.8*inch]),
        Spacer(1, 4),
        Paragraph(
            "No <i>high</i>-confidence rows exist in Phase 1, by design: the site_control module is "
            "missing for every candidate, which caps the data-coverage tier at <i>medium</i>. "
            "Confidence will rise to <i>high</i> for shortlisted candidates after Phase 2 parcel "
            "enrichment plugs in.", BODY),

        Paragraph('Composite score distribution', H2),
        make_table([
            ['Stat',  'Value'],
            ['min',   '25.61'],
            ['p50',   '77.58'],
            ['p90',   '87.25'],
            ['max',   '95.03'],
        ], col_widths=[1.0*inch, 1.0*inch]),

        Paragraph('Sub-score medians', H2),
        make_table([
            ['Sub-score',                 'Weight',         'Median', 'p90'],
            ['utility_score',             '40%',            '91.92',  '100.00'],
            ['buildability_score',        '20%',            '78.75',  '87.75'],
            ['supporting_infra_score',    '15%',            '56.00',  '76.75'],
            ['dev_risk_score',            '15%',            '78.00',  '87.75'],
            ['site_control_score',        '10% (deferred)', '-',      '-'],
        ], col_widths=[2.0*inch, 1.4*inch, 0.9*inch, 0.9*inch]),

        Paragraph('Data coverage (per-row, Owen May 11 ask)', H2),
        make_table([
            wrap_row(['data_coverage_pct', 'Count',  'What is missing']),
            wrap_row(['90',                '67,932', 'site_control only (Phase 2)']),
            wrap_row(['80',                '15,305', 'site_control + 1 other module']),
            wrap_row(['70',                '2,743',  'site_control + 2 modules']),
            wrap_row(['60',                '146',    'site_control + 3 modules']),
            wrap_row(['50',                '61',     'site_control + 4 modules']),
        ], col_widths=[1.6*inch, 1.0*inch, 3.6*inch]),
        Spacer(1, 4),
        Paragraph(
            "I made <font face='Courier'>data_coverage_pct</font> vary per row this run "
            "(previously hardcoded to 90). The 19,455 rows below 90 are mostly candidates without an "
            "ISO-queue anchor in range, or candidates outside any water service district. The "
            "<font face='Courier'>missing_modules</font> list per row spells out which specific "
            "modules contributed no data.", BODY),
        hr(),

        # ============ Top 10 Shortlist ============
        Paragraph('Top 10 Shortlist candidates', H1),
        Paragraph(
            "These are the immediate-attention candidates - they cleared composite &ge; 90, fell in "
            "the 500-5,000 ac campus range, had no slope review flag, had at least three anchors in "
            "range, and reached medium confidence. All are in Texas, where utility activity and clean "
            "flat shrubland/grassland concentrate.", BODY),
        make_table([
            ['candidate_id', 'state', 'county', 'acres', 'composite', 'util', 'build'],
            ['973f2d1a..', 'TX', 'Harris',     '719',   '94.44', '100', '95'],
            ['e767bb30..', 'TX', 'Caldwell',   '1,887', '94.31', '100', '96'],
            ['8138a608..', 'TX', 'Atascosa',   '4,020', '94.31', '100', '96'],
            ['3de93707..', 'TX', 'Cameron',    '559',   '94.28', '100', '95'],
            ['a23d02af..', 'TX', 'Kenedy',     '1,418', '94.19', '100', '93'],
            ['79518297..', 'TX', 'Brazoria',   '1,557', '94.14', '100', '96'],
            ['9601fd96..', 'TX', 'Johnson',    '557',   '94.08', '100', '89'],
            ['64e71f76..', 'TX', 'Galveston',  '627',   '94.07', '100', '94'],
            ['d6b1dc82..', 'TX', 'Johnson',    '897',   '93.92', '100', '86'],
            ['bd5693d9..', 'TX', 'Caldwell',   '579',   '93.69', '100', '94'],
        ], col_widths=[1.0*inch, 0.5*inch, 1.0*inch, 0.7*inch, 0.9*inch, 0.6*inch, 0.6*inch]),
        Spacer(1, 6),
        Paragraph(
            "To replicate the shortlist in QGIS or any other viewer, open "
            "<font face='Courier'>candidates_final.fgb</font> and filter "
            "<font face='Courier'>recommended_action == 'Shortlist'</font>.", BODY),
        hr(),

        # ============ Top reason codes ============
        Paragraph('Top reason codes (frequency across all 86,187 rows)', H1),
        Paragraph(
            "Reason codes are the short, controlled-vocabulary explanations of why a row scored where "
            "it did. Each row carries 3-5 codes. The distribution below shows how often each fires.", BODY),
        make_table([
            ['Code',                              'Sign', 'Count'],
            ['strong_queue_signal',               '+',    '73,888'],
            ['interconnection_agreement_executed','+',    '50,806'],
            ['drought_tier_high',                 '-',    '38,092'],
            ['no_executed_ia_in_range',           '-',    '32,609'],
            ['wetland_adjacent_500m',             '-',    '31,761'],
            ['345kv_anchor_in_range',             '+',    '27,782'],
            ['tier1_pipeline_within_5mi',         '+',    '25,714'],
            ['low_seismic_risk',                  '+',    '24,549'],
            ['elevated_slope_band',               '-',    '23,695'],
            ['ideal_slope',                       '+',    '18,613'],
            ['allocation_risk_possible',          '?',    '15,906'],
            ['class1_rail_within_3mi',            '+',    '11,869'],
            ['500kv_anchor_in_range',             '+',    '10,629'],
            ['building_density_low',              '+',    '7,685'],
            ['no_queue_activity',                 '-',    '5,811'],
            ['padus_adjacent_500m',               '-',    '4,335'],
            ['small_candidate',                   '-',    '3,918'],
            ['high_seismic_zone',                 '-',    '3,711'],
            ['floodway_adjacent_500m',            '-',    '2,921'],
            ['queue_anchor_zone_fallback',        '?',    '2,738'],
        ], col_widths=[3.0*inch, 0.8*inch, 1.2*inch]),
        hr(),

        # ============ Known limitations ============
        Paragraph('Known limitations and caveats', H1),
        Paragraph(
            "Every item below is documented so Owen can decide whether it's worth a Phase 1.5 follow-up "
            "or can be carried forward as-is.", BODY),

        Paragraph('1. Mega-candidate subdivision uses inherited slope', H2),
        Paragraph(
            "<b>What I did:</b> for the 1,238 raw polygons &ge; 5,000 acres, I subdivided into &le; 3,953-acre "
            "fragments using a 4 km grid. Acreage preserved 99.96 percent. Fragments carry "
            "<font face='Courier'>parent_candidate_id</font> back to the original polygon.", BODY),
        Paragraph(
            "<b>Limitation:</b> the 44,725 fragments inherit their parent's "
            "<font face='Courier'>slope_max</font> rather than getting per-fragment 3DEP sampling. "
            "This is conservative - some fragments in the flat parts of a mostly-flat-but-one-corner-steep "
            "parent get hard-excluded by Rule 13 (slope_max &gt; 15) when they shouldn't. Net impact: "
            "probably 5-15 percent of fragments are over-excluded for slope, mostly in mountainous "
            "parents.", BODY),
        Paragraph(
            "<b>Phase 1.5 follow-up:</b> re-sample 3DEP elevation per fragment (~1-2 hours of tile "
            "downloads). Until done, slope filtering on fragments is conservative.", BODY),

        Paragraph('2. GAP-3/4 exclusion uses public/military/rec subset only', H2),
        Paragraph(
            "<b>What I did:</b> added a post-filter that drops candidates intersecting GAP-3 or GAP-4 "
            "PAD-US polygons where <font face='Courier'>Mang_Type</font> is FED/DOD/MIL/LOC or "
            "<font face='Courier'>Des_Tp</font> is LREC/MIL/MPUB/NRA. This catches city parks, military "
            "bases, federal recreation areas. 49,130 polygons across the 5 states; dropped 13,647 "
            "candidates (mostly Cave Buttes-type overlaps).", BODY),
        Paragraph(
            "<b>Limitation:</b> private GAP-4 (conservation easements, hunting clubs) is kept as "
            "adjacency-only since those are still developable in principle. If a stricter rule is "
            "wanted, easy to revise.", BODY),

        Paragraph('3. Road clip uses TIGER PRISECROADS only', H2),
        Paragraph(
            "<b>What I did:</b> downloaded TIGER 2025 PRISECROADS for all 5 states "
            "(34,490 segments: Interstate, US Highway, State Highway, major named roads), buffered 25 m, "
            "and subtracted from candidate geometries (proper geometric difference, not drop-if-intersects). "
            "10,947 candidates had geometry clipped; 61 dropped to &lt; 50 acres after clipping.", BODY),
        Paragraph(
            "<b>Limitation:</b> minor roads (residential streets, dirt roads) are not subtracted "
            "individually. CDL's developed-medium/high pixel class already excludes most residential "
            "land upstream, so this gap is small in practice. If full all-roads coverage is wanted, "
            "TIGER ROADS (per-county, multi-GB) would need to be ingested.", BODY),

        Paragraph('4. NLCD cross-check deferred (PDF Rule 11)', H2),
        Paragraph(
            "<font face='Courier'>nlcd_class</font>, <font face='Courier'>nlcd_label</font>, and "
            "<font face='Courier'>landcover_confidence_score</font> are null in every row. CDL is the "
            "primary land-cover input per PDF Rule 11; NLCD cross-check is a documented Phase 2 follow-up.", BODY),

        Paragraph('5. Phase 2 module placeholders (all null by design)', H2),
        Paragraph(
            "The following columns are reserved schema slots that Phase 2 enrichment populates - they "
            "are 100 percent null in this output by intent. Owen approved this on May 11 with the "
            "<font face='Courier'>data_coverage_pct</font> mechanism.", BODY),
        Paragraph(
            "<b>Parcel/ownership (PDF Rule 21):</b> parcel_count, owner_count, largest_owner_acres, "
            "largest_owner_pct_of_candidate, assessed_value_total, assessed_value_per_acre, "
            "last_sale_date, last_sale_price, land_use_code, zoning_code, road_frontage_flag, "
            "legal_access_flag.", BODY),
        Paragraph(
            "<b>Economic proxy (PDF Rule 22):</b> site_control_score, economic_proxy_score.", BODY),
        Paragraph(
            "<b>Utility feasibility (PDF Rule 23):</b> serving_utility, utility_territory_known, "
            "nearest_load_serving_node, utility_service_feasibility_score, utility_review_required.", BODY),
        Paragraph(
            "<b>Communications (PDF Rule 24):</b> communications_route_distance, "
            "communications_provider_count, communications_access_score.", BODY),
        Paragraph(
            "<b>Water capacity (PDF Rule 25):</b> water_capacity_known, water_capacity_review_required.", BODY),
        Paragraph(
            "<b>Local jurisdiction (PDF Rule 26):</b> jurisdiction_review_required, local_policy_notes.", BODY),
        Paragraph(
            "<b>Manual imagery QA (PDF Rule 28):</b> manual_imagery_review_status, "
            "manual_imagery_review_notes.", BODY),
        Paragraph(
            "<b>Route complexity (PDF Rule 5):</b> route_complexity_score, route_complexity_notes.", BODY),
        Paragraph(
            "<font face='Courier'>parcel_owner_module_status = 'not_built'</font> on every row, so "
            "downstream consumers can see the module status explicitly rather than guessing.", BODY),

        Paragraph('6. Shortlist band activation (May 11 ask)', H2),
        Paragraph(
            "PDF Rule 29 defined Shortlist as the top action label but the original rule "
            "(<font face='Courier'>composite &ge; 85 and utility_review_required</font>) never fired "
            "because <font face='Courier'>utility_review_required</font> is a Phase 2 field "
            "(always null). I activated the band with a tighter criterion: "
            "<font face='Courier'>composite &ge; 90 AND 500 &le; acres &le; 5,000 AND not "
            "slope_review_flag AND not oversized_flag AND confidence &ge; medium AND num_anchors "
            "&ge; 3</font>. 569 candidates qualified.", BODY),

        Paragraph('7. Wetland adjacency rate is high (structural)', H2),
        Paragraph(
            "Roughly 37 percent of the final file (31,761 candidates) is within 500 m of an NWI "
            "wetland polygon. This is structural - the NWI wetland dataset is dense across TX and VA, "
            "and the adjacency search radius is 10 km. Wetlands were already physically subtracted "
            "from candidates upstream; this is <i>adjacency</i>, not overlap. The "
            "<font face='Courier'>near_wetland_flag</font> and the "
            "<font face='Courier'>wetland_adjacent_500m</font> reason code surface this so an analyst "
            "can see the water-feature context without having to overlay layers manually.", BODY),

        Paragraph('8. IA-executed-nearby rate is high but well-distributed', H2),
        Paragraph(
            "50,806 candidates (59 percent of file) carry "
            "<font face='Courier'>ia_executed_nearby = True</font>. The boolean fires when an IA-executed "
            "anchor is anywhere within the 50 km radius. Among those that fire, 99.9 percent have the "
            "IA-executed anchor within 25 km, median distance 14 km - so the boolean isn't capturing "
            "distant noise. The banded distance decay in <font face='Courier'>utility_score</font> "
            "still weights closer anchors much more heavily.", BODY),

        Paragraph('9. Three TX candidates have null county_name', H2),
        Paragraph(
            "Three candidates near the TX Gulf coast (Brownsville area) and the TX/AR/LA tri-state "
            "corner have centroids that fall just outside any county polygon in the Census TIGER "
            "county boundaries file. The candidate_id and state are populated correctly; only "
            "county_name is null. Acceptable as an edge case.", BODY),

        Paragraph('10. Mega-polygon scoring is at-the-fragment, not at-the-region', H2),
        Paragraph(
            "After subdivision, the composite score reflects each &lt; 5,000-acre fragment, not the "
            "underlying 12.3 million-acre region. This is the correct behaviour for siting decisions, "
            "but it does mean that two adjacent fragments of the same parent will often have similar "
            "scores. <font face='Courier'>parent_candidate_id</font> is preserved if regional context "
            "is wanted.", BODY),
        hr(),

        # ============ Reproducibility ============
        Paragraph('Reproducibility and audit trail', H1),
        Paragraph(
            "Every row carries 11 versioning fields populated to the same value across the run, per "
            "PDF Rule 30:", BODY),
        make_table([
            ['Field',                          'Value'],
            ['run_id',                         'UUID (one per execution)'],
            ['run_date',                       '2026-05-16'],
            ['scoring_model_version',          '1.0.0-phase1'],
            ['exclusion_model_version',        '1.0'],
            ['cdl_year',                       '2025'],
            ['padus_version',                  '4.1'],
            ['fema_nfhl_date',                 '2025'],
            ['nwi_date',                       '2025'],
            ['transmission_dataset_version',   'HIFLD-2024-09'],
            ['queue_dataset_date',             '2026-05'],
            ['dem_dataset_version',            'USGS-3DEP-1arcsec'],
        ], col_widths=[2.6*inch, 3.2*inch]),
        Spacer(1, 6),
        Paragraph('Pipeline scripts (all in the GitHub repo)', H2),
        Paragraph(
            "<font face='Courier'>candidate_areas/scripts/Step7d_subdivide_megas.py</font> &mdash; "
            "subdivision of &ge; 5,000-acre polygons.", BODY),
        Paragraph(
            "<font face='Courier'>candidate_areas/scripts/Step7e_gap34_post_filter.py</font> &mdash; "
            "GAP-3/4 public-land post-filter.", BODY),
        Paragraph(
            "<font face='Courier'>candidate_areas/scripts/Step7f_airport_post_filter.py</font> &mdash; "
            "airport runway protection zone exclusion.", BODY),
        Paragraph(
            "<font face='Courier'>candidate_areas/scripts/Step7g_road_clip_exclusion.py</font> &mdash; "
            "TIGER PRISECROADS road clip.", BODY),
        Paragraph(
            "<font face='Courier'>ingestion_scripts/usgs_slope/inherit_slope_for_fragments.py</font> &mdash; "
            "slope inheritance for fragments.", BODY),
        Paragraph(
            "<font face='Courier'>candidate_areas/enrichment_scripts/Step1A-1J.py</font> + "
            "<font face='Courier'>Step1_merge.py</font> &mdash; enrichment.", BODY),
        Paragraph(
            "<font face='Courier'>candidate_areas/scoring_scripts/Step2A-2G.py</font> &mdash; "
            "kill gates, four sub-scores, composite, confidence, actions, reason codes, versioning.", BODY),
        Paragraph(
            "<font face='Courier'>candidate_areas/scoring_scripts/Step3B_apply_slope_hard_exclusion.py</font> "
            "&mdash; final slope hard-exclusion and three-format export.", BODY),
        Paragraph(
            "<font face='Courier'>candidate_areas/scoring_scripts/Step6_final_deep_verify.py</font> "
            "&mdash; 95-check verification (94/95 PASS; 1 informational on structural water-feature "
            "co-occurrence).", BODY),
        hr(),

        # ============ Next steps ============
        Paragraph('What is coming next', H1),
        Paragraph('Phase 2 &mdash; Selective enrichment', H2),
        Paragraph(
            "Targeted Regrid pulls (or county GIS / ArcGIS parcel endpoints) for the 569 Shortlist "
            "candidates first, then expand to Parcel Pull. Populates parcel_count, owner_count, "
            "assessed_value_per_acre, last_sale_date, land_use_code, zoning_code. This is where "
            "<font face='Courier'>data_coverage_pct</font> rises from 90 to 100 and confidence can "
            "tier up to <i>high</i> on shortlisted rows.", BODY),
        Paragraph('Phase 1.5 &mdash; Tighten the corners', H2),
        Paragraph(
            "Per-fragment 3DEP slope sampling (removes the conservative over-exclusion noted in "
            "limitation #1). NLCD cross-check for non-agricultural CDL classes (limitation #4). "
            "Optionally a tighter <font face='Courier'>ia_executed_within_10km</font> boolean as a "
            "second view of the IA signal (limitation #8).", BODY),
        Paragraph('Phase 3 &mdash; Optional overlays', H2),
        Paragraph(
            "Active listings (LandWatch / Lands of America / Crexi) as a motivated-seller overlay "
            "on the existing shortlist. Owen agreed on May 3 that this is a Phase 3 item, not core.", BODY),
    ]

    doc.build(flow)
    print(f'Wrote {OUT_PDF}  ({OUT_PDF.stat().st_size/1024:.1f} KB)')


def write_markdown_mirror():
    """Rewrite qa_summary.md to mirror the PDF content, first-person voice."""
    content = """# Candidate Site Detection -- Phase 1 QA Report

**As of:** 2026-05-17
**States covered:** CA, TX, AZ, NV, VA
**Pipeline version:** 1.0.0-phase1
**Total candidates:** 86,187
**Status:** Phase 1 complete. Ranked candidate output delivered.

---

## Overview

This report covers the Phase 1 ranked candidate output. The pipeline turned 932,456 km^2 of grid-relevant land across CA, TX, AZ, NV, and VA into 86,187 scored candidate polygons, each with composite score, sub-scores, confidence tier, recommended action, actionability status, and 3-5 reason codes. Every field in Owen's PDF (May 6) and his May 11/13 spec adjustments is present in the schema.

I shipped the output in three formats - GeoParquet for analyst tooling, CSV for Excel-friendly review, and FlatGeobuf for QGIS/ArcGIS Pro. All three are byte-equivalent on content and sorted by composite score descending.

---

## Files delivered

| File | Size | Purpose |
|---|---|---|
| `candidates_final.parquet` | 78.3 MB | GeoParquet, EPSG:5070, full 153-column schema |
| `candidates_final.csv` | 287.2 MB | CSV with geometry as WKT and list columns as JSON |
| `candidates_final.fgb` | 304.9 MB | FlatGeobuf, opens directly in QGIS / ArcGIS Pro |
| `Phase1_QA_Report.pdf` | ~80 KB | This document |

All three data files contain the same 86,187 rows. The parquet and FGB carry the geometry natively; the CSV serialises it as WKT. List columns (top_reason_codes, missing_modules) are JSON-encoded in the CSV.

---

## Pipeline stages applied

The pipeline runs as a sequential funnel. Each stage narrows the search area or annotates what survives.

### Stages 1-4 -- Keep-zone construction
**What:** Polygons covering every area worth screening from a grid perspective. Built from active-queue substations (25 km proximity buffers per PDF Rule 4) and 345 kV+ transmission corridors (10 km buffers). 230 kV is a scoring signal, not a spatial anchor.

**Result:** `keep_zones.parquet` -- 932,456 km^2 across the five states.

### Stage 5 -- Hard exclusions
I unioned all exclusion layers into a single mask, then subtracted from keep-zones in one operation per state-tile to avoid topology issues.

| Layer | Source | Treatment |
|---|---|---|
| FEMA regulatory floodway | FEMA NFHL | Hard exclusion (PDF Rule 8) |
| FEMA VE coastal high-risk | FEMA NFHL | Hard exclusion |
| Open water bodies | Census TIGER | Hard exclusion |
| Protected areas (PAD-US GAP 1-2) | USGS PAD-US 4.1 | Hard exclusion |
| Public GAP-3/4 (FED/DOD/MIL/LOC, LREC/MIL/MPUB/NRA) | USGS PAD-US 4.1 | Post-filter (added this run) |
| Slope > 15% | USGS 3DEP | Hard exclusion (PDF Rule 13) |
| NWI wetlands (heavy) | USFWS NWI | Hard exclusion |
| Microsoft Building Footprints | MS USBF | Density flag + manual_review trigger |
| Airport runway protection zones | OurAirports public CSV | Post-filter (large 5km / medium 3km / small 1.5km) |
| Major roads (Interstate/US/State Hwy) | TIGER 2025 PRISECROADS | Clip 25m buffer, drop fragments <50 ac |
| FAA NEXRAD radar | FAA | Review flag at 3 mi (PDF Rule 9) |

*Why airport and road exclusion were added this run:* spot-checks revealed candidates overlapping airports and bisected by highways. PDF Rule 10 lists both as hard-exclusion candidates "if dataset available" - the datasets exist and are free, so I wired them in. The road exclusion does proper geometric subtraction (not just drop-if-intersects).

### Stages 6-7 -- Land-cover candidate generation
**What:** USDA CDL (30 m raster) classifies land that isn't already developed, underwater, or wetland-heavy. CDL is rasterized onto the buildable mask, then connected-component labelling produces candidate polygons of contiguous same-class land at >= 50 acres (PDF Rule 14).

**Subdivision (Owen May 14 disclosure):** the raw output had 1,238 polygons >= 5,000 acres - including a single 12.3 million-acre Texas blob. These aren't sites, they're regions. I subdivided them via a 4 km x 4 km grid aligned to NAD83 Albers, producing 44,725 site-scale fragments with `parent_candidate_id` metadata. Acreage preserved 99.96%.

**Result:** 125,109 candidate polygons after subdivision + GAP-3/4 filtering, narrowed to 86,187 after slope hard exclusion, airport exclusion, and road clip.

### Stage 8 -- Enrichment (Step1A-1J)
Every candidate gets 11 enrichment passes:

| Pass | What it computes | Source |
|---|---|---|
| 1A | slope_mean/max, slope_tier, acreage_tier, size_class | USGS 3DEP per-candidate |
| 1B | seismic_hazard_pga, tier, valley_response | USGS NSHM23 |
| 1C | drought_level, drought_label | USDM weekly |
| 1D | PAD-US/wetland/floodway/FEMA-AE/radar adjacency | 5 source layers, sjoin_nearest |
| 1E | nearest_500/345/230 kV distances + crosses flags | HIFLD-2024-09 |
| 1F | pipeline tier + estimated diameter + distance | PHMSA + operator tier table |
| 1G | Class 1 rail distance + STRACNET flag + n_tracks | FRA / NRHM |
| 1H | within_water_service_area + distance + pop_served | Water district shapefiles |
| 1I | utility anchors with banded distance decay | ISO queues + HIFLD substations |
| 1J | original/net buildable acreage + ratio | Convex hull proxy |
| merge | Assemble 152-column enriched table | All Step1 outputs |

*Why banded distance decay, not flat radius (Owen May 13):* a candidate 2 km from a strong substation shouldn't score the same as one 45 km away. Anchor contributions are banded into 0-5 km (decay 1.00), 5-10 km (0.67), 10-25 km (0.33), 25-50 km (0.10).

### Stage 9 -- Scoring (Step2A-2G)
**Kill gates (Step2A):** three rules tag candidates for manual review - slope_mean > 15%, building_footprint > 5%, or buildable_area_ratio < 0.25.

**Four sub-scores (Step2B-2E):**

| Sub-score | Weight | Inputs |
|---|---|---|
| utility_score | 40% | Anchors within 50 km; queue MW tier x status x voltage x distance decay x match confidence |
| buildability_score | 20% | land_cover_score + constraint_score + slope_tier_score + building_footprint_pct |
| supporting_infra_score | 15% | 0.35*transmission + 0.25*pipeline + 0.20*rail + 0.20*water |
| dev_risk_score | 15% | 0.25*seismic + 0.20*drought + 0.15*radar + 0.15*padus + 0.15*wetland + 0.10*floodway |
| site_control_score | 10% | Deferred to Phase 2 (parcel data) |

**Composite (Step2F):** `composite = (0.40*utility + 0.20*buildability + 0.15*supporting + 0.15*dev_risk) / 0.90`. The 0.90 divisor renormalises over the 90 percent of weight observable in Phase 1.

**Action labels (PDF Rule 29):** Ignore / Monitor / Manual Review / Parcel Pull / Utility Desk Check / Ownership Review / Reuse Diligence / Shortlist.
**Actionability (May 11 vocab):** do_not_pitch / internal_diligence_only / apn_owner_pull_required / broker_verify_required / nda_teaser_possible / buyer_ready_with_caveats.
**Reason codes:** 3-5 short codes per row from a 25-code vocabulary mixing positives, negatives, and uncertain markers. I removed always-fire clutter codes (pipeline_diameter_estimated, owner_data_missing).

---

## Final distribution

### By state
| State | Candidates | Share |
|---|---|---|
| TX | 60,806 | 70.6% |
| VA | 16,100 | 18.7% |
| CA | 6,603 | 7.7% |
| AZ | 2,004 | 2.3% |
| NV | 674 | 0.8% |

Texas dominates not because of bias but because TX has the largest contiguous flat shrubland and grassland, the most active ERCOT queue, and the fewest federal-land carve-outs. NV's count is small because the Great Basin Range fails the 15% slope rule and the federal-land filter at the same time.

### By recommended_action
| Action | Count | Cutoff / meaning |
|---|---|---|
| Shortlist | 569 | composite >= 90 + 500-5,000 ac + clean slope + >= 3 anchors + medium/high confidence |
| Parcel Pull | 46,510 | composite >= 75 (parcel data missing - enrichment recommended) |
| Monitor | 10,746 | 65 <= composite < 75 (interesting but not actionable yet) |
| Manual Review | 231 | Kill gate flagged (slope/footprint/ratio) per PDF Rule 17 |
| Ignore | 28,131 | composite < 65 (do not pursue) |

### By confidence (worst-of-three)
| Confidence | Count | Share |
|---|---|---|
| medium | 67,929 | 78.8% |
| low | 18,258 | 21.2% |

No *high*-confidence rows exist in Phase 1 by design: the site_control module is missing for every candidate, which caps the data-coverage tier at medium.

### Composite score distribution
| Stat | Value |
|---|---|
| min | 25.61 |
| p50 | 77.58 |
| p90 | 87.25 |
| max | 95.03 |

### Sub-score medians
| Sub-score | Weight | Median | p90 |
|---|---|---|---|
| utility_score | 40% | 91.92 | 100.00 |
| buildability_score | 20% | 78.75 | 87.75 |
| supporting_infra_score | 15% | 56.00 | 76.75 |
| dev_risk_score | 15% | 78.00 | 87.75 |
| site_control_score | 10% (deferred) | - | - |

### Data coverage (per-row, Owen May 11 ask)
| data_coverage_pct | Count | What is missing |
|---|---|---|
| 90 | 67,932 | site_control only (Phase 2) |
| 80 | 15,305 | site_control + 1 other module |
| 70 | 2,743 | site_control + 2 modules |
| 60 | 146 | site_control + 3 modules |
| 50 | 61 | site_control + 4 modules |

I made `data_coverage_pct` vary per row this run (previously hardcoded to 90). The 19,455 rows below 90 are mostly candidates without an ISO-queue anchor in range, or candidates outside any water service district.

---

## Top 10 Shortlist candidates

| candidate_id | state | county | acres | composite | util | build |
|---|---|---|---|---|---|---|
| 973f2d1a.. | TX | Harris | 719 | 94.44 | 100 | 95 |
| e767bb30.. | TX | Caldwell | 1,887 | 94.31 | 100 | 96 |
| 8138a608.. | TX | Atascosa | 4,020 | 94.31 | 100 | 96 |
| 3de93707.. | TX | Cameron | 559 | 94.28 | 100 | 95 |
| a23d02af.. | TX | Kenedy | 1,418 | 94.19 | 100 | 93 |
| 79518297.. | TX | Brazoria | 1,557 | 94.14 | 100 | 96 |
| 9601fd96.. | TX | Johnson | 557 | 94.08 | 100 | 89 |
| 64e71f76.. | TX | Galveston | 627 | 94.07 | 100 | 94 |
| d6b1dc82.. | TX | Johnson | 897 | 93.92 | 100 | 86 |
| bd5693d9.. | TX | Caldwell | 579 | 93.69 | 100 | 94 |

Open `candidates_final.fgb` in QGIS and filter `recommended_action == 'Shortlist'` to see all 569 on the map.

---

## Top reason codes (frequency across all 86,187 rows)

| Code | Sign | Count |
|---|---|---|
| strong_queue_signal | + | 73,888 |
| interconnection_agreement_executed | + | 50,806 |
| drought_tier_high | - | 38,092 |
| no_executed_ia_in_range | - | 32,609 |
| wetland_adjacent_500m | - | 31,761 |
| 345kv_anchor_in_range | + | 27,782 |
| tier1_pipeline_within_5mi | + | 25,714 |
| low_seismic_risk | + | 24,549 |
| elevated_slope_band | - | 23,695 |
| ideal_slope | + | 18,613 |
| allocation_risk_possible | ? | 15,906 |
| class1_rail_within_3mi | + | 11,869 |
| 500kv_anchor_in_range | + | 10,629 |
| building_density_low | + | 7,685 |
| no_queue_activity | - | 5,811 |
| padus_adjacent_500m | - | 4,335 |
| small_candidate | - | 3,918 |
| high_seismic_zone | - | 3,711 |
| floodway_adjacent_500m | - | 2,921 |
| queue_anchor_zone_fallback | ? | 2,738 |

---

## Known limitations and caveats

### 1. Mega-candidate subdivision uses inherited slope
**What I did:** for the 1,238 raw polygons >= 5,000 acres, I subdivided into <= 3,953-acre fragments using a 4 km grid. Acreage preserved 99.96%. Fragments carry `parent_candidate_id`.

**Limitation:** the 44,725 fragments inherit their parent's `slope_max` rather than getting per-fragment 3DEP sampling. This is conservative - some fragments in the flat parts of a mostly-flat-but-one-corner-steep parent get hard-excluded by Rule 13 when they shouldn't. Net impact: probably 5-15% of fragments are over-excluded for slope.

**Phase 1.5 follow-up:** re-sample 3DEP elevation per fragment (~1-2 hours of tile downloads).

### 2. GAP-3/4 exclusion uses public/military/rec subset only
**What I did:** post-filter drops candidates intersecting GAP-3 or GAP-4 PAD-US polygons where Mang_Type is FED/DOD/MIL/LOC or Des_Tp is LREC/MIL/MPUB/NRA. 49,130 polygons across 5 states; dropped 13,647 candidates.

**Limitation:** private GAP-4 (conservation easements, hunting clubs) is kept as adjacency-only since those are still developable in principle.

### 3. Road clip uses TIGER PRISECROADS only
**What I did:** downloaded TIGER 2025 PRISECROADS (34,490 segments: Interstate, US Highway, State Highway, major named), buffered 25 m, and subtracted from candidate geometries. 10,947 candidates had geometry clipped; 61 dropped to < 50 acres.

**Limitation:** minor roads (residential streets, dirt roads) are not subtracted individually. CDL's developed-medium/high pixel class already excludes most residential land upstream.

### 4. NLCD cross-check deferred (PDF Rule 11)
`nlcd_class`, `nlcd_label`, `landcover_confidence_score` are null in every row. CDL is the primary input per PDF Rule 11; NLCD cross-check is a Phase 2 follow-up.

### 5. Phase 2 module placeholders (all null by design)
Reserved schema slots populated by Phase 2 enrichment, 100% null in this output by intent. Covered by Owen May 11 with the `data_coverage_pct` mechanism.

* Parcel/ownership (PDF Rule 21): parcel_count, owner_count, largest_owner_acres, largest_owner_pct_of_candidate, assessed_value_total, assessed_value_per_acre, last_sale_date, last_sale_price, land_use_code, zoning_code, road_frontage_flag, legal_access_flag
* Economic proxy (PDF Rule 22): site_control_score, economic_proxy_score
* Utility feasibility (PDF Rule 23): serving_utility, utility_territory_known, nearest_load_serving_node, utility_service_feasibility_score, utility_review_required
* Communications (PDF Rule 24): communications_route_distance, communications_provider_count, communications_access_score
* Water capacity (PDF Rule 25): water_capacity_known, water_capacity_review_required
* Local jurisdiction (PDF Rule 26): jurisdiction_review_required, local_policy_notes
* Manual imagery QA (PDF Rule 28): manual_imagery_review_status, manual_imagery_review_notes
* Route complexity (PDF Rule 5): route_complexity_score, route_complexity_notes

`parcel_owner_module_status = 'not_built'` on every row, so downstream consumers can see the module status explicitly.

### 6. Shortlist band activation (May 11 ask)
PDF Rule 29 defined Shortlist but the original rule never fired because `utility_review_required` is a Phase 2 field. I activated the band with: `composite >= 90 AND 500 <= acres <= 5,000 AND not slope_review_flag AND not oversized_flag AND confidence >= medium AND num_anchors >= 3`. 569 candidates qualified.

### 7. Wetland adjacency rate is high (structural)
~37% of the final file (31,761 candidates) is within 500 m of an NWI wetland polygon. Wetlands were already physically subtracted from candidates upstream; this is *adjacency*, not overlap.

### 8. IA-executed-nearby rate is high but well-distributed
50,806 candidates (59%) carry `ia_executed_nearby = True`. Among those that fire, 99.9% have the IA-executed anchor within 25 km, median 14 km - so the boolean isn't capturing distant noise. The banded distance decay still weights closer anchors much more heavily.

### 9. Three TX candidates have null county_name
Three edge-case candidates near the Gulf coast (Brownsville area) and the TX/AR/LA tri-state corner have centroids that fall just outside any county polygon in the Census TIGER county boundaries file.

### 10. Mega-polygon scoring is at-the-fragment, not at-the-region
After subdivision, the composite score reflects each < 5,000-acre fragment, not the underlying region. This is the correct behaviour for siting decisions. `parent_candidate_id` is preserved if regional context is wanted.

---

## Reproducibility and audit trail

Every row carries 11 versioning fields populated to the same value per run (PDF Rule 30):

| Field | Value |
|---|---|
| run_id | UUID (one per execution) |
| run_date | 2026-05-16 |
| scoring_model_version | 1.0.0-phase1 |
| exclusion_model_version | 1.0 |
| cdl_year | 2025 |
| padus_version | 4.1 |
| fema_nfhl_date | 2025 |
| nwi_date | 2025 |
| transmission_dataset_version | HIFLD-2024-09 |
| queue_dataset_date | 2026-05 |
| dem_dataset_version | USGS-3DEP-1arcsec |

### Pipeline scripts (all in the GitHub repo)
* `candidate_areas/scripts/Step7d_subdivide_megas.py` - subdivision of >= 5,000-acre polygons
* `candidate_areas/scripts/Step7e_gap34_post_filter.py` - GAP-3/4 public-land post-filter
* `candidate_areas/scripts/Step7f_airport_post_filter.py` - airport runway protection zone exclusion
* `candidate_areas/scripts/Step7g_road_clip_exclusion.py` - TIGER PRISECROADS road clip
* `ingestion_scripts/usgs_slope/inherit_slope_for_fragments.py` - slope inheritance for fragments
* `candidate_areas/enrichment_scripts/Step1A-1J.py` + `Step1_merge.py` - enrichment
* `candidate_areas/scoring_scripts/Step2A-2G.py` - kill gates, sub-scores, composite, actions, reason codes, versioning
* `candidate_areas/scoring_scripts/Step3B_apply_slope_hard_exclusion.py` - final slope hard-exclusion and three-format export
* `candidate_areas/scoring_scripts/Step6_final_deep_verify.py` - 95-check verification (94/95 PASS)

---

## What is coming next

### Phase 2 -- Selective enrichment
Targeted Regrid pulls for the 569 Shortlist candidates first, then expand to Parcel Pull. Populates parcel_count, owner_count, assessed_value_per_acre, last_sale_date, land_use_code, zoning_code. This is where `data_coverage_pct` rises from 90 to 100 and confidence can tier up to *high*.

### Phase 1.5 -- Tighten the corners
Per-fragment 3DEP slope sampling (removes the conservative over-exclusion in limitation #1). NLCD cross-check for non-agricultural CDL classes. Optionally a tighter `ia_executed_within_10km` boolean.

### Phase 3 -- Optional overlays
Active listings (LandWatch / Lands of America / Crexi) as a motivated-seller overlay on the existing shortlist. Owen agreed on May 3 that this is a Phase 3 item, not core.
"""
    OUT_MD.write_text(content, encoding='utf-8')
    print(f'Wrote {OUT_MD}  ({OUT_MD.stat().st_size/1024:.1f} KB)')


if __name__ == '__main__':
    write_markdown_mirror()
    build_pdf()
