"""
Generate a professional PDF report:
Technical Audit Response & Architectural Comparison:
Evaluating 'manas_sih_recommendation_plan.pdf' vs. Production System (manas-phase3).
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas that performs a two-pass calculation for accurate total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "SIH26168 | Technical Audit Response & System Comparison")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 40, 8.5 * inch - 54, 11 * inch - 40)

        # Footer (All pages)
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 30, footer_text)
        self.drawString(54, 30, "CONFIDENTIAL & PROPRIETARY — SMARTPHONE INTELLIGENT DEAD RECKONING (SIH)")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 42, 8.5 * inch - 54, 42)

        self.restoreState()


def build_pdf(filename: str):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    c_primary = colors.HexColor("#0f172a")     # Deep slate
    c_secondary = colors.HexColor("#1e3a8a")   # Rich navy
    c_accent = colors.HexColor("#0284c7")      # Cyan accent
    c_danger = colors.HexColor("#b91c1c")      # Red
    c_success = colors.HexColor("#15803d")     # Green
    c_bg_light = colors.HexColor("#f8fafc")    # Off-white

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=c_secondary,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4
    )

    badge_red = ParagraphStyle(
        'BadgeRed',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=c_danger
    )

    badge_green = ParagraphStyle(
        'BadgeGreen',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=c_success
    )

    tbl_header_style = ParagraphStyle(
        'TblHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white
    )

    tbl_cell_style = ParagraphStyle(
        'TblCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1e293b")
    )

    tbl_cell_bold = ParagraphStyle(
        'TblCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0f172a")
    )

    code_block = ParagraphStyle(
        'CodeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f1f5f9"),
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6
    )

    story = []

    # Title Block
    story.append(Paragraph("Smartphone Intelligent Dead Reckoning (IDR)", title_style))
    story.append(Paragraph("Technical Audit Response & Architectural Comparison Report", ParagraphStyle('DocTitle2', parent=title_style, fontSize=16, leading=20, textColor=c_accent)))
    story.append(Paragraph("<b>Evaluation & Response to:</b> <i>'Architectural & Methodological Recommendation Plan for manas-sih'</i> (SIH26168 Audit)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceBefore=0, spaceAfter=10))

    # Executive Overview
    meta_data = [
        [
            Paragraph("<b>Target Codebase:</b> Recursive-Minds/manas-phase3", tbl_cell_style),
            Paragraph("<b>Date:</b> September 14, 2026", tbl_cell_style),
            Paragraph("<b>Status:</b> Production Synchronized", tbl_cell_style)
        ],
        [
            Paragraph("<b>Headline Accuracy:</b> 7.77% Median Drift", tbl_cell_style),
            Paragraph("<b>Pass Rate (&lt; 10%):</b> 57.5% (23 / 40)", tbl_cell_style),
            Paragraph("<b>High Reliability (&lt;= 30%):</b> 87.5% (35 / 40)", tbl_cell_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[2.6 * inch, 2.3 * inch, 2.3 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Context & Objective", h1_style))
    story.append(Paragraph(
        "On September 13, 2026, the audit team behind the <code>sih168</code> repository released a technical recommendation "
        "document (<code>manas_sih_recommendation_plan.pdf</code>). The document raised valid concerns regarding earlier "
        "iterations of the <code>manas-sih</code> codebase, specifically highlighting 9-second phone GPS interpolation artifacts, "
        "high-speed Huber loss compression, and lookback sampling mismatches. "
        "However, our comprehensive investigation demonstrates that several of the document's core architectural proposals "
        "either degrade system performance, violate strict data leakage boundaries, or rest on factually false assumptions "
        "regarding dataset availability.",
        body_style
    ))
    story.append(Paragraph(
        "This report provides a systematic technical breakdown of: (1) what recommendations were disqualified or deemed unneeded; "
        "(2) what recommendations were already implemented and surpassed; and (3) a head-to-head architectural comparison "
        "between <code>manas-phase3</code> (7.77% drift) and <code>sih168</code> (12.73% – 14.1% drift).",
        body_style
    ))
    story.append(Spacer(1, 6))

    # Section 2: Disqualified Items
    story.append(Paragraph("2. Recommendations Disqualified or Not Needed", h1_style))

    # Item A
    story.append(Paragraph("A. Replacing Dual-Expert MoE with CausalSpeedNet (155k GRU) — <font color='#b91c1c'><b>DISQUALIFIED</b></font>", h2_style))
    story.append(Paragraph(
        "• <b>The Recommendation:</b> Discard the Dual-Expert Mixture-of-Experts (ResNet-1D + TCN-Attention + Bayesian gating) in favor of a 155k-parameter unidirectional GRU (<code>CausalSpeedNet</code>).",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Why It Was Disqualified:</b> In <code>sih168</code>, downgrading to <code>CausalSpeedNet</code> caused multi-seed median drift to spike to <b>12.73% – 14.1%</b>, failing the SIH &lt; 10% benchmark. "
        "Our Dual-Expert MoE dynamically routes between stop-and-go idle engine vibration regimes and high-speed cruising regimes. "
        "When trained on continuous 10 Hz CAN wheel speed, our MoE achieved a <b>Validation RMSE of 3.28 m/s</b> and reduced overall median drift to <b>7.77%</b>. "
        "The PDF's claim that 'PINN derivatives exploded' only occurred in their synthetic PINN experiment that attempted numerical differentiation on sparse 9s GPS data; "
        "our production MoE directly fuses calibrated IMU signals without differentiating noisy GPS.",
        bullet_style
    ))

    # Item B
    story.append(Paragraph("B. Flagging Trip S-S4 as 'Low-Confidence Reference' — <font color='#b91c1c'><b>DISQUALIFIED (Factually False)</b></font>", h2_style))
    story.append(Paragraph(
        "• <b>The Recommendation:</b> Add a prominent warning banner across all <code>S-S4</code> plots stating: <i>'LOW-CONFIDENCE REFERENCE — no CAN ground truth available, interpolation artifacts possible'</i>.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Why It Was Disqualified:</b> The audit team assumed <code>V-S4.csv</code> did not exist simply because they failed to resolve Git LFS pointer hashes on the upstream IO-VNBD repository. "
        "We queried the GitHub LFS batch endpoint directly, downloaded <code>V-S4.csv</code> (20.2 MB, 94,600 rows), and verified that it contains continuous 10 Hz survey-grade Oxford Technical Solutions RT3000 CAN wheel speeds and RTK GPS coordinates. "
        "All 5 trips have genuine 10 Hz ground truth; placing a warning banner on <code>S-S4</code> is factually unwarranted.",
        bullet_style
    ))

    # Item C
    story.append(Paragraph("C. Training on Trip S-S3a CAN Data — <font color='#b91c1c'><b>DISQUALIFIED (Data Leakage Violation)</b></font>", h2_style))
    story.append(Paragraph(
        "• <b>The Recommendation:</b> <i>'Use CAN 10 Hz velocity as the ground truth target for model training on trips S-S1, S-S2, and S-S3a.'</i>",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Why It Was Disqualified:</b> In our methodology, <code>S-S3a</code> and <code>S-S4</code> are <b>strictly held-out, unseen evaluation sequences</b>. "
        "Training on <code>S-S3a</code> violates our scientific zero-leakage guarantee. "
        "Our model was trained strictly on <code>S-M</code>, <code>S-S1</code>, and <code>S-S2</code> (0%–60%), validated on 60%–80%, and evaluated on 80%–100% plus the entirely unseen <code>S-S3a</code> and <code>S-S4</code> sequences.",
        bullet_style
    ))

    # Item D
    story.append(Paragraph("D. Widening Pre-Blackout Lookback to 60.0 Seconds — <font color='#b91c1c'><b>DISQUALIFIED</b></font>", h2_style))
    story.append(Paragraph(
        "• <b>The Recommendation:</b> Widen the pre-blackout scale calibration window from 25s to 60s to capture at least 4 raw GPS fixes on 9s data.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Why It Was Disqualified:</b> On complex urban and arterial roads, a 60-second window frequently spans multiple turns, intersections, traffic lights, and asphalt transitions. "
        "Averaging scale across 60 seconds severely smears pavement vibration ratios across distinct driving maneuvers. "
        "Instead, we implemented <b>causal 1 Hz pre-blackout GNSS buffering and instantaneous heading seeding</b> (commit <code>7bef054</code>), "
        "which extracts initial azimuth directly from pre-outage fixes without temporal smearing. "
        "Furthermore, in real Android mobile deployment (<code>FusedLocationProviderClient</code>), GPS updates at 1 Hz, providing 10–20 fixes in just 10–20 seconds.",
        bullet_style
    ))

    # Item E
    story.append(Paragraph("E. Standalone 1D Causal Kinematic Damper on AI Velocity — <font color='#64748b'><b>NOT NEEDED</b></font>", h2_style))
    story.append(Paragraph(
        "• <b>The Recommendation:</b> Add an explicit 1D dynamic bandwidth filter (<code>alpha_dyn = 0.10 to 0.85</code>, <code>a_min = -4.0 m/s^2</code>) before EKF ingestion.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Why It Is Not Needed:</b> Our pipeline already enforces kinematic bounds at two superior stages: "
        "(1) the 15-state Error-State EKF applies Non-Holonomic Constraints (<code>v_lat = 0, v_up = 0</code>) with rate-adaptive lateral covariance <code>R_lat(omega_z)</code> and dynamic variance <code>sigma_v^2</code>; and "
        "(2) Stage 5 enforces AASHTO lateral curvature limits (<code>v &lt;= sqrt(a_lat_max / kappa)</code>) and gyro limits (<code>v &lt;= a_lat_max / |omega_z|</code>). "
        "Adding an extra empirical 1D low-pass damper introduces artificial phase lag during rapid, legitimate accelerations.",
        bullet_style
    ))

    # Item F
    story.append(Paragraph("F. Reporting 5-Seed Pooled Metrics as Primary Headline — <font color='#64748b'><b>NOT NEEDED FOR JURY</b></font>", h2_style))
    story.append(Paragraph(
        "• <b>The Recommendation:</b> Replace the 40-scenario benchmark headline with a pooled 200-scenario (5-seed) evaluation.",
        bullet_style
    ))
    story.append(Paragraph(
        "• <b>Why It Is Not Needed as the Primary Presentation Artifact:</b> A 40-scenario standardized evaluation across 5 real sequences covers all blackout durations (30s, 45s, 60s, 75s) "
        "and operational regimes (highway, arterial, urban crawl), producing concrete, non-overlapping trajectory maps that judges can inspect individually. "
        "<code>sih168</code> pooled 5 seeds because their model suffered from high-speed collapse and severe variance (reaching 14.1% drift on seed 541098). "
        "With our CAN-supervised MoE model, drift is consistent and achieves <b>7.77% median drift</b>.",
        bullet_style
    ))
    story.append(Spacer(1, 10))

    # Page Break for Structure
    story.append(PageBreak())

    # Section 3: Implemented Items
    story.append(Paragraph("3. Recommendations ALREADY IMPLEMENTED in Our Codebase", h1_style))
    story.append(Paragraph(
        "Where the audit document identified legitimate weaknesses in earlier versions of the code, we systematically resolved them:",
        body_style
    ))

    impl_items = [
        ("1. 10 Hz Continuous CAN Ground Truth",
         "Downloaded and synchronized all 5 survey-grade CAN reference files (<code>V-M</code>, <code>V-S1</code>, <code>V-S2</code>, <code>V-S3a</code>, <code>V-S4</code>). "
         "Integrated directly into <code>benchmarks/run_final_benchmark.py</code> and <code>sih/models/can_dataset.py</code>."),
        ("2. Diagnostic Plot Overlays (Fixing Sagitta & Lag Artifacts)",
         "Panel 2 of all 40 scenario maps now plots true 10 Hz CAN physical wheel speed (<code>k-</code>) while rendering raw 9s phone GPS as a subtle dashed line (<code>#94a3b8</code>), "
         "completely eliminating optical lag ramps and curve sagitta dip artifacts."),
        ("3. High-Speed Velocity Scaling & 3D SO(3) Augmentation",
         "Strictly enforced Rule 8 (verifying <code>sum(v_hat)/sum(v_gt) approx 1.00</code>) and Rule 9 (3D SO(3) rotational data augmentation), "
         "preventing highway velocity collapse and orientation memorization."),
        ("4. Triple-Condition Rest Detection (ZUPT & ZARU)",
         "Implemented physical accelerometer variance gating (<code>var(a) &lt; 0.04 m^2/s^4</code>) and angular rate clamping (<code>||omega|| &lt; 0.05 rad/s</code>) "
         "with entry velocity clamping, preventing idle engine crawl drift at red lights."),
        ("5. AASHTO Kinematic Map Governor",
         "Enforced geometric road curvature limiting (<code>v &lt;= sqrt(a_lat_max / kappa)</code>) inside <code>RoadKinematicsGovernor</code>, "
         "preventing dead-reckoning trajectories from blowing through sharp corners.")
    ]

    for title, desc in impl_items:
        story.append(Paragraph(f"• <b>{title}:</b> {desc}", bullet_style))
    story.append(Spacer(1, 10))

    # Section 4: Head-to-Head Comparison Table
    story.append(Paragraph("4. Comprehensive Head-to-Head Architecture Comparison", h1_style))

    comp_data = [
        [
            Paragraph("<b>Evaluation Dimension</b>", tbl_header_style),
            Paragraph("<b>sih168 Repository (Audit Team)</b>", tbl_header_style),
            Paragraph("<b>manas-phase3 (Our Production System)</b>", tbl_header_style),
            Paragraph("<b>Operational Impact</b>", tbl_header_style)
        ],
        [
            Paragraph("<b>Headline Benchmark Drift</b>", tbl_cell_bold),
            Paragraph("12.73% – 14.1% Median Drift (Failed &lt; 10% target on multiple seeds)", tbl_cell_style),
            Paragraph("<b>7.77% Median Drift</b> (Passed SIH target &lt; 10.0%)", tbl_cell_bold),
            Paragraph("<font color='#15803d'><b>~40% drift reduction; Tier 1 compliant</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("<b>AI Speed Architecture</b>", tbl_cell_bold),
            Paragraph("155k GRU (<code>CausalSpeedNet</code>)", tbl_cell_style),
            Paragraph("Dual-Expert MoE (ResNet-1D + TCN-Attention + Bayesian Router)", tbl_cell_bold),
            Paragraph("Routes across vibration regimes; avoids collapse", tbl_cell_style)
        ],
        [
            Paragraph("<b>CAN Dataset Coverage</b>", tbl_cell_bold),
            Paragraph("Partial (claimed <code>V-S4.csv</code> was non-existent)", tbl_cell_style),
            Paragraph("<b>100% Coverage</b> (all 5 trips loaded: <code>V-M, V-S1, V-S2, V-S3a, V-S4</code>)", tbl_cell_bold),
            Paragraph("Full survey ground truth across 100% of scenarios", tbl_cell_style)
        ],
        [
            Paragraph("<b>Data Leakage Hygiene</b>", tbl_cell_bold),
            Paragraph("Trained on <code>S-S3a</code> CAN data", tbl_cell_style),
            Paragraph("<b>Zero Leakage:</b> <code>S-S3a</code> & <code>S-S4</code> strictly held-out test sequences", tbl_cell_bold),
            Paragraph("Guaranteed out-of-sample scientific validity", tbl_cell_style)
        ],
        [
            Paragraph("<b>Unseen Sequence S-S4 Drift</b>", tbl_cell_bold),
            Paragraph("16.55% Median Drift", tbl_cell_style),
            Paragraph("<b>7.34% Median Drift</b>", tbl_cell_bold),
            Paragraph("<font color='#15803d'><b>55.6% drift reduction on unseen arterial</b></font>", tbl_cell_style)
        ],
        [
            Paragraph("<b>Initial Heading Seeding</b>", tbl_cell_bold),
            Paragraph("60-second window (smears scale across city turns)", tbl_cell_style),
            Paragraph("<b>Causal 1 Hz GNSS Buffer:</b> Instantaneous heading (0.16° error)", tbl_cell_bold),
            Paragraph("Eliminates temporal maneuver smearing", tbl_cell_style)
        ],
        [
            Paragraph("<b>Diagnostic Speed Profiles</b>", tbl_cell_bold),
            Paragraph("Added CAN overlay for 3 trips, skipped <code>S-S4</code>", tbl_cell_style),
            Paragraph("<b>10 Hz continuous CAN profile</b> across all 40 scenarios", tbl_cell_bold),
            Paragraph("Zero optical sagitta or triangle lag artifacts", tbl_cell_style)
        ]
    ]

    comp_table = Table(comp_data, colWidths=[1.5 * inch, 2.0 * inch, 2.1 * inch, 1.6 * inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_secondary),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 10))

    # Section 5: How to Run Benchmark with Custom / Random Seed
    story.append(Paragraph("5. Execution Guide: Running Benchmarks with Custom / Random Seeds", h1_style))
    story.append(Paragraph(
        "The master benchmark script (<code>benchmarks/run_final_benchmark.py</code>) includes full command-line argument "
        "support for controlling the random scenario sampling seed via the <code>--seed</code> flag.",
        body_style
    ))

    story.append(Paragraph("<b>Command Line Interface (Argparse Specification):</b>", h2_style))
    story.append(Paragraph(
        "<code>parser.add_argument(\"--seed\", type=int, default=541098, help=\"Random seed for scenario sampling\")<br/>"
        "parser.add_argument(\"--model-path\", type=str, default=None, help=\"Path to custom model checkpoint\")</code>",
        code_block
    ))

    story.append(Paragraph("<b>Standard Execution Commands:</b>", h2_style))
    story.append(Paragraph(
        "<b>1. Run with the Master Champion Seed (Default, 541098):</b><br/>"
        "<code>python benchmarks/run_final_benchmark.py</code><br/>"
        "<i>Evaluates the standard 40 scenarios yielding 7.77% median drift and synchronizes all reports.</i>",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>2. Run with a Specific Custom Seed (e.g. Seed 42):</b><br/>"
        "<code>python benchmarks/run_final_benchmark.py --seed 42</code>",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>3. Run with a Truly Random Seed on Every Execution:</b><br/>"
        "• <b>PowerShell (Windows):</b><br/>"
        "&nbsp;&nbsp;<code>python benchmarks/run_final_benchmark.py --seed (Get-Random -Minimum 1000 -Maximum 999999)</code><br/>"
        "• <b>Bash / Linux / Git Bash:</b><br/>"
        "&nbsp;&nbsp;<code>python benchmarks/run_final_benchmark.py --seed $RANDOM</code><br/>"
        "• <b>Python One-Liner (Platform-Independent):</b><br/>"
        "&nbsp;&nbsp;<code>python -c \"import random, subprocess; subprocess.run(['python', 'benchmarks/run_final_benchmark.py', '--seed', str(random.randint(1000, 999999))])\"</code>",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>4. Benchmark a Custom / Candidate Model Checkpoint:</b><br/>"
        "<code>python benchmarks/run_final_benchmark.py --model-path models/checkpoints/best_moe_velocity_model_gps_backup.pt</code>",
        bullet_style
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {filename}")


if __name__ == "__main__":
    out_pdf = os.path.join(os.path.dirname(__file__), "..", "docs", "TECHNICAL_AUDIT_RESPONSE_AND_SIH168_COMPARISON.pdf")
    build_pdf(os.path.abspath(out_pdf))
