"""
Generate the final project report as a properly IEEE-formatted Word document.
Run from the project root: python code/generate_report_docx.py
"""

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parent.parent
FIG  = ROOT / "figures"
OUT  = ROOT / "docs" / "Fairness_Stress_Testing_Report_IEEE.docx"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BODY_FONT  = "Times New Roman"
TITLE_SIZE = 20
AUTH_SIZE  = 11
ABST_SIZE  = 9
BODY_SIZE  = 10
CAP_SIZE   = 8
REF_SIZE   = 8

# ---------------------------------------------------------------------------
# Low-level XML helpers
# ---------------------------------------------------------------------------

def _el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), str(v))
    return e


def _cols_el(num, space_twips=720):
    c = _el("w:cols", **{"w:num": str(num), "w:space": str(space_twips),
                          "w:equalWidth": "1"})
    return c


def _set_section_cols(section, num, space_twips=720):
    """Set number of columns on an existing section object."""
    sectPr = section._sectPr
    for old in sectPr.findall(qn("w:cols")):
        sectPr.remove(old)
    sectPr.append(_cols_el(num, space_twips))


def _page_setup(section, w_in=8.5, h_in=11,
                top=0.75, bottom=1.0, left=0.75, right=0.75):
    sectPr = section._sectPr
    for tag in ("w:pgSz", "w:pgMar"):
        for old in sectPr.findall(qn(tag)):
            sectPr.remove(old)
    pgSz = _el("w:pgSz", **{"w:w": str(int(w_in * 1440)),
                              "w:h": str(int(h_in * 1440))})
    pgMar = _el("w:pgMar",
                 **{"w:top":    str(int(top    * 1440)),
                    "w:bottom": str(int(bottom * 1440)),
                    "w:left":   str(int(left   * 1440)),
                    "w:right":  str(int(right  * 1440)),
                    "w:header": "720", "w:footer": "720", "w:gutter": "0"})
    sectPr.append(pgSz)
    sectPr.append(pgMar)


def _insert_continuous_break_SINGLE(doc):
    """
    Insert a continuous section break whose sectPr declares SINGLE-column
    layout.  Everything before this paragraph belongs to that single-column
    section; everything after belongs to the *next* section (which we set
    to two-column separately on doc.sections[-1]).
    """
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(0)
    pPr = p._p.get_or_add_pPr()
    sectPr = OxmlElement("w:sectPr")
    # type = continuous
    typ = _el("w:type", **{"w:val": "continuous"})
    sectPr.append(typ)
    # single column
    sectPr.append(_cols_el(1, 0))
    # page size + margins (match the rest of the doc)
    pgSz = _el("w:pgSz", **{"w:w": "12240", "w:h": "15840"})
    pgMar = _el("w:pgMar",
                 **{"w:top": "1080", "w:bottom": "1440",
                    "w:left": "1080", "w:right": "1080",
                    "w:header": "720", "w:footer": "720", "w:gutter": "0"})
    sectPr.append(pgSz)
    sectPr.append(pgMar)
    pPr.append(sectPr)
    return p


# ---------------------------------------------------------------------------
# IEEE table borders (horizontal rules only: thick top/bottom, thin mid)
# ---------------------------------------------------------------------------

def _ieee_table_borders(table):
    """Apply IEEE-style borders: thick top, thin after header, thick bottom."""
    tbl  = table._tbl
    tblPr = tbl.find(qn("w:tblPr"))
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr")
        tbl.insert(0, tblPr)

    # Remove existing border element
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)

    # Table-level: only outside top and bottom, no vertical lines
    tblBorders = OxmlElement("w:tblBorders")
    for side, sz in [("top", "12"), ("bottom", "12"),
                     ("left", "0"), ("right", "0"),
                     ("insideH", "0"), ("insideV", "0")]:
        b = OxmlElement(f"w:{side}")
        if sz == "0":
            b.set(qn("w:val"), "none")
        else:
            b.set(qn("w:val"), "single")
            b.set(qn("w:sz"), sz)
            b.set(qn("w:space"), "0")
            b.set(qn("w:color"), "000000")
        tblBorders.append(b)
    tblPr.append(tblBorders)

    # Header row: add bottom border (thin rule after header)
    hrow = table.rows[0]
    for cell in hrow.cells:
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        for old in tcPr.findall(qn("w:tcBorders")):
            tcPr.remove(old)
        tcBorders = OxmlElement("w:tcBorders")
        bot = OxmlElement("w:bottom")
        bot.set(qn("w:val"), "single")
        bot.set(qn("w:sz"), "6")
        bot.set(qn("w:space"), "0")
        bot.set(qn("w:color"), "000000")
        tcBorders.append(bot)
        tcPr.append(tcBorders)

    # Last row: add bottom border (thick)
    lrow = table.rows[-1]
    for cell in lrow.cells:
        tc   = cell._tc
        tcPr = tc.get_or_add_tcPr()
        for old in tcPr.findall(qn("w:tcBorders")):
            tcPr.remove(old)
        tcBorders = OxmlElement("w:tcBorders")
        bot = OxmlElement("w:bottom")
        bot.set(qn("w:val"), "single")
        bot.set(qn("w:sz"), "12")
        bot.set(qn("w:space"), "0")
        bot.set(qn("w:color"), "000000")
        tcBorders.append(bot)
        tcPr.append(tcBorders)


# ---------------------------------------------------------------------------
# Paragraph / run helpers
# ---------------------------------------------------------------------------

def _fmt(para, align=WD_ALIGN_PARAGRAPH.JUSTIFY, sb=0, sa=3,
         first=None, left=None):
    f = para.paragraph_format
    f.alignment    = align
    f.space_before = Pt(sb)
    f.space_after  = Pt(sa)
    if first is not None:
        f.first_line_indent = Inches(first)
    if left is not None:
        f.left_indent = Inches(left)
    return para


def _run(para, text, size=BODY_SIZE, bold=False, italic=False,
         font=BODY_FONT, color=None, smallcaps=False):
    r = para.add_run(text)
    r.font.name    = font
    r.font.size    = Pt(size)
    r.font.bold    = bold
    r.font.italic  = italic
    if color:
        r.font.color.rgb = RGBColor(*color)
    if smallcaps:
        r.font.small_caps = True
    return r


def _new_para(doc, align=WD_ALIGN_PARAGRAPH.JUSTIFY, sb=0, sa=3,
              first=None, left=None):
    p = doc.add_paragraph()
    return _fmt(p, align=align, sb=sb, sa=sa, first=first, left=left)


# ---------------------------------------------------------------------------
# Semantic helpers
# ---------------------------------------------------------------------------

def body(doc, text, sa=2):
    p = _new_para(doc, sa=sa, first=0.15)
    _run(p, text)
    return p


def body0(doc, text, sa=2):
    """Body paragraph, no first-line indent."""
    p = _new_para(doc, sa=sa, first=0.0)
    _run(p, text)
    return p


def sec_heading(doc, num, title):
    """IEEE section heading: 'I.  INTRODUCTION', centered, bold."""
    p = _new_para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, sb=10, sa=3, first=0.0)
    label = f"{num}.  " if num else ""
    _run(p, label + title.upper(), bold=True, size=10)
    return p


def sub_heading(doc, letter, title):
    """IEEE subsection: 'A.  Title of Subsection', italic, left."""
    p = _new_para(doc, align=WD_ALIGN_PARAGRAPH.LEFT, sb=6, sa=2, first=0.0)
    _run(p, f"{letter}.  ", italic=True, bold=True, size=10)
    _run(p, title, italic=True, size=10)
    return p


def bullet(doc, label, text):
    p = _new_para(doc, sb=0, sa=2, first=0.0, left=0.2)
    _run(p, "•  ", bold=True, size=10)
    if label:
        _run(p, label + ": ", bold=True, size=10)
    _run(p, text, size=10)
    return p


def fig_caption(doc, text):
    p = _new_para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, sb=2, sa=8, first=0.0)
    _run(p, text, size=CAP_SIZE, italic=True)
    return p


def table_caption(doc, roman, text):
    """IEEE table caption ABOVE the table, e.g. TABLE II"""
    p = _new_para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, sb=8, sa=1, first=0.0)
    _run(p, f"TABLE {roman}", bold=True, size=CAP_SIZE)
    p2 = _new_para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, sb=0, sa=2, first=0.0)
    _run(p2, text.upper(), bold=False, size=CAP_SIZE)
    return p


def add_figure(doc, filename, width=3.1, cap=""):
    path = FIG / filename
    if not path.exists():
        p = _new_para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, first=0.0)
        _run(p, f"[Figure not found: {filename}]", italic=True, size=8)
        if cap:
            fig_caption(doc, cap)
        return
    p = _new_para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, sb=6, sa=1, first=0.0)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    if cap:
        fig_caption(doc, cap)


def make_table(doc, headers, rows, col_widths, roman, cap_text):
    """Build an IEEE-style table (horizontal rules only, no shading)."""
    table_caption(doc, roman, cap_text)
    n = len(headers)
    tbl = doc.add_table(rows=1 + len(rows), cols=n)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    # header row
    for i, h in enumerate(headers):
        cell = tbl.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name  = BODY_FONT
        r.font.size  = Pt(CAP_SIZE)
        r.font.bold  = True

    # data rows
    for ri, row_data in enumerate(rows):
        for ci, val in enumerate(row_data):
            cell = tbl.rows[ri + 1].cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            r.font.name = BODY_FONT
            r.font.size = Pt(CAP_SIZE)

    # set column widths
    for row in tbl.rows:
        for ci, cell in enumerate(row.cells):
            if ci < len(col_widths):
                cell.width = Inches(col_widths[ci])

    _ieee_table_borders(tbl)
    _new_para(doc, sb=0, sa=4)   # small gap after table
    return tbl


# ---------------------------------------------------------------------------
# Main document builder
# ---------------------------------------------------------------------------

def build():
    doc = Document()

    # ── Page geometry for the FINAL section (two-column body) ───────────────
    final_sec = doc.sections[0]
    _page_setup(final_sec, top=0.75, bottom=1.0, left=0.75, right=0.75)
    # We'll set this to 2-column after inserting the section break

    # ====================================================================
    # TITLE BLOCK  (single-column — defined by the inline sectPr we'll add)
    # ====================================================================

    # Title
    tp = _new_para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, sb=0, sa=4, first=0.0)
    _run(tp, "Fairness Stress Testing for Tabular Machine Learning\n"
             "in High-Stakes Decision Systems",
         bold=True, size=TITLE_SIZE)

    # Authors  (3-column-like block using a 1-row table for layout)
    auth_data = [
        ("Devansh Verma",   "Project Lead",              "dverma26@asu.edu"),
        ("Naman Dewan",     "Data Engineering Lead",     "ndewan@asu.edu"),
        ("Manan Batra",     "Model Development Lead",    "mbatra@asu.edu"),
        ("Khushi Thakare",  "Debiasing Strategy Lead",   "kthakare@asu.edu"),
        ("Anushka Mote",    "Documentation & Viz Lead",  "amote@asu.edu"),
    ]
    # Emit authors as centered paragraphs (simpler, still correct)
    names_p = _new_para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, sb=4, sa=1, first=0.0)
    for i, (name, role, email) in enumerate(auth_data):
        _run(names_p, name, bold=True, size=AUTH_SIZE)
        if i < len(auth_data) - 1:
            _run(names_p, ",  ", size=AUTH_SIZE)

    aff_p = _new_para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, sb=0, sa=1, first=0.0)
    _run(aff_p, "Arizona State University, Tempe, AZ 85281", italic=True, size=9)

    email_p = _new_para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, sb=0, sa=6, first=0.0)
    emails = ",  ".join(e for _, _, e in auth_data)
    _run(email_p, emails, italic=True, size=8, color=(80, 80, 80))

    # Course line
    cp = _new_para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, sb=0, sa=10, first=0.0)
    _run(cp, "CSE 575 Statistical Machine Learning  |  Spring 2026  |  Project P17",
         italic=True, size=9, color=(100, 100, 100))

    # ── Abstract ─────────────────────────────────────────────────────────────
    # IEEE: abstract indented 0.5 in on each side, 9 pt
    abst_p = _new_para(doc, sb=0, sa=2, first=0.0, left=0.5)
    abst_p.paragraph_format.right_indent = Inches(0.5)
    _run(abst_p, "Abstract—", bold=True, italic=True, size=ABST_SIZE)
    _run(abst_p,
         "Fairness evaluations in machine learning are routinely reported as a single "
         "metric at a single operating point, obscuring the variance practitioners "
         "encounter under real deployment conditions. This work proposes a fairness "
         "stress-testing framework that evaluates three standard debiasing methods—"
         "Reweighing, Exponentiated Gradient, and Threshold Optimization—across "
         "five random seeds, nine classification thresholds, and intersectional subgroups "
         "on three benchmark datasets: Adult Income, German Credit, and COMPAS. We "
         "evaluate Logistic Regression, Random Forest, and XGBoost. Our central finding "
         "is that fairness interventions are real but fragile: Exponentiated Gradient "
         "reduces Logistic Regression demographic parity difference by 89% "
         "(0.174±0.007 → 0.019±0.008) with only 1.9 pp accuracy cost, "
         "yet the same method on Random Forest costs 6.0 pp for a 64% gain. Threshold "
         "choice alone produces a 63% DP-difference swing without any debiasing. "
         "Intersectional analysis reveals that an aggregate DP difference of 0.183 "
         "masks a 4.2× selection-rate ratio between white males (26.6%) and "
         "non-white females (6.4%). We conclude that fairness must be reported as "
         "distributions over seeds, threshold curves, and intersectional subgroups, "
         "not as single-point benchmarks.",
         italic=True, size=ABST_SIZE)

    # Keywords
    kp = _new_para(doc, sb=2, sa=8, first=0.0, left=0.5)
    kp.paragraph_format.right_indent = Inches(0.5)
    _run(kp, "Index Terms—", bold=True, italic=True, size=ABST_SIZE)
    _run(kp,
         "algorithmic fairness, stress testing, demographic parity, threshold "
         "sensitivity, intersectional fairness, reweighing, exponentiated gradient, "
         "threshold optimization, tabular machine learning.",
         italic=True, size=ABST_SIZE)

    # ── Insert continuous section break (marks end of single-column header) ─
    _insert_continuous_break_SINGLE(doc)

    # ── Now configure the FINAL section of the document to be two-column ────
    # After the section break paragraph, doc.sections[-1] is the body section
    _set_section_cols(doc.sections[-1], num=2, space_twips=720)
    _page_setup(doc.sections[-1], top=0.75, bottom=1.0, left=0.75, right=0.75)

    # ====================================================================
    # BODY  (two-column)
    # ====================================================================

    # ── I. INTRODUCTION ──────────────────────────────────────────────────────
    sec_heading(doc, "I", "Introduction")
    body(doc,
         "Machine learning systems are increasingly deployed in high-stakes "
         "domains—credit underwriting, hiring pipelines, and criminal justice "
         "risk scoring. In each setting a single automated prediction can determine "
         "whether a person receives a loan, advances past a screening round, or is "
         "recommended for pretrial release. The growing use of algorithmic "
         "decision-making has elevated fairness from an ethical aspiration to an "
         "engineering requirement.")
    body(doc,
         "The standard response is to apply a fairness intervention and report "
         "the resulting metric improvement: a researcher trains a model, applies "
         "Reweighing or Threshold Optimization, and reports that demographic parity "
         "difference falls from 0.19 to 0.09. The result appears in a benchmark "
         "table; the method is deemed effective.")
    body(doc,
         "The problem is not that the result is wrong—it is that the result "
         "is a point estimate for one seed, one split, and one threshold. Real "
         "deployed systems face the conditions single-point evaluation ignores: "
         "training distributions shift over time, decision thresholds are tuned "
         "to business constraints, and populations contain intersectional subgroups "
         "that aggregate metrics erase.")
    body(doc,
         "This paper proposes a systematic stress-testing framework designed to "
         "expose this fragility. Rather than asking whether a mitigation method "
         "improves a fairness metric, we ask whether the improvement is stable. "
         "We evaluate every (model × method × dataset) combination across "
         "five random seeds, nine classification thresholds (0.30 to 0.70), and "
         "intersectional subgroups on three benchmark datasets.")

    # ── II. RELATED WORK ─────────────────────────────────────────────────────
    sec_heading(doc, "II", "Related Work")

    sub_heading(doc, "A", "Fairness Criteria")
    body(doc,
         "The three primary fairness criteria for tabular classification are "
         "demographic parity, equal opportunity, and equalized odds [1]. These are "
         "mutually incompatible in the general case [4]. Barocas et al. [5] provide "
         "the canonical treatment of these impossibility results.")

    sub_heading(doc, "B", "Debiasing Methods")
    body(doc,
         "Kamiran and Calders [6] introduced Reweighing, a pre-processing method "
         "that assigns sample weights to equalize group representation. "
         "Agarwal et al. [7] proposed the Exponentiated Gradient (in-processing), "
         "which converts a fairness-constrained problem into iteratively weighted "
         "standard classification. Hardt et al. [4] established the equalized odds "
         "post-processor; Fairlearn's ThresholdOptimizer implements it via "
         "group-specific threshold selection.")

    sub_heading(doc, "C", "Single-Point Evaluation and Robustness")
    body(doc,
         "Published fairness benchmarks almost universally report a single seed "
         "and threshold. Uddin et al. [2] and Chen [3] identify this as a critical "
         "gap. The triangular trade-off survey [8] shows that accuracy, robustness, "
         "and fairness interact in ways single-point evaluation cannot capture. "
         "Roberts-Licklider and Trafalis [9] confirm distributional reporting is "
         "rarely practiced.")

    sub_heading(doc, "D", "Intersectional Fairness")
    body(doc,
         "Crenshaw [10] established intersectionality: individuals belong to multiple "
         "protected groups simultaneously, and aggregate metrics mask harm to "
         "multiply-marginalized subgroups. Mehrabi et al. [1] confirm intersectional "
         "analysis is rare in benchmark evaluations despite its ease of implementation.")

    sub_heading(doc, "E", "Dataset Validity")
    body(doc,
         "Adult Income (1994 U.S. Census) reflects historical discrimination. German "
         "Credit (n = 1,000) produces wide metric confidence intervals by "
         "design. COMPAS has been extensively critiqued by Bao et al. [11] "
         "(“It’s COMPASlicated”); we report its results with the "
         "explicit caveat that labels should not be treated as ground truth.")

    # ── III. METHODOLOGY ─────────────────────────────────────────────────────
    sec_heading(doc, "III", "Methodology")

    sub_heading(doc, "A", "Overall Pipeline")
    body(doc,
         "The pipeline has four phases. (1) Data loading: each dataset is returned "
         "as (X, y, sensitive_attrs); sensitive attributes are carried as "
         "metadata and never included as model features. (2) Training: each "
         "(model × method × seed) combination is trained "
         "on a stratified 70/30 split; metrics are recorded to JSON. "
         "(3) Stress testing: multi-seed loop, threshold sweep, and intersectional "
         "breakdown. (4) Visualization: all figures generated from saved CSVs.")

    sub_heading(doc, "B", "Datasets")
    body0(doc, "Table I summarises the three benchmark datasets.")
    make_table(doc,
        headers=["Dataset", "Train", "Test", "Feats", "Sensitive", "Pos."],
        rows=[
            ["Adult Income", "~34,000", "~9,769", "13", "sex, race", "24%"],
            ["German Credit", "700",    "300",    "20", "sex, age",  "70%"],
            ["COMPAS",       "~4,300",  "~1,850", "10", "race",      "46%"],
        ],
        col_widths=[1.0, 0.55, 0.55, 0.45, 0.80, 0.40],
        roman="I", cap_text="Dataset Statistics")

    sub_heading(doc, "C", "Model Suite")
    body(doc,
         "We evaluate Logistic Regression (LR, L2 regularization C = 1.0), "
         "Random Forest (RF, 100 trees), and XGBoost. All expose a common "
         "build_model(name, seed) interface; the seed argument controls all "
         "internal random state.")

    sub_heading(doc, "D", "Fairness Metrics")
    body(doc,
         "Demographic Parity Difference (DP diff) = P(Ŷ=1|A=1) − "
         "P(Ŷ=1|A=0). Equal Opportunity Difference (EO diff) = TPR(A=1) "
         "− TPR(A=0). Equalized Odds Difference = max(|TPR diff|, |FPR diff|). "
         "We also report accuracy, AUC, per-group selection rate, TPR, FPR, and ECE.")

    sub_heading(doc, "E", "Debiasing Methods")
    body(doc,
         "Reweighing [6] upweights underrepresented (group, label) cells in the "
         "training loss. Exponentiated Gradient [7] iteratively reweights the base "
         "classifier subject to a DemographicParity constraint (tolerance 0.01). "
         "ThresholdOptimizer [4] selects group-specific decision thresholds post-hoc.")

    sub_heading(doc, "F", "Stress-Testing Framework")
    body0(doc, "The stress-testing layer has three components:")
    bullet(doc, "Seed sweep",
           "every configuration evaluated across seeds {0,1,2,3,4}; "
           "mean ± std reported for every metric.")
    bullet(doc, "Threshold sweep",
           "threshold swept 0.30–0.70 in steps of 0.05; accuracy "
           "and all fairness metrics recorded at each point.")
    bullet(doc, "Intersectional breakdown",
           "(sex×race) cells on Adult, (sex×age) on German; "
           "cells with n < 100 flagged, n < 30 dropped.")

    # ── IV. EVALUATION AND ANALYSIS ──────────────────────────────────────────
    sec_heading(doc, "IV", "Evaluation and Analysis")

    sub_heading(doc, "A", "Baseline Results")
    body(doc,
         "Table II reports baselines on Adult Income. The LR result "
         "(accuracy 0.854, DP diff 0.174) aligns with published benchmarks [5] "
         "(84–86%, DP diff 0.17–0.19), confirming correct pipeline behaviour. "
         "XGBoost achieves 1.7 pp higher accuracy than LR. All three models "
         "produce similar DP differences (0.174–0.185), confirming that model "
         "family does not substantially change baseline discrimination.")

    make_table(doc,
        headers=["Model", "Acc.", "±", "DP Diff", "±", "AUC"],
        rows=[
            ["LR",      "0.8541", "0.0019", "0.174", "0.007", "0.906"],
            ["RF",      "0.8556", "0.0014", "0.185", "0.006", "0.906"],
            ["XGBoost", "0.8713", "0.0011", "0.179", "0.006", "0.927"],
        ],
        col_widths=[0.65, 0.60, 0.40, 0.65, 0.40, 0.55],
        roman="II", cap_text="Adult Income Baselines (mean ± std, seeds 0–4)")

    add_figure(doc, "baseline_overview.png", width=3.1,
               cap="Fig. 1. Baseline accuracy and DP difference across models and "
                   "datasets. Error bars show std across 5 seeds.")

    body(doc,
         "German Credit (Table III) exhibits extreme small-sample fragility. "
         "LR accuracy swings 0.690–0.800 across seeds; DP difference spans "
         "−0.036 to +0.182—the sign of the fairness gap flips. This is "
         "the correct empirical result for n = 1,000 and is precisely the "
         "finding that should appear in benchmarks but almost never does.")

    make_table(doc,
        headers=["Model", "Acc.", "±", "DP Diff", "±", "AUC"],
        rows=[
            ["LR",      "0.756", "0.038", "0.060", "0.085", "0.784"],
            ["RF",      "0.768", "0.014", "0.047", "0.073", "0.795"],
            ["XGBoost", "0.757", "0.036", "0.036", "0.056", "0.775"],
        ],
        col_widths=[0.65, 0.60, 0.40, 0.65, 0.40, 0.55],
        roman="III", cap_text="German Credit Baselines (mean ± std, seeds 0–4)")

    body(doc,
         "COMPAS (Table IV) shows the largest baseline disparities "
         "(DP diff 0.149–0.230). Results are interpreted with the Bao et al. "
         "caveat [11]; the accuracy ceiling (~66%) reflects label noise.")

    make_table(doc,
        headers=["Model", "Acc.", "DP Diff", "EO Diff", "AUC"],
        rows=[
            ["LR",      "0.6619", "0.230", "0.134", "0.702"],
            ["RF",      "0.6155", "0.149", "0.089", "0.657"],
            ["XGBoost", "0.6468", "0.199", "0.134", "0.670"],
        ],
        col_widths=[0.70, 0.60, 0.65, 0.65, 0.55],
        roman="IV", cap_text="COMPAS Baselines (race attr., seed 0) [labels contested, 11]")

    sub_heading(doc, "B", "Debiasing Results")
    body(doc,
         "Table V shows all three methods on LR across five seeds. Exponentiated "
         "Gradient and ThresholdOptimizer both achieve near-zero DP difference with "
         "less than 2.5 pp accuracy cost, satisfying the a-priori success criterion "
         "(>50% DP reduction, <3% accuracy loss). Critically, the DP-diff standard "
         "deviation grows from 0.007 (baseline) to 0.011 (ThresholdOptimizer): "
         "the most aggressive method is also the most variable.")

    make_table(doc,
        headers=["Method", "Acc.", "±", "DP", "±", "Reduc.", "Cost"],
        rows=[
            ["None",       "0.8541", "0.0019", "0.174", "0.007", "—",  "—"],
            ["Reweigh.",   "0.8483", "0.0016", "0.095", "0.004", "45%",  "0.6 pp"],
            ["Exp. Grad.", "0.8350", "0.0024", "0.019", "0.008", "89%",  "1.9 pp"],
            ["Thresh.Opt", "0.8330", "0.0027", "0.003", "0.011", "98%",  "2.1 pp"],
        ],
        col_widths=[0.70, 0.52, 0.38, 0.45, 0.38, 0.52, 0.50],
        roman="V", cap_text="Debiasing: LR on Adult Income (mean, seeds 0–4)")

    add_figure(doc, "adult_method_tradeoff.png", width=3.1,
               cap="Fig. 2. Accuracy vs. DP difference for each method on Adult Income. "
                   "Each point is one seed (lower DP, higher accuracy is preferred).")

    body(doc,
         "The RF story (Table VI) is qualitatively different. Reweighing has "
         "essentially zero effect on RF (0.185→0.185). Exponentiated Gradient "
         "achieves 64% reduction at 6.0 pp cost—three times the LR cost "
         "for a smaller gain. ThresholdOptimizer reverses the disparity direction "
         "(−0.064 mean) with a standard deviation 4× larger than on LR, "
         "demonstrating strong model-dependence.")

    make_table(doc,
        headers=["Method", "Acc.", "±", "DP", "±", "Reduc."],
        rows=[
            ["None",       "0.8556", "0.0014", "0.185",   "0.006", "—"],
            ["Reweigh.",   "0.8553", "0.0016", "0.185",   "0.008", "~0%"],
            ["Exp. Grad.", "0.7953", "0.0028", "0.067",   "0.007", "64%"],
            ["Thresh.Opt", "0.7870", "0.0062", "−0.064", "0.046", "rev."],
        ],
        col_widths=[0.78, 0.55, 0.38, 0.55, 0.45, 0.54],
        roman="VI", cap_text="Debiasing: Random Forest on Adult Income")

    body(doc,
         "XGBoost with Exponentiated Gradient (Table VII) achieves 85% DP reduction "
         "at 1.9 pp cost while retaining a 1.7 pp accuracy advantage over "
         "the mitigated LR configuration. This combination represents the best "
         "overall trade-off on Adult Income.")

    make_table(doc,
        headers=["Method", "Acc.", "±", "DP", "±", "Reduc."],
        rows=[
            ["None",       "0.8713", "0.0011", "0.179", "0.006", "—"],
            ["Reweigh.",   "0.8683", "0.0022", "0.118", "0.008", "34%"],
            ["Exp. Grad.", "0.8519", "0.0031", "0.027", "0.009", "85%"],
        ],
        col_widths=[0.78, 0.55, 0.38, 0.55, 0.45, 0.54],
        roman="VII", cap_text="Debiasing: XGBoost on Adult Income")

    add_figure(doc, "adult_pareto_front.png", width=3.1,
               cap="Fig. 3. Pareto front: accuracy vs. DP difference across all "
                   "(model × method × seed) configurations, Adult Income.")

    sub_heading(doc, "C", "Seed Variance Analysis")
    body(doc,
         "Table VIII reports the coefficient of variation (CV = std/|mean|) "
         "of DP difference across seeds. Baseline models are stable (CV ~0.03–0.04). "
         "Reweighing preserves that stability. Exponentiated Gradient and "
         "ThresholdOptimizer have dramatically higher CV values; "
         "ThresholdOptimizer reaches CV = 3.44 on LR because its near-zero "
         "mean amplifies small absolute fluctuations.")

    make_table(doc,
        headers=["Model", "None", "Reweigh.", "Exp.Grad.", "Thresh.Opt."],
        rows=[
            ["LR",      "0.04", "0.04", "0.39", "3.44"],
            ["RF",      "0.03", "0.04", "0.10", "sign flip"],
            ["XGBoost", "0.03", "0.07", "0.34", "—"],
        ],
        col_widths=[0.60, 0.48, 0.60, 0.65, 0.80],
        roman="VIII", cap_text="CV of DP Difference Across 5 Seeds (Adult Income)")

    add_figure(doc, "seed_sweep.png", width=3.1,
               cap="Fig. 4. DP difference distribution across 5 seeds per configuration. "
                   "Wider spread indicates greater fragility.")

    sub_heading(doc, "D", "Threshold Sensitivity")
    body(doc,
         "Table IX shows the full threshold sweep for LR on Adult (seed 0). Moving "
         "the threshold 0.30→0.70 reduces DP difference from 0.282 to 0.103—"
         "a 63% reduction—without any debiasing; accuracy cost is only 1.88 pp. "
         "Raising the default threshold from 0.50 to 0.70 alone yields more than half "
         "the fairness gain of Reweighing at comparable accuracy cost.")

    make_table(doc,
        headers=["Thresh.", "Accuracy", "DP Diff", "EO Diff"],
        rows=[
            ["0.30",          "0.8306", "0.282", "0.122"],
            ["0.35",          "0.8436", "0.251", "0.113"],
            ["0.40",          "0.8498", "0.227", "0.100"],
            ["0.45",          "0.8528", "0.200", "0.090"],
            ["0.50 (default)","0.8541", "0.183", "0.103"],
            ["0.55",          "0.8533", "0.164", "0.100"],
            ["0.60",          "0.8499", "0.144", "0.090"],
            ["0.65",          "0.8457", "0.120", "0.065"],
            ["0.70",          "0.8353", "0.103", "0.079"],
        ],
        col_widths=[0.80, 0.75, 0.72, 0.72],
        roman="IX", cap_text="Threshold Sweep: LR on Adult Income, Seed 0")

    add_figure(doc, "threshold_sweep.png", width=3.1,
               cap="Fig. 5. Threshold sweep for LR and RF on Adult Income. "
                   "DP difference decreases monotonically as threshold rises.")

    body(doc,
         "The sweep also reveals a Pareto-improvement zone at thresholds 0.55–0.65 "
         "where both accuracy and fairness are simultaneously better than at the default "
         "0.50—a free lunch invisible to single-point evaluation.")

    sub_heading(doc, "E", "Intersectional Analysis")
    body(doc,
         "Table X breaks the aggregate sex-based DP difference (0.183) into its "
         "sex × race components. The 4.2× selection-rate ratio "
         "between white males (26.6%) and non-white females (6.4%) is entirely "
         "hidden by the aggregate metric.")

    make_table(doc,
        headers=["Subgroup", "N", "Sel. Rate", "Acc.", "TPR"],
        rows=[
            ["Female, Non-White", "642",   "6.4%",  "93.6%", "51.3%"],
            ["Female, White",     "2,589", "7.4%",  "92.9%", "51.6%"],
            ["Male, Non-White",   "804",   "17.9%", "85.6%", "58.2%"],
            ["Male, White",       "5,734", "26.6%", "81.1%", "61.7%"],
        ],
        col_widths=[1.05, 0.50, 0.65, 0.55, 0.50],
        roman="X", cap_text="Intersectional Stats: LR Baseline, Adult Income, Seed 0")

    add_figure(doc, "intersectional_selection_rate.png", width=3.1,
               cap="Fig. 6. Intersectional selection rates by sex × race. "
                   "An aggregate DP diff of 0.183 masks a 4.2× subgroup ratio.")

    body(doc,
         "The per-group accuracy pattern runs opposite to selection rate "
         "(non-white females: 93.6%; white males: 81.1%), reflecting base-rate "
         "effects: high accuracy on a group that is predominantly negative in the "
         "ground truth does not imply equitable treatment. The TPR values "
         "(51–62%) show the disparity is driven by demographic parity failure, "
         "not differential positive-case detection.")

    sub_heading(doc, "F", "Summary of Key Findings")
    bullet(doc, "Debiasing works in expectation",
           "Exp. Gradient achieves 89% DP reduction on LR and 85% on XGBoost "
           "at <2 pp accuracy cost.")
    bullet(doc, "Effectiveness is model-dependent",
           "Reweighing fails on RF (0% reduction); ThresholdOptimizer reverses "
           "disparity on RF with 4× higher variance than on LR.")
    bullet(doc, "Threshold choice rivals method choice",
           "0.30→0.70 threshold shift yields 63% DP reduction with no debiasing.")
    bullet(doc, "Intersectional analysis is necessary",
           "4.2× selection-rate disparity (white males vs. non-white females) "
           "is invisible in the aggregate DP metric.")
    bullet(doc, "Small datasets amplify all fragility",
           "German Credit LR DP diff spans −0.036 to +0.182 across seeds.")

    # ── V. CONCLUSION ────────────────────────────────────────────────────────
    sec_heading(doc, "V", "Conclusion")
    body(doc,
         "Standard fairness evaluation provides an incomplete view of a method's "
         "practical effectiveness. Across three datasets and three model families, "
         "debiasing works in expectation but is fragile in detail: Exponentiated "
         "Gradient reduces LR DP difference by 89% at 1.9 pp cost, but only "
         "64% on RF at 6.0 pp cost, and Reweighing fails entirely on RF. "
         "Threshold choice alone—from 0.30 to 0.70—produces a 63% DP-"
         "difference reduction. Intersectional analysis reveals a 4.2× disparity "
         "hidden behind an aggregate DP difference of 0.183. German Credit's extreme "
         "seed variance (DP diff spanning 0.218 units for LR) shows that small-dataset "
         "fairness benchmarks are unreliable as point estimates.")
    body(doc,
         "The recommendation is methodological: fairness evaluations must report "
         "distributions over seeds, threshold curves over operating points, and "
         "intersectional breakdowns over subgroups. Our stress-testing framework "
         "implements all three components in a single CLI call and is designed to "
         "drop into any sklearn-compatible pipeline.")

    # ── VI. FUTURE WORK ──────────────────────────────────────────────────────
    sec_heading(doc, "VI", "Future Work")
    body(doc,
         "First, completing the FT-Transformer evaluation [12] would add a tabular "
         "transformer—largely absent from fairness benchmarking literature—"
         "to the model comparison. Second, Folktables distribution-shift experiments "
         "would test whether fairness improvements survive covariate shift. Third, "
         "a scalar fragility index derived from seed-threshold-intersectional variance "
         "would provide a single stability summary per (model × method). "
         "Fourth, expanding from 5 to 10 seeds would tighten statistical claims. "
         "Finally, applying ExpGradient to the FT-Transformer (currently supporting "
         "only Reweighing and ThresholdOptimizer) remains an open challenge.")

    # ── VII. TEAM CONTRIBUTIONS ──────────────────────────────────────────────
    sec_heading(doc, "VII", "Team Contributions")
    contribs = [
        ("D. Verma", "designed the end-to-end pipeline, implemented "
         "fairness_metrics.py, integrated all debiasing methods, ran the "
         "full experiment grid, and consolidated the report."),
        ("N. Dewan", "implemented data_loaders.py for all datasets, designed "
         "the (X, y, sensitive_attrs) schema, and ensured stratified "
         "splitting with no sensitive-attribute leakage."),
        ("M. Batra", "implemented models.py with the common build_model interface "
         "and managed the FT-Transformer wrapper."),
        ("K. Thakare", "implemented debiasing.py and stress_tests.py (seed sweep, "
         "threshold sweep, intersectional breakdown)."),
        ("A. Mote", "implemented visualize.py, produced all six figures, conducted "
         "the literature review, and assembled the presentation deck."),
    ]
    for name, desc in contribs:
        p = _new_para(doc, sa=2, first=0.15)
        _run(p, name + ": ", bold=True, size=10)
        _run(p, desc, size=10)

    # ── REFERENCES ───────────────────────────────────────────────────────────
    sec_heading(doc, "", "References")
    refs = [
        ("[1]",  "N. Mehrabi, F. Morstatter, N. Saxena, K. Lerman, and A. Galstyan, "
                 "“A survey on bias and fairness in machine learning,” "
                 "ACM Comput. Surv., vol. 57, no. 1, pp. 1–38, Jan. 2024."),
        ("[2]",  "S. Uddin, H. Lu, and F. Hajati, “Fairness-preserving framework "
                 "for machine learning: data bias quantification, model evaluation, "
                 "and robustness across multiple datasets,” AI Ethics, vol. 6, "
                 "p. 25, 2026, doi: 10.1007/s43681-025-00874-7."),
        ("[3]",  "N. L. Chen, “Robustness and fairness evaluation of machine "
                 "learning models in high-stakes decision systems,” Int. J. Mach. "
                 "Learn. Commun. Netw., vol. 2, no. 1, pp. 8–12, 2025."),
        ("[4]",  "M. Hardt, E. Price, and N. Srebro, “Equality of opportunity "
                 "in supervised learning,” in Adv. Neural Inf. Process. Syst. "
                 "(NeurIPS), vol. 29, 2016."),
        ("[5]",  "S. Barocas, M. Hardt, and A. Narayanan, Fairness and Machine "
                 "Learning: Limitations and Opportunities. Cambridge, MA, USA: "
                 "MIT Press, 2024."),
        ("[6]",  "F. Kamiran and T. Calders, “Data preprocessing techniques "
                 "for classification without discrimination,” Knowl. Inf. Syst., "
                 "vol. 33, no. 1, pp. 1–33, Jan. 2012."),
        ("[7]",  "A. Agarwal, A. Beygelzimer, M. Dudik, J. Langford, and H. Wallach, "
                 "“A reductions approach to fair classification,” in Proc. "
                 "35th Int. Conf. Mach. Learn. (ICML), Stockholm, Sweden, Jul. 2018, "
                 "pp. 60–69."),
        ("[8]",  "T. Zhao, E. Iofinova, J. Frankle, and D. Alistarh, “Symbiotic "
                 "robustness and fairness in deep neural networks: A triangular "
                 "trade-off,” ACM Comput. Surv., vol. 57, no. 2, 2024."),
        ("[9]",  "K. Roberts-Licklider and T. Trafalis, “Fairness in machine "
                 "learning: A survey part 1,” in Dynamics of Information Systems, "
                 "Lecture Notes Comput. Sci., vol. 14661. Springer, 2025."),
        ("[10]", "K. Crenshaw, “Demarginalizing the intersection of race and sex,” "
                 "Univ. Chicago Legal Forum, vol. 1989, no. 1, Art. 8, 1989."),
        ("[11]", "M. Bao et al., “It’s COMPASlicated: The messy relationship "
                 "between RAI datasets and algorithmic fairness benchmarks,” in "
                 "NeurIPS Datasets Benchmarks Track, 2021."),
        ("[12]", "Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, “Revisiting "
                 "deep learning models for tabular data,” in Adv. Neural Inf. Process. "
                 "Syst. (NeurIPS), vol. 34, 2021."),
        ("[13]", "F. Ding, M. Hardt, J. Miller, and L. Schmidt, “Retiring Adult: "
                 "New datasets for fair machine learning,” in Adv. Neural Inf. "
                 "Process. Syst. (NeurIPS), vol. 34, 2021."),
    ]
    for num, text in refs:
        rp = _new_para(doc, align=WD_ALIGN_PARAGRAPH.JUSTIFY, sb=0, sa=1,
                       first=-0.25, left=0.25)
        _run(rp, num + " ", bold=True, size=REF_SIZE)
        _run(rp, text, size=REF_SIZE)

    doc.save(str(OUT))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    build()
