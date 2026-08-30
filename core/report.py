"""
report.py
=========
Automated export of inspection results to PDF.

This covers the *Reporting* extra-effort requirement in the assignment
specification: "automated export of results and findings into PDF format".

Two report shapes are produced:

* :func:`build_single_report` — one board, with the annotated image, the verdict,
  the per-defect table and the pipeline configuration that produced it.
* :func:`build_batch_report`  — a production run, with yield statistics, the
  defect distribution and a per-board results table.

Both return PDF bytes, so the interface can hand them straight to a Streamlit
download button without touching the filesystem.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Sequence

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .analysis import BatchSummary, InspectionSummary
from .detector import Detection
from .viz import encode_jpeg

# --------------------------------------------------------------------------- #
# Shared styling
# --------------------------------------------------------------------------- #
_INK = colors.HexColor("#1f2937")
_MUTED = colors.HexColor("#6b7280")
_RULE = colors.HexColor("#d1d5db")
_HEADER_BG = colors.HexColor("#f3f4f6")

_VERDICT_COLOURS = {
    "PASS": colors.HexColor("#27ae60"),
    "REVIEW": colors.HexColor("#f39c12"),
    "FAIL": colors.HexColor("#c0392b"),
}


def _styles() -> dict[str, ParagraphStyle]:
    """Paragraph styles shared by both report shapes."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=18, leading=22, textColor=_INK, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, leading=12, textColor=_MUTED, spaceAfter=10,
        ),
        "heading": ParagraphStyle(
            "heading", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=11, leading=14, textColor=_INK, spaceBefore=12, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, leading=13, textColor=_INK, alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontName="Helvetica",
            fontSize=7.5, leading=10, textColor=_MUTED,
        ),
    }


def _table_style(header: bool = True) -> TableStyle:
    """Consistent table appearance across the report."""
    commands: list[tuple] = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), _INK),
        ("GRID", (0, 0), (-1, -1), 0.4, _RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    if header:
        commands += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ]
    return TableStyle(commands)


def _verdict_banner(verdict: str, subtitle: str, styles: dict[str, ParagraphStyle]) -> Table:
    """A full-width coloured strip carrying the verdict."""
    colour = _VERDICT_COLOURS.get(verdict, _MUTED)
    cell = Paragraph(
        f'<font color="white" size="15"><b>{verdict}</b></font>'
        f'<br/><font color="white" size="8">{subtitle}</font>',
        styles["body"],
    )
    table = Table([[cell]], colWidths=[170 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colour),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def _image_flowable(image: np.ndarray | None, max_width_mm: float = 165.0,
                    max_height_mm: float = 115.0) -> Image | None:
    """
    Wrap a BGR array as a reportlab flowable, scaled to fit the page while
    preserving its aspect ratio.

    Returns ``None`` when the array is missing or cannot be encoded, so the
    caller can simply skip the figure.
    """
    if image is None:
        return None
    encoded = encode_jpeg(image, quality=88)
    if encoded is None:
        return None

    reader = ImageReader(io.BytesIO(encoded))
    source_w, source_h = reader.getSize()
    scale = min((max_width_mm * mm) / source_w, (max_height_mm * mm) / source_h)
    return Image(io.BytesIO(encoded), width=source_w * scale, height=source_h * scale)


def _footer(canvas, doc) -> None:
    """Page furniture: a rule, the system name and the page number."""
    canvas.saveState()
    canvas.setStrokeColor(_RULE)
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, 14 * mm, A4[0] - 20 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(_MUTED)
    canvas.drawString(20 * mm, 9.5 * mm, "PCB Defect Inspection System — automated report")
    canvas.drawRightString(A4[0] - 20 * mm, 9.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _document(buffer: io.BytesIO, title: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=title,
        author="PCB Defect Inspection System",
    )


def _config_rows(config: dict[str, Any]) -> list[list[str]]:
    """Render the pipeline configuration dictionary as table rows."""
    return [["Setting", "Value"]] + [
        [str(key), "—" if value is None else str(value)] for key, value in config.items()
    ]


# --------------------------------------------------------------------------- #
# Single-board report
# --------------------------------------------------------------------------- #
def build_single_report(
    annotated: np.ndarray | None,
    summary: InspectionSummary,
    detections: Sequence[Detection],
    config: dict[str, Any],
    board_id: str = "Board 01",
    original: np.ndarray | None = None,
) -> bytes:
    """
    Build the PDF inspection certificate for one board.

    Args:
        annotated: the detection overlay shown in the interface.
        summary: the verdict and statistics for this board.
        detections: the individual findings, listed in a table.
        config: pipeline settings (model, thresholds, modules used) recorded for
            traceability.
        board_id: identifier printed in the header — normally the file name.
        original: the operator's input image, included for before/after context.

    Returns:
        The PDF as bytes.
    """
    styles = _styles()
    buffer = io.BytesIO()
    doc = _document(buffer, f"PCB inspection report — {board_id}")
    story: list[Any] = []

    timestamp = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    story.append(Paragraph("PCB Defect Inspection Report", styles["title"]))
    story.append(
        Paragraph(
            f"Board: <b>{board_id}</b> &nbsp;·&nbsp; Generated: {timestamp} "
            f"&nbsp;·&nbsp; Detector: {config.get('Detector weights', 'not specified')}",
            styles["subtitle"],
        )
    )

    subtitle = (
        f"Quality score {summary.quality_score:.0f}/100 · "
        f"{summary.total_defects} defect(s) detected · "
        f"inference {summary.inference_ms:.0f} ms"
    )
    story.append(_verdict_banner(summary.verdict, subtitle, styles))
    story.append(Spacer(1, 10))

    # -- reasoning ---------------------------------------------------------- #
    story.append(Paragraph("Assessment", styles["heading"]))
    for reason in summary.reasons:
        story.append(Paragraph(f"• {reason}", styles["body"]))

    # -- key figures -------------------------------------------------------- #
    story.append(Paragraph("Inspection summary", styles["heading"]))
    height, width = summary.image_shape
    metrics = [
        ["Metric", "Value", "Metric", "Value"],
        ["Verdict", summary.verdict, "Total defects", str(summary.total_defects)],
        ["Quality score", f"{summary.quality_score:.1f} / 100", "Critical defects", str(summary.critical_defects)],
        ["Mean confidence", f"{summary.mean_confidence:.3f}" if summary.total_defects else "—",
         "Low-confidence findings", str(summary.uncertain_defects)],
        ["Inspected resolution", f"{width} x {height} px", "Inference time", f"{summary.inference_ms:.1f} ms"],
    ]
    metrics_table = Table(metrics, colWidths=[42 * mm, 43 * mm, 42 * mm, 43 * mm])
    metrics_table.setStyle(_table_style())
    story.append(metrics_table)

    # -- annotated image ---------------------------------------------------- #
    figure = _image_flowable(annotated)
    if figure is not None:
        story.append(Paragraph("Annotated inspection result", styles["heading"]))
        story.append(figure)
        story.append(Paragraph(
            "Bounding boxes mark every detection above the configured confidence "
            "threshold. Colours identify the defect class.", styles["small"],
        ))

    # -- per-defect table --------------------------------------------------- #
    story.append(Paragraph("Detected defects", styles["heading"]))
    if detections:
        rows: list[list[str]] = [
            # Superscripts are avoided: the built-in Helvetica encoding has no
            # glyph for them and reportlab would silently drop the character.
            ["#", "Defect type", "Confidence", "Centre (x, y)", "Size (w x h)", "Area (sq. px)"]
        ]
        for index, detection in enumerate(detections, start=1):
            cx, cy = detection.centre
            rows.append([
                str(index),
                detection.class_name,
                f"{detection.confidence:.3f}",
                f"({cx:.0f}, {cy:.0f})",
                f"{detection.width:.0f} x {detection.height:.0f}",
                f"{detection.area:.0f}",
            ])
        defect_table = Table(rows, colWidths=[12 * mm, 42 * mm, 26 * mm, 32 * mm, 30 * mm, 28 * mm])
        defect_table.setStyle(_table_style())
        story.append(defect_table)
    else:
        story.append(Paragraph(
            "No defects were detected above the configured confidence threshold.",
            styles["body"],
        ))

    # -- configuration ------------------------------------------------------ #
    story.append(Paragraph("Pipeline configuration", styles["heading"]))
    config_table = Table(_config_rows(config), colWidths=[65 * mm, 105 * mm])
    config_table.setStyle(_table_style())
    story.append(config_table)

    # -- input image on a second page --------------------------------------- #
    original_figure = _image_flowable(original)
    if original_figure is not None:
        story.append(PageBreak())
        story.append(Paragraph("Operator input (before processing)", styles["heading"]))
        story.append(original_figure)
        story.append(Paragraph(
            "The image as supplied, before pre-processing (Module 1) and "
            "alignment (Module 2) were applied.", styles["small"],
        ))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Batch report
# --------------------------------------------------------------------------- #
def build_batch_report(
    batch: BatchSummary,
    rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    run_name: str = "Batch run",
    gallery: Sequence[tuple[str, np.ndarray]] = (),
) -> bytes:
    """
    Build the PDF summary for a production batch.

    Args:
        batch: aggregate statistics for the run.
        rows: one dictionary per board, as produced by
            ``InspectionSummary.as_row(name)``.
        config: pipeline settings recorded for traceability.
        run_name: identifier printed in the header — normally the source folder.
        gallery: up to a handful of ``(caption, annotated_image)`` pairs appended
            as an evidence appendix. Failed boards are the useful choice here.

    Returns:
        The PDF as bytes.
    """
    styles = _styles()
    buffer = io.BytesIO()
    doc = _document(buffer, f"PCB batch inspection report — {run_name}")
    story: list[Any] = []

    timestamp = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    story.append(Paragraph("PCB Batch Inspection Report", styles["title"]))
    story.append(
        Paragraph(
            f"Run: <b>{run_name}</b> &nbsp;·&nbsp; Generated: {timestamp} "
            f"&nbsp;·&nbsp; Boards inspected: {batch.total_boards}",
            styles["subtitle"],
        )
    )

    overall = "PASS" if batch.failed == 0 else "FAIL"
    story.append(_verdict_banner(
        overall,
        f"Yield {batch.yield_pct:.1f}% · {batch.failed} board(s) failed · "
        f"{batch.total_defects} defect(s) in total",
        styles,
    ))
    story.append(Spacer(1, 10))

    # -- run statistics ----------------------------------------------------- #
    story.append(Paragraph("Run statistics", styles["heading"]))
    stats = [
        ["Metric", "Value", "Metric", "Value"],
        ["Boards inspected", str(batch.total_boards), "Yield", f"{batch.yield_pct:.1f} %"],
        ["Passed", str(batch.passed), "Flagged for review", str(batch.review)],
        ["Failed", str(batch.failed), "Defects per board", f"{batch.defect_rate:.2f}"],
        ["Total defects", str(batch.total_defects), "Mean quality score", f"{batch.mean_quality:.1f} / 100"],
        ["Mean inference time", f"{batch.mean_inference_ms:.1f} ms", "", ""],
    ]
    stats_table = Table(stats, colWidths=[42 * mm, 43 * mm, 42 * mm, 43 * mm])
    stats_table.setStyle(_table_style())
    story.append(stats_table)

    # -- defect distribution ------------------------------------------------ #
    story.append(Paragraph("Defect distribution", styles["heading"]))
    if batch.class_counts:
        total = sum(batch.class_counts.values())
        dist_rows: list[list[str]] = [["Defect type", "Count", "Share of all defects"]]
        for name, count in sorted(batch.class_counts.items(), key=lambda item: -item[1]):
            dist_rows.append([name, str(count), f"{100.0 * count / total:.1f} %"])
        dist_table = Table(dist_rows, colWidths=[80 * mm, 45 * mm, 45 * mm])
        dist_table.setStyle(_table_style())
        story.append(dist_table)
    else:
        story.append(Paragraph("No defects were detected across the run.", styles["body"]))

    # -- per-board results -------------------------------------------------- #
    story.append(Paragraph("Per-board results", styles["heading"]))
    if rows:
        columns = ["image", "verdict", "quality_score", "total_defects",
                   "critical_defects", "mean_confidence"]
        headers = ["Image", "Verdict", "Score", "Defects", "Critical", "Mean conf."]
        board_rows: list[list[str]] = [headers]
        for row in rows:
            name = str(row.get("image", ""))
            if len(name) > 34:
                name = name[:31] + "..."
            board_rows.append([name] + [str(row.get(column, "")) for column in columns[1:]])

        board_table = Table(
            board_rows,
            colWidths=[62 * mm, 22 * mm, 18 * mm, 22 * mm, 22 * mm, 24 * mm],
            repeatRows=1,
        )
        style = _table_style()
        # Tint each row by its verdict so failures are findable at a glance.
        for index, row in enumerate(rows, start=1):
            verdict = str(row.get("verdict", ""))
            if verdict in ("FAIL", "REVIEW"):
                tint = colors.HexColor("#fdecea" if verdict == "FAIL" else "#fef5e7")
                style.add("BACKGROUND", (0, index), (-1, index), tint)
        board_table.setStyle(style)
        story.append(board_table)
    else:
        story.append(Paragraph("No boards were inspected in this run.", styles["body"]))

    # -- configuration ------------------------------------------------------ #
    story.append(Paragraph("Pipeline configuration", styles["heading"]))
    config_table = Table(_config_rows(config), colWidths=[65 * mm, 105 * mm])
    config_table.setStyle(_table_style())
    story.append(config_table)

    # -- evidence appendix -------------------------------------------------- #
    if gallery:
        story.append(PageBreak())
        story.append(Paragraph("Appendix — annotated evidence", styles["heading"]))
        for caption, image in gallery:
            figure = _image_flowable(image, max_width_mm=120.0, max_height_mm=85.0)
            if figure is None:
                continue
            story.append(Paragraph(f"<b>{caption}</b>", styles["body"]))
            story.append(figure)
            story.append(Spacer(1, 8))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
