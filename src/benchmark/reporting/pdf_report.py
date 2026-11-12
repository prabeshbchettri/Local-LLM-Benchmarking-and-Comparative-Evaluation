"""
File: pdf_report.py

Description:
ReportLab PDF report renderer for one experiment.

Why this file exists:
PDF is the portable, print-ready format. It must stay readable, keep tables inside
the page, include charts where useful, and always show a header and page numbers.

Responsibilities:
- Register a Unicode TrueType font when the platform provides one.
- Build a styled document with all ten sections plus appendices.
- Keep wide tables inside the page using wrapped cells and repeating headers.
- Draw charts with ReportLab graphics and a header/footer on every page.
- Report page count and layout warnings instead of hiding them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import (
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)

from benchmark.core.exceptions import ReportError
from benchmark.reporting.bundle import ReportBundle
from benchmark.reporting.presentation import build_view

PAGE_SIZE = A4
LEFT_MARGIN = 16 * mm
RIGHT_MARGIN = 16 * mm
TOP_MARGIN = 20 * mm
BOTTOM_MARGIN = 16 * mm
CONTENT_WIDTH = PAGE_SIZE[0] - LEFT_MARGIN - RIGHT_MARGIN


@dataclass
class PdfResult:
    """Result of building one PDF report."""

    data: bytes
    page_count: int
    warnings: list[str] = field(default_factory=list)
    font_name: str = "Helvetica"


@dataclass
class _FontSet:
    body: str
    bold: str
    warnings: list[str]


def _windows_font_dir() -> Path:
    return Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _register_fonts() -> _FontSet:
    """Register a Unicode TrueType font pair when available."""
    warnings: list[str] = []
    fonts = _windows_font_dir()
    registered = set(pdfmetrics.getRegisteredFontNames())
    body_path = _first_existing([fonts / "ARIALUNI.TTF", fonts / "DejaVuSans.ttf"])
    bold_path = _first_existing([fonts / "DejaVuSans-Bold.ttf", fonts / "ARIALUNI.TTF"])
    body = "Helvetica"
    bold = "Helvetica-Bold"
    if body_path is not None:
        try:
            if "ReportBody" not in registered:
                pdfmetrics.registerFont(TTFont("ReportBody", str(body_path)))
            body = "ReportBody"
        except Exception as exc:  # fall back rather than fail the report
            warnings.append(f"Unicode body font unavailable ({exc}); using Helvetica")
    else:
        warnings.append(
            "No Unicode TrueType font was found; PDF text is limited to the "
            "built-in Helvetica character set"
        )
    if bold_path is not None:
        try:
            if "ReportBold" not in registered:
                pdfmetrics.registerFont(TTFont("ReportBold", str(bold_path)))
            bold = "ReportBold"
        except Exception as exc:
            warnings.append(f"Bold font unavailable ({exc}); using Helvetica-Bold")
    pdfmetrics.registerFontFamily(
        "Report", normal=body, bold=bold, italic=body, boldItalic=bold
    )
    return _FontSet(body=body, bold=bold, warnings=warnings)


def _styles(fonts: _FontSet) -> dict[str, ParagraphStyle]:
    """Build the paragraph styles used by the PDF."""
    return {
        "Title": ParagraphStyle(
            "ReportTitle",
            fontName=fonts.bold,
            fontSize=19,
            leading=23,
            spaceAfter=2,
            textColor=colors.HexColor("#0f172a"),
        ),
        "Subtitle": ParagraphStyle(
            "ReportSubtitle",
            fontName=fonts.body,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#475569"),
            spaceAfter=6,
        ),
        "Heading": ParagraphStyle(
            "ReportHeading",
            fontName=fonts.bold,
            fontSize=13,
            leading=16,
            spaceBefore=14,
            spaceAfter=6,
            textColor=colors.HexColor("#0f172a"),
        ),
        "Subheading": ParagraphStyle(
            "ReportSubheading",
            fontName=fonts.bold,
            fontSize=10,
            leading=13,
            spaceBefore=8,
            spaceAfter=4,
            textColor=colors.HexColor("#1e293b"),
        ),
        "Body": ParagraphStyle(
            "ReportBody",
            fontName=fonts.body,
            fontSize=8.6,
            leading=11.6,
            spaceAfter=4,
        ),
        "Small": ParagraphStyle(
            "ReportSmall",
            fontName=fonts.body,
            fontSize=7.4,
            leading=9.6,
            textColor=colors.HexColor("#475569"),
        ),
        "Banner": ParagraphStyle(
            "ReportBanner",
            fontName=fonts.bold,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#92400e"),
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            fontName=fonts.bold,
            fontSize=7.1,
            leading=8.6,
            textColor=colors.HexColor("#0f172a"),
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            fontName=fonts.body,
            fontSize=7.0,
            leading=8.6,
        ),
        "TableCellRight": ParagraphStyle(
            "TableCellRight",
            fontName=fonts.body,
            fontSize=7.0,
            leading=8.6,
            alignment=2,
        ),
        "RecHead": ParagraphStyle(
            "RecHead",
            fontName=fonts.bold,
            fontSize=8.6,
            leading=11,
            spaceBefore=6,
            spaceAfter=2,
        ),
    }


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text), style)


def _table(
    block: dict[str, Any],
    styles: dict[str, ParagraphStyle],
    *,
    first_width: float = 1.7,
) -> LongTable:
    """Build a table whose cells wrap and whose header repeats across pages."""
    columns = block["columns"]
    header = [
        _paragraph(str(column["label"]), styles["TableHeader"]) for column in columns
    ]
    data: list[list[Any]] = [header]
    for row in block["rows"]:
        cells: list[Any] = []
        for column in columns:
            style = (
                styles["TableCellRight"]
                if column["align"] == "right"
                else styles["TableCell"]
            )
            cells.append(_paragraph(str(row.get(column["key"], "")), style))
        data.append(cells)
    weights = [first_width] + [1.0] * (len(columns) - 1)
    total = sum(weights)
    widths = [CONTENT_WIDTH * weight / total for weight in weights]
    table = LongTable(data, colWidths=widths, repeatRows=1, splitByRow=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return table


def _kv_table(
    rows: list[dict[str, str]],
    styles: dict[str, ParagraphStyle],
) -> LongTable:
    data = [
        [
            _paragraph(str(row["label"]), styles["TableHeader"]),
            _paragraph(str(row["value"]), styles["TableCell"]),
        ]
        for row in rows
    ]
    table = LongTable(data, colWidths=[CONTENT_WIDTH * 0.32, CONTENT_WIDTH * 0.68], repeatRows=0)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def _chart_drawing(
    chart: dict[str, Any],
    fonts: _FontSet,
) -> Drawing | None:
    """Build a horizontal bar chart drawing for one chart specification."""
    categories = chart["categories"]
    values = chart["values"]
    if not categories or not values:
        return None
    label_width = min(
        max(
            stringWidth(str(category), fonts.body, 7)
            for category in categories
        )
        + 12,
        CONTENT_WIDTH * 0.45,
    )
    height = max(96.0, 22.0 * len(categories) + 46.0)
    drawing = Drawing(CONTENT_WIDTH, height)
    bars = HorizontalBarChart()
    bars.x = label_width
    bars.y = 26
    bars.width = max(CONTENT_WIDTH - label_width - 52, 80)
    bars.height = height - 48
    bars.data = [values]
    bars.categoryAxis.categoryNames = [str(category) for category in categories]
    bars.categoryAxis.labels.fontName = fonts.body
    bars.categoryAxis.labels.fontSize = 7
    bars.valueAxis.valueMin = 0
    bars.valueAxis.labels.fontName = fonts.body
    bars.valueAxis.labels.fontSize = 7
    bars.bars[0].fillColor = colors.HexColor("#2563eb")
    bars.barLabels.fontName = fonts.body
    bars.barLabels.fontSize = 7
    bars.barLabels.nudge = 6
    digits = chart.get("digits", 2)
    bars.barLabelFormat = f"%.{digits}f"
    drawing.add(bars)
    return drawing


class _ReportCanvas(pdf_canvas.Canvas):
    """Canvas that keeps every page so headers and 'Page X of Y' can be drawn."""

    def __init__(
        self,
        *args: Any,
        header_left: str = "",
        header_right: str = "",
        footer_left: str = "",
        font_body: str = "Helvetica",
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._header_left = header_left
        self._header_right = header_right
        self._footer_left = footer_left
        self._font_body = font_body
        self._state = state if state is not None else {}
        self._saved_states: list[dict[str, Any]] = []

    def showPage(self) -> None:  # noqa: N802 - ReportLab API
        self._saved_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total = len(self._saved_states)
        for index, state in enumerate(self._saved_states, start=1):
            self.__dict__.update(state)
            self._draw_chrome(index, total)
            pdf_canvas.Canvas.showPage(self)
        pdf_canvas.Canvas.save(self)
        self._state["page_count"] = total

    def _draw_chrome(self, page: int, total: int) -> None:
        self.saveState()
        self.setFont(self._font_body, 7.2)
        self.setFillColor(colors.HexColor("#475569"))
        self.drawString(LEFT_MARGIN, PAGE_SIZE[1] - 12 * mm, self._header_left)
        self.drawRightString(
            PAGE_SIZE[0] - RIGHT_MARGIN,
            PAGE_SIZE[1] - 12 * mm,
            self._header_right,
        )
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(
            LEFT_MARGIN,
            PAGE_SIZE[1] - 13.6 * mm,
            PAGE_SIZE[0] - RIGHT_MARGIN,
            PAGE_SIZE[1] - 13.6 * mm,
        )
        self.line(LEFT_MARGIN, 12 * mm, PAGE_SIZE[0] - RIGHT_MARGIN, 12 * mm)
        self.drawString(LEFT_MARGIN, 8.4 * mm, self._footer_left)
        self.drawRightString(
            PAGE_SIZE[0] - RIGHT_MARGIN,
            8.4 * mm,
            f"Page {page} of {total}",
        )
        self.restoreState()


def _story(
    view: dict[str, Any],
    bundle: ReportBundle,
    styles: dict[str, ParagraphStyle],
    fonts: _FontSet,
) -> list[Any]:
    """Build the flowable story for the PDF report."""
    story: list[Any] = []
    story.append(_paragraph(view["title"], styles["Title"]))
    story.append(_paragraph(view["subtitle"], styles["Subtitle"]))
    story.append(
        _paragraph(
            f"Generated {view['generated_at']} · report schema {view['schema_version']}",
            styles["Small"],
        )
    )
    if view["preliminary"]:
        story.append(Spacer(1, 6))
        banner = _paragraph(
            "PRELIMINARY DATA — experiment status is "
            f"{view['experiment']['status']}. This report is not a final benchmark result.",
            styles["Banner"],
        )
        banner_table = LongTable([[banner]], colWidths=[CONTENT_WIDTH])
        banner_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef3c7")),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#f59e0b")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(banner_table)
    story.append(Spacer(1, 4))

    story.append(_paragraph("1. Executive summary", styles["Heading"]))
    story.append(_paragraph(view["summary"]["narrative"], styles["Body"]))
    story.append(
        _kv_table(
            [
                {"label": card["label"], "value": f"{card['value']} — {card['detail']}"}
                for card in view["summary"]["key_metrics"]
            ],
            styles,
        )
    )

    story.append(_paragraph("2. Experimental environment", styles["Heading"]))
    story.append(_kv_table(view["environment"], styles))

    story.append(_paragraph("3. Methodology", styles["Heading"]))
    for paragraph in view["methodology"]["paragraphs"]:
        story.append(_paragraph(paragraph, styles["Body"]))
    story.append(_paragraph("Scoring weights", styles["Subheading"]))
    story.append(_table(view["methodology"]["weights"], styles, first_width=2.2))
    story.append(
        _paragraph(
            "Run profile: "
            f"{view['methodology']['filters']['run_profile']} · cold-start included: "
            f"{view['methodology']['filters']['include_cold_start']} · smoke included: "
            f"{view['methodology']['filters']['include_smoke']}",
            styles["Small"],
        )
    )

    story.append(_paragraph("4. Model metadata", styles["Heading"]))
    story.append(_table(view["tables"]["models"], styles))

    story.append(PageBreak())
    story.append(_paragraph("5. Performance results", styles["Heading"]))
    story.append(_table(view["tables"]["performance"], styles))
    for chart in view["charts"]:
        if chart["key"] not in {"composite", "latency_p95"}:
            continue
        story.append(_paragraph(chart["title"], styles["Subheading"]))
        story.append(_paragraph(chart["unit"], styles["Small"]))
        drawing = _chart_drawing(chart, fonts)
        if drawing is not None:
            story.append(drawing)

    story.append(_paragraph("6. Quality results", styles["Heading"]))
    story.append(_table(view["tables"]["quality"], styles))
    story.append(_paragraph("Category breakdown", styles["Subheading"]))
    story.append(_table(view["tables"]["quality_categories"], styles))
    for chart in view["charts"]:
        if chart["key"] not in {"quality", "throughput"}:
            continue
        story.append(_paragraph(chart["title"], styles["Subheading"]))
        story.append(_paragraph(chart["unit"], styles["Small"]))
        drawing = _chart_drawing(chart, fonts)
        if drawing is not None:
            story.append(drawing)

    story.append(PageBreak())
    story.append(_paragraph("7. Hardware analysis", styles["Heading"]))
    story.append(_paragraph("CPU and RAM", styles["Subheading"]))
    story.append(_table(view["tables"]["hardware_cpu_ram"], styles))
    story.append(_paragraph("GPU, VRAM, and temperature", styles["Subheading"]))
    story.append(_table(view["tables"]["hardware_gpu"], styles))
    story.append(
        _paragraph(
            "Hardware values are system-wide samples and are not exclusively "
            "attributable to the model. Missing sensors appear as a dash.",
            styles["Small"],
        )
    )

    story.append(_paragraph("8. Reliability analysis", styles["Heading"]))
    story.append(_table(view["tables"]["reliability"], styles))

    story.append(_paragraph("9. Recommendations", styles["Heading"]))
    if not view["recommendations"]:
        story.append(_paragraph("No workload recommendation could be computed.", styles["Body"]))
    for result in view["recommendations"]:
        story.append(_paragraph(result["workload"], styles["Subheading"]))
        if not result["available"]:
            story.append(
                _paragraph(
                    f"Unavailable: {result['reason'] or 'insufficient data'}",
                    styles["Body"],
                )
            )
            continue
        for entry in result["entries"]:
            story.append(
                _paragraph(
                    f"#{entry['rank']} {entry['model']} · score {entry['score']} · "
                    f"confidence {entry['confidence']}",
                    styles["RecHead"],
                )
            )
            story.append(_paragraph(entry["rationale"], styles["Body"]))
            if entry["strengths"]:
                story.append(
                    _paragraph(f"Strengths: {'; '.join(entry['strengths'])}", styles["Small"])
                )
            if entry["weaknesses"]:
                story.append(
                    _paragraph(
                        f"Weaknesses: {'; '.join(entry['weaknesses'])}", styles["Small"]
                    )
                )
            if entry["limitations"]:
                story.append(
                    _paragraph(
                        f"Limitations: {'; '.join(entry['limitations'])}", styles["Small"]
                    )
                )
            story.append(
                _paragraph(
                    f"Evidence: {entry['evidence']} responses · quality coverage "
                    f"{entry['coverage']}",
                    styles["Small"],
                )
            )

    story.append(_paragraph("10. Limitations", styles["Heading"]))
    for item in view["limitations"]:
        story.append(_paragraph(f"• {item}", styles["Body"]))

    if view["tables"]["errors"]["rows"]:
        story.append(_paragraph("Appendix — structured errors", styles["Heading"]))
        story.append(_table(view["tables"]["errors"], styles))

    if view["excerpts"]:
        story.append(_paragraph("Appendix — response excerpts", styles["Heading"]))
        story.append(
            _paragraph(
                "Excerpts are truncated to 600 characters; the CSV and JSON exports "
                "contain full response text.",
                styles["Small"],
            )
        )
        for excerpt in view["excerpts"]:
            story.append(
                _paragraph(
                    f"{excerpt['model']} · {excerpt['question']}", styles["Subheading"]
                )
            )
            story.append(_paragraph(excerpt["text"], styles["Body"]))
    return story


def build_pdf(bundle: ReportBundle) -> PdfResult:
    """Build the PDF report and return its bytes, page count, and warnings."""
    view = build_view(bundle)
    fonts = _register_fonts()
    styles = _styles(fonts)
    buffer = BytesIO()
    state: dict[str, Any] = {}
    header_left = "Local LLM Benchmark — experiment report"
    header_right = bundle.experiment_id
    footer_left = f"Generated {view['generated_at']} · {view['summary']['profile']} profile"
    maker = lambda *args, **kwargs: _ReportCanvas(  # noqa: E731
        *args,
        header_left=header_left,
        header_right=header_right,
        footer_left=footer_left,
        font_body=fonts.body,
        state=state,
        **kwargs,
    )
    document = SimpleDocTemplate(
        buffer,
        pagesize=PAGE_SIZE,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title=view["title"],
        author="Local LLM Benchmark",
        subject="Benchmark experiment report",
    )
    try:
        document.build(_story(view, bundle, styles, fonts), canvasmaker=maker)
    except ReportError:
        raise
    except Exception as exc:
        raise ReportError(f"PDF layout failed: {exc}") from exc
    data = buffer.getvalue()
    if not data.startswith(b"%PDF"):
        raise ReportError("PDF build did not produce a valid PDF document")
    warnings = list(fonts.warnings)
    if view["preliminary"]:
        warnings.append(
            "PDF contains preliminary data because the experiment is not completed"
        )
    return PdfResult(
        data=data,
        page_count=int(state.get("page_count", 0)),
        warnings=warnings,
        font_name=fonts.body,
    )


def write_pdf_report(bundle: ReportBundle, path: Path) -> PdfResult:
    """Write the PDF report to a file and return the build result."""
    result = build_pdf(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.data)
    return result


__all__ = ["PdfResult", "build_pdf", "write_pdf_report"]
