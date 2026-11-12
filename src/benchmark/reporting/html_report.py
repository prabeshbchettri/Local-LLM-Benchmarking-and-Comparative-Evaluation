"""
File: html_report.py

Description:
Jinja2 HTML report renderer for one experiment.

Why this file exists:
HTML is the primary human-readable report: it carries the ten required sections,
inline charts, and complete tables without external assets.

Responsibilities:
- Load the packaged template.
- Escape all stored text through Jinja2 autoescaping.
- Return the rendered document as text or UTF-8 bytes.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from benchmark.reporting.bundle import ReportBundle
from benchmark.reporting.presentation import build_view

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATE_NAME = "report.html.j2"


def _environment() -> Environment:
    # The template name ends in .j2, so extension-based autoescape would silently
    # leave untrusted stored model responses unescaped. Enable it explicitly.
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_html(bundle: ReportBundle) -> str:
    """Render the HTML report for one experiment."""
    template = _environment().get_template(TEMPLATE_NAME)
    return template.render(view=build_view(bundle))


def html_bytes(bundle: ReportBundle) -> bytes:
    """Return the rendered HTML report as UTF-8 bytes."""
    return render_html(bundle).encode("utf-8")


def write_html_report(bundle: ReportBundle, path: Path) -> Path:
    """Write the HTML report to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(html_bytes(bundle))
    return path


__all__ = ["html_bytes", "render_html", "write_html_report"]
