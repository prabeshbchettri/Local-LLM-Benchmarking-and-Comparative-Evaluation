"""
File: generate_report.py

Description:
Command-line entry point for generating benchmark reports from stored experiments.

Why this file exists:
Operators need one command that turns a completed experiment in SQLite into CSV,
JSON, HTML, and PDF reports without starting a benchmark.

Responsibilities:
- Parse the experiment id, output directory, and report format options.
- Open the configured database and generate reports through the reporting service.
- Print a reproducible file summary or machine-readable JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from benchmark.core.config import load_settings
from benchmark.core.exceptions import BenchmarkError
from benchmark.database.db import open_database
from benchmark.execution.metadata import format_bytes
from benchmark.reporting import DEFAULT_FORMATS, generate_reports


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate CSV, JSON, HTML, and PDF reports for an experiment."
    )
    parser.add_argument("experiment_id", help="Experiment identifier to report.")
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Database path; defaults to storage/benchmark.db",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to data/exports/<experiment_id>",
    )
    parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        type=str.lower,
        choices=DEFAULT_FORMATS,
        default=None,
        help="Report format to generate; repeatable. Defaults to all four.",
    )
    parser.add_argument("--profile", default=None)
    parser.add_argument("--run-profile", default=None)
    parser.add_argument("--include-cold-start", action="store_true")
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Export a non-completed experiment, flagged as preliminary.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def _print_text(result: object) -> None:
    print(f"Experiment: {result.experiment_id}")
    print(f"Output directory: {result.output_dir}")
    print(f"Preliminary: {'yes' if result.preliminary else 'no'}")
    if result.page_count is not None:
        print(f"PDF pages: {result.page_count}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    print("Files:")
    for generated in result.files:
        print(
            f"  {generated.name} ({generated.format}, "
            f"{format_bytes(generated.size_bytes)}, sha256 {generated.sha256[:12]}...)"
        )
    print(f"Manifest: {result.manifest_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Generate reports and return a process exit code."""
    args = build_parser().parse_args(argv)
    database = None
    try:
        settings = load_settings(config_dir=args.config_dir)
        database = open_database(settings, args.database)
        result = generate_reports(
            database,
            args.experiment_id,
            settings=settings,
            output_dir=args.output_dir,
            formats=tuple(args.formats) if args.formats else DEFAULT_FORMATS,
            profile=args.profile,
            allow_incomplete=args.allow_incomplete,
            run_profile=args.run_profile,
            include_cold_start=args.include_cold_start,
            include_smoke=args.include_smoke,
        )
        if args.json:
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        else:
            _print_text(result)
        return 0
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if database is not None:
            database.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
