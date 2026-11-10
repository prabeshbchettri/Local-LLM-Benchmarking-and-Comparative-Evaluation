"""
File: validate_dataset.py

Description:
Validate the versioned benchmark dataset, probes, hashes, and item-defined JSON Schemas.

Responsibilities:
- Load configuration and resolve the canonical dataset path.
- Validate the dataset and both versioned probes.
- Print a human-readable or JSON validation summary.
- Return a nonzero exit code when validation fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark.core.config import load_settings
from benchmark.core.exceptions import BenchmarkError
from benchmark.dataset.validation import render_validation_summary, validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the benchmark dataset, probes, content hash, and JSON Schemas.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset path; defaults to the configured benchmark dataset.",
    )
    parser.add_argument(
        "--performance-probe",
        type=Path,
        default=None,
        help="Performance probe path; defaults to probes/performance_v1.0.0.json.",
    )
    parser.add_argument(
        "--consistency-probe",
        type=Path,
        default=None,
        help="Consistency probe path; defaults to probes/consistency_v1.0.0.json.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Configuration directory; defaults to BENCHMARK_CONFIG_DIR or ./configs.",
    )
    parser.add_argument(
        "--no-probes",
        action="store_true",
        help="Validate only the dataset and skip probe files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the validation summary as JSON.",
    )
    return parser


def _resolve_path(value: Path | None, project_root: Path, default: Path) -> Path:
    if value is None:
        return default
    return value if value.is_absolute() else (project_root / value)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(config_dir=args.config_dir)
        dataset_path = _resolve_path(
            args.dataset,
            settings.project_root,
            settings.benchmark.dataset_path,
        )
        probe_dir = dataset_path.parent / "probes"
        performance_path = None
        consistency_path = None
        if not args.no_probes:
            performance_path = _resolve_path(
                args.performance_probe,
                settings.project_root,
                probe_dir / "performance_v1.0.0.json",
            )
            consistency_path = _resolve_path(
                args.consistency_probe,
                settings.project_root,
                probe_dir / "consistency_v1.0.0.json",
            )
        summary = validate_dataset(
            dataset_path,
            performance_probe_path=performance_path,
            consistency_probe_path=consistency_path,
        )
        if args.json:
            print(json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False))
        else:
            print(render_validation_summary(summary))
        return 0 if summary.valid else 1
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
