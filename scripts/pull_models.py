"""
File: pull_models.py

Description:
Explicitly pull missing configured Ollama models with progress reporting.

Responsibilities:
- Refuse to download unless --yes is supplied.
- Pull only configured models, and skip models already installed.
- Verify metadata after each successful pull.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark.core.config import load_settings
from benchmark.core.exceptions import BenchmarkError
from benchmark.core.startup import initialize_application
from benchmark.execution.metadata import format_bytes, format_pull_event
from benchmark.execution.registry import ModelManager
from benchmark.models.ollama import OllamaPullEvent


def _progress(event: OllamaPullEvent) -> None:
    print(format_pull_event(event), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pull missing configured Ollama models after explicit approval.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        help="Specific configured model tag; may be repeated.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the model downloads explicitly.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Configuration directory; defaults to BENCHMARK_CONFIG_DIR or ./configs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(config_dir=args.config_dir)
        initialize_application(settings)
        if not args.yes:
            print(
                "ERROR: model downloads require explicit approval; pass --yes",
                file=sys.stderr,
            )
            return 2
        manager = ModelManager(settings)
        try:
            statuses = manager.inventory(include_all=False)
            installed = {status.tag for status in statuses if status.installed}
            if args.tag:
                targets = [manager.resolve(tag).spec.tag for tag in args.tag]
            else:
                targets = [
                    spec.tag
                    for spec in settings.models.candidates
                    if spec.tag not in installed
                ]
            if not targets:
                print("All requested models are already installed.")
                return 0
            print("Models to pull:")
            for tag in targets:
                print(f"  {tag}")
            for tag in targets:
                if tag in installed:
                    print(f"Already installed: {tag}")
                    continue
                print(f"Pulling {tag} ...", flush=True)
                metadata = manager.pull(tag, approved=True, progress=_progress)
                print(
                    f"Ready: {metadata.tag} | {format_bytes(metadata.size_bytes)} | "
                    f"{metadata.quantization or 'quant unknown'}"
                )
            return 0
        finally:
            manager.close()
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
