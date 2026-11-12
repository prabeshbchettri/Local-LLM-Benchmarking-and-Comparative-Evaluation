"""
File: run_experiment.py

Description:
Command-line entry point for executing a persisted benchmark experiment.

Why this file exists:
Operators need one command that connects Ollama, the dataset, the evaluation engine,
and the SQLite database without writing Python.

Responsibilities:
- Parse experiment options such as models, profile, item subset, and repetitions.
- Build the model manager, judge, and inference engine from settings.
- Run the orchestrator and print a summary or JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from benchmark.core.config import load_settings
from benchmark.core.exceptions import BenchmarkError
from benchmark.database.db import open_database
from benchmark.dataset.loader import load_dataset
from benchmark.evaluation.judge import build_ollama_judge
from benchmark.execution.engine import InferenceEngine
from benchmark.execution.ollama_client import OllamaClient
from benchmark.execution.orchestrator import ExperimentOrchestrator
from benchmark.execution.registry import ModelManager
from benchmark.models.inference import InferenceProfile, ResponseRecord


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run a benchmark experiment and persist every response to SQLite."
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Database path; defaults to storage/benchmark.db",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset path; defaults to the configured canonical dataset.",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in InferenceProfile],
        default=InferenceProfile.QUALITY.value,
    )
    parser.add_argument("--items", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--cooldown", type=float, default=None)
    parser.add_argument("--cold-start", action="store_true")
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--resume",
        default=None,
        metavar="EXPERIMENT_ID",
        help="Continue an unfinished experiment without repeating stored requests",
    )
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--no-manager",
        action="store_true",
        help="Skip explicit load/unload calls (Ollama manages the model lifecycle)",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _response_line(record: ResponseRecord) -> str:
    latency = record.timing.total_latency_seconds
    latency_text = f"{latency:.2f}s" if latency is not None else "unknown"
    return (
        f"{record.model_tag} | {record.item_id or '-'} | rep {record.repetition} | "
        f"{record.status.value} | {latency_text}"
    )


def _print_text(report: Any) -> None:
    print(f"Experiment: {report.experiment_id} ({report.status})")
    print(f"Database: {report.database_path}")
    print(
        f"Responses: {report.responses} "
        f"({report.successful_responses} successful, "
        f"{report.failed_responses} failed)"
    )
    print(f"Evaluations: {report.evaluations}")
    print(f"Runs: {report.runs}")
    if report.resumed:
        print("Resumed an existing experiment")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment CLI and return a process exit code."""
    args = build_parser().parse_args(argv)
    database = None
    manager = None
    own_client = None
    try:
        settings = load_settings(config_dir=args.config_dir)
        dataset = load_dataset(args.dataset or settings.benchmark.dataset_path)
        database = open_database(settings, args.database)
        if args.no_manager:
            own_client = OllamaClient(
                str(settings.models.ollama_base_url),
                timeout_seconds=settings.benchmark.request_timeout_seconds,
            )
            client = own_client
        else:
            manager = ModelManager(settings)
            client = manager.client
        judge = (
            None
            if args.no_judge
            else build_ollama_judge(client, settings.evaluation, settings.models.judge)
        )
        engine = InferenceEngine(settings, client)
        orchestrator = ExperimentOrchestrator(
            settings,
            database,
            dataset,
            engine,
            manager=manager,
            judge=judge,
            progress=None if args.quiet else lambda record: print(
                _response_line(record),
                flush=True,
            ),
        )
        report = orchestrator.run(
            args.models,
            profile=args.profile,
            item_ids=args.items,
            limit=args.limit,
            repetitions=args.repeats,
            warmup_requests=args.warmup,
            cold_start=args.cold_start,
            cooldown_seconds=args.cooldown,
            name=args.name,
            resume_experiment=args.resume,
        )
        if args.json:
            print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        else:
            _print_text(report)
        if report.interrupted:
            print(
                "Interrupted; stored results were kept and the experiment can be resumed",
                file=sys.stderr,
            )
            return 130
        return 0
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Interrupted; stored results were kept and the experiment can be resumed",
            file=sys.stderr,
        )
        return 130
    finally:
        if database is not None:
            database.dispose()
        if manager is not None:
            manager.close()
        elif own_client is not None:
            own_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
