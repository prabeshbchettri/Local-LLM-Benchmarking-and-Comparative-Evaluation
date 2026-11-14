"""
File: __main__.py

Description:
Command-line entry point for initialization, health checks, and model management.

Responsibilities:
- Parse init, health, models, infer, monitor, analyze, compare, and report commands.
- Load and validate configuration.
- Initialize runtime directories and logging.
- Manage Ollama models and run or report on benchmark experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from benchmark.analytics import AnalyticsEngine, build_recommendation_set
from benchmark.analytics.models import ExperimentAnalytics, RecommendationSet
from benchmark.core.config import AppSettings, load_settings
from benchmark.core.exceptions import BenchmarkError
from benchmark.core.health import render_health_report, run_health_check
from benchmark.core.startup import initialize_application
from benchmark.database.db import Database
from benchmark.dataset.loader import load_dataset, ordered_items
from benchmark.dataset.models import BenchmarkDataset
from benchmark.execution.engine import (
    CancellationToken,
    InferenceEngine,
    InferenceRunner,
    JsonlResponseWriter,
)
from benchmark.execution.metadata import (
    CompatibilityAssessment,
    ModelInspection,
    ModelStatus,
    ModelVerification,
    format_bytes,
    format_pull_event,
)
from benchmark.execution.registry import ModelManager
from benchmark.models.hardware import HardwareSummary
from benchmark.models.inference import InferenceProfile, PromptSpec, ResponseRecord, RunSummary
from benchmark.models.ollama import OllamaPullEvent
from benchmark.monitoring.sampler import HardwareSampler


def _add_config_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Configuration directory; defaults to BENCHMARK_CONFIG_DIR or ./configs.",
    )


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Local LLM benchmarking application and Ollama model management.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = (
        ("init", "Create runtime directories and initialize logging."),
        ("health", "Validate the environment, configuration, runtime directories, and Ollama."),
    )
    for command, help_text in commands:
        subparser = subparsers.add_parser(command, help=help_text)
        _add_config_dir(subparser)
    models_parser = subparsers.add_parser("models", help="Inspect and manage Ollama models.")
    model_commands = models_parser.add_subparsers(dest="models_command", required=True)

    list_parser = model_commands.add_parser("list", help="List configured and installed models.")
    _add_config_dir(list_parser)
    list_parser.add_argument("--all", action="store_true", help="Include unconfigured models.")
    _add_json(list_parser)

    show_parser = model_commands.add_parser("show", help="Show metadata for one configured model.")
    _add_config_dir(show_parser)
    show_parser.add_argument("tag", help="Model tag, display name, or role keyword.")
    show_parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Show configuration details when the model is not installed.",
    )
    _add_json(show_parser)

    verify_parser = model_commands.add_parser("verify", help="Verify digests and quantization.")
    _add_config_dir(verify_parser)
    verify_parser.add_argument(
        "tag",
        nargs="?",
        default=None,
        help="Optional model tag; defaults to the full configured roster.",
    )
    verify_parser.add_argument(
        "--require-installed",
        action="store_true",
        help="Fail if any selected model is missing.",
    )
    _add_json(verify_parser)

    pull_parser = model_commands.add_parser("pull", help="Pull one model after explicit approval.")
    _add_config_dir(pull_parser)
    pull_parser.add_argument("tag", help="Model tag, display name, or role keyword.")
    pull_parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the download explicitly.",
    )
    _add_json(pull_parser)

    test_parser = model_commands.add_parser("test", help="Run a short inference test.")
    _add_config_dir(test_parser)
    test_parser.add_argument("tag", help="Model tag, display name, or role keyword.")
    test_parser.add_argument(
        "--prompt",
        default="Reply with the single word OK.",
        help="Prompt used for the test request.",
    )
    test_parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Maximum generated tokens.",
    )
    test_parser.add_argument(
        "--unload",
        action="store_true",
        help="Unload the model after the test.",
    )
    _add_json(test_parser)

    load_parser = model_commands.add_parser("load", help="Load a model and confirm /api/ps.")
    _add_config_dir(load_parser)
    load_parser.add_argument("tag", help="Model tag, display name, or role keyword.")
    load_parser.add_argument(
        "--keep-alive",
        default=None,
        help="Override the configured keep_alive value.",
    )
    _add_json(load_parser)

    unload_parser = model_commands.add_parser("unload", help="Unload a model and confirm /api/ps.")
    _add_config_dir(unload_parser)
    unload_parser.add_argument("tag", help="Model tag, display name, or role keyword.")
    _add_json(unload_parser)

    ps_parser = model_commands.add_parser("ps", help="List models currently loaded in Ollama.")
    _add_config_dir(ps_parser)
    _add_json(ps_parser)

    snapshot_parser = model_commands.add_parser(
        "snapshot", help="Write a metadata snapshot for the configured roster."
    )
    _add_config_dir(snapshot_parser)
    snapshot_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Snapshot path; defaults to storage/model_metadata.json.",
    )
    snapshot_parser.add_argument(
        "--all",
        action="store_true",
        help="Include unconfigured installed models.",
    )
    _add_json(snapshot_parser)

    select_parser = model_commands.add_parser(
        "select", help="Resolve a tag, name, or role to a configured model."
    )
    _add_config_dir(select_parser)
    select_parser.add_argument("identifier", help="Model tag, display name, or role keyword.")
    _add_json(select_parser)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run streaming inference over selected dataset items and log response records.",
    )
    _add_config_dir(infer_parser)
    infer_parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Model tag, display name, or role keyword; repeatable. Defaults to the first candidate.",
    )
    infer_parser.add_argument(
        "--item",
        action="append",
        dest="items",
        default=None,
        help="Dataset item ID to run; repeatable. Overrides ordered selection.",
    )
    selection = infer_parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of ordered dataset items; defaults to 5.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run every dataset item.",
    )
    infer_parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Measured repetitions per item; defaults to 1.",
    )
    infer_parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Warm-up requests per model; defaults to the configured value.",
    )
    infer_parser.add_argument(
        "--profile",
        choices=[profile.value for profile in InferenceProfile],
        default=InferenceProfile.SMOKE.value,
        help="Inference profile label stored with each record.",
    )
    infer_parser.add_argument(
        "--cold-start",
        action="store_true",
        help="Unload before each measured request and use keep_alive=0.",
    )
    infer_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional lower output cap; never exceeds the item or configured cap.",
    )
    infer_parser.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help="Seconds to wait between models; defaults to 0 for smoke runs.",
    )
    infer_parser.add_argument(
        "--no-unload",
        action="store_true",
        help="Leave the model loaded after its run.",
    )
    infer_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output path; defaults to data/results/inference_<experiment>.jsonl.",
    )
    infer_parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable hardware sampling for this run.",
    )
    infer_parser.add_argument(
        "--hardware-interval-ms",
        type=int,
        default=None,
        help="Override hardware sampling interval; allowed range is 250-1000 ms.",
    )
    infer_parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=None,
        help="Override the pre-request hardware baseline duration.",
    )
    _add_json(infer_parser)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Compute stored-result analytics and workload recommendations.",
    )
    _add_config_dir(analyze_parser)
    analyze_parser.add_argument("experiment_id", help="Experiment identifier.")
    profile_group = analyze_parser.add_mutually_exclusive_group()
    profile_group.add_argument(
        "--profile",
        default=None,
        help="Scoring profile name; defaults to the configured default.",
    )
    profile_group.add_argument(
        "--all-profiles",
        action="store_true",
        help="Compute every configured workload profile.",
    )
    analyze_parser.add_argument(
        "--run-profile",
        default=None,
        help="Only include stored runs with this inference profile.",
    )
    analyze_parser.add_argument(
        "--include-cold-start",
        action="store_true",
        help="Include cold-start runs; excluded by default.",
    )
    analyze_parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke runs; excluded by default.",
    )
    analyze_parser.add_argument(
        "--persist",
        action="store_true",
        help="Store available recommendations in the database.",
    )
    _add_json(analyze_parser)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare analytics between a baseline and one or more experiments.",
    )
    _add_config_dir(compare_parser)
    compare_parser.add_argument(
        "baseline_experiment_id",
        help="Baseline experiment identifier.",
    )
    compare_parser.add_argument(
        "candidate_experiment_ids",
        nargs="+",
        help="One or more candidate experiment identifiers.",
    )
    compare_parser.add_argument(
        "--profile",
        default=None,
        help="Scoring profile name; defaults to the configured default.",
    )
    compare_parser.add_argument(
        "--run-profile",
        default=None,
        help="Only include stored runs with this inference profile.",
    )
    compare_parser.add_argument(
        "--include-cold-start",
        action="store_true",
        help="Include cold-start runs; excluded by default.",
    )
    compare_parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke runs; excluded by default.",
    )
    _add_json(compare_parser)

    report_parser = subparsers.add_parser(
        "report",
        help="Generate CSV, JSON, HTML, and PDF reports for a completed experiment.",
    )
    _add_config_dir(report_parser)
    report_parser.add_argument(
        "experiment_id",
        help="Experiment identifier to report.",
    )
    report_parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Database path; defaults to storage/benchmark.db.",
    )
    report_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to data/exports/<experiment_id>.",
    )
    report_parser.add_argument(
        "--format",
        action="append",
        dest="formats",
        type=str.lower,
        choices=("csv", "json", "html", "pdf"),
        default=None,
        help="Report format to generate; repeatable. Defaults to all four.",
    )
    report_parser.add_argument(
        "--profile",
        default=None,
        help="Scoring profile name; defaults to the configured default.",
    )
    report_parser.add_argument(
        "--run-profile",
        default=None,
        help="Only include stored runs with this inference profile.",
    )
    report_parser.add_argument(
        "--include-cold-start",
        action="store_true",
        help="Include cold-start runs; excluded by default.",
    )
    report_parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke runs; excluded by default.",
    )
    report_parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Export an experiment that is not completed, flagged as preliminary.",
    )
    _add_json(report_parser)

    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Sample CPU, RAM, GPU, VRAM, and temperature metrics.",
    )
    _add_config_dir(monitor_parser)
    monitor_parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Sampling duration in seconds; defaults to 5.",
    )
    monitor_parser.add_argument(
        "--interval-ms",
        type=int,
        default=None,
        help="Override hardware sampling interval; allowed range is 250-1000 ms.",
    )
    _add_json(monitor_parser)
    return parser


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _create_model_manager(settings: AppSettings) -> ModelManager:
    return ModelManager(settings)


def _status_line(status: ModelStatus) -> str:
    state = "installed" if status.installed else "missing"
    digest = f"{status.digest[:12]}..." if status.digest else "no digest"
    return (
        f"[{state:9}] {status.name} ({status.role.value}) {status.tag} | "
        f"{format_bytes(status.size_bytes)} | {status.quantization or 'quant unknown'} | {digest}"
    )


def _verification_payload(verification: ModelVerification) -> dict[str, Any]:
    return {
        **verification.model_dump(mode="json"),
        "ok": verification.ok,
    }


def _verification_line(verification: ModelVerification) -> str:
    if verification.ok:
        return (
            f"[OK]   {verification.name} ({verification.role.value}) "
            f"{verification.tag}: verified"
        )
    detail = "; ".join(verification.problems) or "verification failed"
    return f"[FAIL] {verification.name} ({verification.role.value}) {verification.tag}: {detail}"


def _compatibility_line(assessment: CompatibilityAssessment) -> str:
    labels = {
        "full_gpu_possible": "full GPU residency possible (heuristic)",
        "partial_cpu_offload": "partial CPU offload likely (heuristic)",
        "insufficient_memory": "insufficient target memory (heuristic)",
        "unknown": "unknown",
    }
    parts = [labels[assessment.fit]]
    if assessment.estimated_total_bytes is not None:
        parts.append(f"estimated total {format_bytes(assessment.estimated_total_bytes)}")
    if assessment.estimated_kv_cache_bytes is not None:
        parts.append(f"KV cache {format_bytes(assessment.estimated_kv_cache_bytes)}")
    return "; ".join(parts)


def _short_license(value: str | None) -> str:
    if not value:
        return "unknown"
    first_line = value.strip().splitlines()[0]
    return first_line if len(first_line) <= 120 else f"{first_line[:117]}..."


def _inspection_text(inspection: ModelInspection) -> str:
    metadata = inspection.metadata
    verification = inspection.verification
    lines = [
        f"Name: {metadata.name}",
        f"Role: {metadata.role.value}",
        f"Tag: {metadata.tag}",
        f"Installed: {'yes' if metadata.installed else 'no'} ({metadata.source.value})",
        f"Digest: {metadata.digest or 'unknown'}",
        f"Size: {format_bytes(metadata.size_bytes)}",
        f"Family: {metadata.family or 'unknown'}",
        f"Parameters: {metadata.parameter_size or metadata.parameter_count or 'unknown'}",
        f"Quantization: {metadata.quantization or 'unknown'}",
        f"Context length: {metadata.context_length or 'unknown'}",
        f"Embedding length: {metadata.embedding_length or 'unknown'}",
        f"Capabilities: {', '.join(metadata.capabilities) or 'unknown'}",
        f"License: {_short_license(metadata.license)}",
        f"Ollama version: {metadata.ollama_version}",
        f"Verification: {'OK' if verification.ok else 'FAIL'}",
        f"Compatibility: {_compatibility_line(inspection.compatibility)}",
    ]
    if verification.problems:
        lines.append(f"Problems: {'; '.join(verification.problems)}")
    lines.extend(f"Note: {note}" for note in inspection.compatibility.notes)
    return "\n".join(lines)


def _progress(event: OllamaPullEvent) -> None:
    print(format_pull_event(event), flush=True)


def _new_experiment_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"


def _select_prompts(
    dataset: BenchmarkDataset,
    args: argparse.Namespace,
    settings: AppSettings,
) -> list[PromptSpec]:
    if args.limit is not None and args.limit < 1:
        raise BenchmarkError("--limit must be at least 1")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise BenchmarkError("--max-tokens must be at least 1")
    by_id = {item.id: item for item in dataset.items}
    if args.items:
        selected = []
        seen: set[str] = set()
        for item_id in args.items:
            if item_id in seen:
                raise BenchmarkError(f"Duplicate dataset item selected: {item_id}")
            seen.add(item_id)
            item = by_id.get(item_id)
            if item is None:
                raise BenchmarkError(f"Unknown dataset item: {item_id}")
            selected.append(item)
    else:
        selected = ordered_items(dataset, dataset.random_seed)
        if not args.all:
            limit = args.limit if args.limit is not None else 5
            selected = selected[:limit]
    prompts: list[PromptSpec] = []
    for item in selected:
        max_tokens = item.maximum_output_tokens
        if args.max_tokens is not None:
            max_tokens = min(max_tokens, args.max_tokens)
        max_tokens = min(max_tokens, settings.benchmark.num_predict)
        prompts.append(
            PromptSpec(
                item_id=item.id,
                prompt=item.prompt,
                system_instruction=item.system_instruction,
                maximum_output_tokens=max_tokens,
                category=item.category.value,
            )
        )
    return prompts


def _optional_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def _optional_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} tok/s"


def _response_line(record: ResponseRecord) -> str:
    marker = "warmup " if record.is_warmup else ""
    return (
        f"[{record.status.value:9}] {marker}{record.model_tag} "
        f"{record.item_id or '-'} r{record.repetition} | "
        f"ttft={_optional_seconds(record.timing.ttft_seconds)} "
        f"total={_optional_seconds(record.timing.total_latency_seconds)} "
        f"output={_optional_rate(record.tokens.output_tokens_per_second)} "
        f"chunks={record.stream_chunk_count}"
    )


def _summary_text(summary: RunSummary) -> str:
    return "\n".join(
        [
            f"Experiment: {summary.experiment_id}",
            f"Profile: {summary.profile.value} | status: {summary.status.value}",
            f"Models: {', '.join(summary.model_tags)}",
            (
                f"Requests: total={summary.total_requests} "
                f"measured={summary.measured_requests} warmup={summary.warmup_count}"
            ),
            (
                f"Measured status: success={summary.success_count} "
                f"empty={summary.empty_count} timeout={summary.timeout_count} "
                f"error={summary.error_count} cancelled={summary.cancelled_count}"
            ),
            f"Output: {summary.output_path or 'not written'}",
        ]
    )


def _optional_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _optional_bytes(value: int | None) -> str:
    return "n/a" if value is None else format_bytes(value)


def _optional_celsius(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} C"


def _optional_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f} ms"


def _hardware_summary_text(summary: HardwareSummary) -> str:
    inference = summary.inference
    gpu_source = summary.gpu_source.value if summary.gpu_source else "unavailable"
    lines = [
        (
            f"Hardware: {summary.sample_count} samples "
            f"({summary.baseline.sample_count} baseline, "
            f"{inference.sample_count} inference) | "
            f"interval {summary.sampling_interval_ms} ms"
        ),
        (
            f"CPU: avg {_optional_percent(inference.cpu_avg_percent)} | "
            f"peak {_optional_percent(inference.cpu_peak_percent)}"
        ),
        (
            f"RAM: avg {_optional_bytes(inference.ram_avg_used_bytes)} | "
            f"peak {_optional_bytes(inference.ram_peak_used_bytes)} | "
            f"total {_optional_bytes(summary.ram_total_bytes)}"
        ),
        (
            f"Ollama RSS: avg {_optional_bytes(inference.ollama_rss_avg_bytes)} | "
            f"peak {_optional_bytes(inference.ollama_rss_peak_bytes)} | "
            f"processes {', '.join(summary.ollama_process_names) or 'not found'}"
        ),
        (
            f"GPU: {summary.gpu_name or 'unavailable'} via {gpu_source} | "
            f"util avg {_optional_percent(inference.gpu_utilization_avg_percent)} | "
            f"peak {_optional_percent(inference.gpu_utilization_peak_percent)}"
        ),
        (
            f"VRAM: avg {_optional_bytes(inference.vram_avg_used_bytes)} | "
            f"peak {_optional_bytes(inference.vram_peak_used_bytes)} | "
            f"total {_optional_bytes(summary.vram_total_bytes)}"
        ),
        (
            "Ollama model VRAM: "
            + (
                f"{_optional_bytes(summary.ollama_model_size_vram_bytes)} / "
                f"{_optional_bytes(summary.ollama_model_size_bytes)}"
                + (
                    f" ({summary.ollama_model_vram_ratio * 100:.1f}%)"
                    if summary.ollama_model_vram_ratio is not None
                    else ""
                )
                if summary.ollama_model_vram_available
                else "not recorded"
            )
        ),
        (
            f"GPU temperature: avg {_optional_celsius(inference.gpu_temperature_avg_c)} | "
            f"peak {_optional_celsius(inference.gpu_temperature_peak_c)}"
        ),
        (
            f"Sampler overhead: avg {_optional_ms(summary.sampler_overhead_avg_ms)} | "
            f"peak {_optional_ms(summary.sampler_overhead_peak_ms)}"
        ),
        f"Attribution: {summary.attribution_note}",
    ]
    if summary.gpu_error:
        lines.append(f"GPU note: {summary.gpu_error}")
    if summary.ollama_model_vram_error:
        lines.append(f"Ollama model VRAM note: {summary.ollama_model_vram_error}")
    if inference.low_sample_count:
        lines.append(
            "Sample note: the inference window has fewer than two samples; "
            "averages and peaks are descriptive only."
        )
    if summary.sampler_error:
        lines.append(f"Sampler note: {summary.sampler_error}")
    return "\n".join(lines)


def _make_hardware_factory(
    args: argparse.Namespace,
    settings: AppSettings,
) -> Any:
    if args.no_monitor or not settings.benchmark.hardware_monitoring_enabled:
        return lambda: None
    interval_ms = (
        args.hardware_interval_ms
        if args.hardware_interval_ms is not None
        else settings.benchmark.hardware_sample_interval_ms
    )
    baseline = (
        args.baseline_seconds
        if args.baseline_seconds is not None
        else settings.benchmark.hardware_baseline_seconds
    )
    if not 250 <= interval_ms <= 1000:
        raise BenchmarkError("--hardware-interval-ms must be between 250 and 1000")
    if not 0.0 <= baseline <= 10.0:
        raise BenchmarkError("--baseline-seconds must be between 0 and 10")
    return lambda: HardwareSampler(
        interval_seconds=interval_ms / 1000,
        baseline_seconds=baseline,
    )


def _handle_infer(args: argparse.Namespace, settings: AppSettings) -> int:
    if args.repeats < 1:
        raise BenchmarkError("--repeats must be at least 1")
    if args.warmup is not None and args.warmup < 0:
        raise BenchmarkError("--warmup must not be negative")
    if args.cooldown < 0:
        raise BenchmarkError("--cooldown must not be negative")
    dataset = load_dataset(settings.benchmark.dataset_path)
    prompts = _select_prompts(dataset, args, settings)
    manager = _create_model_manager(settings)
    try:
        model_tags: list[str] = []
        identifiers = args.models or [settings.models.candidates[0].tag]
        for identifier in identifiers:
            registered = manager.resolve(identifier)
            if registered.spec.tag in model_tags:
                raise BenchmarkError(f"Duplicate model selected: {registered.spec.tag}")
            model_tags.append(registered.spec.tag)
        experiment_id = _new_experiment_id()
        output = args.output or (
            settings.benchmark.results_dir / f"inference_{experiment_id}.jsonl"
        )
        engine = InferenceEngine(
            settings,
            manager.client,
            hardware_factory=_make_hardware_factory(args, settings),
        )
        runner = InferenceRunner(
            engine,
            JsonlResponseWriter(output),
            manager=manager,
            cancellation=CancellationToken(),
        )
        warmup = (
            settings.benchmark.warmup_requests if args.warmup is None else args.warmup
        )
        result = runner.run(
            model_tags,
            prompts,
            repetitions=args.repeats,
            warmup_requests=warmup,
            profile=args.profile,
            experiment_id=experiment_id,
            cold_start=args.cold_start,
            unload_between_models=not args.no_unload,
            cooldown_seconds=args.cooldown,
            progress=None if args.json else lambda record: print(
                _response_line(record),
                flush=True,
            ),
        )
        if args.json:
            _print_json(result.summary.model_dump(mode="json"))
        else:
            print(_summary_text(result.summary))
        return 130 if result.summary.status.value == "cancelled" else 0
    finally:
        manager.close()


def _handle_monitor(args: argparse.Namespace, settings: AppSettings) -> int:
    if args.seconds <= 0:
        raise BenchmarkError("--seconds must be greater than zero")
    interval_ms = (
        args.interval_ms
        if args.interval_ms is not None
        else settings.benchmark.hardware_sample_interval_ms
    )
    if not 250 <= interval_ms <= 1000:
        raise BenchmarkError("--interval-ms must be between 250 and 1000")
    sampler = HardwareSampler(
        interval_seconds=interval_ms / 1000,
        baseline_seconds=0.0,
    )
    sampler.start()
    sampler.mark_request_start()
    if not args.json:
        print(f"Sampling hardware for {args.seconds:.1f}s...", flush=True)
    try:
        time.sleep(args.seconds)
    finally:
        summary = sampler.stop()
    if args.json:
        _print_json(summary.model_dump(mode="json"))
    else:
        print(_hardware_summary_text(summary))
    return 0


def _optional_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _optional_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _analytics_text(analysis: ExperimentAnalytics) -> str:
    lines = [
        f"Experiment: {analysis.experiment_id} ({analysis.experiment_status})",
        (
            f"Profile: {analysis.profile} | responses measured="
            f"{analysis.measured_responses} evaluated={analysis.evaluated_responses}"
        ),
        "Weights: "
        + ", ".join(
            f"{name}={weight:g}" for name, weight in analysis.weights.items()
        ),
    ]
    for model in analysis.models:
        statistics = model.statistics
        lines.append(
            f"  {model.model_tag} | composite={_optional_score(model.composite_score)} "
            f"quality={_optional_score(model.quality_score)} "
            f"P95={_optional_seconds(statistics.latency.p95)} "
            f"throughput={_optional_rate(statistics.output_tokens_per_second.mean)} "
            f"success={_optional_ratio(statistics.success_rate)} "
            f"failure={_optional_ratio(statistics.failure_rate)} "
            f"evaluated={model.quality.evaluated_responses}/"
            f"{statistics.measured_responses}"
        )
    if analysis.limitations:
        lines.extend(f"Limitation: {item}" for item in analysis.limitations)
    return "\n".join(lines)


def _recommendation_text(result: RecommendationSet) -> str:
    lines = [f"Recommendation profile: {result.workload}"]
    if not result.available:
        lines.append(f"  unavailable: {result.reason or 'insufficient data'}")
        return "\n".join(lines)
    for item in result.recommendations:
        tie = f" | tied with {', '.join(item.tied_with)}" if item.tied else ""
        lines.append(
            f"  {item.rank}. {item.model_tag} | score={item.composite_score:.2f} "
            f"| confidence={item.confidence}{tie}"
        )
        lines.append(f"     {item.rationale}")
        if item.strengths:
            lines.append(f"     Strengths: {'; '.join(item.strengths)}")
        if item.weaknesses:
            lines.append(f"     Weaknesses: {'; '.join(item.weaknesses)}")
        if item.limitations:
            lines.append(f"     Limitations: {'; '.join(item.limitations)}")
    return "\n".join(lines)


def _handle_analyze(args: argparse.Namespace, settings: AppSettings) -> int:
    database = Database.from_settings(settings)
    database.initialize()
    try:
        engine = AnalyticsEngine(database, settings=settings.benchmark)
        names = (
            list(engine.profile_names)
            if args.all_profiles
            else [args.profile or engine.default_profile]
        )
        results: list[tuple[ExperimentAnalytics, RecommendationSet]] = []
        for name in names:
            analysis = engine.analyze(
                args.experiment_id,
                profile=name,
                run_profile=args.run_profile,
                include_cold_start=args.include_cold_start,
                include_smoke=args.include_smoke,
            )
            results.append((analysis, build_recommendation_set(analysis)))
        if args.persist:
            engine.persist_recommendations(
                args.experiment_id,
                [result for _, result in results],
            )
        if args.json:
            _print_json(
                {
                    "experiment_id": args.experiment_id,
                    "profiles": [
                        {
                            "analytics": analysis.model_dump(mode="json"),
                            "recommendation": result.model_dump(mode="json"),
                        }
                        for analysis, result in results
                    ],
                }
            )
        else:
            for analysis, result in results:
                print(_analytics_text(analysis))
                print(_recommendation_text(result))
                print()
        return 0
    finally:
        database.dispose()


def _compare_text(comparison: Any) -> str:
    lines = [f"Baseline: {comparison.baseline_experiment_id}"]
    if comparison.limitations:
        lines.extend(f"Limitation: {item}" for item in comparison.limitations)
    for candidate in comparison.comparisons:
        lines.append(
            f"Candidate: {candidate.candidate_experiment_id} "
            f"(profile {candidate.candidate_profile or 'unavailable'})"
        )
        if candidate.limitations:
            lines.extend(f"  Limitation: {item}" for item in candidate.limitations)
        for model in candidate.models:
            lines.append(f"  {model.model_tag}:")
            for metric in model.metrics:
                if not metric.available:
                    lines.append(f"    {metric.metric}: unavailable")
                    continue
                percent = (
                    f", {metric.percent_change:+.1f}%"
                    if metric.percent_change is not None
                    else ""
                )
                lines.append(
                    f"    {metric.metric}: {metric.baseline:.4g} -> "
                    f"{metric.candidate:.4g} ({metric.delta:+.4g}{percent})"
                )
    return "\n".join(lines)


def _handle_compare(args: argparse.Namespace, settings: AppSettings) -> int:
    database = Database.from_settings(settings)
    database.initialize()
    try:
        engine = AnalyticsEngine(database, settings=settings.benchmark)
        comparison = engine.compare(
            args.baseline_experiment_id,
            args.candidate_experiment_ids,
            profile=args.profile,
            run_profile=args.run_profile,
            include_cold_start=args.include_cold_start,
            include_smoke=args.include_smoke,
        )
        if args.json:
            _print_json(comparison.model_dump(mode="json"))
        else:
            print(_compare_text(comparison))
        return 0
    finally:
        database.dispose()


def _report_text(result: Any) -> str:
    lines = [
        f"Experiment: {result.experiment_id}",
        f"Output directory: {result.output_dir}",
        f"Preliminary: {'yes' if result.preliminary else 'no'}",
    ]
    if result.page_count is not None:
        lines.append(f"PDF pages: {result.page_count}")
    if result.warnings:
        lines.extend(f"Warning: {item}" for item in result.warnings)
    lines.append("Files:")
    for generated in result.files:
        lines.append(
            f"  {generated.name} ({generated.format}, "
            f"{format_bytes(generated.size_bytes)}, sha256 {generated.sha256[:12]}...)"
        )
    lines.append(f"Manifest: {result.manifest_path}")
    return "\n".join(lines)


def _handle_report(args: argparse.Namespace, settings: AppSettings) -> int:
    from benchmark.reporting import DEFAULT_FORMATS, generate_reports

    database = (
        Database.from_settings(settings)
        if args.database is None
        else Database(args.database)
    )
    database.initialize()
    try:
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
            _print_json(result.as_dict())
        else:
            print(_report_text(result))
        return 0
    finally:
        database.dispose()


def _handle_models(args: argparse.Namespace, settings: AppSettings) -> int:
    manager = _create_model_manager(settings)
    try:
        if args.models_command == "list":
            statuses = manager.inventory(include_all=args.all)
            if args.json:
                _print_json([status.model_dump(mode="json") for status in statuses])
            else:
                print("Configured and installed models:")
                for status in statuses:
                    print(_status_line(status))
            return 0
        if args.models_command == "show":
            inspection = manager.inspect(
                args.tag,
                require_installed=not args.allow_missing,
            )
            if args.json:
                _print_json(
                    {
                        **inspection.model_dump(mode="json"),
                        "verification_ok": inspection.verification.ok,
                    }
                )
            else:
                print(_inspection_text(inspection))
            return 0 if inspection.metadata.installed else 1
        if args.models_command == "verify":
            if args.tag:
                verifications = [
                    manager.verify(args.tag, require_installed=args.require_installed)
                ]
            else:
                verifications = manager.verify_all(require_installed=args.require_installed)
            if args.json:
                _print_json([_verification_payload(item) for item in verifications])
            else:
                for verification in verifications:
                    print(_verification_line(verification))
            return 0 if all(item.ok for item in verifications) else 1
        if args.models_command == "pull":
            metadata = manager.pull(
                args.tag,
                approved=args.yes,
                progress=None if args.json else _progress,
            )
            if args.json:
                _print_json(metadata.model_dump(mode="json"))
            else:
                print(f"Model ready: {metadata.tag}")
                print(f"Digest: {metadata.digest or 'unknown'}")
                print(f"Size: {format_bytes(metadata.size_bytes)}")
                print(f"Quantization: {metadata.quantization or 'unknown'}")
            return 0
        if args.models_command == "test":
            response = manager.test_inference(
                args.tag,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                unload_after=args.unload,
            )
            if args.json:
                _print_json(response.model_dump(mode="json"))
            else:
                print(response.message.content)
                print(f"eval_count={response.eval_count or 0}")
                if response.eval_duration:
                    tokens_per_second = (response.eval_count or 0) / (
                        response.eval_duration / 1_000_000_000
                    )
                    print(f"eval_tokens_per_second={tokens_per_second:.2f}")
            return 0
        if args.models_command == "load":
            loaded = manager.load(args.tag, keep_alive=args.keep_alive)
            if args.json:
                _print_json(loaded.model_dump(mode="json"))
            else:
                print(f"Loaded: {loaded.tag}")
                print(f"Size: {format_bytes(loaded.size)}")
                print(f"VRAM: {format_bytes(loaded.size_vram)}")
                print(f"Context length: {loaded.context_length or 'unknown'}")
            return 0
        if args.models_command == "unload":
            manager.unload(args.tag)
            if args.json:
                _print_json({"unloaded": args.tag})
            else:
                print(f"Unloaded and absent from /api/ps: {args.tag}")
            return 0
        if args.models_command == "ps":
            models = manager.client.loaded_models()
            if args.json:
                _print_json(
                    [
                        {"tag": model.tag, **model.model_dump(mode="json")}
                        for model in models
                    ]
                )
            else:
                if not models:
                    print("No models are currently loaded.")
                for model in models:
                    print(
                        f"{model.tag} | size {format_bytes(model.size)} | "
                        f"VRAM {format_bytes(model.size_vram)} | "
                        f"context {model.context_length or 'unknown'}"
                    )
            return 0
        if args.models_command == "snapshot":
            snapshot, path = manager.snapshot(include_all=args.all, output=args.output)
            if args.json:
                _print_json(
                    {
                        "path": str(path),
                        "snapshot": snapshot.model_dump(mode="json"),
                    }
                )
            else:
                print(f"Wrote metadata snapshot: {path}")
                print(f"Models: {len(snapshot.models)}")
                print(f"Loaded models: {len(snapshot.loaded_models)}")
            return 0
        if args.models_command == "select":
            status = manager.select(args.identifier)
            if args.json:
                _print_json(status.model_dump(mode="json"))
            else:
                print(_status_line(status))
            return 0
        raise BenchmarkError(f"Unknown models command: {args.models_command}")
    finally:
        manager.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(config_dir=args.config_dir)
        initialize_application(settings)
        if args.command == "init":
            print(f"Initialized project at {settings.project_root}")
            print(f"Configuration: {settings.config_dir}")
            print(f"Runtime storage: {settings.benchmark.storage_dir}")
            return 0
        if args.command == "health":
            report = run_health_check(settings=settings)
            print(render_health_report(report))
            return 0 if report.ok else 1
        if args.command == "models":
            return _handle_models(args, settings)
        if args.command == "monitor":
            return _handle_monitor(args, settings)
        if args.command == "infer":
            return _handle_infer(args, settings)
        if args.command == "analyze":
            return _handle_analyze(args, settings)
        if args.command == "compare":
            return _handle_compare(args, settings)
        if args.command == "report":
            return _handle_report(args, settings)
        raise BenchmarkError(f"Unknown command: {args.command}")
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
