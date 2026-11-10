"""
File: __main__.py

Description:
Command-line entry point for initialization, health checks, and model management.

Responsibilities:
- Parse init, health, and models commands.
- Load and validate configuration.
- Initialize runtime directories and logging.
- Manage Ollama models without starting a benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from benchmark.core.config import AppSettings, load_settings
from benchmark.core.exceptions import BenchmarkError
from benchmark.core.health import render_health_report, run_health_check
from benchmark.core.startup import initialize_application
from benchmark.execution.metadata import (
    CompatibilityAssessment,
    ModelInspection,
    ModelStatus,
    ModelVerification,
    format_bytes,
    format_pull_event,
)
from benchmark.execution.registry import ModelManager
from benchmark.models.ollama import OllamaPullEvent


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
        raise BenchmarkError(f"Unknown command: {args.command}")
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
