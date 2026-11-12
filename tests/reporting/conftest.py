"""
File: conftest.py

Description:
Fixtures for reporting tests: an isolated project, a disposable database, and
seeded completed and incomplete experiments.

Responsibilities:
- Copy the real configuration and canonical dataset into a temporary project.
- Seed deterministic responses, evaluations, and hardware summaries.
- Provide completed, sparse (missing metrics), and incomplete experiments.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from benchmark.core.config import AppSettings, load_settings
from benchmark.database.db import Database
from benchmark.database.repositories import Repositories
from benchmark.dataset.loader import load_dataset
from benchmark.dataset.models import BenchmarkDataset, DatasetItem
from benchmark.evaluation.models import (
    CheckResult,
    EvaluationResult,
    EvaluationStatus,
    JudgeDimensionScore,
    JudgeResult,
    JudgeStatus,
)
from benchmark.execution.metadata import ModelRole, build_missing_metadata
from benchmark.models.hardware import HardwareSummary, WindowSummary
from benchmark.models.inference import (
    GenerationOptions,
    InferenceProfile,
    ResponseRecord,
    ResponseStatus,
    TimingMetrics,
    TokenMetrics,
)

MIB = 1024 * 1024
PLAN_CATEGORIES = (
    "general_knowledge",
    "coding",
    "structured_json",
    "instruction_following",
)


@dataclass
class ReportingContext:
    """Paths and loaded objects for an isolated reporting test project."""

    settings: AppSettings
    database: Database
    dataset: BenchmarkDataset
    project_root: Path


def build_context(env: "ReportingEnvironment") -> ReportingContext:
    """Build a reporting context for an isolated environment."""
    settings = load_settings(config_dir=env.config_dir)
    database = Database.from_settings(settings)
    database.initialize()
    dataset = load_dataset(settings.benchmark.dataset_path)
    return ReportingContext(
        settings=settings,
        database=database,
        dataset=dataset,
        project_root=env.project_root,
    )


@dataclass
class ReportingEnvironment:
    """Paths for an isolated reporting project."""

    config_dir: Path
    project_root: Path
    dataset_path: Path
    database_path: Path


@pytest.fixture()
def reporting_env(tmp_path: Path, project_root: Path) -> ReportingEnvironment:
    config_dir = tmp_path / "configs"
    shutil.copytree(project_root / "configs", config_dir)
    dataset_dir = tmp_path / "data" / "datasets"
    dataset_dir.mkdir(parents=True)
    dataset_path = dataset_dir / "benchmark_v1.json"
    shutil.copy2(project_root / "data" / "datasets" / "benchmark_v1.json", dataset_path)
    benchmark_path = config_dir / "benchmark.yaml"
    payload = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "dataset_path": str(dataset_path),
            "storage_dir": str(tmp_path / "storage"),
            "results_dir": str(tmp_path / "data" / "results"),
            "exports_dir": str(tmp_path / "data" / "exports"),
            "log_dir": str(tmp_path / "storage" / "logs"),
        }
    )
    benchmark_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return ReportingEnvironment(
        config_dir=config_dir,
        project_root=tmp_path,
        dataset_path=dataset_path,
        database_path=tmp_path / "storage" / "benchmark.db",
    )


@pytest.fixture()
def reporting_context(reporting_env: ReportingEnvironment):
    context = build_context(reporting_env)
    yield context
    context.database.dispose()


def _hardware(*, cpu: float, gpu: float, temperature: float) -> HardwareSummary:
    window = WindowSummary(
        sample_count=3,
        duration_seconds=2.0,
        cpu_avg_percent=cpu,
        cpu_peak_percent=cpu + 15.0,
        ram_avg_used_bytes=4 * 1024 * MIB,
        ram_peak_used_bytes=5 * 1024 * MIB,
        gpu_utilization_avg_percent=gpu,
        gpu_utilization_peak_percent=gpu + 10.0,
        vram_avg_used_bytes=2 * 1024 * MIB,
        vram_peak_used_bytes=3 * 1024 * MIB,
        gpu_temperature_avg_c=temperature,
        gpu_temperature_peak_c=temperature + 5.0,
    )
    return HardwareSummary(
        sampling_interval_ms=1000,
        baseline_seconds=1.0,
        sample_count=3,
        baseline=window,
        inference=window,
        overall=window,
        ram_total_bytes=32 * 1024 * MIB,
        vram_total_bytes=8 * 1024 * MIB,
        gpu_name="Test GPU",
        gpu_source="nvml",
        gpu_available=True,
        gpu_utilization_available=True,
        gpu_memory_available=True,
        gpu_temperature_available=True,
        ollama_model_vram_available=True,
        ollama_model_size_vram_bytes=2 * 1024 * MIB,
        ollama_model_vram_ratio=0.25,
        ollama_process_available=True,
        ollama_process_names=["ollama.exe"],
    )


def _evaluation(
    item: DatasetItem,
    model_tag: str,
    score: float,
    *,
    judge_score: float | None,
) -> EvaluationResult:
    judge = None
    if judge_score is not None:
        raw = max(1, min(5, round(judge_score / 20)))
        judge = JudgeResult(
            status=JudgeStatus.SUCCESS,
            dimensions={
                name: JudgeDimensionScore(score=raw, rationale="Seeded judge score")
                for name in (
                    "relevance",
                    "completeness",
                    "coherence",
                    "reasoning_quality",
                )
            },
            raw_score=float(raw),
            normalized_score=judge_score,
            attempts=1,
        )
    passed = score >= 50.0
    return EvaluationResult(
        question_id=item.id,
        model_id=model_tag,
        category=item.category.value,
        evaluation_type=item.evaluation_type,
        evaluator=f"seeded-{item.evaluation_type.value}",
        evaluator_version="1.0.0",
        deterministic_score=score / 100.0,
        deterministic_max_score=1.0,
        deterministic_normalized_score=score,
        checks=[
            CheckResult(
                name=item.evaluation_type.value,
                passed=passed,
                score=1.0 if passed else 0.0,
                max_score=1.0,
                detail="Seeded check",
            )
        ],
        judge=judge,
        judge_eligible=judge_score is not None,
        judge_missing=judge_score is None,
        composite_score=score,
        passed=passed,
        status=EvaluationStatus.PASSED if passed else EvaluationStatus.FAILED,
    )


def _response(
    *,
    experiment_id: str,
    model_tag: str,
    item: DatasetItem,
    latency: float,
    ttft: float,
    response_text: str,
    repetition: int = 1,
    cold_start: bool = False,
    hardware: HardwareSummary | None = None,
) -> ResponseRecord:
    now = datetime.now(timezone.utc)
    return ResponseRecord(
        experiment_id=experiment_id,
        model_tag=model_tag,
        model_name=model_tag,
        item_id=item.id,
        category=item.category.value,
        profile=InferenceProfile.QUALITY,
        repetition=repetition,
        cold_start=cold_start,
        prompt=item.prompt,
        system_prompt=item.system_instruction,
        response_text=response_text,
        finish_reason="stop",
        status=ResponseStatus.SUCCESS,
        started_at=now,
        completed_at=now,
        generation_options=GenerationOptions(),
        timing=TimingMetrics(ttft_seconds=ttft, total_latency_seconds=latency),
        tokens=TokenMetrics(
            prompt_eval_count=40,
            eval_count=25,
            output_tokens_per_second=25.0 / latency,
            prompt_tokens_per_second=40.0 / ttft,
        ),
        hardware=hardware,
    )


def seed_experiment(
    context: ReportingContext,
    *,
    experiment_id: str = "exp-report-a",
    completed: bool = True,
    with_hardware: bool = True,
    with_evaluations: bool = True,
    latency_shift: float = 0.0,
    score_shift: float = 0.0,
) -> tuple[str, list[str]]:
    """Persist one deterministic experiment and return its id and model tags."""
    repos = Repositories(context.database)
    dataset_id = repos.datasets.register(
        context.dataset, context.settings.benchmark.dataset_path
    )
    repos.experiments.create(
        experiment_id=experiment_id,
        name=experiment_id,
        dataset_id=dataset_id,
        config_hash="a" * 64,
        config={"benchmark": "reporting-test"},
        host={"platform": "test", "gpu": "Test GPU"},
        software={"python": "3.13"},
    )
    specs = context.settings.models.candidates[:3]
    registered = []
    for spec in specs:
        existing = repos.models.get_by_tag(spec.tag)
        registered.append(
            existing
            if existing is not None
            else repos.models.register(
                build_missing_metadata(spec, "0.40.15", ModelRole.CANDIDATE)
            )
        )
    by_category: dict[str, DatasetItem] = {}
    for item in context.dataset.items:
        by_category.setdefault(item.category.value, item)
    plan = [by_category[name] for name in PLAN_CATEGORIES]

    for index, model in enumerate(registered):
        run = repos.runs.create(
            experiment_external_id=experiment_id,
            model_id=model.id,
            profile="quality",
            repetitions=1,
        )
        for offset, item in enumerate(plan):
            latency = 1.2 + index * 0.6 + offset * 0.15 + latency_shift
            response_text = (
                f"{model.tag} <b>seeded</b> answer for {item.id} — café naïve 漢字"
                if index == 0
                else f"<b>{model.tag}</b> answer for {item.id} & more"
            )
            hardware = (
                _hardware(
                    cpu=20.0 + index * 18.0,
                    gpu=30.0 + index * 15.0,
                    temperature=40.0 + index * 4.0,
                )
                if with_hardware
                else None
            )
            response_id = repos.responses.save(
                _response(
                    experiment_id=experiment_id,
                    model_tag=model.tag,
                    item=item,
                    latency=latency,
                    ttft=0.15 + index * 0.04 + offset * 0.01,
                    response_text=response_text,
                    hardware=hardware,
                ),
                run.id,
            )
            if with_evaluations:
                score = 92.0 - index * 20.0 + offset + score_shift
                repos.evaluations.save(
                    _evaluation(
                        item,
                        model.tag,
                        score,
                        judge_score=score - 5.0,
                    ),
                    response_id,
                )
        repos.runs.mark_completed(run.id)

    if completed:
        repos.experiments.mark_completed(experiment_id)
    return experiment_id, [model.tag for model in registered]


@pytest.fixture()
def seed(reporting_context: ReportingContext):
    """Seed the shared reporting test database and return the result."""
    return lambda **kwargs: seed_experiment(reporting_context, **kwargs)


@pytest.fixture()
def completed_bundle(reporting_context: ReportingContext, seed):
    """Return a report bundle for one completed, fully populated experiment."""
    from benchmark.reporting import build_bundle

    experiment_id, tags = seed()
    bundle = build_bundle(
        reporting_context.database,
        experiment_id,
        settings=reporting_context.settings,
    )
    return bundle, tags
