"""
File: conftest.py

Description:
Fixtures for dashboard tests: isolated configuration, a disposable database, and
seeded experiments that mirror the rows produced by the orchestrator.

Responsibilities:
- Copy the real configuration and canonical dataset into a temporary project.
- Build a DashboardContext without touching the repository database.
- Seed deterministic responses, evaluations, and hardware summaries.
- Point the dashboard entry point at the temporary configuration.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from benchmark.core.config import load_settings
from benchmark.dashboard.data import DashboardContext
from benchmark.database.db import Database
from benchmark.database.repositories import Repositories
from benchmark.dataset.loader import load_dataset
from benchmark.dataset.models import DatasetItem
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
class DashboardEnv:
    """Paths for an isolated dashboard project."""

    config_dir: Path
    project_root: Path
    dataset_path: Path
    database_path: Path


def build_context(env: DashboardEnv) -> DashboardContext:
    """Build a dashboard context for an isolated environment."""
    settings = load_settings(config_dir=env.config_dir)
    database = Database.from_settings(settings)
    database.initialize()
    dataset = load_dataset(settings.benchmark.dataset_path)
    return DashboardContext(
        settings=settings,
        database=database,
        dataset=dataset,
        dataset_error=None,
    )


@pytest.fixture()
def dashboard_env(tmp_path: Path, project_root: Path) -> DashboardEnv:
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
    return DashboardEnv(
        config_dir=config_dir,
        project_root=tmp_path,
        dataset_path=dataset_path,
        database_path=tmp_path / "storage" / "benchmark.db",
    )


@pytest.fixture()
def dashboard_context(dashboard_env: DashboardEnv):
    context = build_context(dashboard_env)
    yield context
    context.database.dispose()


@pytest.fixture()
def dashboard_app_env(dashboard_env: DashboardEnv, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BENCHMARK_CONFIG_DIR", str(dashboard_env.config_dir))
    monkeypatch.delenv("BENCHMARK_PROJECT_ROOT", raising=False)
    return dashboard_env


def _hardware(
    *,
    cpu: float,
    gpu: float,
    temperature: float,
    ram_gib: int,
    vram_gib: int,
    model_mib: int,
) -> HardwareSummary:
    window = WindowSummary(
        sample_count=3,
        duration_seconds=2.0,
        cpu_avg_percent=cpu,
        cpu_peak_percent=cpu + 15.0,
        ram_avg_used_bytes=ram_gib * 1024 * MIB,
        ram_peak_used_bytes=(ram_gib + 1) * 1024 * MIB,
        gpu_utilization_avg_percent=gpu,
        gpu_utilization_peak_percent=gpu + 10.0,
        vram_avg_used_bytes=vram_gib * 1024 * MIB,
        vram_peak_used_bytes=(vram_gib + 1) * 1024 * MIB,
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
        ollama_model_size_vram_bytes=model_mib * MIB,
        ollama_model_vram_ratio=model_mib / (8 * 1024),
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
        response_text=f"{model_tag} seeded answer for {item.id}",
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
    context: DashboardContext,
    *,
    experiment_id: str = "exp-dashboard-a",
    completed: bool = True,
    with_hardware: bool = True,
    with_evaluations: bool = True,
    with_cold_start: bool = True,
    latency_shift: float = 0.0,
    score_shift: float = 0.0,
) -> tuple[str, list[str]]:
    """Persist one deterministic experiment and return its id and model tags."""
    assert context.dataset is not None
    repos = Repositories(context.database)
    dataset_id = repos.datasets.register(
        context.dataset, context.settings.benchmark.dataset_path
    )
    repos.experiments.create(
        experiment_id=experiment_id,
        name=experiment_id,
        dataset_id=dataset_id,
        config_hash="a" * 64,
        config={"benchmark": "dashboard-test"},
        host={"platform": "test"},
        software={"python": "3.13"},
    )
    specs = context.settings.models.candidates[:3]
    registered = [
        repos.models.register(build_missing_metadata(spec, "0.40.15", ModelRole.CANDIDATE))
        for spec in specs
    ]
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
            hardware = (
                _hardware(
                    cpu=20.0 + index * 18.0,
                    gpu=30.0 + index * 15.0,
                    temperature=40.0 + index * 4.0,
                    ram_gib=2 + index * 2,
                    vram_gib=1 + index,
                    model_mib=512 + index * 256,
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

    if with_cold_start:
        cold = repos.runs.create(
            experiment_external_id=experiment_id,
            model_id=registered[0].id,
            profile="quality",
            cold_start=True,
        )
        item = plan[0]
        response_id = repos.responses.save(
            _response(
                experiment_id=experiment_id,
                model_tag=registered[0].tag,
                item=item,
                latency=9.5,
                ttft=1.4,
                cold_start=True,
                hardware=_hardware(
                    cpu=35.0,
                    gpu=20.0,
                    temperature=45.0,
                    ram_gib=2,
                    vram_gib=1,
                    model_mib=512,
                )
                if with_hardware
                else None,
            ),
            cold.id,
        )
        if with_evaluations:
            repos.evaluations.save(
                _evaluation(item, registered[0].tag, 70.0, judge_score=65.0),
                response_id,
            )
        repos.runs.mark_completed(cold.id)

    if completed:
        repos.experiments.mark_completed(experiment_id)
    return experiment_id, [model.tag for model in registered]


@pytest.fixture()
def seed(dashboard_context: DashboardContext):
    """Seed the shared dashboard test database and return the result."""
    return lambda **kwargs: seed_experiment(dashboard_context, **kwargs)


@pytest.fixture()
def seed_database(dashboard_env: DashboardEnv):
    """Seed the database used by the AppTest dashboard entry point."""

    def _seed(**kwargs):
        context = build_context(dashboard_env)
        try:
            return seed_experiment(context, **kwargs)
        finally:
            context.database.dispose()

    return _seed
