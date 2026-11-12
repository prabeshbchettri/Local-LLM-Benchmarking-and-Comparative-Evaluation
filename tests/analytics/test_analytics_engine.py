"""
File: test_analytics_engine.py

Description:
Integration tests for analytics built exclusively from stored experiment rows.

Responsibilities:
- Seed responses, metrics, evaluations, and hardware in a real SQLite database.
- Verify default smoke/cold-start exclusion and successful-result aggregation.
- Verify recommendations persist evidence details and historical comparisons work.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from benchmark.analytics import AnalyticsEngine
from benchmark.core.exceptions import InsufficientDataError
from benchmark.database.db import Database
from benchmark.database.migrations import SCHEMA_VERSION
from benchmark.database.repositories import Repositories
from benchmark.dataset.loader import load_dataset
from benchmark.dataset.models import EvaluationType
from benchmark.evaluation.models import EvaluationResult, EvaluationStatus
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

pytestmark = pytest.mark.integration

MIB = 1024 * 1024


@pytest.fixture()
def analytics_db(temp_settings, dataset_path: Path):
    database = Database.from_settings(temp_settings)
    database.initialize()
    repos = Repositories(database)
    dataset = load_dataset(dataset_path)
    dataset_id = repos.datasets.register(dataset)
    yield database, repos, dataset, dataset_id, temp_settings
    database.dispose()


def _experiment(repos, dataset_id: int, experiment_id: str) -> None:
    repos.experiments.create(
        experiment_id=experiment_id,
        name=experiment_id,
        dataset_id=dataset_id,
        config_hash="c" * 64,
        config={"benchmark": "test"},
        host={"platform": "test"},
        software={"python": "3.13"},
    )


def _models(repos, temp_settings):
    return [
        repos.models.register(
            build_missing_metadata(spec, "0.40.15", ModelRole.CANDIDATE)
        )
        for spec in temp_settings.models.candidates[:2]
    ]


def _hardware(cpu: float, ram_gib: int, gpu: float, vram_gib: int, model_mib: int):
    window = WindowSummary(
        sample_count=2,
        duration_seconds=2.0,
        cpu_avg_percent=cpu,
        ram_avg_used_bytes=ram_gib * 1024 * MIB,
        gpu_utilization_avg_percent=gpu,
        vram_avg_used_bytes=vram_gib * 1024 * MIB,
    )
    return HardwareSummary(
        sampling_interval_ms=1000,
        sample_count=2,
        baseline=window,
        inference=window,
        overall=window,
        gpu_available=True,
        gpu_utilization_available=True,
        gpu_memory_available=True,
        ollama_model_vram_available=True,
        ollama_model_size_vram_bytes=model_mib * MIB,
    )


def _add_response(
    repos,
    run,
    *,
    experiment_id: str,
    model_tag: str,
    item_id: str,
    category: str,
    repetition: int,
    status: ResponseStatus = ResponseStatus.SUCCESS,
    latency: float = 2.0,
    composite: float | None = 80.0,
    hardware: HardwareSummary | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    record = ResponseRecord(
        experiment_id=experiment_id,
        model_tag=model_tag,
        model_name=model_tag,
        item_id=item_id,
        category=category,
        profile=InferenceProfile.QUALITY,
        repetition=repetition,
        prompt="prompt",
        response_text="answer" if status is ResponseStatus.SUCCESS else "",
        status=status,
        started_at=now,
        completed_at=now,
        generation_options=GenerationOptions(),
        timing=TimingMetrics(ttft_seconds=latency / 4, total_latency_seconds=latency),
        tokens=TokenMetrics(
            prompt_eval_count=5,
            eval_count=10,
            output_tokens_per_second=100.0 / latency,
            prompt_tokens_per_second=500.0 / latency,
        ),
        hardware=hardware,
    )
    response_id = repos.responses.save(record, run.id)
    if composite is not None:
        evaluation = EvaluationResult(
            question_id=item_id,
            model_id=model_tag,
            category=category,
            evaluation_type=EvaluationType.EXACT_MATCH,
            deterministic_normalized_score=composite,
            composite_score=composite,
            status=EvaluationStatus.PASSED,
            judge_missing=True,
        )
        repos.evaluations.save(evaluation, response_id)


def _seed_experiment(
    repos,
    dataset_id: int,
    temp_settings,
    experiment_id: str,
    *,
    latency_shift: float = 0.0,
    score_shift: float = 0.0,
):
    _experiment(repos, dataset_id, experiment_id)
    models = _models(repos, temp_settings)
    runs = []
    for index, model in enumerate(models):
        run = repos.runs.create(
            experiment_external_id=experiment_id,
            model_id=model.id,
            profile="quality",
        )
        runs.append(run)
        hardware = _hardware(
            cpu=20.0 + index * 20.0,
            ram_gib=2 + index * 2,
            gpu=30.0 + index * 20.0,
            vram_gib=1 + index * 2,
            model_mib=512 + index * 512,
        )
        base_score = 90.0 - index * 35.0 + score_shift
        _add_response(
            repos,
            run,
            experiment_id=experiment_id,
            model_tag=model.tag,
            item_id="gk-001",
            category="general_knowledge",
            repetition=1,
            latency=2.0 + index * 1.5 + latency_shift,
            composite=base_score,
            hardware=hardware,
        )
        _add_response(
            repos,
            run,
            experiment_id=experiment_id,
            model_tag=model.tag,
            item_id="code-001",
            category="coding",
            repetition=1,
            latency=3.0 + index * 2.0 + latency_shift,
            composite=base_score - 10.0,
            hardware=hardware,
        )

        smoke = repos.runs.create(
            experiment_external_id=experiment_id,
            model_id=model.id,
            profile="smoke",
        )
        _add_response(
            repos,
            smoke,
            experiment_id=experiment_id,
            model_tag=model.tag,
            item_id="smoke-001",
            category="general_knowledge",
            repetition=1,
            latency=0.01,
            composite=100.0,
        )

        cold = repos.runs.create(
            experiment_external_id=experiment_id,
            model_id=model.id,
            profile="quality",
            cold_start=True,
        )
        _add_response(
            repos,
            cold,
            experiment_id=experiment_id,
            model_tag=model.tag,
            item_id="cold-001",
            category="general_knowledge",
            repetition=1,
            latency=30.0,
            composite=10.0,
        )

        repos.runs.mark_completed(run.id)
        repos.runs.mark_completed(smoke.id)
        repos.runs.mark_completed(cold.id)
    repos.experiments.mark_completed(experiment_id)
    return models


def test_engine_uses_stored_rows_and_excludes_smoke_and_cold_start(
    analytics_db,
) -> None:
    database, repos, _, dataset_id, temp_settings = analytics_db
    _seed_experiment(repos, dataset_id, temp_settings, "exp-a")
    engine = AnalyticsEngine(database, settings=temp_settings.benchmark)

    analysis = engine.analyze("exp-a")

    assert database.initialize() == SCHEMA_VERSION == "1.1.0"
    assert analysis.profile == "general_purpose"
    assert analysis.measured_responses == 4
    assert analysis.evaluated_responses == 4
    assert len(analysis.models) == 2
    assert {model.statistics.measured_responses for model in analysis.models} == {2}
    assert {model.statistics.success_rate for model in analysis.models} == {1.0}
    assert all(
        row.run_profile == "quality"
        for row in repos.reads.analytics_rows("exp-a")
    )
    assert all(
        not row.cold_start for row in repos.reads.analytics_rows("exp-a")
    )
    assert all(
        row.run_profile is not None
        for row in repos.reads.analytics_rows("exp-a")
    )

    first, second = analysis.models
    assert first.model_tag == temp_settings.models.candidates[0].tag
    assert first.quality.composite_score == pytest.approx(85.0)
    assert second.quality.composite_score == pytest.approx(50.0)
    assert first.quality.item_weighted_composite_score == pytest.approx(85.0)
    assert second.quality.item_weighted_composite_score == pytest.approx(50.0)
    assert first.resources.model_vram_avg_mib == pytest.approx(512.0)
    assert second.resources.model_vram_avg_mib == pytest.approx(1024.0)
    assert first.composite_score is not None
    assert first.composite_score > (second.composite_score or 0.0)
    assert first.dimensions["reliability"].no_discrimination is True

    recommendations = engine.recommend(
        "exp-a",
        workloads=["general_purpose"],
        persist=True,
    )
    assert recommendations[0].available is True
    assert recommendations[0].recommendations[0].model_tag == first.model_tag
    stored = repos.recommendations.list_for_experiment("exp-a")
    assert len(stored) == 2
    assert stored[0]["details"] is not None
    assert "sub_scores" in stored[0]["details"]
    assert stored[0]["details"]["evidence_responses"] == 2


def test_engine_compares_historical_experiments(analytics_db) -> None:
    database, repos, _, dataset_id, temp_settings = analytics_db
    _seed_experiment(repos, dataset_id, temp_settings, "exp-a")
    _seed_experiment(
        repos,
        dataset_id,
        temp_settings,
        "exp-b",
        latency_shift=1.0,
        score_shift=5.0,
    )
    engine = AnalyticsEngine(database, settings=temp_settings.benchmark)

    comparison = engine.compare("exp-a", ["exp-b"])

    assert comparison.baseline_experiment_id == "exp-a"
    assert len(comparison.comparisons) == 1
    candidate = comparison.comparisons[0]
    assert candidate.candidate_experiment_id == "exp-b"
    assert candidate.baseline_profile == "general_purpose"
    assert candidate.candidate_profile == "general_purpose"
    assert len(candidate.models) == 2
    for model in candidate.models:
        metrics = {item.metric: item for item in model.metrics}
        assert metrics["p95_latency_seconds"].available is True
        assert metrics["p95_latency_seconds"].delta == pytest.approx(1.0)
        assert metrics["quality_score"].delta == pytest.approx(5.0)
        assert metrics["model_vram_avg_mib"].available is True


def test_engine_reports_insufficient_data_for_smoke_only(analytics_db) -> None:
    database, repos, _, dataset_id, temp_settings = analytics_db
    _experiment(repos, dataset_id, "exp-smoke")
    model = _models(repos, temp_settings)[0]
    smoke = repos.runs.create(
        experiment_external_id="exp-smoke",
        model_id=model.id,
        profile="smoke",
    )
    _add_response(
        repos,
        smoke,
        experiment_id="exp-smoke",
        model_tag=model.tag,
        item_id="smoke-001",
        category="general_knowledge",
        repetition=1,
        composite=100.0,
    )
    repos.runs.mark_completed(smoke.id)
    repos.experiments.mark_completed("exp-smoke")
    engine = AnalyticsEngine(database, settings=temp_settings.benchmark)

    with pytest.raises(InsufficientDataError):
        engine.analyze("exp-smoke")
