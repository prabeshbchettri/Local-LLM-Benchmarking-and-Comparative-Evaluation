"""
File: conftest.py

Description:
Shared builders for analytics unit tests.

Responsibilities:
- Build AnalyticsRow values with explicit missing measurements.
- Build ModelAnalytics values without touching the database.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import pytest

from benchmark.analytics.models import (
    DIMENSIONS,
    DimensionScore,
    ExperimentAnalytics,
    ModelAnalytics,
    QualityAnalysis,
    ResourceUsage,
    ResponseStatistics,
)
from benchmark.database.repositories import AnalyticsRow


@pytest.fixture()
def make_row() -> Callable[..., AnalyticsRow]:
    def _make_row(
        *,
        model_tag: str = "model-a",
        model_name: str = "Model A",
        question_id: str = "q-1",
        category: str = "general_knowledge",
        status: str = "success",
        repetition: int = 1,
        text: str = "answer",
        latency: float | None = 1.0,
        ttft: float | None = 0.2,
        output_tps: float | None = 10.0,
        prompt_tps: float | None = 100.0,
        evaluation_status: str | None = "passed",
        deterministic: float | None = 80.0,
        judge: float | None = 80.0,
        composite: float | None = 80.0,
        judge_missing: bool = False,
        hardware: bool = False,
        cpu: float | None = None,
        ram_bytes: int | None = None,
        gpu: float | None = None,
        vram_bytes: int | None = None,
        model_vram_bytes: int | None = None,
        run_profile: str = "quality",
        cold_start: bool = False,
    ) -> AnalyticsRow:
        return AnalyticsRow(
            model_tag=model_tag,
            model_name=model_name,
            run_profile=run_profile,
            cold_start=cold_start,
            question_id=question_id,
            category=category,
            response_status=status,
            repetition=repetition,
            response_text=text,
            latency_seconds=latency,
            ttft_seconds=ttft,
            output_tokens_per_second=output_tps,
            prompt_tokens_per_second=prompt_tps,
            evaluation_status=evaluation_status,
            evaluation_type="exact_match",
            deterministic_score=deterministic,
            judge_score=judge,
            composite_score=composite,
            judge_missing=judge_missing,
            hardware_present=hardware,
            cpu_avg_percent=cpu,
            ram_avg_used_bytes=ram_bytes,
            gpu_utilization_avg_percent=gpu,
            vram_avg_used_bytes=vram_bytes,
            ollama_model_vram_bytes=model_vram_bytes,
            ollama_model_vram_available=model_vram_bytes is not None,
            gpu_available=gpu is not None,
            gpu_memory_available=vram_bytes is not None,
            gpu_name="Test GPU" if hardware else None,
        )

    return _make_row


@pytest.fixture()
def make_model() -> Callable[..., ModelAnalytics]:
    def _make_model(
        tag: str,
        *,
        composite: float = 80.0,
        quality: float | None = None,
        failure_rate: float = 0.0,
        timeout_rate: float = 0.0,
        measured: int = 10,
        evaluated: int = 10,
        judge_coverage: float = 1.0,
        dimension_scores: dict[str, float | None] | None = None,
        limitations: list[str] | None = None,
    ) -> ModelAnalytics:
        quality_score = composite if quality is None else quality
        dimensions = {
            name: DimensionScore(
                name=name,
                raw_value=(dimension_scores or {}).get(name),
                normalized_score=(
                    dimension_scores.get(name)
                    if dimension_scores is not None
                    else composite
                ),
                weight_percent=20.0,
                available=(
                    dimension_scores.get(name) is not None
                    if dimension_scores is not None
                    else True
                ),
            )
            for name in DIMENSIONS
        }
        failed = round(measured * failure_rate)
        return ModelAnalytics(
            model_tag=tag,
            model_name=tag,
            statistics=ResponseStatistics(
                measured_responses=measured,
                attempted_responses=measured,
                successful_responses=measured - failed,
                failed_responses=failed,
                timed_out_responses=round(measured * timeout_rate),
                failure_rate=failure_rate,
                timeout_rate=timeout_rate,
                success_rate=1.0 - failure_rate,
            ),
            quality=QualityAnalysis(
                measured_responses=measured,
                evaluated_responses=evaluated,
                quality_coverage=evaluated / measured if measured else 0.0,
                judge_coverage=judge_coverage,
                composite_score=quality_score,
            ),
            resources=ResourceUsage(),
            dimensions=dimensions,
            quality_score=quality_score,
            reliability_score=(1.0 - failure_rate) * 100.0,
            composite_score=composite,
            limitations=list(limitations or []),
        )

    return _make_model


@pytest.fixture()
def make_analysis() -> Callable[..., ExperimentAnalytics]:
    def _make_analysis(
        models: list[ModelAnalytics],
        *,
        profile: str = "general_purpose",
    ) -> ExperimentAnalytics:
        return ExperimentAnalytics(
            experiment_id="exp-test",
            experiment_status="completed",
            profile=profile,
            profile_description="Test profile",
            weights={
                "quality": 40.0,
                "latency": 20.0,
                "throughput": 15.0,
                "efficiency": 10.0,
                "reliability": 15.0,
            },
            models=models,
            measured_responses=sum(
                model.statistics.measured_responses for model in models
            ),
            evaluated_responses=sum(
                model.quality.evaluated_responses for model in models
            ),
            generated_at=datetime.now(timezone.utc),
        )

    return _make_analysis
