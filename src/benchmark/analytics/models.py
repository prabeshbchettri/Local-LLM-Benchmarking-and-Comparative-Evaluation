"""
File: models.py

Description:
Typed analytics results for statistics, quality, resources, scoring, and recommendations.

Why this file exists:
Analytics output is consumed by reports, recommendations, and the dashboard; weak
dictionaries would hide missing measurements and make unavailable values easy to
mistake for zeroes.

Responsibilities:
- Represent descriptive statistics with an explicit sample count.
- Keep response quality, resource usage, dimensions, and composite scores separate.
- Carry limitations and evidence counts alongside every aggregate.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from benchmark.models.base import BaseSchema

DIMENSIONS = ("quality", "latency", "throughput", "efficiency", "reliability")


class DescriptiveStats(BaseSchema):
    """Summary of one numeric metric; all values are None for an empty sample."""

    count: int = Field(default=0, ge=0)
    mean: float | None = None
    median: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    std_dev: float | None = None
    p95: float | None = None


class ResponseStatistics(BaseSchema):
    """Latency, throughput, outcome, and consistency statistics for one model."""

    measured_responses: int = Field(default=0, ge=0)
    successful_responses: int = Field(default=0, ge=0)
    failed_responses: int = Field(default=0, ge=0)
    timed_out_responses: int = Field(default=0, ge=0)
    success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    failure_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    timeout_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    latency: DescriptiveStats = Field(default_factory=DescriptiveStats)
    ttft: DescriptiveStats = Field(default_factory=DescriptiveStats)
    output_tokens_per_second: DescriptiveStats = Field(default_factory=DescriptiveStats)
    prompt_tokens_per_second: DescriptiveStats = Field(default_factory=DescriptiveStats)
    response_consistency: float | None = Field(default=None, ge=0.0, le=1.0)


class CategoryQuality(BaseSchema):
    """Deterministic, judge, and composite quality for one category."""

    category: str = Field(min_length=1)
    evaluated_responses: int = Field(default=0, ge=0)
    judge_evaluated_responses: int = Field(default=0, ge=0)
    judge_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    deterministic_score: float | None = Field(default=None, ge=0.0, le=100.0)
    judge_score: float | None = Field(default=None, ge=0.0, le=100.0)
    composite_score: float | None = Field(default=None, ge=0.0, le=100.0)


class QualityAnalysis(BaseSchema):
    """Overall and per-category quality with explicit judge coverage."""

    evaluated_responses: int = Field(default=0, ge=0)
    judge_evaluated_responses: int = Field(default=0, ge=0)
    judge_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    deterministic_score: float | None = Field(default=None, ge=0.0, le=100.0)
    judge_score: float | None = Field(default=None, ge=0.0, le=100.0)
    composite_score: float | None = Field(default=None, ge=0.0, le=100.0)
    categories: list[CategoryQuality] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ResourceUsage(BaseSchema):
    """Average hardware consumption and quality per resource unit."""

    hardware_samples: int = Field(default=0, ge=0)
    cpu_avg_percent: float | None = Field(default=None, ge=0.0)
    ram_avg_mib: float | None = Field(default=None, ge=0.0)
    gpu_utilization_avg_percent: float | None = Field(default=None, ge=0.0)
    vram_avg_mib: float | None = Field(default=None, ge=0.0)
    cpu_available: bool = False
    ram_available: bool = False
    gpu_available: bool = False
    vram_available: bool = False
    quality_per_cpu_point: float | None = Field(default=None, ge=0.0)
    quality_per_ram_gib: float | None = Field(default=None, ge=0.0)
    quality_per_vram_gib: float | None = Field(default=None, ge=0.0)
    limitations: list[str] = Field(default_factory=list)


class DimensionScore(BaseSchema):
    """One weighted scoring dimension before and after cross-model normalization."""

    name: str = Field(min_length=1)
    raw_value: float | None = None
    normalized_score: float | None = Field(default=None, ge=0.0, le=100.0)
    weight_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    available: bool = False


class ModelAnalytics(BaseSchema):
    """All measured analytics for one model in one experiment."""

    model_tag: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    statistics: ResponseStatistics = Field(default_factory=ResponseStatistics)
    quality: QualityAnalysis = Field(default_factory=QualityAnalysis)
    resources: ResourceUsage = Field(default_factory=ResourceUsage)
    dimensions: dict[str, DimensionScore] = Field(default_factory=dict)
    quality_score: float | None = Field(default=None, ge=0.0, le=100.0)
    deterministic_score: float | None = Field(default=None, ge=0.0, le=100.0)
    judge_score: float | None = Field(default=None, ge=0.0, le=100.0)
    reliability_score: float | None = Field(default=None, ge=0.0, le=100.0)
    composite_score: float | None = Field(default=None, ge=0.0, le=100.0)
    limitations: list[str] = Field(default_factory=list)


class ExperimentAnalytics(BaseSchema):
    """Profile-specific analytics for every model stored in one experiment."""

    experiment_id: str = Field(min_length=1)
    experiment_status: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    profile_description: str = Field(min_length=1)
    weights: dict[str, float] = Field(default_factory=dict)
    category_weights: dict[str, float] = Field(default_factory=dict)
    models: list[ModelAnalytics] = Field(default_factory=list)
    measured_responses: int = Field(default=0, ge=0)
    evaluated_responses: int = Field(default=0, ge=0)
    generated_at: datetime
    limitations: list[str] = Field(default_factory=list)

    def model(self, tag: str) -> ModelAnalytics | None:
        """Return one model's analytics by exact tag."""
        return next((item for item in self.models if item.model_tag == tag), None)


class Recommendation(BaseSchema):
    """One explained, workload-specific model recommendation."""

    workload: str = Field(min_length=1)
    rank: int = Field(ge=1)
    model_tag: str = Field(min_length=1)
    composite_score: float = Field(ge=0.0, le=100.0)
    confidence: str = Field(min_length=1)
    tied: bool = False
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    rationale: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class RecommendationSet(BaseSchema):
    """Recommendations for one workload, or an explicit unavailable result."""

    workload: str = Field(min_length=1)
    available: bool
    reason: str | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    category_weights: dict[str, float] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class MetricDelta(BaseSchema):
    """Change in one metric between two experiments."""

    metric: str = Field(min_length=1)
    baseline: float | None = None
    candidate: float | None = None
    delta: float | None = None
    percent_change: float | None = None
    available: bool = False


class ModelComparison(BaseSchema):
    """Per-model metric changes across two experiments."""

    model_tag: str = Field(min_length=1)
    metrics: list[MetricDelta] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ExperimentComparison(BaseSchema):
    """Comparison of one candidate experiment against a baseline."""

    baseline_experiment_id: str = Field(min_length=1)
    candidate_experiment_id: str = Field(min_length=1)
    models: list[ModelComparison] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class HistoricalComparison(BaseSchema):
    """Comparison of a baseline experiment against one or more candidates."""

    baseline_experiment_id: str = Field(min_length=1)
    comparisons: list[ExperimentComparison] = Field(default_factory=list)
    generated_at: datetime
