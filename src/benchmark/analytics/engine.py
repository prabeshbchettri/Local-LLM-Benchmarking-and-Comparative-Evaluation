"""
File: engine.py

Description:
Analytics engine that turns stored experiment rows into scores and recommendations.

Why this file exists:
All computation must come from persisted responses, metrics, hardware, and
evaluations. The engine is the single entry point that guarantees that.

Responsibilities:
- Load measured rows through the persistence layer.
- Build per-model statistics, quality, resources, and normalized dimensions.
- Apply validated workload profiles and rank models.
- Persist recommendations and compare historical experiments.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from benchmark.analytics.comparison import compare_analytics
from benchmark.analytics.models import (
    DIMENSIONS,
    DimensionScore,
    ExperimentAnalytics,
    HistoricalComparison,
    ModelAnalytics,
    RecommendationSet,
)
from benchmark.analytics.normalization import min_max_scale, weighted_average
from benchmark.analytics.quality import quality_analysis
from benchmark.analytics.recommendations import build_recommendation_set
from benchmark.analytics.resources import resource_usage
from benchmark.analytics.scoring import (
    DEFAULT_PROFILE_NAME,
    PROFILES,
    ScoringProfile,
    composite_score,
    get_profile,
)
from benchmark.analytics.statistics import response_statistics
from benchmark.core.exceptions import DatabaseError, InsufficientDataError
from benchmark.database.db import Database
from benchmark.database.repositories import AnalyticsRow, RecommendationInput, Repositories

WORKLOAD_PROFILE_NAMES = (
    "general_purpose",
    "coding",
    "reasoning",
    "json_generation",
    "resource_constrained",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AnalyticsEngine:
    """Compute analytics and recommendations from stored experiment results."""

    def __init__(
        self,
        database: Database,
        *,
        profiles: Mapping[str, ScoringProfile] | None = None,
    ) -> None:
        self.database = database
        self.repos = Repositories(database)
        self.profiles = dict(profiles) if profiles is not None else dict(PROFILES)

    def analyze(
        self,
        experiment_id: str,
        *,
        profile: str = DEFAULT_PROFILE_NAME,
    ) -> ExperimentAnalytics:
        """Build profile-specific analytics for every model in one experiment."""
        view = self.repos.experiments.get(experiment_id)
        if view is None:
            raise DatabaseError(f"experiment {experiment_id} does not exist")
        rows = self.repos.reads.analytics_rows(experiment_id)
        if not rows:
            raise InsufficientDataError(
                f"experiment {experiment_id} has no measured responses"
            )
        scoring_profile = get_profile(profile, self.profiles)
        grouped: dict[str, list[AnalyticsRow]] = defaultdict(list)
        for row in rows:
            grouped[row.model_tag].append(row)
        models = [
            self._model_analytics(tag, grouped[tag]) for tag in sorted(grouped)
        ]
        self._normalize_dimensions(models)
        models = [self._apply_profile(model, scoring_profile) for model in models]
        limitations: list[str] = []
        if view.status != "completed":
            limitations.append(f"experiment status is {view.status}")
        if len(models) < 2:
            limitations.append(
                "cross-model normalization and ranking need at least two models"
            )
        evaluated = sum(model.quality.evaluated_responses for model in models)
        if evaluated == 0:
            limitations.append("no evaluated responses are stored")
        return ExperimentAnalytics(
            experiment_id=experiment_id,
            experiment_status=view.status,
            profile=scoring_profile.name,
            profile_description=scoring_profile.description,
            weights=dict(scoring_profile.weights),
            category_weights=dict(scoring_profile.category_weights),
            models=models,
            measured_responses=len(rows),
            evaluated_responses=evaluated,
            generated_at=_utc_now(),
            limitations=limitations,
        )

    def _model_analytics(
        self,
        tag: str,
        rows: list[AnalyticsRow],
    ) -> ModelAnalytics:
        statistics = response_statistics(rows)
        quality = quality_analysis(rows)
        quality_score = quality.composite_score
        resources = resource_usage(rows, quality_score)
        limitations = list(quality.limitations) + list(resources.limitations)
        return ModelAnalytics(
            model_tag=tag,
            model_name=rows[0].model_name,
            statistics=statistics,
            quality=quality,
            resources=resources,
            quality_score=quality_score,
            deterministic_score=quality.deterministic_score,
            judge_score=quality.judge_score,
            reliability_score=(
                statistics.success_rate * 100.0
                if statistics.success_rate is not None
                else None
            ),
            limitations=limitations,
        )

    def _normalize_dimensions(self, models: list[ModelAnalytics]) -> None:
        """Normalize relative metrics across models and attach dimension scores."""
        latency_raw = {
            model.model_tag: model.statistics.latency.p95 for model in models
        }
        throughput_raw = {
            model.model_tag: model.statistics.output_tokens_per_second.mean
            for model in models
        }
        latency = (
            min_max_scale(latency_raw, higher_is_better=False)
            if len(models) >= 2
            else {tag: None for tag in latency_raw}
        )
        throughput = (
            min_max_scale(throughput_raw, higher_is_better=True)
            if len(models) >= 2
            else {tag: None for tag in throughput_raw}
        )

        cpu = min_max_scale(
            {model.model_tag: model.resources.cpu_avg_percent for model in models},
            higher_is_better=False,
        )
        ram = min_max_scale(
            {model.model_tag: model.resources.ram_avg_mib for model in models},
            higher_is_better=False,
        )
        vram = min_max_scale(
            {model.model_tag: model.resources.vram_avg_mib for model in models},
            higher_is_better=False,
        )
        efficiency: dict[str, float | None] = {}
        for model in models:
            tag = model.model_tag
            if len(models) < 2:
                efficiency[tag] = None
                continue
            efficiency[tag] = weighted_average(
                {"cpu": cpu[tag], "ram": ram[tag], "vram": vram[tag]},
                {"cpu": 1.0, "ram": 1.0, "vram": 1.0},
            )

        for model in models:
            dimensions = {
                "quality": DimensionScore(
                    name="quality",
                    raw_value=model.quality_score,
                    normalized_score=model.quality_score,
                    available=model.quality_score is not None,
                ),
                "latency": DimensionScore(
                    name="latency",
                    raw_value=latency_raw[model.model_tag],
                    normalized_score=latency[model.model_tag],
                    available=latency[model.model_tag] is not None,
                ),
                "throughput": DimensionScore(
                    name="throughput",
                    raw_value=throughput_raw[model.model_tag],
                    normalized_score=throughput[model.model_tag],
                    available=throughput[model.model_tag] is not None,
                ),
                "efficiency": DimensionScore(
                    name="efficiency",
                    raw_value=None,
                    normalized_score=efficiency[model.model_tag],
                    available=efficiency[model.model_tag] is not None,
                ),
                "reliability": DimensionScore(
                    name="reliability",
                    raw_value=model.reliability_score,
                    normalized_score=model.reliability_score,
                    available=model.reliability_score is not None,
                ),
            }
            model.dimensions = dimensions
            if len(models) < 2:
                model.limitations.append(
                    "relative dimensions need at least two models in the experiment"
                )
            if not all(dimensions[name].available for name in DIMENSIONS):
                model.limitations.append(
                    "one or more scoring dimensions are unavailable"
                )

    def _apply_profile(
        self,
        model: ModelAnalytics,
        profile: ScoringProfile,
    ) -> ModelAnalytics:
        quality_value, coverage = profile.weighted_quality(model.quality)
        updated = model.model_copy(deep=True)
        updated.quality_score = quality_value
        quality = updated.dimensions.get("quality")
        if quality is None:
            quality = DimensionScore(name="quality")
        quality.raw_value = quality_value
        quality.normalized_score = quality_value
        quality.available = quality_value is not None
        updated.dimensions["quality"] = quality
        for name, dimension in updated.dimensions.items():
            dimension.weight_percent = profile.weights.get(name, 0.0)
        if profile.category_weights and coverage < 1.0:
            updated.limitations.append(
                f"quality category coverage for {profile.name} is "
                f"{coverage * 100:.1f}%"
            )
        updated.composite_score = composite_score(updated.dimensions, profile)
        return updated

    def recommend(
        self,
        experiment_id: str,
        *,
        workloads: Sequence[str] | None = None,
        persist: bool = False,
    ) -> list[RecommendationSet]:
        """Recommend models for each workload profile, optionally storing the result."""
        names = list(workloads) if workloads else list(WORKLOAD_PROFILE_NAMES)
        sets = [
            build_recommendation_set(self.analyze(experiment_id, profile=name))
            for name in names
        ]
        if persist:
            self.persist_recommendations(experiment_id, sets)
        return sets

    def persist_recommendations(
        self,
        experiment_id: str,
        recommendation_sets: Sequence[RecommendationSet],
    ) -> int:
        """Store available recommendations, returning the number of new rows."""
        stored = 0
        for result in recommendation_sets:
            if not result.available:
                continue
            inputs = [
                RecommendationInput(
                    workload=result.workload,
                    rank=item.rank,
                    model_tag=item.model_tag,
                    score=item.composite_score,
                    confidence=item.confidence,
                    weights={
                        **{
                            f"dimension:{name}": weight
                            for name, weight in result.weights.items()
                        },
                        **{
                            f"category:{name}": weight
                            for name, weight in result.category_weights.items()
                        },
                    },
                    rationale=item.rationale,
                )
                for item in result.recommendations
            ]
            stored += self.repos.recommendations.save_many(experiment_id, inputs)
        return stored

    def compare(
        self,
        baseline_experiment_id: str,
        candidate_experiment_ids: Sequence[str],
    ) -> HistoricalComparison:
        """Compare a baseline experiment against one or more later experiments."""
        baseline = self.analyze(baseline_experiment_id)
        comparisons = []
        for candidate_id in candidate_experiment_ids:
            try:
                candidate = self.analyze(candidate_id)
            except InsufficientDataError as exc:
                from benchmark.analytics.models import ExperimentComparison

                comparisons.append(
                    ExperimentComparison(
                        baseline_experiment_id=baseline_experiment_id,
                        candidate_experiment_id=candidate_id,
                        models=[],
                        limitations=[str(exc)],
                    )
                )
                continue
            comparisons.append(compare_analytics(baseline, candidate))
        return HistoricalComparison(
            baseline_experiment_id=baseline_experiment_id,
            comparisons=comparisons,
            generated_at=_utc_now(),
        )
