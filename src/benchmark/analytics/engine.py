"""
File: engine.py

Description:
Analytics engine that turns stored experiment rows into scores and recommendations.

Why this file exists:
All computation must come from persisted responses, metrics, hardware, and
evaluations. The engine is the single entry point that guarantees that, applies
consistent experiment filters, and records which evidence was excluded.

Responsibilities:
- Load measured rows through the persistence layer, excluding smoke and cold-start
  runs by default.
- Build per-model statistics, quality, resources, and normalized dimensions.
- Drop relative dimensions that were not measured for every compared model.
- Apply validated workload profiles, rank models, and persist explanations.
- Compare historical experiments on the same evidence basis.
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
    ExperimentComparison,
    HistoricalComparison,
    ModelAnalytics,
    RecommendationSet,
)
from benchmark.analytics.normalization import (
    has_discrimination,
    min_max_scale,
    weighted_average,
)
from benchmark.analytics.quality import quality_analysis
from benchmark.analytics.recommendations import build_recommendation_set
from benchmark.analytics.resources import resource_usage
from benchmark.analytics.scoring import (
    DEFAULT_PROFILE_NAME,
    PROFILES,
    ScoringProfile,
    build_profiles_from_settings,
    composite_score,
    get_profile,
)
from benchmark.analytics.statistics import response_statistics
from benchmark.core.exceptions import DatabaseError, InsufficientDataError
from benchmark.database.db import Database
from benchmark.database.repositories import AnalyticsRow, RecommendationInput, Repositories

WORKLOAD_PROFILE_NAMES = tuple(PROFILES)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AnalyticsEngine:
    """Compute analytics and recommendations from stored experiment results."""

    def __init__(
        self,
        database: Database,
        *,
        profiles: Mapping[str, ScoringProfile] | None = None,
        settings: object | None = None,
    ) -> None:
        self.database = database
        self.repos = Repositories(database)
        if profiles is not None:
            self.profiles = dict(profiles)
        elif settings is not None:
            self.profiles = build_profiles_from_settings(settings)
        else:
            self.profiles = dict(PROFILES)
        configured_default = getattr(settings, "default_scoring_profile", None)
        if configured_default is None:
            configured_default = DEFAULT_PROFILE_NAME
        self.default_profile = (
            configured_default
            if configured_default in self.profiles
            else next(iter(self.profiles))
        )

    @property
    def profile_names(self) -> tuple[str, ...]:
        """Return configured profile names in configuration order."""
        return tuple(self.profiles)

    def analyze(
        self,
        experiment_id: str,
        *,
        profile: str | None = None,
        run_profile: str | None = None,
        include_cold_start: bool = False,
        include_smoke: bool = False,
    ) -> ExperimentAnalytics:
        """Build profile-specific analytics for every model in one experiment."""
        view = self.repos.experiments.get(experiment_id)
        if view is None:
            raise DatabaseError(f"experiment {experiment_id} does not exist")
        selected_profile = profile or self.default_profile
        scoring_profile = get_profile(selected_profile, self.profiles)
        rows = self.repos.reads.analytics_rows(
            experiment_id,
            run_profile=run_profile,
            include_cold_start=include_cold_start,
            include_smoke=include_smoke,
        )
        if not rows:
            filters = []
            if run_profile is not None:
                filters.append(f"run profile {run_profile!r}")
            if not include_cold_start:
                filters.append("cold-start runs excluded")
            if not include_smoke:
                filters.append("smoke runs excluded")
            detail = f" ({'; '.join(filters)})" if filters else ""
            raise InsufficientDataError(
                f"experiment {experiment_id} has no measured responses{detail}"
            )
        grouped: dict[str, list[AnalyticsRow]] = defaultdict(list)
        for row in rows:
            grouped[row.model_tag].append(row)
        models = [
            self._model_analytics(
                tag,
                grouped[tag],
                scoring_profile.category_weights,
            )
            for tag in sorted(grouped)
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
        if not include_cold_start:
            limitations.append(
                "cold-start runs were excluded; include them explicitly to analyze "
                "loaded-model behavior separately"
            )
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
            run_profile=run_profile,
            include_cold_start=include_cold_start,
            include_smoke=include_smoke,
            generated_at=_utc_now(),
            limitations=limitations,
        )

    def _model_analytics(
        self,
        tag: str,
        rows: list[AnalyticsRow],
        category_weights: Mapping[str, float],
    ) -> ModelAnalytics:
        statistics = response_statistics(rows)
        quality = quality_analysis(rows, category_weights=category_weights)
        quality_score = quality.composite_score
        resources = resource_usage(rows, quality_score)
        limitations = list(quality.limitations) + list(resources.limitations)
        if statistics.response_consistency is None:
            limitations.append(
                "response consistency needs at least two successful repetitions"
            )
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
        """Normalize relative metrics and drop dimensions missing for any model."""
        tags = [model.model_tag for model in models]
        enough_models = len(models) >= 2

        def complete(values: Mapping[str, float | None]) -> bool:
            return enough_models and all(values.get(tag) is not None for tag in tags)

        latency_raw = {
            model.model_tag: model.statistics.latency.p95 for model in models
        }
        throughput_raw = {
            model.model_tag: model.statistics.output_tokens_per_second.mean
            for model in models
        }
        reliability_raw = {
            model.model_tag: model.reliability_score for model in models
        }
        latency_ok = complete(latency_raw)
        throughput_ok = complete(throughput_raw)
        reliability_ok = complete(reliability_raw)
        latency = (
            min_max_scale(latency_raw, higher_is_better=False)
            if latency_ok
            else {tag: None for tag in tags}
        )
        throughput = (
            min_max_scale(throughput_raw, higher_is_better=True)
            if throughput_ok
            else {tag: None for tag in tags}
        )
        reliability = (
            {
                tag: reliability_raw[tag]
                for tag in tags
            }
            if reliability_ok
            else {tag: None for tag in tags}
        )

        cpu_raw = {
            model.model_tag: model.resources.cpu_avg_percent for model in models
        }
        ram_raw = {model.model_tag: model.resources.ram_avg_mib for model in models}
        model_vram_raw = {
            model.model_tag: model.resources.model_vram_avg_mib for model in models
        }
        system_vram_raw = {
            model.model_tag: model.resources.vram_avg_mib for model in models
        }
        if complete(model_vram_raw):
            vram_raw = model_vram_raw
        elif complete(system_vram_raw):
            vram_raw = system_vram_raw
        else:
            vram_raw = {tag: None for tag in tags}
        resource_values = {
            "cpu": cpu_raw,
            "ram": ram_raw,
            "vram": vram_raw,
        }
        active_resources = [
            name for name, values in resource_values.items() if complete(values)
        ]
        normalized_resources = {
            name: min_max_scale(resource_values[name], higher_is_better=False)
            for name in active_resources
        }
        efficiency: dict[str, float | None] = {}
        efficiency_discriminates = any(
            has_discrimination(resource_values[name]) for name in active_resources
        )
        for tag in tags:
            efficiency[tag] = (
                weighted_average(
                    {
                        name: normalized_resources[name][tag]
                        for name in active_resources
                    },
                    {name: 1.0 for name in active_resources},
                )
                if active_resources
                else None
            )

        for model in models:
            tag = model.model_tag
            dimensions = {
                "quality": DimensionScore(
                    name="quality",
                    raw_value=model.quality_score,
                    normalized_score=model.quality_score,
                    available=model.quality_score is not None,
                ),
                "latency": DimensionScore(
                    name="latency",
                    raw_value=latency_raw[tag],
                    normalized_score=latency[tag],
                    available=latency[tag] is not None,
                    no_discrimination=(
                        latency_ok and not has_discrimination(latency_raw)
                    ),
                ),
                "throughput": DimensionScore(
                    name="throughput",
                    raw_value=throughput_raw[tag],
                    normalized_score=throughput[tag],
                    available=throughput[tag] is not None,
                    no_discrimination=(
                        throughput_ok and not has_discrimination(throughput_raw)
                    ),
                ),
                "efficiency": DimensionScore(
                    name="efficiency",
                    raw_value=None,
                    normalized_score=efficiency[tag],
                    available=efficiency[tag] is not None,
                    no_discrimination=(
                        bool(active_resources) and not efficiency_discriminates
                    ),
                ),
                "reliability": DimensionScore(
                    name="reliability",
                    raw_value=reliability_raw[tag],
                    normalized_score=reliability[tag],
                    available=reliability[tag] is not None,
                    no_discrimination=(
                        reliability_ok and not has_discrimination(reliability_raw)
                    ),
                ),
            }
            model.dimensions = dimensions
            if not enough_models:
                model.limitations.append(
                    "relative dimensions need at least two models in the experiment"
                )
            missing = [name for name in DIMENSIONS if not dimensions[name].available]
            if missing:
                model.limitations.append(
                    "unavailable scoring dimensions: " + ", ".join(missing)
                )
            if active_resources and set(active_resources) != {"cpu", "ram", "vram"}:
                model.limitations.append(
                    "efficiency uses only resources measured for every model: "
                    + ", ".join(sorted(active_resources))
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
        run_profile: str | None = None,
        include_cold_start: bool = False,
        include_smoke: bool = False,
    ) -> list[RecommendationSet]:
        """Recommend models for each workload profile, optionally storing results."""
        names = list(workloads) if workloads else list(self.profile_names)
        sets: list[RecommendationSet] = []
        for name in names:
            profile = get_profile(name, self.profiles)
            try:
                analysis = self.analyze(
                    experiment_id,
                    profile=name,
                    run_profile=run_profile,
                    include_cold_start=include_cold_start,
                    include_smoke=include_smoke,
                )
            except InsufficientDataError as exc:
                sets.append(
                    RecommendationSet(
                        workload=name,
                        available=False,
                        reason=str(exc),
                        weights=dict(profile.weights),
                        category_weights=dict(profile.category_weights),
                        limitations=["no measured rows matched the requested filters"],
                    )
                )
                continue
            sets.append(build_recommendation_set(analysis))
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
                    details=item.model_dump(mode="json"),
                )
                for item in result.recommendations
            ]
            stored += self.repos.recommendations.save_many(experiment_id, inputs)
        return stored

    def compare(
        self,
        baseline_experiment_id: str,
        candidate_experiment_ids: Sequence[str],
        *,
        profile: str | None = None,
        run_profile: str | None = None,
        include_cold_start: bool = False,
        include_smoke: bool = False,
    ) -> HistoricalComparison:
        """Compare a baseline experiment against one or more later experiments."""
        filters = {
            "profile": profile,
            "run_profile": run_profile,
            "include_cold_start": include_cold_start,
            "include_smoke": include_smoke,
        }
        baseline = self.analyze(baseline_experiment_id, **filters)
        comparisons: list[ExperimentComparison] = []
        for candidate_id in candidate_experiment_ids:
            try:
                candidate = self.analyze(candidate_id, **filters)
            except InsufficientDataError as exc:
                comparisons.append(
                    ExperimentComparison(
                        baseline_experiment_id=baseline_experiment_id,
                        candidate_experiment_id=candidate_id,
                        baseline_profile=baseline.profile,
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
