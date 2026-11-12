"""
File: recommendations.py

Description:
Ranked, explained workload recommendations derived only from measured analytics.

Why this file exists:
A ranking without evidence is not actionable. Each recommendation states the scores,
the dimensions that support it, its weaknesses, and what was not measured. Models
with excessive failures cannot win solely on quality, and near-ties are disclosed
instead of being presented as decisive.

Responsibilities:
- Refuse to recommend when fewer than two models have comparable evidence.
- Apply a 20% failure-rate gate before selecting a first-place model.
- Flag composite scores within two points as ties while keeping ranks unique.
- Explain strengths, weaknesses, metrics, evidence coverage, and confidence.
"""

from __future__ import annotations

from collections.abc import Sequence

from benchmark.analytics.models import (
    DIMENSIONS,
    ExperimentAnalytics,
    ModelAnalytics,
    Recommendation,
    RecommendationSet,
)
from benchmark.analytics.scoring import ScoringProfile

DIMENSION_LABELS = {
    "quality": "response quality",
    "latency": "latency",
    "throughput": "generation throughput",
    "efficiency": "hardware efficiency",
    "reliability": "reliability",
}
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
FAILURE_RATE_LIMIT = 0.20
TIMEOUT_RATE_WARNING = 0.10
TIE_TOLERANCE = 2.0
METHODOLOGY_VERSION = "1.0.0"


def _available_dimensions(model: ModelAnalytics) -> dict[str, float]:
    return {
        name: item.normalized_score
        for name, item in model.dimensions.items()
        if item.normalized_score is not None
    }


def _confidence(
    winner: ModelAnalytics,
    model_count: int,
    *,
    tied: bool,
) -> str:
    evaluated = winner.quality.evaluated_responses
    dimensions = len(_available_dimensions(winner))
    judge_covered = winner.quality.judge_coverage >= 0.5
    coverage = winner.quality.quality_coverage
    failure = winner.statistics.failure_rate
    if (
        model_count >= 3
        and evaluated >= 20
        and dimensions == len(DIMENSIONS)
        and judge_covered
        and coverage >= 0.8
        and (failure is None or failure <= 0.1)
        and not tied
    ):
        return CONFIDENCE_HIGH
    if (
        model_count >= 2
        and evaluated >= 5
        and dimensions >= 4
        and coverage >= 0.5
    ):
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def _relative_findings(
    winner: ModelAnalytics,
    others: Sequence[ModelAnalytics],
) -> tuple[list[str], list[str]]:
    winner_scores = _available_dimensions(winner)
    differences: list[tuple[str, float, float]] = []
    for name, score in winner_scores.items():
        peer_scores = [
            _available_dimensions(other)[name]
            for other in others
            if name in _available_dimensions(other)
        ]
        if not peer_scores:
            continue
        field_mean = sum(peer_scores) / len(peer_scores)
        differences.append((name, score, score - field_mean))
    strengths = [
        f"{DIMENSION_LABELS[name]} {score:.1f} vs field average {score - diff:.1f}"
        for name, score, diff in sorted(
            (item for item in differences if item[2] >= 5.0),
            key=lambda item: item[2],
            reverse=True,
        )[:3]
    ]
    weaknesses = [
        f"{DIMENSION_LABELS[name]} {score:.1f} vs field average {score - diff:.1f}"
        for name, score, diff in sorted(
            (item for item in differences if item[2] <= -5.0),
            key=lambda item: item[2],
        )[:2]
    ]
    failure = winner.statistics.failure_rate
    if failure is not None and failure > FAILURE_RATE_LIMIT:
        weaknesses.append(f"failure rate {failure * 100:.1f}% exceeds the 20% gate")
    if not strengths:
        strengths = ["no dimension leads the other measured models by 5 points"]
    if not weaknesses:
        weaknesses = ["no dimension trails the other measured models by 5 points"]
    return strengths, weaknesses


def _metrics(model: ModelAnalytics) -> dict[str, float | None]:
    return {
        "quality_score": model.quality_score,
        "deterministic_score": model.deterministic_score,
        "judge_score": model.judge_score,
        "item_weighted_quality_score": model.quality.item_weighted_composite_score,
        "failure_as_zero_quality_score": (
            model.quality.failure_as_zero_composite_score
        ),
        "quality_coverage": model.quality.quality_coverage,
        "judge_coverage": model.quality.judge_coverage,
        "p95_latency_seconds": model.statistics.latency.p95,
        "mean_ttft_seconds": model.statistics.ttft.mean,
        "mean_output_tokens_per_second": model.statistics.output_tokens_per_second.mean,
        "success_rate": model.statistics.success_rate,
        "failure_rate": model.statistics.failure_rate,
        "timeout_rate": model.statistics.timeout_rate,
        "cpu_avg_percent": model.resources.cpu_avg_percent,
        "ram_avg_mib": model.resources.ram_avg_mib,
        "gpu_utilization_avg_percent": model.resources.gpu_utilization_avg_percent,
        "vram_avg_mib": model.resources.vram_avg_mib,
        "model_vram_avg_mib": model.resources.model_vram_avg_mib,
        "evaluated_responses": float(model.quality.evaluated_responses),
        "measured_responses": float(model.statistics.measured_responses),
    }


def _sub_scores(model: ModelAnalytics) -> dict[str, float | None]:
    return {
        name: model.dimensions[name].normalized_score
        if name in model.dimensions
        else None
        for name in DIMENSIONS
    }


def _rationale(
    winner: ModelAnalytics,
    rank: int,
    comparable_count: int,
    profile: ScoringProfile,
) -> str:
    parts = [
        f"{winner.model_tag} ranks {rank} of {comparable_count} comparable models "
        f"for {profile.name} with a weighted composite of "
        f"{winner.composite_score:.2f}/100 after applying the 20% failure-rate gate."
    ]
    measured: list[str] = []
    if winner.quality_score is not None:
        measured.append(f"quality {winner.quality_score:.1f}")
    latency = winner.statistics.latency.p95
    if latency is not None:
        measured.append(f"P95 latency {latency:.2f} s")
    throughput = winner.statistics.output_tokens_per_second.mean
    if throughput is not None:
        measured.append(f"mean throughput {throughput:.1f} tok/s")
    if winner.statistics.success_rate is not None:
        measured.append(f"success rate {winner.statistics.success_rate * 100:.1f}%")
    if measured:
        parts.append("Selected because of " + ", ".join(measured) + ".")
    failure = winner.statistics.failure_rate
    if failure is not None:
        parts.append(f"Its measured failure rate is {failure * 100:.1f}%.")
    parts.append(
        f"Evidence base: {winner.quality.evaluated_responses} evaluated of "
        f"{winner.statistics.measured_responses} measured responses "
        f"({winner.quality.quality_coverage * 100:.1f}% quality coverage)."
    )
    return " ".join(parts)


def _limitations(
    analysis: ExperimentAnalytics,
    winner: ModelAnalytics,
    profile: ScoringProfile,
    tied_with: Sequence[str],
) -> list[str]:
    limitations = list(
        dict.fromkeys(
            analysis.limitations + winner.limitations + winner.quality.limitations
        )
    )
    if profile.category_weights:
        coverage = sum(
            weight
            for name, weight in profile.category_weights.items()
            if any(
                category.category == name and category.composite_score is not None
                for category in winner.quality.categories
            )
        )
        if coverage < 1.0:
            limitations.append(
                f"profile category coverage is {coverage * 100:.1f}% of the "
                "configured weights"
            )
    if len(analysis.models) < 3:
        limitations.append("fewer than three models were compared")
    if winner.quality.quality_coverage < 1.0:
        limitations.append(
            f"quality coverage is {winner.quality.quality_coverage * 100:.1f}%"
        )
    if winner.quality.judge_coverage < 0.5:
        limitations.append("most evaluated responses have no judge score")
    if not winner.resources.gpu_available:
        limitations.append("GPU utilization was not measured")
    if not winner.resources.model_vram_available:
        limitations.append("model-specific VRAM was not measured")
    failure = winner.statistics.failure_rate
    if failure is not None and failure > FAILURE_RATE_LIMIT:
        limitations.append(
            f"failure rate {failure * 100:.1f}% exceeds the 20% first-place gate"
        )
    timeout = winner.statistics.timeout_rate
    if timeout is not None and timeout > TIMEOUT_RATE_WARNING:
        limitations.append(
            f"timeout rate {timeout * 100:.1f}% exceeds the 10% warning threshold"
        )
    if tied_with:
        limitations.append(
            "composite score is within two points of: " + ", ".join(tied_with)
        )
    missing = [
        DIMENSION_LABELS[name]
        for name in DIMENSIONS
        if name not in _available_dimensions(winner)
    ]
    if missing:
        limitations.append("missing scoring dimensions: " + ", ".join(missing))
    return list(dict.fromkeys(limitations))


def _excluded_reasons(
    analysis: ExperimentAnalytics,
) -> tuple[list[ModelAnalytics], dict[str, str]]:
    eligible: list[ModelAnalytics] = []
    excluded: dict[str, str] = {}
    for model in analysis.models:
        if model.quality.evaluated_responses == 0:
            excluded[model.model_tag] = "no evaluated responses"
        elif model.quality_score is None:
            excluded[model.model_tag] = "no quality score for this profile"
        elif model.composite_score is None:
            excluded[model.model_tag] = "no complete composite score"
        else:
            eligible.append(model)
    return eligible, excluded


def build_recommendation_set(analysis: ExperimentAnalytics) -> RecommendationSet:
    """Rank models for the analysis profile, or explain why that is not possible."""
    profile = ScoringProfile(
        name=analysis.profile,
        description=analysis.profile_description,
        weights=analysis.weights,
        category_weights=analysis.category_weights,
    )
    eligible, excluded = _excluded_reasons(analysis)
    evaluated_models = sum(
        1 for model in analysis.models if model.quality.evaluated_responses > 0
    )
    comparable = len(eligible)
    base = {
        "workload": profile.name,
        "weights": analysis.weights,
        "category_weights": analysis.category_weights,
        "evaluated_models": evaluated_models,
        "comparable_models": comparable,
        "excluded_models": excluded,
        "methodology_version": METHODOLOGY_VERSION,
    }
    if not eligible:
        if analysis.evaluated_responses == 0:
            reason = (
                "no evaluated responses are stored for this experiment; "
                "a recommendation would not be evidence-based"
            )
        elif profile.category_weights:
            reason = (
                "no evaluated responses cover the categories required by the "
                f"{profile.name} profile"
            )
        else:
            reason = "no model has enough measured data for a composite score"
        return RecommendationSet(
            available=False,
            reason=reason,
            limitations=analysis.limitations,
            **base,
        )
    if comparable < 2:
        return RecommendationSet(
            available=False,
            reason=(
                "only one model has a comparable composite score; at least two "
                "models are required for a recommendation"
            ),
            limitations=analysis.limitations,
            **base,
        )

    def fails_gate(model: ModelAnalytics) -> bool:
        failure = model.statistics.failure_rate
        return failure is not None and failure > FAILURE_RATE_LIMIT

    gate_passing = [model for model in eligible if not fails_gate(model)]
    if not gate_passing:
        return RecommendationSet(
            available=False,
            reason=(
                "every comparable model exceeds the 20% failure-rate gate; "
                "no first-place recommendation is defensible"
            ),
            limitations=analysis.limitations,
            **base,
        )

    ordered = sorted(
        eligible,
        key=lambda model: (
            fails_gate(model),
            -model.composite_score,  # type: ignore[operator]
            -(model.reliability_score or 0.0),
            -(model.quality_score or 0.0),
            model.model_tag,
        ),
    )
    recommendations: list[Recommendation] = []
    for index, model in enumerate(ordered):
        assert model.composite_score is not None
        tied_with = [
            other.model_tag
            for other in ordered
            if other.model_tag != model.model_tag
            and abs(other.composite_score - model.composite_score) <= TIE_TOLERANCE
        ]
        tied = bool(tied_with)
        strengths, weaknesses = _relative_findings(
            model,
            [other for other in ordered if other.model_tag != model.model_tag],
        )
        rank = index + 1
        confidence = _confidence(model, len(ordered), tied=tied)
        limitations = _limitations(analysis, model, profile, tied_with)
        rationale = (
            _rationale(model, rank, len(ordered), profile)
            + f" Confidence: {confidence}."
            + (
                " Limitations are recorded with the recommendation."
                if limitations
                else ""
            )
        )
        recommendations.append(
            Recommendation(
                workload=profile.name,
                rank=rank,
                model_tag=model.model_tag,
                composite_score=model.composite_score,
                confidence=confidence,
                tied=tied,
                tied_with=tied_with,
                strengths=strengths,
                weaknesses=weaknesses,
                metrics=_metrics(model),
                sub_scores=_sub_scores(model),
                missing_dimensions=[
                    name
                    for name in DIMENSIONS
                    if name not in _available_dimensions(model)
                ],
                evidence_responses=model.quality.evaluated_responses,
                quality_coverage=model.quality.quality_coverage,
                failure_rate=model.statistics.failure_rate,
                timeout_rate=model.statistics.timeout_rate,
                rationale=rationale,
                limitations=limitations,
            )
        )
    set_limitations = list(analysis.limitations)
    if any(fails_gate(model) for model in ordered[1:]):
        set_limitations.append(
            "one or more models exceed the 20% failure-rate gate and were ranked "
            "below gate-passing models"
        )
    return RecommendationSet(
        available=True,
        recommendations=recommendations,
        limitations=list(dict.fromkeys(set_limitations)),
        **base,
    )
