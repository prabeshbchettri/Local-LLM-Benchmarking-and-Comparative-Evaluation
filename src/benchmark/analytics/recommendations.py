"""
File: recommendations.py

Description:
Ranked, explained workload recommendations derived only from measured analytics.

Why this file exists:
A ranking without evidence is not actionable. Each recommendation states the scores,
the dimensions that support it, its weaknesses, and what was not measured.

Responsibilities:
- Refuse to recommend when there are no evaluated responses.
- Rank models by profile-specific composite score with deterministic tie-breaking.
- Explain strengths, weaknesses, relevant metrics, and confidence.
"""

from __future__ import annotations

from collections.abc import Sequence

from benchmark.analytics.models import (
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


def _available_dimensions(model: ModelAnalytics) -> dict[str, float]:
    return {
        name: item.normalized_score
        for name, item in model.dimensions.items()
        if item.normalized_score is not None
    }


def _confidence(
    winner: ModelAnalytics,
    model_count: int,
) -> str:
    evaluated = winner.quality.evaluated_responses
    dimensions = len(_available_dimensions(winner))
    judge_covered = winner.quality.judge_coverage >= 0.5
    if (
        model_count >= 3
        and evaluated >= 20
        and dimensions == 5
        and judge_covered
    ):
        return CONFIDENCE_HIGH
    if model_count >= 2 and evaluated >= 5 and dimensions >= 4:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def _relative_findings(
    winner: ModelAnalytics,
    others: Sequence[ModelAnalytics],
) -> tuple[list[str], list[str]]:
    winner_scores = _available_dimensions(winner)
    if not others:
        ordered = sorted(winner_scores.items(), key=lambda item: item[1], reverse=True)
        strengths = [
            f"{DIMENSION_LABELS[name]} scored {score:.1f}"
            for name, score in ordered[:2]
        ]
        weaknesses = [
            f"{DIMENSION_LABELS[name]} scored {score:.1f}"
            for name, score in ordered[-1:]
            if score < 50.0
        ]
        return strengths, weaknesses

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
        "p95_latency_seconds": model.statistics.latency.p95,
        "mean_output_tokens_per_second": model.statistics.output_tokens_per_second.mean,
        "success_rate": model.statistics.success_rate,
        "cpu_avg_percent": model.resources.cpu_avg_percent,
        "ram_avg_mib": model.resources.ram_avg_mib,
        "vram_avg_mib": model.resources.vram_avg_mib,
        "evaluated_responses": float(model.quality.evaluated_responses),
    }


def _rationale(
    winner: ModelAnalytics,
    rank: int,
    profile: ScoringProfile,
) -> str:
    parts = [
        f"{winner.model_tag} ranks {rank} for {profile.name} with a weighted score "
        f"of {winner.composite_score:.2f}/100."
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
        parts.append("Measured evidence: " + ", ".join(measured) + ".")
    parts.append(f"Evidence base: {winner.quality.evaluated_responses} evaluated responses.")
    return " ".join(parts)


def _limitations(
    analysis: ExperimentAnalytics,
    winner: ModelAnalytics,
    profile: ScoringProfile,
) -> list[str]:
    limitations = list(dict.fromkeys(
        analysis.limitations + winner.limitations + winner.quality.limitations
    ))
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
                f"profile category coverage is {coverage * 100:.1f}% of the configured weights"
            )
    if len(analysis.models) < 3:
        limitations.append("fewer than three models were compared")
    if winner.quality.judge_coverage < 0.5:
        limitations.append("most evaluated responses have no judge score")
    if not winner.resources.gpu_available:
        limitations.append("GPU utilization was not measured")
    return list(dict.fromkeys(limitations))


def build_recommendation_set(analysis: ExperimentAnalytics) -> RecommendationSet:
    """Rank models for the analysis profile, or explain why that is not possible."""
    profile = ScoringProfile(
        name=analysis.profile,
        description=analysis.profile_description,
        weights=analysis.weights,
        category_weights=analysis.category_weights,
    )
    eligible = [
        model
        for model in analysis.models
        if model.composite_score is not None and model.quality_score is not None
    ]
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
            workload=profile.name,
            available=False,
            reason=reason,
            weights=analysis.weights,
            category_weights=analysis.category_weights,
            limitations=analysis.limitations,
        )

    ordered = sorted(
        eligible,
        key=lambda model: (
            -model.composite_score,  # type: ignore[operator]
            -(model.reliability_score or 0.0),
            -(model.quality_score or 0.0),
            model.model_tag,
        ),
    )
    recommendations: list[Recommendation] = []
    rank = 0
    previous_score: float | None = None
    for index, model in enumerate(ordered):
        assert model.composite_score is not None
        if previous_score is None or abs(model.composite_score - previous_score) > 1e-9:
            rank = index + 1
        previous_score = model.composite_score
        next_score = (
            ordered[index + 1].composite_score if index + 1 < len(ordered) else None
        )
        tied = next_score is not None and abs(next_score - model.composite_score) <= 1e-9
        strengths, weaknesses = _relative_findings(
            model,
            [other for other in ordered if other.model_tag != model.model_tag],
        )
        recommendations.append(
            Recommendation(
                workload=profile.name,
                rank=rank,
                model_tag=model.model_tag,
                composite_score=model.composite_score,
                confidence=_confidence(model, len(ordered)),
                tied=tied,
                strengths=strengths,
                weaknesses=weaknesses,
                metrics=_metrics(model),
                rationale=_rationale(model, rank, profile),
                limitations=_limitations(analysis, model, profile),
            )
        )
    return RecommendationSet(
        workload=profile.name,
        available=True,
        recommendations=recommendations,
        weights=analysis.weights,
        category_weights=analysis.category_weights,
        limitations=analysis.limitations,
    )
