"""
File: test_analytics_recommendations.py

Description:
Unit tests for recommendation gates, near-ties, explanations, and refusals.

Responsibilities:
- Verify a high-failure model cannot rank first.
- Verify scores within two points are flagged as ties with unique ranks.
- Verify insufficient evidence produces an unavailable result.
"""

from __future__ import annotations

import pytest

from benchmark.analytics.recommendations import build_recommendation_set

pytestmark = pytest.mark.unit


def test_failure_gate_prevents_higher_scoring_model_from_ranking_first(
    make_model,
    make_analysis,
) -> None:
    unreliable = make_model("unreliable", composite=90.0, failure_rate=0.31)
    reliable = make_model("reliable", composite=80.0, failure_rate=0.05)
    result = build_recommendation_set(
        make_analysis([unreliable, reliable])
    )

    assert result.available is True
    assert [item.model_tag for item in result.recommendations] == [
        "reliable",
        "unreliable",
    ]
    assert result.recommendations[0].rank == 1
    assert "20% failure-rate gate" in result.recommendations[1].rationale
    assert any(
        "exceed the 20% failure-rate gate" in item for item in result.limitations
    )


def test_scores_within_two_points_are_flagged_as_ties(make_model, make_analysis) -> None:
    first = make_model("first", composite=80.0)
    second = make_model("second", composite=79.0)
    result = build_recommendation_set(make_analysis([first, second]))

    assert result.available is True
    assert result.recommendations[0].tied is True
    assert result.recommendations[0].tied_with == ["second"]
    assert result.recommendations[1].tied is True
    assert result.recommendations[1].tied_with == ["first"]
    assert [item.rank for item in result.recommendations] == [1, 2]
    assert any("within two points" in item for item in result.recommendations[0].limitations)


def test_recommendation_refuses_a_single_comparable_model(
    make_model,
    make_analysis,
) -> None:
    result = build_recommendation_set(make_analysis([make_model("only")]))
    assert result.available is False
    assert "at least two" in (result.reason or "")
    assert result.comparable_models == 1


def test_recommendation_refuses_models_above_failure_gate(
    make_model,
    make_analysis,
) -> None:
    models = [
        make_model("a", composite=90.0, failure_rate=0.25),
        make_model("b", composite=80.0, failure_rate=0.30),
    ]
    result = build_recommendation_set(make_analysis(models))
    assert result.available is False
    assert "every comparable model" in (result.reason or "")


def test_recommendation_refuses_without_evaluations(
    make_model,
    make_analysis,
) -> None:
    models = [
        make_model("a", evaluated=0),
        make_model("b", evaluated=0),
    ]
    result = build_recommendation_set(make_analysis(models))
    assert result.available is False
    assert "no evaluated responses" in (result.reason or "")
    assert result.evaluated_models == 0


def test_explanation_contains_metrics_subscores_and_confidence(
    make_model,
    make_analysis,
) -> None:
    winner = make_model(
        "winner",
        composite=85.0,
        quality=85.0,
        failure_rate=0.0,
        measured=10,
        evaluated=5,
        dimension_scores={
            "quality": 85.0,
            "latency": 70.0,
            "throughput": 75.0,
            "efficiency": 65.0,
            "reliability": 100.0,
        },
    )
    runner_up = make_model("runner", composite=60.0)
    result = build_recommendation_set(make_analysis([winner, runner_up]))
    item = result.recommendations[0]

    assert item.confidence in {"low", "medium", "high"}
    assert "weighted composite" in item.rationale
    assert item.evidence_responses == 5
    assert item.metrics["quality_score"] == 85.0
    assert item.sub_scores["latency"] == 70.0
    assert item.strengths
    assert item.weaknesses
    assert "quality coverage" in " ".join(item.limitations)
