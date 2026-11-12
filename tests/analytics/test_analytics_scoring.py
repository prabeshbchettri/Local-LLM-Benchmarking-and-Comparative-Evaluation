"""
File: test_analytics_scoring.py

Description:
Unit tests for workload profiles, category weighting, and composite scores.

Responsibilities:
- Verify built-in profiles and configured overrides.
- Verify invalid weights and unknown profiles are rejected.
- Verify missing dimensions are renormalized without imputing values.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchmark.analytics.models import CategoryQuality, DimensionScore, QualityAnalysis
from benchmark.analytics.scoring import (
    PROFILES,
    ScoringProfile,
    build_profiles,
    composite_score,
    get_profile,
)
from benchmark.core.config import WorkloadProfileSettings
from benchmark.core.exceptions import AnalyticsError

pytestmark = pytest.mark.unit


def test_builtin_profiles_match_requested_workloads() -> None:
    assert set(PROFILES) == {
        "general_purpose",
        "coding",
        "reasoning",
        "json_generation",
        "resource_constrained",
    }
    general = PROFILES["general_purpose"]
    assert general.weights == {
        "quality": 40.0,
        "latency": 20.0,
        "throughput": 15.0,
        "efficiency": 10.0,
        "reliability": 15.0,
    }
    assert sum(general.weights.values()) == pytest.approx(100.0)


def test_configured_profile_overrides_defaults() -> None:
    profiles = build_profiles(
        {
            "fast": {
                "description": "Fast responses",
                "weights": {
                    "quality": 50,
                    "latency": 30,
                    "throughput": 10,
                    "efficiency": 5,
                    "reliability": 5,
                },
            }
        }
    )
    assert profiles["fast"].weights["latency"] == 30.0
    assert get_profile("fast", profiles).description == "Fast responses"


def test_invalid_profiles_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ScoringProfile(
            name="bad",
            description="Bad weights",
            weights={
                "quality": 50,
                "latency": 20,
                "throughput": 10,
                "efficiency": 10,
                "reliability": 9,
            },
        )
    with pytest.raises(ValidationError):
        ScoringProfile(
            name="bad",
            description="Unknown dimension",
            weights={
                "quality": 40,
                "latency": 20,
                "throughput": 15,
                "efficiency": 10,
                "reliability": 15,
                "speed": 0,
            },
        )
    with pytest.raises(AnalyticsError):
        get_profile("missing")


def test_configured_profile_validation_requires_full_weights() -> None:
    with pytest.raises(ValidationError):
        WorkloadProfileSettings(
            description="Incomplete",
            weights={
                "quality": 40,
                "latency": 20,
                "throughput": 15,
                "efficiency": 10,
                "reliability": 10,
            },
        )
    with pytest.raises(ValidationError):
        WorkloadProfileSettings(
            description="Bad categories",
            weights={
                "quality": 40,
                "latency": 20,
                "throughput": 15,
                "efficiency": 10,
                "reliability": 15,
            },
            category_weights={"coding": 0.6, "unknown": 0.4},
        )


def test_category_weighted_quality_reports_coverage() -> None:
    quality = QualityAnalysis(
        measured_responses=2,
        evaluated_responses=2,
        quality_coverage=1.0,
        judge_coverage=1.0,
        composite_score=70.0,
        categories=[
            CategoryQuality(category="coding", composite_score=90.0),
            CategoryQuality(category="general_knowledge", composite_score=50.0),
        ],
    )
    score, coverage = PROFILES["coding"].weighted_quality(quality)
    assert score == pytest.approx(90.0)
    assert coverage == pytest.approx(0.7)


def test_composite_renormalizes_missing_dimensions_but_requires_quality() -> None:
    dimensions = {
        "quality": DimensionScore(
            name="quality",
            normalized_score=80.0,
            available=True,
        ),
        "latency": DimensionScore(
            name="latency",
            normalized_score=60.0,
            available=True,
        ),
        "throughput": DimensionScore(name="throughput"),
        "efficiency": DimensionScore(
            name="efficiency",
            normalized_score=40.0,
            available=True,
        ),
        "reliability": DimensionScore(
            name="reliability",
            normalized_score=100.0,
            available=True,
        ),
    }
    score = composite_score(dimensions, PROFILES["general_purpose"])
    expected = (80 * 40 + 60 * 20 + 40 * 10 + 100 * 15) / 85
    assert score == pytest.approx(expected)

    dimensions["quality"] = DimensionScore(name="quality")
    assert composite_score(dimensions, PROFILES["general_purpose"]) is None
