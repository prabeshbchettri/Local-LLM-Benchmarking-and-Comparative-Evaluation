"""
File: test_analytics_normalization.py

Description:
Unit tests for metric normalization, discrimination flags, and weighted averages.

Responsibilities:
- Verify direction handling, missing values, and zero-range behavior.
- Verify weights are renormalized over available dimensions.
"""

from __future__ import annotations

import pytest

from benchmark.analytics.normalization import (
    has_discrimination,
    min_max_scale,
    normalize_value,
    percentage,
    weighted_average,
)

pytestmark = pytest.mark.unit


def test_normalize_value_respects_direction_and_missing_values() -> None:
    assert normalize_value(25.0, 10.0, 30.0) == pytest.approx(75.0)
    assert normalize_value(25.0, 10.0, 30.0, higher_is_better=False) == pytest.approx(
        25.0
    )
    assert normalize_value(None, 10.0, 30.0) is None
    assert normalize_value(25.0, None, 30.0) is None


def test_zero_range_returns_neutral_score_not_full_marks() -> None:
    assert normalize_value(5.0, 5.0, 5.0) == 50.0
    assert not has_discrimination({"a": 5.0, "b": 5.0})
    assert has_discrimination({"a": 5.0, "b": 6.0})


def test_min_max_scale_preserves_missing_models() -> None:
    scaled = min_max_scale(
        {"fast": 1.0, "slow": 3.0, "missing": None},
        higher_is_better=False,
    )
    assert scaled["fast"] == pytest.approx(100.0)
    assert scaled["slow"] == pytest.approx(0.0)
    assert scaled["missing"] is None


def test_weighted_average_renormalizes_available_scores() -> None:
    score = weighted_average(
        {"quality": 80.0, "latency": None, "reliability": 40.0},
        {"quality": 0.6, "latency": 0.2, "reliability": 0.2},
    )
    assert score == pytest.approx(70.0)
    assert weighted_average({"quality": None}, {"quality": 1.0}) is None
    assert percentage(1, 4) == pytest.approx(25.0)
    assert percentage(1, 0) is None


def test_weighted_average_rejects_unknown_or_negative_weights() -> None:
    with pytest.raises(ValueError):
        weighted_average({"quality": 80.0}, {"unknown": 1.0})
    with pytest.raises(ValueError):
        weighted_average({"quality": 80.0}, {"quality": -1.0})
