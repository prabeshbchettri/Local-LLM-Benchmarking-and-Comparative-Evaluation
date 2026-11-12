"""
File: test_analytics_statistics.py

Description:
Unit tests for descriptive statistics, outcome rates, and consistency.

Responsibilities:
- Verify successful-only timing and throughput.
- Verify cancelled requests are excluded from reliability denominators.
- Verify timeouts are separated and missing values are ignored.
"""

from __future__ import annotations

import pytest

from benchmark.analytics.statistics import (
    describe,
    response_consistency,
    response_statistics,
)

pytestmark = pytest.mark.unit


def test_describe_ignores_missing_and_non_finite_values() -> None:
    stats = describe([1.0, 3.0, None, float("nan")])
    assert stats.count == 2
    assert stats.mean == pytest.approx(2.0)
    assert stats.median == pytest.approx(2.0)
    assert stats.minimum == 1.0
    assert stats.maximum == 3.0
    assert stats.std_dev == pytest.approx(1.0)
    assert stats.p95 == pytest.approx(2.9)
    assert describe([]).mean is None


def test_outcome_rates_exclude_cancelled_and_timing_uses_successes(make_row) -> None:
    rows = [
        make_row(
            question_id="q1",
            repetition=1,
            latency=2.0,
            ttft=0.5,
            output_tps=20.0,
            prompt_tps=100.0,
        ),
        make_row(
            question_id="q2",
            repetition=1,
            latency=4.0,
            ttft=1.0,
            output_tps=30.0,
            prompt_tps=200.0,
        ),
        make_row(
            question_id="q3",
            status="error",
            latency=6.0,
            ttft=None,
            output_tps=None,
            prompt_tps=None,
            evaluation_status=None,
            deterministic=None,
            judge=None,
            composite=None,
            judge_missing=True,
        ),
        make_row(
            question_id="q4",
            status="timeout",
            latency=8.0,
            ttft=None,
            output_tps=None,
            prompt_tps=None,
            evaluation_status=None,
            deterministic=None,
            judge=None,
            composite=None,
            judge_missing=True,
        ),
        make_row(
            question_id="q5",
            status="cancelled",
            latency=99.0,
            ttft=None,
            output_tps=None,
            prompt_tps=None,
            evaluation_status=None,
            deterministic=None,
            judge=None,
            composite=None,
            judge_missing=True,
        ),
    ]

    stats = response_statistics(rows)

    assert stats.measured_responses == 5
    assert stats.attempted_responses == 4
    assert stats.successful_responses == 2
    assert stats.failed_responses == 2
    assert stats.timed_out_responses == 1
    assert stats.cancelled_responses == 1
    assert stats.success_rate == pytest.approx(0.5)
    assert stats.failure_rate == pytest.approx(0.5)
    assert stats.timeout_rate == pytest.approx(0.25)
    assert stats.cancellation_rate == pytest.approx(0.2)
    assert stats.latency.count == 2
    assert stats.latency.mean == pytest.approx(3.0)
    assert stats.latency.p95 == pytest.approx(3.9)
    assert stats.ttft.mean == pytest.approx(0.75)
    assert stats.output_tokens_per_second.mean == pytest.approx(25.0)
    assert stats.prompt_tokens_per_second.mean == pytest.approx(150.0)


def test_consistency_requires_repeated_successful_answers(make_row) -> None:
    one = [make_row(question_id="q1", text="same answer")]
    assert response_consistency(one) is None

    repeated = [
        make_row(question_id="q1", repetition=1, text="same\nanswer"),
        make_row(question_id="q1", repetition=2, text="same answer"),
    ]
    assert response_consistency(repeated) == pytest.approx(1.0)

    different = [
        make_row(question_id="q1", repetition=1, text="alpha"),
        make_row(question_id="q1", repetition=2, text="omega"),
    ]
    assert response_consistency(different) == pytest.approx(0.2)
