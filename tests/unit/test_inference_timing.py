"""
File: test_inference_timing.py

Description:
Unit tests for monotonic timing and token-rate calculations.

Responsibilities:
- Verify nanosecond conversion and elapsed-time calculations.
- Verify TTFT behavior when no token arrives.
- Verify undefined token rates are represented as None.
"""

from __future__ import annotations

import pytest

from benchmark.execution.timing import (
    elapsed_seconds,
    nanoseconds_to_seconds,
    tokens_per_second,
    ttft_seconds,
)

pytestmark = pytest.mark.unit


def test_nanoseconds_and_elapsed_seconds() -> None:
    assert nanoseconds_to_seconds(None) is None
    assert nanoseconds_to_seconds(1_500_000_000) == 1.5
    assert elapsed_seconds(2_000_000_000, 2_250_000_000) == 0.25


def test_elapsed_seconds_rejects_reversed_samples() -> None:
    with pytest.raises(ValueError):
        elapsed_seconds(2, 1)


def test_ttft_is_none_without_a_token() -> None:
    assert ttft_seconds(100, None) is None
    assert ttft_seconds(100, 250_000_100) == 0.25


def test_tokens_per_second_handles_zero_and_missing_durations() -> None:
    assert tokens_per_second(None, 1_000_000_000) is None
    assert tokens_per_second(10, None) is None
    assert tokens_per_second(10, 0) is None
    assert tokens_per_second(10, 500_000_000) == 20.0
