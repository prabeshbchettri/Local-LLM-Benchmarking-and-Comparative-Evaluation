"""
File: statistics.py

Description:
Descriptive statistics, outcome rates, and response consistency for measured requests.

Why this file exists:
Latency, throughput, and reliability must be computed the same way for every report
and recommendation, with missing measurements excluded rather than treated as zero.

Responsibilities:
- Compute mean, median, minimum, maximum, standard deviation, and P95 with NumPy.
- Compute success, failure, and timeout rates from stored response statuses.
- Estimate repetition consistency from response text similarity.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher

import numpy as np

from benchmark.analytics.models import DescriptiveStats, ResponseStatistics
from benchmark.database.repositories import AnalyticsRow


def describe(values: Iterable[float | None]) -> DescriptiveStats:
    """Summarize finite values, ignoring missing or non-finite entries."""
    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not clean:
        return DescriptiveStats(count=0)
    array = np.asarray(clean, dtype=float)
    return DescriptiveStats(
        count=int(array.size),
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        std_dev=float(np.std(array, ddof=0)),
        p95=float(np.percentile(array, 95, method="linear")),
    )


def _rate(count: int, total: int) -> float | None:
    """Return a safe ratio, or None when there is no denominator."""
    if total <= 0:
        return None
    return count / total


def response_consistency(rows: Sequence[AnalyticsRow]) -> float | None:
    """Return mean pairwise text similarity for successfully repeated questions.

    Questions measured once cannot provide a consistency estimate, so this returns
    None instead of claiming perfect consistency.
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row.response_status == "success" and row.response_text.strip():
            grouped[row.question_id].append(row.response_text)
    scores: list[float] = []
    for texts in grouped.values():
        for index, first in enumerate(texts):
            for second in texts[index + 1 :]:
                scores.append(SequenceMatcher(None, first, second).ratio())
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def response_statistics(rows: Sequence[AnalyticsRow]) -> ResponseStatistics:
    """Compute outcome, timing, throughput, and consistency statistics."""
    measured = len(rows)
    successful = sum(1 for row in rows if row.response_status == "success")
    timed_out = sum(1 for row in rows if row.response_status == "timeout")
    failed = measured - successful - timed_out
    return ResponseStatistics(
        measured_responses=measured,
        successful_responses=successful,
        failed_responses=max(failed, 0),
        timed_out_responses=timed_out,
        success_rate=_rate(successful, measured),
        failure_rate=_rate(max(failed, 0), measured),
        timeout_rate=_rate(timed_out, measured),
        latency=describe(row.latency_seconds for row in rows),
        ttft=describe(row.ttft_seconds for row in rows),
        output_tokens_per_second=describe(
            row.output_tokens_per_second for row in rows
        ),
        prompt_tokens_per_second=describe(
            row.prompt_tokens_per_second for row in rows
        ),
        response_consistency=response_consistency(rows),
    )
