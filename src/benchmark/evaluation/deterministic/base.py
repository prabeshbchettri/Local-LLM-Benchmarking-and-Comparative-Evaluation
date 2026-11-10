"""
File: base.py

Description:
Common protocol and result helpers for deterministic evaluators.

Responsibilities:
- Define the evaluator interface used by the registry.
- Build validated deterministic results and checks consistently.
- Normalize item scores to a 0-100 scale.
"""

from __future__ import annotations

from typing import Protocol

from benchmark.dataset.models import DatasetItem, EvaluationType
from benchmark.evaluation.models import (
    EVALUATOR_VERSION,
    CheckResult,
    DeterministicResult,
)


class Evaluator(Protocol):
    """Protocol implemented by every deterministic evaluator."""

    name: str
    version: str
    evaluation_type: EvaluationType

    def evaluate(self, item: DatasetItem, response_text: str) -> DeterministicResult:
        """Evaluate one response and return a deterministic result."""


def normalized_score(score: float, max_score: float) -> float:
    """Return a 0-100 score, guarding against invalid item configuration."""
    if max_score <= 0:
        return 0.0
    return round(min(100.0, max(0.0, score / max_score * 100.0)), 6)


def make_check(
    name: str,
    passed: bool,
    *,
    score: float | None = None,
    max_score: float = 1.0,
    detail: str = "",
    metadata: dict[str, object] | None = None,
) -> CheckResult:
    """Build a check with an optional partial score."""
    return CheckResult(
        name=name,
        passed=passed,
        score=max_score if score is None and passed else (score or 0.0),
        max_score=max_score,
        detail=detail,
        metadata=metadata or {},
    )


def build_result(
    item: DatasetItem,
    *,
    evaluator: str,
    evaluation_type: EvaluationType,
    score: float,
    checks: list[CheckResult],
    passed: bool | None = None,
    version: str = EVALUATOR_VERSION,
) -> DeterministicResult:
    """Build a deterministic result with normalized score and pass state."""
    bounded_score = round(min(item.max_score, max(0.0, score)), 6)
    resolved_passed = bounded_score >= item.max_score if passed is None else passed
    return DeterministicResult(
        evaluator=evaluator,
        evaluator_version=version,
        evaluation_type=evaluation_type,
        score=bounded_score,
        max_score=item.max_score,
        normalized_score=normalized_score(bounded_score, item.max_score),
        passed=resolved_passed,
        checks=checks,
    )
