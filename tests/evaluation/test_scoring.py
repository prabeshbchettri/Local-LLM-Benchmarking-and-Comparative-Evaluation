"""
File: test_scoring.py

Description:
Tests for category aggregation, composite scoring, and reliability metrics.

Responsibilities:
- Verify category and overall score formulas.
- Verify coverage and judge-missing behavior.
- Verify judge reliability, family-bias, and manual-agreement summaries.
"""

from __future__ import annotations

import pytest

from benchmark.evaluation.models import (
    EvaluationResult,
    EvaluationStatus,
    JudgeDimensionScore,
    JudgeResult,
    JudgeStatus,
    ManualJudgeScore,
)
from benchmark.evaluation.scoring import (
    compare_manual_scores,
    score_results,
    summarize_family_bias,
    summarize_judge_reliability,
)

pytestmark = [pytest.mark.unit, pytest.mark.evaluation]


def _judge(
    normalized: float | None,
    *,
    status: JudgeStatus = JudgeStatus.SUCCESS,
    retry_count: int = 0,
    raw_score: float | None = None,
    same_family: bool | None = None,
    reliability: bool = False,
    dimension_score: int = 3,
) -> JudgeResult:
    scores = {
        "relevance": dimension_score,
        "completeness": dimension_score,
        "coherence": dimension_score,
        "reasoning_quality": dimension_score,
    }
    dimensions = {
        name: JudgeDimensionScore(score=score, rationale="test")
        for name, score in scores.items()
    }
    return JudgeResult(
        status=status,
        dimensions=dimensions if status is JudgeStatus.SUCCESS else {},
        raw_score=raw_score
        if raw_score is not None
        else (float(dimension_score) if status is JudgeStatus.SUCCESS else None),
        normalized_score=normalized,
        retry_count=retry_count,
        same_family=same_family,
        reliability=reliability,
    )


def _result(
    question_id: str,
    *,
    category: str = "summarization",
    score: float = 1.0,
    max_score: float = 1.0,
    judge: JudgeResult | None = None,
    status: EvaluationStatus = EvaluationStatus.PASSED,
) -> EvaluationResult:
    normalized = score / max_score * 100
    return EvaluationResult(
        question_id=question_id,
        model_id="llama3.1:8b",
        category=category,
        evaluation_type="reference_overlap",
        deterministic_score=score,
        deterministic_max_score=max_score,
        deterministic_normalized_score=normalized,
        judge=judge,
        judge_eligible=judge is not None,
        judge_missing=judge is None or judge.status is not JudgeStatus.SUCCESS,
        category_score=normalized,
        composite_score=normalized,
        passed=status is EvaluationStatus.PASSED,
        status=status,
    )


def test_category_and_composite_scores() -> None:
    results = [
        _result("sum-001", score=1.0, judge=_judge(75.0)),
        _result("sum-002", score=0.5, status=EvaluationStatus.PARTIAL, judge=_judge(50.0)),
    ]
    summary = score_results(results)
    category = summary.categories["summarization"]
    assert category.deterministic_score == 75.0
    assert category.judge_score == 62.5
    assert category.composite_score == pytest.approx(70.0)
    assert summary.overall_deterministic_score == 75.0
    assert summary.overall_composite_score == pytest.approx(70.0)


def test_judge_missing_falls_back_to_deterministic() -> None:
    results = [_result("sum-001", score=0.5, judge=_judge(None, status=JudgeStatus.INVALID_OUTPUT))]
    summary = score_results(results)
    category = summary.categories["summarization"]
    assert category.judge_missing is True
    assert category.composite_score == 50.0


def test_category_judge_missing_when_any_result_lacks_a_judge() -> None:
    results = [
        _result("sum-001", score=1.0, judge=_judge(75.0)),
        _result("sum-002", score=0.5, judge=None),
    ]
    summary = score_results(results)
    assert summary.categories["summarization"].judge_missing is True


def test_quality_coverage() -> None:
    results = [
        _result("sum-001"),
        _result("sum-002", status=EvaluationStatus.SKIPPED),
    ]
    results[1] = results[1].model_copy(
        update={
            "deterministic_score": None,
            "deterministic_max_score": None,
            "deterministic_normalized_score": None,
            "category_score": None,
            "composite_score": None,
        }
    )
    summary = score_results(results)
    assert summary.evaluated_items == 1
    assert summary.quality_coverage == 0.5


def test_judge_reliability_summary() -> None:
    results = [
        _result("sum-001", judge=_judge(50.0)),
        _result("sum-002", judge=_judge(None, status=JudgeStatus.INVALID_OUTPUT, retry_count=1)),
        _result("sum-003", judge=_judge(None, status=JudgeStatus.TIMEOUT)),
    ]
    summary = summarize_judge_reliability(results)
    assert summary.total_calls == 3
    assert summary.successful_calls == 1
    assert summary.failed_calls == 2
    assert summary.invalid_output_calls == 1
    assert summary.timeout_calls == 1
    assert summary.retry_calls == 1
    assert summary.parse_failure_rate == pytest.approx(1 / 3)
    assert summary.retry_rate == pytest.approx(1 / 3)


def test_judge_reliability_variance() -> None:
    results = [
        _result("sum-001", judge=_judge(50.0, raw_score=3.0, reliability=True)),
        _result("sum-002", judge=_judge(75.0, raw_score=4.0, reliability=True)),
        _result("sum-003", judge=_judge(100.0, raw_score=5.0, reliability=True)),
    ]
    summary = summarize_judge_reliability(results)
    assert summary.reliability_sample_count == 3
    assert summary.reliability_score_variance == pytest.approx(1.0)


def test_family_bias_summary() -> None:
    results = [
        _result("sum-001", judge=_judge(80.0, same_family=True)),
        _result("sum-002", judge=_judge(40.0, same_family=False)),
    ]
    summary = summarize_family_bias(results)
    assert summary.same_family_count == 1
    assert summary.cross_family_count == 1
    assert summary.difference == pytest.approx(40.0)


def test_manual_agreement() -> None:
    results = [
        _result("sum-001", judge=_judge(50.0)),
        _result("sum-002", judge=_judge(75.0, dimension_score=4)),
    ]
    manual = [
        ManualJudgeScore(
            question_id="sum-001",
            model_id="llama3.1:8b",
            reviewer="reviewer-a",
            scores={
                "relevance": 3,
                "completeness": 3,
                "coherence": 3,
                "reasoning_quality": 3,
            },
        ),
        ManualJudgeScore(
            question_id="sum-002",
            model_id="llama3.1:8b",
            reviewer="reviewer-a",
            scores={
                "relevance": 4,
                "completeness": 4,
                "coherence": 4,
                "reasoning_quality": 4,
            },
        ),
    ]
    summary = compare_manual_scores(manual, results)
    assert summary.sample_count == 2
    assert summary.overall_exact_agreement == 1.0
    assert summary.overall_within_one_agreement == 1.0
    assert summary.rank_correlation == pytest.approx(1.0)
