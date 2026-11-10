"""
File: scoring.py

Description:
Category aggregation, composite scoring, judge reliability, and manual agreement.

Responsibilities:
- Aggregate item scores by category with coverage.
- Apply category judge weights without letting the judge override deterministic failure.
- Summarize judge parse failures, retries, timeouts, variance, and family bias.
- Compare manual labels with primary judge scores without overwriting either.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from benchmark.evaluation.models import (
    CategoryScoreSummary,
    EvaluationResult,
    EvaluationSummary,
    FamilyBiasSummary,
    JudgeReliabilitySummary,
    JudgeStatus,
    ManualAgreementSummary,
    ManualJudgeScore,
)
from benchmark.evaluation.rubric import JUDGE_DIMENSIONS

DEFAULT_CATEGORY_JUDGE_WEIGHTS: dict[str, float] = {
    "general_knowledge": 0.30,
    "logical_reasoning": 0.30,
    "mathematical_reasoning": 0.10,
    "coding": 0.10,
    "instruction_following": 0.10,
    "structured_json": 0.10,
    "summarization": 0.40,
}


def _category_score(results: list[EvaluationResult]) -> float | None:
    scored = [
        result
        for result in results
        if result.deterministic_score is not None
        and result.deterministic_max_score is not None
        and result.deterministic_max_score > 0
    ]
    if not scored:
        return None
    earned = sum(result.deterministic_score or 0.0 for result in scored)
    possible = sum(result.deterministic_max_score or 0.0 for result in scored)
    return round(earned / possible * 100.0, 6) if possible > 0 else None


def _judge_score(results: list[EvaluationResult]) -> float | None:
    scores = [
        result.judge.normalized_score
        for result in results
        if result.judge is not None
        and result.judge.status is JudgeStatus.SUCCESS
        and result.judge.normalized_score is not None
    ]
    return round(statistics.fmean(scores), 6) if scores else None


def summarize_judge_reliability(
    results: list[EvaluationResult],
) -> JudgeReliabilitySummary:
    """Summarize actual judge calls, excluding not-applicable and unconfigured states."""
    judged = [
        result.judge
        for result in results
        if result.judge is not None
        and result.judge.status
        not in {JudgeStatus.NOT_APPLICABLE, JudgeStatus.NOT_CONFIGURED}
    ]
    total = len(judged)
    successful = sum(1 for judge in judged if judge.status is JudgeStatus.SUCCESS)
    invalid = sum(1 for judge in judged if judge.status is JudgeStatus.INVALID_OUTPUT)
    timeouts = sum(1 for judge in judged if judge.status is JudgeStatus.TIMEOUT)
    retries = sum(1 for judge in judged if judge.retry_count > 0)
    reliability_scores = [
        judge.raw_score
        for judge in judged
        if judge.reliability and judge.raw_score is not None
    ]
    variance = (
        round(statistics.variance(reliability_scores), 6)
        if len(reliability_scores) >= 2
        else None
    )
    return JudgeReliabilitySummary(
        total_calls=total,
        successful_calls=successful,
        failed_calls=total - successful,
        invalid_output_calls=invalid,
        timeout_calls=timeouts,
        retry_calls=retries,
        parse_failure_rate=round(invalid / total, 6) if total else 0.0,
        retry_rate=round(retries / total, 6) if total else 0.0,
        reliability_sample_count=len(reliability_scores),
        reliability_score_variance=variance,
    )


def summarize_family_bias(results: list[EvaluationResult]) -> FamilyBiasSummary:
    """Describe same-family and cross-family judge scores without claiming causation."""
    same_scores: list[float] = []
    cross_scores: list[float] = []
    for result in results:
        if (
            result.judge is None
            or result.judge.status is not JudgeStatus.SUCCESS
            or result.judge.normalized_score is None
            or result.judge.same_family is None
        ):
            continue
        if result.judge.same_family:
            same_scores.append(result.judge.normalized_score)
        else:
            cross_scores.append(result.judge.normalized_score)
    same_mean = round(statistics.fmean(same_scores), 6) if same_scores else None
    cross_mean = round(statistics.fmean(cross_scores), 6) if cross_scores else None
    difference = (
        round(same_mean - cross_mean, 6)
        if same_mean is not None and cross_mean is not None
        else None
    )
    return FamilyBiasSummary(
        same_family_count=len(same_scores),
        cross_family_count=len(cross_scores),
        same_family_mean=same_mean,
        cross_family_mean=cross_mean,
        difference=difference,
    )


def score_results(
    results: list[EvaluationResult],
    weights: dict[str, float] | None = None,
) -> EvaluationSummary:
    """Aggregate item results into category and overall quality summaries."""
    resolved_weights = weights or DEFAULT_CATEGORY_JUDGE_WEIGHTS
    grouped: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        if result.category:
            grouped[result.category].append(result)

    categories: dict[str, CategoryScoreSummary] = {}
    for category, category_results in sorted(grouped.items()):
        deterministic = _category_score(category_results)
        judge = _judge_score(category_results)
        weight = resolved_weights.get(category, 0.0)
        composite = deterministic
        if deterministic is not None and judge is not None:
            composite = (1.0 - weight) * deterministic + weight * judge
        evaluated = sum(
            1 for result in category_results if result.deterministic_normalized_score is not None
        )
        judged = sum(
            1
            for result in category_results
            if result.judge is not None and result.judge.status is JudgeStatus.SUCCESS
        )
        categories[category] = CategoryScoreSummary(
            category=category,
            item_count=len(category_results),
            evaluated_count=evaluated,
            judged_count=judged,
            coverage=round(evaluated / len(category_results), 6)
            if category_results
            else 0.0,
            deterministic_score=deterministic,
            judge_score=judge,
            composite_score=round(composite, 6) if composite is not None else None,
            judge_weight=weight,
            judge_missing=judge is None,
        )

    deterministic_scores = [
        summary.deterministic_score
        for summary in categories.values()
        if summary.deterministic_score is not None
    ]
    composite_scores = [
        summary.composite_score
        for summary in categories.values()
        if summary.composite_score is not None
    ]
    evaluated_total = sum(
        1 for result in results if result.deterministic_normalized_score is not None
    )
    return EvaluationSummary(
        total_items=len(results),
        evaluated_items=evaluated_total,
        quality_coverage=round(evaluated_total / len(results), 6) if results else 0.0,
        overall_deterministic_score=round(statistics.fmean(deterministic_scores), 6)
        if deterministic_scores
        else None,
        overall_composite_score=round(statistics.fmean(composite_scores), 6)
        if composite_scores
        else None,
        categories=categories,
        judge_reliability=summarize_judge_reliability(results),
        family_bias=summarize_family_bias(results),
    )


def _rankdata(values: list[float]) -> list[float]:
    sorted_values = sorted(values)
    ranks: dict[float, float] = {}
    index = 0
    while index < len(sorted_values):
        end = index + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[index]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        ranks[sorted_values[index]] = average_rank
        index = end
    return [ranks[value] for value in values]


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left)
    right_variance = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_variance * right_variance)
    if denominator == 0:
        return None
    return numerator / denominator


def compare_manual_scores(
    manual_scores: list[ManualJudgeScore],
    results: list[EvaluationResult],
) -> ManualAgreementSummary:
    """Compare manual labels with primary judge scores on matching responses."""
    judged = {
        (result.question_id, result.model_id): result
        for result in results
        if result.judge is not None
        and result.judge.status is JudgeStatus.SUCCESS
        and result.judge.dimensions
    }
    matched: list[tuple[ManualJudgeScore, EvaluationResult]] = [
        (manual, judged[(manual.question_id, manual.model_id)])
        for manual in manual_scores
        if (manual.question_id, manual.model_id) in judged
    ]
    if not matched:
        return ManualAgreementSummary(
            sample_count=0,
            overall_exact_agreement=0.0,
            overall_within_one_agreement=0.0,
        )

    exact_counts = {dimension: 0 for dimension in JUDGE_DIMENSIONS}
    within_one_counts = {dimension: 0 for dimension in JUDGE_DIMENSIONS}
    overall_exact = 0
    overall_within_one = 0
    manual_means: list[float] = []
    judge_means: list[float] = []
    for manual, result in matched:
        all_exact = True
        all_within_one = True
        manual_values: list[float] = []
        judge_values: list[float] = []
        for dimension in JUDGE_DIMENSIONS:
            manual_value = manual.scores[dimension]
            judge_value = result.judge.dimensions[dimension].score
            manual_values.append(float(manual_value))
            judge_values.append(float(judge_value))
            if manual_value == judge_value:
                exact_counts[dimension] += 1
            else:
                all_exact = False
            if abs(manual_value - judge_value) <= 1:
                within_one_counts[dimension] += 1
            else:
                all_within_one = False
        overall_exact += int(all_exact)
        overall_within_one += int(all_within_one)
        manual_means.append(statistics.fmean(manual_values))
        judge_means.append(statistics.fmean(judge_values))

    sample_count = len(matched)
    correlation = _pearson(_rankdata(manual_means), _rankdata(judge_means))
    return ManualAgreementSummary(
        sample_count=sample_count,
        dimension_exact_agreement={
            dimension: round(exact_counts[dimension] / sample_count, 6)
            for dimension in JUDGE_DIMENSIONS
        },
        dimension_within_one_agreement={
            dimension: round(within_one_counts[dimension] / sample_count, 6)
            for dimension in JUDGE_DIMENSIONS
        },
        overall_exact_agreement=round(overall_exact / sample_count, 6),
        overall_within_one_agreement=round(overall_within_one / sample_count, 6),
        rank_correlation=round(correlation, 6) if correlation is not None else None,
    )
