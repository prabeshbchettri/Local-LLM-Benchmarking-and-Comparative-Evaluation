"""
File: engine.py

Description:
Unified evaluation request, dispatcher, and result assembly.

Responsibilities:
- Convert stored inference responses into evaluation requests.
- Dispatch deterministic methods by evaluation type.
- Run the blind judge only for judge-eligible items.
- Keep deterministic, judge, category, and composite scores separate.
"""

from __future__ import annotations

import random
from math import ceil

from pydantic import Field

from benchmark.core.config import EvaluationSettings
from benchmark.dataset.models import DatasetItem, EvaluationType
from benchmark.evaluation.deterministic.registry import (
    EvaluatorRegistry,
    build_default_registry,
)
from benchmark.evaluation.judge import Judge
from benchmark.evaluation.models import (
    DeterministicResult,
    EvaluationError,
    EvaluationResult,
    EvaluationStatus,
    JudgeRequest,
    JudgeResult,
    JudgeStatus,
)
from benchmark.evaluation.scoring import DEFAULT_CATEGORY_JUDGE_WEIGHTS
from benchmark.models.base import BaseSchema
from benchmark.models.inference import ResponseRecord, ResponseStatus


class EvaluationRequest(BaseSchema):
    """One candidate response and the dataset item it answers."""

    item: DatasetItem
    response_text: str
    model_id: str = Field(min_length=1)
    response_id: str | None = None
    model_name: str | None = None
    model_digest: str | None = None
    candidate_family: str | None = None
    response_status: ResponseStatus = ResponseStatus.SUCCESS

    @classmethod
    def from_response(
        cls,
        item: DatasetItem,
        response: ResponseRecord,
        *,
        candidate_family: str | None = None,
    ) -> "EvaluationRequest":
        """Build an evaluation request from a stored inference response."""
        return cls(
            item=item,
            response_text=response.response_text,
            model_id=response.model_tag,
            response_id=response.response_id,
            model_name=response.model_name,
            model_digest=response.model_digest,
            candidate_family=candidate_family,
            response_status=response.status,
        )


class EvaluationEngine:
    """Evaluate responses with deterministic methods and an optional blind judge."""

    def __init__(
        self,
        *,
        registry: EvaluatorRegistry | None = None,
        judge: Judge | None = None,
        settings: EvaluationSettings | None = None,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.judge = judge
        self.settings = settings
        self.weights = (
            settings.category_judge_weights
            if settings is not None
            else DEFAULT_CATEGORY_JUDGE_WEIGHTS
        )

    def _base_result(
        self,
        request: EvaluationRequest,
        *,
        status: EvaluationStatus,
        errors: list[EvaluationError],
        deterministic: DeterministicResult | None = None,
        judge: JudgeResult | None = None,
        judge_missing: bool = True,
        judge_order_seed: int | None = None,
    ) -> EvaluationResult:
        item = request.item
        deterministic_normalized = (
            deterministic.normalized_score if deterministic is not None else None
        )
        judge_normalized = (
            judge.normalized_score
            if judge is not None and judge.status is JudgeStatus.SUCCESS
            else None
        )
        composite = deterministic_normalized
        if deterministic_normalized is not None and judge_normalized is not None:
            weight = self.weights.get(item.category.value, 0.0)
            composite = (
                (1.0 - weight) * deterministic_normalized + weight * judge_normalized
            )
        return EvaluationResult(
            question_id=item.id,
            response_id=request.response_id,
            model_id=request.model_id,
            model_name=request.model_name,
            model_digest=request.model_digest,
            candidate_family=request.candidate_family,
            category=item.category.value,
            evaluation_type=item.evaluation_type,
            evaluator=deterministic.evaluator if deterministic else None,
            evaluator_version=deterministic.evaluator_version if deterministic else None,
            deterministic_score=deterministic.score if deterministic else None,
            deterministic_max_score=deterministic.max_score if deterministic else None,
            deterministic_normalized_score=deterministic_normalized,
            checks=deterministic.checks if deterministic else [],
            judge=judge,
            judge_eligible=item.judge_eligible,
            judge_missing=judge_missing,
            judge_order_seed=judge_order_seed,
            category_score=deterministic_normalized,
            composite_score=round(composite, 6) if composite is not None else None,
            passed=deterministic.passed if deterministic else None,
            status=status,
            errors=errors,
        )

    def _skipped(
        self,
        request: EvaluationRequest,
        error_type: str,
        message: str,
        *,
        judge_order_seed: int | None = None,
    ) -> EvaluationResult:
        return self._base_result(
            request,
            status=EvaluationStatus.SKIPPED,
            errors=[
                EvaluationError(
                    stage="response",
                    error_type=error_type,
                    message=message,
                )
            ],
            judge_missing=request.item.judge_eligible,
            judge_order_seed=judge_order_seed,
        )

    def _judge_request(self, request: EvaluationRequest) -> JudgeRequest:
        item = request.item
        return JudgeRequest(
            question_id=item.id,
            prompt=item.prompt,
            system_instruction=item.system_instruction,
            reference_answer=item.reference_answer,
            key_facts=item.key_facts,
            candidate_response=request.response_text,
            rubric_version=self.settings.rubric_version
            if self.settings is not None
            else "1.0.0",
        )

    def _run_judge(
        self,
        request: EvaluationRequest,
        *,
        temperature: float | None = None,
        seed: int | None = None,
        reliability: bool = False,
    ) -> tuple[JudgeResult | None, list[EvaluationError]]:
        item = request.item
        if not item.judge_eligible:
            return None, []
        if self.judge is None:
            return (
                JudgeResult(
                    status=JudgeStatus.NOT_CONFIGURED,
                    error="judge is not configured",
                ),
                [
                    EvaluationError(
                        stage="judge",
                        error_type="not_configured",
                        message="judge is not configured",
                    )
                ],
            )

        try:
            judge_result = self.judge.judge(
                self._judge_request(request),
                temperature=temperature,
                seed=seed,
                reliability=reliability,
            )
        except Exception as exc:
            return (
                JudgeResult(
                    status=JudgeStatus.ERROR,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                [
                    EvaluationError(
                        stage="judge",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                ],
            )

        judge_config = judge_result.config
        if (
            judge_result.same_family is None
            and request.candidate_family is not None
            and judge_config is not None
            and judge_config.family is not None
        ):
            judge_result = judge_result.model_copy(
                update={
                    "same_family": request.candidate_family.casefold()
                    == judge_config.family.casefold()
                }
            )
        errors: list[EvaluationError] = []
        if judge_result.status is not JudgeStatus.SUCCESS:
            errors.append(
                EvaluationError(
                    stage="judge",
                    error_type=judge_result.status.value,
                    message=judge_result.error or "judge evaluation failed",
                )
            )
        return judge_result, errors

    def evaluate(
        self,
        request: EvaluationRequest,
        *,
        judge_order_seed: int | None = None,
        judge_temperature: float | None = None,
        judge_seed: int | None = None,
        reliability: bool = False,
    ) -> EvaluationResult:
        """Evaluate one request without retrying or zeroing failed judge calls."""
        if request.response_status is not ResponseStatus.SUCCESS:
            return self._skipped(
                request,
                "non_success_response",
                f"response status is {request.response_status.value}",
                judge_order_seed=judge_order_seed,
            )
        if not request.response_text.strip():
            return self._skipped(
                request,
                "empty_response",
                "response text is empty",
                judge_order_seed=judge_order_seed,
            )

        try:
            deterministic = self.registry.evaluate(
                request.item, request.response_text
            )
        except Exception as exc:
            return self._base_result(
                request,
                status=EvaluationStatus.ERROR,
                errors=[
                    EvaluationError(
                        stage="deterministic",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                ],
                judge_missing=request.item.judge_eligible,
                judge_order_seed=judge_order_seed,
            )

        judge_result, judge_errors = self._run_judge(
            request,
            temperature=judge_temperature,
            seed=judge_seed,
            reliability=reliability,
        )
        judge_missing = (
            request.item.judge_eligible
            and (
                judge_result is None
                or judge_result.status is not JudgeStatus.SUCCESS
            )
        )
        if deterministic.passed:
            status = EvaluationStatus.PASSED
        elif deterministic.score > 0:
            status = EvaluationStatus.PARTIAL
        else:
            status = EvaluationStatus.FAILED
        return self._base_result(
            request,
            status=status,
            errors=judge_errors,
            deterministic=deterministic,
            judge=judge_result,
            judge_missing=judge_missing,
            judge_order_seed=judge_order_seed,
        )

    def evaluate_many(
        self,
        requests: list[EvaluationRequest],
        *,
        randomize_order: bool = True,
        seed: int = 42,
    ) -> list[EvaluationResult]:
        """Evaluate a batch, optionally randomizing judge call order with a seed."""
        ordered = list(requests)
        if randomize_order:
            random.Random(seed).shuffle(ordered)
        return [
            self.evaluate(
                request,
                judge_order_seed=seed if randomize_order else None,
            )
            for request in ordered
        ]

    def _select_sample(
        self,
        requests: list[EvaluationRequest],
        *,
        fraction: float,
        seed: int | None,
        purpose: str,
    ) -> list[EvaluationRequest]:
        """Select a reproducible sample of judge-eligible successful responses."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"{purpose} fraction must be in (0, 1]")
        eligible = [
            request
            for request in requests
            if request.item.judge_eligible
            and request.response_status is ResponseStatus.SUCCESS
            and request.response_text.strip()
        ]
        if not eligible:
            return []
        count = min(len(eligible), max(1, ceil(len(eligible) * fraction)))
        actual_seed = (
            self.settings.judge_seed if self.settings is not None else 42
        ) if seed is None else seed
        return random.Random(actual_seed).sample(eligible, count)

    def select_reliability_sample(
        self,
        requests: list[EvaluationRequest],
        *,
        fraction: float | None = None,
        seed: int | None = None,
    ) -> list[EvaluationRequest]:
        """Select the repeated-judge reliability sample."""
        configured_fraction = (
            self.settings.reliability_sample_fraction
            if self.settings is not None
            else 0.1
        )
        return self._select_sample(
            requests,
            fraction=configured_fraction if fraction is None else fraction,
            seed=seed,
            purpose="reliability sample",
        )

    def select_manual_validation_sample(
        self,
        requests: list[EvaluationRequest],
        *,
        fraction: float | None = None,
        seed: int | None = None,
    ) -> list[EvaluationRequest]:
        """Select the sample a human should label for judge validation.

        A configured fraction of zero disables manual sampling and returns an
        empty list, because manual validation is optional.
        """
        configured_fraction = (
            self.settings.manual_validation_fraction
            if self.settings is not None
            else 0.1
        )
        actual_fraction = configured_fraction if fraction is None else fraction
        if actual_fraction == 0.0:
            return []
        return self._select_sample(
            requests,
            fraction=actual_fraction,
            seed=seed,
            purpose="manual validation sample",
        )

    def evaluate_reliability(
        self,
        requests: list[EvaluationRequest],
        *,
        samples: int = 3,
        fraction: float | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> list[EvaluationResult]:
        """Run independent repeated judge calls on a reproducible sample."""
        if samples < 1:
            raise ValueError("reliability samples must be at least 1")
        sample = self.select_reliability_sample(
            requests,
            fraction=fraction,
            seed=seed,
        )
        actual_temperature = (
            self.settings.reliability_temperature
            if self.settings is not None
            else 0.7
        ) if temperature is None else temperature
        base_seed = (
            self.settings.judge_seed if self.settings is not None else 42
        ) if seed is None else seed
        return [
            self.evaluate(
                request,
                judge_temperature=actual_temperature,
                judge_seed=base_seed + index,
                reliability=True,
            )
            for request in sample
            for index in range(samples)
        ]
