"""
File: models.py

Description:
Typed evaluation records for deterministic checks, judge calls, and aggregates.

Responsibilities:
- Keep deterministic and subjective scores separate.
- Record evaluator and judge identity, configuration, status, and errors.
- Represent item-level results, category summaries, and reliability samples.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, StrictInt, model_validator

from benchmark.dataset.models import EvaluationType
from benchmark.models.base import BaseSchema

EVALUATION_SCHEMA_VERSION = "1.0.0"
EVALUATOR_VERSION = "1.0.0"


class EvaluationStatus(StrEnum):
    """Final status of one item evaluation."""

    PASSED = "passed"
    PARTIAL = "partial"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class JudgeStatus(StrEnum):
    """Final status of one judge call."""

    NOT_APPLICABLE = "not_applicable"
    NOT_CONFIGURED = "not_configured"
    SUCCESS = "success"
    INVALID_OUTPUT = "invalid_output"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    VERIFICATION_ERROR = "verification_error"
    ERROR = "error"


class EvaluationError(BaseSchema):
    """Structured evaluation error for one stage."""

    stage: Literal["response", "deterministic", "judge"]
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)


class CheckResult(BaseSchema):
    """One independently reported deterministic or judge-derived check."""

    name: str = Field(min_length=1)
    passed: bool
    score: float = Field(ge=0.0)
    max_score: float = Field(gt=0.0)
    detail: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_score_bounds(self) -> "CheckResult":
        if self.score > self.max_score:
            raise ValueError("check score must not exceed max_score")
        return self


class DeterministicResult(BaseSchema):
    """Outcome of one deterministic evaluator."""

    evaluator: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    evaluation_type: EvaluationType
    score: float = Field(ge=0.0)
    max_score: float = Field(gt=0.0)
    normalized_score: float = Field(ge=0.0, le=100.0)
    passed: bool
    checks: list[CheckResult] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="after")
    def validate_score_bounds(self) -> "DeterministicResult":
        if self.score > self.max_score:
            raise ValueError("deterministic score must not exceed max_score")
        return self


class JudgeDimensionScore(BaseSchema):
    """Strict rubric score and rationale for one judge dimension."""

    score: Annotated[StrictInt, Field(ge=1, le=5)]
    rationale: str = Field(min_length=1)


class JudgeRawOutput(BaseSchema):
    """Validated JSON object returned by the judge."""

    relevance: JudgeDimensionScore
    completeness: JudgeDimensionScore
    coherence: JudgeDimensionScore
    reasoning_quality: JudgeDimensionScore


class JudgeConfig(BaseSchema):
    """Identity and generation settings for one judge call."""

    name: str = Field(min_length=1)
    tag: str = Field(min_length=1)
    digest: str | None = None
    family: str | None = None
    rubric_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    seed: int
    top_p: float = Field(ge=0.0, le=1.0)
    top_k: int = Field(ge=1)
    repeat_penalty: float = Field(gt=0.0)
    num_ctx: int = Field(ge=512)
    num_predict: int = Field(ge=1)


class JudgeRequest(BaseSchema):
    """Blind judge input containing no candidate identity or ranking hints."""

    question_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    system_instruction: str | None = None
    reference_answer: str | None = None
    key_facts: list[str] = Field(default_factory=list)
    candidate_response: str = Field(min_length=1)
    rubric_version: str = Field(min_length=1)


class JudgeVerification(BaseSchema):
    """Verification result for the configured judge artifact."""

    tag: str
    installed: bool
    digest: str | None = None
    expected_digest: str | None = None
    digest_match: bool | None = None
    quantization: str | None = None
    expected_quantization: str | None = None
    quantization_match: bool | None = None
    problems: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return whether the judge artifact is installed and verified."""
        return self.installed and not self.problems


class JudgeResult(BaseSchema):
    """Separate subjective result for one candidate response."""

    status: JudgeStatus
    config: JudgeConfig | None = None
    verification: JudgeVerification | None = None
    dimensions: dict[str, JudgeDimensionScore] = Field(default_factory=dict)
    raw_score: float | None = Field(default=None, ge=1.0, le=5.0)
    normalized_score: float | None = Field(default=None, ge=0.0, le=100.0)
    attempts: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    raw_output: str | None = None
    error: str | None = None
    same_family: bool | None = None
    reliability: bool = False
    reliability_seed: int | None = None
    order_seed: int | None = None


class EvaluationResult(BaseSchema):
    """Unified item-level evaluation record."""

    schema_version: str = EVALUATION_SCHEMA_VERSION
    question_id: str = Field(min_length=1)
    response_id: str | None = None
    model_id: str = Field(min_length=1)
    model_name: str | None = None
    model_digest: str | None = None
    candidate_family: str | None = None
    category: str | None = None
    evaluation_type: EvaluationType
    evaluator: str | None = None
    evaluator_version: str | None = None
    deterministic_score: float | None = Field(default=None, ge=0.0)
    deterministic_max_score: float | None = Field(default=None, gt=0.0)
    deterministic_normalized_score: float | None = Field(default=None, ge=0.0, le=100.0)
    checks: list[CheckResult] = Field(default_factory=list)
    judge: JudgeResult | None = None
    judge_eligible: bool = False
    judge_missing: bool = True
    judge_order_seed: int | None = None
    category_score: float | None = Field(default=None, ge=0.0, le=100.0)
    composite_score: float | None = Field(default=None, ge=0.0, le=100.0)
    passed: bool | None = None
    status: EvaluationStatus
    errors: list[EvaluationError] = Field(default_factory=list)


class CategoryScoreSummary(BaseSchema):
    """Aggregated deterministic, judge, and composite score for one category."""

    category: str = Field(min_length=1)
    item_count: int = Field(ge=0)
    evaluated_count: int = Field(ge=0)
    judged_count: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    deterministic_score: float | None = Field(default=None, ge=0.0, le=100.0)
    judge_score: float | None = Field(default=None, ge=0.0, le=100.0)
    composite_score: float | None = Field(default=None, ge=0.0, le=100.0)
    judge_weight: float = Field(ge=0.0, le=1.0)
    judge_missing: bool = True


class JudgeReliabilitySummary(BaseSchema):
    """Judge health metrics, not candidate quality metrics."""

    total_calls: int = Field(ge=0)
    successful_calls: int = Field(ge=0)
    failed_calls: int = Field(ge=0)
    invalid_output_calls: int = Field(ge=0)
    timeout_calls: int = Field(ge=0)
    retry_calls: int = Field(ge=0)
    parse_failure_rate: float = Field(ge=0.0, le=1.0)
    retry_rate: float = Field(ge=0.0, le=1.0)
    reliability_sample_count: int = Field(ge=0)
    reliability_score_variance: float | None = Field(default=None, ge=0.0)


class FamilyBiasSummary(BaseSchema):
    """Descriptive comparison of same-family and cross-family judge scores."""

    same_family_count: int = Field(ge=0)
    cross_family_count: int = Field(ge=0)
    same_family_mean: float | None = Field(default=None, ge=0.0, le=100.0)
    cross_family_mean: float | None = Field(default=None, ge=0.0, le=100.0)
    difference: float | None = None


class ManualJudgeScore(BaseSchema):
    """Human-provided rubric score kept separate from judge scores."""

    question_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    scores: dict[str, Annotated[StrictInt, Field(ge=1, le=5)]]
    notes: str | None = None

    @model_validator(mode="after")
    def validate_dimensions(self) -> "ManualJudgeScore":
        required = {"relevance", "completeness", "coherence", "reasoning_quality"}
        missing = required - set(self.scores)
        unknown = set(self.scores) - required
        if missing:
            raise ValueError(f"missing manual judge dimensions: {sorted(missing)}")
        if unknown:
            raise ValueError(f"unknown manual judge dimensions: {sorted(unknown)}")
        return self


class ManualAgreementSummary(BaseSchema):
    """Agreement between manual labels and primary judge scores."""

    sample_count: int = Field(ge=0)
    dimension_exact_agreement: dict[str, float] = Field(default_factory=dict)
    dimension_within_one_agreement: dict[str, float] = Field(default_factory=dict)
    overall_exact_agreement: float = Field(ge=0.0, le=1.0)
    overall_within_one_agreement: float = Field(ge=0.0, le=1.0)
    rank_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)


class EvaluationSummary(BaseSchema):
    """Batch-level quality summary with coverage and judge health."""

    total_items: int = Field(ge=0)
    evaluated_items: int = Field(ge=0)
    quality_coverage: float = Field(ge=0.0, le=1.0)
    overall_deterministic_score: float | None = Field(default=None, ge=0.0, le=100.0)
    overall_composite_score: float | None = Field(default=None, ge=0.0, le=100.0)
    categories: dict[str, CategoryScoreSummary] = Field(default_factory=dict)
    judge_reliability: JudgeReliabilitySummary | None = None
    family_bias: FamilyBiasSummary | None = None
