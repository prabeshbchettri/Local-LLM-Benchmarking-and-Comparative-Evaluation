"""
File: test_engine.py

Description:
Tests for deterministic dispatch, unified results, and judge integration.

Responsibilities:
- Verify registry selection and unsupported-method errors.
- Verify skipped responses do not receive imputed zero scores.
- Verify judge scores remain separate and failed judges never override deterministic results.
- Verify stored inference records convert into evaluation requests without losing identity.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from benchmark.core.exceptions import EvaluationConfigurationError
from benchmark.dataset.models import EvaluationType
from benchmark.evaluation.deterministic.registry import build_default_registry
from benchmark.evaluation.engine import EvaluationEngine, EvaluationRequest
from benchmark.evaluation.models import (
    EvaluationStatus,
    JudgeConfig,
    JudgeDimensionScore,
    JudgeResult,
    JudgeStatus,
)
from benchmark.models.inference import (
    GenerationOptions,
    InferenceProfile,
    ResponseRecord,
    ResponseStatus,
    TimingMetrics,
    TokenMetrics,
)

pytestmark = [pytest.mark.unit, pytest.mark.evaluation]


def _judge_config() -> JudgeConfig:
    return JudgeConfig(
        name="Qwen 2.5 3B Judge",
        tag="qwen2.5:3b",
        digest="a" * 64,
        family="qwen2",
        rubric_version="1.0.0",
        prompt_version="1.0.0",
        temperature=0.0,
        seed=42,
        top_p=1.0,
        top_k=40,
        repeat_penalty=1.1,
        num_ctx=4096,
        num_predict=300,
    )


def _judge_success(score: int = 3) -> JudgeResult:
    dimensions = {
        name: JudgeDimensionScore(score=score, rationale="test rationale")
        for name in ("relevance", "completeness", "coherence", "reasoning_quality")
    }
    normalized = (score - 1) / 4 * 100
    return JudgeResult(
        status=JudgeStatus.SUCCESS,
        config=_judge_config(),
        dimensions=dimensions,
        raw_score=float(score),
        normalized_score=normalized,
        attempts=1,
    )


class FakeJudge:
    def __init__(self, result: JudgeResult) -> None:
        self.result = result
        self.config = result.config or _judge_config()
        self.requests = []
        self.calls = []

    def judge(self, request, **kwargs):
        self.requests.append(request)
        self.calls.append(kwargs)
        reliability = bool(kwargs.get("reliability", False))
        return self.result.model_copy(
            update={
                "reliability": reliability,
                "reliability_seed": kwargs.get("seed") if reliability else None,
            }
        )


def test_registry_dispatches_each_supported_type(item_for) -> None:
    registry = build_default_registry()
    result = registry.evaluate(item_for("mr-001"), "96")
    assert result.evaluation_type.value == "numeric"
    assert result.passed is True


def test_registry_rejects_composite_without_components(item_for) -> None:
    with pytest.raises(EvaluationConfigurationError, match="composite"):
        build_default_registry().get(EvaluationType("composite"))


def test_engine_keeps_deterministic_and_judge_separate(item_for) -> None:
    item = item_for("sum-001")
    judge = FakeJudge(_judge_success(3))
    engine = EvaluationEngine(judge=judge)
    request = EvaluationRequest(
        item=item,
        response_text=" ".join(item.key_facts),
        model_id="llama3.1:8b",
        candidate_family="llama",
    )
    result = engine.evaluate(request)
    assert result.status.value == "passed"
    assert result.deterministic_normalized_score is not None
    assert result.judge is not None
    assert result.judge.normalized_score == 50.0
    assert result.judge.same_family is False
    expected_composite = (
        (1.0 - 0.4) * result.deterministic_normalized_score + 0.4 * 50.0
    )
    assert result.composite_score == pytest.approx(expected_composite)


def test_engine_does_not_turn_deterministic_failure_into_pass(item_for) -> None:
    item = item_for("sum-001")
    engine = EvaluationEngine(judge=FakeJudge(_judge_success(5)))
    result = engine.evaluate(
        EvaluationRequest(item=item, response_text="unrelated text", model_id="llama3.1:8b")
    )
    assert result.passed is False
    assert result.status.value in {"failed", "partial"}


def test_engine_records_judge_failure_without_zeroing_it(item_for) -> None:
    item = item_for("sum-001")
    failed = JudgeResult(
        status=JudgeStatus.INVALID_OUTPUT,
        config=_judge_config(),
        error="invalid output after retry",
        attempts=2,
        retry_count=1,
    )
    engine = EvaluationEngine(judge=FakeJudge(failed))
    result = engine.evaluate(
        EvaluationRequest(
            item=item,
            response_text=" ".join(item.key_facts),
            model_id="llama3.1:8b",
        )
    )
    assert result.judge is not None
    assert result.judge.status is JudgeStatus.INVALID_OUTPUT
    assert result.judge.normalized_score is None
    assert result.judge_missing is True
    assert result.composite_score == result.deterministic_normalized_score
    assert result.errors[0].stage == "judge"


def test_engine_flags_unconfigured_judge(item_for) -> None:
    item = item_for("sum-001")
    engine = EvaluationEngine()
    result = engine.evaluate(
        EvaluationRequest(
            item=item,
            response_text=" ".join(item.key_facts),
            model_id="llama3.1:8b",
        )
    )
    assert result.judge is not None
    assert result.judge.status is JudgeStatus.NOT_CONFIGURED
    assert result.judge_missing is True
    assert result.errors[0].error_type == "not_configured"


def test_engine_skips_non_success_response_without_zero(item_for) -> None:
    engine = EvaluationEngine()
    result = engine.evaluate(
        EvaluationRequest(
            item=item_for("gk-001"),
            response_text="",
            model_id="llama3.1:8b",
            response_status=ResponseStatus.TIMEOUT,
        )
    )
    assert result.status.value == "skipped"
    assert result.deterministic_score is None
    assert result.category_score is None
    assert result.composite_score is None


def test_engine_exact_match_result(item_for) -> None:
    engine = EvaluationEngine()
    result = engine.evaluate(
        EvaluationRequest(
            item=item_for("gk-001"),
            response_text="Paris",
            model_id="llama3.1:8b",
        )
    )
    assert result.status.value == "passed"
    assert result.deterministic_score == 1.0
    assert result.category_score == 100.0


def test_engine_reliability_uses_configured_temperature_and_seeds(item_for) -> None:
    item = item_for("sum-001")
    judge = FakeJudge(_judge_success(3))
    engine = EvaluationEngine(judge=judge)
    request = EvaluationRequest(
        item=item,
        response_text=" ".join(item.key_facts),
        model_id="llama3.1:8b",
    )
    results = engine.evaluate_reliability(
        [request],
        samples=3,
        fraction=1.0,
        temperature=0.7,
        seed=100,
    )
    assert len(results) == 3
    assert [call["temperature"] for call in judge.calls] == [0.7, 0.7, 0.7]
    assert [call["seed"] for call in judge.calls] == [100, 101, 102]
    assert all(call["reliability"] is True for call in judge.calls)
    assert all(result.judge is not None for result in results)
    assert all(result.judge.reliability for result in results if result.judge)
    assert all(
        result.judge.reliability_seed in {100, 101, 102}
        for result in results
        if result.judge
    )


def test_engine_reliability_rejects_empty_sample(item_for) -> None:
    engine = EvaluationEngine()
    request = EvaluationRequest(
        item=item_for("code-001"),
        response_text="def add(a, b):\n    return a + b",
        model_id="llama3.1:8b",
    )
    assert engine.select_reliability_sample([request], fraction=1.0) == []


def _response(
    *,
    text: str,
    status: ResponseStatus = ResponseStatus.SUCCESS,
) -> ResponseRecord:
    timestamp = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
    return ResponseRecord(
        experiment_id="20261110T120000Z-test0001",
        model_name="Llama 3.1 8B Instruct",
        model_tag="llama3.1:8b",
        model_digest="a" * 64,
        item_id="sum-001",
        category="summarization",
        profile=InferenceProfile.QUALITY,
        repetition=1,
        prompt="Summarize the honeybee passage in one sentence.",
        response_text=text,
        status=status,
        started_at=timestamp,
        completed_at=timestamp,
        generation_options=GenerationOptions(),
        timing=TimingMetrics(total_latency_seconds=1.5),
        tokens=TokenMetrics(),
    )


def test_request_from_response_preserves_candidate_identity(item_for) -> None:
    item = item_for("sum-001")
    record = _response(text=" ".join(item.key_facts))
    request = EvaluationRequest.from_response(
        item,
        record,
        candidate_family="llama",
    )
    assert request.model_id == "llama3.1:8b"
    assert request.model_name == "Llama 3.1 8B Instruct"
    assert request.model_digest == "a" * 64
    assert request.candidate_family == "llama"
    assert request.response_status is ResponseStatus.SUCCESS
    assert request.response_id == record.response_id
    assert request.response_text == record.response_text


def test_request_from_response_skips_timeout(item_for) -> None:
    request = EvaluationRequest.from_response(
        item_for("sum-001"),
        _response(text="", status=ResponseStatus.TIMEOUT),
    )
    result = EvaluationEngine().evaluate(request)
    assert result.status is EvaluationStatus.SKIPPED
    assert result.errors[0].error_type == "non_success_response"
    assert result.composite_score is None


def test_engine_judge_request_excludes_candidate_identity(item_for) -> None:
    judge = FakeJudge(_judge_success(3))
    item = item_for("sum-001")
    EvaluationEngine(judge=judge).evaluate(
        EvaluationRequest(
            item=item,
            response_text=" ".join(item.key_facts),
            model_id="llama3.1:8b",
            model_name="Llama 3.1 8B Instruct",
            candidate_family="llama",
        )
    )
    payload = judge.requests[0].model_dump_json()
    assert "llama3.1:8b" not in payload
    assert "Llama 3.1" not in payload


def test_manual_validation_sample_uses_fraction_and_skips_non_eligible(item_for) -> None:
    sum_item = item_for("sum-001")
    requests = [
        EvaluationRequest(
            item=sum_item,
            response_text=" ".join(sum_item.key_facts),
            model_id="llama3.1:8b",
        ),
        EvaluationRequest(
            item=item_for("gk-001"),
            response_text="Paris",
            model_id="llama3.1:8b",
        ),
    ]
    engine = EvaluationEngine()
    sample = engine.select_manual_validation_sample(requests, fraction=1.0)
    assert [request.item.id for request in sample] == ["sum-001"]
    assert engine.select_manual_validation_sample(requests, fraction=0.0) == []
