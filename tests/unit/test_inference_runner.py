"""
File: test_inference_runner.py

Description:
Unit tests for sequential inference batches, warm-up, cold start, and cancellation.

Responsibilities:
- Verify model and prompt execution order.
- Verify warm-up records are stored but excluded from measured counts.
- Verify cold start unloads before each request.
- Verify cooperative cancellation stops the batch and still unloads.
"""

from __future__ import annotations

from benchmark.execution.engine import (
    CancellationToken,
    InferenceBatchResult,
    InferenceRunner,
    utc_now,
)
from benchmark.models.inference import (
    GenerationOptions,
    InferenceProfile,
    PromptSpec,
    ResponseRecord,
    ResponseStatus,
    RunStatus,
    TimingMetrics,
    TokenMetrics,
)

import pytest

pytestmark = pytest.mark.unit


class FakeManager:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def load(self, identifier: str) -> None:
        self.events.append(("load", identifier))

    def unload(self, identifier: str) -> None:
        self.events.append(("unload", identifier))


class FakeEngine:
    interrupted = False

    def __init__(
        self,
        *,
        cancellation: CancellationToken | None = None,
        cancel_after: int | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self.cancellation = cancellation
        self.cancel_after = cancel_after

    def run_request(self, model_tag, prompt, **kwargs) -> ResponseRecord:
        self.calls.append({"model_tag": model_tag, "prompt": prompt, **kwargs})
        if (
            self.cancellation is not None
            and self.cancel_after is not None
            and len(self.calls) == self.cancel_after
        ):
            self.cancellation.cancel()
        return ResponseRecord(
            experiment_id=kwargs["experiment_id"],
            model_tag=model_tag,
            item_id=kwargs["item_id"],
            profile=kwargs["profile"],
            repetition=kwargs["repetition"],
            is_warmup=kwargs["is_warmup"],
            cold_start=kwargs["cold_start"],
            prompt=prompt,
            response_text="ok",
            status=ResponseStatus.SUCCESS,
            started_at=utc_now(),
            completed_at=utc_now(),
            generation_options=GenerationOptions(),
            timing=TimingMetrics(total_latency_seconds=0.1),
            tokens=TokenMetrics(eval_count=1),
        )


def _prompts() -> list[PromptSpec]:
    return [
        PromptSpec(item_id="q1", prompt="one"),
        PromptSpec(item_id="q2", prompt="two"),
    ]


def test_runner_executes_models_and_prompts_sequentially() -> None:
    engine = FakeEngine()
    manager = FakeManager()
    sleeps: list[float] = []
    runner = InferenceRunner(engine, manager=manager, sleep=sleeps.append)
    result = runner.run(
        ["a:1b", "b:2b"],
        _prompts(),
        experiment_id="exp-1",
        cooldown_seconds=2.0,
    )
    assert [
        (call["model_tag"], call["item_id"], call["repetition"], call["is_warmup"])
        for call in engine.calls
    ] == [
        ("a:1b", "q1", 1, False),
        ("a:1b", "q2", 1, False),
        ("b:2b", "q1", 1, False),
        ("b:2b", "q2", 1, False),
    ]
    assert manager.events == [
        ("load", "a:1b"),
        ("unload", "a:1b"),
        ("load", "b:2b"),
        ("unload", "b:2b"),
    ]
    assert sleeps == [2.0]
    assert result.summary.status is RunStatus.COMPLETED
    assert result.summary.total_requests == 4
    assert result.summary.measured_requests == 4
    assert result.summary.success_count == 4


def test_warmup_is_stored_but_excluded_from_measured_counts() -> None:
    engine = FakeEngine()
    runner = InferenceRunner(engine)
    result = runner.run(
        ["a:1b"],
        _prompts(),
        experiment_id="exp-1",
        warmup_requests=1,
        repetitions=1,
    )
    assert len(result.records) == 3
    assert result.records[0].is_warmup is True
    assert result.summary.total_requests == 3
    assert result.summary.measured_requests == 2
    assert result.summary.warmup_count == 1
    assert result.summary.success_count == 2


def test_cold_start_unloads_before_each_measured_request() -> None:
    engine = FakeEngine()
    manager = FakeManager()
    runner = InferenceRunner(engine, manager=manager)
    result = runner.run(
        ["a:1b"],
        _prompts(),
        experiment_id="exp-1",
        warmup_requests=1,
        cold_start=True,
    )
    assert result.summary.warmup_requests == 0
    assert result.summary.warmup_count == 0
    assert manager.events == [
        ("unload", "a:1b"),
        ("unload", "a:1b"),
        ("unload", "a:1b"),
    ]


def test_cancellation_stops_after_current_request_and_unloads() -> None:
    token = CancellationToken()
    engine = FakeEngine(cancellation=token, cancel_after=1)
    manager = FakeManager()
    runner = InferenceRunner(engine, manager=manager, cancellation=token)
    result: InferenceBatchResult = runner.run(
        ["a:1b"],
        _prompts(),
        experiment_id="exp-1",
        repetitions=1,
    )
    assert len(engine.calls) == 1
    assert result.summary.status is RunStatus.CANCELLED
    assert result.summary.total_requests == 1
    assert manager.events == [("load", "a:1b"), ("unload", "a:1b")]


def test_progress_callback_receives_each_record() -> None:
    engine = FakeEngine()
    runner = InferenceRunner(engine)
    seen: list[str] = []
    result = runner.run(
        ["a:1b"],
        _prompts(),
        experiment_id="exp-1",
        progress=lambda record: seen.append(record.item_id or ""),
    )
    assert seen == ["q1", "q2"]
    assert len(result.records) == 2
