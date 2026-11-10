"""
File: test_inference_engine.py

Description:
Unit tests for streaming inference records, failures, cancellation, and JSONL output.

Responsibilities:
- Verify client timing, token metrics, and raw metadata capture.
- Verify empty, malformed, timeout, connection, and cancellation records.
- Verify missing models fail before a request starts.
- Verify response records preserve raw text and append as valid JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.core.exceptions import (
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from benchmark.execution.engine import (
    CancellationToken,
    InferenceEngine,
    JsonlResponseWriter,
    utc_now,
)
from benchmark.models.inference import (
    GenerationOptions,
    InferenceProfile,
    ResponseRecord,
    ResponseStatus,
    TimingMetrics,
    TokenMetrics,
)
from benchmark.models.ollama import OllamaChatChunk, OllamaModelSummary

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        try:
            return next(self._values)
        except StopIteration as exc:
            raise AssertionError("fake clock exhausted") from exc


class FakeClient:
    def __init__(
        self,
        chunks: list[OllamaChatChunk] | None = None,
        *,
        error: Exception | None = None,
        summary: OllamaModelSummary | None = None,
    ) -> None:
        self.chunks = chunks or []
        self.error = error
        self.summary = summary
        self.calls: list[dict] = []

    def find_model(self, tag: str) -> OllamaModelSummary | None:
        return self.summary

    def stream_chat(self, tag, messages, options=None, keep_alive=None):
        self.calls.append(
            {
                "tag": tag,
                "messages": messages,
                "options": options,
                "keep_alive": keep_alive,
            }
        )
        if self.error is not None:
            raise self.error
        yield from self.chunks


class CancellingClient(FakeClient):
    def __init__(self, chunks: list[OllamaChatChunk], token: CancellationToken) -> None:
        super().__init__(chunks)
        self.token = token

    def stream_chat(self, tag, messages, options=None, keep_alive=None):
        self.calls.append(
            {
                "tag": tag,
                "messages": messages,
                "options": options,
                "keep_alive": keep_alive,
            }
        )
        yield self.chunks[0]
        self.token.cancel()
        yield self.chunks[1]


def _summary() -> OllamaModelSummary:
    return OllamaModelSummary(
        name="llama3.1:8b",
        model="llama3.1:8b",
        digest="a" * 64,
    )


def _chunk(content: str = "", done: bool = False, **metadata) -> OllamaChatChunk:
    payload = {
        "model": "llama3.1:8b",
        "message": {"role": "assistant", "content": content},
        "done": done,
        **metadata,
    }
    return OllamaChatChunk.model_validate({**payload, "raw": payload})


def _engine(
    settings,
    chunks: list[OllamaChatChunk] | None = None,
    *,
    error: Exception | None = None,
    summary: OllamaModelSummary | None = None,
    clock_values: list[int] | None = None,
    client: FakeClient | None = None,
) -> tuple[InferenceEngine, FakeClient]:
    fake_client = client or FakeClient(chunks, error=error, summary=summary or _summary())
    engine = InferenceEngine(
        settings,
        fake_client,
        clock_ns=FakeClock(clock_values or [0, 100_000_000, 200_000_000]),
    )
    return engine, fake_client


def _success_chunks() -> list[OllamaChatChunk]:
    return [
        _chunk(""),
        _chunk("Hel"),
        _chunk(
            "lo",
            True,
            done_reason="stop",
            total_duration=400_000_000,
            load_duration=100_000_000,
            prompt_eval_count=4,
            prompt_eval_duration=200_000_000,
            eval_count=10,
            eval_duration=500_000_000,
        ),
    ]


def test_success_record_captures_timing_tokens_and_metadata(settings) -> None:
    engine, client = _engine(
        settings,
        _success_chunks(),
        clock_values=[0, 50_000_000, 100_000_000, 300_000_000],
    )
    record = engine.run_request(
        "llama3.1:8b",
        "  prompt  ",
        experiment_id="exp-1",
        item_id="gk-001",
        category="general_knowledge",
        repetition=2,
        profile=InferenceProfile.SMOKE,
        system_instruction="  system  ",
        max_tokens=64,
    )
    assert record.status is ResponseStatus.SUCCESS
    assert record.response_text == "Hello"
    assert record.prompt == "  prompt  "
    assert record.system_prompt == "  system  "
    assert record.timing.ttft_seconds == pytest.approx(0.1)
    assert record.timing.total_latency_seconds == pytest.approx(0.3)
    assert record.timing.load_duration_seconds == pytest.approx(0.1)
    assert record.tokens.prompt_eval_count == 4
    assert record.tokens.eval_count == 10
    assert record.tokens.output_tokens_per_second == pytest.approx(20.0)
    assert record.tokens.prompt_tokens_per_second == pytest.approx(20.0)
    assert record.stream_chunk_count == 3
    assert record.generation_options.num_predict == 64
    assert record.model_digest == "a" * 64
    assert record.ollama_metadata["done_reason"] == "stop"
    assert client.calls[0]["options"]["num_predict"] == 64
    assert client.calls[0]["messages"] == [
        {"role": "system", "content": "  system  "},
        {"role": "user", "content": "  prompt  "},
    ]


def test_empty_response_is_recorded_as_empty(settings) -> None:
    engine, _ = _engine(
        settings,
        [_chunk("", True)],
        clock_values=[0, 100_000_000],
    )
    record = engine.run_request("llama3.1:8b", "prompt", experiment_id="exp-1")
    assert record.status is ResponseStatus.EMPTY
    assert record.error_type == "EmptyResponseError"
    assert record.timing.ttft_seconds is None
    assert record.timing.total_latency_seconds == pytest.approx(0.1)
    assert record.response_text == ""


def test_missing_final_chunk_is_a_typed_stream_error(settings) -> None:
    engine, _ = _engine(
        settings,
        [_chunk("Partial")],
        clock_values=[0, 50_000_000, 100_000_000],
    )
    record = engine.run_request("llama3.1:8b", "prompt", experiment_id="exp-1")
    assert record.status is ResponseStatus.ERROR
    assert record.error_type == "StreamParseError"
    assert record.response_text == "Partial"
    assert record.timing.ttft_seconds == pytest.approx(0.05)
    assert record.timing.total_latency_seconds == pytest.approx(0.1)


def test_timeout_is_recorded_distinctly(settings) -> None:
    engine, _ = _engine(
        settings,
        error=OllamaTimeoutError("slow"),
        clock_values=[0, 1_000_000_000],
    )
    record = engine.run_request("llama3.1:8b", "prompt", experiment_id="exp-1")
    assert record.status is ResponseStatus.TIMEOUT
    assert record.error_type == "InferenceTimeoutError"
    assert record.timing.ttft_seconds is None
    assert record.timing.total_latency_seconds == pytest.approx(1.0)


def test_connection_and_stream_errors_are_recorded(settings) -> None:
    engine, _ = _engine(
        settings,
        error=OllamaConnectionError("down"),
        clock_values=[0, 10_000_000],
    )
    record = engine.run_request("llama3.1:8b", "prompt", experiment_id="exp-1")
    assert record.status is ResponseStatus.ERROR
    assert record.error_type == "OllamaConnectionError"

    engine, _ = _engine(
        settings,
        error=OllamaResponseError("bad stream"),
        clock_values=[0, 10_000_000],
    )
    record = engine.run_request("llama3.1:8b", "prompt", experiment_id="exp-1")
    assert record.status is ResponseStatus.ERROR
    assert record.error_type == "OllamaResponseError"


def test_cancellation_before_request_skips_streaming(settings) -> None:
    token = CancellationToken()
    token.cancel()
    engine, client = _engine(settings, clock_values=[0])
    record = engine.run_request(
        "llama3.1:8b",
        "prompt",
        experiment_id="exp-1",
        cancellation=token,
    )
    assert record.status is ResponseStatus.CANCELLED
    assert record.error_type == "InferenceCancelledError"
    assert record.timing.total_latency_seconds == 0.0
    assert client.calls == []


def test_cancellation_during_stream_keeps_partial_text(settings) -> None:
    token = CancellationToken()
    client = CancellingClient([_chunk("A"), _chunk("B")], token)
    client.summary = _summary()
    engine, _ = _engine(
        settings,
        client=client,
        clock_values=[0, 50_000_000, 100_000_000, 150_000_000],
    )
    record = engine.run_request(
        "llama3.1:8b",
        "prompt",
        experiment_id="exp-1",
        cancellation=token,
    )
    assert record.status is ResponseStatus.CANCELLED
    assert record.response_text == "AB"
    assert record.timing.ttft_seconds == pytest.approx(0.05)
    assert record.timing.total_latency_seconds == pytest.approx(0.15)


def test_missing_model_fails_before_streaming(settings) -> None:
    engine, client = _engine(settings, clock_values=[0], summary=None)
    client.summary = None
    with pytest.raises(ModelNotFoundError):
        engine.run_request("llama3.1:8b", "prompt", experiment_id="exp-1")
    assert client.calls == []


def test_zero_token_duration_produces_no_rate(settings) -> None:
    engine, _ = _engine(
        settings,
        [
            _chunk(
                "X",
                True,
                eval_count=5,
                eval_duration=0,
                prompt_eval_count=3,
                prompt_eval_duration=0,
            )
        ],
        clock_values=[0, 200_000_000],
    )
    record = engine.run_request("llama3.1:8b", "prompt", experiment_id="exp-1")
    assert record.tokens.output_tokens_per_second is None
    assert record.tokens.prompt_tokens_per_second is None
    assert record.timing.eval_duration_seconds == 0.0


def test_jsonl_writer_appends_valid_records(tmp_path: Path) -> None:
    record = ResponseRecord(
        experiment_id="exp-1",
        model_tag="llama3.1:8b",
        profile=InferenceProfile.SMOKE,
        repetition=1,
        prompt="prompt",
        response_text="  answer  ",
        status=ResponseStatus.SUCCESS,
        started_at=utc_now(),
        completed_at=utc_now(),
        generation_options=GenerationOptions(),
        timing=TimingMetrics(total_latency_seconds=0.5),
        tokens=TokenMetrics(eval_count=5),
    )
    path = tmp_path / "nested" / "responses.jsonl"
    writer = JsonlResponseWriter(path)
    writer.append(record)
    writer.append(record.model_copy(update={"response_id": "second"}))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["response_text"] == "  answer  "
    assert first["status"] == "success"
    assert first["timing"]["total_latency_seconds"] == 0.5
    assert json.loads(lines[1])["response_id"] == "second"
