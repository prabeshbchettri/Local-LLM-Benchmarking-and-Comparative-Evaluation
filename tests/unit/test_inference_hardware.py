"""
File: test_inference_hardware.py

Description:
Unit tests for hardware monitoring integration with inference execution.

Responsibilities:
- Verify monitor start, request marker, and stop calls.
- Verify the hardware summary is attached to the response record.
- Verify monitor failures do not fail the inference request.
"""

from __future__ import annotations

import pytest

from benchmark.execution.engine import InferenceEngine
from benchmark.models.hardware import HardwareSummary
from benchmark.models.inference import ResponseStatus
from benchmark.models.ollama import OllamaChatChunk, OllamaLoadedModel, OllamaModelSummary

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


class FakeClient:
    def find_model(self, tag: str) -> OllamaModelSummary:
        return OllamaModelSummary(name=tag, model=tag, digest="a" * 64)

    def loaded_models(self) -> list[OllamaLoadedModel]:
        return [
            OllamaLoadedModel(
                name="llama3.1:8b",
                model="llama3.1:8b",
                size=5 * 1024**3,
                size_vram=3 * 1024**3,
            )
        ]

    def stream_chat(self, tag, messages, options=None, keep_alive=None):
        payload = {
            "model": tag,
            "message": {"role": "assistant", "content": "OK"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 3,
            "prompt_eval_duration": 100_000_000,
            "eval_count": 1,
            "eval_duration": 100_000_000,
        }
        yield OllamaChatChunk.model_validate({**payload, "raw": payload})


class FakeMonitor:
    def __init__(
        self,
        *,
        stop_error: bool = False,
        baseline_seconds: float = 2.0,
    ) -> None:
        self.started = False
        self.marked = False
        self.stopped = False
        self.stop_error = stop_error
        self.baseline_seconds = baseline_seconds

    def start(self) -> None:
        self.started = True

    def mark_request_start(self) -> None:
        self.marked = True

    def stop(self) -> HardwareSummary:
        self.stopped = True
        if self.stop_error:
            raise RuntimeError("stop failed")
        return HardwareSummary(
            sampling_interval_ms=1000,
            baseline_seconds=self.baseline_seconds,
            sample_count=0,
        )


class EmptyLoadedClient(FakeClient):
    def loaded_models(self) -> list[OllamaLoadedModel]:
        return []


class FailingLoadedClient(FakeClient):
    def loaded_models(self) -> list[OllamaLoadedModel]:
        raise RuntimeError("/api/ps failed")


def _engine(settings, factory, *, sleeps: list[float] | None = None, client=None):
    return InferenceEngine(
        settings,
        client or FakeClient(),
        clock_ns=FakeClock([0, 50_000_000, 100_000_000, 300_000_000]),
        hardware_factory=factory,
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
    )


def test_inference_attaches_hardware_summary(settings) -> None:
    monitor = FakeMonitor()
    sleeps: list[float] = []
    engine = _engine(settings, lambda: monitor, sleeps=sleeps)
    record = engine.run_request("llama3.1:8b", "prompt", experiment_id="exp-1")
    assert record.status is ResponseStatus.SUCCESS
    assert record.hardware is not None
    assert record.hardware.sampling_interval_ms == 1000
    assert record.hardware.ollama_model_vram_available is True
    assert record.hardware.ollama_model_name == "llama3.1:8b"
    assert record.hardware.ollama_model_size_bytes == 5 * 1024**3
    assert record.hardware.ollama_model_size_vram_bytes == 3 * 1024**3
    assert record.hardware.ollama_model_vram_ratio == pytest.approx(0.6)
    assert record.hardware_error is None
    assert monitor.started is True
    assert monitor.marked is True
    assert monitor.stopped is True
    assert sleeps == [settings.benchmark.hardware_baseline_seconds]


def test_uses_sampler_baseline_override(settings) -> None:
    monitor = FakeMonitor(baseline_seconds=1.0)
    sleeps: list[float] = []
    engine = _engine(settings, lambda: monitor, sleeps=sleeps)
    record = engine.run_request("llama3.1:8b", "prompt", experiment_id="exp-1")
    assert record.status is ResponseStatus.SUCCESS
    assert sleeps == [1.0]


def test_missing_model_vram_is_recorded(settings) -> None:
    monitor = FakeMonitor()
    engine = _engine(settings, lambda: monitor, client=EmptyLoadedClient())
    record = engine.run_request("llama3.1:8b", "prompt", experiment_id="exp-1")
    assert record.status is ResponseStatus.SUCCESS
    assert record.hardware is not None
    assert record.hardware.ollama_model_vram_available is False
    assert record.hardware.ollama_model_vram_error is not None
    assert "not listed" in record.hardware.ollama_model_vram_error


def test_model_vram_read_failure_is_recorded(settings) -> None:
    monitor = FakeMonitor()
    engine = _engine(settings, lambda: monitor, client=FailingLoadedClient())
    record = engine.run_request("llama3.1:8b", "prompt", experiment_id="exp-1")
    assert record.status is ResponseStatus.SUCCESS
    assert record.hardware is not None
    assert record.hardware.ollama_model_vram_available is False
    assert record.hardware.ollama_model_vram_error is not None
    assert "/api/ps failed" in record.hardware.ollama_model_vram_error
    assert record.hardware_error is None


def test_monitor_start_failure_does_not_fail_request(settings) -> None:
    def fail():
        raise RuntimeError("sensor unavailable")

    engine = _engine(settings, fail)
    record = engine.run_request("llama3.1:8b", "prompt", experiment_id="exp-1")
    assert record.status is ResponseStatus.SUCCESS
    assert record.hardware is None
    assert record.hardware_error is not None
    assert "sensor unavailable" in record.hardware_error


def test_monitor_stop_failure_does_not_fail_request(settings) -> None:
    monitor = FakeMonitor(stop_error=True)
    engine = _engine(settings, lambda: monitor)
    record = engine.run_request("llama3.1:8b", "prompt", experiment_id="exp-1")
    assert record.status is ResponseStatus.SUCCESS
    assert record.hardware is None
    assert record.hardware_error is not None
    assert "stop failed" in record.hardware_error
