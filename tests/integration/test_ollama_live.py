"""
File: test_ollama_live.py

Description:
Optional live checks against the local Ollama server.

Responsibilities:
- Verify version and model discovery when explicitly enabled.
- Never pull or modify models.
"""

from __future__ import annotations

import os
import time

import pytest

from benchmark.core.config import AppSettings
from benchmark.execution.ollama_client import OllamaClient
from benchmark.monitoring.sampler import HardwareSampler

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ollama,
    pytest.mark.skipif(
        os.getenv("BENCHMARK_LIVE_OLLAMA") != "1",
        reason="set BENCHMARK_LIVE_OLLAMA=1 to run live Ollama checks",
    ),
]


def test_live_version_and_tags(settings: AppSettings) -> None:
    client = OllamaClient(
        str(settings.models.ollama_base_url),
        timeout_seconds=10.0,
    )
    try:
        assert client.version().version
        models = client.list_models()
        assert isinstance(models, list)
    finally:
        client.close()


def test_live_streaming_chat_returns_tokens_and_metrics(settings: AppSettings) -> None:
    """Exercise the real NDJSON stream and Ollama-reported timing metadata."""
    client = OllamaClient(
        str(settings.models.ollama_base_url),
        timeout_seconds=180.0,
    )
    try:
        tag = settings.models.candidates[0].tag
        installed = client.find_model(tag)
        if installed is None:
            pytest.skip(f"{tag} is not installed")
        chunks = list(
            client.stream_chat(
                tag,
                [{"role": "user", "content": "Reply with the single word OK."}],
                options={"num_predict": 8, "temperature": 0.0, "num_ctx": 512},
                keep_alive="1m",
            )
        )
        assert chunks
        final = chunks[-1]
        assert final.done is True
        assert final.eval_count is not None and final.eval_count >= 1
        assert final.eval_duration is not None and final.eval_duration > 0
        text = "".join(chunk.message.content for chunk in chunks if chunk.message)
        assert text.strip()
    finally:
        client.close()


def test_live_hardware_sampler_records_samples() -> None:
    """Verify real psutil/NVML sampling produces baseline and inference samples."""
    sampler = HardwareSampler(interval_seconds=0.25, baseline_seconds=0.25)
    sampler.start()
    try:
        time.sleep(0.4)
        sampler.mark_request_start()
        time.sleep(0.5)
    finally:
        summary = sampler.stop()
    assert summary.sample_count >= 2
    assert summary.inference.sample_count >= 1
    assert summary.sampler_overhead_avg_ms is not None
    assert summary.ram_total_bytes is not None or summary.gpu_available is False
