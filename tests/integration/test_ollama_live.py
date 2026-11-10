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

import pytest

from benchmark.core.config import AppSettings
from benchmark.execution.ollama_client import OllamaClient

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
