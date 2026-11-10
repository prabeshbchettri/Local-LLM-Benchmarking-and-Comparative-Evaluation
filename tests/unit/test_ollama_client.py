"""
File: test_ollama_client.py

Description:
Unit tests for the direct Ollama HTTP client using a mock transport.

Responsibilities:
- Verify request paths, payloads, and response parsing.
- Verify pull progress forwarding and failure handling.
- Verify typed connection, timeout, HTTP, and parse errors.
"""

from __future__ import annotations

import json

import httpx
import pytest

from benchmark.core.exceptions import (
    ModelPullError,
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from benchmark.execution.ollama_client import OllamaClient
from benchmark.models.ollama import OllamaPullEvent

pytestmark = pytest.mark.unit

DIGEST = "a" * 64


def _model_summary() -> dict:
    return {
        "name": "llama3.1:8b",
        "model": "llama3.1:8b",
        "modified_at": "2026-11-10T00:00:00Z",
        "size": 4_920_753_328,
        "digest": DIGEST,
        "details": {
            "family": "llama",
            "parameter_size": "8.0B",
            "quantization_level": "Q4_K_M",
        },
    }


def _show_payload() -> dict:
    return {
        "details": {
            "family": "llama",
            "parameter_size": "8.0B",
            "quantization_level": "Q4_K_M",
        },
        "model_info": {
            "general.architecture": "llama",
            "general.parameter_count": 8_030_261_312,
            "llama.context_length": 131_072,
            "llama.embedding_length": 4096,
        },
        "capabilities": ["completion", "tools"],
        "license": "Llama license",
    }


def _client(handler) -> OllamaClient:
    return OllamaClient(
        "http://127.0.0.1:11434",
        timeout_seconds=5.0,
        transport=httpx.MockTransport(handler),
    )


def test_version_and_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/version"
        return httpx.Response(200, json={"version": "0.40.15"})

    client = _client(handler)
    try:
        assert client.version().version == "0.40.15"
    finally:
        client.close()

    def tags_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [_model_summary()]})

    client = _client(tags_handler)
    try:
        models = client.list_models()
        assert len(models) == 1
        assert models[0].tag == "llama3.1:8b"
        assert client.find_model("llama3.1:8b") is not None
        assert client.find_model("missing:1b") is None
    finally:
        client.close()


def test_show_and_process() -> None:
    def show_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/show"
        assert json.loads(request.content) == {"model": "llama3.1:8b"}
        return httpx.Response(200, json=_show_payload())

    client = _client(show_handler)
    try:
        show = client.show_model("llama3.1:8b")
        assert show.model_info["llama.context_length"] == 131_072
        assert show.capabilities == ["completion", "tools"]
    finally:
        client.close()

    def ps_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "llama3.1:8b",
                        "model": "llama3.1:8b",
                        "size": 5_000_000_000,
                        "size_vram": 3_000_000_000,
                        "context_length": 4096,
                    }
                ]
            },
        )

    client = _client(ps_handler)
    try:
        loaded = client.loaded_models()
        assert loaded[0].size_vram == 3_000_000_000
    finally:
        client.close()


def test_load_and_unload_payloads() -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"model": "llama3.1:8b", "done": True, "done_reason": "load"},
        )

    client = _client(handler)
    try:
        client.load_model("llama3.1:8b", "30m")
        client.unload_model("llama3.1:8b")
    finally:
        client.close()
    assert payloads[0]["keep_alive"] == "30m"
    assert payloads[0]["stream"] is False
    assert payloads[1]["keep_alive"] == 0


def test_chat_payload_and_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["model"] == "qwen2.5:3b"
        assert payload["messages"] == [{"role": "user", "content": "Reply OK"}]
        assert payload["options"]["num_predict"] == 8
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "message": {"role": "assistant", "content": "OK"},
                "done": True,
                "eval_count": 2,
                "eval_duration": 1_000_000_000,
            },
        )

    client = _client(handler)
    try:
        response = client.chat(
            "qwen2.5:3b",
            [{"role": "user", "content": "Reply OK"}],
            options={"num_predict": 8},
            keep_alive="30m",
        )
    finally:
        client.close()
    assert response.message.content == "OK"
    assert response.eval_count == 2


def test_pull_stream_forwards_progress() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/pull"
        assert json.loads(request.content)["model"] == "qwen2.5:7b-instruct"
        content = (
            b'{"status":"pulling manifest"}\n'
            b'{"status":"downloading","digest":"sha256:abc","total":100,"completed":50}\n'
            b'{"status":"success","digest":"sha256:abc"}\n'
        )
        return httpx.Response(200, content=content)

    events: list[OllamaPullEvent] = []
    client = _client(handler)
    try:
        result = client.pull_model("qwen2.5:7b-instruct", progress=events.append)
    finally:
        client.close()
    assert result.success
    assert result.events == 3
    assert len(events) == 3
    assert events[1].completed == 50


def test_pull_error_event_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"status":"pulling manifest"}\n{"error":"download failed"}\n',
        )

    client = _client(handler)
    try:
        with pytest.raises(ModelPullError, match="download failed"):
            client.pull_model("qwen2.5:7b-instruct")
    finally:
        client.close()


def test_http_error_includes_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    client = _client(handler)
    try:
        with pytest.raises(OllamaHTTPError) as exc_info:
            client.show_model("missing:1b")
    finally:
        client.close()
    assert exc_info.value.status_code == 404
    assert "model not found" in str(exc_info.value)


def test_connection_and_timeout_errors_are_typed() -> None:
    def connect_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _client(connect_handler)
    try:
        with pytest.raises(OllamaConnectionError):
            client.version()
    finally:
        client.close()

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = _client(timeout_handler)
    try:
        with pytest.raises(OllamaTimeoutError):
            client.version()
    finally:
        client.close()


def test_invalid_json_raises_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    client = _client(handler)
    try:
        with pytest.raises(OllamaResponseError):
            client.version()
    finally:
        client.close()


def test_invalid_schema_raises_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True})

    client = _client(handler)
    try:
        with pytest.raises(OllamaResponseError, match="invalid version"):
            client.version()
    finally:
        client.close()
