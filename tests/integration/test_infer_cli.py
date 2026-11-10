"""
File: test_infer_cli.py

Description:
Integration test for the benchmark infer command using a fake Ollama client.

Responsibilities:
- Verify the CLI loads a dataset item and runs the inference pipeline.
- Verify response records are written to JSONL.
- Verify JSON summary output is machine-readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmark.__main__ as main_module
from benchmark.models.ollama import OllamaChatChunk, OllamaModelSummary

pytestmark = pytest.mark.integration


class FakeStreamingClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def find_model(self, tag: str) -> OllamaModelSummary:
        return OllamaModelSummary(
            name=tag,
            model=tag,
            digest="a" * 64,
        )

    def stream_chat(self, tag, messages, options=None, keep_alive=None):
        self.requests.append(
            {
                "tag": tag,
                "messages": messages,
                "options": options,
                "keep_alive": keep_alive,
            }
        )
        payload = {
            "model": tag,
            "message": {"role": "assistant", "content": "OK"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 3,
            "prompt_eval_duration": 300_000_000,
            "eval_count": 1,
            "eval_duration": 100_000_000,
        }
        yield OllamaChatChunk.model_validate({**payload, "raw": payload})


class FakeManager:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.client = FakeStreamingClient()
        self.loads: list[str] = []
        self.unloads: list[str] = []

    def resolve(self, identifier: str):
        for spec in [*self.settings.models.candidates, self.settings.models.judge]:
            if identifier in {spec.tag, spec.name, "candidate", "judge"}:
                return SimpleNamespace(spec=spec)
        raise AssertionError(f"unexpected model identifier: {identifier}")

    def load(self, identifier: str) -> None:
        self.loads.append(identifier)

    def unload(self, identifier: str) -> None:
        self.unloads.append(identifier)

    def close(self) -> None:
        pass


def test_infer_command_writes_jsonl_and_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    managers: list[FakeManager] = []

    def create_manager(settings):
        manager = FakeManager(settings)
        managers.append(manager)
        return manager

    monkeypatch.setattr(main_module, "_create_model_manager", create_manager)
    output = tmp_path / "inference.jsonl"
    exit_code = main_module.main(
        [
            "infer",
            "--model",
            "llama3.1:8b",
            "--limit",
            "1",
            "--warmup",
            "0",
            "--output",
            str(output),
            "--json",
        ]
    )
    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "completed"
    assert summary["measured_requests"] == 1
    assert summary["success_count"] == 1
    assert summary["output_path"] == str(output.resolve())
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["model_tag"] == "llama3.1:8b"
    assert records[0]["response_text"] == "OK"
    assert records[0]["status"] == "success"
    assert records[0]["tokens"]["output_tokens_per_second"] == pytest.approx(10.0)
    assert managers[0].loads == ["llama3.1:8b"]
    assert managers[0].unloads == ["llama3.1:8b"]
    assert managers[0].client.requests[0]["options"]["num_predict"] <= 512
