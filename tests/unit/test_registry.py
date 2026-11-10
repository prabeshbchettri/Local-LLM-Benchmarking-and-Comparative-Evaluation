"""
File: test_registry.py

Description:
Unit tests for the model registry and lifecycle manager.

Responsibilities:
- Verify configured model resolution and inventory.
- Verify digest and quantization checks.
- Verify pull approval, load/unload confirmation, inference, and snapshots.
"""

from __future__ import annotations

import pytest

from benchmark.core.config import AppSettings
from benchmark.core.exceptions import (
    InferenceError,
    ModelNotRegisteredError,
    ModelNotFoundError,
    PullApprovalRequiredError,
)
from benchmark.execution.ollama_client import PullResult
from benchmark.execution.registry import ModelManager
from benchmark.models.ollama import (
    OllamaChatResponse,
    OllamaGenerateResponse,
    OllamaLoadedModel,
    OllamaModelDetailsSummary,
    OllamaModelSummary,
    OllamaShowResponse,
)

pytestmark = pytest.mark.unit


def _details(family: str, parameter_size: str, quantization: str = "Q4_K_M"):
    return OllamaModelDetailsSummary(
        family=family,
        parameter_size=parameter_size,
        quantization_level=quantization,
    )


def _summary(tag: str, digest: str, family: str, parameter_size: str) -> OllamaModelSummary:
    return OllamaModelSummary(
        name=tag,
        model=tag,
        size=2_000_000_000,
        digest=digest,
        details=_details(family, parameter_size),
    )


def _show(family: str, parameter_size: str) -> OllamaShowResponse:
    return OllamaShowResponse(
        details=_details(family, parameter_size),
        model_info={
            "general.architecture": family,
            "general.parameter_count": 3_000_000_000,
            f"{family}.context_length": 32768,
            f"{family}.embedding_length": 2048,
            f"{family}.block_count": 36,
            f"{family}.attention.head_count": 16,
            f"{family}.attention.head_count_kv": 2,
        },
        capabilities=["completion", "tools"],
        license="Test license",
    )


def _loaded(tag: str) -> OllamaLoadedModel:
    return OllamaLoadedModel(
        name=tag,
        model=tag,
        size=2_163_568_803,
        size_vram=2_163_568_803,
        context_length=4096,
        details=_details("qwen2", "3.1B"),
    )


class FakeClient:
    """A deterministic in-memory replacement for OllamaClient."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.summaries = [
            _summary(
                settings.models.candidates[0].tag,
                settings.models.candidates[0].expected_digest or "a" * 64,
                "llama",
                "8.0B",
            ),
            _summary(
                settings.models.judge.tag,
                settings.models.judge.expected_digest or "b" * 64,
                "qwen2",
                "3.1B",
            ),
        ]
        self.shows = {
            settings.models.candidates[0].tag: _show("llama", "8.0B"),
            settings.models.judge.tag: _show("qwen2", "3.1B"),
        }
        self.loaded: list[OllamaLoadedModel] = []
        self.pull_summary: OllamaModelSummary | None = None
        self.pull_calls: list[str] = []
        self.chat_content = "OK"
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def version(self, timeout=None) -> object:
        class Version:
            version = "0.40.15"

        return Version()

    def list_models(self) -> list[OllamaModelSummary]:
        return list(self.summaries)

    def show_model(self, tag: str, verbose: bool = False) -> OllamaShowResponse:
        return self.shows[tag]

    def loaded_models(self) -> list[OllamaLoadedModel]:
        return list(self.loaded)

    def load_model(self, tag: str, keep_alive: str) -> OllamaGenerateResponse:
        self.loaded = [_loaded(tag)]
        return OllamaGenerateResponse(model=tag, done=True, done_reason="load")

    def unload_model(self, tag: str) -> OllamaGenerateResponse:
        self.loaded = []
        return OllamaGenerateResponse(model=tag, done=True, done_reason="unload")

    def pull_model(self, tag: str, progress=None) -> PullResult:
        self.pull_calls.append(tag)
        if self.pull_summary is not None:
            self.summaries.append(self.pull_summary)
            self.shows[tag] = _show("qwen2", "7.6B")
        return PullResult(
            tag=tag,
            status="success",
            digest=self.pull_summary.digest if self.pull_summary else None,
            total=1,
            completed=1,
            events=1,
        )

    def chat(self, tag, messages, options=None, keep_alive=None) -> OllamaChatResponse:
        return OllamaChatResponse(
            model=tag,
            message={"role": "assistant", "content": self.chat_content},
            done=True,
            eval_count=2,
            eval_duration=1_000_000_000,
        )


@pytest.fixture()
def fake_client(temp_settings: AppSettings) -> FakeClient:
    return FakeClient(temp_settings)


@pytest.fixture()
def manager(temp_settings: AppSettings, fake_client: FakeClient) -> ModelManager:
    return ModelManager(temp_settings, client=fake_client)  # type: ignore[arg-type]


def test_resolve_and_inventory(manager: ModelManager) -> None:
    assert manager.resolve("llama3.1:8b").role.value == "candidate"
    assert manager.resolve("Qwen 2.5 3B Judge").spec.tag == "qwen2.5:3b"
    assert manager.resolve("judge").role.value == "judge"
    with pytest.raises(ModelNotRegisteredError):
        manager.resolve("unknown:1b")
    statuses = manager.inventory()
    assert len(statuses) == 4
    assert sum(status.installed for status in statuses) == 2
    installed = next(status for status in statuses if status.installed)
    assert installed.context_length == 32768


def test_verify_installed_and_missing(manager: ModelManager) -> None:
    installed = manager.verify("llama3.1:8b")
    assert installed.ok
    missing = manager.verify("qwen2.5:7b-instruct")
    assert not missing.ok
    with pytest.raises(ModelNotFoundError):
        manager.verify("qwen2.5:7b-instruct", require_installed=True)


def test_pull_requires_explicit_approval(manager: ModelManager, fake_client: FakeClient) -> None:
    with pytest.raises(PullApprovalRequiredError, match="explicit approval"):
        manager.pull("qwen2.5:7b-instruct")
    assert fake_client.pull_calls == []


def test_pull_success_records_metadata(
    manager: ModelManager,
    fake_client: FakeClient,
    temp_settings: AppSettings,
) -> None:
    tag = "qwen2.5:7b-instruct"
    fake_client.pull_summary = _summary(tag, "d" * 64, "qwen2", "7.6B")
    metadata = manager.pull(tag, approved=True)
    assert fake_client.pull_calls == [tag]
    assert metadata.tag == tag
    assert metadata.source.value == "pulled"
    assert metadata.installed


def test_load_and_unload_confirm_process_state(manager: ModelManager) -> None:
    loaded = manager.load("qwen2.5:3b", timeout_seconds=1.0)
    assert loaded.tag == "qwen2.5:3b"
    manager.unload("qwen2.5:3b", timeout_seconds=1.0)
    assert manager.client.loaded_models() == []


def test_test_inference(manager: ModelManager, fake_client: FakeClient) -> None:
    response = manager.test_inference("qwen2.5:3b", max_tokens=8)
    assert response.message.content == "OK"
    fake_client.chat_content = ""
    with pytest.raises(InferenceError):
        manager.test_inference("qwen2.5:3b")


def test_snapshot_includes_missing_and_installed_models(
    manager: ModelManager,
    tmp_path,
) -> None:
    snapshot, path = manager.snapshot(output=tmp_path / "snapshot.json")
    assert path.is_file()
    assert len(snapshot.models) == 4
    assert len(snapshot.verifications) == 4
    assert any(model.source.value == "missing" for model in snapshot.models)
    assert any(verification.ok for verification in snapshot.verifications)
    assert snapshot.ollama_version == "0.40.15"
