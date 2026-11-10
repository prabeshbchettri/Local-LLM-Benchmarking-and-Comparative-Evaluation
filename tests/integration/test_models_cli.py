"""
File: test_models_cli.py

Description:
Integration tests for the models command surface using an in-memory manager.

Responsibilities:
- Verify JSON output for inventory, verification, show, select, and ps.
- Verify pull refusal without explicit approval.
- Verify snapshot path reporting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark import __main__ as cli
from benchmark.core.config import AppSettings
from benchmark.core.exceptions import PullApprovalRequiredError
from benchmark.execution.metadata import (
    CompatibilityAssessment,
    ModelInspection,
    ModelMetadata,
    ModelRole,
    ModelSnapshot,
    ModelSource,
    ModelStatus,
    ModelVerification,
    utc_now,
)
from benchmark.models.ollama import (
    OllamaChatResponse,
    OllamaLoadedModel,
    OllamaPullEvent,
)

pytestmark = pytest.mark.integration

DIGEST = "a" * 64


class FakeProcessClient:
    def loaded_models(self) -> list[OllamaLoadedModel]:
        return [
            OllamaLoadedModel(
                name="qwen2.5:3b",
                model="qwen2.5:3b",
                size=2_163_568_803,
                size_vram=2_163_568_803,
                context_length=4096,
            )
        ]


class FakeManager:
    def __init__(self, snapshot_path: Path) -> None:
        self.client = FakeProcessClient()
        self.snapshot_path = snapshot_path
        self.metadata = ModelMetadata(
            tag="llama3.1:8b",
            name="Llama 3.1 8B Instruct",
            role=ModelRole.CANDIDATE,
            source=ModelSource.INSTALLED,
            installed=True,
            digest=DIGEST,
            size_bytes=4_920_753_328,
            family="llama",
            parameter_count=8_030_261_312,
            parameter_size="8.0B",
            quantization="Q4_K_M",
            context_length=131_072,
            embedding_length=4096,
            capabilities=["completion", "tools"],
            license="Llama license",
            ollama_version="0.40.15",
        )
        self.verification = ModelVerification(
            tag="llama3.1:8b",
            name="Llama 3.1 8B Instruct",
            role=ModelRole.CANDIDATE,
            installed=True,
            digest=DIGEST,
            expected_digest=DIGEST,
            digest_match=True,
            quantization="Q4_K_M",
            expected_quantization="Q4_K_M",
            quantization_match=True,
        )
        self.inspection = ModelInspection(
            metadata=self.metadata,
            verification=self.verification,
            compatibility=CompatibilityAssessment(
                fit="partial_cpu_offload",
                target_vram_bytes=4 * 1024**3,
                target_ram_bytes=16 * 1024**3,
                estimated_kv_cache_bytes=1_000_000_000,
                estimated_total_bytes=5_920_753_328,
            ),
        )

    def inventory(self, include_all: bool = False) -> list[ModelStatus]:
        return [
            ModelStatus(
                tag=self.metadata.tag,
                name=self.metadata.name,
                role=self.metadata.role,
                installed=True,
                digest=self.metadata.digest,
                size_bytes=self.metadata.size_bytes,
                parameter_size=self.metadata.parameter_size,
                quantization=self.metadata.quantization,
            )
        ]

    def verify_all(self, require_installed: bool = False) -> list[ModelVerification]:
        return [self.verification]

    def verify(self, identifier: str, require_installed: bool = False) -> ModelVerification:
        return self.verification

    def inspect(self, identifier: str, require_installed: bool = True) -> ModelInspection:
        return self.inspection

    def select(self, identifier: str) -> ModelStatus:
        return self.inventory()[0]

    def pull(self, identifier: str, approved: bool = False, progress=None) -> ModelMetadata:
        if not approved:
            raise PullApprovalRequiredError("explicit approval required")
        if progress is not None:
            progress(OllamaPullEvent(status="pulling", total=100, completed=50))
        return self.metadata

    def test_inference(self, identifier: str, **kwargs) -> OllamaChatResponse:
        return OllamaChatResponse(
            model="llama3.1:8b",
            message={"role": "assistant", "content": "OK"},
            done=True,
            eval_count=2,
        )

    def load(self, identifier: str, keep_alive=None) -> OllamaLoadedModel:
        return self.client.loaded_models()[0]

    def unload(self, identifier: str) -> None:
        return None

    def snapshot(self, include_all: bool = False, output=None) -> tuple[ModelSnapshot, Path]:
        snapshot = ModelSnapshot(
            created_at=utc_now(),
            ollama_version="0.40.15",
            ollama_base_url="http://127.0.0.1:11434",
            config_hash="b" * 64,
            config_dir="configs",
            project_root=".",
            models=[self.metadata],
            verifications=[self.verification],
        )
        return snapshot, self.snapshot_path

    def close(self) -> None:
        return None


@pytest.fixture()
def fake_manager(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> FakeManager:
    manager = FakeManager(tmp_path / "snapshot.json")

    def factory(settings: AppSettings) -> FakeManager:
        return manager

    monkeypatch.setattr(cli, "_create_model_manager", factory)
    return manager


def test_models_list_json(
    fake_manager: FakeManager,
    temp_config_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = cli.main(["models", "list", "--json", "--config-dir", str(temp_config_dir)])
    captured = capsys.readouterr()
    assert result == 0
    payload = json.loads(captured.out)
    assert payload[0]["tag"] == "llama3.1:8b"
    assert payload[0]["installed"] is True


def test_models_verify_and_show_json(
    fake_manager: FakeManager,
    temp_config_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = cli.main(
        ["models", "verify", "--json", "--config-dir", str(temp_config_dir)]
    )
    captured = capsys.readouterr()
    assert result == 0
    payload = json.loads(captured.out)
    assert payload[0]["ok"] is True

    result = cli.main(
        [
            "models",
            "show",
            "llama3.1:8b",
            "--json",
            "--config-dir",
            str(temp_config_dir),
        ]
    )
    captured = capsys.readouterr()
    assert result == 0
    payload = json.loads(captured.out)
    assert payload["metadata"]["tag"] == "llama3.1:8b"
    assert payload["verification_ok"] is True


def test_models_pull_refuses_without_yes(
    fake_manager: FakeManager,
    temp_config_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = cli.main(
        [
            "models",
            "pull",
            "llama3.1:8b",
            "--config-dir",
            str(temp_config_dir),
        ]
    )
    captured = capsys.readouterr()
    assert result == 1
    assert "explicit approval" in captured.err


def test_models_pull_with_yes_json(
    fake_manager: FakeManager,
    temp_config_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = cli.main(
        [
            "models",
            "pull",
            "llama3.1:8b",
            "--yes",
            "--json",
            "--config-dir",
            str(temp_config_dir),
        ]
    )
    captured = capsys.readouterr()
    assert result == 0
    assert json.loads(captured.out)["tag"] == "llama3.1:8b"


def test_models_select_and_ps_json(
    fake_manager: FakeManager,
    temp_config_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = cli.main(
        ["models", "select", "judge", "--json", "--config-dir", str(temp_config_dir)]
    )
    captured = capsys.readouterr()
    assert result == 0
    assert json.loads(captured.out)["tag"] == "llama3.1:8b"

    result = cli.main(["models", "ps", "--json", "--config-dir", str(temp_config_dir)])
    captured = capsys.readouterr()
    assert result == 0
    payload = json.loads(captured.out)
    assert payload[0]["tag"] == "qwen2.5:3b"


def test_models_snapshot_reports_path(
    fake_manager: FakeManager,
    temp_config_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = cli.main(
        ["models", "snapshot", "--json", "--config-dir", str(temp_config_dir)]
    )
    captured = capsys.readouterr()
    assert result == 0
    payload = json.loads(captured.out)
    assert payload["path"].endswith("snapshot.json")
    assert payload["snapshot"]["models"][0]["tag"] == "llama3.1:8b"


def test_models_test_load_and_unload(
    fake_manager: FakeManager,
    temp_config_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = cli.main(
        [
            "models",
            "test",
            "llama3.1:8b",
            "--json",
            "--config-dir",
            str(temp_config_dir),
        ]
    )
    captured = capsys.readouterr()
    assert result == 0
    assert json.loads(captured.out)["message"]["content"] == "OK"

    result = cli.main(
        ["models", "load", "llama3.1:8b", "--json", "--config-dir", str(temp_config_dir)]
    )
    captured = capsys.readouterr()
    assert result == 0
    assert json.loads(captured.out)["name"] == "qwen2.5:3b"

    result = cli.main(
        [
            "models",
            "unload",
            "llama3.1:8b",
            "--json",
            "--config-dir",
            str(temp_config_dir),
        ]
    )
    captured = capsys.readouterr()
    assert result == 0
    assert json.loads(captured.out)["unloaded"] == "llama3.1:8b"
