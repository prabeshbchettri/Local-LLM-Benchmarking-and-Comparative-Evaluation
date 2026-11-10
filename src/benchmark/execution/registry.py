"""
File: registry.py

Description:
Model registry and lifecycle management for configured Ollama models.

Responsibilities:
- Resolve candidate and judge tags against the configuration.
- Discover installed models and verify digests and quantization.
- Pull models only after explicit approval.
- Load, unload, and test models with /api/ps confirmation.
- Build metadata snapshots for later experiment records.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from benchmark.core.config import AppSettings, ModelSpec
from benchmark.core.exceptions import (
    InferenceError,
    InvalidModelTagError,
    ModelLifecycleError,
    ModelNotFoundError,
    ModelNotRegisteredError,
    ModelPullError,
    OllamaHTTPError,
    PullApprovalRequiredError,
)
from benchmark.core.logging import get_logger
from benchmark.execution.metadata import (
    ModelInspection,
    ModelMetadata,
    ModelRole,
    ModelSnapshot,
    ModelSource,
    ModelStatus,
    ModelVerification,
    assess_compatibility,
    build_loaded_snapshot,
    build_missing_metadata,
    build_model_metadata,
    build_verification,
    utc_now,
    write_snapshot,
)
from benchmark.execution.ollama_client import OllamaClient, PullResult
from benchmark.models.ollama import (
    OllamaChatResponse,
    OllamaLoadedModel,
    OllamaModelSummary,
    OllamaPullEvent,
    OllamaShowResponse,
)


LOGGER = get_logger("execution.registry")


@dataclass(frozen=True)
class RegisteredModel:
    """A configured model and its benchmark role."""

    spec: ModelSpec
    role: ModelRole


def normalize_tag(tag: str) -> str:
    """Normalize an Ollama tag to include an explicit latest tag when omitted."""
    value = tag.strip()
    if not value:
        raise InvalidModelTagError("model tag must not be empty")
    return value if ":" in value else f"{value}:latest"


class ModelManager:
    """High-level model registry backed by an Ollama client."""

    def __init__(
        self,
        settings: AppSettings,
        client: OllamaClient | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.client = client or OllamaClient(
            str(settings.models.ollama_base_url),
            timeout_seconds=settings.benchmark.request_timeout_seconds,
        )
        self._sleep = sleep
        self._monotonic = monotonic
        self._owns_client = client is None
        self._version_cache: str | None = None
        self._registry = tuple(
            [
                *(
                    RegisteredModel(spec, ModelRole.CANDIDATE)
                    for spec in settings.models.candidates
                ),
                RegisteredModel(settings.models.judge, ModelRole.JUDGE),
            ]
        )

    def close(self) -> None:
        """Close an internally created Ollama client."""
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "ModelManager":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def registry(self) -> tuple[RegisteredModel, ...]:
        """Return the configured candidate and judge roster."""
        return self._registry

    def resolve(self, identifier: str) -> RegisteredModel:
        """Resolve a tag, display name, or role keyword to a configured model."""
        if not identifier.strip():
            raise InvalidModelTagError("model identifier must not be empty")
        normalized = normalize_tag(identifier)
        for registered in self._registry:
            if identifier == registered.spec.tag:
                return registered
            if normalized == normalize_tag(registered.spec.tag):
                return registered
            if identifier.lower() == registered.spec.name.lower():
                return registered
            if identifier.lower() == registered.role.value:
                return registered
        raise ModelNotRegisteredError(f"Model is not in the configured roster: {identifier}")

    def _version(self) -> str:
        if self._version_cache is None:
            self._version_cache = self.client.version().version
        return self._version_cache

    @staticmethod
    def _tags_equal(left: str, right: str) -> bool:
        try:
            return normalize_tag(left) == normalize_tag(right)
        except InvalidModelTagError:
            return False

    def _find_summary(
        self,
        tag: str,
        summaries: list[OllamaModelSummary] | None = None,
    ) -> OllamaModelSummary | None:
        for summary in summaries if summaries is not None else self.client.list_models():
            if self._tags_equal(summary.tag, tag) or self._tags_equal(summary.name, tag):
                return summary
        return None

    def _find_loaded(
        self,
        tag: str,
        models: list[OllamaLoadedModel] | None = None,
    ) -> OllamaLoadedModel | None:
        for model in models if models is not None else self.client.loaded_models():
            if self._tags_equal(model.tag, tag) or self._tags_equal(model.name, tag):
                return model
        return None

    def _show(self, tag: str) -> OllamaShowResponse:
        try:
            return self.client.show_model(tag)
        except OllamaHTTPError as exc:
            if exc.status_code == 404:
                raise ModelNotFoundError(f"Model is not installed: {tag}") from exc
            raise

    def _metadata(
        self,
        registered: RegisteredModel,
        summary: OllamaModelSummary,
        source: ModelSource = ModelSource.INSTALLED,
    ) -> ModelMetadata:
        return build_model_metadata(
            registered.spec,
            summary,
            self._show(registered.spec.tag),
            self._version(),
            registered.role,
            source=source,
        )

    def inventory(self, include_all: bool = False) -> list[ModelStatus]:
        """Return configured models and optionally all other installed models."""
        summaries = self.client.list_models()
        statuses: list[ModelStatus] = []
        seen: set[str] = set()
        for registered in self._registry:
            summary = self._find_summary(registered.spec.tag, summaries)
            if summary is None:
                statuses.append(
                    ModelStatus(
                        tag=registered.spec.tag,
                        name=registered.spec.name,
                        role=registered.role,
                        installed=False,
                    )
                )
                continue
            seen.add(normalize_tag(summary.tag))
            metadata = self._metadata(registered, summary)
            statuses.append(
                ModelStatus(
                    tag=registered.spec.tag,
                    name=registered.spec.name,
                    role=registered.role,
                    installed=True,
                    digest=summary.digest or None,
                    size_bytes=summary.size or None,
                    parameter_size=summary.details.parameter_size or None,
                    quantization=summary.details.quantization_level or None,
                    context_length=metadata.context_length,
                )
            )
        if include_all:
            for summary in summaries:
                if normalize_tag(summary.tag) in seen:
                    continue
                spec = ModelSpec(name=summary.name, tag=normalize_tag(summary.tag))
                metadata = build_model_metadata(
                    spec,
                    summary,
                    self._show(spec.tag),
                    self._version(),
                    ModelRole.OTHER,
                )
                statuses.append(
                    ModelStatus(
                        tag=spec.tag,
                        name=summary.name,
                        role=ModelRole.OTHER,
                        installed=True,
                        digest=summary.digest or None,
                        size_bytes=summary.size or None,
                        parameter_size=summary.details.parameter_size or None,
                        quantization=summary.details.quantization_level or None,
                        context_length=metadata.context_length,
                    )
                )
        return statuses

    def inspect(self, identifier: str, require_installed: bool = True) -> ModelInspection:
        """Inspect metadata, verification, and compatibility for one model."""
        registered = self.resolve(identifier)
        summary = self._find_summary(registered.spec.tag)
        if summary is None:
            if require_installed:
                raise ModelNotFoundError(f"Model is not installed: {registered.spec.tag}")
            metadata = build_missing_metadata(
                registered.spec,
                self._version(),
                registered.role,
            )
        else:
            metadata = self._metadata(registered, summary)
        verification = build_verification(metadata, registered.spec)
        compatibility = assess_compatibility(metadata, self.settings.benchmark.num_ctx)
        return ModelInspection(
            metadata=metadata,
            verification=verification,
            compatibility=compatibility,
        )

    def verify(self, identifier: str, require_installed: bool = False) -> ModelVerification:
        """Verify one configured model."""
        return self.inspect(identifier, require_installed=require_installed).verification

    def verify_all(self, require_installed: bool = False) -> list[ModelVerification]:
        """Verify every configured candidate and the judge."""
        return [
            self.verify(registered.spec.tag, require_installed=require_installed)
            for registered in self._registry
        ]

    def select(self, identifier: str) -> ModelStatus:
        """Return the inventory entry for a configured model."""
        registered = self.resolve(identifier)
        for status in self.inventory(include_all=False):
            if self._tags_equal(status.tag, registered.spec.tag):
                return status
        raise ModelNotRegisteredError(f"Model is not in the configured roster: {identifier}")

    def pull(
        self,
        identifier: str,
        approved: bool = False,
        progress: Callable[[OllamaPullEvent], None] | None = None,
    ) -> ModelMetadata:
        """Pull a configured model after explicit approval."""
        registered = self.resolve(identifier)
        if not approved:
            raise PullApprovalRequiredError(
                f"Pulling {registered.spec.tag} requires explicit approval; pass --yes"
            )
        summary = self._find_summary(registered.spec.tag)
        if summary is not None:
            return self._metadata(registered, summary)
        LOGGER.info("Pulling model %s", registered.spec.tag)
        result: PullResult = self.client.pull_model(registered.spec.tag, progress=progress)
        if not result.success:
            raise ModelPullError(
                f"Pull for {registered.spec.tag} ended with status: {result.status}"
            )
        summary = self._find_summary(registered.spec.tag)
        if summary is None:
            raise ModelPullError(
                f"Ollama reported success but {registered.spec.tag} is not installed"
            )
        return self._metadata(registered, summary, source=ModelSource.PULLED)

    def _wait_for_state(
        self,
        tag: str,
        loaded: bool,
        timeout_seconds: float,
        interval_seconds: float,
    ) -> OllamaLoadedModel | None:
        deadline = self._monotonic() + timeout_seconds
        while True:
            found = self._find_loaded(tag)
            if (found is not None) == loaded:
                return found
            if self._monotonic() >= deadline:
                expected = "loaded" if loaded else "unloaded"
                raise ModelLifecycleError(
                    f"Timed out waiting for {tag} to become {expected}"
                )
            self._sleep(interval_seconds)

    def load(
        self,
        identifier: str,
        keep_alive: str | None = None,
        timeout_seconds: float = 120.0,
        interval_seconds: float = 0.5,
    ) -> OllamaLoadedModel:
        """Load a model and confirm it appears in /api/ps."""
        registered = self.resolve(identifier)
        summary = self._find_summary(registered.spec.tag)
        if summary is None:
            raise ModelNotFoundError(f"Model is not installed: {registered.spec.tag}")
        keep_alive_value = keep_alive or self.settings.models.keep_alive
        LOGGER.info("Loading model %s with keep_alive=%s", registered.spec.tag, keep_alive_value)
        self.client.load_model(registered.spec.tag, keep_alive_value)
        loaded = self._wait_for_state(
            registered.spec.tag,
            loaded=True,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
        )
        if loaded is None:
            raise ModelLifecycleError(f"Model did not load: {registered.spec.tag}")
        return loaded

    def unload(
        self,
        identifier: str,
        timeout_seconds: float = 30.0,
        interval_seconds: float = 0.5,
    ) -> None:
        """Unload a model and confirm it disappears from /api/ps."""
        registered = self.resolve(identifier)
        summary = self._find_summary(registered.spec.tag)
        if summary is None:
            raise ModelNotFoundError(f"Model is not installed: {registered.spec.tag}")
        LOGGER.info("Unloading model %s", registered.spec.tag)
        self.client.unload_model(registered.spec.tag)
        self._wait_for_state(
            registered.spec.tag,
            loaded=False,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
        )

    def test_inference(
        self,
        identifier: str,
        prompt: str = "Reply with the single word OK.",
        max_tokens: int = 16,
        unload_after: bool = False,
    ) -> OllamaChatResponse:
        """Run a short non-streaming chat request against a configured model."""
        registered = self.resolve(identifier)
        summary = self._find_summary(registered.spec.tag)
        if summary is None:
            raise ModelNotFoundError(f"Model is not installed: {registered.spec.tag}")
        response = self.client.chat(
            registered.spec.tag,
            [{"role": "user", "content": prompt}],
            options={
                "temperature": 0.0,
                "num_predict": max_tokens,
                "num_ctx": self.settings.benchmark.num_ctx,
            },
            keep_alive=self.settings.models.keep_alive,
        )
        if not response.message.content.strip():
            raise InferenceError(
                f"Model {registered.spec.tag} returned an empty response"
            )
        if unload_after:
            self.unload(registered.spec.tag)
        return response

    def snapshot(
        self,
        include_all: bool = False,
        output: Path | None = None,
    ) -> tuple[ModelSnapshot, Path]:
        """Build and write a metadata snapshot."""
        version = self._version()
        summaries = self.client.list_models()
        models: list[ModelMetadata] = []
        verifications: list[ModelVerification] = []
        seen: set[str] = set()
        for registered in self._registry:
            summary = self._find_summary(registered.spec.tag, summaries)
            if summary is None:
                metadata = build_missing_metadata(registered.spec, version, registered.role)
            else:
                seen.add(normalize_tag(summary.tag))
                metadata = self._metadata(registered, summary)
            models.append(metadata)
            verifications.append(build_verification(metadata, registered.spec))
        if include_all:
            for summary in summaries:
                if normalize_tag(summary.tag) in seen:
                    continue
                spec = ModelSpec(name=summary.name, tag=normalize_tag(summary.tag))
                metadata = build_model_metadata(
                    spec,
                    summary,
                    self._show(spec.tag),
                    version,
                    ModelRole.OTHER,
                )
                models.append(metadata)
                verifications.append(build_verification(metadata, spec))
        loaded = [
            build_loaded_snapshot(model)
            for model in self.client.loaded_models()
        ]
        snapshot = ModelSnapshot(
            created_at=utc_now(),
            ollama_version=version,
            ollama_base_url=str(self.settings.models.ollama_base_url).rstrip("/"),
            config_hash=self.settings.config_hash,
            config_dir=str(self.settings.config_dir),
            project_root=str(self.settings.project_root),
            models=models,
            verifications=verifications,
            loaded_models=loaded,
        )
        target = output or (self.settings.benchmark.storage_dir / "model_metadata.json")
        path = write_snapshot(snapshot, target)
        LOGGER.info("Wrote model metadata snapshot to %s", path)
        return snapshot, path
