"""
File: ollama_client.py

Description:
Direct HTTP client for the local Ollama API.

Responsibilities:
- Wrap httpx with a base URL, timeouts, and typed errors.
- Implement version, tags, show, process, pull, chat, and generate endpoints.
- Parse NDJSON pull streams without hiding partial failures.
- Support injected transports for network-free tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from benchmark.core.exceptions import (
    ModelPullError,
    OllamaConnectionError,
    OllamaError,
    OllamaHTTPError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from benchmark.models.ollama import (
    OllamaChatResponse,
    OllamaGenerateResponse,
    OllamaLoadedModel,
    OllamaModelSummary,
    OllamaPsResponse,
    OllamaPullEvent,
    OllamaShowResponse,
    OllamaTagsResponse,
    OllamaVersionResponse,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True)
class PullResult:
    """Outcome of a completed model pull stream."""

    tag: str
    status: str
    digest: str | None
    total: int | None
    completed: int | None
    events: int

    @property
    def success(self) -> bool:
        """Return whether Ollama reported a successful pull."""
        return self.status == "success"


class OllamaClient:
    """A small synchronous client for the local Ollama API."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(
                method,
                path,
                json=payload,
                timeout=timeout if timeout is not None else self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(f"Ollama request timed out: {path}") from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.base_url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama request failed for {path}: {exc}") from exc
        if response.status_code >= 400:
            raise OllamaHTTPError(response.status_code, self._error_detail(response))
        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaResponseError(f"Ollama returned invalid JSON for {path}") from exc
        if not isinstance(data, dict):
            raise OllamaResponseError(f"Ollama returned a non-object response for {path}")
        return data

    @staticmethod
    def _validated(
        schema: type[SchemaT],
        payload: dict[str, Any],
        label: str,
    ) -> SchemaT:
        try:
            return schema.model_validate(payload)
        except ValidationError as exc:
            raise OllamaResponseError(
                f"Ollama returned invalid {label} data"
            ) from exc

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            text = response.text.strip()
            return text or response.reason_phrase
        if isinstance(data, dict) and isinstance(data.get("error"), str):
            return data["error"]
        return response.text.strip() or response.reason_phrase

    def version(self, timeout: float | None = None) -> OllamaVersionResponse:
        """Return the Ollama runtime version."""
        return self._validated(
            OllamaVersionResponse,
            self._request_json("GET", "/api/version", timeout=timeout),
            "version",
        )

    def list_models(self) -> list[OllamaModelSummary]:
        """Return models installed in the local Ollama store."""
        payload = self._request_json("GET", "/api/tags")
        return self._validated(OllamaTagsResponse, payload, "model list").models

    def find_model(self, tag: str) -> OllamaModelSummary | None:
        """Return an installed model by exact tag, or None."""
        for model in self.list_models():
            if model.tag == tag or model.name == tag:
                return model
        return None

    def show_model(self, tag: str, verbose: bool = False) -> OllamaShowResponse:
        """Return detailed metadata for an installed model."""
        payload: dict[str, Any] = {"model": tag}
        if verbose:
            payload["verbose"] = True
        return self._validated(
            OllamaShowResponse,
            self._request_json("POST", "/api/show", payload),
            "model metadata",
        )

    def loaded_models(self) -> list[OllamaLoadedModel]:
        """Return models currently loaded by Ollama."""
        payload = self._request_json("GET", "/api/ps")
        return self._validated(OllamaPsResponse, payload, "process list").models

    def load_model(self, tag: str, keep_alive: str) -> OllamaGenerateResponse:
        """Load a model and set its keep-alive value."""
        payload = {
            "model": tag,
            "prompt": "",
            "stream": False,
            "keep_alive": keep_alive,
        }
        return self._validated(
            OllamaGenerateResponse,
            self._request_json("POST", "/api/generate", payload),
            "generate",
        )

    def unload_model(self, tag: str) -> OllamaGenerateResponse:
        """Unload a model by setting keep_alive to zero."""
        payload = {
            "model": tag,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }
        return self._validated(
            OllamaGenerateResponse,
            self._request_json("POST", "/api/generate", payload),
            "generate",
        )

    def chat(
        self,
        tag: str,
        messages: Sequence[dict[str, str]],
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> OllamaChatResponse:
        """Send a non-streaming chat request."""
        payload: dict[str, Any] = {
            "model": tag,
            "messages": list(messages),
            "stream": False,
        }
        if options is not None:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        return self._validated(
            OllamaChatResponse,
            self._request_json("POST", "/api/chat", payload),
            "chat",
        )

    def pull_model(
        self,
        tag: str,
        progress: Callable[[OllamaPullEvent], None] | None = None,
    ) -> PullResult:
        """Pull a model, forwarding NDJSON progress events."""
        payload = {"model": tag, "stream": True}
        pull_timeout = httpx.Timeout(30.0, read=None)
        try:
            with self._client.stream(
                "POST",
                "/api/pull",
                json=payload,
                timeout=pull_timeout,
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise OllamaHTTPError(response.status_code, self._error_detail(response))
                events = 0
                last = OllamaPullEvent()
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = OllamaPullEvent.model_validate(json.loads(line))
                    except (ValueError, TypeError) as exc:
                        raise OllamaResponseError(
                            f"Ollama returned an invalid pull event for {tag}"
                        ) from exc
                    events += 1
                    last = event
                    if event.error:
                        raise ModelPullError(f"Pull failed for {tag}: {event.error}")
                    if progress is not None:
                        progress(event)
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(f"Ollama pull timed out for {tag}") from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.base_url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama pull request failed for {tag}: {exc}") from exc
        if events == 0:
            raise ModelPullError(f"Pull returned no progress events for {tag}")
        if not last.status:
            raise ModelPullError(f"Pull ended without a status for {tag}")
        return PullResult(
            tag=tag,
            status=last.status,
            digest=last.digest,
            total=last.total,
            completed=last.completed,
            events=events,
        )
