"""
File: ollama.py

Description:
Typed schemas for Ollama HTTP API requests and responses.

Responsibilities:
- Model version, tags, show, process, pull, chat, and generate responses.
- Ignore unknown API fields so newer Ollama versions remain compatible.
- Preserve raw model_info data needed for metadata extraction.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OllamaSchema(BaseModel):
    """Base schema for untrusted Ollama API responses."""

    model_config = ConfigDict(extra="ignore")


class OllamaModelDetailsSummary(OllamaSchema):
    """Summary details returned by /api/tags and /api/ps."""

    parent_model: str = ""
    format: str = ""
    family: str = ""
    families: list[str] = Field(default_factory=list)
    parameter_size: str = ""
    quantization_level: str = ""


class OllamaModelSummary(OllamaSchema):
    """An installed model entry returned by /api/tags."""

    name: str
    model: str | None = None
    modified_at: str | None = None
    size: int = 0
    digest: str = ""
    details: OllamaModelDetailsSummary = Field(default_factory=OllamaModelDetailsSummary)

    @property
    def tag(self) -> str:
        """Return the canonical model tag reported by Ollama."""
        return self.model or self.name


class OllamaTagsResponse(OllamaSchema):
    """Response from /api/tags."""

    models: list[OllamaModelSummary] = Field(default_factory=list)


class OllamaVersionResponse(OllamaSchema):
    """Response from /api/version."""

    version: str


class OllamaShowResponse(OllamaSchema):
    """Response from /api/show."""

    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: OllamaModelDetailsSummary = Field(default_factory=OllamaModelDetailsSummary)
    model_info: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    license: str = ""
    modified_at: str | None = None


class OllamaLoadedModel(OllamaSchema):
    """A loaded model entry returned by /api/ps."""

    name: str
    model: str | None = None
    size: int = 0
    digest: str = ""
    details: OllamaModelDetailsSummary = Field(default_factory=OllamaModelDetailsSummary)
    expires_at: str | None = None
    size_vram: int = 0
    context_length: int | None = None

    @property
    def tag(self) -> str:
        """Return the canonical model tag reported by Ollama."""
        return self.model or self.name


class OllamaPsResponse(OllamaSchema):
    """Response from /api/ps."""

    models: list[OllamaLoadedModel] = Field(default_factory=list)


class OllamaPullEvent(OllamaSchema):
    """One NDJSON event from /api/pull."""

    status: str = ""
    digest: str | None = None
    total: int | None = None
    completed: int | None = None
    error: str | None = None


class OllamaChatMessage(OllamaSchema):
    """A chat message exchanged with /api/chat."""

    role: str
    content: str = ""


class OllamaChatResponse(OllamaSchema):
    """A non-streaming response from /api/chat."""

    model: str = ""
    created_at: str | None = None
    message: OllamaChatMessage
    done: bool = True
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class OllamaGenerateResponse(OllamaSchema):
    """A non-streaming response from /api/generate used for lifecycle calls."""

    model: str = ""
    created_at: str | None = None
    response: str = ""
    done: bool = True
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
