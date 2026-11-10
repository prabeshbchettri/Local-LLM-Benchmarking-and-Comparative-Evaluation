"""
File: __init__.py

Description:
Shared Pydantic schema package for configuration, metadata, and Ollama API data.

Responsibilities:
- Export the project base schema and Ollama response schemas.
"""

from benchmark.models.base import BaseSchema
from benchmark.models.hardware import (
    GpuSource,
    HardwareSample,
    HardwareSummary,
    SamplePhase,
    WindowSummary,
)
from benchmark.models.ollama import (
    OllamaChatMessage,
    OllamaChatResponse,
    OllamaGenerateResponse,
    OllamaLoadedModel,
    OllamaModelDetailsSummary,
    OllamaModelSummary,
    OllamaPsResponse,
    OllamaPullEvent,
    OllamaSchema,
    OllamaShowResponse,
    OllamaTagsResponse,
    OllamaVersionResponse,
)

__all__ = [
    "BaseSchema",
    "GpuSource",
    "HardwareSample",
    "HardwareSummary",
    "OllamaChatMessage",
    "OllamaChatResponse",
    "OllamaGenerateResponse",
    "OllamaLoadedModel",
    "OllamaModelDetailsSummary",
    "OllamaModelSummary",
    "OllamaPsResponse",
    "OllamaPullEvent",
    "OllamaSchema",
    "OllamaShowResponse",
    "OllamaTagsResponse",
    "OllamaVersionResponse",
    "SamplePhase",
    "WindowSummary",
]
