"""
File: streaming.py

Description:
NDJSON parsing for streaming Ollama chat responses.

Responsibilities:
- Ignore blank keep-alive lines.
- Reject malformed or non-object stream chunks with typed errors.
- Surface Ollama stream error events without hiding partial failures.
- Preserve each raw chunk for response-record metadata.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

from pydantic import ValidationError

from benchmark.core.exceptions import OllamaResponseError
from benchmark.models.ollama import OllamaChatChunk


def iter_chat_chunks(lines: Iterable[str], tag: str) -> Iterator[OllamaChatChunk]:
    """Yield validated chat chunks from an NDJSON line stream."""
    for line in lines:
        if not line.strip():
            continue
        try:
            data: Any = json.loads(line)
        except (ValueError, TypeError) as exc:
            raise OllamaResponseError(
                f"Ollama returned an invalid chat stream chunk for {tag}"
            ) from exc
        if not isinstance(data, dict):
            raise OllamaResponseError(
                f"Ollama returned a non-object chat stream chunk for {tag}"
            )
        error = data.get("error")
        if isinstance(error, str) and error:
            raise OllamaResponseError(f"Ollama stream error for {tag}: {error}")
        try:
            yield OllamaChatChunk.model_validate({**data, "raw": data})
        except ValidationError as exc:
            raise OllamaResponseError(
                f"Ollama returned invalid chat stream data for {tag}"
            ) from exc
