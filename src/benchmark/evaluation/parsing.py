"""
File: parsing.py

Description:
Shared text, code-block, and JSON extraction helpers for evaluators and the judge.

Responsibilities:
- Normalize exact-match text without changing raw response storage.
- Extract the first fenced code block or JSON value.
- Tokenize and normalize text for lexical checks.
- Count sentences, lines, and bullet lines deterministically.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python|py)?[ \t]*\r?\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
GENERIC_FENCE_PATTERN = re.compile(
    r"```(?:python|py)?[ \t]*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
JSON_FENCE_PATTERN = re.compile(
    r"```(?:json)?[ \t]*\r?\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
WORD_PATTERN = re.compile(r"[a-z0-9]+")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
BULLET_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
WHITESPACE = re.compile(r"\s+")


class JSONExtractionError(ValueError):
    """Raised when no valid JSON value can be extracted from text."""


def collapse_whitespace(text: str) -> str:
    """Collapse all whitespace runs to one space and trim the result."""
    return WHITESPACE.sub(" ", text).strip()


def _strip_edge_punctuation(text: str) -> str:
    start = 0
    end = len(text)
    while start < end and unicodedata.category(text[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(text[end - 1]).startswith("P"):
        end -= 1
    return text[start:end]


def normalize_exact_text(text: str) -> str:
    """Normalize text for documented exact-match comparison."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = collapse_whitespace(normalized)
    return _strip_edge_punctuation(normalized).strip()


def normalize_phrase(text: str) -> str:
    """Normalize text for case-insensitive phrase checks."""
    return collapse_whitespace(unicodedata.normalize("NFKC", text).casefold())


def tokenize_words(text: str) -> list[str]:
    """Return lowercase alphanumeric word tokens."""
    return WORD_PATTERN.findall(unicodedata.normalize("NFKC", text).casefold())


def extract_code_block(text: str) -> str:
    """Return the first fenced Python block, or the full response if absent."""
    match = CODE_BLOCK_PATTERN.search(text)
    if match is not None:
        return match.group(1).strip()
    match = GENERIC_FENCE_PATTERN.search(text)
    if match is not None:
        return match.group(1).strip()
    return text.strip()


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = JSON_FENCE_PATTERN.fullmatch(stripped)
    if match is not None:
        return match.group(1).strip()
    return stripped


def extract_json_value(text: str) -> Any:
    """Extract and parse the first valid JSON value without repairing it."""
    stripped = _strip_json_fence(text)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise JSONExtractionError("no valid JSON object or array found")


def count_sentences(text: str) -> int:
    """Count non-empty sentence-like segments separated by terminal punctuation."""
    stripped = text.strip()
    if not stripped:
        return 0
    segments = SENTENCE_BOUNDARY.split(stripped)
    return sum(1 for segment in segments if any(char.isalnum() for char in segment))


def non_empty_lines(text: str) -> list[str]:
    """Return stripped non-empty lines in order."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def count_bullet_lines(text: str) -> int:
    """Count lines that begin with a bullet or numbered-list marker."""
    return sum(1 for line in text.splitlines() if BULLET_LINE.match(line))


def count_occurrences(text: str, phrase: str) -> int:
    """Count non-overlapping case-insensitive phrase occurrences."""
    normalized_text = normalize_phrase(text)
    normalized_phrase = normalize_phrase(phrase)
    if not normalized_phrase:
        return 0
    return normalized_text.count(normalized_phrase)


def truncate_output(text: str, limit: int = 10_000) -> str:
    """Truncate captured output to a fixed character limit."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[output truncated]"
