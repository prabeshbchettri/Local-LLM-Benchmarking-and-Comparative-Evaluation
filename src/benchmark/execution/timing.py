"""
File: timing.py

Description:
Monotonic timing and throughput calculations for streaming inference.

Responsibilities:
- Convert nanosecond durations to seconds.
- Calculate TTFT and total latency from monotonic clock samples.
- Calculate token rates while representing undefined rates as None.
"""

from __future__ import annotations

NANOSECONDS_PER_SECOND = 1_000_000_000


def nanoseconds_to_seconds(value: int | None) -> float | None:
    """Convert a nanosecond duration to seconds, preserving missing values."""
    if value is None:
        return None
    return value / NANOSECONDS_PER_SECOND


def elapsed_seconds(start_ns: int, end_ns: int) -> float:
    """Return the elapsed seconds between two monotonic clock samples."""
    if end_ns < start_ns:
        raise ValueError("end timestamp must not precede start timestamp")
    return (end_ns - start_ns) / NANOSECONDS_PER_SECOND


def ttft_seconds(start_ns: int, first_token_ns: int | None) -> float | None:
    """Return time to first token, or None when no token was received."""
    if first_token_ns is None:
        return None
    return elapsed_seconds(start_ns, first_token_ns)


def tokens_per_second(count: int | None, duration_ns: int | None) -> float | None:
    """Return tokens per second, or None when count or duration is undefined."""
    if count is None or duration_ns is None or duration_ns <= 0:
        return None
    return count / nanoseconds_to_seconds(duration_ns)
