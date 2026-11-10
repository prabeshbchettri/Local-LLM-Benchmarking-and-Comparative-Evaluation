"""
File: system.py

Description:
System-wide CPU and RAM sampling plus Ollama process discovery.

Responsibilities:
- Prime and read system CPU utilization with psutil.
- Read total, available, used, and percentage RAM.
- Discover matching Ollama processes and sum their resident memory.
- Return explicit nulls when a metric or process is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Iterable

try:
    import psutil as _PSUTIL
except ImportError:
    _PSUTIL = None


@dataclass(frozen=True)
class OllamaProcess:
    """One process that appears to belong to Ollama."""

    pid: int
    name: str
    rss_bytes: int | None


@dataclass(frozen=True)
class SystemReading:
    """One system-wide and Ollama-process-specific measurement."""

    cpu_percent: float | None
    ram_total_bytes: int | None
    ram_available_bytes: int | None
    ram_used_bytes: int | None
    ram_percent: float | None
    ollama_processes: tuple[OllamaProcess, ...]

    @property
    def ollama_process_count(self) -> int:
        return len(self.ollama_processes)

    @property
    def ollama_process_names(self) -> list[str]:
        return sorted({process.name for process in self.ollama_processes if process.name})

    @property
    def ollama_rss_bytes(self) -> int | None:
        values = [
            process.rss_bytes
            for process in self.ollama_processes
            if process.rss_bytes is not None
        ]
        return sum(values) if values else None


ProcessMatcher = Callable[[str, str | None, tuple[str, ...]], bool]


def _default_matcher(name: str, exe: str | None, cmdline: tuple[str, ...]) -> bool:
    haystack = " ".join([name, exe or "", *cmdline]).lower()
    return "ollama" in haystack


def _safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


class SystemSampler:
    """Collect CPU, RAM, and Ollama process metrics through psutil."""

    def __init__(
        self,
        psutil_module: ModuleType | None = None,
        *,
        process_matcher: ProcessMatcher | None = None,
    ) -> None:
        self._psutil = psutil_module if psutil_module is not None else _PSUTIL
        self._matcher = process_matcher or _default_matcher

    @property
    def available(self) -> bool:
        return self._psutil is not None

    def prime(self) -> None:
        """Initialize psutil's CPU counter before the first measurement."""
        if self._psutil is None:
            return
        try:
            self._psutil.cpu_percent(interval=None)
        except (AttributeError, OSError, ValueError):
            return

    def sample(self) -> SystemReading:
        """Return one system and Ollama process measurement."""
        if self._psutil is None:
            return SystemReading(None, None, None, None, None, ())
        cpu_percent = self._sample_cpu()
        memory = self._sample_memory()
        processes = self._sample_ollama_processes()
        return SystemReading(
            cpu_percent=cpu_percent,
            ram_total_bytes=_safe_int(getattr(memory, "total", None)),
            ram_available_bytes=_safe_int(getattr(memory, "available", None)),
            ram_used_bytes=_safe_int(getattr(memory, "used", None)),
            ram_percent=_safe_float(getattr(memory, "percent", None)),
            ollama_processes=processes,
        )

    def _sample_cpu(self) -> float | None:
        try:
            value = self._psutil.cpu_percent(interval=None)
        except (AttributeError, OSError, ValueError):
            return None
        return _safe_float(value)

    def _sample_memory(self) -> Any | None:
        try:
            return self._psutil.virtual_memory()
        except (AttributeError, OSError, RuntimeError):
            return None

    def _sample_ollama_processes(self) -> tuple[OllamaProcess, ...]:
        try:
            iterator: Iterable[Any] = self._psutil.process_iter(
                attrs=["pid", "name", "exe", "cmdline"]
            )
        except (AttributeError, OSError):
            return ()
        found: list[OllamaProcess] = []
        for process in iterator:
            try:
                info = process.info
                name = str(info.get("name") or "")
                exe = info.get("exe")
                raw_cmdline = info.get("cmdline") or ()
                cmdline = tuple(str(part) for part in raw_cmdline)
                if not self._matcher(name, exe, cmdline):
                    continue
                rss = None
                try:
                    rss = _safe_int(process.memory_info().rss)
                except (AttributeError, OSError, RuntimeError):
                    rss = None
                found.append(
                    OllamaProcess(
                        pid=int(info.get("pid") or 0),
                        name=name or "ollama",
                        rss_bytes=rss,
                    )
                )
            except (AttributeError, KeyError, OSError, RuntimeError, TypeError):
                continue
        return tuple(found)
