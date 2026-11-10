"""
File: test_monitoring_system.py

Description:
Unit tests for system-wide CPU, RAM, and Ollama process sampling.

Responsibilities:
- Verify CPU and memory readings.
- Verify Ollama process matching and RSS aggregation.
- Verify missing psutil and inaccessible process memory produce nulls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import benchmark.monitoring.system as system_module
from benchmark.monitoring.system import SystemSampler

pytestmark = pytest.mark.unit


class FakeProcess:
    def __init__(self, info: dict, rss: int | None = None, error: bool = False) -> None:
        self.info = info
        self._rss = rss
        self._error = error

    def memory_info(self):
        if self._error:
            raise OSError("access denied")
        return SimpleNamespace(rss=self._rss)


class FakePsutil:
    def __init__(
        self,
        *,
        cpu: float = 25.0,
        processes: list[FakeProcess] | None = None,
    ) -> None:
        self.cpu = cpu
        self.processes = processes or []
        self.cpu_calls = 0

    def cpu_percent(self, interval=None):
        self.cpu_calls += 1
        return self.cpu

    def virtual_memory(self):
        return SimpleNamespace(
            total=16 * 1024**3,
            available=8 * 1024**3,
            used=8 * 1024**3,
            percent=50.0,
        )

    def process_iter(self, attrs=None):
        return self.processes


def test_system_sampler_reads_cpu_ram_and_ollama_rss() -> None:
    psutil = FakePsutil(
        cpu=37.5,
        processes=[
            FakeProcess(
                {
                    "pid": 1,
                    "name": "ollama.exe",
                    "exe": r"C:\Ollama\ollama.exe",
                    "cmdline": ["ollama.exe", "serve"],
                },
                rss=100,
            ),
            FakeProcess(
                {
                    "pid": 2,
                    "name": "ollama app.exe",
                    "exe": r"C:\Ollama\ollama app.exe",
                    "cmdline": [],
                },
                rss=50,
            ),
            FakeProcess(
                {
                    "pid": 3,
                    "name": "notepad.exe",
                    "exe": r"C:\Windows\notepad.exe",
                    "cmdline": ["notepad.exe"],
                },
                rss=999,
            ),
        ],
    )
    sampler = SystemSampler(psutil_module=psutil)
    sampler.prime()
    reading = sampler.sample()
    assert psutil.cpu_calls == 2
    assert reading.cpu_percent == pytest.approx(37.5)
    assert reading.ram_total_bytes == 16 * 1024**3
    assert reading.ram_available_bytes == 8 * 1024**3
    assert reading.ram_used_bytes == 8 * 1024**3
    assert reading.ram_percent == pytest.approx(50.0)
    assert reading.ollama_process_count == 2
    assert reading.ollama_process_names == ["ollama app.exe", "ollama.exe"]
    assert reading.ollama_rss_bytes == 150


def test_process_without_memory_access_still_counts() -> None:
    psutil = FakePsutil(
        processes=[
            FakeProcess(
                {
                    "pid": 1,
                    "name": "ollama.exe",
                    "exe": None,
                    "cmdline": [],
                },
                error=True,
            )
        ]
    )
    reading = SystemSampler(psutil_module=psutil).sample()
    assert reading.ollama_process_count == 1
    assert reading.ollama_rss_bytes is None


def test_missing_psutil_returns_nulls(monkeypatch) -> None:
    monkeypatch.setattr(system_module, "_PSUTIL", None)
    sampler = SystemSampler()
    sampler.prime()
    reading = sampler.sample()
    assert reading.cpu_percent is None
    assert reading.ram_total_bytes is None
    assert reading.ollama_processes == ()
