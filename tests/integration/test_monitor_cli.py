"""
File: test_monitor_cli.py

Description:
Integration test for the benchmark monitor command with a fake sampler.

Responsibilities:
- Verify the monitor command starts and stops the sampler.
- Verify JSON output contains the hardware summary.
"""

from __future__ import annotations

import json

import pytest

import benchmark.__main__ as main_module
from benchmark.models.hardware import HardwareSummary

pytestmark = pytest.mark.integration


class FakeSampler:
    def __init__(self, *, interval_seconds: float, baseline_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.baseline_seconds = baseline_seconds
        self.started = False
        self.marked = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def mark_request_start(self) -> None:
        self.marked = True

    def stop(self) -> HardwareSummary:
        self.stopped = True
        return HardwareSummary(
            sampling_interval_ms=round(self.interval_seconds * 1000),
            baseline_seconds=self.baseline_seconds,
            sample_count=1,
        )


def test_monitor_command_outputs_json(monkeypatch, capsys) -> None:
    samplers: list[FakeSampler] = []

    def create_sampler(*, interval_seconds: float, baseline_seconds: float):
        sampler = FakeSampler(
            interval_seconds=interval_seconds,
            baseline_seconds=baseline_seconds,
        )
        samplers.append(sampler)
        return sampler

    monkeypatch.setattr(main_module, "HardwareSampler", create_sampler)
    exit_code = main_module.main(
        ["monitor", "--seconds", "0.01", "--interval-ms", "500", "--json"]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sampling_interval_ms"] == 500
    assert payload["sample_count"] == 1
    assert samplers[0].started is True
    assert samplers[0].marked is True
    assert samplers[0].stopped is True
