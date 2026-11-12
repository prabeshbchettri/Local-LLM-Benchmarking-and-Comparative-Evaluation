"""
File: test_analytics_resources.py

Description:
Unit tests for CPU, RAM, GPU, system VRAM, and model VRAM aggregation.

Responsibilities:
- Verify byte-to-MiB conversion and averaging over measured rows only.
- Verify quality-per-resource values and safe missing-measurement handling.
"""

from __future__ import annotations

import pytest

from benchmark.analytics.resources import BYTES_PER_MIB, resource_usage

pytestmark = pytest.mark.unit


def test_resource_usage_averages_measured_hardware(make_row) -> None:
    rows = [
        make_row(
            hardware=True,
            cpu=20.0,
            ram_bytes=2 * 1024 * BYTES_PER_MIB,
            gpu=30.0,
            vram_bytes=1 * 1024 * BYTES_PER_MIB,
            model_vram_bytes=512 * BYTES_PER_MIB,
        ),
        make_row(
            question_id="q-2",
            hardware=True,
            cpu=40.0,
            ram_bytes=4 * 1024 * BYTES_PER_MIB,
            gpu=50.0,
            vram_bytes=3 * 1024 * BYTES_PER_MIB,
            model_vram_bytes=1536 * BYTES_PER_MIB,
        ),
    ]

    resources = resource_usage(rows, quality_score=80.0)

    assert resources.hardware_samples == 2
    assert resources.cpu_avg_percent == pytest.approx(30.0)
    assert resources.ram_avg_mib == pytest.approx(3072.0)
    assert resources.gpu_utilization_avg_percent == pytest.approx(40.0)
    assert resources.vram_avg_mib == pytest.approx(2048.0)
    assert resources.model_vram_avg_mib == pytest.approx(1024.0)
    assert resources.quality_per_cpu_point == pytest.approx(80.0 / 30.0)
    assert resources.quality_per_ram_gib == pytest.approx(80.0 / 3.0)
    assert resources.quality_per_vram_gib == pytest.approx(40.0)
    assert resources.quality_per_model_vram_gib == pytest.approx(80.0)
    assert resources.model_vram_available is True


def test_resource_usage_preserves_unavailable_measurements(make_row) -> None:
    resources = resource_usage(
        [make_row(hardware=True, cpu=None, ram_bytes=None)],
        quality_score=80.0,
    )

    assert resources.hardware_samples == 1
    assert resources.cpu_avg_percent is None
    assert resources.ram_avg_mib is None
    assert resources.gpu_available is False
    assert resources.model_vram_available is False
    assert resources.quality_per_cpu_point is None
    assert resources.quality_per_model_vram_gib is None
    assert any("CPU" in item for item in resources.limitations)
    assert any("model-specific VRAM" in item for item in resources.limitations)
