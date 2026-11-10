"""
File: conftest.py

Description:
Shared fixtures for deterministic evaluator and judge tests.

Responsibilities:
- Load the versioned dataset once per test session.
- Provide item lookup by stable dataset ID.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from benchmark.dataset.loader import load_dataset
from benchmark.dataset.models import BenchmarkDataset, DatasetItem


@pytest.fixture(scope="session")
def evaluation_dataset(dataset_path) -> BenchmarkDataset:
    return load_dataset(dataset_path)


@pytest.fixture(scope="session")
def item_for(evaluation_dataset: BenchmarkDataset) -> Callable[[str], DatasetItem]:
    items = {item.id: item for item in evaluation_dataset.items}

    def lookup(item_id: str) -> DatasetItem:
        return items[item_id]

    return lookup
