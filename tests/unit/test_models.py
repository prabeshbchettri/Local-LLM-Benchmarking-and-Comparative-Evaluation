"""
File: test_models.py

Description:
Unit tests for the shared Pydantic base schema.

Responsibilities:
- Verify unknown fields are rejected.
- Verify string values are stripped.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchmark.models.base import BaseSchema

pytestmark = pytest.mark.unit


class SampleSchema(BaseSchema):
    name: str


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SampleSchema(name="ok", extra="nope")


def test_strings_are_stripped() -> None:
    schema = SampleSchema(name="  padded  ")
    assert schema.name == "padded"
