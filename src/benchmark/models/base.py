"""
File: base.py

Description:
Shared Pydantic base model for configuration and data schemas.

Responsibilities:
- Reject unknown fields.
- Strip surrounding whitespace from string values.
- Validate assignments so configuration objects stay consistent.
"""

from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """Base class for all project Pydantic schemas."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )
