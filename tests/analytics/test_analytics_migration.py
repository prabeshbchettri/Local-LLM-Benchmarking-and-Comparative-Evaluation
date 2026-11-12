"""
File: test_analytics_migration.py

Description:
Regression coverage for the additive analytics persistence migration.

Responsibilities:
- Create a schema-1.0.0 recommendation table without details_json.
- Verify opening it migrates to 1.1.0 and adds the nullable evidence column.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from benchmark.database.migrations import (
    SCHEMA_VERSION,
    apply_schema,
    current_schema_version,
    utc_now,
)
from benchmark.database.tables import Base, SchemaVersionRow

pytestmark = pytest.mark.unit


def test_schema_1_0_0_migrates_recommendation_details(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE recommendations DROP COLUMN details_json"))
    with Session(engine) as session:
        session.add(SchemaVersionRow(version="1.0.0", applied_at=utc_now()))
        session.commit()

    assert current_schema_version(engine) == "1.0.0"
    assert "details_json" not in {
        column["name"] for column in inspect(engine).get_columns("recommendations")
    }

    assert apply_schema(engine) == SCHEMA_VERSION == "1.1.0"
    assert current_schema_version(engine) == "1.1.0"
    assert "details_json" in {
        column["name"] for column in inspect(engine).get_columns("recommendations")
    }
