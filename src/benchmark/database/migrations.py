"""
File: migrations.py

Description:
Schema version tracking and additive schema creation for the SQLite database.

Why this file exists:
The database must be created on first use and must refuse to open a schema that is
newer than the running code, instead of silently misreading it.

Responsibilities:
- Create all tables through SQLAlchemy metadata.
- Record the applied schema version once.
- Reject an unknown or downgraded schema version.
- Provide a registration point for future additive migrations.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from benchmark.core.exceptions import DatabaseError
from benchmark.database.tables import Base, SchemaVersionRow

SCHEMA_VERSION = "1.0.0"

# Future additive migrations keyed by the version they produce. Each callable
# receives a live connection and must only add tables, indexes, or nullable columns.
MIGRATIONS: dict[str, Callable[[Engine], None]] = {}


def utc_now() -> datetime:
    """Return the current UTC time for stored timestamps."""
    return datetime.now(timezone.utc)


def current_schema_version(engine: Engine) -> str | None:
    """Return the stored schema version, or None for a new database."""
    with Session(engine) as session:
        row = session.scalar(
            select(SchemaVersionRow).order_by(SchemaVersionRow.id.desc()).limit(1)
        )
        return row.version if row is not None else None


def apply_schema(engine: Engine) -> str:
    """Create missing tables and verify that the stored schema version is usable."""
    Base.metadata.create_all(engine)
    stored = current_schema_version(engine)
    if stored is None:
        with Session(engine) as session:
            session.add(SchemaVersionRow(version=SCHEMA_VERSION, applied_at=utc_now()))
            session.commit()
        return SCHEMA_VERSION
    if stored == SCHEMA_VERSION:
        return stored
    if stored not in MIGRATIONS:
        raise DatabaseError(
            f"database schema version {stored} is not supported by this build "
            f"(expected {SCHEMA_VERSION}); back up the database before upgrading"
        )
    MIGRATIONS[stored](engine)
    with Session(engine) as session:
        row = session.scalar(
            select(SchemaVersionRow).order_by(SchemaVersionRow.id.desc()).limit(1)
        )
        if row is None:
            raise DatabaseError("database schema version record disappeared")
        row.version = SCHEMA_VERSION
        session.commit()
    return SCHEMA_VERSION
