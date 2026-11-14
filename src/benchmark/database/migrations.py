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

from sqlalchemy import Engine, select, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from benchmark.core.exceptions import DatabaseError
from benchmark.database.tables import Base, SchemaVersionRow

SCHEMA_VERSION = "1.1.0"


def _add_recommendation_details(connection: Connection) -> None:
    """Add persisted recommendation evidence columns without rewriting history."""
    columns = {
        str(row[1]).lower()
        for row in connection.execute(text("PRAGMA table_info(recommendations)"))
    }
    if "details_json" not in columns:
        connection.execute(
            text("ALTER TABLE recommendations ADD COLUMN details_json JSON")
        )


# Additive migrations keyed by the version they upgrade from. Each entry names
# the version it produces and receives a live connection inside a transaction.
MIGRATIONS: dict[str, tuple[str, Callable[[Connection], None]]] = {
    "1.0.0": ("1.1.0", _add_recommendation_details),
}


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
    """Create missing tables and apply every migration needed for this build."""
    Base.metadata.create_all(engine)
    stored = current_schema_version(engine)
    if stored is None:
        with Session(engine) as session:
            session.add(SchemaVersionRow(version=SCHEMA_VERSION, applied_at=utc_now()))
            session.commit()
        return SCHEMA_VERSION
    while stored != SCHEMA_VERSION:
        step = MIGRATIONS.get(stored)
        if step is None:
            raise DatabaseError(
                f"database schema version {stored} is not supported by this build "
                f"(expected {SCHEMA_VERSION}); back up the database before upgrading"
            )
        target, migrate = step
        with engine.begin() as connection:
            migrate(connection)
            connection.execute(
                text(
                    "UPDATE schema_versions SET version = :version "
                    "WHERE id = (SELECT MAX(id) FROM schema_versions)"
                ),
                {"version": target},
            )
        stored = target
    return SCHEMA_VERSION
