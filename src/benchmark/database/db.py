"""
File: db.py

Description:
Reusable SQLite engine, connection pragmas, and transaction scope management.

Why this file exists:
The application must not open a new database connection for every field write, and
it must enforce foreign keys, WAL journaling, and short write transactions in one place.

Responsibilities:
- Build one engine per database file and reuse it.
- Enable foreign keys, WAL mode, and a busy timeout on every connection.
- Provide a session scope that commits once or rolls back completely.
- Resolve the default database path under storage/benchmark.db.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from benchmark.core.config import AppSettings
from benchmark.core.exceptions import DatabaseError
from benchmark.database.migrations import apply_schema

DATABASE_FILENAME = "benchmark.db"


def _enable_sqlite_features(engine: Engine) -> None:
    """Apply SQLite pragmas to every new connection on this engine."""

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")
        finally:
            cursor.close()


class Database:
    """Owns the SQLAlchemy engine and provides short, explicit sessions."""

    def __init__(self, path: Path, *, echo: bool = False) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.path.as_posix()}",
            echo=echo,
            future=True,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        _enable_sqlite_features(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            future=True,
        )

    @classmethod
    def from_settings(cls, settings: AppSettings, *, echo: bool = False) -> "Database":
        """Open the configured database, defaulting to storage/benchmark.db."""
        return cls(settings.benchmark.storage_dir / DATABASE_FILENAME, echo=echo)

    def initialize(self) -> str:
        """Create the schema if needed and return the active schema version."""
        return apply_schema(self.engine)

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """Commit once on success, roll back completely on any failure."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def table_names(self) -> list[str]:
        """Return the table names present in the database file."""
        with self.engine.connect() as connection:
            rows = connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            )
            return [row[0] for row in rows if not row[0].startswith("sqlite_")]

    def foreign_keys_enabled(self) -> bool:
        """Return whether foreign-key enforcement is active on a connection."""
        with self.engine.connect() as connection:
            return bool(connection.execute(text("PRAGMA foreign_keys")).scalar())

    def journal_mode(self) -> str:
        """Return the active SQLite journal mode."""
        with self.engine.connect() as connection:
            return str(connection.execute(text("PRAGMA journal_mode")).scalar())

    def dispose(self) -> None:
        """Close pooled connections; the file remains on disk."""
        self.engine.dispose()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *args: object) -> None:
        self.dispose()


def open_database(settings: AppSettings, path: Path | None = None) -> Database:
    """Open and initialize a database, raising a typed error on failure."""
    database = Database.from_settings(settings) if path is None else Database(path)
    try:
        database.initialize()
    except DatabaseError:
        database.dispose()
        raise
    except Exception as exc:  # pragma: no cover - defensive wrapper
        database.dispose()
        raise DatabaseError(f"Could not initialize database {database.path}: {exc}") from exc
    return database
