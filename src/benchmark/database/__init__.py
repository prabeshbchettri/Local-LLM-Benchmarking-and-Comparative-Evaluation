"""
File: __init__.py

Description:
Public exports for the persistence layer.

Why this file exists:
Callers should import the database handle, repositories, and read models from one
package instead of reaching into individual modules.

Responsibilities:
- Export the connection helpers and schema version.
- Export the repository container and its read models.
"""

from benchmark.database.db import DATABASE_FILENAME, Database, open_database
from benchmark.database.migrations import SCHEMA_VERSION, utc_now
from benchmark.database.repositories import (
    AnalyticsRow,
    ExperimentView,
    ModelResultView,
    PendingEvaluation,
    QuestionResultView,
    RecommendationInput,
    RegisteredModel,
    Repositories,
    RunView,
)

__all__ = [
    "DATABASE_FILENAME",
    "SCHEMA_VERSION",
    "AnalyticsRow",
    "Database",
    "ExperimentView",
    "ModelResultView",
    "PendingEvaluation",
    "QuestionResultView",
    "RecommendationInput",
    "RegisteredModel",
    "Repositories",
    "RunView",
    "open_database",
    "utc_now",
]
