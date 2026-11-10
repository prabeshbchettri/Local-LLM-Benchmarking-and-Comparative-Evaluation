"""
File: scoring.py

Description:
Validated scoring profiles and weighted composite scores.

Why this file exists:
Different workloads need different weights, but every profile must use the same five
dimensions, sum to 100%, and renormalize around missing measurements consistently.

Responsibilities:
- Define the general-purpose and workload-specific profiles.
- Validate dimensions, weights, and category weights.
- Compute category-weighted response quality.
- Combine dimension scores into a composite score.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from pydantic import Field, model_validator

from benchmark.analytics.models import DIMENSIONS, DimensionScore, QualityAnalysis
from benchmark.analytics.normalization import weighted_average
from benchmark.core.config import CATEGORY_NAMES
from benchmark.core.exceptions import AnalyticsError
from benchmark.models.base import BaseSchema

DEFAULT_PROFILE_NAME = "general_purpose"
WEIGHT_TOLERANCE = 1e-6


class ScoringProfile(BaseSchema):
    """Named weights for the five scoring dimensions and optional categories."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    weights: dict[str, float]
    category_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_profile(self) -> "ScoringProfile":
        unknown = set(self.weights) - set(DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown scoring dimensions: {sorted(unknown)}")
        missing = set(DIMENSIONS) - set(self.weights)
        if missing:
            raise ValueError(f"missing scoring dimensions: {sorted(missing)}")
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("scoring weights must not be negative")
        total = sum(self.weights.values())
        if not math.isclose(total, 100.0, abs_tol=WEIGHT_TOLERANCE):
            raise ValueError(f"scoring weights must total 100, got {total}")
        if self.category_weights:
            unknown_categories = set(self.category_weights) - set(CATEGORY_NAMES)
            if unknown_categories:
                raise ValueError(
                    f"unknown quality categories: {sorted(unknown_categories)}"
                )
            if any(weight < 0 for weight in self.category_weights.values()):
                raise ValueError("category weights must not be negative")
            category_total = sum(self.category_weights.values())
            if not math.isclose(category_total, 1.0, abs_tol=WEIGHT_TOLERANCE):
                raise ValueError(
                    f"category weights must total 1.0, got {category_total}"
                )
        return self

    @property
    def weight_fractions(self) -> dict[str, float]:
        """Return dimension weights as fractions that sum to one."""
        return {name: weight / 100.0 for name, weight in self.weights.items()}

    def weighted_quality(self, quality: QualityAnalysis) -> tuple[float | None, float]:
        """Return profile-specific quality and the category-weight coverage."""
        if not self.category_weights:
            if quality.composite_score is None:
                return None, 0.0
            return quality.composite_score, 1.0
        available = {
            category.category: category.composite_score
            for category in quality.categories
            if category.composite_score is not None
        }
        scores = {
            name: available[name]
            for name in self.category_weights
            if name in available
        }
        weights = {
            name: weight
            for name, weight in self.category_weights.items()
            if name in available
        }
        if not scores:
            return None, 0.0
        coverage = sum(weights.values()) / sum(self.category_weights.values())
        value = weighted_average(scores, weights)
        return value, coverage


def build_profiles() -> dict[str, ScoringProfile]:
    """Return the built-in scoring profiles keyed by workload name."""
    return {
        "general_purpose": ScoringProfile(
            name="general_purpose",
            description="Balanced profile for everyday local assistance.",
            weights={
                "quality": 40.0,
                "latency": 20.0,
                "throughput": 15.0,
                "efficiency": 10.0,
                "reliability": 15.0,
            },
        ),
        "coding": ScoringProfile(
            name="coding",
            description="Code generation and editing with correctness emphasis.",
            weights={
                "quality": 45.0,
                "latency": 15.0,
                "throughput": 15.0,
                "efficiency": 10.0,
                "reliability": 15.0,
            },
            category_weights={
                "coding": 0.7,
                "instruction_following": 0.2,
                "logical_reasoning": 0.1,
            },
        ),
        "reasoning": ScoringProfile(
            name="reasoning",
            description="Analytical tasks that reward multi-step reasoning.",
            weights={
                "quality": 50.0,
                "latency": 10.0,
                "throughput": 10.0,
                "efficiency": 10.0,
                "reliability": 20.0,
            },
            category_weights={
                "logical_reasoning": 0.45,
                "mathematical_reasoning": 0.35,
                "general_knowledge": 0.2,
            },
        ),
        "json_generation": ScoringProfile(
            name="json_generation",
            description="Structured output tasks with schema compliance emphasis.",
            weights={
                "quality": 50.0,
                "latency": 10.0,
                "throughput": 20.0,
                "efficiency": 5.0,
                "reliability": 15.0,
            },
            category_weights={
                "structured_json": 0.8,
                "instruction_following": 0.2,
            },
        ),
        "resource_constrained": ScoringProfile(
            name="resource_constrained",
            description="Memory- and compute-sensitive deployments.",
            weights={
                "quality": 30.0,
                "latency": 20.0,
                "throughput": 15.0,
                "efficiency": 25.0,
                "reliability": 10.0,
            },
        ),
    }


PROFILES = build_profiles()


def get_profile(
    name: str,
    profiles: Mapping[str, ScoringProfile] | None = None,
) -> ScoringProfile:
    """Return a profile by name or raise an analytics configuration error."""
    available = profiles if profiles is not None else PROFILES
    try:
        return available[name]
    except KeyError as exc:
        raise AnalyticsError(
            f"unknown scoring profile {name!r}; available: {', '.join(sorted(available))}"
        ) from exc


def composite_score(
    dimensions: Mapping[str, DimensionScore],
    profile: ScoringProfile,
) -> float | None:
    """Combine normalized dimensions, renormalizing over available measurements."""
    quality = dimensions.get("quality")
    if quality is None or quality.normalized_score is None:
        return None
    scores = {name: item.normalized_score for name, item in dimensions.items()}
    return weighted_average(scores, profile.weight_fractions)
