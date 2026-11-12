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
from benchmark.core.config import (
    CATEGORY_NAMES,
    DEFAULT_PROFILE_NAME,
    DEFAULT_WORKLOAD_PROFILES,
    WorkloadProfileSettings,
)
from benchmark.core.exceptions import AnalyticsError
from benchmark.models.base import BaseSchema

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


def build_profiles(
    configured: Mapping[str, WorkloadProfileSettings | Mapping[str, object]] | None = None,
) -> dict[str, ScoringProfile]:
    """Build validated scoring profiles from configured weights.

    The defaults mirror ``configs/benchmark.yaml``. Callers that loaded settings
    should pass ``settings.workload_profiles`` so overrides take effect.
    """
    source = configured if configured is not None else DEFAULT_WORKLOAD_PROFILES
    profiles: dict[str, ScoringProfile] = {}
    for name, raw in source.items():
        if isinstance(raw, WorkloadProfileSettings):
            data = raw.model_dump()
        else:
            data = dict(raw)
        profiles[name] = ScoringProfile(name=name, **data)
    return profiles


def build_profiles_from_settings(settings: object) -> dict[str, ScoringProfile]:
    """Build profiles from an object exposing ``workload_profiles``."""
    configured = getattr(settings, "workload_profiles", None)
    if configured is None:
        raise AnalyticsError("settings do not define workload_profiles")
    return build_profiles(configured)


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
