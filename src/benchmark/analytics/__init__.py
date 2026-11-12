"""
File: __init__.py

Description:
Public exports for analytics, scoring, recommendations, and comparisons.

Why this file exists:
Callers should import one engine and the result types rather than individual modules.

Responsibilities:
- Export the analytics engine and profile helpers.
- Export statistics, normalization, and recommendation result models.
"""

from benchmark.analytics.engine import WORKLOAD_PROFILE_NAMES, AnalyticsEngine
from benchmark.analytics.models import (
    CategoryQuality,
    DescriptiveStats,
    DimensionScore,
    ExperimentAnalytics,
    ExperimentComparison,
    HistoricalComparison,
    MetricDelta,
    ModelAnalytics,
    ModelComparison,
    QualityAnalysis,
    Recommendation,
    RecommendationSet,
    ResourceUsage,
    ResponseStatistics,
)
from benchmark.analytics.normalization import (
    has_discrimination,
    min_max_scale,
    normalize_value,
    percentage,
    weighted_average,
)
from benchmark.analytics.recommendations import build_recommendation_set
from benchmark.analytics.scoring import (
    DEFAULT_PROFILE_NAME,
    PROFILES,
    ScoringProfile,
    build_profiles,
    build_profiles_from_settings,
    composite_score,
    get_profile,
)
from benchmark.analytics.statistics import (
    describe,
    response_consistency,
    response_statistics,
)

__all__ = [
    "DEFAULT_PROFILE_NAME",
    "PROFILES",
    "WORKLOAD_PROFILE_NAMES",
    "AnalyticsEngine",
    "CategoryQuality",
    "DescriptiveStats",
    "DimensionScore",
    "ExperimentAnalytics",
    "ExperimentComparison",
    "HistoricalComparison",
    "MetricDelta",
    "ModelAnalytics",
    "ModelComparison",
    "QualityAnalysis",
    "Recommendation",
    "RecommendationSet",
    "ResourceUsage",
    "ResponseStatistics",
    "ScoringProfile",
    "build_profiles",
    "build_profiles_from_settings",
    "build_recommendation_set",
    "composite_score",
    "describe",
    "get_profile",
    "has_discrimination",
    "min_max_scale",
    "normalize_value",
    "percentage",
    "response_consistency",
    "response_statistics",
    "weighted_average",
]
