"""
File: normalization.py

Description:
Reusable metric normalization and weighted averaging.

Why this file exists:
Recommendations compare metrics with different units and directions. Percent scale
and missing-value rules must be identical everywhere to keep scores explainable.

Responsibilities:
- Normalize a value or a set of values to 0-100.
- Support higher-is-better and lower-is-better metrics.
- Preserve missing values instead of converting them to zero.
- Renormalize weights over available dimensions and reject invalid weights.
"""

from __future__ import annotations

from collections.abc import Mapping


def normalize_value(
    value: float | None,
    minimum: float | None,
    maximum: float | None,
    *,
    higher_is_better: bool = True,
) -> float | None:
    """Normalize one value to 0-100; return None if it cannot be scaled.

    When all models share the same value, min-max scaling cannot distinguish them.
    Returning the neutral midpoint avoids awarding an artificial 100 to every model.
    """
    if value is None or minimum is None or maximum is None:
        return None
    if maximum < minimum:
        raise ValueError("maximum must be greater than or equal to minimum")
    if maximum == minimum:
        return 50.0
    ratio = (value - minimum) / (maximum - minimum)
    if not higher_is_better:
        ratio = 1.0 - ratio
    return round(ratio * 100.0, 6)


def has_discrimination(values: Mapping[str, float | None]) -> bool:
    """Return whether the observed values contain more than one distinct value."""
    present = [value for value in values.values() if value is not None]
    return bool(present) and max(present) > min(present)


def min_max_scale(
    values: Mapping[str, float | None],
    *,
    higher_is_better: bool = True,
) -> dict[str, float | None]:
    """Scale values relative to the observed minimum and maximum."""
    present = [value for value in values.values() if value is not None]
    if not present:
        return {key: None for key in values}
    minimum = min(present)
    maximum = max(present)
    return {
        key: normalize_value(
            value,
            minimum,
            maximum,
            higher_is_better=higher_is_better,
        )
        for key, value in values.items()
    }


def weighted_average(
    scores: Mapping[str, float | None],
    weights: Mapping[str, float],
) -> float | None:
    """Average available scores with weights renormalized over what was measured."""
    unknown = set(weights) - set(scores)
    if unknown:
        raise ValueError(f"unknown weighted metrics: {sorted(unknown)}")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("weights must not be negative")
    active = {
        name: weight
        for name, weight in weights.items()
        if weight > 0 and scores.get(name) is not None
    }
    total_weight = sum(active.values())
    if total_weight <= 0:
        return None
    total = sum(scores[name] * weight for name, weight in active.items())  # type: ignore[operator]
    return round(total / total_weight, 6)


def percentage(part: float, total: float) -> float | None:
    """Return part/total as a percentage, or None when total is zero."""
    if total == 0:
        return None
    return part / total * 100.0
