"""Objective-direction helpers shared by the CALM protocol.

Raw target values always retain their reported physical units.  Acquisition
models consume an oriented utility for which larger is better, while metrics
and plots continue to expose the raw target scale.
"""

from __future__ import annotations

import numpy as np


OBJECTIVE_DIRECTIONS = ("maximize", "minimize")
OBJECTIVE_DIRECTION_CONTRACT_VERSION = (
    "registered_raw_target_direction_oriented_utility_v1"
)


def validate_objective_direction(value: str) -> str:
    direction = str(value).strip().lower()
    if direction not in OBJECTIVE_DIRECTIONS:
        raise ValueError(
            "objective_direction must be exactly 'maximize' or 'minimize', "
            f"got {value!r}"
        )
    return direction


def objective_sign(direction: str) -> float:
    """Return the multiplier mapping raw targets to larger-is-better utility."""
    return 1.0 if validate_objective_direction(direction) == "maximize" else -1.0


def to_utility(values, direction: str) -> np.ndarray:
    """Orient raw physical target values so larger always means better."""
    return np.asarray(values, dtype=np.float64) * objective_sign(direction)


def to_raw_target(values, direction: str) -> np.ndarray:
    """Invert :func:`to_utility` without changing the physical target scale."""
    return np.asarray(values, dtype=np.float64) * objective_sign(direction)


def objective_optimum(values, direction: str) -> float:
    raw = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise ValueError("Objective values must be finite")
    if validate_objective_direction(direction) == "maximize":
        return float(np.max(raw))
    return float(np.min(raw))


def best_observed(values, direction: str) -> float:
    """Return best raw physical value under the registered direction."""
    return objective_optimum(values, direction)


def objective_regret(
    best_values,
    *,
    global_min: float,
    global_max: float,
    direction: str,
) -> np.ndarray:
    """Return non-negative simple regret on the raw physical target scale."""
    best = np.asarray(best_values, dtype=np.float64)
    if validate_objective_direction(direction) == "maximize":
        return np.maximum(float(global_max) - best, 0.0)
    return np.maximum(best - float(global_min), 0.0)


def is_global_optimum(
    value: float,
    *,
    global_min: float,
    global_max: float,
    direction: str,
) -> bool:
    optimum = (
        float(global_max)
        if validate_objective_direction(direction) == "maximize"
        else float(global_min)
    )
    return bool(np.isclose(float(value), optimum, rtol=1e-9, atol=1e-6))


def best_curve_is_monotonic(values, direction: str, atol: float = 1e-10) -> bool:
    """Check best-so-far monotonicity on the raw target scale."""
    delta = np.diff(np.asarray(values, dtype=np.float64))
    if validate_objective_direction(direction) == "maximize":
        return bool(np.all(delta >= -float(atol)))
    return bool(np.all(delta <= float(atol)))
