"""Fischer et al. (2020) cloned-dynamics benchmark systems."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np


FISCHER_2020_DOI = "10.1016/j.apnum.2020.03.027"

FISCHER_BENCHMARKS: dict[str, dict[str, object]] = {
    "jerk": {
        "parameters": {"a": 0.5, "Ic": 10e-9, "VT": 0.026, "n": 2.0},
        "x0": [0.1, 0.1, 0.1],
        "nonsmooth_rhs": False,
    },
    "financial": {
        "parameters": {"a": 1.0, "b": 0.15, "c": 1.0},
        "x0": [1.0, 1.0, 1.0],
        "nonsmooth_rhs": True,
    },
    "four_wing": {
        "parameters": {"b": 0.53, "c": 3.0},
        "x0": [0.1, 0.1, 0.1],
        "nonsmooth_rhs": False,
    },
}


def _parameters(system_id: str, overrides: Mapping[str, float] | None) -> dict[str, float]:
    values = dict(FISCHER_BENCHMARKS[system_id]["parameters"])
    if overrides is not None:
        values.update(overrides)
    return values


def jerk_rhs(state: np.ndarray, parameters: Mapping[str, float] | None = None) -> np.ndarray:
    """Jerk benchmark with exponential diode nonlinearity."""

    x, y, z = np.asarray(state, dtype=float)
    p = _parameters("jerk", parameters)
    diode = p["Ic"] * (np.exp(y / (p["n"] * p["VT"])) - 1.0)
    return np.asarray([y, z, -p["a"] * z - diode - x], dtype=float)


def financial_rhs(state: np.ndarray, parameters: Mapping[str, float] | None = None) -> np.ndarray:
    """Nonsmooth financial benchmark; no Jacobian is required."""

    x, y, z = np.asarray(state, dtype=float)
    p = _parameters("financial", parameters)
    return np.asarray([z + (y - p["a"]) * x, 1.0 - p["b"] * y - abs(x), -x - p["c"] * z])


def four_wing_rhs(state: np.ndarray, parameters: Mapping[str, float] | None = None) -> np.ndarray:
    """Four-wing benchmark system."""

    x, y, z = np.asarray(state, dtype=float)
    p = _parameters("four_wing", parameters)
    return np.asarray([-x + y * z, y - x * z, p["b"] - p["c"] * z + x * y])


def get_fischer_benchmark(system_id: str) -> tuple[Callable, np.ndarray, dict[str, object]]:
    """Return RHS, initial state, and metadata for one Fischer benchmark."""

    functions = {"jerk": jerk_rhs, "financial": financial_rhs, "four_wing": four_wing_rhs}
    if system_id not in functions:
        raise KeyError(f"unknown Fischer benchmark system: {system_id}")
    metadata = dict(FISCHER_BENCHMARKS[system_id])
    return functions[system_id], np.asarray(metadata["x0"], dtype=float), metadata


__all__ = [
    "FISCHER_2020_DOI",
    "FISCHER_BENCHMARKS",
    "financial_rhs",
    "four_wing_rhs",
    "get_fischer_benchmark",
    "jerk_rhs",
]
