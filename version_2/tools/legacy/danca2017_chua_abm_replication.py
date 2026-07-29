#!/usr/bin/env python3
"""Closed partial validation helpers for the Danca 2017 Chua case.

The cited article fixes the non-smooth Chua equations, parameters, fractional
order, ABM method, and equilibrium-neighbourhood radius.  It does not disclose
an initial condition for the reported attractor.  Consequently this module
validates only the published algebraic and numerical-method contract; it does
not run dynamics or certify chaos or hiddenness.

This file is repository validation support.  It is not part of the installed
library or a user-facing workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class DancaChuaConfig:
    """Numerical values disclosed for the partial published-case validation."""

    q: float = 0.9998
    h: float = 0.01
    t_final: float = 500.0
    transient: float = 250.0
    alpha: float = 8.4562
    beta: float = 12.0732
    gamma_chua: float = 0.0052
    m0: float = -0.1768
    m1: float = -1.1468
    delta: float = 0.01

    def params(self) -> dict[str, float | str]:
        return {
            "model": "piecewise",
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "gamma": float(self.gamma_chua),
            "m0": float(self.m0),
            "m1": float(self.m1),
        }


def validate_config(config: DancaChuaConfig) -> None:
    """Validate the disclosed scalar contract without running trajectories."""

    if not (0.0 < float(config.q) <= 1.0):
        raise ValueError("q must satisfy 0 < q <= 1")
    if float(config.h) <= 0.0 or float(config.t_final) <= 0.0:
        raise ValueError("h and t_final must be positive")
    if not (0.0 <= float(config.transient) < float(config.t_final)):
        raise ValueError("transient must satisfy 0 <= transient < t_final")
    if float(config.delta) <= 0.0:
        raise ValueError("delta must be positive")


def solve_equilibria(parameters: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Solve the three equilibria of the disclosed non-smooth Chua model."""

    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    m0 = float(parameters["m0"])
    m1 = float(parameters["m1"])
    saturation_gain = m0 - m1

    def point(x_value: float) -> np.ndarray:
        y_value = gamma / (beta + gamma) * x_value
        z_value = -beta / (beta + gamma) * x_value
        return np.array([x_value, y_value, z_value], dtype=float)

    equilibria = {"E0": point(0.0)}
    denominator = (beta + gamma) * m1 + beta
    if abs(denominator) <= 1.0e-14:
        return equilibria

    positive_x = -((beta + gamma) * saturation_gain) / denominator
    positive = point(positive_x)
    negative = point(-positive_x)
    if positive[0] > 1.0 - 1.0e-10:
        equilibria["E+"] = positive
    if negative[0] < -1.0 + 1.0e-10:
        equilibria["E-"] = negative
    return equilibria


def local_jacobian(
    parameters: Mapping[str, Any],
    equilibrium: np.ndarray,
) -> np.ndarray:
    """Return the region-local Jacobian used by the algebraic check."""

    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    m0 = float(parameters["m0"])
    m1 = float(parameters["m1"])
    x_value = float(np.asarray(equilibrium, dtype=float)[0])
    saturation_slope = 1.0 if abs(x_value) < 1.0 - 1.0e-10 else 0.0
    effective_slope = m1 + (m0 - m1) * saturation_slope
    return np.array(
        [
            [-alpha * (1.0 + effective_slope), alpha, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, -beta, -gamma],
        ],
        dtype=float,
    )


__all__ = [
    "DancaChuaConfig",
    "local_jacobian",
    "solve_equilibria",
    "validate_config",
]
