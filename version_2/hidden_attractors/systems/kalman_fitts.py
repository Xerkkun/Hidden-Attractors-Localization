"""Kalman--Fitts hidden-limit-cycle benchmark in exact scalar Lur'e form.

The maintained parameter set is the fourth-order counterexample reported by
Kuznetsov et al. (2019), DOI 10.1016/j.ifacol.2019.11.747.  Its target
nonlinearity is ``tanh(sigma / epsilon)``.  The paper reaches the hidden
cycles through the explicit ``sign -> tanh`` nonlinearity homotopy; classical
describing-function closure is deliberately incompatible for this example.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import ChaoticSystem
from .lure import LureSystem


KALMAN_FITTS_2019_DOI = "10.1016/j.ifacol.2019.11.747"
KALMAN_FITTS_2019_PARAMETERS: dict[str, float] = {
    "m1": 0.9,
    "m2": 1.1,
    "beta": 0.03,
    "epsilon": 0.01,
}


def kalman_fitts_coefficients(parameters: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Return the companion-polynomial coefficients for the active parameters."""

    values = dict(KALMAN_FITTS_2019_PARAMETERS)
    if parameters:
        values.update({key: float(value) for key, value in parameters.items() if key in values})
    m1 = values["m1"]
    m2 = values["m2"]
    beta = values["beta"]
    return {
        **values,
        "a0": (m1 * m1 + beta * beta) * (m2 * m2 + beta * beta),
        "a1": 2.0 * beta * (m1 * m1 + m2 * m2 + 2.0 * beta * beta),
        "a2": m1 * m1 + m2 * m2 + 6.0 * beta * beta,
        "a3": 4.0 * beta,
    }


def kalman_fitts_matrix(parameters: Mapping[str, Any] | None = None) -> np.ndarray:
    """Return the fourth-order companion matrix ``A``."""

    p = kalman_fitts_coefficients(parameters)
    return np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-p["a0"], -p["a1"], -p["a2"], -p["a3"]],
        ],
        dtype=float,
    )


def kalman_fitts_rhs(state: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate ``x' = A x + b tanh(c^T x / epsilon)``."""

    x = np.asarray(state, dtype=float)
    p = kalman_fitts_coefficients(parameters)
    matrix = kalman_fitts_matrix(p)
    sigma = -float(x[2])
    out = matrix @ x
    out[3] += float(np.tanh(sigma / p["epsilon"]))
    return out


def kalman_fitts_jacobian(state: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    """Return the analytic Jacobian of the smooth target system."""

    x = np.asarray(state, dtype=float)
    p = kalman_fitts_coefficients(parameters)
    matrix = kalman_fitts_matrix(p)
    sigma = -float(x[2])
    tanh_value = float(np.tanh(sigma / p["epsilon"]))
    derivative = (1.0 - tanh_value * tanh_value) / p["epsilon"]
    matrix[3, 2] -= derivative
    return matrix


def kalman_fitts_equilibria(_parameters: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return the unique equilibrium of the maintained model."""

    return {"E0": np.zeros(4, dtype=float)}


def kalman_fitts_lure_system(parameters: Mapping[str, Any] | None = None) -> LureSystem:
    """Build the exact scalar Lur'e declaration and its classical DF."""

    p = kalman_fitts_coefficients(parameters)
    matrix = kalman_fitts_matrix(p)
    input_vector = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    output_vector = np.array([0.0, 0.0, -1.0, 0.0], dtype=float)
    epsilon = p["epsilon"]

    # Fixed Gauss--Legendre rule for the odd static nonlinearity DF.
    nodes, weights = np.polynomial.legendre.leggauss(256)
    theta = 0.5 * np.pi * (nodes + 1.0)
    theta_weights = 0.5 * np.pi * weights
    sin_theta = np.sin(theta)

    def nonlinearity(sigma: float) -> float:
        return float(np.tanh(float(sigma) / epsilon))

    def describing_function(amplitude: float) -> float:
        amp = float(amplitude)
        if amp <= 0.0 or not np.isfinite(amp):
            raise ValueError("amplitude must be positive and finite.")
        integrand = np.tanh((amp / epsilon) * sin_theta) * sin_theta
        return float((2.0 / (np.pi * amp)) * np.sum(theta_weights * integrand))

    def gain_compatible(gain: float) -> bool:
        value = float(gain)
        return bool(np.isfinite(value) and 0.0 < value < 1.0 / epsilon)

    def amplitude_from_gain(gain: float) -> float:
        target = float(gain)
        if not gain_compatible(target):
            raise RuntimeError("gain is outside the positive tanh describing-function range.")
        left = 1.0e-12
        right = max(1.0, 4.0 / target)
        while describing_function(right) > target and right < 1.0e8:
            right *= 2.0
        if describing_function(right) > target:
            raise RuntimeError("failed to bracket the tanh describing-function amplitude.")
        for _ in range(120):
            midpoint = 0.5 * (left + right)
            if describing_function(midpoint) > target:
                left = midpoint
            else:
                right = midpoint
        return float(0.5 * (left + right))

    return LureSystem(
        name="kalman-fitts-2019-lure",
        matrix=matrix,
        input_vector=input_vector,
        output_vector=output_vector,
        nonlinearity=nonlinearity,
        describing_function=describing_function,
        gain_compatible=gain_compatible,
        amplitude_from_gain=amplitude_from_gain,
        description="Exact scalar Lur'e Kalman--Fitts benchmark; target reached by sign-to-tanh continuation.",
    )


def kalman_fitts_2019_system() -> ChaoticSystem:
    """Return the maintained integer-order Kalman--Fitts reference system."""

    parameters = dict(KALMAN_FITTS_2019_PARAMETERS)
    return ChaoticSystem(
        name="kalman-fitts-2019",
        dimension=4,
        rhs=kalman_fitts_rhs,
        equilibria=kalman_fitts_equilibria,
        jacobian=kalman_fitts_jacobian,
        parameters=parameters,
        description="Fourth-order non-Chua Lur'e system with hidden limit cycles and one stable equilibrium.",
        tags=("integer", "lure", "non-chua", "hidden-limit-cycle", "kalman"),
        workflows={"hidden": "sign-to-tanh nonlinearity continuation"},
        lure=kalman_fitts_lure_system(parameters),
        state_names=("x1", "x2", "x3", "x4"),
        reference={
            "doi": KALMAN_FITTS_2019_DOI,
            "title": "Coexistence of hidden attractors and multistability in counterexamples to the Kalman conjecture",
            "year": 2019,
        },
        metadata={
            "lure_form": "exact_scalar",
            "primary_route": "direct_df_incompatible_by_sign",
            "alternative_route": "andronov_switching_map_then_sign_to_tanh",
        },
    )


__all__ = [
    "KALMAN_FITTS_2019_DOI",
    "KALMAN_FITTS_2019_PARAMETERS",
    "kalman_fitts_2019_system",
    "kalman_fitts_coefficients",
    "kalman_fitts_equilibria",
    "kalman_fitts_jacobian",
    "kalman_fitts_lure_system",
    "kalman_fitts_matrix",
    "kalman_fitts_rhs",
]
