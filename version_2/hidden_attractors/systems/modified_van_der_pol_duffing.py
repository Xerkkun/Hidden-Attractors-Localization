"""Modified autonomous Van der Pol--Duffing system in scalar Lur'e form.

The maintained integer-order declaration follows Matouk et al. (2023),
DOI 10.3390/math11030591.  With ``sigma = y1`` its only nonlinear term is
represented exactly by ``b * psi(sigma)``, where ``psi(sigma) = sigma**3``
and ``b = (-delta, 0, 0)``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import ChaoticSystem
from .lure import LureSystem


MAVPD_2023_DOI = "10.3390/math11030591"
MAVPD_2023_PARAMETERS: dict[str, float] = {
    "gamma": 0.1,
    "delta": 100.0,
    "rho": 200.0,
    "xi": 3.5,
}


def mavpd_parameters(parameters: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Return the published parameter set with optional finite overrides."""

    values = dict(MAVPD_2023_PARAMETERS)
    if parameters:
        values.update({key: float(value) for key, value in parameters.items() if key in values})
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("MAVPD parameters must be finite.")
    if values["delta"] == 0.0:
        raise ValueError("delta must be nonzero for the maintained MAVPD form.")
    if values["rho"] == 0.0:
        raise ValueError("rho must be nonzero for isolated MAVPD equilibria.")
    return values


def mavpd_matrix(parameters: Mapping[str, Any] | None = None) -> np.ndarray:
    """Return the linear matrix ``A`` of the exact Lur'e decomposition."""

    p = mavpd_parameters(parameters)
    return np.array(
        [
            [p["delta"] * p["gamma"], p["delta"], 0.0],
            [1.0, -p["xi"], -1.0],
            [0.0, p["rho"], 0.0],
        ],
        dtype=float,
    )


def mavpd_rhs(state: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate the integer-order modified Van der Pol--Duffing equations."""

    y = np.asarray(state, dtype=float)
    p = mavpd_parameters(parameters)
    y1, y2, y3 = y
    return np.array(
        [
            p["delta"] * p["gamma"] * y1
            + p["delta"] * y2
            - p["delta"] * y1**3,
            y1 - p["xi"] * y2 - y3,
            p["rho"] * y2,
        ],
        dtype=float,
    )


def mavpd_jacobian(state: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    """Return the analytic Jacobian of the integer-order MAVPD vector field."""

    y = np.asarray(state, dtype=float)
    p = mavpd_parameters(parameters)
    y1 = float(y[0])
    return np.array(
        [
            [p["delta"] * (p["gamma"] - 3.0 * y1 * y1), p["delta"], 0.0],
            [1.0, -p["xi"], -1.0],
            [0.0, p["rho"], 0.0],
        ],
        dtype=float,
    )


def mavpd_equilibria(parameters: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return all real isolated equilibria for the active parameter set."""

    p = mavpd_parameters(parameters)
    equilibria = {"E0": np.zeros(3, dtype=float)}
    if p["gamma"] > 0.0:
        root = float(np.sqrt(p["gamma"]))
        equilibria["E+"] = np.array([root, 0.0, root], dtype=float)
        equilibria["E-"] = np.array([-root, 0.0, -root], dtype=float)
    return equilibria


def mavpd_nonzero_equilibrium_characteristic_coefficients(
    parameters: Mapping[str, Any] | None = None,
) -> tuple[float, float, float]:
    """Return ``(a1,a2,a3)`` for ``lambda^3+a1 lambda^2+a2 lambda+a3`` at E±."""

    p = mavpd_parameters(parameters)
    doubled_delta_gamma = 2.0 * p["delta"] * p["gamma"]
    return (
        p["xi"] + doubled_delta_gamma,
        p["rho"] - p["delta"] + doubled_delta_gamma * p["xi"],
        doubled_delta_gamma * p["rho"],
    )


def mavpd_hopf_gamma_boundaries(
    parameters: Mapping[str, Any] | None = None,
) -> tuple[float, ...]:
    """Derive positive Routh--Hurwitz Hopf boundaries in ``gamma`` for E±.

    Writing ``g=2*delta*gamma``, the equality ``a1*a2=a3`` reduces to
    ``xi*g^2 + (xi^2-delta)*g + xi*(rho-delta)=0``.  The function solves
    this polynomial directly; it does not use values copied from a report.
    """

    p = mavpd_parameters(parameters)
    coefficients = np.array(
        [p["xi"], p["xi"] ** 2 - p["delta"], p["xi"] * (p["rho"] - p["delta"])],
        dtype=float,
    )
    roots = np.roots(coefficients)
    boundaries = sorted(
        float(root.real / (2.0 * p["delta"]))
        for root in roots
        if abs(float(root.imag)) <= 1.0e-10 and float(root.real) > 0.0
    )
    return tuple(boundaries)


def mavpd_lure_system(parameters: Mapping[str, Any] | None = None) -> LureSystem:
    """Build the exact scalar Lur'e declaration and cubic describing function."""

    p = mavpd_parameters(parameters)

    def nonlinearity(sigma: float) -> float:
        value = float(sigma)
        return value**3

    def describing_function(amplitude: float) -> float:
        value = float(amplitude)
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError("amplitude must be positive and finite.")
        return 0.75 * value * value

    def gain_compatible(gain: float) -> bool:
        value = float(gain)
        return bool(np.isfinite(value) and value > 0.0)

    def amplitude_from_gain(gain: float) -> float:
        value = float(gain)
        if not gain_compatible(value):
            raise RuntimeError("gain is outside the positive cubic describing-function range.")
        return float(np.sqrt((4.0 / 3.0) * value))

    return LureSystem(
        name="modified-van-der-pol-duffing-lure",
        matrix=mavpd_matrix(p),
        input_vector=np.array([-p["delta"], 0.0, 0.0], dtype=float),
        output_vector=np.array([1.0, 0.0, 0.0], dtype=float),
        nonlinearity=nonlinearity,
        describing_function=describing_function,
        gain_compatible=gain_compatible,
        amplitude_from_gain=amplitude_from_gain,
        description="Exact scalar Lur'e split of the integer MAVPD system with psi(sigma)=sigma^3.",
    )


def mavpd_2023_system(parameters: Mapping[str, Any] | None = None) -> ChaoticSystem:
    """Return the maintained integer-order MAVPD reference system."""

    values = mavpd_parameters(parameters)
    return ChaoticSystem(
        name="modified-van-der-pol-duffing",
        dimension=3,
        rhs=mavpd_rhs,
        equilibria=mavpd_equilibria,
        jacobian=mavpd_jacobian,
        parameters=values,
        description="Integer modified autonomous Van der Pol--Duffing system with one cubic nonlinearity.",
        tags=("integer", "lure", "non-chua", "mavpd", "cubic", "hidden-attractors"),
        workflows={"hidden": "direct integer Lur'e describing-function route"},
        lure=mavpd_lure_system(values),
        state_names=("y1", "y2", "y3"),
        reference={
            "doi": MAVPD_2023_DOI,
            "title": (
                "Existence of Self-Excited and Hidden Attractors in the Modified "
                "Autonomous Van Der Pol-Duffing Systems"
            ),
            "year": 2023,
        },
        metadata={
            "lure_form": "exact_scalar",
            "primary_route": "direct_integer_transfer",
            "published_xi_cases": (3.5, 3.1),
        },
    )


__all__ = [
    "MAVPD_2023_DOI",
    "MAVPD_2023_PARAMETERS",
    "mavpd_2023_system",
    "mavpd_equilibria",
    "mavpd_jacobian",
    "mavpd_lure_system",
    "mavpd_matrix",
    "mavpd_hopf_gamma_boundaries",
    "mavpd_nonzero_equilibrium_characteristic_coefficients",
    "mavpd_parameters",
    "mavpd_rhs",
]
