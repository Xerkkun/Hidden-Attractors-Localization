"""Two-phase lead-lag PLL in an exact shifted scalar Lur'e form.

The maintained integer-order parameters reproduce the phase-space model in
Bianchi et al. (2015), DOI 10.1109/ICUMT.2015.7382409.  The registered state
is ``(u, v) = (x - x_e, theta_delta - theta_s)`` around the stable locked
equilibrium.  The angle is integrated on its unwrapped lift; comparisons of
states and equilibria must use the cylinder metric declared in ``metadata``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.special import j0, j1

from .base import ChaoticSystem
from .lure import LureSystem


PLL_LEAD_LAG_2015_DOI = "10.1109/ICUMT.2015.7382409"
PLL_LEAD_LAG_2015_PARAMETERS: dict[str, float] = {
    "tau1": 0.0448,
    "tau2": 0.0185,
    "loop_gain": 500.0,
    "omega_delta": 178.9,
}


def pll_lead_lag_parameters(parameters: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Return validated PLL parameters and derived equilibrium quantities."""

    values = dict(PLL_LEAD_LAG_2015_PARAMETERS)
    if parameters:
        values.update({key: float(value) for key, value in parameters.items() if key in values})
    tau1 = float(values["tau1"])
    tau2 = float(values["tau2"])
    loop_gain = float(values["loop_gain"])
    omega_delta = float(values["omega_delta"])
    if not all(np.isfinite(value) for value in (tau1, tau2, loop_gain, omega_delta)):
        raise ValueError("PLL parameters must be finite.")
    if tau1 <= 0.0 or tau2 < 0.0 or loop_gain <= 0.0:
        raise ValueError("expected tau1 > 0, tau2 >= 0, and loop_gain > 0.")
    sine_equilibrium = 2.0 * omega_delta / loop_gain
    if abs(sine_equilibrium) >= 1.0:
        raise ValueError("the shifted Lur'e form requires two isolated locked equilibria.")
    total = tau1 + tau2
    theta_focus = float(np.arcsin(sine_equilibrium))
    theta_saddle = float(np.pi - theta_focus)
    cosine_focus = float(np.cos(theta_focus))
    x_equilibrium = float(0.5 * tau1 * sine_equilibrium)
    return {
        **values,
        "total_time_constant": total,
        "sine_equilibrium": sine_equilibrium,
        "cosine_focus": cosine_focus,
        "theta_focus": theta_focus,
        "theta_saddle": theta_saddle,
        "x_equilibrium": x_equilibrium,
        "saddle_offset": float(theta_saddle - theta_focus),
    }


def wrap_pll_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angular difference to ``[-pi, pi)``."""

    values = (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    if np.ndim(angle) == 0:
        return float(values)
    return values


def pll_original_to_shifted(
    state: np.ndarray, parameters: Mapping[str, Any] | None = None
) -> np.ndarray:
    """Convert ``(x, theta_delta)`` to registered coordinates ``(u, v)``."""

    point = np.asarray(state, dtype=float)
    if point.shape != (2,):
        raise ValueError("PLL state must have shape (2,).")
    p = pll_lead_lag_parameters(parameters)
    return np.array(
        [point[0] - p["x_equilibrium"], point[1] - p["theta_focus"]], dtype=float
    )


def pll_shifted_to_original(
    state: np.ndarray, parameters: Mapping[str, Any] | None = None
) -> np.ndarray:
    """Convert registered coordinates ``(u, v)`` to ``(x, theta_delta)``."""

    point = np.asarray(state, dtype=float)
    if point.shape != (2,):
        raise ValueError("PLL state must have shape (2,).")
    p = pll_lead_lag_parameters(parameters)
    return np.array(
        [point[0] + p["x_equilibrium"], point[1] + p["theta_focus"]], dtype=float
    )


def pll_shifted_nonlinearity(
    sigma: float, parameters: Mapping[str, Any] | None = None
) -> float:
    """Return ``sin(theta_s + sigma) - sin(theta_s)``."""

    p = pll_lead_lag_parameters(parameters)
    value = float(sigma)
    return float(np.sin(p["theta_focus"] + value) - p["sine_equilibrium"])


def pll_original_rhs(state: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate the published equations in ``(x, theta_delta)`` coordinates."""

    point = np.asarray(state, dtype=float)
    p = pll_lead_lag_parameters(parameters)
    total = p["total_time_constant"]
    sine = float(np.sin(point[1]))
    return np.array(
        [
            -point[0] / total + p["tau1"] * sine / (2.0 * total),
            p["omega_delta"]
            - p["loop_gain"] * point[0] / total
            - p["tau2"] * p["loop_gain"] * sine / (2.0 * total),
        ],
        dtype=float,
    )


def pll_lead_lag_rhs(state: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    """Evaluate the exact shifted Lur'e vector field in ``(u, v)``."""

    point = np.asarray(state, dtype=float)
    p = pll_lead_lag_parameters(parameters)
    total = p["total_time_constant"]
    phi = float(np.sin(p["theta_focus"] + point[1]) - p["sine_equilibrium"])
    return np.array(
        [
            -point[0] / total + p["tau1"] * phi / (2.0 * total),
            -p["loop_gain"] * point[0] / total
            - p["tau2"] * p["loop_gain"] * phi / (2.0 * total),
        ],
        dtype=float,
    )


def pll_lead_lag_jacobian(state: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    """Return the analytic Jacobian in shifted coordinates."""

    point = np.asarray(state, dtype=float)
    p = pll_lead_lag_parameters(parameters)
    total = p["total_time_constant"]
    cosine = float(np.cos(p["theta_focus"] + point[1]))
    return np.array(
        [
            [-1.0 / total, p["tau1"] * cosine / (2.0 * total)],
            [
                -p["loop_gain"] / total,
                -p["tau2"] * p["loop_gain"] * cosine / (2.0 * total),
            ],
        ],
        dtype=float,
    )


def pll_lead_lag_equilibria(parameters: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return the focus and saddle representatives on one phase cylinder."""

    p = pll_lead_lag_parameters(parameters)
    return {
        "E_focus": np.array([0.0, 0.0], dtype=float),
        "E_saddle": np.array([0.0, p["saddle_offset"]], dtype=float),
    }


def pll_shifted_sine_harmonics(
    amplitude: float,
    parameters: Mapping[str, Any] | None = None,
    *,
    bias: float = 0.0,
) -> dict[str, float]:
    """Return the exact DC and first-harmonic terms for a biased sinusoid.

    For ``v = bias + amplitude*sin(omega*t)``, ``dc`` is the average of the
    shifted sine characteristic and ``gain`` is its in-phase fundamental
    coefficient divided by ``amplitude``.  The gain is real; a static sine
    characteristic introduces no quadrature fundamental.
    """

    amp = float(amplitude)
    if amp <= 0.0 or not np.isfinite(amp):
        raise ValueError("amplitude must be positive and finite.")
    offset = float(bias)
    if not np.isfinite(offset):
        raise ValueError("bias must be finite.")
    p = pll_lead_lag_parameters(parameters)
    phase = p["theta_focus"] + offset
    return {
        "dc": float(np.sin(phase) * j0(amp) - p["sine_equilibrium"]),
        "fundamental": float(2.0 * np.cos(phase) * j1(amp)),
        "gain": float(2.0 * np.cos(phase) * j1(amp) / amp),
        "quadrature": 0.0,
    }


def pll_lure_transfer(
    spectral_parameter: complex,
    parameters: Mapping[str, Any] | None = None,
    *,
    convention: str = "standard",
) -> complex:
    """Return the exact scalar Lur'e transfer in either package convention.

    ``standard`` is ``c^T (sI-A)^(-1)b``.  ``code`` is the historical
    Kuznetsov convention ``c^T (A-sI)^(-1)b = -standard``.
    """

    if convention not in {"standard", "code"}:
        raise ValueError("convention must be 'standard' or 'code'.")
    p = pll_lead_lag_parameters(parameters)
    s = complex(spectral_parameter)
    total = p["total_time_constant"]
    value = -0.5 * p["loop_gain"] * (1.0 + p["tau2"] * s) / (
        s * (1.0 + total * s)
    )
    return complex(value if convention == "standard" else -value)


def pll_lead_lag_lure_system(parameters: Mapping[str, Any] | None = None) -> LureSystem:
    """Build the exact shifted scalar Lur'e declaration."""

    p = pll_lead_lag_parameters(parameters)
    total = p["total_time_constant"]
    matrix = np.array(
        [[-1.0 / total, 0.0], [-p["loop_gain"] / total, 0.0]], dtype=float
    )
    input_vector = np.array(
        [
            p["tau1"] / (2.0 * total),
            -p["tau2"] * p["loop_gain"] / (2.0 * total),
        ],
        dtype=float,
    )
    output_vector = np.array([0.0, 1.0], dtype=float)

    def nonlinearity(sigma: float) -> float:
        return pll_shifted_nonlinearity(sigma, p)

    def describing_function(amplitude: float) -> float:
        return pll_shifted_sine_harmonics(amplitude, p)["gain"]

    return LureSystem(
        name="pll-lead-lag-2015-shifted-lure",
        matrix=matrix,
        input_vector=input_vector,
        output_vector=output_vector,
        nonlinearity=nonlinearity,
        describing_function=describing_function,
        description=(
            "Exact T1 scalar Lur'e form after shifting the stable locked equilibrium; "
            "the declared DF is the centered first harmonic and its DC term is tracked separately."
        ),
    )


def pll_lead_lag_2015_system() -> ChaoticSystem:
    """Return the maintained integer two-phase lead-lag PLL reference."""

    parameters = dict(PLL_LEAD_LAG_2015_PARAMETERS)
    return ChaoticSystem(
        name="pll-lead-lag-2015",
        dimension=2,
        rhs=pll_lead_lag_rhs,
        equilibria=pll_lead_lag_equilibria,
        jacobian=pll_lead_lag_jacobian,
        parameters=parameters,
        description=(
            "Two-phase lead-lag PLL on a phase cylinder with a stable hidden "
            "running cycle and a stable locked equilibrium."
        ),
        tags=("integer", "lure", "non-chua", "pll", "hidden-limit-cycle", "cylinder"),
        workflows={"hidden": "Andronov return map and loop-gain continuation"},
        lure=pll_lead_lag_lure_system(parameters),
        state_names=("u", "v"),
        reference={
            "doi": PLL_LEAD_LAG_2015_DOI,
            "title": "Limitations of PLL simulation: hidden oscillations in MatLab and SPICE",
            "year": 2015,
        },
        metadata={
            "lure_form": "T1_exact_scalar_after_locked_equilibrium_shift",
            "state_coordinates": "u=x-x_e; v=theta_delta-theta_focus_unwrapped",
            "state_space": "R_times_S1",
            "cylinder_scales": {"u": parameters["tau1"] / 2.0, "v": float(np.pi)},
            "primary_route": "direct_integer_transfer_analytically_incompatible",
            "alternative_route": "andronov_phase_map_then_loop_gain_continuation",
            "published_initial_conditions_role": "post_derivation_regression_only",
        },
    )


__all__ = [
    "PLL_LEAD_LAG_2015_DOI",
    "PLL_LEAD_LAG_2015_PARAMETERS",
    "pll_lead_lag_2015_system",
    "pll_lead_lag_equilibria",
    "pll_lead_lag_jacobian",
    "pll_lead_lag_lure_system",
    "pll_lead_lag_parameters",
    "pll_lead_lag_rhs",
    "pll_lure_transfer",
    "pll_original_rhs",
    "pll_original_to_shifted",
    "pll_shifted_nonlinearity",
    "pll_shifted_sine_harmonics",
    "pll_shifted_to_original",
    "wrap_pll_angle",
]
