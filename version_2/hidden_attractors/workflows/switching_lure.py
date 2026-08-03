"""Alternative integer Lur'e route based on a switching map and homotopy.

This module is intentionally separate from the direct describing-function
route.  It supports published constructions that first localize a periodic
orbit for ``sign(c^T x)`` and then continue that orbit to a smooth target
nonlinearity.  No frequency scan is used by the switching-map seed finder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq

from ..solvers.integer import efork_q1_integrate
from ..systems.base import ChaoticSystem
from ..systems.lure import LureSystem, ScalarNonlinearity
from .integer_lure import require_lure
from .protocol import ContinuationPlan


def sign_nonlinearity(sigma: float) -> float:
    """Return the deterministic two-level precursor used at the switching surface."""

    return 1.0 if float(sigma) >= 0.0 else -1.0


@dataclass(frozen=True)
class SwitchingMapSeed:
    """Periodic switching-map seed obtained from the declared linear system."""

    seed: np.ndarray
    initial_section_state: np.ndarray
    crossing_states: np.ndarray
    crossing_times: np.ndarray
    return_period: int
    convergence_error: float
    iterations: int
    method: str = "andronov_switching_point_map"


@dataclass(frozen=True)
class NonlinearityContinuationStep:
    """One step of an explicit source-to-target nonlinearity homotopy."""

    lambda_value: float
    x_in: np.ndarray
    x_out: np.ndarray
    trajectory: np.ndarray
    status: str
    provenance: Mapping[str, Any]


class _ExactConstantInputFlow:
    """Exact flow of ``x' = A x + b u`` for repeated scalar event calls."""

    def __init__(self, matrix: np.ndarray, input_vector: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=float)
        self.input_vector = np.asarray(input_vector, dtype=float)
        if abs(float(np.linalg.det(self.matrix))) <= 1.0e-14:
            raise ValueError("the switching-map implementation requires an invertible Lur'e matrix.")
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.matrix)
        if np.linalg.cond(self.eigenvectors) > 1.0e10:
            raise ValueError("the Lur'e matrix eigenbasis is too ill-conditioned for exact switching flow.")
        self.inverse_eigenvectors = np.linalg.inv(self.eigenvectors)

    def evaluate(self, state: np.ndarray, input_value: float, time_value: float) -> np.ndarray:
        equilibrium = -np.linalg.solve(self.matrix, self.input_vector * float(input_value))
        modal = self.inverse_eigenvectors @ (np.asarray(state, dtype=float) - equilibrium)
        evolved = equilibrium + self.eigenvectors @ (
            np.exp(self.eigenvalues * float(time_value)) * modal
        )
        return np.asarray(np.real_if_close(evolved, tol=1000), dtype=float)


def _project_to_switching_section(state: np.ndarray, output_vector: np.ndarray) -> np.ndarray:
    cvec = np.asarray(output_vector, dtype=float)
    value = float(cvec @ state)
    return np.asarray(state, dtype=float) - cvec * (value / float(cvec @ cvec))


def _select_departure_side(lure: LureSystem, state: np.ndarray) -> float:
    matrix = np.asarray(lure.matrix, dtype=float)
    bvec = np.asarray(lure.input_vector, dtype=float)
    cvec = np.asarray(lure.output_vector, dtype=float)
    base = float(cvec @ (matrix @ state))
    feedback = float(cvec @ bvec)
    candidates = [
        side
        for side in (-1.0, 1.0)
        if side * (base + feedback * side) > 1.0e-12
    ]
    if len(candidates) == 1:
        return float(candidates[0])
    if len(candidates) == 2 and abs(base) > 1.0e-12:
        return float(np.sign(base))
    raise RuntimeError("switching-section state has no unique transverse departure side.")


def _next_switching_crossing(
    lure: LureSystem,
    flow: _ExactConstantInputFlow,
    state: np.ndarray,
    *,
    bracket_step: float,
    max_crossing_time: float,
    root_tolerance: float,
) -> tuple[float, np.ndarray]:
    cvec = np.asarray(lure.output_vector, dtype=float)
    section = _project_to_switching_section(np.asarray(state, dtype=float), cvec)
    side = _select_departure_side(lure, section)
    previous_time = max(1.0e-8, 1.0e-4 * float(bracket_step))
    previous_value = float(cvec @ flow.evaluate(section, side, previous_time))
    grid = np.arange(
        previous_time + float(bracket_step),
        float(max_crossing_time) + float(bracket_step),
        float(bracket_step),
    )
    for current_time in grid:
        current_value = float(cvec @ flow.evaluate(section, side, float(current_time)))
        if previous_value * current_value < 0.0:
            crossing_time = float(
                brentq(
                    lambda value: float(cvec @ flow.evaluate(section, side, value)),
                    previous_time,
                    float(current_time),
                    xtol=float(root_tolerance),
                )
            )
            crossing_state = _project_to_switching_section(
                flow.evaluate(section, side, crossing_time), cvec
            )
            return crossing_time, crossing_state
        previous_time = float(current_time)
        previous_value = current_value
    raise RuntimeError("no transverse return to the switching section was found.")


def find_sign_switching_cycle_seed(
    system: ChaoticSystem | LureSystem,
    initial_section_state: Sequence[float] | np.ndarray,
    *,
    max_crossings: int = 400,
    max_return_period: int = 8,
    convergence_window: int = 4,
    convergence_tolerance: float = 1.0e-8,
    bracket_step: float = 0.02,
    max_crossing_time: float = 20.0,
    root_tolerance: float = 1.0e-12,
) -> SwitchingMapSeed:
    """Find a stable cycle of the ``sign(c^T x)`` precursor by point mapping.

    The supplied state is only a generic starting point on the switching
    section.  The returned cycle state is recomputed from ``(A,b,c)`` at every
    run and is not a stored attractor seed.
    """

    lure = require_lure(system)
    initial = np.asarray(initial_section_state, dtype=float)
    if initial.shape != (lure.dimension,):
        raise ValueError(f"initial_section_state must have shape ({lure.dimension},).")
    if int(max_crossings) < 2 * int(max_return_period):
        raise ValueError("max_crossings is too small for the requested return-period search.")
    if bracket_step <= 0.0 or max_crossing_time <= bracket_step:
        raise ValueError("expected 0 < bracket_step < max_crossing_time.")
    if convergence_window < 2:
        raise ValueError("convergence_window must be at least 2.")
    if convergence_tolerance <= 0.0:
        raise ValueError("convergence_tolerance must be positive.")

    initial = _project_to_switching_section(initial, lure.output_vector)
    flow = _ExactConstantInputFlow(lure.matrix, lure.input_vector)
    crossings: list[np.ndarray] = []
    crossing_times: list[float] = []
    current = initial.copy()
    detected_period: int | None = None
    detected_error = float("inf")
    for _iteration in range(1, int(max_crossings) + 1):
        interval, current = _next_switching_crossing(
            lure,
            flow,
            current,
            bracket_step=float(bracket_step),
            max_crossing_time=float(max_crossing_time),
            root_tolerance=float(root_tolerance),
        )
        crossings.append(current.copy())
        crossing_times.append(float(interval))
        for period in range(1, int(max_return_period) + 1):
            window = int(convergence_window) * period
            if len(crossings) < window + period:
                continue
            recent = np.asarray(crossings[-window:], dtype=float)
            prior = np.asarray(crossings[-window - period : -period], dtype=float)
            error = float(np.max(np.linalg.norm(recent - prior, axis=1)))
            if error <= float(convergence_tolerance):
                detected_period = period
                detected_error = error
                break
        if detected_period is not None:
            break
    if detected_period is None:
        raise RuntimeError("the switching point map did not converge to a periodic return.")
    crossing_array = np.asarray(crossings, dtype=float)
    time_array = np.asarray(crossing_times, dtype=float)
    return SwitchingMapSeed(
        seed=crossing_array[-1].copy(),
        initial_section_state=initial.copy(),
        crossing_states=crossing_array,
        crossing_times=time_array,
        return_period=int(detected_period),
        convergence_error=float(detected_error),
        iterations=len(crossings),
    )


def integer_lure_nonlinearity_homotopy_rhs(
    system: ChaoticSystem | LureSystem,
    source_nonlinearity: ScalarNonlinearity,
    lambda_value: float,
):
    """Return ``A x + b[(1-lambda)phi0 + lambda psi](c^T x)``."""

    lure = require_lure(system)
    lam = float(lambda_value)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda_value must lie in [0, 1].")
    matrix = np.asarray(lure.matrix, dtype=float)
    bvec = np.asarray(lure.input_vector, dtype=float)
    cvec = np.asarray(lure.output_vector, dtype=float)

    def rhs(state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        sigma = float(cvec @ x)
        source = float(source_nonlinearity(sigma))
        target = float(lure.nonlinearity(sigma))
        return matrix @ x + bvec * ((1.0 - lam) * source + lam * target)

    return rhs


def continue_integer_lure_nonlinearity(
    system: ChaoticSystem | LureSystem,
    seed: SwitchingMapSeed | Sequence[float] | np.ndarray,
    source_nonlinearity: ScalarNonlinearity,
    *,
    plan: ContinuationPlan,
    t_transient: float = 80.0,
    t_keep: float = 20.0,
    h: float = 0.01,
    div_threshold: float | None = None,
) -> list[NonlinearityContinuationStep]:
    """Continue a switching-cycle state to the registered smooth target."""

    lure = require_lure(system)
    errors = plan.validate()
    if errors:
        raise ValueError("; ".join(errors))
    x_in = np.asarray(seed.seed if isinstance(seed, SwitchingMapSeed) else seed, dtype=float)
    if x_in.shape != (lure.dimension,):
        raise ValueError(f"seed must have shape ({lure.dimension},).")
    provenance = {
        "public_parameter": "lambda",
        "source_nonlinearity": getattr(source_nonlinearity, "__name__", "callable"),
        "target_nonlinearity": lure.name,
        "mapping": dict(plan.mapping),
    }
    steps: list[NonlinearityContinuationStep] = []
    for lambda_value in plan.lambda_values:
        rhs = integer_lure_nonlinearity_homotopy_rhs(
            lure, source_nonlinearity, float(lambda_value)
        )
        transient, transient_status = efork_q1_integrate(
            rhs,
            x_in,
            t_final=float(t_transient),
            h=float(h),
            div_threshold=div_threshold,
        )
        if transient_status != "ok":
            steps.append(
                NonlinearityContinuationStep(
                    lambda_value=float(lambda_value),
                    x_in=x_in.copy(),
                    x_out=transient[-1, 1:].copy(),
                    trajectory=transient,
                    status=transient_status,
                    provenance=provenance,
                )
            )
            break
        kept, kept_status = efork_q1_integrate(
            rhs,
            transient[-1, 1:],
            t_final=float(t_keep),
            h=float(h),
            div_threshold=div_threshold,
        )
        x_out = kept[-1, 1:].copy()
        steps.append(
            NonlinearityContinuationStep(
                lambda_value=float(lambda_value),
                x_in=x_in.copy(),
                x_out=x_out,
                trajectory=kept,
                status=kept_status,
                provenance=provenance,
            )
        )
        if kept_status != "ok":
            break
        x_in = x_out
    return steps


__all__ = [
    "NonlinearityContinuationStep",
    "SwitchingMapSeed",
    "continue_integer_lure_nonlinearity",
    "find_sign_switching_cycle_seed",
    "integer_lure_nonlinearity_homotopy_rhs",
    "sign_nonlinearity",
]
