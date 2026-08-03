"""Caputo--Hadamard ABM/PECE solver on a uniform logarithmic grid.

Stability: experimental

For ``u = log(t/a)`` and ``y(u) = x(a*exp(u))``, the Caputo--Hadamard IVP

``CH_D_a^q x(t) = f(t, x(t)),  x(a) = x0``

is transformed into the ordinary Caputo IVP

``C_D_0^q y(u) = f(a*exp(u), y(u)),  y(0) = x0``.

This module reuses HAFO's canonical full-history Adams--Bashforth--Moulton
PECE implementation in ``u``.  It does not reinterpret a physical time step
as a logarithmic step, and it returns both coordinates.  Only uniform
logarithmic grids and commensurate ``0 < q < 1`` are implemented here; graded
meshes require different quadrature weights and remain a separate method.

References
----------
K. Diethelm, N. J. Ford, and A. D. Freed, "Detailed Error Analysis for a
Fractional Adams Method", Numerical Algorithms 36 (2004),
https://doi.org/10.1023/B:NUMA.0000027736.85078.be.
X. Zheng, "Logarithmic transformation between (variable-order) Caputo and
Caputo--Hadamard fractional problems and applications", Applied Mathematics
Letters 121 (2021), https://doi.org/10.1016/j.aml.2021.107366.
C. W. H. Green, Y. Liu, and Y. Yan, "Numerical Methods for Caputo--Hadamard
Fractional Differential Equations with Graded and Non-Uniform Meshes",
Mathematics 9 (2021), https://doi.org/10.3390/math9212728.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from .._rhs import bind_rhs
from ..integrations.fractional_c import fractional_integrate
from ._log_grid import (
    physical_times_from_log,
    stable_log_ratio,
    uniform_step_grid_metrics,
)
from .hadamard import CAPUTO_HADAMARD_INITIAL_CONDITION


_REFERENCES = (
    "https://doi.org/10.1186/1687-1847-2012-142",
    "https://doi.org/10.1023/B:NUMA.0000027736.85078.be",
    "https://doi.org/10.1016/j.aml.2021.107366",
    "https://doi.org/10.3390/math9212728",
)


@dataclass(frozen=True, slots=True)
class CaputoHadamardSimulationResult:
    """Finite trajectory and its logarithmic-grid numerical contract.

    In ``solver_info``, ``n_steps`` and ``n_steps_completed`` count completed
    grid increments. ``n_samples`` counts stored points including the initial
    point (and any supplied prehistory in the underlying generic engine), while
    ``n_samples_returned`` counts the points exposed to this caller.
    """

    times: np.ndarray
    log_times: np.ndarray
    states: np.ndarray
    order: float
    method: str
    backend: str
    status: str
    lower_terminal: float
    requested_upper_terminal: float
    actual_upper_terminal: float
    log_step: float
    n_steps_requested: int
    memory_policy: str
    initial_condition_semantics: str
    grid_coordinate: str
    rhs_time_coordinate: str
    physical_sampling_uniform: bool
    solver_info: Mapping[str, Any]
    references: tuple[str, ...]
    scope: str = "finite_numerical_trajectory_only"

    @property
    def trajectory(self) -> np.ndarray:
        """Return conventional physical-time/state columns."""

        return np.column_stack((self.times, self.states))


def _real_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise TypeError(f"{name} must be a real number, not Boolean or complex.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _initial_state(value: Any) -> np.ndarray:
    if np.iscomplexobj(value):
        raise TypeError("initial_state must be real-valued.")
    state = np.asarray(value, dtype=np.float64).reshape(-1)
    if state.size < 1 or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain at least one finite value.")
    return np.ascontiguousarray(state)


def integrate_caputo_hadamard_abm(
    rhs: Callable,
    initial_state: Any,
    order: float,
    parameters: Any = None,
    *,
    lower_terminal: float,
    upper_terminal: float,
    log_step: float,
    initial_condition_semantics: str = CAPUTO_HADAMARD_INITIAL_CONDITION,
    use_acceleration: bool = True,
    allow_python_fallback: bool = True,
    divergence_norm: float | None = 120.0,
) -> CaputoHadamardSimulationResult:
    """Integrate a commensurate Caputo--Hadamard IVP with ABM/PECE.

    ``log_step`` is the uniform increment in ``u=log(t/a)``.  The ratio
    ``log(upper_terminal/lower_terminal)/log_step`` must be integral within a
    floating-point grid tolerance.  The RHS receives physical time ``t``, not
    logarithmic time ``u``.

    ``use_acceleration=True`` attempts the existing C ABM history kernel and
    falls back to the NumPy reference path.  A generic transformed RHS still
    crosses the Python callback boundary, which is recorded in ``backend``;
    this is not advertised as a pure-C vector-field implementation.
    """

    if not callable(rhs):
        raise TypeError("rhs must be callable.")
    bound_rhs = bind_rhs(rhs, parameters)
    state = _initial_state(initial_state)
    normalized_order = _real_scalar(order, name="order")
    if normalized_order <= 0.0 or normalized_order >= 1.0:
        raise ValueError("order must lie strictly in (0, 1) for this ABM solver.")
    terminal = _real_scalar(lower_terminal, name="lower_terminal")
    upper = _real_scalar(upper_terminal, name="upper_terminal")
    if terminal <= 0.0:
        raise ValueError("lower_terminal must be strictly positive.")
    if upper <= terminal:
        raise ValueError("upper_terminal must be greater than lower_terminal.")
    normalized_step = _real_scalar(log_step, name="log_step")
    if normalized_step <= 0.0:
        raise ValueError("log_step must be positive.")
    semantics = str(initial_condition_semantics).strip().lower()
    if semantics != CAPUTO_HADAMARD_INITIAL_CONDITION:
        raise ValueError(
            "Caputo--Hadamard ABM requires "
            f"initial_condition_semantics={CAPUTO_HADAMARD_INITIAL_CONDITION!r}."
        )
    if not isinstance(use_acceleration, (bool, np.bool_)):
        raise TypeError("use_acceleration must be Boolean.")
    if divergence_norm is None:
        normalized_divergence = np.inf
    else:
        normalized_divergence = _real_scalar(
            divergence_norm, name="divergence_norm"
        )
        if normalized_divergence <= 0.0:
            raise ValueError("divergence_norm must be positive or None.")

    log_duration = float(stable_log_ratio(upper, terminal))
    n_steps, residual, tolerance = uniform_step_grid_metrics(
        log_duration,
        normalized_step,
    )
    if n_steps < 1 or residual > tolerance:
        raise ValueError(
            "log(upper_terminal/lower_terminal) must contain an integer "
            "number of log_step increments."
        )

    def transformed_rhs(log_time: float, current_state: np.ndarray) -> np.ndarray:
        physical_time = float(physical_times_from_log(terminal, log_time))
        if not np.isfinite(physical_time):
            raise FloatingPointError("physical time overflowed under exp(log(t/a)).")
        raw_derivative = bound_rhs(physical_time, current_state)
        if np.iscomplexobj(raw_derivative):
            raise TypeError("rhs must return real-valued derivatives.")
        derivative = np.asarray(raw_derivative, dtype=np.float64).reshape(-1)
        if derivative.shape != state.shape:
            raise ValueError("rhs output shape must match initial_state.")
        if not np.all(np.isfinite(derivative)):
            raise ValueError("rhs must return only finite derivatives.")
        return derivative

    # Validate the callable before crossing a ctypes callback boundary.
    transformed_rhs(0.0, state.copy())
    integration_horizon = float(
        np.nextafter(n_steps * normalized_step, -np.inf)
    )
    log_times, states, status, info = fractional_integrate(
        rhs=transformed_rhs,
        x0=state,
        q=normalized_order,
        h=normalized_step,
        t_final=integration_horizon,
        method="abm",
        memory_mode="full",
        use_c_backend=bool(use_acceleration),
        divergence_norm=normalized_divergence,
        return_history=True,
        allow_python_fallback=bool(allow_python_fallback),
        early_stop_config={"enabled": False},
    )
    log_times = np.asarray(log_times, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    times = np.asarray(
        physical_times_from_log(terminal, log_times),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(times)) or (
        times.size > 1 and np.any(np.diff(times) <= 0.0)
    ):
        raise FloatingPointError(
            "The computed logarithmic grid does not map to finite increasing times."
        )
    used_c = bool(info.get("used_c_backend", False))
    backend = "c_abm_with_python_time_transform" if used_c else "python_numpy_abm"
    detached_info = {
        str(key): value
        for key, value in info.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    return CaputoHadamardSimulationResult(
        times=times,
        log_times=log_times,
        states=states,
        order=normalized_order,
        method="caputo_hadamard_abm_pece",
        backend=backend,
        status=status,
        lower_terminal=terminal,
        requested_upper_terminal=upper,
        actual_upper_terminal=float(times[-1]),
        log_step=normalized_step,
        n_steps_requested=n_steps,
        memory_policy="full_history",
        initial_condition_semantics=semantics,
        grid_coordinate="uniform_log_t_over_lower_terminal",
        rhs_time_coordinate="physical_time_t_equals_a_exp_u",
        physical_sampling_uniform=False,
        solver_info=MappingProxyType(detached_info),
        references=_REFERENCES,
    )


__all__ = [
    "CaputoHadamardSimulationResult",
    "integrate_caputo_hadamard_abm",
]
