r"""Tempered-Caputo ABM solver through exponential conjugation.

Stability: experimental

For ``0 < alpha < 1`` and ``lambda >= 0``, HAFO uses the shifted but
equivalent conjugation

.. math::

   {}^{C}D_{a+}^{\alpha,\lambda}x(t)
   =e^{-\lambda(t-a)}{}^{C}D_{a+}^{\alpha}
     \left[e^{\lambda(t-a)}x(t)\right].

Thus ``v(t)=exp(lambda*(t-a))*x(t)`` satisfies an ordinary Caputo problem
whose right-hand side is

.. math::

   g(t,v)=e^{\lambda(t-a)}
          f\!\left(t,e^{-\lambda(t-a)}v\right).

HAFO applies the algebraically equivalent factors
``exp(-lambda*(t_n-t_j))`` directly to the ABM/PECE history.  Both the C and
Python kernels therefore store the physical state rather than the potentially
huge transformed state, and physical divergence can stop the recurrence at the
first accepted crossing.  This is not the Jacobi predictor--corrector algorithm
proposed by Li, Deng, and Zhao.  ``lambda=0`` is an exact software reduction to
the canonical Caputo lane.

References
----------
C. Li, W. Deng, L. Zhao, "Well-posedness and numerical algorithm for the
tempered fractional differential equations", DCDS-B 24 (2019), 1989--2015,
https://doi.org/10.3934/dcdsb.2019026.

F. Sabzikar, M. M. Meerschaert, J. Chen, "Tempered fractional calculus",
Journal of Computational Physics 293 (2015), 14--28,
https://doi.org/10.1016/j.jcp.2014.04.024.

K. Diethelm, N. J. Ford, A. D. Freed, "Detailed Error Analysis for a
Fractional Adams Method", Numerical Algorithms 36 (2004),
https://doi.org/10.1023/B:NUMA.0000027736.85078.be.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import operator
import sys
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from scipy.special import gamma

from .._rhs import bind_rhs
from .._time_grid import checked_array_capacity, exact_fixed_step_count
from ..integrations.fractional_c import GeneralFractionalCBackend, fractional_integrate


TEMPERED_CAPUTO_ABM_REFERENCES = (
    "https://doi.org/10.3934/dcdsb.2019026",
    "https://doi.org/10.1016/j.jcp.2014.04.024",
    "https://doi.org/10.1023/B:NUMA.0000027736.85078.be",
)


class _TemperedNativeBackendError(RuntimeError):
    """Internal marker for infrastructure failures, never RHS failures."""


@dataclass(frozen=True, slots=True)
class TemperedCaputoSimulationResult:
    """Finite physical trajectory plus the transformed Caputo trajectory."""

    times: np.ndarray
    states: np.ndarray
    transformed_states: np.ndarray | None
    order: float
    tempering: float
    lower_terminal: float
    requested_upper_terminal: float
    actual_upper_terminal: float
    step: float
    n_steps_requested: int
    method: str
    backend: str
    status: str
    memory_policy: str
    history_window: int | None
    grid_coordinate: str
    solver_info: Mapping[str, Any]
    references: tuple[str, ...] = TEMPERED_CAPUTO_ABM_REFERENCES
    scope: str = "finite_numerical_trajectory_only"

    @property
    def trajectory(self) -> np.ndarray:
        """Return conventional physical-time/state columns."""

        return np.column_stack((self.times, self.states))


def _real_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise TypeError(f"{name} must be a real scalar, not Boolean or complex.")
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


def _python_tempered_abm(
    evaluate_rhs: Callable[[float, np.ndarray], np.ndarray],
    state: np.ndarray,
    *,
    order: float,
    tempering: float,
    step: float,
    n_steps: int,
    memory_policy: str,
    history_window: int | None,
    divergence_norm: float | None,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    """Reference ABM with exponentially damped history in physical state."""

    dimension = state.size
    times = np.zeros(n_steps + 1, dtype=np.float64)
    if n_steps:
        times[1:] = np.cumsum(
            np.full(n_steps, step, dtype=np.float64),
            dtype=np.float64,
        )
    states = np.zeros((n_steps + 1, dimension), dtype=np.float64)
    derivatives = np.zeros_like(states)
    states[0] = state

    initial_norm = float(np.hypot.reduce(np.abs(state)))
    if divergence_norm is not None and initial_norm > divergence_norm:
        returned_times = times[:1]
        returned_states = states[:1]
        return returned_times, returned_states, "diverged", {
            "used_c_backend": False,
            "rhs_source": "python_native",
            "status_code": 1,
            "n_steps": 0,
            "n_steps_completed": 0,
            "n_samples": 1,
            "n_samples_returned": 1,
        }

    derivatives[0] = evaluate_rhs(0.0, state)
    powers = np.arange(n_steps + 3, dtype=np.float64)
    pow_q = powers**order
    pow_q1 = powers ** (order + 1.0)
    predictor_scale = step**order / float(gamma(order + 1.0))
    corrector_scale = step**order / float(gamma(order + 2.0))
    status = "ok"
    last_index = 0

    for current in range(n_steps):
        if memory_policy == "finite_window":
            assert history_window is not None
            start = max(0, current - history_window + 1)
        else:
            start = 0
        indices = np.arange(start, current + 1, dtype=np.int64)
        lags = (current + 1 - indices).astype(np.float64)
        raw_predictor_weights = (
            pow_q[current + 1 - indices] - pow_q[current - indices]
        )
        if tempering == 0.0:
            damping = np.ones_like(lags)
            predictor_weights = raw_predictor_weights
            anchor_damping = 1.0
        else:
            damping = np.exp(-tempering * lags * step)
            predictor_weights = raw_predictor_weights * damping
            anchor_damping = float(
                np.exp(-tempering * (current + 1 - start) * step)
            )
        with np.errstate(over="ignore", invalid="ignore"):
            predictor = (
                anchor_damping * states[start]
                + predictor_scale
                * (predictor_weights @ derivatives[start : current + 1])
            )
        if not np.all(np.isfinite(predictor)):
            status = "nonfinite_solution"
            break

        next_time = float(times[current + 1])
        predicted_derivative = evaluate_rhs(next_time, predictor)
        relative_count = current - start
        anchor_weight = (
            pow_q1[relative_count]
            - (float(relative_count) - order) * pow_q[relative_count + 1]
        )
        if relative_count > 0:
            middle_indices = np.arange(start + 1, current + 1, dtype=np.int64)
            middle_lags = current - middle_indices
            raw_middle_weights = (
                pow_q1[middle_lags + 2]
                + pow_q1[middle_lags]
                - 2.0 * pow_q1[middle_lags + 1]
            )
            if tempering == 0.0:
                corrector_weights = np.concatenate(
                    (np.array([anchor_weight]), raw_middle_weights)
                )
                history_sum = corrector_weights @ derivatives[start : current + 1]
            else:
                middle_weights = raw_middle_weights * np.exp(
                    -tempering * (current + 1 - middle_indices) * step
                )
                history_sum = (
                    anchor_weight * anchor_damping * derivatives[start]
                    + middle_weights @ derivatives[start + 1 : current + 1]
                )
        else:
            history_sum = anchor_weight * anchor_damping * derivatives[start]
        with np.errstate(over="ignore", invalid="ignore"):
            corrected = (
                anchor_damping * states[start]
                + corrector_scale * (history_sum + predicted_derivative)
            )
        states[current + 1] = corrected
        last_index = current + 1

        if not np.all(np.isfinite(corrected)):
            status = "nonfinite_solution"
            break
        physical_norm = float(np.hypot.reduce(np.abs(corrected)))
        if not np.isfinite(physical_norm):
            status = "nonfinite_solution"
            break
        if divergence_norm is not None and physical_norm > divergence_norm:
            status = "diverged"
            break
        derivatives[current + 1] = evaluate_rhs(next_time, corrected)

    returned_times = times[: last_index + 1]
    returned_states = states[: last_index + 1]
    completed = len(returned_times) - 1
    status_code = {"ok": 0, "diverged": 1, "nonfinite_solution": 2}[status]
    return returned_times, returned_states, status, {
        "used_c_backend": False,
        "rhs_source": "python_native",
        "status_code": status_code,
        "n_steps": completed,
        "n_steps_completed": completed,
        "n_samples": len(returned_times),
        "n_samples_returned": len(returned_times),
    }


def _native_tempered_abm(
    evaluate_rhs: Callable[[float, np.ndarray], np.ndarray],
    state: np.ndarray,
    *,
    order: float,
    tempering: float,
    step: float,
    n_steps: int,
    memory_policy: str,
    history_window: int | None,
    divergence_norm: float | None,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    """Call the direct physical-coordinate tempered ABM C kernel."""

    try:
        backend = GeneralFractionalCBackend.get_instance()
        function = backend.lib.integrate_tempered_caputo_abm_c
        vector = np.ctypeslib.ndpointer(
            dtype=np.float64,
            ndim=1,
            flags="C_CONTIGUOUS",
        )
        function.argtypes = [
            backend.RHS_CALLBACK,
            ctypes.c_void_p,
            ctypes.c_int,
            vector,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            vector,
            vector,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        function.restype = ctypes.c_int
    except Exception as exc:
        raise _TemperedNativeBackendError(
            f"The native tempered-Caputo backend is unavailable: {exc}"
        ) from exc

    dimension = state.size
    callback_failures: list[BaseException] = []

    def callback(
        local_time: float,
        state_pointer: "ctypes.POINTER(ctypes.c_double)",
        derivative_pointer: "ctypes.POINTER(ctypes.c_double)",
        callback_dimension: int,
        parameters_pointer: int,
    ) -> None:
        del parameters_pointer
        output = np.ctypeslib.as_array(
            derivative_pointer,
            shape=(callback_dimension,),
        )
        if callback_failures:
            output.fill(np.nan)
            return
        try:
            local_state = np.array(
                np.ctypeslib.as_array(
                    state_pointer,
                    shape=(callback_dimension,),
                ),
                dtype=np.float64,
                copy=True,
            )
            output[:] = evaluate_rhs(float(local_time), local_state)
        except BaseException as exc:
            callback_failures.append(exc)
            output.fill(np.nan)

    c_callback = backend.RHS_CALLBACK(callback)
    output_times = np.zeros(n_steps + 1, dtype=np.float64)
    output_states = np.zeros((n_steps + 1) * dimension, dtype=np.float64)
    output_samples = ctypes.c_int(0)
    status_code = ctypes.c_int(0)
    effective_divergence = (
        sys.float_info.max if divergence_norm is None else float(divergence_norm)
    )
    try:
        return_code = function(
            c_callback,
            ctypes.c_void_p(None),
            ctypes.c_int(dimension),
            state,
            ctypes.c_double(order),
            ctypes.c_double(tempering),
            ctypes.c_double(step),
            ctypes.c_int(n_steps),
            ctypes.c_int(0 if memory_policy == "full_history" else 1),
            ctypes.c_int(0 if history_window is None else history_window),
            ctypes.c_double(effective_divergence),
            output_times,
            output_states,
            ctypes.c_size_t(output_times.size),
            ctypes.c_size_t(output_states.size),
            ctypes.byref(output_samples),
            ctypes.byref(status_code),
        )
    except Exception as exc:
        if callback_failures:
            raise callback_failures[0]
        raise _TemperedNativeBackendError(
            f"The native tempered-Caputo call failed: {exc}"
        ) from exc
    if callback_failures:
        raise callback_failures[0]
    if return_code < 0:
        raise _TemperedNativeBackendError(
            "integrate_tempered_caputo_abm_c returned "
            f"error code {return_code}."
        )

    sample_count = int(output_samples.value)
    returned_times = output_times[:sample_count].copy()
    returned_states = output_states[: sample_count * dimension].reshape(
        sample_count,
        dimension,
    ).copy()
    status = {
        0: "ok",
        1: "diverged",
        2: "nonfinite_solution",
    }.get(status_code.value, f"unknown_{status_code.value}")
    completed = max(0, sample_count - 1)
    return returned_times, returned_states, status, {
        "used_c_backend": True,
        "rhs_source": "python_callback_wrapped",
        "status_code": int(status_code.value),
        "n_steps": completed,
        "n_steps_completed": completed,
        "n_samples": sample_count,
        "n_samples_returned": sample_count,
    }


def integrate_tempered_caputo_abm(
    rhs: Callable,
    initial_state: Any,
    order: float,
    parameters: Any = None,
    *,
    tempering: float,
    lower_terminal: float,
    upper_terminal: float,
    step: float,
    memory_policy: str = "full_history",
    history_window: int | None = None,
    use_acceleration: bool = True,
    allow_python_fallback: bool = True,
    divergence_norm: float | None = 120.0,
) -> TemperedCaputoSimulationResult:
    """Integrate a commensurate tempered-Caputo IVP via Caputo ABM/PECE.

    The fixed ``step`` is measured in physical time.  ``full_history`` uses
    the conventional hereditary sum.  ``finite_window`` is an explicit model
    approximation and requires ``history_window >= 2``.

    Divergence is assessed on every accepted physical state before evaluating
    the next history derivative.  ``finite_window`` uses a sliding restart with
    the exact tempered anchor ``exp(-lambda*(t-t_s))*x(t_s)``; it is not the
    fixed-terminal full-history equation.
    """

    if not callable(rhs):
        raise TypeError("rhs must be callable.")
    state = _initial_state(initial_state)
    if state.size > np.iinfo(np.int32).max:
        raise ValueError("initial_state dimension exceeds the native INT32 range.")
    alpha = _real_scalar(order, name="order")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("order must lie strictly in (0, 1) for this ABM solver.")
    lam = _real_scalar(tempering, name="tempering")
    if lam < 0.0:
        raise ValueError("tempering must be nonnegative.")
    terminal = _real_scalar(lower_terminal, name="lower_terminal")
    upper = _real_scalar(upper_terminal, name="upper_terminal")
    if upper <= terminal:
        raise ValueError("upper_terminal must be greater than lower_terminal.")
    normalized_step = _real_scalar(step, name="step")
    if normalized_step <= 0.0:
        raise ValueError("step must be positive.")
    if not isinstance(use_acceleration, (bool, np.bool_)):
        raise TypeError("use_acceleration must be Boolean.")
    if not isinstance(allow_python_fallback, (bool, np.bool_)):
        raise TypeError("allow_python_fallback must be Boolean.")

    policy = str(memory_policy).strip().lower()
    if policy not in {"full_history", "finite_window"}:
        raise ValueError("memory_policy must be 'full_history' or 'finite_window'.")
    if policy == "finite_window":
        if isinstance(history_window, (bool, np.bool_)):
            raise ValueError("finite_window requires history_window >= 2 samples.")
        try:
            normalized_window = operator.index(history_window)
        except TypeError as exc:
            raise ValueError(
                "finite_window requires history_window >= 2 samples."
            ) from exc
        if normalized_window < 2:
            raise ValueError("finite_window requires history_window >= 2 samples.")
        if normalized_window > np.iinfo(np.int32).max:
            raise ValueError("history_window must not exceed INT32_MAX.")
        normalized_window = int(normalized_window)
    else:
        if history_window is not None:
            raise ValueError(
                "history_window is only valid with memory_policy='finite_window'."
            )
        normalized_window = None

    if divergence_norm is None:
        physical_divergence = None
    else:
        physical_divergence = _real_scalar(divergence_norm, name="divergence_norm")
        if physical_divergence <= 0.0:
            raise ValueError("divergence_norm must be positive or None.")

    duration = upper - terminal
    native_step_limit = int(np.iinfo(np.int32).max) - 3
    n_steps = exact_fixed_step_count(
        normalized_step,
        duration,
        caller="integrate_tempered_caputo_abm",
        max_steps=native_step_limit,
    )
    if n_steps < 1:
        raise ValueError(
            "upper_terminal-lower_terminal must contain an integer number "
            "of step increments."
        )
    checked_array_capacity(
        (n_steps + 1,),
        np.float64,
        caller="integrate_tempered_caputo_abm output times",
    )
    checked_array_capacity(
        (n_steps + 1, state.size),
        np.float64,
        caller="integrate_tempered_caputo_abm output states",
    )
    maximum_exponent = lam * duration
    bound_rhs = bind_rhs(rhs, parameters)

    def evaluate_physical_rhs(
        local_time: float,
        physical_state: np.ndarray,
    ) -> np.ndarray:
        raw = bound_rhs(
            terminal + local_time,
            np.array(physical_state, dtype=np.float64, copy=True),
        )
        if np.iscomplexobj(raw):
            raise TypeError("rhs must return real-valued derivatives.")
        derivative = np.asarray(raw, dtype=np.float64).reshape(-1)
        if derivative.shape != state.shape:
            raise ValueError("rhs output shape must match initial_state.")
        if not np.all(np.isfinite(derivative)):
            raise ValueError("rhs must return only finite derivatives.")
        return np.ascontiguousarray(derivative)

    c_backend_error: str | None = None
    lambda_zero_canonical_lane = lam == 0.0
    if lambda_zero_canonical_lane:
        local_times, states, status, info = fractional_integrate(
            rhs=evaluate_physical_rhs,
            x0=state,
            q=alpha,
            h=normalized_step,
            t_final=float(np.nextafter(normalized_step * n_steps, -np.inf)),
            method="abm",
            memory_mode="full" if policy == "full_history" else "window",
            memory_window_length=normalized_window,
            use_c_backend=bool(use_acceleration),
            divergence_norm=(
                float("inf")
                if physical_divergence is None
                else physical_divergence
            ),
            return_history=True,
            allow_python_fallback=bool(allow_python_fallback),
            early_stop_config={"enabled": False},
        )
    elif use_acceleration:
        try:
            local_times, states, status, info = _native_tempered_abm(
                evaluate_physical_rhs,
                state,
                order=alpha,
                tempering=lam,
                step=normalized_step,
                n_steps=n_steps,
                memory_policy=policy,
                history_window=normalized_window,
                divergence_norm=physical_divergence,
            )
        except _TemperedNativeBackendError as exc:
            if not allow_python_fallback:
                raise RuntimeError(
                    "C backend failed and allow_python_fallback=False. "
                    f"Original error: {exc}"
                ) from exc
            c_backend_error = str(exc)
            local_times, states, status, info = _python_tempered_abm(
                evaluate_physical_rhs,
                state,
                order=alpha,
                tempering=lam,
                step=normalized_step,
                n_steps=n_steps,
                memory_policy=policy,
                history_window=normalized_window,
                divergence_norm=physical_divergence,
            )
    else:
        local_times, states, status, info = _python_tempered_abm(
            evaluate_physical_rhs,
            state,
            order=alpha,
            tempering=lam,
            step=normalized_step,
            n_steps=n_steps,
            memory_policy=policy,
            history_window=normalized_window,
            divergence_norm=physical_divergence,
        )

    local_times = np.asarray(local_times, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    if not np.all(np.isfinite(states)):
        status = "nonfinite_solution"
    times = terminal + local_times

    used_c = bool(info.get("used_c_backend", False))
    if lambda_zero_canonical_lane:
        backend = (
            "native_c_caputo_abm_canonical"
            if used_c
            else "python_numpy_caputo_abm_canonical"
        )
    else:
        backend = (
            "native_c_tempered_abm_physical"
            if used_c
            else "python_numpy_tempered_abm_physical"
        )
    logarithmic_limit = float(np.log(np.finfo(np.float64).max))
    transformed_states: np.ndarray | None
    if maximum_exponent < logarithmic_limit - 2.0:
        with np.errstate(over="ignore", invalid="ignore"):
            candidate_transformed = states * np.exp(lam * local_times)[:, None]
        transformed_states = (
            np.asarray(candidate_transformed, dtype=np.float64)
            if np.all(np.isfinite(candidate_transformed))
            else None
        )
    else:
        transformed_states = None
    detached_info = {
        str(key): value
        for key, value in info.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    if c_backend_error is not None:
        detached_info["c_backend_error"] = c_backend_error
    detached_info.update(
        {
            "underlying_status": str(status),
            "underlying_method": "caputo_abm_pece_damped_history",
            "conjugation": "v=exp(tempering*(t-a))*x",
            "direct_damped_history": True,
            "maximum_exponent": maximum_exponent,
            "physical_divergence_norm": physical_divergence,
            "divergence_coordinate": "physical_state",
            "transformed_divergence_norm": None,
            "n_steps_requested": n_steps,
            "n_steps_completed": max(0, len(times) - 1),
            "n_steps": max(0, len(times) - 1),
            "n_samples": len(times),
            "n_samples_returned": len(times),
            "lambda_zero_reduction": lam == 0.0,
            "lambda_zero_lane": (
                "canonical_fractional_integrate"
                if lambda_zero_canonical_lane
                else None
            ),
            "memory_mode": "full" if policy == "full_history" else "window",
            "history_window": normalized_window,
            "truncated_memory": policy == "finite_window",
            "window_semantics": (
                "sliding_restart_in_physical_state_with_tempered_anchor"
                if policy == "finite_window"
                else None
            ),
            "effective_window_duration": (
                None
                if normalized_window is None
                else normalized_window * normalized_step
            ),
            "transformed_states_stored": transformed_states is not None,
        }
    )
    return TemperedCaputoSimulationResult(
        times=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        transformed_states=transformed_states,
        order=alpha,
        tempering=lam,
        lower_terminal=terminal,
        requested_upper_terminal=upper,
        actual_upper_terminal=float(times[-1]),
        step=normalized_step,
        n_steps_requested=n_steps,
        method="tempered_caputo_abm_pece_transform",
        backend=backend,
        status=status,
        memory_policy=policy,
        history_window=normalized_window,
        grid_coordinate="physical_time",
        solver_info=MappingProxyType(detached_info),
    )


__all__ = [
    "TEMPERED_CAPUTO_ABM_REFERENCES",
    "TemperedCaputoSimulationResult",
    "integrate_tempered_caputo_abm",
]
