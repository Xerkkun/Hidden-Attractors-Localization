r"""Variable-order Caputo type-III solver on a uniform physical-time grid.

Stability: experimental

The derivative is fixed to the third Tavares--Almeida--Torres convention

.. math::

   {}^{C,III}_{a}D_t^{\alpha(t)}x(t)
   =\frac{1}{\Gamma(1-\alpha(t))}
    \int_a^t(t-s)^{-\alpha(t)}x'(s)\,ds,

so the current value ``alpha(t_n)`` is used throughout the entire history at
output ``n``.  It is not interchangeable with type-I or type-II definitions.

HAFO discretizes the derivative with direct L1 weights and solves the implicit
state equation by a reported Picard PECE iteration.  The L1 formula is from the
published variable-order literature; the nonlinear ODE-system corrector and
its failure policy are an explicit HAFO adaptation.

References
----------
D. Tavares, R. Almeida, D. F. M. Torres, "Caputo derivatives of fractional
variable order: numerical approximations", CNSNS 35 (2016), 69--87,
https://doi.org/10.1016/j.cnsns.2015.10.027.

Z. W. Fang, H. W. Sun, H. Wang, "A fast method for variable-order Caputo
fractional derivative with applications to time-fractional diffusion
equations", CAMWA 80 (2020), 1443--1458,
https://doi.org/10.1016/j.camwa.2020.07.009.
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Signature, signature
import operator
from types import MappingProxyType
from typing import Any, Callable, Mapping
import warnings

import numpy as np
from numba import njit
from scipy.special import gamma

from .._rhs import bind_rhs


VARIABLE_ORDER_CAPUTO_TYPE3_REFERENCES = (
    "https://doi.org/10.1016/j.cnsns.2015.10.027",
    "https://doi.org/10.1016/j.camwa.2020.07.009",
)


class VariableOrderInitialCompatibilityWarning(UserWarning):
    """A requested smooth start conflicts with the endpoint equation."""


class VariableOrderCorrectorError(RuntimeError):
    """The implicit L1 Picard corrector did not meet its declared tolerance."""


@dataclass(frozen=True, slots=True)
class VariableOrderCaputoType3Result:
    """Finite type-III L1 trajectory and complete numerical diagnostics."""

    times: np.ndarray
    states: np.ndarray
    orders: np.ndarray
    corrector_iterations: np.ndarray
    corrector_residuals: np.ndarray
    lower_terminal: float
    requested_upper_terminal: float
    actual_upper_terminal: float
    step: float
    n_steps_requested: int
    method: str
    backend: str
    status: str
    memory_policy: str
    grid_coordinate: str
    order_function_name: str
    initial_regularity: str
    solver_info: Mapping[str, Any]
    references: tuple[str, ...] = VARIABLE_ORDER_CAPUTO_TYPE3_REFERENCES
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


def _strict_count(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return int(normalized)


def _vector_norm(values: np.ndarray) -> float:
    return float(np.hypot.reduce(np.abs(values)))


_DUMMY_TIME = object()
_DUMMY_STATE = object()
_DUMMY_PARAMETERS = object()


def _accepts_positional(signature_value: Signature, *arguments: object) -> bool:
    try:
        signature_value.bind(*arguments)
    except TypeError:
        return False
    return True


def _bind_order_function(
    order_function: Callable[..., Any],
    parameters: Any,
    reference_state: np.ndarray,
) -> Callable[[float], Any]:
    """Bind a prescribed time schedule without probing it by execution.

    HAFO accepts ``alpha(time)``, ``alpha(time, initial_state)`` and, when
    parameters are supplied, ``alpha(time, initial_state, parameters)``.  The
    state argument is always a detached copy of the *initial* state.  It exists
    for callback interoperability and must not be interpreted as a
    state-dependent fractional order.
    """

    if not callable(order_function):
        raise TypeError("order_function must be callable.")
    try:
        order_signature = signature(order_function)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "order_function must expose an inspectable positional signature; "
            "wrap opaque callables as alpha(time), alpha(time, initial_state), "
            "or alpha(time, initial_state, parameters)."
        ) from exc

    if parameters is not None and _accepts_positional(
        order_signature,
        _DUMMY_TIME,
        _DUMMY_STATE,
        _DUMMY_PARAMETERS,
    ):

        def bound(time: float) -> Any:
            return order_function(
                time,
                np.array(reference_state, dtype=np.float64, copy=True),
                parameters,
            )

        mode = "time_initial_state_parameters"
    elif _accepts_positional(order_signature, _DUMMY_TIME, _DUMMY_STATE):

        def bound(time: float) -> Any:
            return order_function(
                time,
                np.array(reference_state, dtype=np.float64, copy=True),
            )

        mode = "time_initial_state"
    elif _accepts_positional(order_signature, _DUMMY_TIME):

        def bound(time: float) -> Any:
            return order_function(time)

        mode = "time"
    else:
        expected = (
            "alpha(time), alpha(time, initial_state), or "
            "alpha(time, initial_state, parameters)"
            if parameters is not None
            else "alpha(time) or alpha(time, initial_state)"
        )
        raise TypeError(
            "order_function does not support a recognized signature; "
            f"expected {expected}."
        )

    setattr(bound, "__hafo_order_signature__", mode)
    return bound


def variable_order_l1_weight(order: float, lag: int) -> float:
    r"""Return ``(lag+1)^(1-order)-lag^(1-order)`` stably.

    The time-step and Gamma prefactor are deliberately not included.  For
    At moderate lags the expression is evaluated directly, preserving the
    conventional L1 recurrence bit-for-bit.  At large lags,
    ``expm1``/``log1p`` avoid cancellation when the order is close to one.
    """

    alpha = _real_scalar(order, name="order")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("order must lie strictly in (0, 1).")
    index = _strict_count(lag, name="lag", minimum=0)
    if index == 0:
        return 1.0
    exponent = 1.0 - alpha
    if index <= 1024:
        direct = float((index + 1.0) ** exponent - index**exponent)
        cancellation_scale = exponent * float(np.log1p(1.0 / float(index)))
        if direct > 0.0 and cancellation_scale >= 1.0e-8:
            return direct
    return float(
        np.exp(exponent * np.log(float(index)))
        * np.expm1(exponent * np.log1p(1.0 / float(index)))
    )


@njit(cache=True)
def _l1_weight_numba(order: float, lag: int) -> float:
    if lag == 0:
        return 1.0
    exponent = 1.0 - order
    if lag <= 1024:
        direct = (lag + 1.0) ** exponent - lag**exponent
        cancellation_scale = exponent * np.log1p(1.0 / float(lag))
        if direct > 0.0 and cancellation_scale >= 1.0e-8:
            return direct
    return np.exp(exponent * np.log(float(lag))) * np.expm1(
        exponent * np.log1p(1.0 / float(lag))
    )


@njit(cache=True)
def _history_sum_numba(
    states: np.ndarray,
    output_index: int,
    order: float,
) -> np.ndarray:
    dimension = states.shape[1]
    history = np.zeros(dimension, dtype=np.float64)
    for history_index in range(output_index - 1):
        lag = output_index - history_index - 1
        weight = _l1_weight_numba(order, lag)
        for component in range(dimension):
            history[component] += weight * (
                states[history_index + 1, component]
                - states[history_index, component]
            )
    return history


def _history_sum_python(
    states: np.ndarray,
    output_index: int,
    order: float,
) -> np.ndarray:
    history = np.zeros(states.shape[1], dtype=np.float64)
    for history_index in range(output_index - 1):
        lag = output_index - history_index - 1
        history += variable_order_l1_weight(order, lag) * (
            states[history_index + 1] - states[history_index]
        )
    return history


def integrate_variable_order_caputo_type3_l1(
    rhs: Callable,
    initial_state: Any,
    parameters: Any = None,
    *,
    step: float,
    n_steps: int,
    lower_terminal: float = 0.0,
    order_function: Callable[..., Any],
    order_function_name: str | None = None,
    declared_initial_order: float | None = None,
    corrector_atol: float = 1.0e-12,
    corrector_rtol: float = 1.0e-10,
    corrector_max_iterations: int = 50,
    on_nonconvergence: str = "raise",
    initial_regularity: str = "unknown",
    compatibility_tolerance: float = 1.0e-10,
    use_acceleration: bool = True,
    allow_python_fallback: bool = True,
    divergence_norm: float | None = 120.0,
) -> VariableOrderCaputoType3Result:
    """Solve a commensurate type-III variable-order IVP with implicit L1.

    ``order_function`` defines a prescribed physical-time schedule.  It may
    accept ``(time)``, ``(time, initial_state)`` or
    ``(time, initial_state, parameters)``; the state context is fixed and does
    not make the order state-dependent.  ``full_history`` is the sole
    implemented memory policy.  When ``initial_regularity='smooth'``, a
    nonzero endpoint RHS emits :class:`VariableOrderInitialCompatibilityWarning`;
    nonsmooth starts remain admissible but do not inherit the smooth L1 error
    rate.  ``weak`` is retained as an alias for ``nonsmooth``.
    """

    if not callable(rhs):
        raise TypeError("rhs must be callable.")
    if not callable(order_function):
        raise TypeError("order_function must be callable as a prescribed alpha(time).")
    if not isinstance(use_acceleration, (bool, np.bool_)):
        raise TypeError("use_acceleration must be Boolean.")
    if not isinstance(allow_python_fallback, (bool, np.bool_)):
        raise TypeError("allow_python_fallback must be Boolean.")

    if np.iscomplexobj(initial_state):
        raise TypeError("initial_state must be real-valued.")
    state0 = np.asarray(initial_state, dtype=np.float64).reshape(-1)
    if state0.size < 1 or not np.all(np.isfinite(state0)):
        raise ValueError("initial_state must contain at least one finite value.")
    state0 = np.ascontiguousarray(state0)

    normalized_step = _real_scalar(step, name="step")
    if normalized_step <= 0.0:
        raise ValueError("step must be positive.")
    count = _strict_count(n_steps, name="n_steps", minimum=1)
    terminal = _real_scalar(lower_terminal, name="lower_terminal")
    atol = _real_scalar(corrector_atol, name="corrector_atol")
    rtol = _real_scalar(corrector_rtol, name="corrector_rtol")
    if atol < 0.0:
        raise ValueError("corrector_atol must be nonnegative.")
    if rtol < 0.0:
        raise ValueError("corrector_rtol must be nonnegative.")
    if atol == 0.0 and rtol == 0.0:
        raise ValueError(
            "corrector_atol and corrector_rtol must not both be zero."
        )
    maximum_iterations = _strict_count(
        corrector_max_iterations,
        name="corrector_max_iterations",
        minimum=1,
    )
    nonconvergence = str(on_nonconvergence).strip().lower()
    if nonconvergence not in {"raise", "return"}:
        raise ValueError("on_nonconvergence must be 'raise' or 'return'.")
    regularity = str(initial_regularity).strip().lower()
    if regularity == "weak":
        regularity = "nonsmooth"
    if regularity not in {"unknown", "smooth", "nonsmooth"}:
        raise ValueError(
            "initial_regularity must be 'unknown', 'smooth', or 'nonsmooth' "
            "('weak' is an alias)."
        )
    compatibility_limit = _real_scalar(
        compatibility_tolerance,
        name="compatibility_tolerance",
    )
    if compatibility_limit < 0.0:
        raise ValueError("compatibility_tolerance must be nonnegative.")
    if divergence_norm is None:
        physical_divergence = None
    else:
        physical_divergence = _real_scalar(
            divergence_norm,
            name="divergence_norm",
        )
        if physical_divergence <= 0.0:
            raise ValueError("divergence_norm must be positive or None.")

    bound_rhs = bind_rhs(rhs, parameters)

    def evaluate_rhs(time: float, state: np.ndarray) -> np.ndarray:
        raw = bound_rhs(time, np.array(state, dtype=np.float64, copy=True))
        if np.iscomplexobj(raw):
            raise TypeError("rhs must return real-valued derivatives.")
        derivative = np.asarray(raw, dtype=np.float64).reshape(-1)
        if derivative.shape != state0.shape:
            raise ValueError("rhs output shape must match initial_state.")
        if not np.all(np.isfinite(derivative)):
            raise ValueError("rhs must return only finite derivatives.")
        return np.ascontiguousarray(derivative)

    initial_norm = _vector_norm(state0)
    initially_diverged = (
        physical_divergence is not None and initial_norm > physical_divergence
    )
    initial_derivative = (
        None if initially_diverged else evaluate_rhs(terminal, state0)
    )

    declared_name = (
        str(order_function_name).strip()
        if order_function_name is not None
        else str(
            getattr(
                order_function,
                "__qualname__",
                getattr(order_function, "__name__", "order_function"),
            )
        )
    )
    if not declared_name:
        raise ValueError("order_function_name must not be empty.")

    evaluate_order = _bind_order_function(order_function, parameters, state0)
    order_signature = str(
        getattr(evaluate_order, "__hafo_order_signature__", "unknown")
    )
    times = terminal + np.arange(count + 1, dtype=np.float64) * normalized_step
    orders = np.empty(count + 1, dtype=np.float64)
    for index, time in enumerate(times):
        raw_order = evaluate_order(float(time))
        alpha = _real_scalar(raw_order, name=f"order_function(t[{index}])")
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError(
                "order_function must return values strictly in (0, 1); "
                f"received {alpha} at t={time}."
            )
        orders[index] = alpha
    if declared_initial_order is None:
        normalized_declared_initial_order = None
    else:
        normalized_declared_initial_order = _real_scalar(
            declared_initial_order,
            name="declared_initial_order",
        )
        if not 0.0 < normalized_declared_initial_order < 1.0:
            raise ValueError("declared_initial_order must lie strictly in (0, 1).")
        order_tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0,
            abs(normalized_declared_initial_order),
            abs(float(orders[0])),
        )
        if abs(float(orders[0]) - normalized_declared_initial_order) > order_tolerance:
            raise ValueError(
                "declared_initial_order does not match order_function at the "
                "lower terminal."
            )

    states = np.zeros((count + 1, state0.size), dtype=np.float64)
    iterations = np.zeros(count + 1, dtype=np.int64)
    residuals = np.full(count + 1, np.nan, dtype=np.float64)
    states[0] = state0
    initial_residual: float | None = None
    status = "ok"
    last_index = 0
    failure_step: int | None = None
    failure_time: float | None = None
    failure_iterations: int | None = None
    failure_residual: float | None = None
    failure_residual_nonfinite = False
    termination_step: int | None = None
    termination_time: float | None = None
    nonfinite_stage: str | None = None
    numba_error: str | None = None
    numba_history_requested = bool(use_acceleration)
    numba_history_attempted = False
    numba_history_used = False
    numba_history_steps = 0
    numba_fallback_used = False
    history_evaluations = 0
    use_numba_history = numba_history_requested

    if initially_diverged:
        status = "diverged"
        termination_step = 0
        termination_time = terminal
    else:
        if initial_derivative is None:  # pragma: no cover - narrowed invariant
            raise RuntimeError("Missing initial RHS after a non-divergent start.")
        initial_residual = _vector_norm(initial_derivative)
        if regularity == "smooth" and initial_residual > compatibility_limit:
            warnings.warn(
                "A C1 type-III Caputo solution has zero derivative at the lower "
                "terminal, but ||f(a,x0)|| exceeds compatibility_tolerance. "
                "Use initial_regularity='nonsmooth' only when a weakly singular start "
                "is part of the model.",
                VariableOrderInitialCompatibilityWarning,
                stacklevel=2,
            )

    if status == "ok":
        for output_index in range(1, count + 1):
            alpha = float(orders[output_index])
            if use_numba_history:
                numba_history_attempted = True
                try:
                    history = _history_sum_numba(states, output_index, alpha)
                except Exception as exc:
                    if not allow_python_fallback:
                        raise RuntimeError(
                            "Numba variable-order history failed and "
                            f"allow_python_fallback=False: {exc}"
                        ) from exc
                    numba_error = str(exc)
                    numba_fallback_used = True
                    use_numba_history = False
                    history = _history_sum_python(states, output_index, alpha)
                else:
                    numba_history_used = True
                    numba_history_steps += 1
            else:
                history = _history_sum_python(states, output_index, alpha)
            history_evaluations += 1

            scale = float(gamma(2.0 - alpha) * normalized_step**alpha)
            base = states[output_index - 1] - history
            current_time = float(times[output_index])
            estimate = base + scale * evaluate_rhs(
                current_time,
                states[output_index - 1],
            )
            if not np.all(np.isfinite(estimate)):
                status = "nonfinite_solution"
                termination_step = output_index
                termination_time = current_time
                nonfinite_stage = "predictor"
                break

            estimate_rhs = evaluate_rhs(current_time, estimate)
            converged = False
            last_residual = float("inf")
            candidate = estimate
            for iteration in range(1, maximum_iterations + 1):
                candidate = base + scale * estimate_rhs
                if not np.all(np.isfinite(candidate)):
                    break
                candidate_rhs = evaluate_rhs(current_time, candidate)
                residual_vector = candidate - base - scale * candidate_rhs
                last_residual = _vector_norm(residual_vector)
                tolerance = atol + rtol * _vector_norm(candidate)
                if last_residual <= tolerance:
                    converged = True
                    if iteration < maximum_iterations:
                        polished = base + scale * candidate_rhs
                        if np.all(np.isfinite(polished)):
                            polished_rhs = evaluate_rhs(current_time, polished)
                            polished_residual = _vector_norm(
                                polished - base - scale * polished_rhs
                            )
                            if (
                                np.isfinite(polished_residual)
                                and polished_residual <= last_residual
                            ):
                                candidate = polished
                                candidate_rhs = polished_rhs
                                last_residual = polished_residual
                                iteration += 1
                    break
                estimate = candidate
                estimate_rhs = candidate_rhs

            iterations[output_index] = iteration
            residuals[output_index] = last_residual
            if not np.all(np.isfinite(candidate)):
                status = "nonfinite_solution"
                termination_step = output_index
                termination_time = current_time
                nonfinite_stage = "corrector"
                break
            if not converged:
                failure_step = output_index
                failure_time = current_time
                failure_iterations = int(iteration)
                if np.isfinite(last_residual):
                    failure_residual = float(last_residual)
                else:
                    failure_residual_nonfinite = True
                if nonconvergence == "raise":
                    raise VariableOrderCorrectorError(
                        "Variable-order type-III L1 Picard corrector failed at "
                        f"step {output_index}, t={current_time}, residual="
                        f"{last_residual:.17g}, max_iterations={maximum_iterations}."
                    )
                status = "corrector_nonconvergence"
                break

            physical_norm = _vector_norm(candidate)
            if not np.isfinite(physical_norm):
                status = "nonfinite_solution"
                termination_step = output_index
                termination_time = current_time
                nonfinite_stage = "physical_norm"
                break
            states[output_index] = candidate
            last_index = output_index
            if physical_divergence is not None and physical_norm > physical_divergence:
                status = "diverged"
                termination_step = output_index
                termination_time = current_time
                break

    returned_times = times[: last_index + 1]
    returned_states = states[: last_index + 1]
    returned_orders = orders[: last_index + 1]
    returned_iterations = iterations[: last_index + 1]
    returned_residuals = residuals[: last_index + 1]
    finite_residuals = returned_residuals[np.isfinite(returned_residuals)]
    residual_candidates = finite_residuals.tolist()
    if failure_residual is not None:
        residual_candidates.append(failure_residual)
    completed = max(0, len(returned_times) - 1)
    if numba_history_used and numba_fallback_used:
        backend = "numba_then_python_history_python_picard"
        history_backend = "numba_then_python"
    elif numba_history_used:
        backend = "numba_history_python_picard"
        history_backend = "numba"
    else:
        backend = "python_numpy_l1_picard"
        history_backend = "python" if history_evaluations > 0 else "not_executed"
    solver_info: dict[str, Any] = {
        "definition": "tavares_type_iii_current_time",
        "discretization": "uniform_l1",
        "corrector": "picard",
        "history_complexity": "O(N^2)",
        "history_component_work": "O(N^2*d)",
        "history_storage": "O(N*d)",
        "numba_history_requested": numba_history_requested,
        "numba_history_attempted": numba_history_attempted,
        "used_numba_history": numba_history_used,
        "numba_history_steps": numba_history_steps,
        "numba_fallback_used": numba_fallback_used,
        "numba_fallback_error": numba_error,
        "history_evaluations": history_evaluations,
        "history_backend": history_backend,
        "n_steps_requested": count,
        "n_steps": completed,
        "n_steps_completed": completed,
        "n_samples": len(returned_times),
        "n_samples_returned": len(returned_times),
        "order_min": float(np.min(returned_orders)),
        "order_max": float(np.max(returned_orders)),
        "order_function_signature": order_signature,
        "declared_initial_order": normalized_declared_initial_order,
        "initial_compatibility_residual": initial_residual,
        "compatibility_tolerance": compatibility_limit,
        "initial_regularity": regularity,
        "corrector_atol": atol,
        "corrector_rtol": rtol,
        "corrector_max_iterations": maximum_iterations,
        "max_corrector_iterations_used": max(
            int(np.max(returned_iterations)),
            0 if failure_iterations is None else failure_iterations,
        ),
        "max_corrector_residual": (
            None if not residual_candidates else float(max(residual_candidates))
        ),
        "on_nonconvergence": nonconvergence,
        "nonconverged_step": failure_step,
        "failure_time": failure_time,
        "failure_iterations": failure_iterations,
        "failure_residual": failure_residual,
        "failure_residual_nonfinite": failure_residual_nonfinite,
        "termination_step": termination_step,
        "termination_time": termination_time,
        "nonfinite_stage": nonfinite_stage,
        "physical_divergence_norm": physical_divergence,
        "validation_scope": "entire_requested_order_schedule_and_initial_rhs",
        "order_regularity_check": "pointwise_range_only",
        "picard_contractivity_check": "not_inferred",
        "published_scope": "type_iii_definition_and_l1_discretization",
        "hafo_adaptation": "implicit_system_picard_corrector",
    }
    return VariableOrderCaputoType3Result(
        times=np.asarray(returned_times, dtype=np.float64),
        states=np.asarray(returned_states, dtype=np.float64),
        orders=np.asarray(returned_orders, dtype=np.float64),
        corrector_iterations=np.asarray(returned_iterations, dtype=np.int64),
        corrector_residuals=np.asarray(returned_residuals, dtype=np.float64),
        lower_terminal=terminal,
        requested_upper_terminal=float(times[-1]),
        actual_upper_terminal=float(returned_times[-1]),
        step=normalized_step,
        n_steps_requested=count,
        method="vo_caputo_type3_l1",
        backend=backend,
        status=status,
        memory_policy="full_history",
        grid_coordinate="physical_time",
        order_function_name=declared_name,
        initial_regularity=regularity,
        solver_info=MappingProxyType(solver_info),
    )


__all__ = [
    "VARIABLE_ORDER_CAPUTO_TYPE3_REFERENCES",
    "VariableOrderCaputoType3Result",
    "VariableOrderCorrectorError",
    "VariableOrderInitialCompatibilityWarning",
    "integrate_variable_order_caputo_type3_l1",
    "variable_order_l1_weight",
]
