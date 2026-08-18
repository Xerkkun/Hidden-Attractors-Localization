r"""Caputo distributed-order solver on a uniform physical-time grid.

Stability: experimental

HAFO solves the finite order-quadrature model

.. math::

   \sum_{r=1}^{R}\Omega_r\,{}_{a}^{C}D_t^{\alpha_r}x(t)
   = f(t,x(t)), \qquad 0 < \alpha_r \le 1,

where the caller supplies the quadrature nodes and declares whether the input
weights are already discrete masses or quadrature weights multiplied by an
explicit density.  Every Caputo term below order one is discretized by the
uniform L1 formula.  A node exactly at one is handled as the classical
derivative and therefore reduces to backward Euler after time discretization.

For fixed nodes the per-order L1 histories can be combined before stepping:

.. math::

   K_k = \sum_r \frac{\Omega_r h^{-\alpha_r}}
                          {\Gamma(2-\alpha_r)}
         \left[(k+1)^{1-\alpha_r}-k^{1-\alpha_r}\right].

This removes an order dimension from the time loop.  Kernel construction costs
``O(R*N)`` and the direct trajectory history costs ``O(N**2*d)`` instead of a
naive ``O(R*N**2*d)`` implementation.  No order-by-time-by-state tensor is
materialized.  The nonlinear implicit state equation is solved by a reported
Picard iteration; this vector-system corrector and the combined-kernel
optimization are explicit HAFO adaptations.

References
----------
M. Caputo, "Distributed order differential equations modelling dielectric
induction and diffusion", Fractional Calculus and Applied Analysis 4 (2001),
421--442.  No DOI is asserted.

K. Diethelm and N. J. Ford, "Numerical analysis for distributed-order
differential equations", Journal of Computational and Applied Mathematics
225 (2009), 96--104, https://doi.org/10.1016/j.cam.2008.07.018.

C. Huang, H. Chen and N. An, "Beta-robust superconvergent analysis of a finite
element method for the distributed order time-fractional diffusion equation",
Journal of Scientific Computing (2022),
https://doi.org/10.1007/s10915-021-01726-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping
import warnings

import numpy as np
from numba import njit
from scipy.special import gamma

from ._validation import strict_count as _strict_count

from .._rhs import bind_rhs
from .distributed_order import _order_measure


DISTRIBUTED_ORDER_CAPUTO_L1_REFERENCES = (
    "https://www.math.bas.bg/complan/fcaa/volume4/index.html",
    "https://doi.org/10.1016/j.cam.2008.07.018",
    "https://doi.org/10.1007/s10915-021-01726-2",
)


class DistributedOrderInitialCompatibilityWarning(UserWarning):
    """A requested smooth fractional start conflicts with the endpoint equation."""


class DistributedOrderCorrectorError(RuntimeError):
    """The implicit distributed-order L1 corrector did not converge."""


@dataclass(frozen=True, slots=True)
class DistributedOrderCaputoResult:
    """Finite trajectory and complete distributed-order diagnostics."""

    times: np.ndarray
    states: np.ndarray
    corrector_iterations: np.ndarray
    corrector_residuals: np.ndarray
    order_nodes: np.ndarray
    quadrature_weights: np.ndarray
    density_values: np.ndarray | None
    effective_weights: np.ndarray
    l1_coefficients: np.ndarray
    combined_l1_kernel: np.ndarray
    weight_semantics: str
    normalization: str
    raw_mass: float
    raw_l1_norm: float
    mass: float
    l1_norm: float
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
    order_quadrature_name: str
    initial_regularity: str
    solver_info: Mapping[str, Any]
    references: tuple[str, ...] = DISTRIBUTED_ORDER_CAPUTO_L1_REFERENCES
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


def _vector_norm(values: np.ndarray) -> float:
    return float(np.hypot.reduce(np.abs(values)))


def distributed_order_l1_weight(order: float, lag: int) -> float:
    r"""Return the unscaled uniform-L1 weight at one order and lag.

    For ``0 < order < 1`` the value is
    ``(lag+1)**(1-order)-lag**(1-order)``.  At ``order == 1`` HAFO uses the
    exact backward-Euler limit: one for lag zero and zero for every older lag.
    """

    alpha = _real_scalar(order, name="order")
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("order must lie in (0, 1].")
    index = _strict_count(lag, name="lag", minimum=0)
    if index == 0:
        return 1.0
    if alpha == 1.0:
        return 0.0
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
    if order == 1.0:
        return 0.0
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
def _combined_kernel_numba(
    order_nodes: np.ndarray,
    l1_coefficients: np.ndarray,
    n_lags: int,
) -> np.ndarray:
    kernel = np.zeros(n_lags, dtype=np.float64)
    for order_index in range(order_nodes.size):
        alpha = order_nodes[order_index]
        coefficient = l1_coefficients[order_index]
        for lag in range(n_lags):
            kernel[lag] += coefficient * _l1_weight_numba(alpha, lag)
    return kernel


def _combined_kernel_python(
    order_nodes: np.ndarray,
    l1_coefficients: np.ndarray,
    n_lags: int,
) -> np.ndarray:
    kernel = np.zeros(n_lags, dtype=np.float64)
    for alpha, coefficient in zip(order_nodes, l1_coefficients, strict=True):
        for lag in range(n_lags):
            kernel[lag] += float(coefficient) * distributed_order_l1_weight(
                float(alpha), lag
            )
    return kernel


@njit(cache=True)
def _history_sum_numba(
    states: np.ndarray,
    output_index: int,
    combined_kernel: np.ndarray,
) -> np.ndarray:
    dimension = states.shape[1]
    history = np.zeros(dimension, dtype=np.float64)
    for component in range(dimension):
        total = 0.0
        for history_index in range(output_index - 1):
            lag = output_index - history_index - 1
            total += combined_kernel[lag] * (
                states[history_index + 1, component]
                - states[history_index, component]
            )
        history[component] = total
    return history


def _history_sum_python(
    states: np.ndarray,
    output_index: int,
    combined_kernel: np.ndarray,
) -> np.ndarray:
    history = np.zeros(states.shape[1], dtype=np.float64)
    for history_index in range(output_index - 1):
        lag = output_index - history_index - 1
        history += combined_kernel[lag] * (
            states[history_index + 1] - states[history_index]
        )
    return history


def integrate_distributed_order_caputo_l1(
    rhs: Callable,
    initial_state: Any,
    parameters: Any = None,
    *,
    order_nodes: Any,
    order_weights: Any,
    step: float,
    n_steps: int,
    lower_terminal: float = 0.0,
    weight_semantics: str = "nonnegative_mass",
    density_values: Any | None = None,
    normalization: str = "none",
    order_quadrature_name: str = "explicit_nodes_and_declared_weights",
    corrector_atol: float = 1.0e-12,
    corrector_rtol: float = 1.0e-10,
    corrector_max_iterations: int = 50,
    on_nonconvergence: str = "raise",
    initial_regularity: str = "unknown",
    compatibility_tolerance: float = 1.0e-10,
    use_acceleration: bool = True,
    allow_python_fallback: bool = True,
    divergence_norm: float | None = 120.0,
) -> DistributedOrderCaputoResult:
    """Solve a finite positive-measure Caputo distributed-order IVP.

    The order rule is explicit and serializable.  Signed measures are retained
    by :func:`distributed_order_gl_derivative` for operator diagnostics but are
    deliberately rejected by this solver because positivity and invertibility
    of the current-step coefficient would otherwise require a separate
    well-posedness contract.  Only full history is implemented.

    ``initial_regularity='smooth'`` requests the endpoint compatibility check
    ``f(a,x0)=0`` when all effective mass lies below order one.  If the rule has
    mass at order one, the classical derivative contributes at the endpoint
    and that zero-residual check is not applicable.
    """

    if not callable(rhs):
        raise TypeError("rhs must be callable.")
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
        raise ValueError("corrector_atol and corrector_rtol must not both be zero.")
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
        physical_divergence = _real_scalar(divergence_norm, name="divergence_norm")
        if physical_divergence <= 0.0:
            raise ValueError("divergence_norm must be positive or None.")

    semantics = str(weight_semantics).strip().lower()
    if semantics not in {
        "nonnegative_mass",
        "nonnegative_quadrature_density",
    }:
        raise ValueError(
            "The Caputo distributed-order solver requires nonnegative_mass or "
            "nonnegative_quadrature_density semantics; signed measures remain "
            "operator-only."
        )
    normalization_value = str(normalization).strip().lower()
    (
        nodes,
        quadrature_weights,
        density,
        effective_weights,
        raw_mass,
        raw_l1_norm,
        mass,
        l1_norm,
    ) = _order_measure(
        order_nodes,
        order_weights,
        density_values,
        semantics,
        normalization_value,
    )

    quadrature_name = str(order_quadrature_name).strip()
    if not quadrature_name:
        raise ValueError("order_quadrature_name must not be empty.")

    l1_coefficients = effective_weights * np.power(
        normalized_step, -nodes
    ) / gamma(2.0 - nodes)
    l1_coefficients = np.ascontiguousarray(l1_coefficients, dtype=np.float64)
    if not np.all(np.isfinite(l1_coefficients)):
        raise ValueError(
            "The step/order measure produces non-finite L1 coefficients."
        )
    current_coefficient = float(np.sum(l1_coefficients, dtype=np.float64))
    coefficient_norm = float(
        np.sum(np.abs(l1_coefficients), dtype=np.float64)
    )
    if current_coefficient <= 0.0:
        raise ValueError(
            "The distributed-order current-step coefficient must remain "
            "positive after floating-point evaluation."
        )

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

    times = terminal + np.arange(count + 1, dtype=np.float64) * normalized_step

    numba_error: str | None = None
    numba_requested = bool(use_acceleration)
    numba_kernel_attempted = numba_requested
    numba_kernel_used = False
    numba_history_steps = 0
    use_numba_history = numba_requested
    if use_numba_history:
        try:
            combined_kernel = _combined_kernel_numba(
                nodes,
                l1_coefficients,
                count,
            )
            numba_kernel_used = True
        except Exception as exc:
            if not allow_python_fallback:
                raise RuntimeError(
                    "Numba distributed-order kernel construction failed and "
                    f"allow_python_fallback=False: {exc}"
                ) from exc
            numba_error = str(exc)
            use_numba_history = False
            combined_kernel = _combined_kernel_python(
                nodes,
                l1_coefficients,
                count,
            )
    else:
        combined_kernel = _combined_kernel_python(
            nodes,
            l1_coefficients,
            count,
        )
    combined_kernel = np.ascontiguousarray(combined_kernel, dtype=np.float64)
    if not np.all(np.isfinite(combined_kernel)):
        raise ValueError("The combined L1 kernel contains non-finite values.")
    kernel_tolerance = 64.0 * np.finfo(np.float64).eps * max(
        np.finfo(np.float64).tiny,
        abs(current_coefficient),
        abs(float(combined_kernel[0])),
    )
    if abs(float(combined_kernel[0]) - current_coefficient) > kernel_tolerance:
        raise RuntimeError("Combined L1 kernel failed its current-coefficient invariant.")

    states = np.zeros((count + 1, state0.size), dtype=np.float64)
    iterations = np.zeros(count + 1, dtype=np.int64)
    residuals = np.full(count + 1, np.nan, dtype=np.float64)
    states[0] = state0
    status = "ok"
    last_index = 0
    failure_step: int | None = None
    failure_time: float | None = None
    failure_iterations: int | None = None
    failure_residual: float | None = None
    failure_residual_nonfinite = False

    initial_norm = _vector_norm(state0)
    initial_residual: float | None = None
    alpha_one_mask = nodes == 1.0
    alpha_one_effective_mass = float(
        np.sum(effective_weights[alpha_one_mask], dtype=np.float64)
    )
    alpha_one_coefficient = float(
        np.sum(l1_coefficients[alpha_one_mask], dtype=np.float64)
    )
    compatibility_applies = alpha_one_effective_mass == 0.0
    if physical_divergence is not None and initial_norm > physical_divergence:
        status = "diverged"
    else:
        initial_derivative = evaluate_rhs(terminal, state0)
        initial_residual = _vector_norm(initial_derivative)
        if (
            regularity == "smooth"
            and compatibility_applies
            and initial_residual > compatibility_limit
        ):
            warnings.warn(
                "A C1 Caputo distributed-order solution with all effective mass "
                "strictly below order one has zero operator value at the lower "
                "terminal, but ||f(a,x0)|| exceeds compatibility_tolerance. Use "
                "initial_regularity='nonsmooth' only when a weakly singular start "
                "is part of the model.",
                DistributedOrderInitialCompatibilityWarning,
                stacklevel=2,
            )

    if status == "ok":
        for output_index in range(1, count + 1):
            if use_numba_history:
                try:
                    history = _history_sum_numba(
                        states,
                        output_index,
                        combined_kernel,
                    )
                    numba_history_steps += 1
                except Exception as exc:
                    if not allow_python_fallback:
                        raise RuntimeError(
                            "Numba distributed-order history failed and "
                            f"allow_python_fallback=False: {exc}"
                        ) from exc
                    numba_error = str(exc)
                    use_numba_history = False
                    history = _history_sum_python(
                        states,
                        output_index,
                        combined_kernel,
                    )
            else:
                history = _history_sum_python(
                    states,
                    output_index,
                    combined_kernel,
                )

            base = states[output_index - 1] - history / current_coefficient
            current_time = float(times[output_index])
            estimate = base + evaluate_rhs(
                current_time,
                states[output_index - 1],
            ) / current_coefficient
            if not np.all(np.isfinite(estimate)):
                failure_step = output_index
                failure_time = current_time
                failure_residual_nonfinite = True
                status = "nonfinite_solution"
                break

            estimate_rhs = evaluate_rhs(current_time, estimate)
            converged = False
            last_residual = float("inf")
            candidate = estimate
            iteration = 0
            for iteration in range(1, maximum_iterations + 1):
                candidate = base + estimate_rhs / current_coefficient
                if not np.all(np.isfinite(candidate)):
                    break
                candidate_rhs = evaluate_rhs(current_time, candidate)
                residual_vector = (
                    current_coefficient * (candidate - base) - candidate_rhs
                )
                last_residual = _vector_norm(residual_vector)
                tolerance = atol + rtol * max(
                    _vector_norm(candidate_rhs),
                    current_coefficient * _vector_norm(candidate),
                )
                if last_residual <= tolerance:
                    converged = True
                    if iteration < maximum_iterations:
                        polished = base + candidate_rhs / current_coefficient
                        if np.all(np.isfinite(polished)):
                            polished_rhs = evaluate_rhs(current_time, polished)
                            polished_residual = _vector_norm(
                                current_coefficient * (polished - base)
                                - polished_rhs
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
                estimate_rhs = candidate_rhs

            iterations[output_index] = iteration
            residuals[output_index] = last_residual
            if not np.all(np.isfinite(candidate)):
                failure_step = output_index
                failure_time = current_time
                failure_iterations = int(iteration)
                failure_residual_nonfinite = True
                status = "nonfinite_solution"
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
                    raise DistributedOrderCorrectorError(
                        "Distributed-order Caputo L1 Picard corrector failed at "
                        f"step {output_index}, t={current_time}, residual="
                        f"{last_residual:.17g}, max_iterations={maximum_iterations}."
                    )
                status = "corrector_nonconvergence"
                break

            physical_norm = _vector_norm(candidate)
            if not np.isfinite(physical_norm):
                failure_step = output_index
                failure_time = current_time
                failure_iterations = int(iteration)
                failure_residual_nonfinite = True
                status = "nonfinite_solution"
                break
            states[output_index] = candidate
            last_index = output_index
            if physical_divergence is not None and physical_norm > physical_divergence:
                status = "diverged"
                break

    returned_times = times[: last_index + 1]
    returned_states = states[: last_index + 1]
    returned_iterations = iterations[: last_index + 1]
    returned_residuals = residuals[: last_index + 1]
    finite_residuals = returned_residuals[np.isfinite(returned_residuals)]
    residual_candidates = finite_residuals.tolist()
    if failure_residual is not None:
        residual_candidates.append(failure_residual)
    completed = max(0, len(returned_times) - 1)
    fallback_occurred = numba_error is not None
    if fallback_occurred and (numba_kernel_used or numba_history_steps > 0):
        backend = "hybrid_numba_python_combined_l1_picard"
    elif numba_history_steps > 0:
        backend = "numba_combined_history_python_picard"
    elif numba_kernel_used:
        backend = "numba_kernel_no_history_python_picard"
    else:
        backend = "python_numpy_combined_l1_picard"
    solver_info: dict[str, Any] = {
        "definition": "caputo_distributed_order_discrete_measure",
        "discretization": "uniform_l1_with_exact_alpha_one_backward_euler",
        "corrector": "picard",
        "order_quadrature": quadrature_name,
        "order_quadrature_error_estimated": False,
        "time_discretization_error_estimated": False,
        "kernel_precomputation_complexity": "O(R*N)",
        "history_complexity": "O(N^2*d)",
        "total_structural_complexity": "O(R*N + N^2*d)",
        "history_storage": "O(N*d + N + R)",
        "naive_order_time_state_tensor_avoided": True,
        "numba_requested": numba_requested,
        "numba_kernel_attempted": numba_kernel_attempted,
        "numba_kernel_used": numba_kernel_used,
        "numba_history_steps": numba_history_steps,
        "used_numba_history": numba_history_steps > 0,
        "numba_fallback_occurred": fallback_occurred,
        "numba_fallback_error": numba_error,
        "n_steps_requested": count,
        "n_steps": completed,
        "n_steps_completed": completed,
        "n_samples": len(returned_times),
        "n_samples_returned": len(returned_times),
        "n_order_nodes": int(nodes.size),
        "order_min": float(np.min(nodes)),
        "order_max": float(np.max(nodes)),
        "weight_semantics": semantics,
        "normalization": normalization_value,
        "raw_mass": raw_mass,
        "raw_l1_norm": raw_l1_norm,
        "mass": mass,
        "l1_norm": l1_norm,
        "current_step_coefficient": current_coefficient,
        "current_step_coefficient_l1_norm": coefficient_norm,
        "alpha_one_handling": "exact_backward_euler_limit",
        "alpha_one_effective_mass": alpha_one_effective_mass,
        "alpha_one_current_coefficient": alpha_one_coefficient,
        "initial_compatibility_check_applies": compatibility_applies,
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
        "physical_divergence_norm": physical_divergence,
        "validation_scope": "entire_order_rule_and_initial_rhs",
        "published_scope": "distributed_order_quadrature_and_single_order_l1",
        "hafo_adaptation": "combined_kernel_and_implicit_system_picard_corrector",
        "picard_contractivity_check": "not_inferred",
    }
    return DistributedOrderCaputoResult(
        times=np.asarray(returned_times, dtype=np.float64),
        states=np.asarray(returned_states, dtype=np.float64),
        corrector_iterations=np.asarray(returned_iterations, dtype=np.int64),
        corrector_residuals=np.asarray(returned_residuals, dtype=np.float64),
        order_nodes=np.array(nodes, dtype=np.float64, copy=True),
        quadrature_weights=np.array(
            quadrature_weights,
            dtype=np.float64,
            copy=True,
        ),
        density_values=(
            None
            if density is None
            else np.array(density, dtype=np.float64, copy=True)
        ),
        effective_weights=np.array(
            effective_weights,
            dtype=np.float64,
            copy=True,
        ),
        l1_coefficients=np.array(
            l1_coefficients,
            dtype=np.float64,
            copy=True,
        ),
        combined_l1_kernel=np.array(
            combined_kernel,
            dtype=np.float64,
            copy=True,
        ),
        weight_semantics=semantics,
        normalization=normalization_value,
        raw_mass=raw_mass,
        raw_l1_norm=raw_l1_norm,
        mass=mass,
        l1_norm=l1_norm,
        lower_terminal=terminal,
        requested_upper_terminal=float(times[-1]),
        actual_upper_terminal=float(returned_times[-1]),
        step=normalized_step,
        n_steps_requested=count,
        method="distributed_order_caputo_l1",
        backend=backend,
        status=status,
        memory_policy="full_history",
        grid_coordinate="physical_time",
        order_quadrature_name=quadrature_name,
        initial_regularity=regularity,
        solver_info=MappingProxyType(solver_info),
    )


__all__ = [
    "DISTRIBUTED_ORDER_CAPUTO_L1_REFERENCES",
    "DistributedOrderCaputoResult",
    "DistributedOrderCorrectorError",
    "DistributedOrderInitialCompatibilityWarning",
    "distributed_order_l1_weight",
    "integrate_distributed_order_caputo_l1",
]
