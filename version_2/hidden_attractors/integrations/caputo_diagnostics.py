"""Independent full-history Caputo L1 and stable PECE diagnostic solvers.

These maintained, generic counterparts of the audit experiments retain every
history term from t=0. They are O(N**2) work and O(N*d) storage. Stable analytic
weights and compensated history sums reduce cancellation; neither solver is a
ground truth for a nonlinear system or an asymptotic basin classifier.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.special import gamma

from .._rhs import bind_rhs
from .._time_grid import exact_fixed_step_count


def _kahan_history(weights: np.ndarray, history: np.ndarray) -> np.ndarray:
    total = np.zeros(history.shape[1], dtype=np.float64)
    correction = np.zeros_like(total)
    for j in range(len(history)):
        for d in range(history.shape[1]):
            term = weights[len(history) - 1 - j] * history[j, d] - correction[d]
            updated = total[d] + term
            correction[d] = (updated - total[d]) - term
            total[d] = updated
    return total


try:
    from numba import njit
    _compiled_history = njit(cache=True, fastmath=False)(_kahan_history)
except ImportError:  # pragma: no cover - supported pure Python fallback
    _compiled_history = None


def stable_caputo_weights(q: float, count: int, method: str) -> tuple[np.ndarray, ...]:
    """Compute analytic weights without subtracting adjacent large powers.

    PECE second differences use the convergent even binomial series around
    k+1; a0 uses its positive binomial remainder. Long-double intermediate
    precision is platform dependent and is reported by the integration API.
    All stored weights and states are binary64.
    """
    order = np.longdouble(q)
    lag = np.arange(count, dtype=np.longdouble)
    if method == "l1_implicit":
        weights = np.ones(count, dtype=np.longdouble)
        if count > 1:
            k = lag[1:]
            exponent = 1 - order
            weights[1:] = np.power(k, exponent) * np.expm1(exponent * np.log1p(1 / k))
        return (np.asarray(weights, dtype=float),)
    if method != "abm_stable":
        raise ValueError("method must be l1_implicit or abm_stable")
    predictor = np.ones(count, dtype=np.longdouble)
    corrector = np.empty(count, dtype=np.longdouble)
    first = np.empty(count, dtype=np.longdouble)
    if not count:
        return tuple(np.asarray(v, dtype=float) for v in (predictor, corrector, first))
    corrector[0] = 2 * np.expm1(order * np.log(np.longdouble(2)))
    first[0] = order
    if count > 1:
        k = lag[1:]
        predictor[1:] = np.power(k, order) * np.expm1(order * np.log1p(1 / k))
        u = 1 / (k + 1)
        # (1+u)^p+(1-u)^p-2, p=q+1. No subtraction of O(1) terms.
        term = order * (order + 1) * u * u
        even_sum = term.copy()
        for degree in range(4, 104, 2):
            term *= ((order + 3 - degree) * (order + 2 - degree)
                     / ((degree - 1) * degree)) * u * u
            even_sum += term
        corrector[1:] = np.power(k + 1, order + 1) * even_sum
        # (1-u)^p - 1 + p*u: all terms from degree two are positive.
        term = order * (order + 1) * u * u / 2
        remainder = term.copy()
        for degree in range(3, 104):
            term *= (degree - 2 - order) * u / degree
            remainder += term
        first[1:] = np.power(k + 1, order + 1) * remainder
    return tuple(np.asarray(v, dtype=float) for v in (predictor, corrector, first))


def integrate_caputo_diagnostic(
    rhs: Callable,
    x0: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    *,
    method: str,
    jacobian: Callable | None = None,
    divergence_norm: float | None = 120.0,
    newton_tol: float = 2e-14,
    newton_maxiter: int = 20,
    fd_relative_step: float = np.sqrt(np.finfo(float).eps),
    use_numba: bool = True,
    diagnostics: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Integrate D_C^q x=f(t,x), 0<q<1, with a fixed lower limit zero.

    The optional mutable ``diagnostics`` receives the same schema for both
    methods. L1 solves its nonlinear equation by Newton with an optional
    analytic Jacobian; otherwise a declared central finite difference is used.
    A failed nonlinear solve returns only accepted rows, never a guessed step.
    The PECE method takes exactly one predictor and one corrector per step.
    """
    if method not in {"l1_implicit", "abm_stable"}:
        raise ValueError("method must be l1_implicit or abm_stable")
    if isinstance(q, (bool, np.bool_)) or not np.isscalar(q) or not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("diagnostic Caputo solvers require finite 0 < q < 1")
    anchor = np.asarray(x0, dtype=float)
    if anchor.ndim != 1 or not anchor.size or not np.all(np.isfinite(anchor)):
        raise ValueError("x0 must be a nonempty finite one-dimensional vector")
    if divergence_norm is not None and (not np.isscalar(divergence_norm)
            or isinstance(divergence_norm, (bool, np.bool_))
            or not np.isfinite(divergence_norm) or divergence_norm <= 0):
        raise ValueError("divergence_norm must be positive and finite, or None")
    if not np.isfinite(newton_tol) or newton_tol <= 0:
        raise ValueError("newton_tol must be positive and finite")
    if isinstance(newton_maxiter, (bool, np.bool_)) or not isinstance(newton_maxiter, (int, np.integer)) or newton_maxiter < 1:
        raise ValueError("newton_maxiter must be a positive integer")
    if not np.isfinite(fd_relative_step) or fd_relative_step <= 0:
        raise ValueError("fd_relative_step must be positive and finite")
    if not isinstance(use_numba, (bool, np.bool_)):
        raise ValueError("use_numba must be a boolean")
    count = exact_fixed_step_count(h, t_final, caller="integrate_caputo_diagnostic")
    bound_rhs = bind_rhs(rhs)
    bound_jacobian = bind_rhs(jacobian) if jacobian is not None else None
    record = diagnostics if diagnostics is not None else {}
    record.clear()
    record.update(
        schema_version="caputo_diagnostic_v1", method=method, order=float(q),
        step=float(h), requested_horizon=float(t_final), caputo_lower_limit=0.0,
        memory_mode="full", history_restarted=False, history_truncated=False,
        state_dtype="float64", weight_intermediate_precision_bits=int(np.finfo(np.longdouble).nmant + 1),
        reduction="kahan_numba" if use_numba and _compiled_history is not None else "kahan_python",
        nfev=0, njev=0, history_terms=0, newton_iterations_max=0,
        estimated_history_terms=count * count if method == "abm_stable" else count * (count - 1) // 2,
        work_complexity="O(N^2*d)", storage_complexity="O(N*d)",
        newton_residual_max=0.0, newton_failures=0,
        newton_tolerance=float(newton_tol) if method == "l1_implicit" else None,
        newton_maxiter=int(newton_maxiter) if method == "l1_implicit" else None,
        newton_residual_definition="norm_inf(x-base-c*f)/(1+norm_inf(x))" if method == "l1_implicit" else None,
        jacobian=("analytic" if jacobian is not None else "central_finite_difference") if method == "l1_implicit" else "not_used",
        fd_relative_step=float(fd_relative_step) if method == "l1_implicit" and jacobian is None else None,
        divergence_norm=None if divergence_norm is None else float(divergence_norm),
        failure_detail=None,
    )
    started = time.perf_counter()
    times = np.arange(count + 1, dtype=float) * h
    states = np.zeros((count + 1, anchor.size), dtype=float)
    history = np.zeros_like(states)
    states[0] = anchor
    weights = stable_caputo_weights(q, count, method)
    reduce_history = _compiled_history if use_numba and _compiled_history is not None else _kahan_history

    def field(t, state):
        record["nfev"] += 1
        value = np.asarray(bound_rhs(t, state), dtype=float)
        if value.shape != anchor.shape:
            raise ValueError(f"RHS shape {value.shape} must equal x0 shape {anchor.shape}")
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("RHS contains nonfinite values")
        return value

    def matrix(t, state):
        record["njev"] += 1
        if bound_jacobian is not None:
            value = np.asarray(bound_jacobian(t, state), dtype=float)
        else:
            value = np.empty((anchor.size, anchor.size))
            for d in range(anchor.size):
                delta = fd_relative_step * max(1.0, abs(state[d]))
                plus, minus = state.copy(), state.copy()
                plus[d] += delta
                minus[d] -= delta
                value[:, d] = (field(t, plus) - field(t, minus)) / (2 * delta)
        if value.shape != (anchor.size, anchor.size):
            raise ValueError("Jacobian must have shape (dimension, dimension)")
        if not np.all(np.isfinite(value)):
            raise FloatingPointError("Jacobian contains nonfinite values")
        return value

    def finish(last, status):
        record.update(status=status, final_time=float(times[last]), rows=int(last + 1),
                      completed_steps=int(last), full_horizon_reached=last == count and status == "ok",
                      elapsed_seconds=time.perf_counter() - started)
        return times[:last + 1], states[:last + 1], status

    # t=0 belongs to the declared experiment. An already crossed finite
    # initial state is the first threshold event, even if a stiff subsequent
    # step would return inside the bound. No RHS evaluation is needed.
    if divergence_norm is not None and np.linalg.norm(anchor) > divergence_norm:
        return finish(0, "diverged")

    # Validate even L1's unused initial value; malformed callbacks are contracts,
    # not scalar values to be broadcast into a state vector.
    try:
        initial = field(0.0, anchor)
    except Exception as exc:
        record["failure_detail"] = f"{type(exc).__name__}: {exc}"
        finish(0, "invalid_initial_rhs")
        raise ValueError(f"invalid initial RHS: {exc}") from exc
    if method == "abm_stable":
        history[0] = initial
    l1_scale = float(gamma(2 - q) * h**q)
    predictor_scale = float(h**q / gamma(q + 1))
    corrector_scale = float(h**q / gamma(q + 2))
    for n in range(1, count + 1):
        try:
            if method == "l1_implicit":
                # history[j-1] = x_j-x_(j-1), j=1,...,n-1;
                # reversed weights must begin at lag one, not lag zero.
                past = reduce_history(weights[0][1:n], history[:n - 1])
                record["history_terms"] += n - 1
                base = states[n - 1] - past
                current = states[n - 1].copy()
                success = False
                for iteration in range(newton_maxiter + 1):
                    residual = current - base - l1_scale * field(times[n], current)
                    normalized = float(np.max(abs(residual)) / (1 + np.max(abs(current))))
                    if normalized <= newton_tol:
                        record["newton_residual_max"] = max(record["newton_residual_max"], normalized)
                        success = True
                        break
                    if iteration == newton_maxiter:
                        break
                    current -= np.linalg.solve(np.eye(anchor.size) - l1_scale * matrix(times[n], current), residual)
                    if not np.all(np.isfinite(current)):
                        raise FloatingPointError("Newton iterate contains nonfinite values")
                record["newton_iterations_max"] = max(record["newton_iterations_max"], iteration)
                if not success:
                    record["newton_failures"] += 1
                    record["failure_detail"] = f"Newton did not reach tolerance; last normalized residual={normalized:.17g}"
                    return finish(n - 1, "newton_failure")
                history[n - 1] = current - states[n - 1]
            else:
                predicted = anchor + predictor_scale * reduce_history(weights[0][:n], history[:n])
                if not np.all(np.isfinite(predicted)):
                    raise FloatingPointError("predicted state contains nonfinite values")
                predicted_rhs = field(times[n], predicted)
                old = reduce_history(weights[1][:n - 1], history[1:n])
                current = anchor + corrector_scale * (weights[2][n - 1] * history[0] + old + predicted_rhs)
                record["history_terms"] += 2 * n - 1
                if not np.all(np.isfinite(current)):
                    raise FloatingPointError("corrected state contains nonfinite values")
                if divergence_norm is not None and np.linalg.norm(current) > divergence_norm:
                    # The finite state crosses the declared norm before a new
                    # RHS is required; do not turn that event into a later
                    # overflow of f(current).
                    states[n] = current
                    return finish(n, "diverged")
                # Validate before assignment; malformed later RHS leaves only
                # accepted rows, consistently across both maintained methods.
                history[n] = field(times[n], current)
            states[n] = current
            if divergence_norm is not None and np.linalg.norm(current) > divergence_norm:
                return finish(n, "diverged")
        except FloatingPointError as exc:
            record["failure_detail"] = f"{type(exc).__name__}: {exc}"
            return finish(n - 1, "nonfinite_solution")
        except np.linalg.LinAlgError as exc:
            record["newton_failures"] += 1
            record["failure_detail"] = f"{type(exc).__name__}: {exc}"
            return finish(n - 1, "newton_failure")
        except Exception as exc:
            record["failure_detail"] = f"{type(exc).__name__}: {exc}"
            return finish(n - 1, "solver_exception")
    return finish(count, "ok")


def integrate_l1_implicit(rhs, x0, q, h, t_final, **kwargs):
    """Generic implicit L1 derivative discretization, full Caputo memory."""
    return integrate_caputo_diagnostic(rhs, x0, q, h, t_final, method="l1_implicit", **kwargs)


def integrate_abm_stable(rhs, x0, q, h, t_final, **kwargs):
    """Generic stable-weight, Kahan-summed one-correction PECE from t=0."""
    return integrate_caputo_diagnostic(rhs, x0, q, h, t_final, method="abm_stable", **kwargs)


__all__ = ["integrate_caputo_diagnostic", "integrate_l1_implicit", "integrate_abm_stable", "stable_caputo_weights"]
