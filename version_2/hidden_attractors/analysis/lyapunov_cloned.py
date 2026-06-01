"""F3 cloned-dynamics finite-time Lyapunov spectrum estimators.

The implementation follows the block-restarted cloned-dynamics contract from
Fischer, Zourmba, and Mohamadou (2020).  It does not use a Jacobian or a
variational system.  Fractional results are block-restarted approximations,
not full-memory Caputo-aware estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from ..integrations.abm_fractional import (
    classify_component_orders,
    integrate_fractional_abm,
    normalize_component_orders,
)
from ..integrations.rk4 import rk4_integrate

FISCHER_2020_REFERENCE = (
    "Fischer, Zourmba, and Mohamadou 2020 - Lyapunov exponents spectrum "
    "estimation of fractional order nonlinear systems using Cloned Dynamics "
    "(Applied Numerical Mathematics 154, 187-204; DOI: 10.1016/j.apnum.2020.03.027)"
)

_WARNINGS = (
    "Finite-time local Lyapunov indicators; convergence depends on delta, t_clone, K, h, and orders.",
    "Fractional memory protocol is published_block_restart; this is not a full-memory Caputo-aware claim.",
    "Does not certify chaos; does not certify hiddenness of attractors.",
)


@dataclass(frozen=True)
class ClonedDynamicsResult:
    """Finite-time spectrum returned by :func:`compute_cloned_dynamics_spectrum`."""

    exponents: np.ndarray
    times: np.ndarray
    convergence: np.ndarray
    status: str
    method_id: str
    derivative_model: str
    q: float
    finite_time_local: bool = True
    jacobian_required: bool = False
    orthonormalization: str = "gs"
    reference_ids: tuple[str, ...] = (FISCHER_2020_REFERENCE,)
    methodological_warnings: tuple[str, ...] = _WARNINGS
    method_metadata: dict[str, object] = field(default_factory=dict)
    bounded_trajectory: bool = True


def _modified_gram_schmidt(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return orthonormal columns and residual norms."""

    q = np.zeros_like(vectors)
    residual_norms = np.zeros(vectors.shape[1], dtype=float)
    for col in range(vectors.shape[1]):
        residual = vectors[:, col].copy()
        for previous in range(col):
            residual -= np.dot(q[:, previous], residual) * q[:, previous]
        residual_norms[col] = np.linalg.norm(residual)
        if not np.isfinite(residual_norms[col]) or residual_norms[col] <= 0.0:
            return q, residual_norms
        q[:, col] = residual / residual_norms[col]
    return q, residual_norms


def _classical_gram_schmidt(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return classical Gram-Schmidt columns for diagnostic comparisons."""

    q = np.zeros_like(vectors)
    residual_norms = np.zeros(vectors.shape[1], dtype=float)
    for col in range(vectors.shape[1]):
        source = vectors[:, col]
        residual = source.copy()
        for previous in range(col):
            residual -= np.dot(q[:, previous], source) * q[:, previous]
        residual_norms[col] = np.linalg.norm(residual)
        if not np.isfinite(residual_norms[col]) or residual_norms[col] <= 0.0:
            return q, residual_norms
        q[:, col] = residual / residual_norms[col]
    return q, residual_norms


def _orthonormalize(vectors: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method in {"gs", "gs_modified"}:
        return _modified_gram_schmidt(vectors)
    if method == "gs_classical":
        return _classical_gram_schmidt(vectors)
    if method == "qr":
        q, r = np.linalg.qr(vectors)
        diagonal = np.diag(r)
        signs = np.where(diagonal < 0.0, -1.0, 1.0)
        return q * signs, np.abs(diagonal)
    raise ValueError("method must be 'gs', 'gs_modified', 'gs_classical', or 'qr'.")


def compute_cloned_dynamics_spectrum(
    rhs: Callable,
    x0: np.ndarray,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    h: float,
    t_clone: float,
    n_clones: int | None,
    k_blocks: int,
    delta: float,
    method: str = "gs",
    memory_protocol: str = "published_block_restart",
    system_id: str | None = None,
    parameters: Mapping[str, float] | None = None,
    return_history: bool = False,
    random_seed: int | None = None,
    divergence_norm: float | None = None,
    integration_mode: str = "fractional_abm",
) -> ClonedDynamicsResult:
    """Estimate a spectrum from perturbed clones without a Jacobian.

    The number of clones is the state dimension.  ``n_clones`` is explicit in
    the API so callers cannot accidentally confuse it with the number of ABM
    steps inside a clone interval.
    """

    state = np.asarray(x0, dtype=float).reshape(-1)
    dimension = state.size
    component_orders = normalize_component_orders(orders, dimension)
    n_clones = dimension if n_clones is None else int(n_clones)
    h = float(h)
    t_clone = float(t_clone)
    delta = float(delta)
    k_blocks = int(k_blocks)
    if n_clones != dimension:
        raise ValueError(f"n_clones must equal the state dimension ({dimension}).")
    if h <= 0.0 or t_clone <= 0.0 or delta <= 0.0 or k_blocks < 1:
        raise ValueError("h, t_clone, delta, and k_blocks must be positive.")
    n_steps = int(round(t_clone / h))
    if n_steps < 1 or not np.isclose(n_steps * h, t_clone):
        raise ValueError("t_clone must be an integer multiple of h.")
    if memory_protocol not in {
        "published_block_restart",
        "published_block_restart_or_experimental_qr",
        "experimental_qr_block_restart",
    }:
        raise ValueError("unsupported cloned-dynamics memory protocol.")
    if integration_mode not in {"fractional_abm", "integer_rk4_reference"}:
        raise ValueError("integration_mode must be 'fractional_abm' or 'integer_rk4_reference'.")
    if integration_mode == "integer_rk4_reference" and not np.allclose(component_orders, 1.0):
        raise ValueError("integer_rk4_reference is available only when all orders are q=1.")

    direction_basis = np.eye(dimension, dtype=float)
    sum_logs = np.zeros(dimension, dtype=float)
    convergence: list[np.ndarray] = []
    times: list[float] = []
    block_history: list[dict[str, object]] = []
    bounded_trajectory = True
    status = "ok"

    for block in range(1, k_blocks + 1):
        initial_copies = np.vstack((state, state + delta * direction_basis.T))
        augmented_orders = np.tile(component_orders, dimension + 1)

        def augmented_rhs(t: float, flattened: np.ndarray) -> np.ndarray:
            copies = np.asarray(flattened, dtype=float).reshape(dimension + 1, dimension)
            derivatives = []
            for copy in copies:
                if parameters is not None:
                    try:
                        value = rhs(t, copy, parameters)
                    except TypeError:
                        value = rhs(copy, parameters)
                else:
                    try:
                        value = rhs(t, copy)
                    except TypeError:
                        value = rhs(copy)
                derivatives.append(np.asarray(value, dtype=float))
            return np.asarray(derivatives, dtype=float).reshape(-1)

        if integration_mode == "integer_rk4_reference":
            _, trajectories, integration_status, _ = rk4_integrate(
                augmented_rhs,
                initial_copies.reshape(-1),
                h,
                n_steps,
                divergence_norm=float("inf") if divergence_norm is None else divergence_norm,
            )
        else:
            _, trajectories, integration_status = integrate_fractional_abm(
                augmented_rhs,
                initial_copies.reshape(-1),
                augmented_orders,
                h,
                n_steps,
                memory_protocol="published_block_restart",
                divergence_norm=divergence_norm,
            )
        if integration_status != "ok":
            status = "numerical_failure"
            bounded_trajectory = integration_status != "diverged"
            break

        final_copies = trajectories[-1].reshape(dimension + 1, dimension)
        fiducial_end = final_copies[0]
        differences = (final_copies[1:] - fiducial_end).T
        directions, residual_norms = _orthonormalize(differences, method)
        if (
            not np.all(np.isfinite(residual_norms))
            or np.any(residual_norms <= 0.0)
            or not np.all(np.isfinite(directions))
        ):
            status = "numerical_failure"
            break

        sum_logs += np.log(np.maximum(residual_norms, np.finfo(float).tiny) / delta)
        elapsed = block * t_clone
        running = sum_logs / elapsed
        times.append(elapsed)
        convergence.append(running.copy())
        if return_history:
            block_history.append(
                {
                    "block": block,
                    "time": elapsed,
                    "fiducial_state": fiducial_end.tolist(),
                    "residual_norms": residual_norms.tolist(),
                    "exponents": running.tolist(),
                }
            )
        state = fiducial_end
        direction_basis = directions

    if status == "ok":
        exponents = sum_logs / (k_blocks * t_clone)
    else:
        exponents = np.full(dimension, np.nan, dtype=float)

    order_class = classify_component_orders(component_orders)
    method_id = (
        "fractional_cloned_dynamics_abm_gs_published"
        if method != "qr"
        else "fractional_cloned_dynamics_abm_qr"
    )
    return ClonedDynamicsResult(
        exponents=np.asarray(exponents, dtype=float),
        times=np.asarray(times, dtype=float),
        convergence=np.asarray(convergence, dtype=float).reshape(-1, dimension),
        status=status,
        method_id=method_id,
        derivative_model="integer" if order_class == "integer" else "caputo",
        q=float(component_orders[0]),
        orthonormalization=method,
        bounded_trajectory=bounded_trajectory,
        method_metadata={
            "no_jacobian_required": True,
            "memory_protocol": memory_protocol,
            "orthonormalization": method,
            "integration_mode": integration_mode,
            "delta": delta,
            "t_clone": t_clone,
            "k_blocks": k_blocks,
            "h": h,
            "orders": component_orders.tolist(),
            "order_class": order_class,
            "incommensurate_fractional_experimental": order_class == "incommensurate_fractional",
            "finite_time": True,
            "system_id": system_id,
            "n_clones": n_clones,
            "random_seed": random_seed,
            "history": block_history if return_history else None,
        },
    )


__all__ = [
    "ClonedDynamicsResult",
    "FISCHER_2020_REFERENCE",
    "compute_cloned_dynamics_spectrum",
]
