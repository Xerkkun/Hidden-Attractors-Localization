"""Cloned-dynamics finite-time Lyapunov spectrum estimators.

The legacy protocol chains short fiducial endpoints. The versioned Fischer
reference-rail protocol instead anchors each short fiducial/clone block to a
separately integrated reference. Clone memory is restarted in both protocols;
neither is a full-memory Caputo variational exponent estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Callable, Mapping

import numpy as np

from .._rhs import bind_rhs
from .._time_grid import exact_fixed_step_count
from ..integrations.abm_fractional import (
    classify_component_orders,
    integrate_fractional_abm,
    normalize_component_orders,
)
from ..integrations.rk4 import rk4_integrate
from ..integrations.abm_reference import integrate_reference_abm

LEGACY_EXECUTION_PROTOCOL = "legacy_block_restart_v1"
REFERENCE_RAIL_PROTOCOL = "fischer_reference_rail_v1"
CLONED_EXECUTION_PROTOCOLS = (LEGACY_EXECUTION_PROTOCOL, REFERENCE_RAIL_PROTOCOL)

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
    reference_times: np.ndarray | None = None
    reference_states: np.ndarray | None = None


def _sample_reference_rail(
    times: np.ndarray, states: np.ndarray, physical_times: np.ndarray,
    *, sampling: str,
) -> np.ndarray:
    """Select rail states by physical time; never confuse reference/clone indices."""
    if sampling not in {"grid_exact", "linear"}:
        raise ValueError("reference_sampling must be 'grid_exact' or explicit 'linear'")
    if (times.ndim != 1 or len(times) < 2 or states.ndim != 2
            or len(states) != len(times) or not np.all(np.isfinite(times))
            or not np.all(np.isfinite(states)) or np.any(np.diff(times) <= 0)):
        raise ValueError("reference rail must have finite states and strictly increasing times")
    tolerance = 64 * np.finfo(float).eps * max(1.0, abs(times[-1]))
    if abs(times[0]) > tolerance or physical_times[-1] > times[-1] + tolerance:
        raise ValueError("reference rail must cover the complete physical horizon from zero")
    if sampling == "linear":
        return np.column_stack([np.interp(physical_times, times, column) for column in states.T])
    right = np.clip(np.searchsorted(times, physical_times), 0, len(times) - 1)
    left = np.maximum(0, right - 1)
    indices = np.where(abs(times[left] - physical_times) < abs(times[right] - physical_times), left, right)
    if np.any(abs(times[indices] - physical_times) > tolerance):
        raise ValueError("clone boundary misses reference grid; declare linear sampling explicitly")
    return states[indices].copy()


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
    execution_protocol: str = LEGACY_EXECUTION_PROTOCOL,
    reference_h: float | None = None,
    reference_times: np.ndarray | None = None,
    reference_states: np.ndarray | None = None,
    reference_sampling: str = "grid_exact",
    reference_convolution_block_size: int = 256,
) -> ClonedDynamicsResult:
    """Estimate a spectrum from perturbed clones without a Jacobian.

    The number of clones is the state dimension.  ``n_clones`` is explicit in
    the API so callers cannot accidentally confuse it with the number of ABM
    steps inside a clone interval.
    """

    state = np.asarray(x0, dtype=float).reshape(-1)
    initial_state = state.copy()
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
    n_steps = exact_fixed_step_count(
        h,
        t_clone,
        caller="compute_cloned_dynamics_spectrum(t_clone)",
    )
    if n_steps < 1:
        raise ValueError("t_clone must span at least one fixed step.")
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
    if execution_protocol not in CLONED_EXECUTION_PROTOCOLS:
        raise ValueError("unsupported cloned-dynamics execution protocol")
    use_reference_rail = execution_protocol == REFERENCE_RAIL_PROTOCOL
    if not use_reference_rail and (reference_times is not None or reference_states is not None or reference_h is not None):
        raise ValueError("reference arguments require the reference-rail execution protocol")
    if use_reference_rail and integration_mode != "fractional_abm":
        raise ValueError("reference-rail V1 uses fractional_abm for clone blocks")
    requested_memory_protocol = str(memory_protocol)
    effective_memory_protocol = (
        "not_applicable_integer_rk4_reference"
        if integration_mode == "integer_rk4_reference"
        else "published_block_restart"
    )

    direction_basis = np.eye(dimension, dtype=float)
    sum_logs = np.zeros(dimension, dtype=float)
    convergence: list[np.ndarray] = []
    times: list[float] = []
    block_history: list[dict[str, object]] = []
    bounded_trajectory = True
    status = "ok"
    bound_rhs = bind_rhs(rhs, parameters)
    rail_times = rail_states = rail_anchors = None
    reference_metadata: dict[str, object] = {}
    if use_reference_rail:
        if (reference_times is None) != (reference_states is None):
            raise ValueError("provide both reference_times and reference_states, or neither")
        if reference_times is None:
            reference_h = h if reference_h is None else float(reference_h)
            rail_times, rail_states, reference_status = integrate_reference_abm(
                bound_rhs, initial_state, component_orders, reference_h, k_blocks * t_clone,
                convolution_block_size=reference_convolution_block_size,
                divergence_norm=divergence_norm,
            )
            reference_source = "generated_fixed_lower_limit_pece_componentwise_block_fft_v1"
        else:
            rail_times = np.asarray(reference_times, dtype=float)
            rail_states = np.asarray(reference_states, dtype=float)
            reference_status = "ok"
            reference_source = "provided_reference_trajectory"
        if reference_status != "ok":
            status = "reference_numerical_failure"
            bounded_trajectory = reference_status != "diverged"
        else:
            if rail_states.ndim != 2 or rail_states.shape[1] != dimension:
                raise ValueError("reference state dimension must match x0")
            rail_anchors = _sample_reference_rail(
                rail_times, rail_states, np.arange(k_blocks + 1) * t_clone,
                sampling=reference_sampling,
            )
            if not np.allclose(rail_anchors[0], initial_state, rtol=0, atol=64 * np.finfo(float).eps):
                raise ValueError("reference rail must begin at x0")
        reference_metadata = {
            "source": reference_source, "status": reference_status,
            "memory_protocol": ("fixed_lower_limit_full_history" if reference_source != "provided_reference_trajectory"
                                else "provided_trajectory_memory_policy_not_verified"), "lower_limit": 0.0,
            "fft_blocks_restart_memory": False,
            "convolution_block_size": reference_convolution_block_size,
            "h": reference_h, "sampling": reference_sampling,
            "anchor_time_rule": "physical_time_k_times_t_clone",
            "samples": len(rail_times),
            "times_float64_le_sha256": hashlib.sha256(np.asarray(rail_times, dtype="<f8").tobytes()).hexdigest(),
            "states_float64_le_sha256": hashlib.sha256(np.asarray(rail_states, dtype="<f8").tobytes()).hexdigest(),
            "provided_memory_policy_verified_by_solver": reference_source != "provided_reference_trajectory",
        }

    for block in range(1, k_blocks + 1):
        if status != "ok":
            break
        block_start = (block - 1) * t_clone
        if use_reference_rail:
            state = rail_anchors[block - 1].copy()
        initial_copies = np.vstack((state, state + delta * direction_basis.T))
        augmented_orders = np.tile(component_orders, dimension + 1)

        def augmented_rhs(t: float, flattened: np.ndarray) -> np.ndarray:
            copies = np.asarray(flattened, dtype=float).reshape(dimension + 1, dimension)
            derivatives = []
            for copy in copies:
                value = bound_rhs(t + block_start if use_reference_rail else t, copy)
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
                memory_protocol=effective_memory_protocol,
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
                    **({"reference_anchor_state": state.tolist(),
                        "reference_anchor_time": block_start} if use_reference_rail else {}),
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
    if use_reference_rail:
        method_id += "_reference_rail_v1"
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
        reference_times=rail_times,
        reference_states=rail_states,
        method_metadata={
            "execution_protocol": execution_protocol,
            "reference_rail": reference_metadata if use_reference_rail else None,
            "clone_memory_protocol": effective_memory_protocol,
            "clone_lower_limit": "each_block_start",
            "no_jacobian_required": True,
            "memory_protocol": effective_memory_protocol,
            "requested_memory_protocol": requested_memory_protocol,
            "effective_memory_protocol": effective_memory_protocol,
            "requested_protocol_is_alias": (
                requested_memory_protocol != effective_memory_protocol
            ),
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
    "CLONED_EXECUTION_PROTOCOLS", "LEGACY_EXECUTION_PROTOCOL", "REFERENCE_RAIL_PROTOCOL",
    "compute_cloned_dynamics_spectrum",
]
