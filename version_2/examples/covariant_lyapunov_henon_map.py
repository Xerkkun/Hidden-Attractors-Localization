"""Finite-time covariant Lyapunov vectors for the Hénon map.

This q=1 example exercises the nonlinear map facade, the Ginelli backward
recursion and the unoriented angle postprocessor.  Its output is finite
numerical evidence, not an automatic claim of chaos, attraction or hiddenness.

References
----------
Ginelli et al. (2007), doi:10.1103/PhysRevLett.99.130601.
Kuptsov & Parlitz (2012), doi:10.1007/s00332-012-9126-5.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from hidden_attractors import (
    covariant_lyapunov_angles,
    integer_map_covariant_lyapunov_vectors,
)


A = 1.4
B = 0.3
INITIAL_STATE = np.array([0.0, 0.0], dtype=float)
SEED = 20260803


def henon_map(state: np.ndarray) -> np.ndarray:
    """Return one classical Hénon-map iterate."""

    x, y = np.asarray(state, dtype=float)
    return np.array([1.0 - A * x * x + y, B * x], dtype=float)


def henon_map_jacobian(state: np.ndarray) -> np.ndarray:
    """Return the analytic Jacobian at the pre-iterate state."""

    x = float(np.asarray(state, dtype=float)[0])
    return np.array([[-2.0 * A * x, 1.0], [B, 0.0]], dtype=float)


def _line_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    cosine = float(np.clip(abs(first @ second), 0.0, 1.0))
    return float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))


def _covariance_residual(states: np.ndarray, vectors: np.ndarray) -> float:
    residual = 0.0
    for sample in range(vectors.shape[0] - 1):
        tangent = henon_map_jacobian(states[sample])
        for vector in range(vectors.shape[1]):
            residual = max(
                residual,
                _line_distance(
                    tangent @ vectors[sample, vector],
                    vectors[sample + 1, vector],
                ),
            )
    return float(residual)


def run_example(
    *,
    iterations: int = 1000,
    transient_iterations: int = 1000,
    forward_transient_iterations: int = 500,
    backward_transient_iterations: int = 500,
    backend: str = "auto",
) -> dict[str, Any]:
    """Compute a reproducible CLV record for the Hénon map."""

    result = integer_map_covariant_lyapunov_vectors(
        henon_map,
        henon_map_jacobian,
        INITIAL_STATE,
        iterations=iterations,
        transient_iterations=transient_iterations,
        forward_transient_iterations=forward_transient_iterations,
        backward_transient_iterations=backward_transient_iterations,
        n_vectors=2,
        seed=SEED,
        backend=backend,
        q=1.0,
    )
    if result.status != "ok":
        raise RuntimeError(f"Hénon CLV failed: {result.status}: {result.error_message}")
    angles = covariant_lyapunov_angles(
        result.vectors,
        coordinates=result.coordinates,
        pairs=((0, 1),),
        unoriented=True,
    )
    pair_angles = angles.pair_angles[:, 0]
    return {
        "system": "Henon map",
        "parameters": {"a": A, "b": B},
        "initial_state": INITIAL_STATE.tolist(),
        "order": 1.0,
        "derivative_model": "integer_discrete_map",
        "method_id": result.method_id,
        "backend": result.backend,
        "propagation_backend": result.propagation_backend,
        "sample_count": int(result.coordinates.size),
        "observation_iterations": int(iterations),
        "state_transient_iterations": int(transient_iterations),
        "forward_transient_iterations": int(forward_transient_iterations),
        "backward_transient_iterations": int(backward_transient_iterations),
        "finite_time_exponents_per_iteration": result.exponents.tolist(),
        "clv_pair_angle_radians": {
            "minimum": float(np.min(pair_angles)),
            "median": float(np.median(pair_angles)),
            "maximum": float(np.max(pair_angles)),
        },
        "maximum_projective_covariance_residual": _covariance_residual(
            result.sampled_states, result.vectors
        ),
        "near_degenerate_finite_time_spectrum": bool(
            result.metadata["near_degenerate_finite_time_spectrum"]
        ),
        "auto_transient_stopping": bool(result.metadata["auto_transient_stopping"]),
        "evidence_scope": (
            "finite q=1 nonlinear-map CLV and angle diagnostic; no automatic "
            "chaos, attraction, hiddenness, hyperbolicity, or fractional-memory claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--transient-iterations", type=int, default=1000)
    parser.add_argument("--forward-transient-iterations", type=int, default=500)
    parser.add_argument("--backward-transient-iterations", type=int, default=500)
    parser.add_argument("--backend", choices=("auto", "numpy", "numba"), default="auto")
    args = parser.parse_args()
    record = run_example(
        iterations=args.iterations,
        transient_iterations=args.transient_iterations,
        forward_transient_iterations=args.forward_transient_iterations,
        backward_transient_iterations=args.backward_transient_iterations,
        backend=args.backend,
    )
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

