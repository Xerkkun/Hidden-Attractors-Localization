"""Finite-time SALI/GALI comparison on the Hénon--Heiles Hamiltonian.

The example compares the q=1 variational and multi-particle propagation
routes on the same deterministic initial deviations.  It is a numerical API
example, not a classification of the selected orbit and not evidence of an
attractor or hiddenness.

References
----------
Skokos, Bountis & Antonopoulos (2007), doi:10.1016/j.physd.2007.04.004.
Manda, Hillebrand & Skokos (2025), doi:10.1016/j.cnsns.2025.108635.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from hidden_attractors import integer_flow_alignment_indices


INITIAL_STATE = np.array([0.0, 0.1, 0.35, 0.0], dtype=float)
GALI_ORDERS = (2, 3, 4)
SEED = 20260803


def henon_heiles_rhs(state: np.ndarray) -> np.ndarray:
    """Return ``(xdot, ydot, pxdot, pydot)`` for Hénon--Heiles."""

    x, y, px, py = np.asarray(state, dtype=float)
    return np.array(
        [
            px,
            py,
            -x - 2.0 * x * y,
            -y - x * x + y * y,
        ],
        dtype=float,
    )


def henon_heiles_jacobian(state: np.ndarray) -> np.ndarray:
    """Return the analytic Jacobian of :func:`henon_heiles_rhs`."""

    x, y, _px, _py = np.asarray(state, dtype=float)
    return np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0 - 2.0 * y, -2.0 * x, 0.0, 0.0],
            [-2.0 * x, -1.0 + 2.0 * y, 0.0, 0.0],
        ],
        dtype=float,
    )


def henon_heiles_energy(states: np.ndarray) -> np.ndarray:
    """Evaluate the conserved Hamiltonian on one state or a state matrix."""

    values = np.asarray(states, dtype=float)
    x = values[..., 0]
    y = values[..., 1]
    px = values[..., 2]
    py = values[..., 3]
    return (
        0.5 * (px * px + py * py + x * x + y * y)
        + x * x * y
        - (y * y * y) / 3.0
    )


def _record(result: Any) -> dict[str, Any]:
    if result.status != "ok" or result.sampled_states is None:
        raise RuntimeError(
            f"alignment computation failed: {result.status}: {result.error_message}"
        )
    energies = henon_heiles_energy(result.sampled_states)
    energy_scale = max(abs(float(energies[0])), np.finfo(float).tiny)
    relative_energy_drift = float(
        np.max(np.abs(energies - energies[0])) / energy_scale
    )
    return {
        "method_id": result.method_id,
        "evolution_method": result.evolution_method,
        "backend": result.backend,
        "volume_method": result.volume_method,
        "sample_count": int(result.coordinates.size),
        "final_time": float(result.coordinates[-1]),
        "final_sali": float(result.sali[-1]),
        "final_gali": {
            str(int(order)): float(value)
            for order, value in zip(result.gali_orders, result.gali[-1], strict=True)
        },
        "minimum_gali": {
            str(int(order)): float(value)
            for order, value in zip(
                result.gali_orders,
                np.min(result.gali, axis=0),
                strict=True,
            )
        },
        "relative_energy_drift": relative_energy_drift,
        "censored_cells": int(np.count_nonzero(result.censored)),
        "jacobian_source": result.metadata["jacobian_source"],
    }


def run_example(*, duration: float = 4.0) -> dict[str, Any]:
    """Run both propagation methods and return a strict JSON-safe record."""

    shared = {
        "t_final": float(duration),
        "renormalization_time": 0.25,
        "gali_orders": GALI_ORDERS,
        "n_vectors": 4,
        "seed": SEED,
        "rtol": 1.0e-10,
        "atol": 1.0e-12,
        "max_step": 0.02,
        "backend": "auto",
        "q": 1.0,
    }
    variational = integer_flow_alignment_indices(
        henon_heiles_rhs,
        henon_heiles_jacobian,
        INITIAL_STATE,
        method="variational",
        **shared,
    )
    multi_particle = integer_flow_alignment_indices(
        henon_heiles_rhs,
        None,
        INITIAL_STATE,
        method="multi_particle",
        deviation_size=float(np.sqrt(np.finfo(float).eps)),
        **shared,
    )
    if variational.status != "ok" or multi_particle.status != "ok":
        raise RuntimeError(
            "Hénon--Heiles comparison failed: "
            f"variational={variational.status}, "
            f"multi_particle={multi_particle.status}"
        )
    if not np.array_equal(variational.coordinates, multi_particle.coordinates):
        raise RuntimeError("the two propagation methods produced different sample grids")

    return {
        "system": "Henon-Heiles Hamiltonian",
        "state_order": ["x", "y", "px", "py"],
        "initial_state": INITIAL_STATE.tolist(),
        "initial_energy": float(henon_heiles_energy(INITIAL_STATE)),
        "order": 1.0,
        "derivative": "ordinary_first_derivative",
        "gali_orders": list(GALI_ORDERS),
        "variational": _record(variational),
        "multi_particle": _record(multi_particle),
        "finite_window_comparison": {
            "maximum_absolute_sali_difference": float(
                np.max(np.abs(variational.sali - multi_particle.sali))
            ),
            "maximum_absolute_gali_difference": float(
                np.max(np.abs(variational.gali - multi_particle.gali))
            ),
        },
        "evidence_scope": (
            "finite-time q=1 alignment-index implementation comparison; no "
            "automatic orbit classification, attraction, hiddenness, or "
            "fractional-memory claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=4.0)
    args = parser.parse_args()
    print(json.dumps(run_example(duration=args.duration), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

