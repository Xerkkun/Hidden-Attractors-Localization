#!/usr/bin/env python
"""Bandt--Pompe diagnostics for integer and fractional oscillator samples.

The public simulation facades generate a classical harmonic oscillator and a
componentwise Caputo counterpart.  Each ``SimulationResult`` is adapted to the
common ``TrajectoryInput`` contract before the position coordinate is passed
to HAFO's permutation-entropy analysis.

The two finite values are descriptive diagnostics of their recorded scalar
projections.  Sharing a vector field, sampling grid, and ordinal parameters
does not make the ordinary and hereditary problems physically equivalent, and
neither value proves chaos, attraction, or hiddenness.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from hidden_attractors import (
    ChaoticSystem,
    SimulationResult,
    TrajectoryInput,
    permutation_entropy,
    simulate,
    simulate_fractional,
)
from hidden_attractors.fractional import FractionalProblem


STEP_SIZE = 0.05
DURATION = 4.0
FRACTIONAL_ORDER = 0.85
EMBEDDING_DIMENSION = 4
DELAY = 2
TIE_POLICY = "stable_index"
LOG_BASE = 2.0
ANALYSIS_BACKEND = "python"
OBSERVABLE = "position"


def _oscillator_rhs(
    state: np.ndarray,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    """Return the unit-mass harmonic-oscillator vector field."""

    angular_frequency = float(parameters["angular_frequency"])
    position, velocity = np.asarray(state, dtype=float)
    return np.array(
        [velocity, -(angular_frequency**2) * position],
        dtype=float,
    )


def _oscillator() -> ChaoticSystem:
    """Build one public system definition shared by both solver facades."""

    return ChaoticSystem(
        name="harmonic-oscillator-permutation-entropy-example",
        dimension=2,
        rhs=_oscillator_rhs,
        parameters={"angular_frequency": 1.0},
        initial_state=(1.0, 0.0),
        state_names=("position", "velocity"),
        description=(
            "Unit-mass undamped harmonic oscillator used for a finite "
            "permutation-entropy API demonstration."
        ),
        tags=("example", "integer", "fractional"),
    )


def _as_trajectory(
    simulation: SimulationResult,
    *,
    example_role: str,
) -> TrajectoryInput:
    """Adapt a public simulation result and add example-level provenance."""

    return TrajectoryInput.from_simulation_result(
        simulation,
        projection=("position", "velocity"),
        metadata={
            "example_role": example_role,
            "physical_model": "unit-mass undamped harmonic oscillator",
            "selected_observable": OBSERVABLE,
            "transient_handling": "none; full short trajectory retained",
            "evidence_boundary": (
                "finite scalar-projection diagnostic; no chaos, attraction, "
                "or hiddenness claim"
            ),
        },
    )


def _trajectory_record(trajectory: TrajectoryInput) -> dict[str, Any]:
    """Return the compact JSON-safe provenance used by the example."""

    serialized = trajectory.to_serializable()
    history = serialized["lower_terminal_and_prehistory"]
    solver = serialized["solver_and_tolerances"]
    return {
        "sample_count": trajectory.sample_count,
        "dimension": trajectory.dimension,
        "system_kind": trajectory.system_kind,
        "time_coordinate": trajectory.time_coordinate,
        "sampled_uniformly": bool(trajectory.sampled_uniformly),
        "uniform_step": trajectory.uniform_step,
        "projection": list(trajectory.projection),
        "derivative_definition": trajectory.derivative_definition,
        "order": serialized["order"],
        "memory_policy": trajectory.memory_policy,
        "prehistory_kind": history["kind"],
        "lower_terminal": history["lower_terminal"],
        "solver_method": solver["method"],
        "solver_backend": solver["backend"],
        "metadata": serialized["metadata"],
        "fingerprint": trajectory.fingerprint(),
    }


def _entropy_record(trajectory: TrajectoryInput) -> dict[str, Any]:
    """Evaluate one scalar component under the shared ordinal contract."""

    result = permutation_entropy(
        trajectory,
        component=OBSERVABLE,
        embedding_dimension=EMBEDDING_DIMENSION,
        delay=DELAY,
        tie_policy=TIE_POLICY,
        log_base=LOG_BASE,
        backend=ANALYSIS_BACKEND,
        fallback=False,
    )
    distribution = result.distribution
    envelope = result.as_analysis_result().to_serializable()
    return {
        "entropy": float(result.entropy),
        "normalized_entropy": float(result.normalized_entropy),
        "maximum_entropy": float(result.maximum_entropy),
        "log_base": float(result.log_base),
        "estimator": result.estimator,
        "normalization": result.normalization,
        "sample_count": distribution.sample_count,
        "total_windows": distribution.total_windows,
        "valid_windows": distribution.valid_windows,
        "tied_windows": distribution.tied_windows,
        "omitted_windows": distribution.omitted_windows,
        "possible_patterns": distribution.possible_patterns,
        "observed_patterns": distribution.observed_patterns,
        "missing_patterns": distribution.missing_patterns,
        "counts": distribution.counts.tolist(),
        "probabilities": distribution.probabilities.tolist(),
        "requested_backend": distribution.requested_backend,
        "backend": distribution.backend,
        "sampling": distribution.sampling,
        "projection": distribution.projection,
        "trajectory_fingerprint": result.trajectory_fingerprint,
        "trajectory_system_kind": distribution.trajectory_system_kind,
        "derivative_definition": distribution.derivative_definition,
        "memory_policy": distribution.memory_policy,
        "status": result.status,
        "evidence_scope": result.evidence_scope,
        "warnings": list(result.warnings),
        "references": list(result.references),
        "analysis_envelope": {
            "method": envelope["method"],
            "backend": envelope["backend"],
            "status": envelope["status"],
            "evidence_scope": envelope["evidence_scope"],
            "trajectory_fingerprint": envelope["trajectory_fingerprint"],
        },
    }


def run_example() -> dict[str, Any]:
    """Run two deterministic simulations and return a JSON-safe record."""

    system = _oscillator()
    integer_simulation = simulate(
        system,
        step_size=STEP_SIZE,
        duration=DURATION,
        method="rk4",
        use_acceleration=False,
        divergence_norm=None,
    )
    fractional_problem = FractionalProblem(
        derivative="caputo",
        method="caputo_abm_pece",
        orders=FRACTIONAL_ORDER,
        initial_state=system.initial_state,
        step=STEP_SIZE,
        t_span=(0.0, DURATION),
        memory_policy="full_history",
        problem_id="permutation-entropy-fractional-oscillator-example",
    )
    fractional_simulation = simulate_fractional(
        system,
        fractional_problem,
        use_acceleration=False,
        allow_python_fallback=True,
        divergence_norm=None,
    )
    if integer_simulation.status != "ok" or fractional_simulation.status != "ok":
        raise RuntimeError(
            "trajectory generation failed: "
            f"integer={integer_simulation.status}, "
            f"fractional={fractional_simulation.status}"
        )

    integer_trajectory = _as_trajectory(
        integer_simulation,
        example_role="ordinary q=1 reference trajectory",
    )
    fractional_trajectory = _as_trajectory(
        fractional_simulation,
        example_role="Caputo q=0.85 hereditary trajectory",
    )
    integer_entropy = _entropy_record(integer_trajectory)
    fractional_entropy = _entropy_record(fractional_trajectory)

    return {
        "system": system.name,
        "scope": "finite_sample_empirical_ordinal_pattern_diagnostic",
        "analysis_contract": {
            "observable": OBSERVABLE,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "delay_samples": DELAY,
            "window_order": "forward_x_s_plus_k_delay",
            "tie_policy": TIE_POLICY,
            "ordinal_encoding": "zero_based_lexicographic_lehmer_rank",
            "log_base": LOG_BASE,
            "estimator": "relative_frequency_plugin_shannon_entropy",
            "backend": ANALYSIS_BACKEND,
        },
        "integer": {
            "declared_order": 1.0,
            "derivative": "ordinary_first_derivative",
            "trajectory_status": integer_simulation.status,
            "trajectory": _trajectory_record(integer_trajectory),
            "permutation_entropy": integer_entropy,
        },
        "fractional": {
            "declared_order": FRACTIONAL_ORDER,
            "derivative": fractional_problem.derivative,
            "method": fractional_problem.method,
            "memory_policy": fractional_problem.memory_policy,
            "trajectory_status": fractional_simulation.status,
            "trajectory": _trajectory_record(fractional_trajectory),
            "permutation_entropy": fractional_entropy,
        },
        "comparison_policy": (
            "Same observable and ordinal parameters are used for an API-level "
            "comparison only; ordinary and hereditary trajectories are not "
            "asserted to be physically equivalent."
        ),
        "claims": (
            "Finite projected-series diagnostics only. The reported plugin "
            "entropies are not entropy rates. They are not proof of chaos, "
            "attraction, or hiddenness, and the fractional position coordinate "
            "is not the complete hereditary state."
        ),
    }


def main() -> None:
    """Print the complete example record as strict JSON."""

    print(json.dumps(run_example(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
