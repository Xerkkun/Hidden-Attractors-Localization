#!/usr/bin/env python
"""Compare finite correlation-dimension diagnostics for two trajectory types.

The same linear oscillator is integrated once as an ordinary differential
equation and once as a Caputo fractional differential equation.  The radii,
Theiler window and regression interval are all caller-specified.  The reported
slopes characterize only these finite, standardized projections; they are not
proof of chaos, attraction or hiddenness.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from hidden_attractors import simulate, simulate_fractional
from hidden_attractors.analysis import estimate_correlation_dimension
from hidden_attractors.fractional import FractionalProblem
from hidden_attractors.systems import ChaoticSystem


RADII = np.array([0.08, 0.12, 0.18, 0.27, 0.40, 0.60, 0.90, 1.35, 2.00])
FIT_RADIUS_RANGE = (0.27, 0.90)
THEILER_WINDOW = 2
TRANSIENT_SAMPLES = 10


def _oscillator() -> ChaoticSystem:
    """Return ``x'=-y, y'=x`` with one shared integer/fractional contract."""

    return ChaoticSystem(
        name="linear-oscillator-correlation-dimension-example",
        dimension=2,
        rhs=lambda state, _parameters: np.array([-state[1], state[0]]),
        initial_state=(1.0, 0.0),
        description="Linear oscillator used only for a finite API demonstration.",
    )


def _standardize(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    scale = np.std(values, axis=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise RuntimeError("trajectory coordinates must have finite positive scales")
    return (values - np.mean(values, axis=0)) / scale


def _diagnostic_record(
    states: np.ndarray,
    *,
    sampling: str,
    projection: str,
) -> dict[str, Any]:
    points = _standardize(states[TRANSIENT_SAMPLES:])
    result = estimate_correlation_dimension(
        points,
        RADII,
        fit_radius_range=FIT_RADIUS_RANGE,
        minimum_points=3,
        theiler_window=THEILER_WINDOW,
        metric="euclidean",
        backend="python",
        fallback=False,
        sampling=sampling,
        projection=projection,
    )
    curve = result.curve
    return {
        "samples_supplied": curve.sample_count,
        "feature_dimension": curve.feature_dimension,
        "metric": curve.metric,
        "backend": curve.backend,
        "eligible_pairs": curve.eligible_pairs,
        "counts": curve.counts.tolist(),
        "correlation_sums": curve.correlation_sums.tolist(),
        "fit_indices": result.fit_indices.tolist(),
        "dimension_slope": result.slope,
        "intercept": result.intercept,
        "r_squared": result.r_squared,
        "regression_standard_error": result.regression_standard_error,
        "sampling": curve.sampling,
        "projection": curve.projection,
        "evidence_scope": result.evidence_scope,
        "fractional_state_caveat": result.fractional_state_caveat,
    }


def run_example() -> dict[str, Any]:
    """Run two small deterministic simulations and return JSON-safe results."""

    system = _oscillator()
    integer = simulate(
        system,
        step_size=0.05,
        duration=4.0,
        method="rk4",
        use_acceleration=False,
        divergence_norm=None,
    )
    fractional_problem = FractionalProblem(
        derivative="caputo",
        method="caputo_abm_pece",
        orders=0.85,
        initial_state=system.initial_state,
        step=0.05,
        t_span=(0.0, 4.0),
        memory_policy="full_history",
        problem_id="correlation-dimension-fractional-example",
    )
    fractional = simulate_fractional(
        system,
        fractional_problem,
        use_acceleration=False,
        allow_python_fallback=True,
        divergence_norm=None,
    )
    if integer.status != "ok" or fractional.status != "ok":
        raise RuntimeError(
            "trajectory generation failed: "
            f"integer={integer.status}, fractional={fractional.status}"
        )

    integer_record = _diagnostic_record(
        integer.states,
        sampling="uniform physical time, dt=0.05; first 10 samples discarded",
        projection="standardized x and y coordinates of the q=1 trajectory",
    )
    integer_record.update(
        {
            "order": 1.0,
            "derivative": "ordinary_first_derivative",
            "method": integer.method,
            "trajectory_status": integer.status,
        }
    )
    fractional_record = _diagnostic_record(
        fractional.states,
        sampling="uniform physical time, dt=0.05; first 10 samples discarded",
        projection=(
            "standardized x and y coordinates of the Caputo q=0.85 trajectory"
        ),
    )
    fractional_record.update(
        {
            "order": 0.85,
            "derivative": fractional_problem.derivative,
            "method": fractional.method,
            "memory_policy": fractional_problem.memory_policy,
            "trajectory_status": fractional.status,
        }
    )

    return {
        "system": system.name,
        "scope": "finite_sample_empirical_trajectory_diagnostic",
        "analysis_contract": {
            "radii": RADII.tolist(),
            "fit_radius_range": list(FIT_RADIUS_RANGE),
            "fit_range_selection": "explicit_caller_supplied_inclusive_range",
            "minimum_points": 3,
            "theiler_window_samples": THEILER_WINDOW,
            "threshold": "distance_strictly_less_than_radius",
        },
        "integer": integer_record,
        "fractional": fractional_record,
        "claims": (
            "Finite projected-trajectory diagnostic only; the fitted slopes are "
            "not proof of chaos, attraction or hiddenness (ocultedad), and the "
            "fractional coordinates are not the complete hereditary state."
        ),
    }


def main() -> None:
    print(json.dumps(run_example(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
