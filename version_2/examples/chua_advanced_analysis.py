#!/usr/bin/env python
"""Apply shared integer/fractional-ready diagnostics to finite dynamical data.

Delay reconstruction and RQA use an integer Chua trajectory.  Basin entropy
and uncertainty use the exactly classified bistable flow ``x'=x-x^3,
y'=-y``.  The latter has two asymptotic basins separated by ``x=0``.  These
finite diagnostics do not establish chaos, Wada boundaries, or hiddenness.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from hidden_attractors.analysis.basin_uncertainty import (
    basin_entropy,
    estimate_uncertainty_exponent,
    uncertainty_fraction,
)
from hidden_attractors.analysis.delay_embedding import (
    estimate_delay_autocorrelation,
    estimate_delay_mutual_information,
    false_nearest_neighbors,
    generalized_delay_embedding,
)
from hidden_attractors.analysis.recurrence_advanced import (
    auto_recurrence_matrix,
    cross_recurrence_matrix,
    joint_recurrence_matrix,
    recurrence_quantification_advanced,
)
from hidden_attractors.solvers.integer import dop853_q1_integrate
from hidden_attractors.systems.builtins import chua_system


def _standardize(values: np.ndarray) -> np.ndarray:
    scale = np.std(values, axis=0)
    scale[scale == 0.0] = 1.0
    return (values - np.mean(values, axis=0)) / scale


def _bistable_labels(x_coordinates: np.ndarray) -> np.ndarray:
    """Exact asymptotic classes for x'=x-x^3 away from x=0."""

    return (np.asarray(x_coordinates) > 0.0).astype(np.int64)


def run_example() -> dict[str, Any]:
    """Return a compact record exercising basin, delay and recurrence APIs."""

    system = chua_system("nonsmooth")
    trajectory, status = dop853_q1_integrate(
        lambda state: system.evaluate(state),
        np.array([0.1, 0.0, 0.0]),
        t_final=8.0,
        h=0.01,
        max_step=0.01,
        div_threshold=120.0,
    )
    if status != "ok":
        raise RuntimeError(f"integer Chua example did not finish: {status}")
    times = trajectory[200:, 0]
    states = trajectory[200:, 1:]

    autocorrelation = estimate_delay_autocorrelation(
        states,
        column=0,
        max_lag=80,
        times=times,
        time_unit="model_time",
    )
    mutual_information = estimate_delay_mutual_information(
        states,
        column=0,
        max_lag=60,
        bins=24,
        minimum_pairs=64,
        times=times,
        time_unit="model_time",
    )
    selected_delay = mutual_information.lag_samples or autocorrelation.lag_samples or 1
    selected_delay = max(1, int(selected_delay))
    fnn = false_nearest_neighbors(
        states,
        column=0,
        delay=selected_delay,
        max_dimension=5,
        theiler_window=selected_delay,
        minimum_valid_neighbors=32,
        times=times,
        time_unit="model_time",
    )
    embedding = generalized_delay_embedding(
        states,
        [(0, 0), (0, selected_delay), (0, 2 * selected_delay)],
        times=times,
        time_unit="model_time",
    )
    points = _standardize(embedding.vectors)

    auto = auto_recurrence_matrix(
        points,
        target_rate=0.04,
        theiler_window=selected_delay,
        block_rows=64,
        projection="standardized three-coordinate delay embedding of Chua x",
    )
    rqa = recurrence_quantification_advanced(
        auto,
        min_diagonal=2,
        min_vertical=2,
        trend_border=10,
    )
    midpoint = points.shape[0] // 2
    cross = cross_recurrence_matrix(
        points[:midpoint],
        points[midpoint:],
        target_rate=0.04,
        block_rows=64,
        projection="first versus second half of the same Chua embedding",
    )
    normalized_x = _standardize(states[:, [0]])
    normalized_y = _standardize(states[:, [1]])
    joint = joint_recurrence_matrix(
        normalized_x,
        normalized_y,
        radius=(0.2, 0.2),
        theiler_window=selected_delay,
        block_rows=64,
        projection="separate standardized Chua x and y observables",
    )

    coarse_axis = np.linspace(-1.45, 1.45, 30)
    coarse_x, _ = np.meshgrid(coarse_axis, coarse_axis, indexing="xy")
    basin_labels = _bistable_labels(coarse_x)
    basin = basin_entropy(basin_labels, box_size=6)

    fine_axis = np.linspace(-1.5, 1.5, 241)
    fine_x, _ = np.meshgrid(fine_axis, fine_axis, indexing="xy")
    reference_labels = _bistable_labels(fine_x)
    declared_scale = 0.1
    perturbed_labels = _bistable_labels(fine_x + declared_scale)
    uncertainty = uncertainty_fraction(
        reference_labels,
        perturbed_labels,
        perturbation_scale=declared_scale,
        scale_units="bistable_x_coordinate",
        perturbation_norm="euclidean",
        perturbation_direction=(1.0, 0.0),
    )
    scales = np.array([0.05, 0.1, 0.2, 0.4])
    fractions = np.array(
        [
            np.mean(reference_labels != _bistable_labels(fine_x + scale))
            for scale in scales
        ],
        dtype=float,
    )
    exponent = estimate_uncertainty_exponent(
        scales,
        fractions,
        sampling_space_dimension=2.0,
    )

    return {
        "scope": "finite_trajectory_and_finite_grid_diagnostics",
        "chua": {
            "system": system.name,
            "integer_solver_status": status,
            "samples_after_transient": int(states.shape[0]),
            "delay": {
                "autocorrelation_samples": autocorrelation.lag_samples,
                "mutual_information_samples": mutual_information.lag_samples,
                "selected_samples": selected_delay,
                "fnn_selected_dimension": fnn.selected_dimension,
                "embedding_shape": list(embedding.vectors.shape),
            },
            "recurrence": {
                "auto_rate": auto.achieved_recurrence_rate,
                "cross_rate": cross.achieved_recurrence_rate,
                "joint_rate": joint.achieved_recurrence_rate,
                "determinism": rqa.determinism,
                "laminarity": rqa.laminarity,
                "trend": rqa.trend,
                "normalized_absolute_trend": rqa.normalized_absolute_trend,
            },
        },
        "bistable_flow": {
            "equations": ["dx/dt=x-x^3", "dy/dt=-y"],
            "basin_entropy": basin.basin_entropy,
            "boundary_basin_entropy": basin.boundary_basin_entropy,
            "log_two_criterion_applicable": basin.log_two_criterion_applicable,
            "log_two_criterion_reason": basin.log_two_criterion_reason,
            "uncertainty_fraction_at_0.1": uncertainty.fraction,
            "uncertainty_interval": list(uncertainty.confidence_interval),
            "uncertainty_exponent": exponent.exponent,
            "estimated_boundary_dimension": exponent.estimated_boundary_dimension,
            "scales": scales.tolist(),
            "fractions": fractions.tolist(),
        },
        "claims": (
            "The Chua values are finite sampled diagnostics. The bistable labels "
            "use the known separatrix x=0; neither result proves hiddenness or Wada."
        ),
    }


def main() -> None:
    print(json.dumps(run_example(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

