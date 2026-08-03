from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.analysis.basin_uncertainty import (
    basin_entropy,
    estimate_uncertainty_exponent,
    uncertainty_fraction,
)


def test_homogeneous_basin_has_zero_entropy() -> None:
    result = basin_entropy(
        np.zeros((5, 7), dtype=int),
        box_size=(2, 3),
        partial_boxes="include_equal",
    )
    assert result.basin_entropy == 0.0
    assert result.boundary_basin_entropy == 0.0
    assert result.boundary_boxes == 0
    assert result.total_boxes == 9
    assert result.classified_samples == 35
    assert result.partial_box_policy == "include_equal"
    assert not result.boundary_entropy_defined
    assert not result.log_two_criterion_applicable
    assert result.log_two_criterion_reason == "no_boundary_boxes"


def test_checkerboard_boxes_recover_log_two_entropy() -> None:
    labels = np.indices((4, 4)).sum(axis=0) % 2
    result = basin_entropy(labels, box_size=2)
    assert result.basin_entropy == pytest.approx(np.log(2.0))
    assert result.boundary_basin_entropy == pytest.approx(np.log(2.0))
    assert result.boundary_box_fraction == 1.0
    assert not result.boundary_entropy_above_log_two
    assert result.boundary_entropy_defined
    assert not result.log_two_criterion_applicable
    assert result.log_two_criterion_reason == "fewer_than_three_observed_basins"

    bits = basin_entropy(labels, box_size=2, entropy_base=2.0)
    assert bits.basin_entropy == pytest.approx(1.0)
    assert bits.log_two_threshold == pytest.approx(1.0)


def test_ignored_labels_are_removed_and_empty_boxes_reported() -> None:
    labels = np.array([[0, 1, -1, -1], [0, 1, -1, -1]])
    result = basin_entropy(labels, box_size=2, ignored_labels=(-1,))
    assert result.evaluated_boxes == 1
    assert result.empty_boxes == 1
    assert result.ignored_samples == 4
    assert result.boundary_basin_entropy == pytest.approx(np.log(2.0))


def test_uncertainty_fraction_counts_valid_pairs_and_wilson_interval() -> None:
    reference = np.array([0, 0, 1, 1, -1])
    perturbed = np.array([0, 1, 1, 0, 0])
    result = uncertainty_fraction(
        reference,
        perturbed,
        ignored_labels=(-1,),
        perturbation_scale=1.0e-4,
        scale_units="dimensionless_state",
        perturbation_norm="euclidean",
        perturbation_direction=(1.0, 0.0),
    )
    assert result.fraction == 0.5
    assert result.uncertain_pairs == 2
    assert result.evaluated_pairs == 4
    assert result.ignored_pairs == 1
    assert result.perturbation_scale == pytest.approx(1.0e-4)
    assert result.scale_units == "dimensionless_state"
    assert result.perturbation_norm == "euclidean"
    assert result.perturbation_direction == (1.0, 0.0)
    lower, upper = result.confidence_interval
    assert 0.0 < lower < result.fraction < upper < 1.0


def test_uncertainty_exponent_recovers_synthetic_power_law() -> None:
    scales = np.array([1e-3, 2e-3, 4e-3, 8e-3])
    fractions = 0.7 * scales**0.4
    result = estimate_uncertainty_exponent(
        scales,
        fractions,
        sampling_space_dimension=2.0,
    )
    assert result.exponent == pytest.approx(0.4)
    assert result.coefficient == pytest.approx(0.7)
    assert result.r_squared == pytest.approx(1.0)
    assert result.estimated_boundary_dimension == pytest.approx(1.6)
    assert result.dimension_estimate_admissible


def test_basin_diagnostics_reject_ambiguous_inputs() -> None:
    with pytest.raises(TypeError, match="integer basin"):
        basin_entropy(np.zeros((2, 2), dtype=float), box_size=1)
    with pytest.raises(ValueError, match="no classified"):
        basin_entropy(np.full((2, 2), -1), box_size=1, ignored_labels=(-1,))
    with pytest.raises(ValueError, match="distinct scales"):
        estimate_uncertainty_exponent([0.1, 0.1], [0.2, 0.3])


@pytest.mark.parametrize("base", [0.5, 1.0, 0.0, -2.0, np.inf, np.nan])
def test_entropy_base_must_be_greater_than_one(base: float) -> None:
    with pytest.raises(ValueError, match="greater than one"):
        basin_entropy(np.zeros((2, 2), dtype=int), box_size=1, entropy_base=base)


def test_partial_box_policy_is_explicit() -> None:
    labels = np.zeros((3, 5), dtype=int)
    with pytest.raises(ValueError, match="divisible"):
        basin_entropy(labels, box_size=2)
    dropped = basin_entropy(labels, box_size=2, partial_boxes="drop")
    assert dropped.total_boxes == 2
    assert dropped.classified_samples == 8
    assert dropped.dropped_samples == 7


def test_log_two_flag_is_gated_by_documented_finite_grid_preconditions() -> None:
    labels = np.tile(np.array([[0, 1], [2, 0]], dtype=int), (2, 2))
    result = basin_entropy(labels, box_size=2)
    assert result.observed_basins == 3
    assert result.log_two_criterion_applicable
    assert result.log_two_criterion_reason == "finite_grid_preconditions_only"
    assert result.boundary_entropy_above_log_two

    ignored = basin_entropy(labels, box_size=2, ignored_labels=(2,))
    assert not ignored.log_two_criterion_applicable
    assert ignored.log_two_criterion_reason == "fewer_than_three_observed_basins"
    assert not ignored.boundary_entropy_above_log_two


def test_uncertainty_scale_metadata_is_validated() -> None:
    labels = np.array([0, 1], dtype=int)
    with pytest.raises(ValueError, match="scale_units"):
        uncertainty_fraction(labels, labels, perturbation_scale=0.1)
    with pytest.raises(ValueError, match="positive"):
        uncertainty_fraction(
            labels,
            labels,
            perturbation_scale=0.0,
            scale_units="state",
            perturbation_norm="euclidean",
        )
    with pytest.raises(ValueError, match="nonzero"):
        uncertainty_fraction(
            labels,
            labels,
            perturbation_scale=0.1,
            scale_units="state",
            perturbation_norm="euclidean",
            perturbation_direction=(0.0, 0.0),
        )


def test_constant_uncertainty_fraction_has_undefined_r_squared() -> None:
    result = estimate_uncertainty_exponent([0.01, 0.02, 0.04], [0.2, 0.2, 0.2])
    assert np.isnan(result.r_squared)
    assert result.status == "degenerate_constant_response"
