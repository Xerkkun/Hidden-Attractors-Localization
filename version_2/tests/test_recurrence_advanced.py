from __future__ import annotations

import math

import numpy as np
import pytest

from hidden_attractors.analysis.recurrence_advanced import (
    auto_recurrence_matrix,
    cross_recurrence_matrix,
    joint_recurrence_matrix,
    recurrence_quantification_advanced,
)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (
            "euclidean",
            [[True, True, False], [True, True, False], [False, False, True]],
        ),
        (
            "chebyshev",
            [[True, True, True], [True, True, True], [True, True, True]],
        ),
        (
            "manhattan",
            [[True, True, False], [True, True, True], [False, True, True]],
        ),
    ],
)
def test_exact_auto_matrices_for_all_metrics(metric: str, expected: list[list[bool]]) -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    result = auto_recurrence_matrix(
        points,
        radius=2.0 if metric != "euclidean" else 1.0,
        metric=metric,
        theiler_window=None,
        block_rows=1,
    )
    assert np.array_equal(result.matrix, np.asarray(expected, dtype=bool))
    assert np.array_equal(result.matrix, result.matrix.T)


def test_theiler_window_excludes_general_index_band() -> None:
    result = auto_recurrence_matrix(
        np.arange(5, dtype=float),
        radius=100.0,
        theiler_window=1,
        backend="numpy",
    )
    expected = np.fromfunction(lambda row, column: np.abs(row - column) > 1, (5, 5))
    assert np.array_equal(result.matrix, expected)
    assert result.eligible_points == 12
    assert result.recurrent_points == 12
    assert result.achieved_recurrence_rate == 1.0
    assert result.metadata["theiler"]["policy"] == (
        "exclude_abs_row_minus_column_le_window"
    )


def test_cross_matrix_is_rectangular_and_reverses_by_transpose() -> None:
    left = np.array([0.0, 2.0])
    right = np.array([1.0, 2.0, 4.0])
    forward = cross_recurrence_matrix(left, right, radius=1.0, metric="manhattan")
    reverse = cross_recurrence_matrix(right, left, radius=1.0, metric="manhattan")
    assert np.array_equal(
        forward.matrix,
        [[True, False, False], [True, True, False]],
    )
    assert np.array_equal(forward.matrix, reverse.matrix.T)
    assert forward.mode == "cross"
    assert forward.source_shapes == ((2, 1), (3, 1))
    assert forward.metadata["matrix_shape"] == (2, 3)
    assert forward.metadata["theiler"]["policy"] == "include_all_entries"


def test_joint_matrix_is_componentwise_logical_and() -> None:
    first = np.array([0.0, 0.0, 2.0])
    second = np.array([1.0, 1.0, 3.0])
    joint = joint_recurrence_matrix(first, second, radius=(0.0, 0.0))
    expected = np.array(
        [[False, True, False], [True, False, False], [False, False, False]]
    )
    assert np.array_equal(joint.matrix, expected)
    assert joint.mode == "joint"
    assert joint.radius == (0.0, 0.0)
    assert "logical_and" in joint.joint_policy
    assert joint.metadata["source_shapes"] == ((3, 1), (3, 1))


def test_global_target_rate_has_deterministic_inclusive_tie_policy() -> None:
    # Of the six entries outside the identity line, two have zero distance.
    # A 10% target asks for one entry, but radius thresholding must include the
    # symmetric tie pair and therefore reaches 2/6 without breaking symmetry.
    points = np.array([0.0, 0.0, 1.0])
    first = auto_recurrence_matrix(points, target_rate=0.1, backend="numpy")
    second = auto_recurrence_matrix(points, target_rate=0.1, backend="numpy")
    assert first.radius == 0.0
    assert first.recurrent_points == 2
    assert first.eligible_points == 6
    assert first.achieved_recurrence_rate == pytest.approx(1.0 / 3.0)
    assert first.achieved_recurrence_rate > first.target_recurrence_rate
    assert first.tie_policy == "include_all_entries_at_selected_radius"
    assert np.array_equal(first.matrix, second.matrix)
    assert np.array_equal(first.matrix, first.matrix.T)


def test_cross_and_joint_target_rates_use_their_global_distance_population() -> None:
    cross = cross_recurrence_matrix(
        np.array([0.0, 2.0]),
        np.array([0.0, 1.0]),
        target_rate=0.5,
        backend="numpy",
    )
    assert cross.radius == 1.0
    assert cross.achieved_recurrence_rate == 0.75

    joint = joint_recurrence_matrix(
        np.array([0.0, 1.0, 3.0]),
        np.array([0.0, 2.0, 2.0]),
        target_rate=0.5,
        backend="numpy",
    )
    assert joint.radius == 2.0
    assert joint.achieved_recurrence_rate == pytest.approx(2.0 / 3.0)
    assert "common radius" in joint.joint_policy


@pytest.mark.parametrize("metric", ["euclidean", "chebyshev", "manhattan"])
def test_numba_and_numpy_block_backends_match(metric: str) -> None:
    points = np.array([[0.0, -1.0], [0.2, 0.0], [1.0, 0.25], [2.0, 1.0]])
    accelerated = auto_recurrence_matrix(
        points,
        radius=1.1,
        metric=metric,
        theiler_window=0,
        block_rows=2,
        backend="numba",
    )
    fallback = auto_recurrence_matrix(
        points,
        radius=1.1,
        metric=metric,
        theiler_window=0,
        block_rows=1,
        backend="numpy",
    )
    assert np.array_equal(accelerated.matrix, fallback.matrix)
    assert accelerated.backend == "numba"
    assert fallback.backend == "numpy"


def test_rqa_periodic_lines_have_exact_standard_measures() -> None:
    recurrence = auto_recurrence_matrix(
        np.array([0.0, 1.0, 0.0, 1.0]),
        radius=0.0,
        theiler_window=0,
        backend="numpy",
    )
    result = recurrence_quantification_advanced(
        recurrence,
        min_diagonal=2,
        min_vertical=1,
        trend_border=0,
    )
    assert result.recurrence_rate == pytest.approx(1.0 / 3.0)
    assert result.determinism == 1.0
    assert result.mean_diagonal_length == 2.0
    assert result.longest_diagonal == 2
    assert result.divergence == 0.5
    assert result.diagonal_entropy == 0.0
    assert result.laminarity == 1.0
    assert result.trapping_time == 1.0
    assert result.longest_vertical == 1
    assert result.vertical_entropy == 0.0
    assert result.trend == pytest.approx(0.0, abs=1e-12)
    assert result.normalized_absolute_trend == pytest.approx(0.0, abs=1e-12)


def test_standard_and_normalized_trend_conventions_are_separate() -> None:
    recurrence = auto_recurrence_matrix(
        np.arange(4, dtype=float),
        radius=1.0,
        theiler_window=0,
        backend="numpy",
    )
    result = recurrence_quantification_advanced(recurrence, trend_border=0)
    # Diagonal recurrence densities at separations 1, 2, 3 are 1, 0, 0.
    assert result.normalized_absolute_trend == pytest.approx(-1.5)
    assert result.trend == pytest.approx(-1000.0 / 2.75)
    assert "per 1000" in result.metadata["rqa"]["trend"]
    assert "scale-free" in result.metadata["rqa"]["normalized_absolute_trend"]


def test_rqa_no_qualifying_lines_has_defined_divergence() -> None:
    recurrence = auto_recurrence_matrix(
        np.array([0.0, 2.0, 5.0]),
        radius=0.1,
        backend="numpy",
    )
    result = recurrence_quantification_advanced(recurrence)
    assert result.determinism == 0.0
    assert result.longest_diagonal == 0
    assert math.isinf(result.divergence)
    assert result.laminarity == 0.0
    assert np.isnan(result.diagonal_entropy)
    assert np.isnan(result.vertical_entropy)


def test_metadata_preserves_sampling_projection_and_evidence_boundary() -> None:
    recurrence = auto_recurrence_matrix(
        np.array([0.0, 0.1, 0.2]),
        radius=0.11,
        sampling="uniform dt=0.01 after 1000-step transient",
        projection="coordinates x and z",
        backend="numpy",
    )
    metadata = recurrence.metadata
    assert metadata["sampling"] == "uniform dt=0.01 after 1000-step transient"
    assert metadata["projection"] == "coordinates x and z"
    assert "not proof of chaos or hiddenness" in metadata["diagnostic_scope"]
    assert "Markovian" in metadata["fractional_state_warning"]
    assert "marwan_romano_thiel_kurths_2007" in metadata["references"]


def test_dense_and_target_selection_memory_guards_are_explicit() -> None:
    points = np.zeros((10, 1))
    with pytest.raises(MemoryError, match="above max_bytes=99"):
        auto_recurrence_matrix(points, radius=0.0, max_bytes=99)
    with pytest.raises(MemoryError, match="above max_threshold_bytes=700"):
        auto_recurrence_matrix(
            points,
            target_rate=0.5,
            max_bytes=100,
            max_threshold_bytes=700,
        )


def test_invalid_policies_and_shapes_fail_before_computation() -> None:
    points = np.arange(4, dtype=float)
    with pytest.raises(ValueError, match="exactly one"):
        auto_recurrence_matrix(points, radius=1.0, target_rate=0.2)
    with pytest.raises(ValueError, match="exactly one"):
        auto_recurrence_matrix(points)
    with pytest.raises(ValueError, match="interval"):
        auto_recurrence_matrix(points, target_rate=0.0)
    with pytest.raises(ValueError, match="metric"):
        auto_recurrence_matrix(points, radius=1.0, metric="cosine")
    with pytest.raises(ValueError, match="feature dimension"):
        cross_recurrence_matrix(np.zeros((2, 1)), np.zeros((3, 2)), radius=1.0)
    with pytest.raises(ValueError, match="at least two"):
        joint_recurrence_matrix(points, radius=1.0)
    with pytest.raises(ValueError, match="equal sample counts"):
        joint_recurrence_matrix(points, points[:-1], radius=1.0)
    with pytest.raises(ValueError, match="eligible"):
        auto_recurrence_matrix(points, radius=1.0, theiler_window=99)
    with pytest.raises(TypeError, match="integer"):
        auto_recurrence_matrix(points, radius=1.0, theiler_window=0.9)


def test_recurrence_matrix_is_immutable_and_contract_counts_stay_coherent() -> None:
    recurrence = auto_recurrence_matrix(
        np.arange(5.0),
        radius=1.0,
        backend="numpy",
    )
    assert not recurrence.matrix.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        recurrence.matrix[0, 2] = True
    result = recurrence_quantification_advanced(recurrence, trend_border=0)
    assert result.recurrence_rate == recurrence.achieved_recurrence_rate


def test_small_matrix_reports_undefined_default_border_trend() -> None:
    recurrence = auto_recurrence_matrix(np.arange(5.0), radius=1.0)
    result = recurrence_quantification_advanced(recurrence)
    assert np.isnan(result.trend)
    assert result.trend_border == 10
