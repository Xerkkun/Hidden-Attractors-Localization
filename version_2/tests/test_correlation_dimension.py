from __future__ import annotations

from dataclasses import is_dataclass
import math

import numpy as np
import pytest

from hidden_attractors.analysis.correlation_dimension import (
    CorrelationDimensionResult,
    CorrelationSumResult,
    correlation_sum_curve,
    estimate_correlation_dimension,
    fit_correlation_dimension,
)


@pytest.mark.parametrize(
    ("metric", "expected_counts"),
    [
        ("euclidean", [0, 1, 2, 3, 3]),
        ("manhattan", [0, 1, 2, 2, 3]),
        ("chebyshev", [0, 1, 3, 3, 3]),
    ],
)
def test_manual_pair_counts_for_all_metrics(
    metric: str,
    expected_counts: list[int],
) -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    radii = np.array([1.0, 2.0, 2.0000001, 2.25, 3.1])

    curve = correlation_sum_curve(
        points,
        radii,
        metric=metric,
        backend="python",
    )

    assert isinstance(curve, CorrelationSumResult)
    assert is_dataclass(curve)
    assert curve.metric == metric
    assert curve.backend == "python"
    assert curve.eligible_pairs == 3
    assert np.array_equal(curve.counts, expected_counts)
    assert np.allclose(curve.correlation_sums, np.asarray(expected_counts) / 3.0)


def test_threshold_is_strict_and_excludes_distance_equal_to_radius() -> None:
    curve = correlation_sum_curve(
        np.array([0.0, 1.0, 3.0]),
        np.array([1.0, 2.0, 3.0, np.nextafter(3.0, math.inf)]),
        backend="python",
    )

    # Pair distances are exactly 1, 2 and 3.  Equality is never recurrent.
    assert np.array_equal(curve.counts, [0, 1, 2, 3])
    assert np.allclose(curve.correlation_sums, [0.0, 1 / 3, 2 / 3, 1.0])


@pytest.mark.parametrize("theiler_window", [0, 1, 2, 4])
def test_theiler_window_uses_declared_eligible_pair_formula(
    theiler_window: int,
) -> None:
    sample_count = 6
    expected_pairs = (
        (sample_count - theiler_window)
        * (sample_count - theiler_window - 1)
        // 2
    )
    curve = correlation_sum_curve(
        np.arange(sample_count, dtype=float),
        np.array([10.0]),
        theiler_window=theiler_window,
        backend="python",
    )

    assert curve.theiler_window == theiler_window
    assert curve.eligible_pairs == expected_pairs
    assert curve.counts.item() == expected_pairs
    assert curve.correlation_sums.item() == 1.0


def test_duplicate_points_are_counted_only_at_a_positive_radius() -> None:
    curve = correlation_sum_curve(
        np.array([0.0, 0.0, 1.0]),
        np.array([np.nextafter(0.0, 1.0), 1.0, np.nextafter(1.0, 2.0)]),
        backend="python",
    )

    assert np.array_equal(curve.counts, [1, 1, 3])


def test_counts_and_correlation_sums_are_monotone_in_radius() -> None:
    rng = np.random.default_rng(20260803)
    curve = correlation_sum_curve(
        rng.normal(size=(31, 3)),
        np.geomspace(1.0e-3, 10.0, 41),
        theiler_window=2,
        backend="python",
    )

    assert np.all(np.diff(curve.counts) >= 0)
    assert np.all(np.diff(curve.correlation_sums) >= 0.0)
    assert np.all((curve.correlation_sums >= 0.0) & (curve.correlation_sums <= 1.0))


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_translation_and_positive_scaling_preserve_pair_counts(metric: str) -> None:
    points = np.array(
        [[-1.0, 2.0], [0.25, -0.5], [1.5, 0.75], [3.0, -2.0]]
    )
    radii = np.array([0.25, 1.0, 2.5, 5.0])
    reference = correlation_sum_curve(
        points,
        radii,
        metric=metric,
        backend="python",
    )
    translated = correlation_sum_curve(
        points + np.array([17.0, -9.0]),
        radii,
        metric=metric,
        backend="python",
    )
    scaled = correlation_sum_curve(
        points * 3.5,
        radii * 3.5,
        metric=metric,
        backend="python",
    )

    assert np.array_equal(translated.counts, reference.counts)
    assert np.array_equal(scaled.counts, reference.counts)


def test_euclidean_counts_are_invariant_under_rotation() -> None:
    angle = 0.731
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    points = np.array(
        [[-1.0, 0.5], [0.0, 0.0], [0.25, 2.0], [3.0, -1.0]]
    )
    radii = np.array([0.5, 1.25, 2.75, 5.0])

    original = correlation_sum_curve(
        points,
        radii,
        metric="euclidean",
        backend="python",
    )
    rotated = correlation_sum_curve(
        points @ rotation.T,
        radii,
        metric="euclidean",
        backend="python",
    )

    assert np.array_equal(rotated.counts, original.counts)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_numba_and_python_backends_have_identical_counts(metric: str) -> None:
    rng = np.random.default_rng(73)
    points = rng.normal(size=(29, 4))
    radii = np.geomspace(0.05, 8.0, 23)

    python_curve = correlation_sum_curve(
        points,
        radii,
        theiler_window=3,
        metric=metric,
        backend="python",
        fallback=False,
    )
    numba_curve = correlation_sum_curve(
        points,
        radii,
        theiler_window=3,
        metric=metric,
        backend="numba",
        fallback=False,
    )

    assert python_curve.backend == "python"
    assert numba_curve.backend == "numba"
    assert np.array_equal(numba_curve.counts, python_curve.counts)
    assert np.array_equal(
        numba_curve.correlation_sums,
        python_curve.correlation_sums,
    )


def test_auto_backend_preserves_the_public_curve_contract() -> None:
    curve = correlation_sum_curve(
        np.arange(5.0),
        np.array([0.5, 1.5, 5.0]),
        backend="auto",
    )

    assert curve.backend in {"native_c", "numba", "python"}
    assert np.array_equal(curve.counts, [0, 4, 10])


@pytest.mark.parametrize(
    "points",
    [
        np.array([True, False, True]),
        np.array([1.0 + 0.0j, 2.0 + 0.0j]),
    ],
)
def test_points_reject_boolean_and_complex_arrays(points: np.ndarray) -> None:
    with pytest.raises(TypeError):
        correlation_sum_curve(points, np.array([1.0]), backend="python")


@pytest.mark.parametrize(
    "radii",
    [
        np.array([False, True]),
        np.array([1.0 + 0.0j, 2.0 + 0.0j]),
    ],
)
def test_radii_reject_boolean_and_complex_arrays(radii: np.ndarray) -> None:
    with pytest.raises(TypeError):
        correlation_sum_curve(np.arange(4.0), radii, backend="python")


@pytest.mark.parametrize(
    "points",
    [
        np.array([0.0, np.nan, 1.0]),
        np.array([[0.0], [np.inf], [1.0]]),
        np.array([[0.0], [-np.inf], [1.0]]),
    ],
)
def test_points_reject_nonfinite_values(points: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        correlation_sum_curve(points, np.array([1.0]), backend="python")


@pytest.mark.parametrize(
    "radii",
    [
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1.0, -np.inf]),
    ],
)
def test_radii_reject_nonfinite_values(radii: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        correlation_sum_curve(np.arange(4.0), radii, backend="python")


@pytest.mark.parametrize(
    "points",
    [
        np.array(1.0),
        np.empty((0,)),
        np.array([1.0]),
        np.empty((2, 0)),
        np.zeros((2, 1, 1)),
    ],
)
def test_invalid_point_shapes_are_rejected(points: np.ndarray) -> None:
    with pytest.raises(ValueError):
        correlation_sum_curve(points, np.array([1.0]), backend="python")


@pytest.mark.parametrize(
    "radii",
    [
        np.array(1.0),
        np.empty((0,)),
        np.ones((2, 1)),
        np.array([-1.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([2.0, 1.0]),
    ],
)
def test_invalid_radius_contracts_are_rejected(radii: np.ndarray) -> None:
    with pytest.raises(ValueError):
        correlation_sum_curve(np.arange(4.0), radii, backend="python")


@pytest.mark.parametrize("theiler_window", [True, 0.5, -1, 3])
def test_invalid_theiler_windows_are_rejected(
    theiler_window: object,
) -> None:
    expected_error = TypeError if isinstance(theiler_window, (bool, float)) else ValueError
    with pytest.raises(expected_error):
        correlation_sum_curve(
            np.arange(4.0),
            np.array([1.0]),
            theiler_window=theiler_window,  # type: ignore[arg-type]
            backend="python",
        )


@pytest.mark.parametrize("metric", ["cosine", "Euclidean", "", True])
def test_unknown_or_non_string_metrics_are_rejected(metric: object) -> None:
    with pytest.raises((TypeError, ValueError), match="metric"):
        correlation_sum_curve(
            np.arange(4.0),
            np.array([1.0]),
            metric=metric,  # type: ignore[arg-type]
            backend="python",
        )


@pytest.mark.parametrize("backend", ["numpy", "gpu", "", True])
def test_unknown_or_non_string_backends_are_rejected(backend: object) -> None:
    with pytest.raises((TypeError, ValueError), match="backend"):
        correlation_sum_curve(
            np.arange(4.0),
            np.array([1.0]),
            backend=backend,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("fallback", [0, 1, "yes", None])
def test_fallback_must_be_a_real_boolean(fallback: object) -> None:
    with pytest.raises(TypeError, match="fallback"):
        correlation_sum_curve(
            np.arange(4.0),
            np.array([1.0]),
            backend="python",
            fallback=fallback,  # type: ignore[arg-type]
        )


def _exact_quadratic_scaling_curve() -> CorrelationSumResult:
    # Six eligible distances are 0.5, 1.1, 1.6, 1.7, 2.8 and 3.3.
    # At radii 1, sqrt(2), 2 the counts are 1, 2 and 4, hence
    # C(r) = r**2 / 6 exactly at all three fit points.
    return correlation_sum_curve(
        np.array([0.0, 0.5, 1.6, 3.3]),
        np.array([1.0, math.sqrt(2.0), 2.0]),
        backend="python",
    )


def test_explicit_inclusive_fit_recovers_an_exact_power_law() -> None:
    curve = _exact_quadratic_scaling_curve()
    result = fit_correlation_dimension(
        curve,
        fit_radius_range=(1.0, 2.0),
        minimum_points=3,
    )

    assert isinstance(result, CorrelationDimensionResult)
    assert is_dataclass(result)
    assert result.curve is curve
    assert result.fit_radius_range == (1.0, 2.0)
    assert result.minimum_points == 3
    assert result.slope == pytest.approx(2.0, abs=1.0e-13)
    assert result.intercept == pytest.approx(-math.log(6.0), abs=1.0e-13)
    assert result.r_squared == pytest.approx(1.0, abs=1.0e-14)
    assert result.regression_standard_error == pytest.approx(0.0, abs=1.0e-13)
    assert np.array_equal(result.fit_indices, [0, 1, 2])
    assert np.allclose(result.log_radii, np.log(curve.radii))
    assert np.allclose(
        result.log_correlation_sums,
        np.log(curve.correlation_sums),
    )
    assert np.allclose(result.local_slopes, [2.0, 2.0], atol=1.0e-13)


def test_constant_nontrivial_correlation_sum_reports_degenerate_r_squared() -> None:
    curve = correlation_sum_curve(
        np.array([0.0, 0.5, 10.0]),
        np.array([1.0, 2.0, 3.0]),
        backend="python",
    )
    assert np.allclose(curve.correlation_sums, 1.0 / 3.0)

    result = fit_correlation_dimension(
        curve,
        fit_radius_range=(1.0, 3.0),
    )

    assert result.slope == pytest.approx(0.0, abs=1.0e-14)
    assert result.intercept == pytest.approx(math.log(1.0 / 3.0))
    assert math.isnan(result.r_squared)
    assert result.regression_standard_error == pytest.approx(0.0, abs=1.0e-14)
    assert np.allclose(result.local_slopes, 0.0, atol=1.0e-14)


def test_fit_excludes_zero_and_one_correlation_sums() -> None:
    curve = correlation_sum_curve(
        np.array([0.0, 0.5, 1.6, 3.3]),
        np.array([0.1, 1.0, 4.0]),
        backend="python",
    )
    assert np.array_equal(curve.correlation_sums, [0.0, 1.0 / 6.0, 1.0])

    with pytest.raises(ValueError, match="minimum_points|usable"):
        fit_correlation_dimension(
            curve,
            fit_radius_range=(0.1, 4.0),
            minimum_points=3,
        )


def test_fit_radius_range_is_mandatory_and_never_selected_automatically() -> None:
    curve = _exact_quadratic_scaling_curve()
    with pytest.raises(TypeError):
        fit_correlation_dimension(curve)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "fit_radius_range",
    [
        True,
        (1.0,),
        (2.0, 1.0),
        (1.0, 1.0),
        (np.nan, 2.0),
        (1.0, np.inf),
        (1.0 + 0.0j, 2.0),
        (True, 2.0),
    ],
)
def test_invalid_fit_radius_ranges_are_rejected(
    fit_radius_range: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        fit_correlation_dimension(
            _exact_quadratic_scaling_curve(),
            fit_radius_range=fit_radius_range,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("minimum_points", [True, 2.5, 0, 1, 2])
def test_minimum_points_requires_an_integer_of_at_least_three(
    minimum_points: object,
) -> None:
    expected_error = TypeError if isinstance(minimum_points, (bool, float)) else ValueError
    with pytest.raises(expected_error):
        fit_correlation_dimension(
            _exact_quadratic_scaling_curve(),
            fit_radius_range=(1.0, 2.0),
            minimum_points=minimum_points,  # type: ignore[arg-type]
        )


def test_estimate_convenience_function_matches_explicit_two_stage_analysis() -> None:
    points = np.array([0.0, 0.5, 1.6, 3.3])
    radii = np.array([1.0, math.sqrt(2.0), 2.0])
    explicit_curve = correlation_sum_curve(points, radii, backend="python")
    explicit_fit = fit_correlation_dimension(
        explicit_curve,
        fit_radius_range=(1.0, 2.0),
    )

    combined = estimate_correlation_dimension(
        points,
        radii,
        fit_radius_range=(1.0, 2.0),
        minimum_points=3,
        backend="python",
    )

    assert isinstance(combined, CorrelationDimensionResult)
    assert np.array_equal(combined.curve.counts, explicit_curve.counts)
    assert np.array_equal(
        combined.curve.correlation_sums,
        explicit_curve.correlation_sums,
    )
    assert combined.slope == pytest.approx(explicit_fit.slope)
    assert combined.intercept == pytest.approx(explicit_fit.intercept)


def test_result_arrays_are_owned_read_only_copies() -> None:
    input_radii = np.array([1.0, math.sqrt(2.0), 2.0])
    curve = correlation_sum_curve(
        np.array([0.0, 0.5, 1.6, 3.3]),
        input_radii,
        backend="python",
    )
    fit = fit_correlation_dimension(curve, fit_radius_range=(1.0, 2.0))
    input_radii[0] = 0.25

    assert curve.radii[0] == 1.0
    arrays = (
        curve.radii,
        curve.counts,
        curve.correlation_sums,
        fit.fit_indices,
        fit.log_radii,
        fit.log_correlation_sums,
        fit.local_slopes,
    )
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = array.flat[0]


def test_metadata_preserves_sampling_projection_and_evidence_boundary() -> None:
    sampling = "uniform dt=0.01 after a 1000-sample transient"
    projection = "delay coordinates x(t), x(t-7), x(t-14)"
    curve = correlation_sum_curve(
        np.arange(12.0).reshape(4, 3),
        np.array([1.0, 10.0]),
        sampling=sampling,
        projection=projection,
        backend="python",
    )
    fit_curve = _exact_quadratic_scaling_curve()
    fit = fit_correlation_dimension(
        fit_curve,
        fit_radius_range=(1.0, 2.0),
    )

    assert curve.sampling == sampling
    assert curve.projection == projection
    assert curve.evidence_scope == "finite_sample_empirical_trajectory_diagnostic"
    assert "hereditary state" in curve.fractional_state_caveat
    assert fit.evidence_scope == curve.evidence_scope
    assert fit.fractional_state_caveat == curve.fractional_state_caveat
    assert curve.metadata["sampling"] == sampling
    assert curve.metadata["projection"] == projection
    metadata_text = repr(curve.metadata).lower()
    assert "strict" in metadata_text
    assert "not proof of chaos or hiddenness" in metadata_text
    assert "fractional" in metadata_text
    fit_metadata_text = repr(fit.metadata).lower()
    assert "explicit" in fit_metadata_text
    assert "not proof of chaos or hiddenness" in fit_metadata_text
