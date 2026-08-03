from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.analysis.delay_embedding import (
    EVIDENCE_SCOPE,
    FDE_RECONSTRUCTION_CAVEAT,
    estimate_delay_autocorrelation,
    estimate_delay_mutual_information,
    false_nearest_neighbors,
    generalized_delay_embedding,
)


def test_generalized_multivariate_embedding_has_exact_anchor_alignment() -> None:
    trajectory = np.column_stack((np.arange(8.0), 100.0 + np.arange(8.0)))
    result = generalized_delay_embedding(
        trajectory,
        ((0, 0), (1, 2), (0, -1)),
        dt=0.25,
        time_unit="s",
    )

    assert np.array_equal(result.anchor_indices, np.arange(2, 7))
    assert np.array_equal(
        result.vectors,
        np.array(
            [
                [2.0, 100.0, 3.0],
                [3.0, 101.0, 4.0],
                [4.0, 102.0, 5.0],
                [5.0, 103.0, 6.0],
                [6.0, 104.0, 7.0],
            ]
        ),
    )
    assert np.array_equal(result.aligned_times, result.anchor_indices * 0.25)
    assert result.projection == ((0, 0), (1, 2), (0, -1))
    assert result.lag_times == (0.0, 0.5, -0.25)
    assert result.lag_unit == "samples"
    assert result.time_unit == "s"


def test_embedding_accepts_irregular_times_without_inventing_a_physical_lag() -> None:
    trajectory = np.arange(6.0)
    times = np.array([0.0, 0.1, 0.25, 0.4, 0.7, 1.0])
    result = generalized_delay_embedding(
        trajectory,
        ((0, 0), (0, 2)),
        times=times,
        time_unit="s",
    )

    assert np.array_equal(result.anchor_indices, [2, 3, 4, 5])
    assert np.array_equal(result.aligned_times, times[2:])
    assert result.dt is None
    assert result.lag_times is None
    assert result.time_source == "supplied_irregular_times"


def test_embedding_validation_rejects_ambiguous_or_impossible_contracts() -> None:
    with pytest.raises(ValueError, match="at least one"):
        generalized_delay_embedding(np.arange(5.0), ())
    with pytest.raises(ValueError, match="outside trajectory width"):
        generalized_delay_embedding(np.arange(5.0), ((1, 0),))
    with pytest.raises(ValueError, match="too short"):
        generalized_delay_embedding(np.arange(5.0), ((0, 5),))
    with pytest.raises(ValueError, match="inconsistent"):
        generalized_delay_embedding(
            np.arange(5.0),
            ((0, 0),),
            times=np.arange(5.0) * 0.2,
            dt=0.1,
        )


def test_periodic_autocorrelation_delay_has_expected_crossing_and_minimum() -> None:
    period_samples = 40
    series = np.sin(2.0 * np.pi * np.arange(800) / period_samples)

    crossing = estimate_delay_autocorrelation(
        series,
        max_lag=30,
        criterion="first_zero_crossing",
        dt=0.02,
        time_unit="s",
    )
    minimum = estimate_delay_autocorrelation(
        series,
        max_lag=30,
        criterion="first_local_minimum",
        dt=0.02,
        time_unit="s",
    )

    # Sampling chooses the first non-positive value just after the analytic
    # quarter-period zero; the first minimum is at half a period.
    assert crossing.lag_samples in {10, 11}
    assert minimum.lag_samples == 20
    assert minimum.lag_time == pytest.approx(0.4)
    assert crossing.scores[0] == 1.0
    assert crossing.backend == "numpy.fft"
    assert crossing.evidence_scope == EVIDENCE_SCOPE


def test_histogram_mutual_information_is_declared_and_deterministic() -> None:
    series = np.sin(2.0 * np.pi * np.arange(800) / 40.0)
    first = estimate_delay_mutual_information(
        series,
        max_lag=30,
        bins=16,
        fallback="none",
        dt=0.05,
        time_unit="s",
    )
    second = estimate_delay_mutual_information(
        series,
        max_lag=30,
        bins=16,
        fallback="none",
        dt=0.05,
        time_unit="s",
    )

    assert first.lag_samples is not None
    selected = int(np.flatnonzero(first.lags == first.lag_samples)[0])
    assert 0 < selected < first.scores.size - 1
    assert first.scores[selected] < first.scores[selected - 1]
    assert first.scores[selected] <= first.scores[selected + 1]
    assert first.lag_samples == second.lag_samples
    assert np.array_equal(first.scores, second.scores)
    assert first.parameters["binning"] == 16
    assert first.parameters["n_bins"] == 16
    assert first.parameters["edges_reused_for_all_lags"] is True
    assert first.score_name == "mutual_information_nats"


def test_mutual_information_fallback_and_constant_signal_validation() -> None:
    rng = np.random.default_rng(321)
    series = rng.normal(size=80)
    result = estimate_delay_mutual_information(
        series,
        max_lag=1,
        bins="sturges",
        fallback="global_minimum",
    )
    assert result.lag_samples == 1
    assert result.selection == "global_minimum_fallback"
    assert result.status == "selected_by_fallback"

    with pytest.raises(ValueError, match="constant series"):
        estimate_delay_autocorrelation(np.ones(20))
    with pytest.raises(ValueError, match="constant series"):
        estimate_delay_mutual_information(np.ones(20), bins=4)


def test_mutual_information_can_select_lag_one_against_zero_lag_entropy() -> None:
    series = np.random.default_rng(1).normal(size=200)
    result = estimate_delay_mutual_information(
        series,
        max_lag=2,
        bins=8,
        fallback="none",
    )
    assert result.lag_samples == 1
    assert result.status == "selected"
    assert result.parameters["zero_lag_information"] > result.scores[0]
    assert result.scores[0] < result.scores[1]
    assert result.method == "fixed_bin_plugin_mutual_information"
    assert result.parameters["estimator"] == "fixed_bin_plugin_not_adaptive_partition"


def test_mutual_information_rejects_tail_with_too_few_pairs() -> None:
    with pytest.raises(ValueError, match="minimum_pairs"):
        estimate_delay_mutual_information(
            np.arange(30.0),
            max_lag=29,
            bins=4,
            minimum_pairs=8,
        )


def _henon_observable(n_samples: int = 500) -> np.ndarray:
    x = np.empty(n_samples, dtype=float)
    y = np.empty(n_samples, dtype=float)
    x[0] = 0.1
    y[0] = 0.1
    for index in range(n_samples - 1):
        x[index + 1] = 1.0 - 1.4 * x[index] ** 2 + y[index]
        y[index + 1] = 0.3 * x[index]
    return x[100:]


def test_fnn_selects_a_low_dimension_for_a_henon_observable() -> None:
    result = false_nearest_neighbors(
        _henon_observable(),
        delay=1,
        min_dimension=1,
        max_dimension=4,
        theiler_window=8,
        rtol=10.0,
        atol=2.0,
        metric="euclidean",
        selection_threshold=0.05,
        dt=0.1,
        time_unit="s",
    )

    assert result.fractions[0] > 0.5
    assert result.fractions[1] < 0.05
    assert result.selected_dimension == 2
    assert result.delay_time == pytest.approx(0.1)
    assert result.theiler_window == 8
    assert all(record.status == "ok" for record in result.records)
    assert all(
        record.false_neighbors <= record.valid_neighbors
        for record in result.records
    )
    assert result.backend == "scipy.spatial.cKDTree"


def test_fnn_is_deterministic_and_reports_insufficient_neighbors() -> None:
    observable = _henon_observable(300)
    first = false_nearest_neighbors(
        observable,
        max_dimension=3,
        theiler_window=5,
        metric="chebyshev",
    )
    second = false_nearest_neighbors(
        observable,
        max_dimension=3,
        theiler_window=5,
        metric="chebyshev",
    )
    assert np.array_equal(first.fractions, second.fractions)
    assert first.selected_dimension == second.selected_dimension

    insufficient = false_nearest_neighbors(
        np.arange(6.0) ** 2,
        delay=2,
        max_dimension=4,
        theiler_window=20,
    )
    assert insufficient.selected_dimension is None
    assert all(record.valid_neighbors == 0 for record in insufficient.records)
    assert all(
        record.status == "insufficient_neighbors"
        for record in insufficient.records
    )
    assert np.all(np.isnan(insufficient.fractions))


def test_result_scope_does_not_overclaim_fractional_state_reconstruction() -> None:
    result = generalized_delay_embedding(np.arange(6.0), ((0, 0), (0, 1)))
    assert result.time_unit == "sample_index"
    assert result.time_source == "sample_indices"
    assert result.evidence_scope == "finite_sample_empirical_trajectory_diagnostic"
    assert "hereditary state" in result.fractional_state_caveat
    assert result.fractional_state_caveat == FDE_RECONSTRUCTION_CAVEAT

    irregular_times = np.cumsum(np.linspace(0.1, 0.2, 40))
    delay = estimate_delay_autocorrelation(
        np.sin(np.arange(40.0)),
        max_lag=8,
        times=irregular_times,
    )
    assert delay.time_source == "supplied_irregular_times"
    assert delay.lag_time is None
    assert "not a constant physical-time delay" in delay.sampling_caveat


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_fnn_supported_metrics_have_a_transparent_name(metric: str) -> None:
    result = false_nearest_neighbors(
        np.linspace(0.0, 1.0, 30),
        max_dimension=2,
        theiler_window=2,
        metric=metric,
        rtol=2.0,
        atol=2.0,
        selection_threshold=0.0,
    )
    assert result.metric == metric
    assert result.selected_dimension == 1
    expected = (
        "kennel_brown_abarbanel_euclidean"
        if metric == "euclidean"
        else "generalized_lp_fnn_using_kba_threshold_form"
    )
    assert result.algorithm == expected
