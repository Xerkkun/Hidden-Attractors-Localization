from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np
import pytest

from hidden_attractors.analysis.contracts import PrehistorySpec, TrajectoryInput
from hidden_attractors.analysis.permutation_entropy import (
    PERMUTATION_ENTROPY_EVIDENCE_SCOPE,
    PERMUTATION_ENTROPY_MAX_EMBEDDING_DIMENSION,
    PERMUTATION_ENTROPY_MAX_PATTERN_STATES,
    PERMUTATION_ENTROPY_NATIVE_AUTO_MIN_WINDOWS,
    PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS,
    OrdinalPatternDistribution,
    PermutationEntropyResult,
    _automatic_backend,
    ordinal_pattern_distribution,
    permutation_entropy,
    permutation_entropy_from_distribution,
)


def test_increasing_signal_has_only_lexicographic_rank_zero() -> None:
    distribution = ordinal_pattern_distribution(
        np.arange(8, dtype=float),
        embedding_dimension=3,
        backend="python",
    )
    np.testing.assert_array_equal(distribution.counts, [6, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(distribution.probabilities, [1, 0, 0, 0, 0, 0])
    assert distribution.total_windows == 6
    assert distribution.valid_windows == 6
    assert distribution.observed_patterns == 1
    assert distribution.missing_patterns == 5
    result = permutation_entropy_from_distribution(distribution)
    assert result.entropy == 0.0
    assert result.normalized_entropy == 0.0


def test_decreasing_signal_has_only_last_lexicographic_rank() -> None:
    distribution = ordinal_pattern_distribution(
        np.arange(8, 0, -1, dtype=float),
        embedding_dimension=3,
        backend="python",
    )
    np.testing.assert_array_equal(distribution.counts, [0, 0, 0, 0, 0, 6])


def test_forward_delay_windows_are_explicit() -> None:
    signal = np.array([0.0, 10.0, 2.0, 8.0, 4.0, 6.0, 12.0])
    tau_one = ordinal_pattern_distribution(
        signal,
        embedding_dimension=3,
        delay=1,
        backend="python",
    )
    tau_two = ordinal_pattern_distribution(
        signal,
        embedding_dimension=3,
        delay=2,
        backend="python",
    )
    assert tau_one.total_windows == 5
    assert tau_two.total_windows == 3
    assert not np.array_equal(tau_one.counts, tau_two.counts)
    assert tau_two.metadata["window_order"] == "forward_x_s_plus_k_delay"


def test_stable_index_ties_use_temporal_index_as_secondary_key() -> None:
    distribution = ordinal_pattern_distribution(
        [1.0, 1.0, 0.0],
        embedding_dimension=3,
        tie_policy="stable_index",
        backend="python",
    )
    # Stable ascending argsort is [2, 0, 1], lexicographic Lehmer rank 4.
    np.testing.assert_array_equal(distribution.counts, [0, 0, 0, 0, 1, 0])
    assert distribution.tied_windows == 1
    assert distribution.valid_windows == 1
    assert distribution.omitted_windows == 0


def test_omit_ties_changes_the_denominator_without_relabeling() -> None:
    distribution = ordinal_pattern_distribution(
        [1.0, 1.0, 0.0, 2.0],
        embedding_dimension=3,
        tie_policy="omit",
        backend="python",
    )
    # First window is omitted; [1,0,2] maps to permutation [1,0,2], rank 2.
    np.testing.assert_array_equal(distribution.counts, [0, 0, 1, 0, 0, 0])
    assert distribution.total_windows == 2
    assert distribution.valid_windows == 1
    assert distribution.tied_windows == 1
    assert distribution.omitted_windows == 1


@pytest.mark.parametrize("backend", ["python", "numba", "native_c"])
def test_raise_policy_rejects_ties_for_every_backend(backend: str) -> None:
    with pytest.raises(ValueError, match="tie_policy='raise'"):
        ordinal_pattern_distribution(
            [1.0, 1.0, 0.0, 2.0],
            tie_policy="raise",
            backend=backend,
            fallback=True,
        )


def test_omit_rejects_a_signal_with_no_valid_windows() -> None:
    with pytest.raises(ValueError, match="No valid ordinal windows"):
        ordinal_pattern_distribution(
            np.ones(8),
            tie_policy="omit",
            backend="python",
        )


@pytest.mark.parametrize("embedding_dimension", range(2, 9))
@pytest.mark.parametrize("delay", [1, 2, 4])
@pytest.mark.parametrize("tie_policy", ["stable_index", "omit"])
def test_python_numba_and_native_counts_are_identical(
    embedding_dimension: int,
    delay: int,
    tie_policy: str,
) -> None:
    rng = np.random.default_rng(20_260_803 + embedding_dimension + delay)
    signal = np.round(rng.normal(size=96), decimals=1)
    kwargs = dict(
        embedding_dimension=embedding_dimension,
        delay=delay,
        tie_policy=tie_policy,
    )
    python = ordinal_pattern_distribution(signal, backend="python", **kwargs)
    numba = ordinal_pattern_distribution(signal, backend="numba", **kwargs)
    native = ordinal_pattern_distribution(
        signal,
        backend="native_c",
        fallback=False,
        **kwargs,
    )
    np.testing.assert_array_equal(numba.counts, python.counts)
    np.testing.assert_array_equal(native.counts, python.counts)
    np.testing.assert_array_equal(numba.probabilities, python.probabilities)
    np.testing.assert_array_equal(native.probabilities, python.probabilities)
    assert numba.tied_windows == native.tied_windows == python.tied_windows
    assert numba.valid_windows == native.valid_windows == python.valid_windows
    assert native.backend == "native_c"


def test_auto_uses_numba_below_native_window_threshold() -> None:
    distribution = ordinal_pattern_distribution(
        np.sin(np.linspace(0.0, 20.0, 2000)),
        backend="auto",
    )
    assert distribution.requested_backend == "auto"
    assert distribution.backend == "numba"


@pytest.mark.parametrize(
    ("embedding_dimension", "windows", "expected"),
    [
        (2, 131_071, "numba"),
        (2, 131_072, "native_c"),
        (3, 32_767, "numba"),
        (3, 32_768, "native_c"),
        (5, 32_767, "numba"),
        (5, 32_768, "native_c"),
        (8, 131_071, "numba"),
        (8, 131_072, "native_c"),
        (9, 10_000_000, "numba"),
        (10, 10_000_000, "numba"),
    ],
)
def test_auto_backend_policy_is_dimension_aware(
    embedding_dimension: int,
    windows: int,
    expected: str,
) -> None:
    assert _automatic_backend(windows, embedding_dimension) == expected
    assert PERMUTATION_ENTROPY_NATIVE_AUTO_MIN_WINDOWS == 32_768
    assert PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS[9] is None


def test_auto_metadata_records_the_dimension_specific_threshold() -> None:
    distribution = ordinal_pattern_distribution(
        np.sin(np.linspace(0.0, 20.0, 2000)),
        embedding_dimension=5,
        backend="auto",
    )
    execution = distribution.metadata["execution"]
    assert execution["auto_policy"] == (
        "dimension_aware_native_window_threshold_otherwise_numba"
    )
    assert execution["auto_native_window_threshold"] == 32_768


def test_entropy_is_invariant_under_strictly_increasing_transforms() -> None:
    rng = np.random.default_rng(7)
    signal = rng.normal(size=500)
    transformed = np.exp(signal)
    original = permutation_entropy(signal, embedding_dimension=5, backend="numba")
    mapped = permutation_entropy(transformed, embedding_dimension=5, backend="numba")
    np.testing.assert_array_equal(original.distribution.counts, mapped.distribution.counts)
    assert original.entropy == pytest.approx(mapped.entropy, abs=0.0)


def test_strictly_decreasing_transform_preserves_entropy_not_rank_labels() -> None:
    rng = np.random.default_rng(8)
    signal = rng.normal(size=500)
    original = permutation_entropy(signal, embedding_dimension=4, backend="python")
    reversed_order = permutation_entropy(-signal, embedding_dimension=4, backend="python")
    assert not np.array_equal(
        original.distribution.counts,
        reversed_order.distribution.counts,
    )
    assert original.entropy == pytest.approx(reversed_order.entropy, abs=2e-15)


@pytest.mark.parametrize("log_base", [2.0, math.e, 10.0])
def test_logarithm_base_changes_units_but_not_normalized_entropy(
    log_base: float,
) -> None:
    distribution = ordinal_pattern_distribution(
        [0.0, 2.0, 1.0, 3.0, 0.5, 2.5, 1.5],
        backend="python",
    )
    result = permutation_entropy_from_distribution(
        distribution,
        log_base=log_base,
    )
    expected_maximum = math.log(math.factorial(3), log_base)
    assert result.maximum_entropy == pytest.approx(expected_maximum)
    assert result.normalized_entropy == pytest.approx(
        result.entropy / expected_maximum
    )


def test_normalization_uses_all_factorial_patterns_not_observed_patterns() -> None:
    result = permutation_entropy(np.arange(10.0), embedding_dimension=4, backend="python")
    assert result.distribution.observed_patterns == 1
    assert result.maximum_entropy == pytest.approx(math.log2(math.factorial(4)))
    assert result.normalization == "log_factorial_outcome_space"
    assert result.estimator == "plugin"


def test_convenience_result_reuses_its_distribution_contract() -> None:
    result = permutation_entropy(
        [0.0, 1.0, 3.0, 2.0, 4.0, 1.5],
        embedding_dimension=3,
        delay=1,
        tie_policy="stable_index",
        log_base=2.0,
        backend="python",
        sampling="fixture index",
        projection="fixture scalar",
    )
    assert isinstance(result, PermutationEntropyResult)
    assert isinstance(result.distribution, OrdinalPatternDistribution)
    assert result.log_base == 2.0
    assert result.distribution.sampling == "fixture index"
    assert result.distribution.projection == "fixture scalar"
    assert result.distribution.sample_count == 6
    assert result.backend == "python"
    assert result.trajectory_fingerprint == result.distribution.trajectory_fingerprint
    assert result.sampling == "fixture index"
    assert result.evidence_scope == PERMUTATION_ENTROPY_EVIDENCE_SCOPE


def test_multivariable_trajectory_requires_explicit_component() -> None:
    t = np.linspace(0.0, 1.0, 20)
    trajectory = TrajectoryInput(
        t=t,
        x=np.column_stack((np.sin(t), np.cos(t))),
        system_kind="integer_flow",
        projection=("sine", "cosine"),
    )
    with pytest.raises(ValueError, match="component is required"):
        permutation_entropy(trajectory)
    by_name = permutation_entropy(trajectory, component="cosine", backend="python")
    by_index = permutation_entropy(trajectory, component=1, backend="python")
    np.testing.assert_array_equal(
        by_name.distribution.counts,
        by_index.distribution.counts,
    )
    assert by_name.distribution.projection == "cosine"
    assert by_name.distribution.trajectory_fingerprint == trajectory.fingerprint()


def test_fractional_trajectory_retains_hereditary_warning_and_provenance() -> None:
    t = np.linspace(0.0, 5.0, 200)
    trajectory = TrajectoryInput(
        t=t,
        x=np.sin(t),
        system_kind="fractional_continuous",
        derivative_definition="caputo_hadamard",
        order=0.82,
        memory_policy="full_history",
        lower_terminal_and_prehistory=PrehistorySpec(
            kind="point_initial_value",
            lower_terminal=0.0,
        ),
        solver_and_tolerances={"method": "fixture"},
    )
    result = permutation_entropy(trajectory, backend="numba")
    distribution = result.distribution
    assert distribution.trajectory_system_kind == "fractional_continuous"
    assert distribution.derivative_definition == "caputo_hadamard"
    assert distribution.memory_policy == "full_history"
    assert any("hereditary state" in warning for warning in distribution.warnings)
    assert distribution.metadata["trajectory_contract"]["order"] == 0.82


def test_result_arrays_and_nested_metadata_are_immutable() -> None:
    distribution = ordinal_pattern_distribution(
        [0.0, 1.0, 2.0, 1.0, 0.0],
        backend="python",
    )
    assert not distribution.counts.flags.writeable
    assert not distribution.probabilities.flags.writeable
    assert isinstance(distribution.metadata, MappingProxyType)
    assert isinstance(distribution.metadata["trajectory_contract"], MappingProxyType)
    with pytest.raises(ValueError):
        distribution.counts[0] = 99
    with pytest.raises(TypeError):
        distribution.metadata["new"] = "value"
    with pytest.raises(TypeError):
        distribution.metadata["trajectory_contract"]["new"] = "value"


def test_analysis_envelopes_preserve_method_backend_and_references() -> None:
    result = permutation_entropy(
        [0.0, 2.0, 1.0, 3.0, 0.5],
        backend="numba",
    )
    distribution_envelope = result.distribution.as_analysis_result()
    entropy_envelope = result.as_analysis_result()
    assert distribution_envelope.method == "bandt_pompe_ordinal_pattern_distribution"
    assert entropy_envelope.method == "bandt_pompe_permutation_entropy"
    assert entropy_envelope.backend == "hafo_numba"
    assert entropy_envelope.status == "finite_numerical_diagnostic"
    assert "10.1103/PhysRevLett.88.174102" in entropy_envelope.references
    assert entropy_envelope.trajectory_fingerprint == result.distribution.trajectory_fingerprint


@pytest.mark.parametrize(
    ("kwargs", "exception", "match"),
    [
        ({"embedding_dimension": True}, TypeError, "integer"),
        ({"embedding_dimension": 1}, ValueError, "at least"),
        ({"embedding_dimension": 11}, ValueError, "at most"),
        ({"delay": 0}, ValueError, "at least"),
        ({"delay": 1.5}, TypeError, "integer"),
        ({"tie_policy": "random"}, ValueError, "tie_policy"),
        ({"backend": "gpu"}, ValueError, "backend"),
        ({"fallback": 1}, TypeError, "fallback"),
    ],
)
def test_distribution_contract_rejects_invalid_parameters(
    kwargs: dict[str, object],
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        ordinal_pattern_distribution(np.arange(20.0), **kwargs)


@pytest.mark.parametrize("log_base", [1.0, 0.0, -2.0, np.inf, np.nan])
def test_entropy_rejects_invalid_logarithm_bases(log_base: float) -> None:
    distribution = ordinal_pattern_distribution(np.arange(10.0), backend="python")
    with pytest.raises(ValueError, match="greater than one"):
        permutation_entropy_from_distribution(distribution, log_base=log_base)
    

def test_entropy_rejects_boolean_logarithm_base() -> None:
    distribution = ordinal_pattern_distribution(np.arange(10.0), backend="python")
    with pytest.raises(TypeError, match="real number"):
        permutation_entropy_from_distribution(distribution, log_base=True)


@pytest.mark.parametrize(
    ("data", "exception", "match"),
    [
        (np.ones((8, 1)), ValueError, "one-dimensional"),
        ([0.0, 1.0, np.nan, 2.0], ValueError, "finite"),
        ([0.0, 1.0 + 2.0j, 2.0], TypeError, "real-valued"),
        ([True, False, True, False], TypeError, "Boolean"),
    ],
)
def test_signal_validation_is_strict(
    data: object,
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        ordinal_pattern_distribution(data, embedding_dimension=2)


def test_short_signal_and_raw_component_are_rejected() -> None:
    with pytest.raises(ValueError, match="too short"):
        ordinal_pattern_distribution([0.0, 1.0], embedding_dimension=3)
    with pytest.raises(ValueError, match="only valid with TrajectoryInput"):
        ordinal_pattern_distribution(np.arange(10.0), component=0)


def test_pattern_state_guard_is_explicit() -> None:
    assert PERMUTATION_ENTROPY_MAX_EMBEDDING_DIMENSION == 10
    assert PERMUTATION_ENTROPY_MAX_PATTERN_STATES == math.factorial(10)
    with pytest.raises(ValueError, match="at most 10"):
        ordinal_pattern_distribution(
            np.arange(20.0),
            embedding_dimension=11,
        )


def test_warnings_state_finite_bias_and_no_chaos_certificate() -> None:
    result = permutation_entropy(np.arange(10.0), backend="python")
    joined = " ".join(result.warnings)
    assert "finite-sample" in joined
    assert "statistically dependent" in joined
    assert "not by itself proof of chaos" in joined
