"""Native C/OpenMP Grassberger--Procaccia pair-count tests."""

from __future__ import annotations

import ctypes
from typing import Callable

import numpy as np
import pytest

from hidden_attractors.analysis.native_correlation_sum import (
    NativeCorrelationBackendUnavailable,
    NativeCorrelationSumBackend,
    native_correlation_sum_counts,
)


def _reference_counts(
    points: np.ndarray,
    radii: np.ndarray,
    *,
    theiler_window: int,
    metric: str,
) -> np.ndarray:
    distance_functions: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "euclidean": lambda left, right: float(np.linalg.norm(left - right)),
        "chebyshev": lambda left, right: float(np.max(np.abs(left - right))),
        "manhattan": lambda left, right: float(np.sum(np.abs(left - right))),
    }
    distance = distance_functions[metric]
    counts = np.zeros(radii.size, dtype=np.uint64)
    for left_index in range(points.shape[0]):
        for right_index in range(left_index + 1, points.shape[0]):
            if right_index - left_index <= theiler_window:
                continue
            separation = distance(points[left_index], points[right_index])
            counts += np.asarray(separation < radii, dtype=np.uint64)
    return counts


@pytest.fixture(scope="module")
def native_backend() -> NativeCorrelationSumBackend:
    try:
        return NativeCorrelationSumBackend.build()
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"Native compiler or shared-library loader unavailable: {exc}")


@pytest.mark.parametrize(
    ("metric", "metric_code"),
    [("euclidean", 0), ("chebyshev", 1), ("manhattan", 2)],
)
def test_native_counts_match_numba_and_direct_reference_for_each_metric(
    native_backend: NativeCorrelationSumBackend,
    metric: str,
    metric_code: int,
) -> None:
    from hidden_attractors.analysis.correlation_dimension import (
        _correlation_counts_numba,
    )

    rng = np.random.default_rng(20260803)
    points = rng.normal(size=(41, 4))
    radii = np.asarray([0.15, 0.4, 0.9, 1.7, 3.5], dtype=np.float64)
    expected = _reference_counts(
        points,
        radii,
        theiler_window=2,
        metric=metric,
    )
    numba_counts = _correlation_counts_numba(
        np.ascontiguousarray(points),
        radii,
        2,
        metric_code,
    )
    result = native_correlation_sum_counts(
        points,
        radii,
        theiler_window=2,
        metric=metric,
        fallback=False,
    )

    np.testing.assert_array_equal(result.counts, expected)
    np.testing.assert_array_equal(result.counts, numba_counts)
    assert result.eligible_pairs == (38 * 39) // 2
    assert result.metric == metric
    assert result.theiler_window == 2
    assert result.backend == "native_c"
    assert result.status == "ok"


def test_strict_boundary_and_differential_multiradius_bins(
    native_backend: NativeCorrelationSumBackend,
) -> None:
    points = np.asarray([[0.0], [1.0], [2.0]])
    radii = np.asarray([1.0, 1.0 + 1.0e-12, 2.0, 2.0 + 1.0e-12])
    result = native_correlation_sum_counts(
        points,
        radii,
        fallback=False,
    )

    np.testing.assert_array_equal(result.counts, [0, 2, 2, 3])
    assert result.eligible_pairs == 3
    assert result.counts.flags.writeable is False


@pytest.mark.parametrize(
    ("theiler_window", "eligible_pairs"),
    [(0, 15), (1, 10), (2, 6), (4, 1), (5, 0), (50, 0)],
)
def test_theiler_window_uses_strict_index_separation(
    native_backend: NativeCorrelationSumBackend,
    theiler_window: int,
    eligible_pairs: int,
) -> None:
    points = np.arange(6, dtype=np.float64)[:, None]
    radii = np.asarray([0.5, 1.5, 10.0])
    result = native_correlation_sum_counts(
        points,
        radii,
        theiler_window=theiler_window,
        fallback=False,
    )
    expected = _reference_counts(
        points,
        radii,
        theiler_window=theiler_window,
        metric="euclidean",
    )

    np.testing.assert_array_equal(result.counts, expected)
    assert result.eligible_pairs == eligible_pairs


def test_native_metadata_identifies_reproducible_openmp_abi(
    native_backend: NativeCorrelationSumBackend,
) -> None:
    points = np.asarray([[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]])
    result = native_correlation_sum_counts(
        points,
        [0.1, 1.0, 2.0],
        fallback=False,
    )

    assert result.build.available is True
    assert result.build.backend == "native_c"
    assert result.build.abi_version == 1
    assert result.build.kernel_id == "hafo_correlation_sum_q2_v1"
    assert result.build.openmp_requested is True
    assert isinstance(result.build.openmp_active, bool)
    assert len(result.build.source_sha256) == 64
    assert result.build.library_path
    assert result.build.compiler
    assert result.build.compile_command
    assert native_backend.lib.hafo_correlation_status(0) == b"ok"
    assert native_backend.lib.hafo_correlation_status(-7) == b"aliased_buffers"


def test_noncontiguous_points_are_normalized_before_native_call(
    native_backend: NativeCorrelationSumBackend,
) -> None:
    base = np.arange(60, dtype=np.float64).reshape(10, 6)
    points = base[:, ::2]
    assert points.flags.c_contiguous is False
    radii = np.asarray([1.0, 10.0, 100.0])
    result = native_correlation_sum_counts(
        points,
        radii,
        metric="manhattan",
        fallback=False,
    )
    expected = _reference_counts(
        points,
        radii,
        theiler_window=0,
        metric="manhattan",
    )
    np.testing.assert_array_equal(result.counts, expected)


def test_missing_compiler_or_loader_falls_back_lazily_to_numba(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(cls, output_name=None):
        raise OSError("synthetic compiler absence")

    monkeypatch.setattr(
        NativeCorrelationSumBackend,
        "build",
        classmethod(unavailable),
    )
    points = np.asarray([[0.0], [0.5], [1.0], [2.0]])
    radii = np.asarray([0.5, 0.5001, 1.5, 3.0])
    expected = _reference_counts(
        points,
        radii,
        theiler_window=1,
        metric="euclidean",
    )
    result = native_correlation_sum_counts(
        points,
        radii,
        theiler_window=1,
        fallback=True,
    )

    np.testing.assert_array_equal(result.counts, expected)
    assert result.backend == "numba_fallback"
    assert result.build.available is False
    assert result.build.abi_version is None
    assert "synthetic compiler absence" in (result.build.fallback_reason or "")
    with pytest.raises(NativeCorrelationBackendUnavailable, match="unavailable"):
        native_correlation_sum_counts(
            points,
            radii,
            fallback=False,
        )


@pytest.mark.parametrize(
    ("points", "message"),
    [
        ([0.0, 1.0], "shape"),
        ([[0.0]], "n_points"),
        ([[0.0], [np.nan]], "finite"),
        ([[0.0], [np.inf]], "finite"),
        ([[False], [True]], "real-valued"),
        ([[0.0j], [1.0j]], "real-valued"),
    ],
)
def test_invalid_points_fail_before_native_execution(points, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        native_correlation_sum_counts(points, [1.0])


@pytest.mark.parametrize(
    ("radii", "message"),
    [
        ([], "non-empty"),
        ([[1.0]], "one-dimensional"),
        ([0.0, 1.0], "positive"),
        ([-1.0, 1.0], "positive"),
        ([1.0, 1.0], "strictly increasing"),
        ([2.0, 1.0], "strictly increasing"),
        ([1.0, np.nan], "finite"),
        ([False, True], "real-valued"),
    ],
)
def test_invalid_radii_fail_before_native_execution(radii, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        native_correlation_sum_counts([[0.0], [1.0]], radii)


@pytest.mark.parametrize("theiler_window", [-1, 1.5, True])
def test_invalid_theiler_window_is_rejected(theiler_window) -> None:
    with pytest.raises((TypeError, ValueError), match="nonnegative integer"):
        native_correlation_sum_counts(
            [[0.0], [1.0]],
            [1.0],
            theiler_window=theiler_window,
        )


def test_invalid_metric_and_fallback_flag_are_rejected() -> None:
    with pytest.raises(ValueError, match="metric must be one of"):
        native_correlation_sum_counts(
            [[0.0], [1.0]],
            [1.0],
            metric="minkowski",
        )
    with pytest.raises(TypeError, match="fallback must be Boolean"):
        native_correlation_sum_counts(
            [[0.0], [1.0]],
            [1.0],
            fallback="yes",
        )


def test_c_abi_rejects_nonfinite_and_aliased_buffers(
    native_backend: NativeCorrelationSumBackend,
) -> None:
    radii = np.asarray([0.5, 2.0], dtype=np.float64)
    counts = np.zeros(radii.size, dtype=np.uint64)
    eligible = ctypes.c_uint64(0)
    nonfinite_points = np.asarray([0.0, np.nan, 1.0, 2.0], dtype=np.float64)
    status = native_backend.lib.hafo_correlation_sum_counts(
        nonfinite_points,
        2,
        2,
        radii,
        radii.size,
        0,
        0,
        counts,
        ctypes.byref(eligible),
    )
    assert status == -6

    points = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    aliased_counts = points.view(np.uint64)[: radii.size]
    status = native_backend.lib.hafo_correlation_sum_counts(
        points,
        2,
        2,
        radii,
        radii.size,
        0,
        0,
        aliased_counts,
        ctypes.byref(eligible),
    )
    assert status == -7


def test_c_abi_rejects_shape_products_that_overflow_size_t(
    native_backend: NativeCorrelationSumBackend,
) -> None:
    points = np.asarray([0.0, 1.0], dtype=np.float64)
    radii = np.asarray([1.0], dtype=np.float64)
    counts = np.zeros(1, dtype=np.uint64)
    eligible = ctypes.c_uint64(0)
    size_t_max = ctypes.c_size_t(-1).value
    status = native_backend.lib.hafo_correlation_sum_counts(
        points,
        size_t_max,
        2,
        radii,
        1,
        0,
        0,
        counts,
        ctypes.byref(eligible),
    )
    assert status == -8
