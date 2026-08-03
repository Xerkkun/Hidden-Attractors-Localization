"""Native C/OpenMP ordinal-pattern counting tests."""

from __future__ import annotations

import ctypes
import itertools
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hidden_attractors.analysis.native_permutation_entropy import (
    NativePermutationBackendUnavailable,
    NativePermutationEntropyBackend,
    NativePermutationTieError,
    native_permutation_counts,
)


def _lehmer_rank(permutation: list[int]) -> int:
    rank = 0
    dimension = len(permutation)
    for index, value in enumerate(permutation[:-1]):
        smaller_to_right = sum(
            later < value for later in permutation[index + 1 :]
        )
        rank += smaller_to_right * math.factorial(dimension - index - 1)
    return rank


def _reference_counts(
    signal: np.ndarray,
    *,
    embedding_dimension: int,
    delay: int,
    tie_policy: str,
) -> tuple[np.ndarray, int, int, int]:
    total_windows = signal.size - (embedding_dimension - 1) * delay
    counts = np.zeros(math.factorial(embedding_dimension), dtype=np.uint64)
    tied_windows = 0
    for start in range(total_windows):
        window = [
            float(signal[start + index * delay])
            for index in range(embedding_dimension)
        ]
        tied = any(
            window[left] == window[right]
            for left in range(embedding_dimension)
            for right in range(left + 1, embedding_dimension)
        )
        if tied:
            tied_windows += 1
            if tie_policy == "raise":
                raise ValueError("reference tied window")
            if tie_policy == "omit":
                continue
        permutation = sorted(
            range(embedding_dimension),
            key=lambda index: (window[index], index),
        )
        counts[_lehmer_rank(permutation)] += np.uint64(1)
    return counts, total_windows, int(np.sum(counts)), tied_windows


@pytest.fixture(scope="module")
def native_backend() -> NativePermutationEntropyBackend:
    try:
        return NativePermutationEntropyBackend.build()
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"Native compiler or shared-library loader unavailable: {exc}")


def test_all_three_dimensional_patterns_use_lexicographic_lehmer_bins(
    native_backend: NativePermutationEntropyBackend,
) -> None:
    for expected_rank, permutation_tuple in enumerate(
        itertools.permutations(range(3))
    ):
        permutation = list(permutation_tuple)
        signal = np.empty(3, dtype=np.float64)
        for order, original_index in enumerate(permutation):
            signal[original_index] = float(order)

        result = native_permutation_counts(
            signal,
            embedding_dimension=3,
            fallback=False,
        )

        assert _lehmer_rank(permutation) == expected_rank
        assert result.counts[expected_rank] == 1
        assert np.count_nonzero(result.counts) == 1
        assert result.total_windows == result.valid_windows == 1
        assert result.tied_windows == 0


@pytest.mark.parametrize("embedding_dimension", [2, 3, 4, 5, 7, 8])
@pytest.mark.parametrize("delay", [1, 2, 4])
def test_native_counts_match_independent_python_oracle(
    native_backend: NativePermutationEntropyBackend,
    embedding_dimension: int,
    delay: int,
) -> None:
    rng = np.random.default_rng(20260803 + 10 * embedding_dimension + delay)
    signal = rng.normal(size=(embedding_dimension - 1) * delay + 37)
    expected, total, valid, tied = _reference_counts(
        signal,
        embedding_dimension=embedding_dimension,
        delay=delay,
        tie_policy="stable_index",
    )

    result = native_permutation_counts(
        signal,
        embedding_dimension=embedding_dimension,
        delay=delay,
        tie_policy="stable_index",
        fallback=False,
    )

    np.testing.assert_array_equal(result.counts, expected)
    assert result.total_windows == total
    assert result.valid_windows == valid
    assert result.tied_windows == tied
    assert result.backend == "native_c"
    assert result.status == "ok"


def test_maximum_embedding_dimension_ten_is_supported(
    native_backend: NativePermutationEntropyBackend,
) -> None:
    signal = np.arange(10.0, 0.0, -1.0)
    result = native_permutation_counts(
        signal,
        embedding_dimension=10,
        fallback=False,
    )

    assert result.counts.size == math.factorial(10)
    assert result.counts[-1] == 1
    assert np.count_nonzero(result.counts) == 1
    assert result.counts.flags.writeable is False


@pytest.mark.parametrize("embedding_dimension", [4, 10])
def test_large_window_batch_is_exact_across_openmp_histogram_strategies(
    native_backend: NativePermutationEntropyBackend,
    embedding_dimension: int,
) -> None:
    signal = np.arange(1100, dtype=np.float64)
    result = native_permutation_counts(
        signal,
        embedding_dimension=embedding_dimension,
        fallback=False,
    )
    expected_windows = signal.size - embedding_dimension + 1
    assert result.total_windows == expected_windows
    assert result.valid_windows == expected_windows
    assert result.tied_windows == 0
    assert result.counts[0] == expected_windows
    assert np.count_nonzero(result.counts) == 1


@pytest.mark.parametrize("tie_policy", ["stable_index", "omit"])
def test_tie_policies_match_oracle_and_report_tied_windows(
    native_backend: NativePermutationEntropyBackend,
    tie_policy: str,
) -> None:
    signal = np.asarray([3.0, 1.0, 1.0, 2.0, 5.0, 5.0, 4.0])
    expected, total, valid, tied = _reference_counts(
        signal,
        embedding_dimension=3,
        delay=1,
        tie_policy=tie_policy,
    )

    result = native_permutation_counts(
        signal,
        embedding_dimension=3,
        tie_policy=tie_policy,
        fallback=False,
    )

    np.testing.assert_array_equal(result.counts, expected)
    assert result.total_windows == total
    assert result.valid_windows == valid
    assert result.tied_windows == tied == 4
    if tie_policy == "stable_index":
        assert valid == total
    else:
        assert valid + tied == total


def test_raise_policy_reports_all_tied_windows_without_histogram_result(
    native_backend: NativePermutationEntropyBackend,
) -> None:
    signal = np.asarray([3.0, 1.0, 1.0, 2.0, 5.0, 5.0, 4.0])
    with pytest.raises(NativePermutationTieError) as error:
        native_permutation_counts(
            signal,
            embedding_dimension=3,
            tie_policy="raise",
            fallback=False,
        )

    assert error.value.total_windows == 5
    assert error.value.valid_windows == 1
    assert error.value.tied_windows == 4
    assert error.value.first_tied_window is None


def test_exact_zero_ties_include_signed_zero(
    native_backend: NativePermutationEntropyBackend,
) -> None:
    signal = np.asarray([-0.0, +0.0, 1.0, 2.0])
    result = native_permutation_counts(
        signal,
        embedding_dimension=2,
        tie_policy="omit",
        fallback=False,
    )
    assert result.total_windows == 3
    assert result.tied_windows == 1
    assert result.valid_windows == 2


def test_noncontiguous_signal_is_normalized_before_native_call(
    native_backend: NativePermutationEntropyBackend,
) -> None:
    base = np.linspace(-2.0, 3.0, 80)
    signal = base[::3]
    assert signal.flags.c_contiguous is False
    expected, total, valid, tied = _reference_counts(
        signal,
        embedding_dimension=4,
        delay=2,
        tie_policy="stable_index",
    )
    result = native_permutation_counts(
        signal,
        embedding_dimension=4,
        delay=2,
        fallback=False,
    )
    np.testing.assert_array_equal(result.counts, expected)
    assert (result.total_windows, result.valid_windows, result.tied_windows) == (
        total,
        valid,
        tied,
    )


def test_native_metadata_identifies_reproducible_abi(
    native_backend: NativePermutationEntropyBackend,
) -> None:
    result = native_permutation_counts(
        [0.0, 2.0, 1.0, 3.0],
        embedding_dimension=3,
        fallback=False,
    )

    assert result.build.available is True
    assert result.build.backend == "native_c"
    assert result.build.abi_version == 1
    assert result.build.kernel_id == "hafo_permutation_entropy_counts_v1"
    assert result.build.openmp_requested is True
    assert isinstance(result.build.openmp_active, bool)
    assert len(result.build.source_sha256) == 64
    assert result.build.library_path
    assert result.build.compiler
    assert result.build.compile_command
    assert native_backend.lib.hafo_permutation_status(0) == b"ok"
    assert native_backend.lib.hafo_permutation_status(-11) == b"tied_window"
    assert native_backend.lib.hafo_permutation_max_embedding_dimension() == 10
    assert native_backend.lib.hafo_permutation_pattern_count(10) == math.factorial(10)


@pytest.mark.parametrize("tie_policy", ["stable_index", "omit"])
def test_missing_native_backend_falls_back_to_self_contained_numba(
    monkeypatch: pytest.MonkeyPatch,
    tie_policy: str,
) -> None:
    def unavailable(cls, output_name=None):
        raise OSError("synthetic compiler absence")

    monkeypatch.setattr(
        NativePermutationEntropyBackend,
        "build",
        classmethod(unavailable),
    )
    signal = np.asarray([2.0, 1.0, 1.0, 4.0, 3.0, 5.0])
    expected, total, valid, tied = _reference_counts(
        signal,
        embedding_dimension=3,
        delay=1,
        tie_policy=tie_policy,
    )
    result = native_permutation_counts(
        signal,
        embedding_dimension=3,
        tie_policy=tie_policy,
        fallback=True,
    )

    np.testing.assert_array_equal(result.counts, expected)
    assert (result.total_windows, result.valid_windows, result.tied_windows) == (
        total,
        valid,
        tied,
    )
    assert result.backend == "numba_fallback"
    assert result.build.available is False
    assert "synthetic compiler absence" in (result.build.fallback_reason or "")
    with pytest.raises(NativePermutationBackendUnavailable, match="unavailable"):
        native_permutation_counts(
            signal,
            embedding_dimension=3,
            fallback=False,
        )


def test_numba_fallback_raise_reports_first_tied_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(cls, output_name=None):
        raise RuntimeError("synthetic loader absence")

    monkeypatch.setattr(
        NativePermutationEntropyBackend,
        "build",
        classmethod(unavailable),
    )
    with pytest.raises(NativePermutationTieError) as error:
        native_permutation_counts(
            [0.0, 1.0, 1.0, 2.0, 3.0],
            embedding_dimension=3,
            tie_policy="raise",
            fallback=True,
        )
    assert error.value.first_tied_window == 0
    assert error.value.tied_windows == 2


@pytest.mark.parametrize(
    ("signal", "message"),
    [
        ([], "non-empty"),
        ([[0.0, 1.0]], "one-dimensional"),
        ([0.0, np.nan, 1.0], "finite"),
        ([0.0, np.inf, 1.0], "finite"),
        ([False, True, False], "real-valued"),
        ([0.0j, 1.0j, 2.0j], "real-valued"),
    ],
)
def test_invalid_signals_fail_before_native_execution(signal, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        native_permutation_counts(signal, embedding_dimension=2)


@pytest.mark.parametrize(
    ("embedding_dimension", "exception", "message"),
    [
        (1, ValueError, "between 2 and 10"),
        (11, ValueError, "between 2 and 10"),
        (2.5, TypeError, "integer"),
        (True, TypeError, "integer"),
    ],
)
def test_invalid_embedding_dimension_is_rejected(
    embedding_dimension,
    exception,
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        native_permutation_counts(
            [0.0, 1.0, 2.0],
            embedding_dimension=embedding_dimension,
        )


@pytest.mark.parametrize(
    ("delay", "exception", "message"),
    [
        (0, ValueError, "at least 1"),
        (-1, ValueError, "at least 1"),
        (1.5, TypeError, "integer"),
        (True, TypeError, "integer"),
    ],
)
def test_invalid_delay_is_rejected(delay, exception, message: str) -> None:
    with pytest.raises(exception, match=message):
        native_permutation_counts(
            [0.0, 1.0, 2.0],
            embedding_dimension=2,
            delay=delay,
        )


def test_short_signal_invalid_policy_and_fallback_flag_are_rejected() -> None:
    with pytest.raises(ValueError, match="too short"):
        native_permutation_counts(
            [0.0, 1.0, 2.0],
            embedding_dimension=3,
            delay=2,
        )
    with pytest.raises(ValueError, match="stable_index"):
        native_permutation_counts(
            [0.0, 1.0, 2.0],
            embedding_dimension=2,
            tie_policy="average_rank",
        )
    with pytest.raises(TypeError, match="string token"):
        native_permutation_counts(
            [0.0, 1.0, 2.0],
            embedding_dimension=2,
            tie_policy=0,
        )
    with pytest.raises(TypeError, match="Boolean"):
        native_permutation_counts(
            [0.0, 1.0, 2.0],
            embedding_dimension=2,
            fallback="yes",
        )


def test_c_abi_rejects_invalid_arguments_before_counting(
    native_backend: NativePermutationEntropyBackend,
) -> None:
    signal = np.arange(8, dtype=np.float64)
    counts = np.zeros(math.factorial(3), dtype=np.uint64)
    total = ctypes.c_uint64(0)
    valid = ctypes.c_uint64(0)
    tied = ctypes.c_uint64(0)
    call = native_backend.lib.hafo_permutation_entropy_counts

    output_pointers = (
        ctypes.byref(total),
        ctypes.byref(valid),
        ctypes.byref(tied),
    )
    assert call(signal, 8, 1, 1, 0, counts, 6, *output_pointers) == -3
    assert call(signal, 8, 11, 1, 0, counts, 6, *output_pointers) == -3
    assert call(signal, 8, 3, 0, 0, counts, 6, *output_pointers) == -4
    assert call(signal, 8, 3, 1, 7, counts, 6, *output_pointers) == -5
    assert call(signal, 8, 3, 1, 0, counts, 5, *output_pointers) == -6
    assert call(signal, 3, 3, 2, 0, counts, 6, *output_pointers) == -2


def test_c_abi_rejects_nonfinite_aliased_and_overflowing_buffers(
    native_backend: NativePermutationEntropyBackend,
) -> None:
    counts = np.zeros(math.factorial(3), dtype=np.uint64)
    total = ctypes.c_uint64(0)
    valid = ctypes.c_uint64(0)
    tied = ctypes.c_uint64(0)
    call = native_backend.lib.hafo_permutation_entropy_counts

    nonfinite_signal = np.asarray([0.0, 1.0, np.nan, 2.0], dtype=np.float64)
    assert call(
        nonfinite_signal,
        nonfinite_signal.size,
        3,
        1,
        0,
        counts,
        counts.size,
        ctypes.byref(total),
        ctypes.byref(valid),
        ctypes.byref(tied),
    ) == -7

    signal = np.arange(8, dtype=np.float64)
    aliased_counts = signal.view(np.uint64)[: math.factorial(3)]
    assert call(
        signal,
        signal.size,
        3,
        1,
        0,
        aliased_counts,
        aliased_counts.size,
        ctypes.byref(total),
        ctypes.byref(valid),
        ctypes.byref(tied),
    ) == -8

    tiny_signal = np.asarray([0.0, 1.0], dtype=np.float64)
    two_counts = np.zeros(2, dtype=np.uint64)
    size_t_max = ctypes.c_size_t(-1).value
    assert call(
        tiny_signal,
        size_t_max,
        2,
        1,
        0,
        two_counts,
        2,
        ctypes.byref(total),
        ctypes.byref(valid),
        ctypes.byref(tied),
    ) == -9


def test_c_abi_raise_status_reports_counts_and_leaves_histogram_zero(
    native_backend: NativePermutationEntropyBackend,
) -> None:
    signal = np.asarray([3.0, 1.0, 1.0, 2.0, 5.0, 5.0, 4.0])
    counts = np.full(math.factorial(3), 99, dtype=np.uint64)
    total = ctypes.c_uint64(0)
    valid = ctypes.c_uint64(0)
    tied = ctypes.c_uint64(0)
    status = native_backend.lib.hafo_permutation_entropy_counts(
        signal,
        signal.size,
        3,
        1,
        2,
        counts,
        counts.size,
        ctypes.byref(total),
        ctypes.byref(valid),
        ctypes.byref(tied),
    )
    assert status == -11
    assert (total.value, valid.value, tied.value) == (5, 1, 4)
    np.testing.assert_array_equal(counts, 0)


@pytest.mark.parametrize("openmp", [False, True])
def test_c11_source_compiles_with_strict_warning_flags(
    tmp_path: Path,
    openmp: bool,
) -> None:
    compiler = shutil.which(os.environ.get("CC", "gcc"))
    if compiler is None:
        pytest.skip("No C compiler is available for the strict source audit.")
    source = (
        Path(__file__).resolve().parents[1]
        / "hidden_attractors"
        / "native"
        / "csrc"
        / "permutation_entropy.c"
    )
    output = tmp_path / ("permutation_entropy_omp.o" if openmp else "permutation_entropy.o")
    command = [
        compiler,
        "-std=c11",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
    ]
    if openmp:
        command.append("-fopenmp")
    command.extend(["-c", str(source), "-o", str(output)])
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if openmp and completed.returncode != 0:
        pytest.skip(
            "The available compiler has no usable OpenMP toolchain: "
            + completed.stderr.strip()
        )
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    assert output.stat().st_size > 0
