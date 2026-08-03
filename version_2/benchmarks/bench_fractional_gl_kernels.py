"""Synthetic performance checks for direct GL history kernels.

Build and JIT warm-up are deliberately excluded from timed repetitions.  The
workloads measure sampled-data operators, not complete FDE solves, and timing
results are engineering diagnostics rather than evidence of chaos or
hidden-attractor existence.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hidden_attractors.fractional import (
    fast_grunwald_letnikov_derivative,
    grunwald_letnikov_derivative,
    native_grunwald_letnikov_derivative,
)


ORDERS = np.asarray([0.41, 0.73, 0.96], dtype=float)
STEP = 0.0025


def _sample_history(n_times: int) -> np.ndarray:
    rng = np.random.default_rng(20260802 + n_times)
    return np.ascontiguousarray(rng.normal(size=(n_times, 3)).cumsum(axis=0))


FULL_HISTORY = _sample_history(1800)
WINDOW_HISTORY = _sample_history(6000)


def _numba_full() -> np.ndarray:
    return grunwald_letnikov_derivative(
        FULL_HISTORY,
        STEP,
        ORDERS,
        definition="caputo_shifted",
    ).values


def _native_full() -> np.ndarray:
    return native_grunwald_letnikov_derivative(
        FULL_HISTORY,
        STEP,
        ORDERS,
        definition="caputo_shifted",
        fallback=False,
    ).values


def _fft_full() -> np.ndarray:
    return fast_grunwald_letnikov_derivative(
        FULL_HISTORY,
        STEP,
        ORDERS,
        definition="caputo_shifted",
        backend="fft",
    ).values


def _numba_window() -> np.ndarray:
    return grunwald_letnikov_derivative(
        WINDOW_HISTORY,
        STEP,
        ORDERS,
        definition="caputo_shifted",
        history_window=128,
    ).values


def _native_window() -> np.ndarray:
    return native_grunwald_letnikov_derivative(
        WINDOW_HISTORY,
        STEP,
        ORDERS,
        definition="caputo_shifted",
        history_window=128,
        fallback=False,
    ).values


def test_gl_numba_full_history(benchmark) -> None:
    """Measure the warmed Numba full-history operator."""

    _numba_full()
    values = benchmark(_numba_full)
    assert values.shape == FULL_HISTORY.shape


def test_gl_native_full_history(benchmark) -> None:
    """Measure the warmed native-C full-history operator."""

    _native_full()
    values = benchmark(_native_full)
    assert values.shape == FULL_HISTORY.shape


def test_gl_fft_full_history(benchmark) -> None:
    """Measure the zero-padded offline FFT full-history operator."""

    _fft_full()
    values = benchmark(_fft_full)
    assert values.shape == FULL_HISTORY.shape


def test_gl_numba_finite_window(benchmark) -> None:
    """Measure the warmed Numba operator with a 128-sample window."""

    _numba_window()
    values = benchmark(_numba_window)
    assert values.shape == WINDOW_HISTORY.shape


def test_gl_native_finite_window(benchmark) -> None:
    """Measure the warmed native-C operator with a 128-sample window."""

    _native_window()
    values = benchmark(_native_window)
    assert values.shape == WINDOW_HISTORY.shape


def _median_seconds(operation: Callable[[], np.ndarray], repeats: int = 11) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return float(statistics.median(samples))


def _standalone(*, repeats: int = 11) -> dict[str, object]:
    """Return one reproducible, warm-kernel benchmark record."""

    numba_full = _numba_full()
    native_full = _native_full()
    fft_full = _fft_full()
    numba_window = _numba_window()
    native_window = _native_window()
    # Different OpenMP/SIMD accumulation orders produce ordinary floating-point
    # roundoff on long histories; the tolerance is scaled to that reduction.
    np.testing.assert_allclose(native_full, numba_full, rtol=5e-13, atol=5e-12)
    np.testing.assert_allclose(fft_full, numba_full, rtol=1e-11, atol=2e-11)
    np.testing.assert_allclose(native_window, numba_window, rtol=5e-13, atol=5e-12)

    build = native_grunwald_letnikov_derivative(
        FULL_HISTORY[:2], STEP, ORDERS, fallback=False
    ).build
    timings = {
        "full_history_numba_seconds": _median_seconds(_numba_full, repeats),
        "full_history_native_c_seconds": _median_seconds(_native_full, repeats),
        "full_history_fft_seconds": _median_seconds(_fft_full, repeats),
        "window_128_numba_seconds": _median_seconds(_numba_window, repeats),
        "window_128_native_c_seconds": _median_seconds(_native_window, repeats),
    }
    timings["full_history_native_speedup"] = (
        timings["full_history_numba_seconds"]
        / timings["full_history_native_c_seconds"]
    )
    timings["window_128_native_speedup"] = (
        timings["window_128_numba_seconds"]
        / timings["window_128_native_c_seconds"]
    )
    timings["full_history_fft_speedup_over_numba"] = (
        timings["full_history_numba_seconds"] / timings["full_history_fft_seconds"]
    )

    fft_crossover: dict[str, dict[str, float]] = {}
    for sample_count in (16, 32, 64, 128, 256, 512, 1024, 4096):
        history = _sample_history(sample_count)

        def direct_operation(history=history) -> np.ndarray:
            return grunwald_letnikov_derivative(
                history, STEP, ORDERS, definition="caputo_shifted"
            ).values

        def fft_operation(history=history) -> np.ndarray:
            return fast_grunwald_letnikov_derivative(
                history,
                STEP,
                ORDERS,
                definition="caputo_shifted",
                backend="fft",
            ).values

        direct_values = direct_operation()
        fft_values = fft_operation()
        np.testing.assert_allclose(fft_values, direct_values, rtol=1e-11, atol=2e-11)
        direct_seconds = _median_seconds(direct_operation, repeats)
        fft_seconds = _median_seconds(fft_operation, repeats)
        fft_crossover[str(sample_count)] = {
            "numba_direct_seconds": direct_seconds,
            "fft_seconds": fft_seconds,
            "fft_speedup": direct_seconds / fft_seconds,
        }
    record = {
        "scope": "synthetic_sampled_data_GL_kernel",
        "evidence_boundary": "software performance only; no dynamics claim",
        "warmup_excluded": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repetitions": repeats,
        "workloads": {
            "full_history": {"n_times": 1800, "dimension": 3},
            "window_128": {"n_times": 6000, "dimension": 3},
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "compiler": build.compiler,
            "openmp_requested": build.openmp_requested,
            "openmp_active": build.openmp_active,
            "kernel_id": build.kernel_id,
            "source_sha256": build.source_sha256,
        },
        "timings": timings,
        "fft_crossover": fft_crossover,
    }
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    record = _standalone(repeats=args.repeats)
    rendered = json.dumps(record, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
