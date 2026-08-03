"""Reproducible software benchmark for finite correlation-sum backends.

The benchmark compares the public Python, warmed Numba, and native-C/OpenMP
paths on deterministic synthetic sampled point sets.  Native compilation and
Numba JIT warm-up are recorded separately and excluded from the timed samples.

These measurements describe one host, workload, compiler, and runtime
configuration.  They are engineering diagnostics only: they do not establish
universal backend superiority or provide evidence of chaos, attraction,
hiddenness, or a fractional system's complete hereditary-state dimension.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np

from hidden_attractors.analysis.correlation_dimension import (
    CorrelationSumResult,
    correlation_sum_curve,
)


BACKENDS = ("python", "numba", "native_c")
METRIC = "euclidean"


@dataclass(frozen=True, slots=True)
class Workload:
    """Deterministic finite point-cloud workload."""

    name: str
    n_points: int
    dimension: int
    n_radii: int
    theiler_window: int
    seed: int


WORKLOADS = (
    Workload(
        name="small_256x3_r24",
        n_points=256,
        dimension=3,
        n_radii=24,
        theiler_window=4,
        seed=2026080301,
    ),
    Workload(
        name="medium_512x3_r48",
        n_points=512,
        dimension=3,
        n_radii=48,
        theiler_window=12,
        seed=2026080302,
    ),
)


def _make_inputs(workload: Workload) -> tuple[np.ndarray, np.ndarray]:
    """Return one deterministic correlated sample and fixed radius grid."""

    rng = np.random.default_rng(workload.seed)
    innovations = rng.normal(scale=0.55, size=(workload.n_points, workload.dimension))
    points = np.empty_like(innovations)
    points[0] = innovations[0]
    for sample_index in range(1, workload.n_points):
        points[sample_index] = 0.82 * points[sample_index - 1] + innovations[sample_index]
    radii = np.geomspace(0.05, 5.0, workload.n_radii, dtype=np.float64)
    return (
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(radii, dtype=np.float64),
    )


def _run_backend(
    points: np.ndarray,
    radii: np.ndarray,
    workload: Workload,
    backend: str,
) -> CorrelationSumResult:
    result = correlation_sum_curve(
        points,
        radii,
        theiler_window=workload.theiler_window,
        metric=METRIC,
        backend=backend,
        fallback=False,
        sampling="deterministic synthetic AR(1)-like samples in row order",
        projection=f"{workload.dimension} supplied synthetic coordinates",
    )
    if result.backend != backend:
        raise RuntimeError(
            f"requested backend {backend!r}, but execution used {result.backend!r}"
        )
    return result


def _timed_call(
    operation: Callable[[], CorrelationSumResult],
) -> tuple[float, CorrelationSumResult]:
    started_ns = time.perf_counter_ns()
    result = operation()
    elapsed_seconds = (time.perf_counter_ns() - started_ns) * 1.0e-9
    return elapsed_seconds, result


def _summary(samples: list[float]) -> dict[str, object]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "samples_seconds": [float(value) for value in values],
        "minimum_seconds": float(np.min(values)),
        "median_seconds": float(np.median(values)),
        "mean_seconds": float(np.mean(values)),
        "population_stdev_seconds": float(np.std(values, ddof=0)),
        "q25_seconds": float(np.quantile(values, 0.25)),
        "q75_seconds": float(np.quantile(values, 0.75)),
    }


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _selected_native_build(result: CorrelationSumResult) -> dict[str, object]:
    raw_build = result.metadata.get("native_build")
    if not isinstance(raw_build, Mapping):
        return {"available": False, "reason": "native build metadata unavailable"}
    keys = (
        "available",
        "abi_version",
        "kernel_id",
        "source_sha256",
        "compiler",
        "compile_command",
        "openmp_requested",
        "openmp_active",
    )
    return {key: raw_build.get(key) for key in keys}


def _benchmark_workload(
    workload: Workload,
    *,
    repeats: int,
    warmup_calls: int,
) -> tuple[dict[str, object], dict[str, object]]:
    points, radii = _make_inputs(workload)
    operations = {
        backend: (
            lambda backend=backend: _run_backend(points, radii, workload, backend)
        )
        for backend in BACKENDS
    }

    warmup_samples: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
    warmup_results: dict[str, CorrelationSumResult] = {}
    for backend in BACKENDS:
        for _ in range(warmup_calls):
            elapsed, result = _timed_call(operations[backend])
            warmup_samples[backend].append(elapsed)
            warmup_results[backend] = result

    reference_counts = warmup_results["python"].counts
    for backend in BACKENDS[1:]:
        np.testing.assert_array_equal(warmup_results[backend].counts, reference_counts)

    measured_samples: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
    last_results = dict(warmup_results)
    for repeat_index in range(repeats):
        rotation = repeat_index % len(BACKENDS)
        order = BACKENDS[rotation:] + BACKENDS[:rotation]
        for backend in order:
            gc.collect()
            elapsed, result = _timed_call(operations[backend])
            measured_samples[backend].append(elapsed)
            last_results[backend] = result

    for backend in BACKENDS[1:]:
        np.testing.assert_array_equal(last_results[backend].counts, reference_counts)

    measurements = {
        backend: {
            "warmup": _summary(warmup_samples[backend]),
            "measurement": _summary(measured_samples[backend]),
        }
        for backend in BACKENDS
    }
    medians = {
        backend: float(measurements[backend]["measurement"]["median_seconds"])
        for backend in BACKENDS
    }
    ratios = {
        "python_over_numba": medians["python"] / medians["numba"],
        "python_over_native_c": medians["python"] / medians["native_c"],
        "numba_over_native_c": medians["numba"] / medians["native_c"],
    }
    record = {
        "workload": asdict(workload),
        "metric": METRIC,
        "eligible_pairs": int(last_results["python"].eligible_pairs),
        "input": {
            "generator": "numpy.default_rng Gaussian innovations; AR coefficient 0.82",
            "points_sha256": _array_sha256(points),
            "radii_sha256": _array_sha256(radii),
            "radii_minimum": float(radii[0]),
            "radii_maximum": float(radii[-1]),
        },
        "identical_counts": True,
        "final_count": int(reference_counts[-1]),
        "backends": measurements,
        "median_time_ratios": ratios,
        "ratio_interpretation": (
            "numerator median divided by denominator median for this run; "
            "greater than one means the numerator took longer"
        ),
    }
    return record, _selected_native_build(last_results["native_c"])


def _host_record() -> dict[str, object]:
    clock = time.get_clock_info("perf_counter")
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "numba_threads": numba.get_num_threads(),
        "omp_num_threads_environment": os.environ.get("OMP_NUM_THREADS"),
        "perf_counter": {
            "implementation": clock.implementation,
            "resolution_seconds": clock.resolution,
            "monotonic": clock.monotonic,
        },
    }


def run_benchmark(*, repeats: int = 7, warmup_calls: int = 1) -> dict[str, object]:
    """Run all fixed workloads and return a JSON-serializable record."""

    if isinstance(repeats, bool) or repeats < 3:
        raise ValueError("repeats must be an integer of at least 3")
    if isinstance(warmup_calls, bool) or warmup_calls < 1:
        raise ValueError("warmup_calls must be a positive integer")

    workload_records: list[dict[str, object]] = []
    native_build: dict[str, object] | None = None
    for workload in WORKLOADS:
        record, current_build = _benchmark_workload(
            workload,
            repeats=repeats,
            warmup_calls=warmup_calls,
        )
        workload_records.append(record)
        if native_build is None:
            native_build = current_build

    script_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "benchmark": "hafo_correlation_sum_backends",
        "scope": "finite_q2_correlation_sum_software_performance",
        "evidence_boundary": (
            "host- and workload-specific engineering timing only; no universal "
            "backend ranking and no scientific dynamics claim"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "measurement_protocol": {
            "warmup_excluded": True,
            "warmup_calls_per_backend_and_workload": warmup_calls,
            "measured_repetitions_per_backend_and_workload": repeats,
            "backend_order": "deterministic cyclic rotation per repetition",
            "garbage_collection": "collected before, outside, each timed call",
            "clock": "time.perf_counter_ns",
            "fallback": False,
        },
        "host": _host_record(),
        "native_build": native_build,
        "workloads": workload_records,
    }


def _pytest_benchmark(benchmark, backend: str) -> None:
    workload = WORKLOADS[0]
    points, radii = _make_inputs(workload)
    warm_result = _run_backend(points, radii, workload, backend)
    result = benchmark(_run_backend, points, radii, workload, backend)
    np.testing.assert_array_equal(result.counts, warm_result.counts)


def test_correlation_sum_python(benchmark) -> None:
    """Measure the public Python reference path after one untimed call."""

    _pytest_benchmark(benchmark, "python")


def test_correlation_sum_numba(benchmark) -> None:
    """Measure the public Numba path after JIT warm-up."""

    _pytest_benchmark(benchmark, "numba")


def test_correlation_sum_native_c(benchmark) -> None:
    """Measure native C/OpenMP after build and load warm-up."""

    _pytest_benchmark(benchmark, "native_c")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup-calls", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    if args.warmup_calls < 1:
        parser.error("--warmup-calls must be at least 1")

    record = run_benchmark(repeats=args.repeats, warmup_calls=args.warmup_calls)
    rendered = json.dumps(record, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
