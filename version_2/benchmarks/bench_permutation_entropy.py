"""Reproducible backend benchmark for Bandt--Pompe ordinal histograms.

The benchmark times the same deterministic finite signal through the public
Python, warmed Numba, and native-C/OpenMP paths. Native compilation and Numba
JIT warm-up are recorded separately and excluded from measured repetitions.
Counts must agree exactly before any timing ratio is reported.

Results describe one host, compiler, runtime, and workload. They are software
engineering evidence only; they neither establish universal backend superiority
nor certify an entropy rate, chaos, attraction, or hiddenness.
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

from hidden_attractors.analysis.permutation_entropy import (
    PERMUTATION_ENTROPY_NATIVE_AUTO_MIN_WINDOWS,
    PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS,
    OrdinalPatternDistribution,
    _ordinal_pattern_counts_numba,
    _ordinal_pattern_counts_python,
    ordinal_pattern_distribution,
)
from hidden_attractors.analysis.native_permutation_entropy import (
    native_permutation_counts,
)


BACKENDS = ("python", "numba", "native_c")
TIE_POLICY = "stable_index"
PRACTICAL_EQUIVALENCE_RELATIVE = 0.05


@dataclass(frozen=True, slots=True)
class Workload:
    """One deterministic finite ordinal-pattern workload."""

    name: str
    total_windows: int
    embedding_dimension: int
    delay: int
    seed: int

    @property
    def sample_count(self) -> int:
        return self.total_windows + (self.embedding_dimension - 1) * self.delay


WORKLOADS = (
    Workload("small_w4096_m5", 4_096, 5, 1, 2026080311),
    Workload("crossover_w8192_m5", 8_192, 5, 1, 2026080314),
    Workload("crossover_w16384_m3", 16_384, 3, 1, 2026080315),
    Workload("crossover_w16384_m5", 16_384, 5, 1, 2026080316),
    Workload("crossover_w16384_m7", 16_384, 7, 1, 2026080317),
    Workload("medium_w32768_m2", 32_768, 2, 1, 2026080318),
    Workload("medium_w32768_m3", 32_768, 3, 1, 2026080319),
    Workload("medium_w32768_m4", 32_768, 4, 1, 2026080322),
    Workload("medium_w32768_m5", 32_768, 5, 1, 2026080312),
    Workload("medium_w32768_m6", 32_768, 6, 1, 2026080323),
    Workload("medium_w32768_m7", 32_768, 7, 1, 2026080320),
    Workload("medium_w32768_m8", 32_768, 8, 1, 2026080324),
    Workload("medium_w32768_m9", 32_768, 9, 1, 2026080325),
    Workload("medium_w32768_m10", 32_768, 10, 1, 2026080321),
    Workload("large_w131072_m2", 131_072, 2, 1, 2026080326),
    Workload("large_w131072_m3", 131_072, 3, 1, 2026080327),
    Workload(
        "large_w131072_m5",
        131_072,
        5,
        1,
        2026080313,
    ),
    Workload("large_w131072_m7", 131_072, 7, 1, 2026080328),
    Workload("large_w131072_m8", 131_072, 8, 1, 2026080329),
    Workload("large_w131072_m9", 131_072, 9, 1, 2026080330),
    Workload("large_w131072_m10", 131_072, 10, 1, 2026080331),
)


def _make_signal(workload: Workload) -> np.ndarray:
    """Return a reproducible correlated signal with practically no exact ties."""

    rng = np.random.default_rng(workload.seed)
    innovations = rng.normal(scale=0.35, size=workload.sample_count)
    signal = np.empty(workload.sample_count, dtype=np.float64)
    signal[0] = innovations[0]
    for index in range(1, signal.size):
        signal[index] = (
            0.86 * signal[index - 1]
            + innovations[index]
            + 0.08 * np.sin(0.017 * index)
        )
    return np.ascontiguousarray(signal)


def _run_backend(
    signal: np.ndarray,
    workload: Workload,
    backend: str,
) -> OrdinalPatternDistribution:
    result = ordinal_pattern_distribution(
        signal,
        embedding_dimension=workload.embedding_dimension,
        delay=workload.delay,
        tie_policy=TIE_POLICY,
        backend=backend,
        fallback=False,
        sampling="deterministic synthetic correlated samples in row order",
        projection="one supplied scalar observable",
    )
    if result.backend != backend:
        raise RuntimeError(
            f"requested backend {backend!r}, but execution used {result.backend!r}"
        )
    return result


def _timed_call(
    operation: Callable[[], OrdinalPatternDistribution],
) -> tuple[float, OrdinalPatternDistribution]:
    started_ns = time.perf_counter_ns()
    result = operation()
    return (time.perf_counter_ns() - started_ns) * 1.0e-9, result


def _run_count_kernel(
    signal: np.ndarray,
    workload: Workload,
    backend: str,
) -> np.ndarray:
    """Run only the backend-specific count layer used by the auto policy."""

    if backend == "python":
        counts, _, _ = _ordinal_pattern_counts_python(
            signal,
            workload.embedding_dimension,
            workload.delay,
            0,
        )
        return np.asarray(counts, dtype=np.uint64)
    if backend == "numba":
        counts, _, _ = _ordinal_pattern_counts_numba(
            signal,
            workload.embedding_dimension,
            workload.delay,
            0,
        )
        return np.asarray(counts, dtype=np.uint64)
    native = native_permutation_counts(
        signal,
        embedding_dimension=workload.embedding_dimension,
        delay=workload.delay,
        tie_policy=TIE_POLICY,
        fallback=False,
    )
    if native.backend != "native_c":
        raise RuntimeError("native count-kernel benchmark unexpectedly fell back")
    return np.asarray(native.counts, dtype=np.uint64)


def _timed_count_call(
    operation: Callable[[], np.ndarray],
) -> tuple[float, np.ndarray]:
    started_ns = time.perf_counter_ns()
    result = operation()
    return (time.perf_counter_ns() - started_ns) * 1.0e-9, result


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


def _native_build(result: OrdinalPatternDistribution) -> dict[str, object]:
    raw = result.metadata.get("native_build")
    if not isinstance(raw, Mapping):
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
    return {key: raw.get(key) for key in keys}


def _benchmark_workload(
    workload: Workload,
    *,
    repeats: int,
    warmup_calls: int,
) -> tuple[dict[str, object], dict[str, object]]:
    signal = _make_signal(workload)
    operations = {
        backend: (
            lambda backend=backend: _run_backend(signal, workload, backend)
        )
        for backend in BACKENDS
    }

    warmup_samples: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
    warmup_results: dict[str, OrdinalPatternDistribution] = {}
    for backend in BACKENDS:
        for _ in range(warmup_calls):
            elapsed, result = _timed_call(operations[backend])
            warmup_samples[backend].append(elapsed)
            warmup_results[backend] = result

    reference_counts = warmup_results["python"].counts
    for backend in BACKENDS[1:]:
        np.testing.assert_array_equal(warmup_results[backend].counts, reference_counts)

    measured: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
    last_results = dict(warmup_results)
    for repeat_index in range(repeats):
        rotation = repeat_index % len(BACKENDS)
        order = BACKENDS[rotation:] + BACKENDS[:rotation]
        for backend in order:
            gc.collect()
            elapsed, result = _timed_call(operations[backend])
            measured[backend].append(elapsed)
            last_results[backend] = result

    for backend in BACKENDS[1:]:
        np.testing.assert_array_equal(last_results[backend].counts, reference_counts)

    kernel_operations = {
        backend: (
            lambda backend=backend: _run_count_kernel(signal, workload, backend)
        )
        for backend in BACKENDS
    }
    kernel_warmup_samples: dict[str, list[float]] = {
        backend: [] for backend in BACKENDS
    }
    kernel_warmup_results: dict[str, np.ndarray] = {}
    for backend in BACKENDS:
        for _ in range(warmup_calls):
            elapsed, counts = _timed_count_call(kernel_operations[backend])
            kernel_warmup_samples[backend].append(elapsed)
            kernel_warmup_results[backend] = counts
    for backend in BACKENDS:
        np.testing.assert_array_equal(kernel_warmup_results[backend], reference_counts)

    kernel_measured: dict[str, list[float]] = {
        backend: [] for backend in BACKENDS
    }
    for repeat_index in range(repeats):
        rotation = repeat_index % len(BACKENDS)
        order = BACKENDS[rotation:] + BACKENDS[:rotation]
        for backend in order:
            gc.collect()
            elapsed, counts = _timed_count_call(kernel_operations[backend])
            kernel_measured[backend].append(elapsed)
            np.testing.assert_array_equal(counts, reference_counts)

    auto = ordinal_pattern_distribution(
        signal,
        embedding_dimension=workload.embedding_dimension,
        delay=workload.delay,
        tie_policy=TIE_POLICY,
        backend="auto",
        fallback=False,
        sampling="deterministic synthetic correlated samples in row order",
        projection="one supplied scalar observable",
    )
    np.testing.assert_array_equal(auto.counts, reference_counts)

    timing = {
        backend: {
            "warmup": _summary(warmup_samples[backend]),
            "measurement": _summary(measured[backend]),
        }
        for backend in BACKENDS
    }
    medians = {
        backend: float(timing[backend]["measurement"]["median_seconds"])
        for backend in BACKENDS
    }
    kernel_timing = {
        backend: {
            "warmup": _summary(kernel_warmup_samples[backend]),
            "measurement": _summary(kernel_measured[backend]),
        }
        for backend in BACKENDS
    }
    kernel_medians = {
        backend: float(
            kernel_timing[backend]["measurement"]["median_seconds"]
        )
        for backend in BACKENDS
    }
    record = {
        "workload": {**asdict(workload), "sample_count": workload.sample_count},
        "input": {
            "generator": "Gaussian innovations, AR coefficient 0.86, sinusoidal drift",
            "signal_sha256": _array_sha256(signal),
        },
        "tie_policy": TIE_POLICY,
        "possible_patterns": int(reference_counts.size),
        "observed_patterns": int(np.count_nonzero(reference_counts)),
        "identical_counts": True,
        "counts_sha256": _array_sha256(reference_counts),
        "auto_selected_backend": auto.backend,
        "public_pipeline_backends": timing,
        "backends": timing,
        "count_kernel_backends": kernel_timing,
        "median_time_ratios": {
            "python_over_numba": medians["python"] / medians["numba"],
            "python_over_native_c": medians["python"] / medians["native_c"],
            "numba_over_native_c": medians["numba"] / medians["native_c"],
        },
        "count_kernel_median_time_ratios": {
            "python_over_numba": (
                kernel_medians["python"] / kernel_medians["numba"]
            ),
            "python_over_native_c": (
                kernel_medians["python"] / kernel_medians["native_c"]
            ),
            "numba_over_native_c": (
                kernel_medians["numba"] / kernel_medians["native_c"]
            ),
        },
        "ratio_interpretation": (
            "numerator median divided by denominator median for this run; "
            "greater than one means the numerator took longer"
        ),
    }
    return record, _native_build(last_results["native_c"])


def _threshold_assessment(workloads: list[dict[str, object]]) -> dict[str, object]:
    native_policy_records: list[dict[str, object]] = []
    numba_policy_records: list[dict[str, object]] = []
    for record in workloads:
        windows = int(record["workload"]["total_windows"])
        dimension = int(record["workload"]["embedding_dimension"])
        threshold = PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS[dimension]
        target = (
            native_policy_records
            if threshold is not None and windows >= threshold
            else numba_policy_records
        )
        target.append(record)

    threshold_native_wins = bool(native_policy_records) and all(
        float(record["count_kernel_median_time_ratios"]["numba_over_native_c"])
        >= 1.0 + PRACTICAL_EQUIVALENCE_RELATIVE
        for record in native_policy_records
    )
    disabled_native_records = [
        record
        for record in numba_policy_records
        if PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS[
            int(record["workload"]["embedding_dimension"])
        ]
        is None
    ]
    disabled_numba_wins = bool(disabled_native_records) and all(
        float(record["count_kernel_median_time_ratios"]["numba_over_native_c"])
        <= 1.0 + PRACTICAL_EQUIVALENCE_RELATIVE
        for record in disabled_native_records
    )
    conservative_kernel_opportunities = [
        str(record["workload"]["name"])
        for record in numba_policy_records
        if (
            PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS[
                int(record["workload"]["embedding_dimension"])
            ]
            is not None
            and int(record["workload"]["total_windows"])
            < int(
                PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS[
                    int(record["workload"]["embedding_dimension"])
                ]
            )
            and
            float(
                record["count_kernel_median_time_ratios"][
                    "numba_over_native_c"
                ]
            )
            > 1.0 + PRACTICAL_EQUIVALENCE_RELATIVE
        )
    ]
    auto_matches_policy = all(
        record["auto_selected_backend"]
        == (
            "native_c"
            if (
                PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS[
                    int(record["workload"]["embedding_dimension"])
                ]
                is not None
                and int(record["workload"]["total_windows"])
                >= int(
                    PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS[
                        int(record["workload"]["embedding_dimension"])
                    ]
                )
            )
            else "numba"
        )
        for record in workloads
    )
    supported = (
        threshold_native_wins
        and disabled_numba_wins
        and auto_matches_policy
    )
    return {
        "minimum_configured_threshold_windows": (
            PERMUTATION_ENTROPY_NATIVE_AUTO_MIN_WINDOWS
        ),
        "configured_thresholds_windows": {
            str(dimension): threshold
            for dimension, threshold in (
                PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS.items()
            )
        },
        "assessment": (
            "supported_on_this_host_for_measured_workloads"
            if supported
            else "not_supported_on_this_host_for_measured_workloads"
        ),
        "practical_equivalence_relative": PRACTICAL_EQUIVALENCE_RELATIVE,
        "native_materially_faster_at_or_above_threshold": threshold_native_wins,
        "numba_not_slower_when_native_auto_is_disabled": disabled_numba_wins,
        "conservative_kernel_opportunities_below_threshold": (
            conservative_kernel_opportunities
        ),
        "below_threshold_policy": (
            "conservative crossover guard; isolated faster-C points are reported "
            "but do not redefine a threshold without repeatable separation"
        ),
        "auto_selection_matches_configured_policy": auto_matches_policy,
        "scope": (
            "native selections require a material count-kernel advantage; "
            "dimensions with native auto disabled require a Numba advantage; "
            "lower crossovers remain conservative for deterministic m=2 "
            "through m=10 workloads on this host only; "
            "rerun before changing a portable auto policy"
        ),
    }


def _host_record() -> dict[str, object]:
    clock = time.get_clock_info("perf_counter")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
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


def run_benchmark(*, repeats: int = 5, warmup_calls: int = 1) -> dict[str, object]:
    """Run fixed workloads and return a JSON-serializable measurement record."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 3:
        raise ValueError("repeats must be an integer of at least 3")
    if (
        isinstance(warmup_calls, bool)
        or not isinstance(warmup_calls, int)
        or warmup_calls < 1
    ):
        raise ValueError("warmup_calls must be a positive integer")

    records: list[dict[str, object]] = []
    native_build: dict[str, object] | None = None
    for workload in WORKLOADS:
        record, current_build = _benchmark_workload(
            workload,
            repeats=repeats,
            warmup_calls=warmup_calls,
        )
        records.append(record)
        if native_build is None:
            native_build = current_build

    script_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "benchmark": "hafo_permutation_entropy_backends",
        "scope": "finite_bandt_pompe_ordinal_histogram_software_performance",
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
            "timing_scopes": (
                "public pipeline including common contract/provenance work; "
                "backend-specific ordinal count kernel used for auto threshold"
            ),
        },
        "host": _host_record(),
        "native_build": native_build,
        "auto_threshold_assessment": _threshold_assessment(records),
        "workloads": records,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
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
