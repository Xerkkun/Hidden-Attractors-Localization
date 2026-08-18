"""Benchmark integer-order CLV backends on synthetic tangent histories.

The benchmark builds deterministic, dynamically consistent positive-diagonal
QR histories from constant nonnormal linear maps.  Backend parity is enforced
before any timing ratio is reported.  Numba runtime initialization/JIT is
measured once and excluded from the repeated measurements.

Two timing scopes are kept separate: reconstruction through the public QR
history API, and the complete public integer-map CLV pipeline (map/Jacobian
calls, QR propagation, validation, allocation, and backward reconstruction).
Results are host- and workload-specific software-engineering evidence.  They
do not establish universal backend superiority or certify chaos, attraction,
hiddenness, hyperbolicity, or fractional-memory dynamics.
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
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numba
import numpy as np
import scipy

from hidden_attractors.analysis.covariant_lyapunov import (
    CovariantLyapunovResult,
    CovariantQRHistoryResult,
    integer_covariant_vectors_from_qr_history,
    integer_map_covariant_lyapunov_vectors,
)


BACKENDS = ("numpy", "numba")
PARITY_RTOL = 2.0e-12
PARITY_ATOL = 2.0e-13


@dataclass(frozen=True, slots=True)
class Workload:
    """One deterministic finite QR-history and integer-map workload."""

    name: str
    observed_segments: int
    future_segments: int
    dimension: int
    n_vectors: int
    seed: int


@dataclass(frozen=True, slots=True)
class LinearCLVProblem:
    """Inputs shared by the direct-history and end-to-end measurements."""

    matrix: np.ndarray
    initial_basis: np.ndarray
    terminal_coefficients: np.ndarray
    orthonormal_bases: np.ndarray
    observed_r_factors: np.ndarray
    future_r_factors: np.ndarray


WORKLOADS = (
    Workload("small_s64_f16_k4_d8", 64, 16, 8, 4, 2026080341),
    Workload("medium_s512_f128_k4_d8", 512, 128, 8, 4, 2026080342),
    Workload("large_s4096_f1024_k4_d8", 4_096, 1_024, 8, 4, 2026080343),
)

_WARMUP_WORKLOAD = Workload("jit_warmup", 4, 2, 4, 2, 2026080340)


def _positive_reduced_qr(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    basis, upper = np.linalg.qr(values, mode="reduced")
    signs = np.where(np.diag(upper) < 0.0, -1.0, 1.0)
    basis = basis * signs[None, :]
    upper = signs[:, None] * upper
    return np.ascontiguousarray(basis), np.ascontiguousarray(upper)


def _make_problem(workload: Workload) -> LinearCLVProblem:
    """Build one reproducible cocycle and its exact finite QR history."""

    if workload.observed_segments < 1 or workload.future_segments < 0:
        raise ValueError("observed_segments must be positive and future_segments nonnegative")
    if not 1 <= workload.n_vectors <= workload.dimension:
        raise ValueError("n_vectors must satisfy 1 <= n_vectors <= dimension")

    generator = np.random.default_rng(workload.seed)
    coordinate_basis, _ = np.linalg.qr(
        generator.standard_normal((workload.dimension, workload.dimension))
    )
    logarithmic_rates = np.linspace(2.0e-3, -2.0e-3, workload.dimension)
    schur_form = np.diag(np.exp(logarithmic_rates))
    schur_form += np.triu(
        generator.normal(scale=2.5e-3, size=schur_form.shape), k=1
    )
    matrix = np.ascontiguousarray(
        coordinate_basis @ schur_form @ coordinate_basis.T,
        dtype=np.float64,
    )

    initial_basis, _ = _positive_reduced_qr(
        generator.standard_normal((workload.dimension, workload.n_vectors))
    )
    terminal = np.triu(
        generator.normal(scale=0.2, size=(workload.n_vectors, workload.n_vectors))
    )
    np.fill_diagonal(terminal, 1.0)
    terminal = np.ascontiguousarray(terminal, dtype=np.float64)

    bases = np.empty(
        (
            workload.observed_segments + 1,
            workload.dimension,
            workload.n_vectors,
        ),
        dtype=np.float64,
    )
    observed = np.empty(
        (
            workload.observed_segments,
            workload.n_vectors,
            workload.n_vectors,
        ),
        dtype=np.float64,
    )
    future = np.empty(
        (
            workload.future_segments,
            workload.n_vectors,
            workload.n_vectors,
        ),
        dtype=np.float64,
    )
    bases[0] = initial_basis
    current_basis = initial_basis
    for segment in range(workload.observed_segments):
        current_basis, upper = _positive_reduced_qr(matrix @ current_basis)
        observed[segment] = upper
        bases[segment + 1] = current_basis
    for segment in range(workload.future_segments):
        current_basis, upper = _positive_reduced_qr(matrix @ current_basis)
        future[segment] = upper

    return LinearCLVProblem(
        matrix=matrix,
        initial_basis=initial_basis,
        terminal_coefficients=terminal,
        orthonormal_bases=np.ascontiguousarray(bases),
        observed_r_factors=np.ascontiguousarray(observed),
        future_r_factors=np.ascontiguousarray(future),
    )


def _run_history(
    problem: LinearCLVProblem,
    backend: str,
) -> CovariantQRHistoryResult:
    return integer_covariant_vectors_from_qr_history(
        problem.orthonormal_bases,
        problem.observed_r_factors,
        future_r_factors=problem.future_r_factors,
        terminal_coefficients=problem.terminal_coefficients,
        backend=backend,
        q=1.0,
    )


def _run_end_to_end(
    problem: LinearCLVProblem,
    workload: Workload,
    backend: str,
) -> CovariantLyapunovResult:
    matrix = problem.matrix
    return integer_map_covariant_lyapunov_vectors(
        lambda state: matrix @ state,
        lambda _state: matrix,
        np.zeros(workload.dimension, dtype=np.float64),
        iterations=workload.observed_segments,
        backward_transient_iterations=workload.future_segments,
        n_vectors=workload.n_vectors,
        initial_basis=problem.initial_basis,
        terminal_coefficients=problem.terminal_coefficients,
        qr_interval_iterations=1,
        backend=backend,
        q=1.0,
    )


def _timed(operation: Callable[[], Any]) -> tuple[float, Any]:
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


def _maximum_absolute_difference(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.max(np.abs(first - second))) if first.size else 0.0


def _assert_history_parity(
    numpy_result: CovariantQRHistoryResult,
    numba_result: CovariantQRHistoryResult,
    workload: Workload,
) -> dict[str, float]:
    vector_difference = _maximum_absolute_difference(
        numpy_result.vectors, numba_result.vectors
    )
    coefficient_difference = _maximum_absolute_difference(
        numpy_result.coefficients, numba_result.coefficients
    )
    np.testing.assert_allclose(
        numba_result.vectors,
        numpy_result.vectors,
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
        err_msg=f"CLV vector backend mismatch for {workload.name}",
    )
    np.testing.assert_allclose(
        numba_result.coefficients,
        numpy_result.coefficients,
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
        err_msg=f"CLV coefficient backend mismatch for {workload.name}",
    )
    return {
        "maximum_absolute_vector_difference": vector_difference,
        "maximum_absolute_coefficient_difference": coefficient_difference,
    }


def _assert_end_to_end_parity(
    numpy_result: CovariantLyapunovResult,
    numba_result: CovariantLyapunovResult,
    workload: Workload,
) -> dict[str, float]:
    if numpy_result.status != "ok" or numba_result.status != "ok":
        raise RuntimeError(
            f"end-to-end CLV workload {workload.name} failed: "
            f"numpy={numpy_result.status}, numba={numba_result.status}"
        )
    vector_difference = _maximum_absolute_difference(
        numpy_result.vectors, numba_result.vectors
    )
    exponent_difference = _maximum_absolute_difference(
        numpy_result.exponents, numba_result.exponents
    )
    np.testing.assert_allclose(
        numba_result.vectors,
        numpy_result.vectors,
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
        err_msg=f"end-to-end CLV vector mismatch for {workload.name}",
    )
    np.testing.assert_allclose(
        numba_result.exponents,
        numpy_result.exponents,
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
        err_msg=f"end-to-end exponent mismatch for {workload.name}",
    )
    return {
        "maximum_absolute_vector_difference": vector_difference,
        "maximum_absolute_exponent_difference": exponent_difference,
    }


def _measure_alternating(
    operations: dict[str, Callable[[], Any]],
    *,
    repeats: int,
    validator: Callable[[Any, str], None],
) -> dict[str, dict[str, object]]:
    samples: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
    for repeat in range(repeats):
        order = BACKENDS if repeat % 2 == 0 else tuple(reversed(BACKENDS))
        for backend in order:
            gc.collect()
            elapsed, result = _timed(operations[backend])
            validator(result, backend)
            samples[backend].append(elapsed)
    return {backend: _summary(samples[backend]) for backend in BACKENDS}


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _benchmark_workload(workload: Workload, *, repeats: int) -> dict[str, object]:
    problem = _make_problem(workload)

    history_reference = _run_history(problem, "numpy")
    history_accelerated = _run_history(problem, "numba")
    history_parity = _assert_history_parity(
        history_reference, history_accelerated, workload
    )

    end_to_end_reference = _run_end_to_end(problem, workload, "numpy")
    end_to_end_accelerated = _run_end_to_end(problem, workload, "numba")
    end_to_end_parity = _assert_end_to_end_parity(
        end_to_end_reference, end_to_end_accelerated, workload
    )

    history_operations = {
        backend: (lambda backend=backend: _run_history(problem, backend))
        for backend in BACKENDS
    }

    def validate_history(result: CovariantQRHistoryResult, backend: str) -> None:
        if result.backend != backend:
            raise RuntimeError(
                f"history backend mismatch for {workload.name}: "
                f"requested={backend}, returned={result.backend}"
            )

    history_timing = _measure_alternating(
        history_operations,
        repeats=repeats,
        validator=validate_history,
    )

    end_to_end_operations = {
        backend: (
            lambda backend=backend: _run_end_to_end(problem, workload, backend)
        )
        for backend in BACKENDS
    }

    def validate_end_to_end(result: CovariantLyapunovResult, backend: str) -> None:
        if result.status != "ok" or result.backend != backend:
            raise RuntimeError(
                f"end-to-end result mismatch for {workload.name}: "
                f"status={result.status}, requested={backend}, returned={result.backend}"
            )

    end_to_end_timing = _measure_alternating(
        end_to_end_operations,
        repeats=repeats,
        validator=validate_end_to_end,
    )

    history_numpy = float(history_timing["numpy"]["median_seconds"])
    history_numba = float(history_timing["numba"]["median_seconds"])
    end_to_end_numpy = float(end_to_end_timing["numpy"]["median_seconds"])
    end_to_end_numba = float(end_to_end_timing["numba"]["median_seconds"])
    reconstruction_fraction = history_numba / end_to_end_numba
    idealized_zero_cost_reconstruction_speedup = (
        end_to_end_numba / (end_to_end_numba - history_numba)
        if history_numba < end_to_end_numba
        else None
    )
    return {
        "workload": asdict(workload),
        "fixture": {
            "description": (
                "constant deterministic nonnormal linear map; reduced QR at every "
                "iteration; strictly positive R diagonal"
            ),
            "input_sha256": _hash_arrays(
                problem.matrix,
                problem.initial_basis,
                problem.terminal_coefficients,
                problem.orthonormal_bases,
                problem.observed_r_factors,
                problem.future_r_factors,
            ),
        },
        "parity_checked_before_measurement": True,
        "history_public_api": {
            "parity": history_parity,
            "timing": history_timing,
            "numpy_over_numba_median_ratio": history_numpy / history_numba,
        },
        "integer_map_end_to_end": {
            "scope": (
                "public map/Jacobian callbacks, tangent propagation, reduced QR, "
                "validation, allocations, and backward CLV reconstruction"
            ),
            "parity": end_to_end_parity,
            "timing": end_to_end_timing,
            "numpy_over_numba_median_ratio": end_to_end_numpy / end_to_end_numba,
            "observed_numba_time_reduction_relative_to_numpy": (
                1.0 - end_to_end_numba / end_to_end_numpy
            ),
            "warmed_numba_public_reconstruction_fraction_of_end_to_end": (
                reconstruction_fraction
            ),
            "idealized_zero_cost_public_reconstruction_speedup_ceiling": (
                idealized_zero_cost_reconstruction_speedup
            ),
            "speedup_ceiling_interpretation": (
                "Ratio of independently measured medians. It assumes the entire "
                "public reconstruction phase costs zero, so it is an idealized "
                "engineering ceiling rather than a statistical confidence bound; "
                "a C kernel could replace only part of that phase."
            ),
        },
        "ratio_interpretation": (
            "NumPy median divided by warmed-Numba median; greater than one means "
            "NumPy took longer for this finite run"
        ),
    }


def _native_c_assessment(records: Sequence[dict[str, object]]) -> dict[str, object]:
    largest = records[-1]
    history_ratio = float(
        largest["history_public_api"]["numpy_over_numba_median_ratio"]
    )
    end_to_end_ratio = float(
        largest["integer_map_end_to_end"]["numpy_over_numba_median_ratio"]
    )
    reconstruction_fraction = float(
        largest["integer_map_end_to_end"][
            "warmed_numba_public_reconstruction_fraction_of_end_to_end"
        ]
    )
    zero_cost_ceiling = largest["integer_map_end_to_end"][
        "idealized_zero_cost_public_reconstruction_speedup_ceiling"
    ]
    return {
        "assessment": "insufficient_evidence_to_justify_a_native_c_clv_backend",
        "native_c_backend_implemented_or_measured": False,
        "current_recommendation": "retain_numpy_and_warmed_numba_backends",
        "largest_workload_observed_ratios": {
            "history_numpy_over_numba": history_ratio,
            "end_to_end_numpy_over_numba": end_to_end_ratio,
            "warmed_numba_public_reconstruction_fraction_of_end_to_end": (
                reconstruction_fraction
            ),
            "idealized_zero_cost_public_reconstruction_speedup_ceiling": (
                zero_cost_ceiling
            ),
        },
        "reasoning": (
            "This protocol measures NumPy versus Numba, including the complete "
            "integer-map pipeline, but contains no native-C implementation or "
            "native-C timing. The largest-workload record also bounds the entire "
            "warmed-Numba public reconstruction phase relative to end-to-end time. "
            "It cannot establish that C would outperform warmed Numba."
        ),
        "evidence_required_before_adding_c": (
            "Profile representative application systems first. If backward CLV "
            "reconstruction remains a material residual bottleneck, benchmark an "
            "independently verified C candidate with compilation separated, exact "
            "parity gates, multiple dimensions/horizons, and repeated end-to-end "
            "timings before changing the backend policy."
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
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "numba_threads": numba.get_num_threads(),
        "perf_counter": {
            "implementation": clock.implementation,
            "resolution_seconds": clock.resolution,
            "monotonic": clock.monotonic,
        },
    }


def run_benchmark(
    *,
    repeats: int = 7,
    workloads: Sequence[Workload] = WORKLOADS,
) -> dict[str, object]:
    """Run parity-gated CLV measurements and return a JSON-safe record."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 3:
        raise ValueError("repeats must be an integer of at least 3")
    selected_workloads = tuple(workloads)
    if not selected_workloads:
        raise ValueError("workloads must not be empty")

    warmup_problem = _make_problem(_WARMUP_WORKLOAD)
    numba_first_call_seconds, warmup_result = _timed(
        lambda: _run_history(warmup_problem, "numba")
    )
    if warmup_result.backend != "numba":
        raise RuntimeError("the explicit Numba warm-up did not use Numba")

    records = [
        _benchmark_workload(workload, repeats=repeats)
        for workload in selected_workloads
    ]
    script_path = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "benchmark_id": "hafo_covariant_lyapunov_numpy_numba_20260803",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "scope": "finite_q1_covariant_lyapunov_software_backend_performance",
        "evidence_boundary": (
            "host- and workload-specific engineering timing only; no universal "
            "backend ranking and no chaos, hyperbolicity, attraction, hiddenness, "
            "or fractional-memory claim"
        ),
        "measurement_protocol": {
            "backend_parity_before_measurement": True,
            "parity_rtol": PARITY_RTOL,
            "parity_atol": PARITY_ATOL,
            "numba_first_public_call_seconds_including_initialization_or_jit": (
                numba_first_call_seconds
            ),
            "numba_warmup_excluded_from_repeated_measurements": True,
            "measured_repetitions_per_backend_and_scope": repeats,
            "backend_order": "NumPy/Numba alternated and reversed each repetition",
            "garbage_collection": "collected before and outside each timed call",
            "clock": "time.perf_counter_ns",
            "timing_scopes": (
                "public QR-history reconstruction and complete public integer-map "
                "CLV pipeline"
            ),
        },
        "host": _host_record(),
        "workloads": records,
        "native_c_assessment": _native_c_assessment(records),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")

    payload = run_benchmark(repeats=args.repeats)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output)
        return
    print(rendered)


if __name__ == "__main__":
    main()
