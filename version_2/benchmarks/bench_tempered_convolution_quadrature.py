"""Parity-gated benchmark for HAFO tempered convolution quadrature.

The protocol compares the public direct-Python, warmed direct-Numba, and
zero-padded FFT batch backends for both HAFO tempered definitions and for
Lubich BDF1/BDF2 weights.  Every backend is evaluated on the same deterministic
sample history, component-wise fractional orders, and component-wise
tempering parameters.  Numerical parity is required before timing begins.

The measured scope is a complete sampled-operator call: validation, weight
generation, damping, optional conjugated-Caputo correction, scaling, and
convolution are all included.  It is not an FDE solve.  The FFT backend is a
one-shot linear batch convolution; it is not a streaming, online, or
fast-history algorithm.

Timings are host- and workload-specific engineering evidence.  Since this
protocol implements and measures neither a native-C nor a Julia candidate, it
cannot justify adding either backend.  Such a decision requires an isolated,
independently verified candidate and representative end-to-end profiling.
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

from hidden_attractors.fractional.tempered_convolution_quadrature import (
    TEMPERED_CAPUTO_INITIAL_CONDITION,
    TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
    TemperedConvolutionQuadratureResult,
    tempered_convolution_quadrature,
)


BACKENDS = ("python", "numba", "fft")
PARITY_RTOL = 5.0e-11
PARITY_ATOL = 5.0e-10
STEP = 0.004


@dataclass(frozen=True, slots=True)
class Workload:
    """Deterministic sampled-history workload."""

    name: str
    n_times: int
    dimension: int
    seed: int


@dataclass(frozen=True, slots=True)
class OperatorCase:
    """One tempered definition and Lubich generating polynomial."""

    name: str
    definition: str
    bdf_order: int
    initial_condition_semantics: str


@dataclass(frozen=True, slots=True)
class Problem:
    """Fully materialized, reusable benchmark input."""

    samples: np.ndarray
    orders: np.ndarray
    tempering: np.ndarray
    step: float


WORKLOADS = (
    Workload("small_n64_d2", 64, 2, 2026080311),
    Workload("medium_n256_d3", 256, 3, 2026080312),
    Workload("large_n768_d4", 768, 4, 2026080313),
)

OPERATOR_CASES = (
    OperatorCase(
        "tempered_rl_bdf1",
        "tempered_riemann_liouville",
        1,
        TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
    ),
    OperatorCase(
        "tempered_rl_bdf2",
        "tempered_riemann_liouville",
        2,
        TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
    ),
    OperatorCase(
        "tempered_caputo_bdf1",
        "tempered_caputo",
        1,
        TEMPERED_CAPUTO_INITIAL_CONDITION,
    ),
    OperatorCase(
        "tempered_caputo_bdf2",
        "tempered_caputo",
        2,
        TEMPERED_CAPUTO_INITIAL_CONDITION,
    ),
)

_WARMUP_WORKLOAD = Workload("numba_warmup_excluded", 8, 2, 2026080310)


def _validate_workload(workload: Workload) -> None:
    if not isinstance(workload.name, str) or not workload.name:
        raise ValueError("workload.name must be a non-empty string")
    if (
        isinstance(workload.n_times, bool)
        or not isinstance(workload.n_times, int)
        or workload.n_times < 2
    ):
        raise ValueError("workload.n_times must be an integer of at least 2")
    if (
        isinstance(workload.dimension, bool)
        or not isinstance(workload.dimension, int)
        or workload.dimension < 1
    ):
        raise ValueError("workload.dimension must be a positive integer")
    if isinstance(workload.seed, bool) or not isinstance(workload.seed, int):
        raise ValueError("workload.seed must be an integer")


def _make_problem(workload: Workload) -> Problem:
    """Build a fixed, nontrivial smooth-plus-noise sampled history."""

    _validate_workload(workload)
    rng = np.random.default_rng(workload.seed)
    tau = STEP * np.arange(workload.n_times, dtype=np.float64)[:, None]
    component = np.arange(workload.dimension, dtype=np.float64)[None, :]
    phase = rng.uniform(-0.7, 0.7, size=(1, workload.dimension))
    frequency = 0.8 + 0.37 * component
    offset = 0.35 + 0.11 * component
    smooth = (
        offset
        + (0.7 + 0.05 * component) * np.sin(frequency * tau + phase)
        + 0.23 * np.cos((1.9 + 0.21 * component) * tau - 0.5 * phase)
        + 0.04 * tau * (1.0 + component)
    )
    innovations = rng.normal(0.0, 1.0, size=smooth.shape)
    stochastic = 0.0025 * np.cumsum(innovations, axis=0)
    samples = np.ascontiguousarray(smooth + stochastic, dtype=np.float64)
    orders = np.ascontiguousarray(
        np.linspace(0.43, 0.91, workload.dimension, dtype=np.float64)
    )
    tempering = np.ascontiguousarray(
        np.linspace(0.17, 1.07, workload.dimension, dtype=np.float64)
    )
    return Problem(samples=samples, orders=orders, tempering=tempering, step=STEP)


def _run_case(
    problem: Problem,
    operator_case: OperatorCase,
    backend: str,
) -> TemperedConvolutionQuadratureResult:
    return tempered_convolution_quadrature(
        problem.samples,
        problem.orders,
        tempering=problem.tempering,
        bdf_order=operator_case.bdf_order,
        definition=operator_case.definition,
        step=problem.step,
        initial_condition_semantics=operator_case.initial_condition_semantics,
        backend=backend,
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


def _assert_parity(
    results: dict[str, TemperedConvolutionQuadratureResult],
    workload: Workload,
    operator_case: OperatorCase,
) -> dict[str, object]:
    reference = results["python"]
    comparisons: dict[str, object] = {}
    for backend in ("numba", "fft"):
        candidate = results[backend]
        if candidate.backend != backend:
            raise RuntimeError(
                f"backend mismatch for {workload.name}/{operator_case.name}: "
                f"requested={backend}, returned={candidate.backend}"
            )
        if (
            candidate.definition != reference.definition
            or candidate.bdf_order != reference.bdf_order
            or candidate.initial_condition_semantics
            != reference.initial_condition_semantics
        ):
            raise RuntimeError(
                f"semantic metadata mismatch for "
                f"{workload.name}/{operator_case.name}/{backend}"
            )
        np.testing.assert_array_equal(candidate.times, reference.times)
        np.testing.assert_array_equal(candidate.orders, reference.orders)
        np.testing.assert_array_equal(candidate.tempering, reference.tempering)
        np.testing.assert_array_equal(candidate.weights, reference.weights)
        np.testing.assert_array_equal(candidate.base_weights, reference.base_weights)
        np.testing.assert_allclose(
            candidate.values,
            reference.values,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
            err_msg=(
                f"tempered CQ mismatch for "
                f"{workload.name}/{operator_case.name}/{backend}"
            ),
        )
        comparisons[backend] = {
            "maximum_absolute_value_difference_from_python": (
                _maximum_absolute_difference(candidate.values, reference.values)
            ),
            "weights_exactly_equal_to_python": True,
            "base_weights_exactly_equal_to_python": True,
        }
    return {
        "reference_backend": "python",
        "rtol": PARITY_RTOL,
        "atol": PARITY_ATOL,
        "comparisons": comparisons,
    }


def _measure_rotating(
    operations: dict[str, Callable[[], TemperedConvolutionQuadratureResult]],
    *,
    repeats: int,
    workload: Workload,
    operator_case: OperatorCase,
) -> tuple[dict[str, dict[str, object]], list[list[str]]]:
    samples: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
    execution_orders: list[list[str]] = []
    for repeat in range(repeats):
        shift = repeat % len(BACKENDS)
        order = BACKENDS[shift:] + BACKENDS[:shift]
        execution_orders.append(list(order))
        for backend in order:
            gc.collect()
            elapsed, result = _timed(operations[backend])
            if (
                result.backend != backend
                or result.definition != operator_case.definition
                or result.bdf_order != operator_case.bdf_order
                or result.values.shape
                != (workload.n_times, workload.dimension)
                or not np.all(np.isfinite(result.values))
            ):
                raise RuntimeError(
                    f"invalid timed result for "
                    f"{workload.name}/{operator_case.name}/{backend}"
                )
            samples[backend].append(elapsed)
    return (
        {backend: _summary(samples[backend]) for backend in BACKENDS},
        execution_orders,
    )


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _benchmark_case(
    problem: Problem,
    workload: Workload,
    operator_case: OperatorCase,
    *,
    repeats: int,
) -> dict[str, object]:
    parity_results = {
        backend: _run_case(problem, operator_case, backend)
        for backend in BACKENDS
    }
    parity = _assert_parity(parity_results, workload, operator_case)
    operations = {
        backend: (
            lambda backend=backend: _run_case(problem, operator_case, backend)
        )
        for backend in BACKENDS
    }
    timing, execution_orders = _measure_rotating(
        operations,
        repeats=repeats,
        workload=workload,
        operator_case=operator_case,
    )
    medians = {
        backend: float(timing[backend]["median_seconds"])
        for backend in BACKENDS
    }
    return {
        "operator_case": asdict(operator_case),
        "parity_checked_before_measurement": True,
        "parity": parity,
        "timing": timing,
        "backend_execution_order_by_repetition": execution_orders,
        "finite_run_ratios": {
            "python_over_numba_median": medians["python"] / medians["numba"],
            "python_over_fft_median": medians["python"] / medians["fft"],
            "numba_over_fft_median": medians["numba"] / medians["fft"],
        },
        "observed_fastest_backend_by_median": min(medians, key=medians.get),
        "ratio_interpretation": (
            "numerator median divided by denominator median for this finite "
            "host run; greater than one means the numerator took longer"
        ),
        "fft_scope": (
            "offline zero-padded one-shot linear batch convolution; not "
            "streaming, online, fast-history, or an FDE solver"
        ),
    }


def _benchmark_workload(workload: Workload, *, repeats: int) -> dict[str, object]:
    problem = _make_problem(workload)
    return {
        "workload": asdict(workload),
        "fixture": {
            "description": (
                "fixed smooth multi-frequency history plus a seeded low-amplitude "
                "integrated perturbation; nonzero initial values"
            ),
            "step": problem.step,
            "orders": [float(value) for value in problem.orders],
            "tempering": [float(value) for value in problem.tempering],
            "input_sha256": _hash_arrays(
                problem.samples, problem.orders, problem.tempering
            ),
        },
        "cases": [
            _benchmark_case(
                problem,
                workload,
                operator_case,
                repeats=repeats,
            )
            for operator_case in OPERATOR_CASES
        ],
    }


def _backend_policy_assessment(
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    observed_winners = {backend: 0 for backend in BACKENDS}
    for record in records:
        for case in record["cases"]:
            observed_winners[str(case["observed_fastest_backend_by_median"])] += 1
    return {
        "decision": "do_not_add_native_c_or_julia_from_this_evidence",
        "retain_backends": ["python_direct", "numba_direct", "fft_batch"],
        "native_c_candidate_implemented_or_measured": False,
        "julia_candidate_implemented_or_measured": False,
        "observed_fastest_backend_counts": observed_winners,
        "fft_contract": (
            "FFT is an offline zero-padded linear batch convolution. It is not "
            "a fast-history compression, streaming recurrence, online operator, "
            "or fractional differential-equation solver."
        ),
        "reasoning": (
            "The measurements compare only the three existing public software "
            "routes. They provide no C or Julia implementation, parity result, "
            "build cost, call-overhead measurement, or end-to-end application "
            "profile, so they cannot establish that either additional language "
            "would improve HAFO. A finite fastest-backend count is descriptive, "
            "not a universal crossover rule."
        ),
        "c_admission_gate": (
            "First profile representative HAFO and Toolbox Chaos workloads. Add "
            "a native-C candidate only if a residual convolution bottleneck "
            "remains after warmed Numba/FFT selection, then require independent "
            "parity, compiler/build provenance, repeated kernel and end-to-end "
            "timings, and a measured gain larger than inter-run noise."
        ),
        "julia_admission_gate": (
            "Use Julia only when a materially better algorithm or maintained "
            "implementation cannot be reproduced efficiently in the Python/" 
            "Numba/C stack, and measure process startup, data transfer, deployment "
            "cost, parity, and end-to-end benefit before adopting the bridge."
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
        "perf_counter": {
            "implementation": clock.implementation,
            "resolution_seconds": clock.resolution,
            "monotonic": clock.monotonic,
        },
    }


def run_benchmark(
    *,
    repeats: int = 5,
    workloads: Sequence[Workload] = WORKLOADS,
) -> dict[str, object]:
    """Run deterministic parity gates and return a portable JSON-safe record."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 3:
        raise ValueError("repeats must be an integer of at least 3")
    selected_workloads = tuple(workloads)
    if not selected_workloads:
        raise ValueError("workloads must not be empty")
    for workload in selected_workloads:
        _validate_workload(workload)

    warmup_problem = _make_problem(_WARMUP_WORKLOAD)
    warmup_case = OPERATOR_CASES[0]
    numba_first_call_seconds, warmup_result = _timed(
        lambda: _run_case(warmup_problem, warmup_case, "numba")
    )
    if warmup_result.backend != "numba":
        raise RuntimeError("the explicit Numba warm-up did not use Numba")
    numba_warmed_confirmation_seconds, warmed_result = _timed(
        lambda: _run_case(warmup_problem, warmup_case, "numba")
    )
    np.testing.assert_array_equal(warmed_result.values, warmup_result.values)

    records = [
        _benchmark_workload(workload, repeats=repeats)
        for workload in selected_workloads
    ]
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "benchmark_id": "hafo_tempered_convolution_quadrature_backends_20260803",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": script_sha256,
        "scope": "tempered_fractional_sampled_operator_backend_performance",
        "evidence_boundary": (
            "host- and workload-specific finite software timing only; no FDE "
            "solver, convergence-order, stability, dynamics, chaos, attraction, "
            "or hiddenness claim"
        ),
        "portable_json_contract": (
            "the payload contains no repository, source, output, cache, or "
            "temporary absolute filesystem path"
        ),
        "measurement_protocol": {
            "public_call_scope": (
                "validation, BDF weight generation, exponential damping, "
                "optional Caputo correction, scaling, and convolution"
            ),
            "definitions": [
                "tempered_riemann_liouville",
                "tempered_caputo",
            ],
            "bdf_orders": [1, 2],
            "backends": list(BACKENDS),
            "backend_parity_before_measurement": True,
            "parity_rtol": PARITY_RTOL,
            "parity_atol": PARITY_ATOL,
            "numba_first_public_call_seconds_including_initialization_or_jit": (
                numba_first_call_seconds
            ),
            "numba_warmed_confirmation_seconds_excluded_from_measurements": (
                numba_warmed_confirmation_seconds
            ),
            "numba_warmup_workload_excluded_from_measurements": True,
            "measured_repetitions_per_backend_case": repeats,
            "backend_order": (
                "Python/Numba/FFT cyclically rotated across repetitions"
            ),
            "garbage_collection": "collected before and outside each timed call",
            "clock": "time.perf_counter_ns",
            "fft_scope": (
                "zero-padded offline linear batch convolution; not fast-history"
            ),
        },
        "host": _host_record(),
        "workloads": records,
        "backend_policy_assessment": _backend_policy_assessment(records),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
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
