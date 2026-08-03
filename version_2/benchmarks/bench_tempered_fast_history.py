"""Parity-gated end-to-end benchmark for tempered recurrent fast history.

The benchmark covers the public HAFO Fast Method II implementation with the
FBDF1 and GNGF2 generators, for raw tempered Riemann--Liouville and conjugated
tempered-Caputo sampled operators.  It is not an FDE-solver benchmark.

FBDF1 is compared with the public direct-Numba and zero-padded offline-FFT
convolution-quadrature routes.  GNGF2 is compared with independent exact-weight
direct and FFT batch references implemented in this file; GNGF2 is deliberately
not relabelled as fractional BDF2.  Fast Python and warmed Fast Numba use the
same calibrated quadrature-point count ``Q``.  Numerical parity is required
before any timing is recorded.

Automatic selection of ``Q`` and Numba initialization/JIT are measured
separately.  Repeated timings are complete public calls for a fixed, previously
validated ``Q`` and therefore include validation, exact local weights,
finite-grid weight calibration, recurrence construction, allocation, and the
returned sampled values.  Analytical active-memory models report ``N``, ``Q``,
``d``, and ``n0`` explicitly and exclude the common input and returned output.

FFT is an offline batch baseline, not an online or streaming fast-history
method.  No native-C or Julia candidate is implemented or measured here, so
this protocol cannot justify either backend by inference from Numba/FFT timing.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import tempfile
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
    tempered_convolution_quadrature,
)
from hidden_attractors.fractional.tempered_fast_history import (
    TemperedFastHistoryResult,
    tempered_fast_multistep_history,
)


FAST_BACKENDS = ("fast_python", "fast_numba")
FBDF1_BASELINES = ("direct_numba", "fft_batch")
GNGF2_BASELINES = ("direct_reference", "fft_reference")
PARITY_RTOL = 2.0e-11
PARITY_ATOL = 5.0e-10
FAST_BACKEND_RTOL = 2.0e-13
FAST_BACKEND_ATOL = 2.0e-12
STEP = 0.006
TAIL_CUTOFF = 1.0e-18
FLOAT64_BYTES = np.dtype(np.float64).itemsize
COMPLEX128_BYTES = np.dtype(np.complex128).itemsize


@dataclass(frozen=True, slots=True)
class Workload:
    """Deterministic history and compression-control workload."""

    name: str
    n_times: int
    dimension: int
    local_history_steps: int
    relative_tolerance: float
    max_quadrature_points: int
    seed: int


@dataclass(frozen=True, slots=True)
class DefinitionCase:
    """One explicit tempered initial-condition convention."""

    name: str
    definition: str
    initial_condition_semantics: str


@dataclass(frozen=True, slots=True)
class Problem:
    """Fully materialized benchmark input."""

    samples: np.ndarray
    orders: np.ndarray
    tempering: np.ndarray
    step: float


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    """Independent exact-weight GNGF2 batch evaluation."""

    values: np.ndarray
    base_weights: np.ndarray
    tempered_weights: np.ndarray
    backend: str
    fft_length: int | None


WORKLOADS = (
    Workload("small_n128_qauto_d2_n016", 128, 2, 16, 1.0e-6, 513, 2026080341),
    Workload("medium_n512_qauto_d3_n032", 512, 3, 32, 1.0e-6, 513, 2026080342),
    Workload("large_n2048_qauto_d4_n050", 2048, 4, 50, 1.0e-6, 1025, 2026080343),
)

DEFINITION_CASES = (
    DefinitionCase(
        "tempered_rl",
        "tempered_riemann_liouville",
        TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
    ),
    DefinitionCase(
        "tempered_caputo",
        "tempered_caputo",
        TEMPERED_CAPUTO_INITIAL_CONDITION,
    ),
)

MULTISTEP_METHODS = ("fbdf1", "gngf2")
_WARMUP_WORKLOAD = Workload(
    "numba_warmup_excluded", 24, 2, 4, 1.0e-4, 257, 2026080340
)


def _validate_workload(workload: Workload) -> None:
    if not isinstance(workload.name, str) or not workload.name:
        raise ValueError("workload.name must be a non-empty string")
    integer_fields = (
        ("n_times", workload.n_times, 4),
        ("dimension", workload.dimension, 1),
        ("local_history_steps", workload.local_history_steps, 2),
        ("max_quadrature_points", workload.max_quadrature_points, 65),
    )
    for name, value, minimum in integer_fields:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"workload.{name} must be an integer >= {minimum}")
    if workload.local_history_steps >= workload.n_times - 1:
        raise ValueError(
            "workload.local_history_steps must leave at least one compressed lag"
        )
    if (
        not np.isfinite(workload.relative_tolerance)
        or not 0.0 < workload.relative_tolerance < 1.0
    ):
        raise ValueError("workload.relative_tolerance must lie in (0, 1)")
    if isinstance(workload.seed, bool) or not isinstance(workload.seed, int):
        raise ValueError("workload.seed must be an integer")


def _make_problem(workload: Workload) -> Problem:
    """Return deterministic smooth, nonzero, componentwise sampled data."""

    _validate_workload(workload)
    rng = np.random.default_rng(workload.seed)
    tau = STEP * np.arange(workload.n_times, dtype=np.float64)[:, None]
    component = np.arange(workload.dimension, dtype=np.float64)[None, :]
    phase = rng.uniform(-0.55, 0.55, size=(1, workload.dimension))
    smooth = (
        0.45
        + 0.09 * component
        + (0.65 + 0.03 * component)
        * np.sin((0.7 + 0.31 * component) * tau + phase)
        + 0.21 * np.cos((1.5 + 0.17 * component) * tau - 0.4 * phase)
        + 0.025 * (1.0 + component) * tau**1.4
    )
    perturbation = 0.0015 * np.cumsum(
        rng.normal(size=smooth.shape), axis=0
    )
    samples = np.ascontiguousarray(smooth + perturbation, dtype=np.float64)
    orders = np.ascontiguousarray(
        np.linspace(0.41, 0.87, workload.dimension, dtype=np.float64)
    )
    tempering = np.ascontiguousarray(
        np.linspace(0.13, 0.83, workload.dimension, dtype=np.float64)
    )
    return Problem(samples, orders, tempering, STEP)


def _run_fast(
    problem: Problem,
    workload: Workload,
    definition_case: DefinitionCase,
    method: str,
    backend: str,
    *,
    quadrature_points: int | None,
) -> TemperedFastHistoryResult:
    return tempered_fast_multistep_history(
        problem.samples,
        problem.orders,
        tempering=problem.tempering,
        multistep_method=method,
        definition=definition_case.definition,
        step=problem.step,
        initial_condition_semantics=definition_case.initial_condition_semantics,
        local_history_steps=workload.local_history_steps,
        quadrature_points=quadrature_points,
        relative_tolerance=workload.relative_tolerance,
        tail_cutoff=TAIL_CUTOFF,
        max_quadrature_points=workload.max_quadrature_points,
        backend=backend,
    )


def _run_fbdf1_baseline(
    problem: Problem,
    definition_case: DefinitionCase,
    backend: str,
):
    return tempered_convolution_quadrature(
        problem.samples,
        problem.orders,
        tempering=problem.tempering,
        bdf_order=1,
        definition=definition_case.definition,
        step=problem.step,
        initial_condition_semantics=definition_case.initial_condition_semantics,
        backend=backend,
    )


def _gngf2_base_weights(orders: np.ndarray, count: int) -> np.ndarray:
    """Construct GNGF2 weights independently from the public fast module."""

    result = np.empty((count, orders.size), dtype=np.float64)
    for component, order in enumerate(orders):
        gl_previous = 1.0
        result[0, component] = 1.0 + 0.5 * float(order)
        for lag in range(1, count):
            gl_current = ((lag - 1.0 - float(order)) / lag) * gl_previous
            result[lag, component] = (
                (1.0 + 0.5 * float(order)) * gl_current
                - 0.5 * float(order) * gl_previous
            )
            gl_previous = gl_current
    return np.ascontiguousarray(result)


def _fft_length(count: int) -> int:
    required = 2 * count - 1
    return 1 << max(0, (required - 1).bit_length())


def _run_gngf2_reference(
    problem: Problem,
    definition_case: DefinitionCase,
    backend: str,
) -> ReferenceResult:
    """Evaluate exact GNGF2 weights by direct dot products or linear FFT."""

    if backend not in GNGF2_BASELINES:
        raise ValueError(f"unknown GNGF2 reference backend {backend!r}")
    count, dimension = problem.samples.shape
    lags = np.arange(count, dtype=np.float64)
    base = _gngf2_base_weights(problem.orders, count)
    damping = np.exp(
        -problem.step * lags[:, None] * problem.tempering[None, :]
    )
    tempered = np.ascontiguousarray(base * damping)
    unscaled = np.empty_like(problem.samples)
    fft_length: int | None = None
    if backend == "direct_reference":
        for component in range(dimension):
            for index in range(count):
                unscaled[index, component] = np.dot(
                    tempered[: index + 1, component],
                    problem.samples[index::-1, component],
                )
    else:
        fft_length = _fft_length(count)
        for component in range(dimension):
            sample_spectrum = np.fft.rfft(
                problem.samples[:, component], n=fft_length
            )
            weight_spectrum = np.fft.rfft(
                tempered[:, component], n=fft_length
            )
            unscaled[:, component] = np.fft.irfft(
                sample_spectrum * weight_spectrum, n=fft_length
            )[:count]
    if definition_case.definition == "tempered_caputo":
        unscaled -= (
            problem.samples[0:1, :]
            * damping
            * np.cumsum(base, axis=0)
        )
        unscaled[0, :] = 0.0
    values = unscaled * np.power(problem.step, -problem.orders)[None, :]
    if not np.all(np.isfinite(values)):
        raise RuntimeError("the independent GNGF2 reference became non-finite")
    return ReferenceResult(
        values=np.ascontiguousarray(values),
        base_weights=base,
        tempered_weights=tempered,
        backend=backend,
        fft_length=fft_length,
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


def _componentwise_max_abs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.max(np.abs(left - right), axis=0)


def _absolute_accumulation_scale(
    problem: Problem,
    definition_case: DefinitionCase,
    base_weights: np.ndarray,
    tempered_weights: np.ndarray,
) -> np.ndarray:
    """Bound magnitudes accumulated by the exact full-history formula.

    The scale deliberately ignores cancellation.  It is used only to attach a
    finite-precision margin to the independently reported compression bound.
    """

    count, dimension = problem.samples.shape
    lags = np.arange(count, dtype=np.float64)
    scale = np.empty(dimension, dtype=np.float64)
    for component in range(dimension):
        absolute_history = np.convolve(
            np.abs(problem.samples[:, component]),
            np.abs(tempered_weights[:, component]),
        )[:count]
        if definition_case.definition == "tempered_caputo":
            absolute_history += (
                abs(problem.samples[0, component])
                * np.exp(
                    -problem.tempering[component] * problem.step * lags
                )
                * np.abs(np.cumsum(base_weights[:, component]))
            )
        scale[component] = (
            np.max(absolute_history)
            * problem.step ** (-problem.orders[component])
        )
    return scale


def _assert_parity(
    fast_python: TemperedFastHistoryResult,
    fast_numba: TemperedFastHistoryResult,
    direct_values: np.ndarray,
    fft_values: np.ndarray,
    absolute_accumulation_scale: np.ndarray,
    *,
    workload: Workload,
    definition_case: DefinitionCase,
    method: str,
) -> dict[str, object]:
    if fast_python.quadrature_points != fast_numba.quadrature_points:
        raise RuntimeError("Fast Python and Numba used different Q")
    if not (
        fast_python.compression_tolerance_satisfied
        and fast_numba.compression_tolerance_satisfied
    ):
        raise RuntimeError("fast-history finite-grid calibration did not pass")
    np.testing.assert_allclose(
        fast_numba.values,
        fast_python.values,
        rtol=FAST_BACKEND_RTOL,
        atol=FAST_BACKEND_ATOL,
        err_msg=(
            f"Fast Python/Numba mismatch for "
            f"{workload.name}/{method}/{definition_case.name}"
        ),
    )
    np.testing.assert_allclose(
        fft_values,
        direct_values,
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
        err_msg=(
            f"direct/FFT mismatch for "
            f"{workload.name}/{method}/{definition_case.name}"
        ),
    )
    observed_by_component = _componentwise_max_abs(
        fast_numba.values, direct_values
    )
    operation_count_model = 8 * (
        workload.n_times
        + fast_numba.quadrature_points
        + fast_numba.local_history_steps
        + 1
    )
    epsilon = np.finfo(np.float64).eps
    gamma = (operation_count_model * epsilon) / (
        1.0 - operation_count_model * epsilon
    )
    roundoff_margin = gamma * np.maximum(1.0, absolute_accumulation_scale)
    allowed_by_component = (
        fast_numba.operator_absolute_error_bound + roundoff_margin
    )
    if np.any(observed_by_component > allowed_by_component):
        raise AssertionError(
            "fast-history error exceeded its finite-grid operator bound for "
            f"{workload.name}/{method}/{definition_case.name}: "
            f"observed={observed_by_component.tolist()}, "
            f"allowed={allowed_by_component.tolist()}"
        )
    return {
        "fast_python_numba_max_abs_by_component": [
            float(value)
            for value in _componentwise_max_abs(
                fast_python.values, fast_numba.values
            )
        ],
        "direct_fft_max_abs_by_component": [
            float(value)
            for value in _componentwise_max_abs(direct_values, fft_values)
        ],
        "fast_numba_direct_max_abs_by_component": [
            float(value) for value in observed_by_component
        ],
        "fast_operator_bound_by_component": [
            float(value) for value in fast_numba.operator_absolute_error_bound
        ],
        "roundoff_margin_by_component": [
            float(value) for value in roundoff_margin
        ],
        "absolute_accumulation_scale_by_component": [
            float(value) for value in absolute_accumulation_scale
        ],
        "roundoff_operation_count_model": operation_count_model,
        "roundoff_model": (
            "gamma_k*max(1, absolute full-history accumulation scale), "
            "k=8*(N+Q+n0+1); compression and roundoff remain separate"
        ),
        "compression_l1_relative_error_by_component": [
            float(value) for value in fast_numba.l1_relative_weight_error
        ],
        "compression_tolerance_satisfied": True,
    }


def _measure_rotating(
    operations: dict[str, Callable[[], Any]],
    *,
    repeats: int,
    validator: Callable[[Any, str], None],
) -> tuple[dict[str, dict[str, object]], list[list[str]]]:
    backends = tuple(operations)
    samples: dict[str, list[float]] = {backend: [] for backend in backends}
    execution_orders: list[list[str]] = []
    for repeat in range(repeats):
        shift = repeat % len(backends)
        order = backends[shift:] + backends[:shift]
        execution_orders.append(list(order))
        for backend in order:
            gc.collect()
            elapsed, result = _timed(operations[backend])
            validator(result, backend)
            samples[backend].append(elapsed)
    return (
        {backend: _summary(samples[backend]) for backend in backends},
        execution_orders,
    )


def _fast_active_memory(
    result: TemperedFastHistoryResult,
    *,
    n_times: int,
    dimension: int,
) -> dict[str, object]:
    q_points = result.quadrature_points
    local_count = result.local_history_steps + 1
    recurrence_state_values = q_points * dimension
    recurrence_coefficients_values = 2 * q_points * dimension
    exact_local_values = local_count * dimension
    evaluator_values = (
        recurrence_state_values
        + recurrence_coefficients_values
        + exact_local_values
    )
    retained_arrays = (
        result.local_base_weights,
        result.local_tempered_weights,
        result.quadrature_nodes,
        result.quadrature_weights,
        result.final_history_state,
        result.orders,
        result.tempering,
        result.l1_absolute_weight_error,
        result.l1_relative_weight_error,
        result.max_absolute_weight_error,
        result.max_relative_weight_error,
        result.operator_absolute_error_bound,
    )
    return {
        "N": n_times,
        "Q": q_points,
        "d": dimension,
        "n0": result.local_history_steps,
        "float64_bytes": FLOAT64_BYTES,
        "recurrence_state_bytes": recurrence_state_values * FLOAT64_BYTES,
        "recurrence_coefficients_bytes": (
            recurrence_coefficients_values * FLOAT64_BYTES
        ),
        "exact_local_tempered_weights_bytes": (
            exact_local_values * FLOAT64_BYTES
        ),
        "evaluator_active_history_bytes_excluding_input_and_output": (
            evaluator_values * FLOAT64_BYTES
        ),
        "retained_audit_arrays_bytes_excluding_times_and_values": int(
            sum(array.nbytes for array in retained_arrays)
        ),
        "analytic_evaluator_formula": "8*d*(3*Q+n0+1) bytes",
        "complexity": "O(d*(Q+n0)*N) time; O(d*(Q+n0)) active history",
    }


def _batch_memory_model(n_times: int, dimension: int) -> dict[str, object]:
    fft_length = _fft_length(n_times)
    rfft_bins = fft_length // 2 + 1
    weight_arrays = 2 * n_times * dimension * FLOAT64_BYTES
    fft_component_work = (
        3 * rfft_bins * COMPLEX128_BYTES
        + fft_length * FLOAT64_BYTES
    )
    return {
        "N": n_times,
        "d": dimension,
        "full_base_and_tempered_weight_arrays_bytes": weight_arrays,
        "fft_length": fft_length,
        "fft_complex_bins_per_component": rfft_bins,
        "fft_component_work_lower_bound_bytes_excluding_output": (
            fft_component_work
        ),
        "fft_memory_note": (
            "analytical lower-bound model for two input spectra, their product, "
            "and one inverse real work array; allocator/library temporaries may "
            "increase peak resident memory"
        ),
    }


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
    definition_case: DefinitionCase,
    method: str,
    *,
    repeats: int,
) -> dict[str, object]:
    calibration_seconds, automatic = _timed(
        lambda: _run_fast(
            problem,
            workload,
            definition_case,
            method,
            "python",
            quadrature_points=None,
        )
    )
    q_points = automatic.quadrature_points
    fast_python = _run_fast(
        problem,
        workload,
        definition_case,
        method,
        "python",
        quadrature_points=q_points,
    )
    fast_numba = _run_fast(
        problem,
        workload,
        definition_case,
        method,
        "numba",
        quadrature_points=q_points,
    )

    if method == "fbdf1":
        direct = _run_fbdf1_baseline(problem, definition_case, "numba")
        fft = _run_fbdf1_baseline(problem, definition_case, "fft")
        direct_values = np.asarray(direct.values)
        fft_values = np.asarray(fft.values)
        base_weights = np.asarray(direct.base_weights)
        tempered_weights = np.asarray(direct.weights)
        operations: dict[str, Callable[[], Any]] = {
            "fast_python": lambda: _run_fast(
                problem,
                workload,
                definition_case,
                method,
                "python",
                quadrature_points=q_points,
            ),
            "fast_numba": lambda: _run_fast(
                problem,
                workload,
                definition_case,
                method,
                "numba",
                quadrature_points=q_points,
            ),
            "direct_numba": lambda: _run_fbdf1_baseline(
                problem, definition_case, "numba"
            ),
            "fft_batch": lambda: _run_fbdf1_baseline(
                problem, definition_case, "fft"
            ),
        }

        def validator(result: Any, backend: str) -> None:
            expected = {
                "fast_python": "python",
                "fast_numba": "numba",
                "direct_numba": "numba",
                "fft_batch": "fft",
            }[backend]
            if result.backend != expected or np.asarray(result.values).shape != problem.samples.shape:
                raise RuntimeError(f"invalid timed FBDF1 result for {backend}")

        baseline_contract = (
            "public tempered CQ FBDF1 direct Numba and offline FFT batch"
        )
    else:
        direct = _run_gngf2_reference(
            problem, definition_case, "direct_reference"
        )
        fft = _run_gngf2_reference(problem, definition_case, "fft_reference")
        direct_values = direct.values
        fft_values = fft.values
        base_weights = direct.base_weights
        tempered_weights = direct.tempered_weights
        operations = {
            "fast_python": lambda: _run_fast(
                problem,
                workload,
                definition_case,
                method,
                "python",
                quadrature_points=q_points,
            ),
            "fast_numba": lambda: _run_fast(
                problem,
                workload,
                definition_case,
                method,
                "numba",
                quadrature_points=q_points,
            ),
            "direct_reference": lambda: _run_gngf2_reference(
                problem, definition_case, "direct_reference"
            ),
            "fft_reference": lambda: _run_gngf2_reference(
                problem, definition_case, "fft_reference"
            ),
        }

        def validator(result: Any, backend: str) -> None:
            expected = {
                "fast_python": "python",
                "fast_numba": "numba",
                "direct_reference": "direct_reference",
                "fft_reference": "fft_reference",
            }[backend]
            if result.backend != expected or np.asarray(result.values).shape != problem.samples.shape:
                raise RuntimeError(f"invalid timed GNGF2 result for {backend}")

        baseline_contract = (
            "independent exact GNGF2 recurrence weights with direct NumPy dot "
            "products or zero-padded NumPy FFT"
        )

    parity = _assert_parity(
        fast_python,
        fast_numba,
        direct_values,
        fft_values,
        _absolute_accumulation_scale(
            problem,
            definition_case,
            base_weights,
            tempered_weights,
        ),
        workload=workload,
        definition_case=definition_case,
        method=method,
    )
    timing, execution_orders = _measure_rotating(
        operations, repeats=repeats, validator=validator
    )
    medians = {
        backend: float(summary["median_seconds"])
        for backend, summary in timing.items()
    }
    baseline_direct_name = (
        "direct_numba" if method == "fbdf1" else "direct_reference"
    )
    baseline_fft_name = "fft_batch" if method == "fbdf1" else "fft_reference"
    return {
        "definition_case": asdict(definition_case),
        "multistep_method": method,
        "generator_contract": (
            "Omega(z)=(1-z)^q"
            if method == "fbdf1"
            else "Omega(z)=(1-z)^q*(1+q*(1-z)/2); not fractional BDF2"
        ),
        "baseline_contract": baseline_contract,
        "Q_selection": {
            "automatic_selection_seconds_excluded_from_repeated_timings": (
                calibration_seconds
            ),
            "selected_quadrature_points": q_points,
            "selection_backend": "python",
            "selected_Q_fixed_during_all_repeated_fast_calls": True,
        },
        "parity_checked_before_measurement": True,
        "parity": parity,
        "timing": timing,
        "backend_execution_order_by_repetition": execution_orders,
        "finite_run_ratios": {
            "direct_over_fast_numba_median": (
                medians[baseline_direct_name] / medians["fast_numba"]
            ),
            "fft_over_fast_numba_median": (
                medians[baseline_fft_name] / medians["fast_numba"]
            ),
            "fast_python_over_fast_numba_median": (
                medians["fast_python"] / medians["fast_numba"]
            ),
        },
        "observed_fastest_backend_by_median": min(medians, key=medians.get),
        "fast_active_memory": _fast_active_memory(
            fast_numba,
            n_times=workload.n_times,
            dimension=workload.dimension,
        ),
        "direct_and_fft_batch_memory": _batch_memory_model(
            workload.n_times, workload.dimension
        ),
        "scope": (
            "sampled tempered operator only; fast routes are recurrent batch "
            "evaluations and FFT routes are offline batch convolutions"
        ),
    }


def _benchmark_workload(workload: Workload, *, repeats: int) -> dict[str, object]:
    problem = _make_problem(workload)
    return {
        "workload": asdict(workload),
        "fixture": {
            "description": (
                "fixed smooth multicomponent history plus a seeded low-amplitude "
                "integrated perturbation; nonzero componentwise initial values"
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
                definition_case,
                method,
                repeats=repeats,
            )
            for method in MULTISTEP_METHODS
            for definition_case in DEFINITION_CASES
        ],
    }


def _backend_policy_assessment(
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    observed_winners: dict[str, int] = {}
    largest_ratios: dict[str, dict[str, float]] = {}
    for record in records:
        workload_name = str(record["workload"]["name"])
        for case in record["cases"]:
            winner = str(case["observed_fastest_backend_by_median"])
            observed_winners[winner] = observed_winners.get(winner, 0) + 1
            if record is records[-1]:
                key = (
                    f"{case['multistep_method']}_"
                    f"{case['definition_case']['name']}"
                )
                largest_ratios[key] = {
                    name: float(value)
                    for name, value in case["finite_run_ratios"].items()
                }
    return {
        "decision": "insufficient_evidence_to_add_native_c_or_julia",
        "native_c_candidate_implemented_or_measured": False,
        "julia_candidate_implemented_or_measured": False,
        "measured_routes": [
            "fast_python",
            "fast_numba",
            "fbdf1_direct_numba",
            "fbdf1_fft_batch",
            "gngf2_direct_numpy_reference",
            "gngf2_fft_numpy_reference",
        ],
        "observed_fastest_backend_counts": observed_winners,
        "largest_workload": str(records[-1]["workload"]["name"]),
        "largest_workload_finite_ratios": largest_ratios,
        "c_decision_gate": (
            "Profile representative HAFO/Toolbox long-history workloads first. "
            "Only benchmark a native-C recurrence if warmed Fast Numba remains "
            "a material end-to-end bottleneck; require identical N/Q/d/tolerance, "
            "finite-grid parity and operator-bound checks, compiler provenance, "
            "repeated timings, active-memory accounting, and a robust end-to-end "
            "gain beyond run-to-run uncertainty before adoption."
        ),
        "julia_decision_gate": (
            "No Julia implementation with the identical FBDF1/GNGF2, local-window, "
            "finite-grid calibration, and Caputo-anchor contract was measured. "
            "Julia may serve as a pinned whole-batch comparison only after startup, "
            "data transfer, parity, memory, and end-to-end time are measured; never "
            "call Julia inside the history recurrence."
        ),
        "fft_interpretation": (
            "FFT is an offline zero-padded batch baseline with O(N) stored weights; "
            "it is not streaming or active-memory-compressed fast history. A lower "
            "batch median does not invalidate the recurrent memory contract."
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


def _numba_warmup() -> dict[str, object]:
    workload = _WARMUP_WORKLOAD
    problem = _make_problem(workload)
    case = DEFINITION_CASES[0]
    _, calibrated = _timed(
        lambda: _run_fast(
            problem,
            workload,
            case,
            "fbdf1",
            "python",
            quadrature_points=None,
        )
    )
    q_points = calibrated.quadrature_points
    fast_first, fast_result = _timed(
        lambda: _run_fast(
            problem,
            workload,
            case,
            "fbdf1",
            "numba",
            quadrature_points=q_points,
        )
    )
    fast_warmed, fast_confirmation = _timed(
        lambda: _run_fast(
            problem,
            workload,
            case,
            "fbdf1",
            "numba",
            quadrature_points=q_points,
        )
    )
    np.testing.assert_array_equal(fast_result.values, fast_confirmation.values)
    direct_first, direct_result = _timed(
        lambda: _run_fbdf1_baseline(problem, case, "numba")
    )
    direct_warmed, direct_confirmation = _timed(
        lambda: _run_fbdf1_baseline(problem, case, "numba")
    )
    np.testing.assert_array_equal(direct_result.values, direct_confirmation.values)
    return {
        "workload": asdict(workload),
        "Q": q_points,
        "fast_numba_first_public_call_seconds_including_jit": fast_first,
        "fast_numba_warmed_confirmation_seconds": fast_warmed,
        "direct_numba_first_public_call_seconds_including_jit": direct_first,
        "direct_numba_warmed_confirmation_seconds": direct_warmed,
        "all_warmup_calls_excluded_from_repeated_measurements": True,
    }


def run_benchmark(
    *,
    repeats: int = 5,
    workloads: Sequence[Workload] = WORKLOADS,
) -> dict[str, object]:
    """Run parity-gated fast-history measurements and return JSON-safe data."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 3:
        raise ValueError("repeats must be an integer of at least 3")
    selected_workloads = tuple(workloads)
    if not selected_workloads:
        raise ValueError("workloads must not be empty")
    for workload in selected_workloads:
        _validate_workload(workload)

    warmup = _numba_warmup()
    records = [
        _benchmark_workload(workload, repeats=repeats)
        for workload in selected_workloads
    ]
    return {
        "schema_version": 1,
        "benchmark_id": "hafo_tempered_fast_history_backends_20260803",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "tempered_fast_multistep_sampled_operator_performance",
        "evidence_boundary": (
            "finite host- and workload-specific sampled-operator engineering "
            "evidence only; no FDE solver, universal crossover, convergence-order, "
            "stability, dynamics, chaos, attraction, or hiddenness claim"
        ),
        "portable_json_contract": (
            "payload contains no repository, output, cache, or temporary absolute "
            "filesystem path"
        ),
        "measurement_protocol": {
            "backend_parity_before_measurement": True,
            "methods": list(MULTISTEP_METHODS),
            "definitions": [case.definition for case in DEFINITION_CASES],
            "fast_backends": list(FAST_BACKENDS),
            "fbdf1_baselines": list(FBDF1_BASELINES),
            "gngf2_baselines": list(GNGF2_BASELINES),
            "gngf2_is_fractional_bdf2": False,
            "automatic_Q_selection_timed_separately": True,
            "fixed_validated_Q_in_repeated_fast_calls": True,
            "numba_warmup": warmup,
            "measured_repetitions_per_backend_case": repeats,
            "backend_order": "cyclically rotated across repetitions",
            "garbage_collection": "collected before and outside each timed call",
            "clock": "time.perf_counter_ns",
            "repeated_fast_call_scope": (
                "complete public call for fixed Q: validation, local weights, "
                "finite-grid calibration, recurrence construction and evaluation"
            ),
            "fft_scope": "offline zero-padded batch convolution; not fast-history",
            "parity_rtol": PARITY_RTOL,
            "parity_atol": PARITY_ATOL,
        },
        "host": _host_record(),
        "workloads": records,
        "backend_policy_assessment": _backend_policy_assessment(records),
    }


def _default_output_path() -> Path:
    root = Path(r"C:\tmp") if os.name == "nt" else Path(tempfile.gettempdir())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return root / f"hafo_tempered_fast_history_benchmark_{stamp}_{os.getpid()}.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit JSON path; default is a unique file under the OS temp root.",
    )
    args = parser.parse_args(argv)
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")

    payload = run_benchmark(repeats=args.repeats)
    output = args.output if args.output is not None else _default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(rendered + "\n")
    print(output)


if __name__ == "__main__":
    main()
