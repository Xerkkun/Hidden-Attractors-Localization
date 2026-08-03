"""Reproducible benchmark for the multi-term Caputo semantic facade.

The deterministic workload is synthetic.  This benchmark measures two
deliberately separate questions:

1. the end-to-end overhead of ``integrate_multi_term_caputo_l1`` relative to
   calling the distributed-order solver with already canonical atomic terms;
2. the one-time benefit of coalescing repeated orders before constructing the
   combined L1 kernel.

The direct ``O(N**2*d)`` history sweep is timed independently so that it is not
misreported as part of the ``O(R*N)`` kernel-construction comparison.  Numba
compilation and explicit warm-up calls are excluded from measured samples.
Every ratio is conditional on finite numerical parity checks passing first.

Results are host-, runtime-, and workload-specific software-engineering
evidence.  They do not establish a universal speedup, convergence theorem,
chaos, attraction, or hiddenness.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numba
import numpy as np
import scipy
from scipy.special import gamma

import hidden_attractors.fractional.distributed_order_caputo_solver as core_module
import hidden_attractors.fractional.multi_term_caputo as facade_module
from hidden_attractors.fractional.distributed_order_caputo_solver import (
    DistributedOrderCaputoResult,
    integrate_distributed_order_caputo_l1,
)
from hidden_attractors.fractional.multi_term_caputo import (
    MultiTermCaputoResult,
    MultiTermCaputoTerms,
    canonicalize_multi_term_caputo_terms,
    integrate_multi_term_caputo_l1,
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Finite deterministic workload and timing protocol."""

    repeats: int = 7
    warmup_calls: int = 1
    n_steps: int = 256
    kernel_lags: int = 4096
    dimension: int = 3
    unique_orders: int = 8
    duplicate_factor: int = 16
    canonicalization_batch: int = 64
    step: float = 0.01
    seed: int = 2026080341


def _validate_config(config: BenchmarkConfig) -> None:
    integer_bounds = {
        "repeats": (config.repeats, 3),
        "warmup_calls": (config.warmup_calls, 1),
        "n_steps": (config.n_steps, 2),
        "kernel_lags": (config.kernel_lags, 2),
        "dimension": (config.dimension, 1),
        "unique_orders": (config.unique_orders, 1),
        "duplicate_factor": (config.duplicate_factor, 2),
        "canonicalization_batch": (config.canonicalization_batch, 1),
    }
    for name, (value, minimum) in integer_bounds.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if config.kernel_lags < config.n_steps:
        raise ValueError("kernel_lags must be >= n_steps for the history check.")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int):
        raise ValueError("seed must be an integer.")
    if not np.isfinite(config.step) or config.step <= 0.0:
        raise ValueError("step must be finite and positive.")


def _affine_rhs(
    current_time: float,
    state: np.ndarray,
    parameters: tuple[float, float],
) -> np.ndarray:
    """Stable deterministic vector field used by both public solver calls."""

    damping, forcing = parameters
    return damping * state + forcing * np.cos(0.37 * current_time)


def _make_terms(config: BenchmarkConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return permuted positive duplicate terms with deterministic provenance."""

    if config.unique_orders == 1:
        unique_orders = np.array([0.63], dtype=np.float64)
    else:
        unique_orders = np.linspace(
            0.18,
            1.0,
            config.unique_orders,
            dtype=np.float64,
        )
    # These are equation coefficients, deliberately not probability-normalized.
    unique_coefficients = 0.35 + 0.09 * np.arange(
        config.unique_orders,
        dtype=np.float64,
    )
    duplicate_orders = np.repeat(unique_orders, config.duplicate_factor)
    duplicate_coefficients = np.repeat(
        unique_coefficients / float(config.duplicate_factor),
        config.duplicate_factor,
    )
    permutation = np.random.default_rng(config.seed).permutation(
        duplicate_orders.size
    )
    return (
        np.ascontiguousarray(duplicate_orders[permutation]),
        np.ascontiguousarray(duplicate_coefficients[permutation]),
    )


def _initial_state(config: BenchmarkConfig) -> np.ndarray:
    values = np.linspace(0.08, 0.24, config.dimension, dtype=np.float64)
    values[1::2] *= -1.0
    return np.ascontiguousarray(values)


def _l1_coefficients(
    orders: np.ndarray,
    coefficients: np.ndarray,
    step: float,
) -> np.ndarray:
    values = coefficients * np.power(step, -orders) / gamma(2.0 - orders)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("The benchmark produced invalid positive L1 coefficients.")
    return np.ascontiguousarray(values, dtype=np.float64)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _source_sha256(module: Any) -> tuple[str, str]:
    path = Path(module.__file__).resolve()
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _timed_call(operation: Callable[[], Any]) -> tuple[float, Any]:
    """Time one synchronous call with cyclic GC disabled only in the interval."""

    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        started_ns = time.perf_counter_ns()
        result = operation()
        elapsed_ns = time.perf_counter_ns() - started_ns
    finally:
        if gc_was_enabled:
            gc.enable()
    return elapsed_ns * 1.0e-9, result


def _summary(samples: list[float]) -> dict[str, object]:
    if not samples:
        raise ValueError("At least one timing sample is required.")
    acquisition_order = [float(value) for value in samples]
    ordered = sorted(acquisition_order)
    return {
        "samples_seconds": acquisition_order,
        "sorted_samples_seconds": ordered,
        "count": len(ordered),
        "median_seconds": float(statistics.median(ordered)),
        "mean_seconds": float(statistics.fmean(ordered)),
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "population_stdev_seconds": float(statistics.pstdev(ordered)),
    }


def _measure_group(
    operations: Mapping[str, Callable[[], Any]],
    *,
    repeats: int,
    warmup_calls: int,
    validator: Callable[[str, Any], None],
) -> tuple[dict[str, dict[str, object]], dict[str, Any]]:
    """Warm and measure operations with deterministic cyclic order rotation."""

    names = list(operations)
    warmup: dict[str, list[float]] = {name: [] for name in names}
    measured: dict[str, list[float]] = {name: [] for name in names}
    last_results: dict[str, Any] = {}

    for warmup_index in range(warmup_calls):
        rotation = warmup_index % len(names)
        for name in names[rotation:] + names[:rotation]:
            gc.collect()
            elapsed, result = _timed_call(operations[name])
            validator(name, result)
            warmup[name].append(elapsed)
            last_results[name] = result

    for repeat_index in range(repeats):
        rotation = repeat_index % len(names)
        for name in names[rotation:] + names[:rotation]:
            gc.collect()
            elapsed, result = _timed_call(operations[name])
            validator(name, result)
            measured[name].append(elapsed)
            last_results[name] = result

    records = {
        name: {
            "warmup_excluded": _summary(warmup[name]),
            "measurement": _summary(measured[name]),
        }
        for name in names
    }
    return records, last_results


def _max_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    difference = np.abs(candidate_array - reference_array)
    scale = np.maximum(np.abs(reference_array), np.finfo(np.float64).tiny)
    return {
        "max_absolute_error": float(np.max(difference, initial=0.0)),
        "max_elementwise_relative_error": float(
            np.max(difference / scale, initial=0.0)
        ),
    }


def _history_sweep(states: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Run the current direct history kernel for every output index."""

    accumulator = np.zeros(states.shape[1], dtype=np.float64)
    for output_index in range(1, states.shape[0]):
        accumulator += core_module._history_sum_numba(
            states,
            output_index,
            kernel,
        )
    return accumulator


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
        "numba_cache_dir_environment": os.environ.get("NUMBA_CACHE_DIR"),
        "python_pycache_prefix_environment": os.environ.get(
            "PYTHONPYCACHEPREFIX"
        ),
        "omp_num_threads_environment": os.environ.get("OMP_NUM_THREADS"),
        "perf_counter": {
            "implementation": clock.implementation,
            "resolution_seconds": clock.resolution,
            "monotonic": clock.monotonic,
        },
    }


def run_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    """Run the finite benchmark and return a JSON-serializable evidence record."""

    _validate_config(config)
    duplicate_orders, duplicate_coefficients = _make_terms(config)
    terms = canonicalize_multi_term_caputo_terms(
        duplicate_orders,
        duplicate_coefficients,
    )
    if terms.original_term_count != config.unique_orders * config.duplicate_factor:
        raise RuntimeError("Unexpected original term count in benchmark fixture.")
    if terms.term_count != config.unique_orders:
        raise RuntimeError(
            "Exact duplicate coalescence did not produce the fixture rule."
        )

    initial_state = _initial_state(config)
    parameters = (-0.18, 0.035)
    common_solver_arguments = {
        "rhs": _affine_rhs,
        "initial_state": initial_state,
        "parameters": parameters,
        "step": config.step,
        "n_steps": config.n_steps,
        "lower_terminal": 0.0,
        "corrector_atol": 1.0e-12,
        "corrector_rtol": 1.0e-10,
        "corrector_max_iterations": 80,
        "on_nonconvergence": "raise",
        "initial_regularity": "nonsmooth",
        "compatibility_tolerance": 1.0e-10,
        "use_acceleration": True,
        "allow_python_fallback": False,
        "divergence_norm": None,
    }

    def facade_operation() -> MultiTermCaputoResult:
        return integrate_multi_term_caputo_l1(
            **common_solver_arguments,
            orders=duplicate_orders,
            coefficients=duplicate_coefficients,
        )

    def direct_canonical_operation() -> DistributedOrderCaputoResult:
        return integrate_distributed_order_caputo_l1(
            **common_solver_arguments,
            order_nodes=terms.orders,
            order_weights=terms.coefficients,
            weight_semantics="nonnegative_mass",
            density_values=None,
            normalization="none",
            order_quadrature_name="benchmark_direct_canonical_atomic_terms",
        )

    setup_started_ns = time.perf_counter_ns()
    facade_reference = facade_operation()
    direct_reference = direct_canonical_operation()
    setup_and_jit_seconds = (time.perf_counter_ns() - setup_started_ns) * 1.0e-9
    if facade_reference.status != "ok" or direct_reference.status != "ok":
        raise RuntimeError(
            "Public solver parity setup did not finish with status='ok'."
        )
    if not bool(facade_reference.solver_info.get("numba_kernel_used")):
        raise RuntimeError("Facade benchmark did not use the requested Numba kernel.")
    if not bool(direct_reference.solver_info.get("numba_kernel_used")):
        raise RuntimeError("Direct benchmark did not use the requested Numba kernel.")
    np.testing.assert_array_equal(facade_reference.times, direct_reference.times)
    np.testing.assert_allclose(
        facade_reference.states,
        direct_reference.states,
        rtol=3.0e-13,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        facade_reference.combined_l1_kernel,
        direct_reference.combined_l1_kernel,
        rtol=3.0e-15,
        atol=3.0e-15,
    )

    def validate_public(name: str, result: Any) -> None:
        if result.status != "ok":
            raise RuntimeError(f"{name} returned unexpected status {result.status!r}.")
        np.testing.assert_array_equal(result.times, direct_reference.times)
        np.testing.assert_allclose(
            result.states,
            direct_reference.states,
            rtol=3.0e-13,
            atol=3.0e-14,
        )

    public_timings, public_results = _measure_group(
        {
            "facade_from_duplicate_terms": facade_operation,
            "direct_from_canonical_terms": direct_canonical_operation,
        },
        repeats=config.repeats,
        warmup_calls=config.warmup_calls,
        validator=validate_public,
    )

    def canonicalization_batch_operation() -> MultiTermCaputoTerms:
        result: MultiTermCaputoTerms | None = None
        for _ in range(config.canonicalization_batch):
            result = canonicalize_multi_term_caputo_terms(
                duplicate_orders,
                duplicate_coefficients,
            )
        if result is None:  # Guard retained for static analysis and future edits.
            raise RuntimeError("canonicalization_batch must be positive.")
        return result

    def validate_canonicalization(name: str, result: Any) -> None:
        del name
        np.testing.assert_array_equal(result.orders, terms.orders)
        np.testing.assert_array_equal(result.coefficients, terms.coefficients)

    canonicalization_timings, _ = _measure_group(
        {"canonicalization_batch": canonicalization_batch_operation},
        repeats=config.repeats,
        warmup_calls=config.warmup_calls,
        validator=validate_canonicalization,
    )
    canonicalization_record = canonicalization_timings["canonicalization_batch"]
    batch_measurement = canonicalization_record["measurement"]
    batch_median = float(batch_measurement["median_seconds"])
    canonicalization_per_call = batch_median / config.canonicalization_batch

    duplicate_l1 = _l1_coefficients(
        duplicate_orders,
        duplicate_coefficients,
        config.step,
    )
    canonical_l1 = _l1_coefficients(
        terms.orders,
        terms.coefficients,
        config.step,
    )

    def duplicate_kernel_operation() -> np.ndarray:
        return core_module._combined_kernel_numba(
            duplicate_orders,
            duplicate_l1,
            config.kernel_lags,
        )

    def canonical_kernel_operation() -> np.ndarray:
        return core_module._combined_kernel_numba(
            terms.orders,
            canonical_l1,
            config.kernel_lags,
        )

    duplicate_kernel_reference = duplicate_kernel_operation()
    canonical_kernel_reference = canonical_kernel_operation()
    kernel_atol = 128.0 * np.finfo(np.float64).eps * max(
        1.0,
        float(np.max(np.abs(canonical_kernel_reference))),
    )
    np.testing.assert_allclose(
        duplicate_kernel_reference,
        canonical_kernel_reference,
        rtol=2.0e-13,
        atol=kernel_atol,
    )

    def validate_kernel(name: str, result: Any) -> None:
        reference = (
            duplicate_kernel_reference
            if name == "uncoalesced_duplicate_terms"
            else canonical_kernel_reference
        )
        np.testing.assert_array_equal(result, reference)

    kernel_timings, kernel_results = _measure_group(
        {
            "uncoalesced_duplicate_terms": duplicate_kernel_operation,
            "coalesced_canonical_terms": canonical_kernel_operation,
        },
        repeats=config.repeats,
        warmup_calls=config.warmup_calls,
        validator=validate_kernel,
    )

    rng = np.random.default_rng(config.seed + 1)
    increments = rng.normal(
        scale=0.015,
        size=(config.n_steps, config.dimension),
    )
    history_states = np.empty(
        (config.n_steps + 1, config.dimension),
        dtype=np.float64,
    )
    history_states[0] = initial_state
    history_states[1:] = initial_state + np.cumsum(increments, axis=0)
    history_states = np.ascontiguousarray(history_states)
    duplicate_history_kernel = np.ascontiguousarray(
        duplicate_kernel_reference[: config.n_steps]
    )
    canonical_history_kernel = np.ascontiguousarray(
        canonical_kernel_reference[: config.n_steps]
    )

    def duplicate_history_operation() -> np.ndarray:
        return _history_sweep(history_states, duplicate_history_kernel)

    def canonical_history_operation() -> np.ndarray:
        return _history_sweep(history_states, canonical_history_kernel)

    duplicate_history_reference = duplicate_history_operation()
    canonical_history_reference = canonical_history_operation()
    history_atol = 256.0 * np.finfo(np.float64).eps * max(
        1.0,
        float(np.max(np.abs(canonical_history_reference))),
    )
    np.testing.assert_allclose(
        duplicate_history_reference,
        canonical_history_reference,
        rtol=3.0e-13,
        atol=history_atol,
    )

    def validate_history(name: str, result: Any) -> None:
        reference = (
            duplicate_history_reference
            if name == "from_uncoalesced_kernel"
            else canonical_history_reference
        )
        np.testing.assert_array_equal(result, reference)

    history_timings, history_results = _measure_group(
        {
            "from_uncoalesced_kernel": duplicate_history_operation,
            "from_coalesced_kernel": canonical_history_operation,
        },
        repeats=config.repeats,
        warmup_calls=config.warmup_calls,
        validator=validate_history,
    )

    facade_median = float(
        public_timings["facade_from_duplicate_terms"]["measurement"][
            "median_seconds"
        ]
    )
    direct_median = float(
        public_timings["direct_from_canonical_terms"]["measurement"][
            "median_seconds"
        ]
    )
    uncoalesced_kernel_median = float(
        kernel_timings["uncoalesced_duplicate_terms"]["measurement"][
            "median_seconds"
        ]
    )
    coalesced_kernel_median = float(
        kernel_timings["coalesced_canonical_terms"]["measurement"][
            "median_seconds"
        ]
    )
    uncoalesced_history_median = float(
        history_timings["from_uncoalesced_kernel"]["measurement"][
            "median_seconds"
        ]
    )
    coalesced_history_median = float(
        history_timings["from_coalesced_kernel"]["measurement"][
            "median_seconds"
        ]
    )

    facade_source_path, facade_source_hash = _source_sha256(facade_module)
    core_source_path, core_source_hash = _source_sha256(core_module)
    script_path = Path(__file__).resolve()
    public_state_errors = _max_error(
        direct_reference.states,
        facade_reference.states,
    )
    kernel_errors = _max_error(
        canonical_kernel_reference,
        duplicate_kernel_reference,
    )
    history_errors = _max_error(
        canonical_history_reference,
        duplicate_history_reference,
    )

    return {
        "schema_version": 1,
        "benchmark": "hafo_multi_term_caputo_facade_and_coalescence",
        "scope": "finite_software_engineering_timing_for_one_atomic_caputo_workload",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_boundary": (
            "Host-, runtime-, configuration-, and workload-specific timing only; "
            "no universal facade-overhead or coalescence-speedup claim, no "
            "convergence theorem, and no claim of chaos, attraction, or hiddenness."
        ),
        "host": _host_record(),
        "configuration": asdict(config),
        "workload": {
            "input_term_count_R": terms.original_term_count,
            "canonical_term_count_R": terms.term_count,
            "duplicate_terms_coalesced": terms.duplicate_terms_coalesced,
            "duplicate_factor": config.duplicate_factor,
            "coefficient_normalization": "none",
            "order_min": float(np.min(terms.orders)),
            "order_max": float(np.max(terms.orders)),
            "coefficient_sum": terms.coefficient_sum,
            "n_steps_N": config.n_steps,
            "kernel_lags_N": config.kernel_lags,
            "state_dimension_d": config.dimension,
            "input_orders_sha256": _array_sha256(duplicate_orders),
            "input_coefficients_sha256": _array_sha256(
                duplicate_coefficients
            ),
            "canonical_orders_sha256": _array_sha256(terms.orders),
            "canonical_coefficients_sha256": _array_sha256(
                terms.coefficients
            ),
            "history_states_sha256": _array_sha256(history_states),
        },
        "measurement_protocol": {
            "clock": "time.perf_counter_ns",
            "warmup_excluded": True,
            "untimed_setup_and_jit_seconds": setup_and_jit_seconds,
            "additional_warmup_calls_per_operation": config.warmup_calls,
            "measured_repetitions_per_operation": config.repeats,
            "operation_order": "deterministic cyclic rotation per repetition",
            "garbage_collection": (
                "gc.collect before each call outside timing; cyclic GC disabled "
                "only during each timed interval"
            ),
            "public_solver_scope": (
                "complete facade call versus complete distributed-order solver "
                "call from already canonical terms; both include identical "
                "combined-kernel, direct-history, RHS, and Picard work"
            ),
            "kernel_scope": (
                "internal existing Numba combined-L1 construction only, O(R*N); "
                "no trajectory history, RHS, or corrector"
            ),
            "history_scope": (
                "existing Numba history sum called for output indices 1..N on "
                "fixed states and a precomputed kernel, O(N^2*d); no kernel "
                "construction, RHS, or corrector"
            ),
            "canonicalization_scope": (
                f"{config.canonicalization_batch} facade canonicalizations per "
                "timed batch; reported per-call median divides the batch median"
            ),
        },
        "numerical_parity": {
            "required_before_ratios": True,
            "passed": True,
            "public_times_exact": True,
            "public_states": {
                **public_state_errors,
                "rtol": 3.0e-13,
                "atol": 3.0e-14,
                "facade_states_sha256": _array_sha256(
                    facade_reference.states
                ),
                "direct_states_sha256": _array_sha256(
                    direct_reference.states
                ),
            },
            "duplicate_vs_canonical_combined_kernel": {
                **kernel_errors,
                "rtol": 2.0e-13,
                "atol": kernel_atol,
                "duplicate_kernel_sha256": _array_sha256(
                    duplicate_kernel_reference
                ),
                "canonical_kernel_sha256": _array_sha256(
                    canonical_kernel_reference
                ),
            },
            "duplicate_vs_canonical_history_sweep": {
                **history_errors,
                "rtol": 3.0e-13,
                "atol": history_atol,
                "duplicate_checksum_sha256": _array_sha256(
                    duplicate_history_reference
                ),
                "canonical_checksum_sha256": _array_sha256(
                    canonical_history_reference
                ),
            },
        },
        "timings": {
            "public_solver_calls": {
                "operations": public_timings,
                "facade_over_direct_median_ratio": facade_median / direct_median,
                "facade_minus_direct_median_seconds": (
                    facade_median - direct_median
                ),
                "ratio_interpretation": (
                    "facade median divided by direct-canonical median for this "
                    "run; noise can make the observed difference negative"
                ),
            },
            "facade_canonicalization_only": {
                "batch": canonicalization_record,
                "calls_per_batch": config.canonicalization_batch,
                "median_seconds_per_call": canonicalization_per_call,
                "excludes_result_provenance_wrapper": True,
            },
            "combined_kernel_construction": {
                "operations": kernel_timings,
                "uncoalesced_over_coalesced_median_ratio": (
                    uncoalesced_kernel_median / coalesced_kernel_median
                ),
                "uncoalesced_minus_coalesced_median_seconds": (
                    uncoalesced_kernel_median - coalesced_kernel_median
                ),
                "ratio_interpretation": (
                    "uncoalesced R-term median divided by coalesced canonical-R "
                    "median on this host; greater than one means the measured "
                    "uncoalesced construction took longer"
                ),
            },
            "direct_history_sweep_separate": {
                "operations": history_timings,
                "uncoalesced_kernel_over_coalesced_kernel_median_ratio": (
                    uncoalesced_history_median / coalesced_history_median
                ),
                "expected_structural_dependence_on_original_R": "none",
                "interpretation": (
                    "both inputs are already one combined kernel of equal "
                    "length; this timing isolates O(N^2*d) history and should "
                    "not be counted as an R-coalescence benefit"
                ),
            },
        },
        "provenance": {
            "script_path": str(script_path),
            "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
            "facade_source_path": facade_source_path,
            "facade_source_sha256": facade_source_hash,
            "distributed_core_source_path": core_source_path,
            "distributed_core_source_sha256": core_source_hash,
            "facade_implementation_reuse": facade_reference.solver_info.get(
                "implementation_reuse"
            ),
            "facade_underlying_method": facade_reference.solver_info.get(
                "underlying_method"
            ),
            "measured_public_backends": {
                "facade": public_results[
                    "facade_from_duplicate_terms"
                ].backend,
                "direct": public_results["direct_from_canonical_terms"].backend,
            },
            "measured_kernel_hashes": {
                name: _array_sha256(result)
                for name, result in kernel_results.items()
            },
            "measured_history_checksum_hashes": {
                name: _array_sha256(result)
                for name, result in history_results.items()
            },
        },
        "interpretation_limits": [
            (
                "The facade/direct difference is small relative to a complete "
                "solve and may be timing noise."
            ),
            (
                "The isolated kernel comparison uses private existing kernels "
                "and is an implementation diagnostic, not public API."
            ),
            (
                "Coalescing exact duplicate orders preserves the finite equation "
                "only up to floating-point summation order."
            ),
            (
                "Near-equal orders are not coalesced and are outside this "
                "duplicate-order workload."
            ),
            (
                "History cost is measured separately and does not decrease with "
                "original R after one combined kernel exists."
            ),
            (
                "Numba compilation, cache population, and explicit warm-up "
                "samples are excluded from measured repetitions."
            ),
            (
                "Rerun on each target host and workload before making an "
                "engineering policy decision."
            ),
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup-calls", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--kernel-lags", type=int, default=4096)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--unique-orders", type=int, default=8)
    parser.add_argument("--duplicate-factor", type=int, default=16)
    parser.add_argument("--canonicalization-batch", type=int, default=64)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026080341)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    config = BenchmarkConfig(
        repeats=args.repeats,
        warmup_calls=args.warmup_calls,
        n_steps=args.n_steps,
        kernel_lags=args.kernel_lags,
        dimension=args.dimension,
        unique_orders=args.unique_orders,
        duplicate_factor=args.duplicate_factor,
        canonicalization_batch=args.canonicalization_batch,
        step=args.step,
        seed=args.seed,
    )
    try:
        _validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))

    record = run_benchmark(config)
    rendered = json.dumps(
        record,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
