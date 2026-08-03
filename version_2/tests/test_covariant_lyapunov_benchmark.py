from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


BENCHMARK = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_covariant_lyapunov.py"


def _load_benchmark():
    specification = importlib.util.spec_from_file_location(
        "hafo_bench_covariant_lyapunov", BENCHMARK
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_clv_benchmark_fixture_is_a_valid_deterministic_qr_history() -> None:
    benchmark = _load_benchmark()
    workload = benchmark.Workload("fixture", 8, 3, 5, 3, 12345)
    first = benchmark._make_problem(workload)
    second = benchmark._make_problem(workload)

    np.testing.assert_array_equal(first.matrix, second.matrix)
    np.testing.assert_array_equal(first.orthonormal_bases, second.orthonormal_bases)
    np.testing.assert_array_equal(first.observed_r_factors, second.observed_r_factors)
    np.testing.assert_array_equal(first.future_r_factors, second.future_r_factors)
    np.testing.assert_allclose(
        np.einsum(
            "sdi,sdj->sij",
            first.orthonormal_bases,
            first.orthonormal_bases,
        ),
        np.repeat(
            np.eye(workload.n_vectors)[None, :, :],
            workload.observed_segments + 1,
            axis=0,
        ),
        atol=2.0e-15,
    )
    assert np.all(np.diagonal(first.observed_r_factors, axis1=1, axis2=2) > 0.0)
    assert np.all(np.diagonal(first.future_r_factors, axis1=1, axis2=2) > 0.0)
    np.testing.assert_allclose(np.tril(first.observed_r_factors, -1), 0.0)
    np.testing.assert_allclose(np.tril(first.future_r_factors, -1), 0.0)


def test_clv_benchmark_small_protocol_checks_parity_and_reports_both_scopes() -> None:
    benchmark = _load_benchmark()
    workload = benchmark.Workload("test_s8_f3_k2_d4", 8, 3, 4, 2, 2026080399)
    payload = benchmark.run_benchmark(repeats=3, workloads=(workload,))

    assert payload["schema_version"] == 1
    assert payload["measurement_protocol"]["backend_parity_before_measurement"] is True
    assert payload["measurement_protocol"]["measured_repetitions_per_backend_and_scope"] == 3
    record = payload["workloads"][0]
    assert record["parity_checked_before_measurement"] is True
    assert set(record["history_public_api"]["timing"]) == {"numpy", "numba"}
    assert set(record["integer_map_end_to_end"]["timing"]) == {"numpy", "numba"}
    assert (
        record["integer_map_end_to_end"][
            "warmed_numba_public_reconstruction_fraction_of_end_to_end"
        ]
        > 0.0
    )
    assert record["history_public_api"]["parity"][
        "maximum_absolute_vector_difference"
    ] < 1.0e-11
    assert record["integer_map_end_to_end"]["parity"][
        "maximum_absolute_vector_difference"
    ] < 1.0e-11
    assert payload["native_c_assessment"]["native_c_backend_implemented_or_measured"] is False
