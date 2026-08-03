from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "bench_tempered_convolution_quadrature.py"
)
PERSISTED_BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "outputs"
    / "benchmarks"
    / "tempered_convolution_quadrature_backends_20260803.json"
)


def _load_benchmark():
    specification = importlib.util.spec_from_file_location(
        "hafo_bench_tempered_convolution_quadrature", BENCHMARK
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _all_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _all_strings(key)
            yield from _all_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _all_strings(item)


def test_tempered_cq_benchmark_fixture_is_deterministic_and_nontrivial() -> None:
    benchmark = _load_benchmark()
    workload = benchmark.Workload("fixture_n12_d3", 12, 3, 2026080391)
    first = benchmark._make_problem(workload)
    second = benchmark._make_problem(workload)

    np.testing.assert_array_equal(first.samples, second.samples)
    np.testing.assert_array_equal(first.orders, second.orders)
    np.testing.assert_array_equal(first.tempering, second.tempering)
    assert first.samples.shape == (12, 3)
    assert np.all(np.isfinite(first.samples))
    assert np.all((first.orders > 0.0) & (first.orders <= 1.0))
    assert np.all(first.tempering > 0.0)
    assert np.all(first.samples[0] != 0.0)
    assert len({case.name for case in benchmark.OPERATOR_CASES}) == 4
    assert {
        (case.definition, case.bdf_order)
        for case in benchmark.OPERATOR_CASES
    } == {
        ("tempered_riemann_liouville", 1),
        ("tempered_riemann_liouville", 2),
        ("tempered_caputo", 1),
        ("tempered_caputo", 2),
    }


def test_tempered_cq_benchmark_minimal_protocol_schema_and_semantics() -> None:
    benchmark = _load_benchmark()
    workload = benchmark.Workload("test_n10_d2", 10, 2, 2026080392)
    payload = benchmark.run_benchmark(repeats=3, workloads=(workload,))

    assert payload["schema_version"] == 1
    assert payload["scope"] == "tempered_fractional_sampled_operator_backend_performance"
    protocol = payload["measurement_protocol"]
    assert protocol["backend_parity_before_measurement"] is True
    assert protocol["numba_warmup_workload_excluded_from_measurements"] is True
    assert protocol["measured_repetitions_per_backend_case"] == 3
    assert protocol["backends"] == ["python", "numba", "fft"]
    assert "not fast-history" in protocol["fft_scope"]

    assert len(payload["workloads"]) == 1
    record = payload["workloads"][0]
    assert record["workload"]["name"] == workload.name
    assert len(record["fixture"]["input_sha256"]) == 64
    assert len(record["cases"]) == 4
    for case in record["cases"]:
        assert case["parity_checked_before_measurement"] is True
        assert set(case["timing"]) == {"python", "numba", "fft"}
        assert set(case["parity"]["comparisons"]) == {"numba", "fft"}
        assert "not streaming" in case["fft_scope"]
        assert len(case["backend_execution_order_by_repetition"]) == 3
        for backend in ("python", "numba", "fft"):
            timing = case["timing"][backend]
            assert len(timing["samples_seconds"]) == 3
            assert np.all(np.asarray(timing["samples_seconds"]) >= 0.0)
            assert np.isfinite(timing["median_seconds"])
        for comparison in case["parity"]["comparisons"].values():
            assert comparison["weights_exactly_equal_to_python"] is True
            assert comparison["base_weights_exactly_equal_to_python"] is True
            assert np.isfinite(
                comparison["maximum_absolute_value_difference_from_python"]
            )

    policy = payload["backend_policy_assessment"]
    assert policy["decision"] == "do_not_add_native_c_or_julia_from_this_evidence"
    assert policy["native_c_candidate_implemented_or_measured"] is False
    assert policy["julia_candidate_implemented_or_measured"] is False
    assert "not a fast-history" in policy["fft_contract"]
    assert set(policy["observed_fastest_backend_counts"]) == {
        "python",
        "numba",
        "fft",
    }
    assert sum(policy["observed_fastest_backend_counts"].values()) == 4

    rendered = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert json.loads(rendered)["benchmark_id"] == payload["benchmark_id"]
    drive_absolute = re.compile(r"^[A-Za-z]:[\\/]")
    posix_absolute = re.compile(r"^/(?:home|tmp|Users|workspace)(?:/|$)")
    assert not any(
        drive_absolute.search(text) or posix_absolute.search(text)
        for text in _all_strings(payload)
    )


def test_persisted_tempered_cq_benchmark_matches_current_protocol() -> None:
    payload = json.loads(PERSISTED_BENCHMARK.read_text(encoding="utf-8"))
    assert payload["script_sha256"] == hashlib.sha256(
        BENCHMARK.read_bytes()
    ).hexdigest()
    assert len(payload["workloads"]) == 3
    cases = [
        case
        for workload in payload["workloads"]
        for case in workload["cases"]
    ]
    assert len(cases) == 12
    assert all(case["parity_checked_before_measurement"] for case in cases)
    assert max(
        case["parity"]["comparisons"]["numba"]
        ["maximum_absolute_value_difference_from_python"]
        for case in cases
    ) == 0.0
    assert max(
        case["parity"]["comparisons"]["fft"]
        ["maximum_absolute_value_difference_from_python"]
        for case in cases
    ) < 2.0e-12
    policy = payload["backend_policy_assessment"]
    assert policy["decision"] == "do_not_add_native_c_or_julia_from_this_evidence"
    assert policy["observed_fastest_backend_counts"] == {
        "python": 0,
        "numba": 0,
        "fft": 12,
    }


@pytest.mark.parametrize("repeats", [True, 0, 2, 3.5])
def test_tempered_cq_benchmark_rejects_invalid_repeat_counts(repeats: Any) -> None:
    benchmark = _load_benchmark()
    with pytest.raises(ValueError, match="at least 3"):
        benchmark.run_benchmark(repeats=repeats, workloads=(benchmark.WORKLOADS[0],))
