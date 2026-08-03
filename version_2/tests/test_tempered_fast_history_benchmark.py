from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "bench_tempered_fast_history.py"
)


def _load_benchmark():
    specification = importlib.util.spec_from_file_location(
        "hafo_bench_tempered_fast_history", BENCHMARK
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


def test_fast_history_benchmark_fixture_and_gngf2_reference_are_deterministic() -> None:
    benchmark = _load_benchmark()
    workload = benchmark.Workload(
        "fixture_n40_d2", 40, 2, 6, 1.0e-5, 257, 2026080393
    )
    first = benchmark._make_problem(workload)
    second = benchmark._make_problem(workload)
    np.testing.assert_array_equal(first.samples, second.samples)
    np.testing.assert_array_equal(first.orders, second.orders)
    np.testing.assert_array_equal(first.tempering, second.tempering)
    assert first.samples.shape == (40, 2)
    assert np.all(first.samples[0] != 0.0)

    for case in benchmark.DEFINITION_CASES:
        direct = benchmark._run_gngf2_reference(
            first, case, "direct_reference"
        )
        fft = benchmark._run_gngf2_reference(first, case, "fft_reference")
        np.testing.assert_allclose(
            fft.values,
            direct.values,
            rtol=benchmark.PARITY_RTOL,
            atol=benchmark.PARITY_ATOL,
        )
        np.testing.assert_array_equal(fft.base_weights, direct.base_weights)
        np.testing.assert_array_equal(
            fft.tempered_weights, direct.tempered_weights
        )
        assert direct.fft_length is None
        assert fft.fft_length == benchmark._fft_length(workload.n_times)


def test_fast_history_benchmark_minimal_protocol_and_memory_semantics() -> None:
    benchmark = _load_benchmark()
    workload = benchmark.Workload(
        "test_n72_d2", 72, 2, 8, 2.0e-5, 257, 2026080394
    )
    payload = benchmark.run_benchmark(repeats=3, workloads=(workload,))

    assert payload["schema_version"] == 1
    protocol = payload["measurement_protocol"]
    assert protocol["backend_parity_before_measurement"] is True
    assert protocol["automatic_Q_selection_timed_separately"] is True
    assert protocol["fixed_validated_Q_in_repeated_fast_calls"] is True
    assert protocol["gngf2_is_fractional_bdf2"] is False
    assert protocol["numba_warmup"][
        "all_warmup_calls_excluded_from_repeated_measurements"
    ] is True
    assert "not fast-history" in protocol["fft_scope"]

    record = payload["workloads"][0]
    assert record["workload"]["n_times"] == 72
    assert len(record["fixture"]["input_sha256"]) == 64
    assert len(record["cases"]) == 4
    for case in record["cases"]:
        assert case["parity_checked_before_measurement"] is True
        assert case["parity"]["compression_tolerance_satisfied"] is True
        assert case["Q_selection"][
            "selected_Q_fixed_during_all_repeated_fast_calls"
        ] is True
        q_points = case["Q_selection"]["selected_quadrature_points"]
        assert q_points >= 65
        memory = case["fast_active_memory"]
        assert memory["N"] == 72
        assert memory["Q"] == q_points
        assert memory["d"] == 2
        expected_active = 8 * memory["d"] * (
            3 * memory["Q"] + memory["n0"] + 1
        )
        assert (
            memory["evaluator_active_history_bytes_excluding_input_and_output"]
            == expected_active
        )
        assert memory["recurrence_state_bytes"] == 8 * memory["Q"] * 2
        if case["multistep_method"] == "fbdf1":
            assert set(case["timing"]) == {
                "fast_python",
                "fast_numba",
                "direct_numba",
                "fft_batch",
            }
        else:
            assert "not fractional BDF2" in case["generator_contract"]
            assert set(case["timing"]) == {
                "fast_python",
                "fast_numba",
                "direct_reference",
                "fft_reference",
            }
        for timing in case["timing"].values():
            assert len(timing["samples_seconds"]) == 3
            assert np.all(np.asarray(timing["samples_seconds"]) >= 0.0)
            assert np.isfinite(timing["median_seconds"])

    policy = payload["backend_policy_assessment"]
    assert policy["decision"] == "insufficient_evidence_to_add_native_c_or_julia"
    assert policy["native_c_candidate_implemented_or_measured"] is False
    assert policy["julia_candidate_implemented_or_measured"] is False
    assert "not streaming" in policy["fft_interpretation"]
    assert sum(policy["observed_fastest_backend_counts"].values()) == 4

    rendered = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert json.loads(rendered)["benchmark_id"] == payload["benchmark_id"]
    drive_absolute = re.compile(r"^[A-Za-z]:[\\/]")
    posix_absolute = re.compile(r"^/(?:home|tmp|Users|workspace)(?:/|$)")
    assert not any(
        drive_absolute.search(text) or posix_absolute.search(text)
        for text in _all_strings(payload)
    )


def test_fast_history_benchmark_default_output_is_unique_and_temp_scoped() -> None:
    benchmark = _load_benchmark()
    first = benchmark._default_output_path()
    second = benchmark._default_output_path()
    expected_root = Path(r"C:\tmp") if os.name == "nt" else Path(tempfile.gettempdir())
    assert first.parent == expected_root
    assert second.parent == expected_root
    assert first != second
    assert first.suffix == ".json"
    assert "tempered_fast_history" in first.name


@pytest.mark.parametrize("repeats", [True, 0, 2, 3.5])
def test_fast_history_benchmark_rejects_invalid_repeat_counts(repeats: Any) -> None:
    benchmark = _load_benchmark()
    with pytest.raises(ValueError, match="at least 3"):
        benchmark.run_benchmark(
            repeats=repeats, workloads=(benchmark.WORKLOADS[0],)
        )
