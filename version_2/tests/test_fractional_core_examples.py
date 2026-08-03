from __future__ import annotations

from examples.caputo_hadamard_chua import run_example as run_caputo_hadamard_chua
from examples.chua_advanced_analysis import run_example as run_advanced_analysis
from examples.fractional_core_catalog import run_catalog


def test_fractional_core_catalog_example_runs_with_distinct_operator_contracts() -> None:
    record = run_catalog(sample_count=65)
    assert record["manufactured_gl_solver"]["status"] == "ok"
    assert record["manufactured_conformable_solver"]["status"] == "ok"
    assert record["manufactured_conformable_solver"]["memory_policy"] == (
        "none_local_operator"
    )
    assert record["manufactured_abc_solver"]["status"] == "ok"
    assert record["manufactured_abc_solver"]["compatibility_residual"] == 0.0
    assert record["manufactured_tempered_caputo_solver"]["status"] == "ok"
    assert record["manufactured_tempered_caputo_solver"]["tempering"] == 0.4
    assert record["manufactured_variable_order_type3_solver"]["status"] == "ok"
    assert record["manufactured_variable_order_type3_solver"]["definition"] == (
        "tavares_type_iii_current_time"
    )
    assert record["manufactured_variable_order_type3_solver"]["order_min"] == 0.52
    assert record["manufactured_variable_order_type3_solver"]["order_max"] == 0.62
    assert record["registry"]["hadamard_cq_execution_kind"] == "sampled_operator"
    assert record["backend_checks"]["gl_direct_vs_fft_max_abs"] < 1.0e-11
    assert "caputo_fabrizio" in record["operator_endpoints"]
    assert "atangana_baleanu_caputo" in record["operator_endpoints"]
    assert record["backend_checks"]["atangana_baleanu_backend"] == "numpy_fft_offline"


def test_caputo_hadamard_chua_example_keeps_claim_boundary() -> None:
    record = run_caputo_hadamard_chua(
        log_duration=0.02,
        log_step=0.01,
        use_acceleration=False,
    )
    assert record["fractional"]["status"] == "ok"
    assert record["integer_q1_log_coordinate_reference"]["status"] == "ok"
    assert record["fractional"]["samples"] == 3
    assert "no chaos" in record["claims"].lower()


def test_advanced_analysis_example_exercises_real_trajectory_and_basin_metrics() -> None:
    record = run_advanced_analysis()
    assert record["chua"]["integer_solver_status"] == "ok"
    assert record["chua"]["recurrence"]["auto_rate"] > 0.0
    assert record["bistable_flow"]["log_two_criterion_reason"] == (
        "fewer_than_three_observed_basins"
    )
    assert record["bistable_flow"]["uncertainty_fraction_at_0.1"] > 0.0
