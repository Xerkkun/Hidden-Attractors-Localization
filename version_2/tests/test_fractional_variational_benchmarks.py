import sys
from pathlib import Path
_VALIDATION_PYTHON_DIR = str(Path(__file__).resolve().parents[1] / "validation" / "python")
if _VALIDATION_PYTHON_DIR not in sys.path:
    sys.path.insert(0, _VALIDATION_PYTHON_DIR)

from fractional_variational_benchmarks import (
    load_benchmark_case,
    run_benchmark_case
)
import os
import numpy as np
import pytest
from hidden_attractors.analysis.lyapunov_methods import LYAPUNOV_METHODS
from lyapunov_method_registry import VALIDATION_LYAPUNOV_METHODS


# Resolve default benchmarks directory
BENCHMARKS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "validation", "lyapunov_benchmarks", "fractional_variational_abm_qr")
)

# ---------------------------------------------------------------------------
# 1. test_benchmark_yaml_files_exist
# ---------------------------------------------------------------------------

def test_benchmark_yaml_files_exist() -> None:
    """Verify that all completed benchmark configuration files exist."""
    expected_files = [
        "synthetic_zero_rhs.yaml",
        "synthetic_linear_stable.yaml",
        "published_dk2018_rabinovich_fabrikant_q0999.yaml",
        "published_dk2018_lorenz_q0985.yaml",
        "published_dk2018_4d_nonsmooth_q098_qualitative.yaml",
    ]
    for filename in expected_files:
        path = os.path.join(BENCHMARKS_DIR, filename)
        assert os.path.isfile(path), f"Benchmark file {filename} does not exist at {path}."


# ---------------------------------------------------------------------------
# 2. test_synthetic_zero_rhs_benchmark_passes_fast
# ---------------------------------------------------------------------------

def test_synthetic_zero_rhs_benchmark_passes_fast() -> None:
    """Verify that the synthetic_zero_rhs benchmark passes in fast mode."""
    yaml_path = os.path.join(BENCHMARKS_DIR, "synthetic_zero_rhs.yaml")
    case_data = load_benchmark_case(yaml_path)
    res = run_benchmark_case(case_data, fast=True)
    assert res["status"] == "synthetic_benchmark_passed"
    assert res["computed_exponents"] is not None
    assert np.all(np.abs(res["computed_exponents"]) < 1e-3)


# ---------------------------------------------------------------------------
# 3. test_synthetic_linear_stable_benchmark_passes_fast
# ---------------------------------------------------------------------------

def test_synthetic_linear_stable_benchmark_passes_fast() -> None:
    """Verify that the synthetic_linear_stable benchmark passes in fast mode."""
    yaml_path = os.path.join(BENCHMARKS_DIR, "synthetic_linear_stable.yaml")
    case_data = load_benchmark_case(yaml_path)
    res = run_benchmark_case(case_data, fast=True)
    # Expected: synthetic_benchmark_passed or benchmark_inconclusive
    # But definitely not failed. If it completes successfully, check max.
    assert res["status"] in ("synthetic_benchmark_passed", "benchmark_inconclusive")
    if res["status"] == "synthetic_benchmark_passed":
        assert np.max(res["computed_exponents"]) < 1e-2


# ---------------------------------------------------------------------------
# 4. test_published_template_missing_data_not_validated
# ---------------------------------------------------------------------------

def test_qualitative_reference_missing_data_not_validated() -> None:
    """Verify that an incomplete qualitative reference is not promoted."""
    yaml_path = os.path.join(
        BENCHMARKS_DIR,
        "published_dk2018_4d_nonsmooth_q098_qualitative.yaml",
    )
    case_data = load_benchmark_case(yaml_path)
    res = run_benchmark_case(case_data, fast=True)
    assert res["status"] == "published_reference_data_missing_qualitative_only"
    assert res["computed_exponents"] is None


# ---------------------------------------------------------------------------
# 5. test_no_published_validated_without_complete_data
# ---------------------------------------------------------------------------

def test_no_published_validated_without_complete_data() -> None:
    """Verify that a case cannot be marked as published_benchmark_passed_quantitative if data_complete is false."""
    yaml_path = os.path.join(
        BENCHMARKS_DIR,
        "published_dk2018_4d_nonsmooth_q098_qualitative.yaml",
    )
    case_data = load_benchmark_case(yaml_path)
    # Explicitly verify that data_complete is False in the specification
    assert case_data["reference"]["data_complete"] is False
    res = run_benchmark_case(case_data, fast=True)
    assert res["status"] != "published_benchmark_passed_quantitative"


# ---------------------------------------------------------------------------
# 6. test_registry_keeps_validated_false_until_published_benchmark
# ---------------------------------------------------------------------------

def test_registry_keeps_validated_false_until_published_benchmark() -> None:
    """Verify that the global validated flag remains False in the methods registry."""
    info = LYAPUNOV_METHODS["fractional_variational_abm_qr"]
    assert info.validated is False
    assert info.implemented is True
    assert info.validated_against_synthetic_tests is True
    assert info.validated_against_published_benchmarks is False
    assert info.benchmark_status == "synthetic_validation_only"


def test_dk2018_registry_reports_reproduced_rf_lambda_3_discrepancy() -> None:
    assert "fractional_variational_dk2018_block_restart_abm_gs" not in LYAPUNOV_METHODS
    info = VALIDATION_LYAPUNOV_METHODS[
        "fractional_variational_dk2018_block_restart_abm_gs"
    ]
    assert info.validated is False
    assert info.validated_against_published_benchmarks is False
    assert info.benchmark_status == "recorded_published_discrepancy"


# ---------------------------------------------------------------------------
# 7. test_benchmark_summary_no_chaos_hidden_verified
# ---------------------------------------------------------------------------

def test_benchmark_summary_no_chaos_hidden_verified() -> None:
    """Verify that no verification summary contains positive chaos/hidden attractor validation claims."""
    # Check registry warning texts
    info = LYAPUNOV_METHODS["fractional_variational_abm_qr"]
    combined_warnings = " ".join(info.warnings).lower()
    
    assert "chaos_verified: true" not in combined_warnings
    assert "hidden_verified: true" not in combined_warnings
    
    # Check that the finite-time scope note remains centralized.
    assert "scope: finite-time numerical lyapunov evidence" in combined_warnings


# ---------------------------------------------------------------------------
# 8. test_non_aligned_burn_time_is_rejected
# ---------------------------------------------------------------------------

def test_non_aligned_burn_time_is_rejected() -> None:
    """Do not silently round a Caputo burn horizon onto another fixed grid."""
    from hidden_attractors.analysis.lyapunov_fractional import fractional_variational_abm_qr
    rhs = lambda x: np.array([-x[0]])
    jac = lambda x: np.array([[-1.0]])
    with pytest.raises(ValueError, match="integer number of fixed steps"):
        fractional_variational_abm_qr(
            rhs, jac, np.ones(1),
            q=0.9, h=0.02, t_burn=0.13, t_final=0.2,
            reorthonormalization_time=0.10
        )


@pytest.mark.native
@pytest.mark.parametrize(
    "filename",
    [
        "published_dk2018_rabinovich_fabrikant_q0999.yaml",
        "published_dk2018_lorenz_q0985.yaml",
    ],
)
def test_published_native_cases_smoke_without_quantitative_promotion(filename: str) -> None:
    case_data = load_benchmark_case(os.path.join(BENCHMARKS_DIR, filename))
    res = run_benchmark_case(case_data, fast=True)
    assert res["status"] == "published_benchmark_smoke_passed"
    assert res["numerical_route"] == "native_c"
    assert res["execution_contract"] == "dk2018_block_restart_abm_gs"
    assert res["validation_run_class"] == "published_smoke_fast"


def test_nonsmooth_4d_case_remains_qualitative_only() -> None:
    case_data = load_benchmark_case(
        os.path.join(BENCHMARKS_DIR, "published_dk2018_4d_nonsmooth_q098_qualitative.yaml")
    )
    res = run_benchmark_case(case_data, fast=True)
    assert res["status"] == "published_reference_data_missing_qualitative_only"
    assert res["computed_exponents"] is None


@pytest.mark.slow
@pytest.mark.published
@pytest.mark.native
@pytest.mark.skipif(
    os.environ.get("RUN_PUBLISHED_LYAPUNOV") != "1",
    reason="Set RUN_PUBLISHED_LYAPUNOV=1 to run extensive published-value benchmarks.",
)
@pytest.mark.parametrize(
    ("filename", "expected_status", "expected_failing_components"),
    [
        (
            "published_dk2018_rabinovich_fabrikant_q0999.yaml",
            "published_benchmark_failed",
            ["lambda_3"],
        ),
        (
            "published_dk2018_lorenz_q0985.yaml",
            "published_benchmark_passed_quantitative",
            [],
        ),
    ],
)
def test_published_native_cases_match_or_record_discrepancy(
    filename: str,
    expected_status: str,
    expected_failing_components: list[str],
) -> None:
    case_data = load_benchmark_case(os.path.join(BENCHMARKS_DIR, filename))
    res = run_benchmark_case(case_data, fast=False)
    assert res["status"] == expected_status
    assert res["failing_components"] == expected_failing_components
    assert np.all(np.isfinite(res["computed_exponents"]))

    if expected_failing_components:
        differences = np.asarray(res["absolute_differences"], dtype=float)
        assert np.all(differences[:2] < res["absolute_tolerance"])
        assert differences[2] >= res["absolute_tolerance"]
        info = VALIDATION_LYAPUNOV_METHODS[
            "fractional_variational_dk2018_block_restart_abm_gs"
        ]
        assert info.benchmark_status == "recorded_published_discrepancy"
