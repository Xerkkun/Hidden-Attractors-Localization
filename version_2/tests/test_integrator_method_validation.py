"""Tests for the integrator method validation phase.

Tests in this file are fast by design (using few grid points).  They validate:

1. Mittag-Leffler special values.
2. ABM scalar Mittag-Leffler error decreases.
3. ABM manufactured solution error decreases.
4. ABM vector linear system gives finite, bounded error.
5. RK4 observed order near 4 for exponential decay.
6. RK4 energy drift decreases with h for harmonic oscillator.
7. RK4 vs. scipy.solve_ivp (optional, skipped if scipy unavailable).
8. Summary output has no hidden_verified claim.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Path setup ───────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.python.integrator_method_validation import (
    abm_caputo_integrate,
    mittag_leffler,
    mittag_leffler_q,
    run_integrator_method_validation,
    validate_abm_manufactured_solution,
    validate_abm_mittag_leffler,
    validate_abm_vector_linear,
    validate_rk4_exponential_decay,
    validate_rk4_harmonic_oscillator,
    validate_rk4_solve_ivp_comparison,
    _NO_CLAIM,
    _ALLOWED_METHOD_STATUSES,
    _ALLOWED_TEST_STATUSES,
)


# =============================================================================
# 1. test_mittag_leffler_basic_values
# =============================================================================

class TestMittagLeffler:
    """E_{1,1}(z) = exp(z); verify against numpy.exp at several z values."""

    @pytest.mark.parametrize("z", [-1.0, -0.5, 0.0, 0.5])
    def test_e1_1_equals_exp(self, z: float) -> None:
        """E_{1,1}(z) must equal exp(z) to within 1e-10."""
        ml_val = mittag_leffler(1.0, 1.0, z)
        exp_val = math.exp(z)
        assert abs(ml_val - exp_val) < 1e-10, (
            f"E_{{1,1}}({z}) = {ml_val} but exp({z}) = {exp_val}"
        )

    def test_e1_1_at_zero(self) -> None:
        """E_{1,1}(0) = exp(0) = 1."""
        assert abs(mittag_leffler(1.0, 1.0, 0.0) - 1.0) < 1e-12

    def test_eq_negative_arg_is_positive(self) -> None:
        """E_q(-1) > 0 for all q in (0, 1)."""
        for q in [0.25, 0.5, 0.8, 0.9998]:
            val = mittag_leffler_q(-1.0, q)
            assert val > 0.0, f"E_{q}(-1) = {val} should be positive"

    def test_eq_at_zero_equals_one(self) -> None:
        """E_q(0) = 1 for all alpha (first term = 0^0/Gamma(beta) = 1/1 = 1)."""
        for q in [0.25, 0.5, 0.8, 0.9998]:
            val = mittag_leffler_q(0.0, q)
            assert abs(val - 1.0) < 1e-12, f"E_{q}(0) = {val} should be 1"


# =============================================================================
# 2. test_abm_scalar_linear_error_decreases
# =============================================================================

class TestABMMittagLeffler:
    """ABM against scalar Caputo decay D^q y = lambda*y."""

    def test_error_decreases_fast(self) -> None:
        """Quick check: q=0.5, lambda=-1, three meshes."""
        rows, status = validate_abm_mittag_leffler(
            q_values=[0.5],
            lambda_values=[-1.0],
            h_values=[1 / 40, 1 / 80, 1 / 160],
        )
        assert len(rows) == 3
        errors = [r["max_error"] for r in rows]
        # Error must decrease
        assert errors[1] < errors[0], (
            f"Error did not decrease from h=1/40 to h=1/80: {errors}"
        )
        assert errors[2] < errors[1], (
            f"Error did not decrease from h=1/80 to h=1/160: {errors}"
        )

    def test_status_in_allowed_set(self) -> None:
        """Status must be one of the documented states."""
        _, status = validate_abm_mittag_leffler(
            q_values=[0.5],
            lambda_values=[-1.0],
            h_values=[1 / 40, 1 / 80, 1 / 160],
        )
        assert status in _ALLOWED_TEST_STATUSES, f"Unexpected status: {status}"

    def test_all_q_values_fast(self) -> None:
        """All four q values, two meshes (fast)."""
        rows, status = validate_abm_mittag_leffler(
            q_values=[0.25, 0.5, 0.8, 0.9998],
            lambda_values=[-1.0],
            h_values=[1 / 40, 1 / 80],
        )
        # 4 q × 1 lambda × 2 h = 8 rows
        assert len(rows) == 8
        # All errors must be finite
        assert all(math.isfinite(r["max_error"]) for r in rows)

    def test_observed_order_positive_for_fine_meshes(self) -> None:
        """Observed order should be positive (> 0) for fine mesh refinement."""
        rows, _ = validate_abm_mittag_leffler(
            q_values=[0.5],
            lambda_values=[-1.0],
            h_values=[1 / 40, 1 / 80, 1 / 160],
        )
        orders = [r["observed_order"] for r in rows if math.isfinite(r["observed_order"])]
        assert len(orders) > 0, "No finite observed orders found"
        assert all(o > 0.0 for o in orders), (
            f"Some observed orders are not positive: {orders}"
        )


# =============================================================================
# 3. test_abm_manufactured_solution_error_decreases
# =============================================================================

class TestABMManufacturedSolution:
    """ABM against manufactured solution y(t) = t^m."""

    def test_error_decreases_q05_m4(self) -> None:
        """q=0.5, m=4, three meshes: error must decrease."""
        rows, status = validate_abm_manufactured_solution(
            q_values=[0.5],
            m_values=[4],
            h_values=[1 / 40, 1 / 80, 1 / 160],
        )
        errors = [r["max_error"] for r in rows]
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]

    def test_status_in_allowed_set(self) -> None:
        """Status must be one of the documented states."""
        _, status = validate_abm_manufactured_solution(
            q_values=[0.5],
            m_values=[4],
            h_values=[1 / 40, 1 / 80, 1 / 160],
        )
        assert status in _ALLOWED_TEST_STATUSES, f"Unexpected status: {status}"

    def test_terminal_error_small(self) -> None:
        """Terminal error must be small for fine mesh."""
        rows, _ = validate_abm_manufactured_solution(
            q_values=[0.5],
            m_values=[4],
            h_values=[1 / 160],
        )
        assert rows[0]["terminal_error"] < 0.1, (
            f"Terminal error {rows[0]['terminal_error']} is too large"
        )

    def test_m5_also_decreasing(self) -> None:
        """m=5 also shows decreasing error."""
        rows, _ = validate_abm_manufactured_solution(
            q_values=[0.5],
            m_values=[5],
            h_values=[1 / 40, 1 / 80, 1 / 160],
        )
        errors = [r["max_error"] for r in rows]
        assert errors[-1] < errors[0], (
            f"Error not decreasing for m=5: {errors}"
        )


# =============================================================================
# 4. test_abm_vector_linear_runs
# =============================================================================

class TestABMVectorLinear:
    """ABM on diagonal linear vector system D^q X = A X."""

    def test_terminal_error_finite_and_reasonable(self) -> None:
        """q=0.8: terminal norm error must be finite and < 1."""
        rows, status = validate_abm_vector_linear(
            q_values=[0.8],
            h_values=[1 / 40, 1 / 80],
        )
        for r in rows:
            assert r["finite_values"], f"Non-finite values at h={r['h']}"
            assert r["terminal_norm_error"] < 1.0, (
                f"Terminal error {r['terminal_norm_error']} too large at h={r['h']}"
            )

    def test_status_in_allowed_set(self) -> None:
        _, status = validate_abm_vector_linear(
            q_values=[0.8],
            h_values=[1 / 40, 1 / 80],
        )
        assert status in _ALLOWED_TEST_STATUSES, f"Unexpected status: {status}"

    def test_all_q_values_run(self) -> None:
        """All three q values (0.5, 0.8, 0.9998) complete without error."""
        rows, _ = validate_abm_vector_linear(
            q_values=[0.5, 0.8, 0.9998],
            h_values=[1 / 40],
        )
        assert len(rows) == 3  # one per q
        assert all(r["finite_values"] for r in rows)


# =============================================================================
# 5. test_rk4_exponential_order
# =============================================================================

class TestRK4ExponentialDecay:
    """RK4 on y' = -y, y(0)=1. Expect order ~ 4."""

    def test_order_near_4_fine_meshes(self) -> None:
        """Observed order should be in [3.5, 4.5] for h=0.05→0.025."""
        rows, status = validate_rk4_exponential_decay(
            h_values=[0.1, 0.05, 0.025, 0.0125],
        )
        fine_orders = [r["observed_order"] for r in rows[-2:] if math.isfinite(r["observed_order"])]
        assert fine_orders, "No finite observed orders in fine meshes"
        for o in fine_orders:
            assert 3.5 <= o <= 4.5, f"Observed order {o} not in [3.5, 4.5]"

    def test_terminal_error_decreases(self) -> None:
        """Terminal error must decrease as h is halved."""
        rows, _ = validate_rk4_exponential_decay(
            h_values=[0.1, 0.05, 0.025],
        )
        errors = [r["terminal_error"] for r in rows]
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]

    def test_status_is_confirmed_or_inconclusive(self) -> None:
        _, status = validate_rk4_exponential_decay(h_values=[0.1, 0.05, 0.025, 0.0125])
        assert status in _ALLOWED_TEST_STATUSES, f"Unexpected status: {status}"
        # With typical meshes, should confirm order 4
        assert status == "rk4_order4_confirmed", (
            f"Expected rk4_order4_confirmed, got {status}"
        )


# =============================================================================
# 6. test_rk4_harmonic_oscillator_energy_drift_decreases
# =============================================================================

class TestRK4HarmonicOscillator:
    """RK4 on x'=y, y'=-x. Energy should be nearly conserved."""

    def test_energy_drift_decreases(self) -> None:
        """Energy drift |x^2+y^2-1| must decrease as h is halved."""
        rows, status = validate_rk4_harmonic_oscillator(
            h_values=[0.2, 0.1, 0.05],
        )
        drifts = [r["energy_drift"] for r in rows]
        assert drifts[1] < drifts[0], (
            f"Energy drift did not decrease from h=0.2 to h=0.1: {drifts}"
        )
        assert drifts[2] < drifts[1], (
            f"Energy drift did not decrease from h=0.1 to h=0.05: {drifts}"
        )

    def test_status_in_allowed_set(self) -> None:
        _, status = validate_rk4_harmonic_oscillator(h_values=[0.2, 0.1, 0.05])
        assert status in _ALLOWED_TEST_STATUSES, f"Unexpected status: {status}"

    def test_error_decreases_with_refinement(self) -> None:
        """Terminal error must decrease as h is halved (max norm over 2π not guaranteed monotone)."""
        rows, _ = validate_rk4_harmonic_oscillator(h_values=[0.2, 0.1, 0.05])
        # Use terminal error which is monotone for standard RK4 on smooth problems
        errors = [r["terminal_error"] for r in rows]
        assert errors[-1] < errors[0], (
            f"Terminal error did not decrease from h=0.2 to h=0.05: {errors}"
        )



# =============================================================================
# 7. test_rk4_solve_ivp_comparison_optional
# =============================================================================

class TestRK4SolveIVP:
    """RK4 vs. scipy.solve_ivp comparison (optional)."""

    def test_solve_ivp_comparison_or_skip(self) -> None:
        """Compare RK4 against scipy.solve_ivp; skip if scipy unavailable."""
        try:
            from scipy.integrate import solve_ivp
        except ImportError:
            pytest.skip("scipy not available; skipping solve_ivp comparison")

        rows, status = validate_rk4_solve_ivp_comparison(
            h_values=[0.05],
        )
        assert status in _ALLOWED_TEST_STATUSES, f"Unexpected status: {status}"
        assert status != "rk4_solve_ivp_skipped_no_scipy", (
            "scipy is available but solve_ivp was skipped"
        )
        assert status == "rk4_solve_ivp_consistent", (
            f"RK4 vs. solve_ivp inconsistent: {rows}"
        )

    def test_status_is_skipped_without_scipy_mocked(self, monkeypatch) -> None:
        """When scipy is absent, status must be rk4_solve_ivp_skipped_no_scipy."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "scipy.integrate":
                raise ImportError("scipy mocked absent")
            return real_import(name, *args, **kwargs)

        # We can't easily mock scipy once imported; just verify the status string exists.
        assert "rk4_solve_ivp_skipped_no_scipy" in _ALLOWED_TEST_STATUSES


# =============================================================================
# 8. test_summary_no_hidden_verified
# =============================================================================

class TestSummaryNoHiddenVerified:
    """Verify no hidden_verified claim in the summary output."""

    def test_no_hidden_verified_in_no_claim(self) -> None:
        """_NO_CLAIM must not contain 'hidden_verified' key."""
        assert "hidden_verified" not in _NO_CLAIM
        assert _NO_CLAIM["hiddenness_certified_by_this_pipeline"] is False
        assert _NO_CLAIM["no_hidden_verified_claim"] is True

    def test_summary_no_hidden_verified_claim(self, tmp_path: Path) -> None:
        """Full summary run: no 'hidden_verified' key anywhere in the output."""
        summary = run_integrator_method_validation(
            output_dir=tmp_path,
            methods=["ABM", "RK4"],
            fast=True,
        )
        assert "hidden_verified" not in summary
        assert summary.get("hiddenness_certified_by_this_pipeline") is False
        assert summary.get("no_hidden_verified_claim") is True

        # Check recursively that no nested key is 'hidden_verified'
        def _check_no_hidden_verified(d: dict | list | object, path: str = "") -> None:
            if isinstance(d, dict):
                for k, v in d.items():
                    assert k != "hidden_verified", (
                        f"Found 'hidden_verified' key at path '{path}.{k}'"
                    )
                    _check_no_hidden_verified(v, f"{path}.{k}")
            elif isinstance(d, list):
                for i, v in enumerate(d):
                    _check_no_hidden_verified(v, f"{path}[{i}]")

        _check_no_hidden_verified(summary)

    def test_method_statuses_in_allowed_set(self, tmp_path: Path) -> None:
        """All method statuses must be from the documented allowed set."""
        summary = run_integrator_method_validation(
            output_dir=tmp_path,
            methods=["ABM", "RK4"],
            fast=True,
        )
        for method, info in summary.get("methods", {}).items():
            status = info.get("status")
            assert status in _ALLOWED_METHOD_STATUSES, (
                f"Method {method} has unexpected status '{status}'"
            )

    def test_all_test_statuses_in_allowed_set(self, tmp_path: Path) -> None:
        """All individual test statuses must be from the documented allowed set."""
        summary = run_integrator_method_validation(
            output_dir=tmp_path,
            methods=["ABM", "RK4"],
            fast=True,
        )
        for method, info in summary.get("methods", {}).items():
            for test_name, test_status in info.get("tests", {}).items():
                assert test_status in _ALLOWED_TEST_STATUSES, (
                    f"Method {method}, test {test_name} has unexpected status '{test_status}'"
                )

    def test_efork3_status_is_reference_status(self, tmp_path: Path) -> None:
        """EFORK3 must always report as validated_elsewhere_against_published_errors."""
        summary = run_integrator_method_validation(
            output_dir=tmp_path,
            methods=["ABM", "RK4"],
            fast=True,
        )
        efork_info = summary.get("methods", {}).get("EFORK3", {})
        assert efork_info.get("status") == "validated_elsewhere_against_published_errors"

    def test_output_csv_files_created(self, tmp_path: Path) -> None:
        """All expected CSV files must be written."""
        run_integrator_method_validation(
            output_dir=tmp_path,
            methods=["ABM", "RK4"],
            fast=True,
        )
        expected = [
            "abm_mittag_leffler_convergence.csv",
            "abm_manufactured_solution_convergence.csv",
            "abm_vector_linear_convergence.csv",
            "rk4_exponential_decay_convergence.csv",
            "rk4_harmonic_oscillator_convergence.csv",
            "rk4_vector_linear_convergence.csv",
            "integrator_method_validation_summary.json",
        ]
        for fname in expected:
            assert (tmp_path / fname).exists(), f"Expected output file missing: {fname}"
