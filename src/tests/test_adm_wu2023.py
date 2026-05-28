"""Unit tests for adm_wu2023 integrator and simulate_attractor_only workflow.

Test A: RHS sign/value consistency between adm_wu2023.rhs_chua_arctan and
        ChuaArctanSystem.evaluate_rhs.

Test B: Symmetry — if X0 generates trajectory X(t), then -X0 generates
        approximately -X(t) (up to numerical noise).

Test C: Run integrator with X0_plus, q=0.99, h=0.01, N=1000 → finite output
        and CSV files produced.

Test D: simulate_attractor_only workflow does NOT call seed generation,
        describing function, or continuation.
"""

from __future__ import annotations

import math
import os
import json
import tempfile
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Official Wu2023 parameters
# ---------------------------------------------------------------------------

WU2023_PARAMS: Dict[str, float] = {
    "alpha": 8.4562,
    "beta":  12.0732,
    "gamma": 0.0052,
    "m":     0.4,
    "n":    -1.1585,
}
WU2023_Q   = 0.99
WU2023_H   = 0.01
WU2023_N   = 1000   # short for tests

X0_PLUS  = np.array([13.8,   0.7093, -19.8768])
X0_MINUS = np.array([-13.8, -0.7093,  19.8768])
X0_FIG   = np.array([13.0,   0.7,    -19.0])


# ---------------------------------------------------------------------------
# Test A — RHS consistency
# ---------------------------------------------------------------------------

class TestRHSConsistency:
    """Verify adm_wu2023.rhs_chua_arctan agrees with ChuaArctanSystem.evaluate_rhs."""

    def _adm_rhs(self, x):
        from src.integrators.adm_wu2023 import rhs_chua_arctan
        return rhs_chua_arctan(x, WU2023_PARAMS)

    def _system_rhs(self, x):
        from src.systems.chua_arctan import ChuaArctanSystem
        sys = ChuaArctanSystem(
            alpha=WU2023_PARAMS["alpha"],
            beta=WU2023_PARAMS["beta"],
            gamma=WU2023_PARAMS["gamma"],
            m=WU2023_PARAMS["m"],
            n=WU2023_PARAMS["n"],
            q=WU2023_Q,
        )
        return sys.evaluate_rhs(x)

    @pytest.mark.parametrize("x", [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.5, -1.0]),
        X0_PLUS,
        X0_MINUS,
        np.array([0.5, 0.001, -0.5]),   # near equilibrium E+
    ])
    def test_rhs_matches_system(self, x):
        """F1 from adm_wu2023 must match ChuaArctanSystem.evaluate_rhs at same point."""
        f_adm = self._adm_rhs(x)
        f_sys = self._system_rhs(x)
        np.testing.assert_allclose(
            f_adm, f_sys, rtol=1e-12, atol=1e-12,
            err_msg=f"RHS mismatch at x={x}: adm={f_adm}, sys={f_sys}"
        )

    def test_f1_formula(self):
        """Verify F1 = -alpha*(1+m)*x + alpha*y - alpha*(n-m)*arctan(x)."""
        from src.integrators.adm_wu2023 import rhs_chua_arctan
        x = np.array([2.0, 1.0, -3.0])
        p = WU2023_PARAMS
        alpha, m, n = p["alpha"], p["m"], p["n"]
        expected_f1 = (
            -alpha * (1.0 + m) * x[0]
            + alpha * x[1]
            - alpha * (n - m) * math.atan(x[0])
        )
        f = rhs_chua_arctan(x, p)
        assert abs(f[0] - expected_f1) < 1e-12, f"F1 mismatch: got {f[0]}, expected {expected_f1}"

    def test_f2_formula(self):
        """Verify F2 = x - y + z."""
        from src.integrators.adm_wu2023 import rhs_chua_arctan
        x = np.array([3.0, -1.5, 2.0])
        f = rhs_chua_arctan(x, WU2023_PARAMS)
        expected_f2 = x[0] - x[1] + x[2]
        assert abs(f[1] - expected_f2) < 1e-12

    def test_f3_formula(self):
        """Verify F3 = -beta*y - gamma*z."""
        from src.integrators.adm_wu2023 import rhs_chua_arctan
        x = np.array([-1.0, 2.0, 5.0])
        p = WU2023_PARAMS
        f = rhs_chua_arctan(x, p)
        expected_f3 = -p["beta"] * x[1] - p["gamma"] * x[2]
        assert abs(f[2] - expected_f3) < 1e-12

    def test_equilibrium_residual_E0(self):
        """RHS at E0=(0,0,0) must be (0,0,0)."""
        from src.integrators.adm_wu2023 import rhs_chua_arctan
        f = rhs_chua_arctan(np.zeros(3), WU2023_PARAMS)
        np.testing.assert_allclose(f, 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Test B — Symmetry
# ---------------------------------------------------------------------------

class TestSymmetry:
    """If the system is odd-symmetric, X0 → X(t) implies -X0 → -X(t)."""

    def test_approximate_symmetry(self):
        """Trajectories from X0_plus and X0_minus should be approximately antisymmetric."""
        from src.integrators.adm_wu2023 import adm_wu2023_integrate

        N_test = 200   # short integration for speed

        t_pos, x_pos, status_pos, _ = adm_wu2023_integrate(
            params=WU2023_PARAMS, x0=X0_PLUS,
            q=WU2023_Q, h=WU2023_H, N=N_test, divergence_norm=200.0
        )
        t_neg, x_neg, status_neg, _ = adm_wu2023_integrate(
            params=WU2023_PARAMS, x0=X0_MINUS,
            q=WU2023_Q, h=WU2023_H, N=N_test, divergence_norm=200.0
        )

        assert status_pos in ("ok", "diverged"), f"Unexpected status from X0_plus: {status_pos}"
        assert status_neg in ("ok", "diverged"), f"Unexpected status from X0_minus: {status_neg}"

        # If both have the same length, check antisymmetry at early steps
        n_common = min(len(t_pos), len(t_neg), 50)
        if n_common > 2:
            diff = x_pos[:n_common] + x_neg[:n_common]   # should be ~0 for odd-symmetric system
            max_err = float(np.abs(diff).max())
            # Allow generous tolerance for 4th-order ADM accumulation
            assert max_err < 1e-8, (
                f"Antisymmetry violated: max |X_+(t) + X_-(t)| = {max_err:.2e}"
            )

    def test_exact_antisymmetry_step1(self):
        """After exactly one ADM step, -X0 must give exactly -(X0_result)."""
        from src.integrators.adm_wu2023 import adm_wu2023_integrate

        t_p, x_p, _, _ = adm_wu2023_integrate(
            params=WU2023_PARAMS, x0=X0_PLUS,
            q=WU2023_Q, h=WU2023_H, N=1, divergence_norm=500.0
        )
        t_m, x_m, _, _ = adm_wu2023_integrate(
            params=WU2023_PARAMS, x0=-X0_PLUS,
            q=WU2023_Q, h=WU2023_H, N=1, divergence_norm=500.0
        )

        # After 1 step: x_p[1] and x_m[1] must be exact negatives
        np.testing.assert_allclose(
            x_p[1], -x_m[1], rtol=1e-12, atol=1e-12,
            err_msg="Single-step antisymmetry failed"
        )


# ---------------------------------------------------------------------------
# Test C — Execution and CSV output
# ---------------------------------------------------------------------------

class TestExecution:
    """Integration runs, produces finite results, and writes CSV files."""

    def test_finite_output_x0_plus(self):
        """With X0_plus, N=1000, output must be finite (no NaN/Inf)."""
        from src.integrators.adm_wu2023 import adm_wu2023_integrate

        times, states, status, info = adm_wu2023_integrate(
            params=WU2023_PARAMS, x0=X0_PLUS,
            q=WU2023_Q, h=WU2023_H, N=WU2023_N, divergence_norm=120.0
        )

        assert len(times) > 1, "No steps completed"
        assert np.all(np.isfinite(states)), "Non-finite values in states"
        assert times[0] == 0.0
        assert info["integrator"] == "adm_wu2023"
        assert info["integrator_class"] == "adm_local_reproduction"
        assert info["hidden_verified"] is False

    def test_output_shape(self):
        """Output shapes must be (N+1,) for times and (N+1, 3) for states."""
        from src.integrators.adm_wu2023 import adm_wu2023_integrate

        N = 100
        times, states, status, _ = adm_wu2023_integrate(
            params=WU2023_PARAMS, x0=X0_PLUS,
            q=WU2023_Q, h=WU2023_H, N=N, divergence_norm=120.0
        )
        assert times.ndim == 1
        assert states.ndim == 2
        assert states.shape[1] == 3

    def test_csv_files_created(self, tmp_path):
        """simulate_attractor_only workflow creates CSV files for each IC."""
        from src.cli.simulate_attractor import run_simulate_attractor_workflow

        cfg = {
            "system_id": "chua_fractional_arctan",
            "workflow_mode": "simulate_attractor_only",
            "integrator": "adm_wu2023",
            "alpha": 8.4562, "beta": 12.0732, "gamma": 0.0052,
            "m": 0.4, "n": -1.1585,
            "q": 0.99, "h": 0.01, "N": 100,
            "t_burn": 0.1,
            "divergence_norm": 120.0,
            "output_dir": str(tmp_path),
            "save_timeseries": True,
            "save_attractor": True,
            "plot_3d": False,
            "plot_projections": False,
            "diagnostics": True,
            "initial_conditions": {
                "x0_plus":  [13.8,  0.7093, -19.8768],
                "x0_minus": [-13.8, -0.7093, 19.8768],
            },
            "scientific_label": "Test run",
        }

        summary = run_simulate_attractor_workflow(cfg)

        # Check summary.json
        summary_path = os.path.join(str(tmp_path), "summary.json")
        assert os.path.exists(summary_path), "summary.json not created"
        with open(summary_path) as f:
            data = json.load(f)
        assert data["hidden_verified"] is False
        assert data["integrator_class"] == "adm_local_reproduction"

        # Check CSV files
        for ic_label in ["x0_plus", "x0_minus"]:
            ts_csv = os.path.join(str(tmp_path), f"{ic_label}_timeseries.csv")
            assert os.path.exists(ts_csv), f"Missing {ic_label}_timeseries.csv"

    def test_divergence_detection(self):
        """If divergence_norm is very small, status should be 'diverged'."""
        from src.integrators.adm_wu2023 import adm_wu2023_integrate

        # Set very tight divergence threshold so first step triggers it
        _, _, status, _ = adm_wu2023_integrate(
            params=WU2023_PARAMS, x0=X0_PLUS,
            q=WU2023_Q, h=WU2023_H, N=WU2023_N, divergence_norm=0.001
        )
        assert status == "diverged", f"Expected 'diverged', got '{status}'"

    def test_nonfinite_detection(self):
        """If step size is enormous, the integrator should detect NaN/Inf."""
        from src.integrators.adm_wu2023 import adm_wu2023_integrate

        # h = 1000 will cause polynomial overflow in arctan expansion
        _, _, status, _ = adm_wu2023_integrate(
            params=WU2023_PARAMS, x0=X0_PLUS,
            q=WU2023_Q, h=1000.0, N=10, divergence_norm=1e30
        )
        assert status in ("nonfinite", "diverged"), (
            f"Expected 'nonfinite' or 'diverged' with h=1000, got '{status}'"
        )


# ---------------------------------------------------------------------------
# Test D — Mode isolation (no seed, DF, continuation calls)
# ---------------------------------------------------------------------------

class TestModeIsolation:
    """simulate_attractor_only must not invoke Ruta A modules."""

    FORBIDDEN_MODULES = [
        "src.lure.nyquist",
        "src.lure.seeds",
        "src.continuation.continuation_fractional",
        "src.continuation.continuation_integer",
        "version_2.hidden_attractors.seed_generation",
    ]

    def test_no_seed_generation_import(self, tmp_path):
        """Running simulate_attractor_only should not import seed-generation modules."""
        import importlib, sys

        # Record modules before
        before = set(sys.modules.keys())

        from src.cli.simulate_attractor import run_simulate_attractor_workflow

        cfg = {
            "system_id": "chua_fractional_arctan",
            "workflow_mode": "simulate_attractor_only",
            "integrator": "adm_wu2023",
            "alpha": 8.4562, "beta": 12.0732, "gamma": 0.0052,
            "m": 0.4, "n": -1.1585,
            "q": 0.99, "h": 0.01, "N": 50,
            "t_burn": 0.1, "divergence_norm": 120.0,
            "output_dir": str(tmp_path),
            "save_timeseries": False, "save_attractor": False,
            "plot_3d": False, "plot_projections": False, "diagnostics": False,
            "initial_conditions": {"x0_test": [1.0, 0.0, 0.0]},
            "scientific_label": "isolation test",
        }

        run_simulate_attractor_workflow(cfg)

        after = set(sys.modules.keys())
        newly_loaded = after - before

        for forbidden in self.FORBIDDEN_MODULES:
            loaded = any(forbidden.replace(".", "/") in m or forbidden in m
                         for m in newly_loaded)
            assert not loaded, (
                f"Forbidden module '{forbidden}' was loaded during "
                f"simulate_attractor_only workflow."
            )

    def test_hidden_verified_always_false(self, tmp_path):
        """summary.json must always have hidden_verified=false."""
        from src.cli.simulate_attractor import run_simulate_attractor_workflow

        cfg = {
            "system_id": "chua_fractional_arctan",
            "workflow_mode": "simulate_attractor_only",
            "integrator": "adm_wu2023",
            "alpha": 8.4562, "beta": 12.0732, "gamma": 0.0052,
            "m": 0.4, "n": -1.1585,
            "q": 0.99, "h": 0.01, "N": 50,
            "t_burn": 0.1, "divergence_norm": 120.0,
            "output_dir": str(tmp_path),
            "save_timeseries": False, "save_attractor": False,
            "plot_3d": False, "plot_projections": False, "diagnostics": False,
            "initial_conditions": {"x0_plus": [13.8, 0.7093, -19.8768]},
            "scientific_label": "hidden_verified test",
        }

        summary = run_simulate_attractor_workflow(cfg)
        assert summary["hidden_verified"] is False
        for r in summary["results"]:
            assert r.get("hidden_verified", False) is False

    def test_integrator_class_label_adm(self, tmp_path):
        """ADM integrator must be labeled 'adm_local_reproduction'."""
        from src.cli.simulate_attractor import run_simulate_attractor_workflow

        cfg = {
            "system_id": "chua_fractional_arctan",
            "workflow_mode": "simulate_attractor_only",
            "integrator": "adm_wu2023",
            "alpha": 8.4562, "beta": 12.0732, "gamma": 0.0052,
            "m": 0.4, "n": -1.1585,
            "q": 0.99, "h": 0.01, "N": 50,
            "t_burn": 0.1, "divergence_norm": 120.0,
            "output_dir": str(tmp_path),
            "save_timeseries": False, "save_attractor": False,
            "plot_3d": False, "plot_projections": False, "diagnostics": False,
            "initial_conditions": {"x0_plus": [13.8, 0.7093, -19.8768]},
            "scientific_label": "label test",
        }

        summary = run_simulate_attractor_workflow(cfg)
        assert summary["integrator_class"] == "adm_local_reproduction"

    def test_integrator_class_label_abm(self, tmp_path):
        """ABM integrator must be labeled 'caputo_full_memory'."""
        from src.cli.simulate_attractor import run_simulate_attractor_workflow

        cfg = {
            "system_id": "chua_fractional_arctan",
            "workflow_mode": "simulate_attractor_only",
            "integrator": "abm",
            "alpha": 8.4562, "beta": 12.0732, "gamma": 0.0052,
            "m": 0.4, "n": -1.1585,
            "q": 0.99, "h": 0.01, "N": 50,
            "t_burn": 0.1, "divergence_norm": 120.0,
            "output_dir": str(tmp_path),
            "save_timeseries": False, "save_attractor": False,
            "plot_3d": False, "plot_projections": False, "diagnostics": False,
            "initial_conditions": {"x0_plus": [13.8, 0.7093, -19.8768]},
            "scientific_label": "abm label test",
        }

        summary = run_simulate_attractor_workflow(cfg)
        assert summary["integrator_class"] == "caputo_full_memory"


# ---------------------------------------------------------------------------
# Test E — Runge-Kutta 4th order (RK4)
# ---------------------------------------------------------------------------

class TestRK4:
    """Explicit verification of RK4 mathematical properties (scalar, RHS, symmetry)."""

    def test_linear_scalar_system(self):
        """Prueba A: Integra dx/dt = -x, x(0)=1, con h=1e-3, t_final=1, y compara con exp(-t).
        
        El error maximo esperado debe ser estrictamente menor que 1e-5.
        """
        from src.integrators.rk4 import rk4_integrate

        def scalar_rhs(t, x):
            return -1.0 * x

        x0 = np.array([1.0])
        h = 0.001
        N = 1000  # t_final = 1.0

        times, states, status, _ = rk4_integrate(
            rhs=scalar_rhs, x0=x0, h=h, N=N, divergence_norm=50.0
        )

        assert status == "ok"
        assert len(times) == 1001

        # Compare with exact analytical solution x(t) = exp(-t)
        exact = np.exp(-times)
        errors = np.abs(states[:, 0] - exact)
        max_error = float(np.max(errors))

        assert max_error < 1e-5, f"Prueba A fallida: error maximo = {max_error:.2e} (limite 1e-5)"
        print(f"\n[TestRK4] Prueba A - OK: error maximo = {max_error:.2e} (< 1e-5)")

    def test_rhs_consistency_formulas(self):
        """Prueba B: Verifica que el RHS de Chua arctan use exactamente las tres formulas expandidas.
        
        F1 = -alpha*(1+m)*x + alpha*y - alpha*(n-m)*atan(x)
        F2 = x - y + z
        F3 = -beta*y - gamma*z
        """
        from src.integrators.adm_wu2023 import rhs_chua_arctan

        # Select a non-trivial test state
        x = np.array([3.5, -2.1, 7.8])
        
        alpha = WU2023_PARAMS["alpha"]
        beta  = WU2023_PARAMS["beta"]
        gamma = WU2023_PARAMS["gamma"]
        m     = WU2023_PARAMS["m"]
        n     = WU2023_PARAMS["n"]
        nm    = n - m

        # Exact algebraic implementation of the equations
        F1_expected = -alpha * (1.0 + m) * x[0] + alpha * x[1] - alpha * nm * math.atan(x[0])
        F2_expected = x[0] - x[1] + x[2]
        F3_expected = -beta * x[1] - gamma * x[2]

        F_actual = rhs_chua_arctan(x, WU2023_PARAMS)

        assert abs(F_actual[0] - F1_expected) < 1e-12, "RHS F1 mismatch"
        assert abs(F_actual[1] - F2_expected) < 1e-12, "RHS F2 mismatch"
        assert abs(F_actual[2] - F3_expected) < 1e-12, "RHS F3 mismatch"
        print("\n[TestRK4] Prueba B - OK: el RHS evalua exactamente las formulas del atractor de Chua.")

    def test_rk4_symmetry(self):
        """Prueba C: Verifica que si X0 produce X(t), -X0 produce aproximadamente -X(t)."""
        from src.integrators.rk4 import rk4_integrate
        from src.integrators.adm_wu2023 import rhs_chua_arctan

        N_steps = 500
        h_val = 0.002

        # Integrate from positive initial condition X0
        t_pos, states_pos, status_pos, _ = rk4_integrate(
            rhs=lambda t, x: rhs_chua_arctan(x, WU2023_PARAMS),
            x0=X0_PLUS, h=h_val, N=N_steps, divergence_norm=150.0
        )
        
        # Integrate from negative initial condition -X0
        t_neg, states_neg, status_neg, _ = rk4_integrate(
            rhs=lambda t, x: rhs_chua_arctan(x, WU2023_PARAMS),
            x0=-X0_PLUS, h=h_val, N=N_steps, divergence_norm=150.0
        )

        assert status_pos == "ok"
        assert status_neg == "ok"
        assert len(states_pos) == len(states_neg)

        # The sum of states should be zero for an exact odd-symmetric vector field solver
        diff = states_pos + states_neg
        max_symmetry_error = float(np.max(np.abs(diff)))

        assert max_symmetry_error < 1e-10, f"Prueba C fallida: max error simetria = {max_symmetry_error:.2e}"
        print(f"\n[TestRK4] Prueba C - OK: error de simetria maximo = {max_symmetry_error:.2e} (< 1e-10)")

    def test_rk4_divergence_detection(self):
        """RK4 must stop and return 'diverged' if state exceeds divergence_norm."""
        from src.integrators.rk4 import rk4_integrate
        from src.integrators.adm_wu2023 import rhs_chua_arctan

        times, states, status, _ = rk4_integrate(
            rhs=lambda t, x: rhs_chua_arctan(x, WU2023_PARAMS),
            x0=X0_PLUS, h=0.01, N=10, divergence_norm=1.0
        )
        assert status == "diverged"

    def test_simulate_attractor_only_rk4(self, tmp_path):
        """simulate_attractor_only workflow runs with rk4 and q=1.0."""
        from src.cli.simulate_attractor import run_simulate_attractor_workflow

        cfg = {
            "system_id": "chua_fractional_arctan",
            "workflow_mode": "simulate_attractor_only",
            "integrator": "rk4",
            "alpha": 8.4562, "beta": 12.0732, "gamma": 0.0052,
            "m": 0.4, "n": -1.1585,
            "q": 1.0, "h": 0.001, "N": 100,
            "t_burn": 0.01,
            "divergence_norm": 120.0,
            "output_dir": str(tmp_path),
            "save_timeseries": True,
            "save_attractor": False,
            "plot_3d": False,
            "plot_projections": False,
            "diagnostics": False,
            "initial_conditions": {
                "x0_rk4": [13.8, 0.7093, -19.8768],
            },
            "scientific_label": "RK4 Integer Test",
        }

        summary = run_simulate_attractor_workflow(cfg)
        assert summary["integrator_class"] == "integer_order_solver"
        assert summary["results"][0]["status"] == "ok"


