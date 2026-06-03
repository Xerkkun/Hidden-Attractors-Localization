from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hidden_attractors.seed_generation.lure import find_lure_harmonic_seed
from hidden_attractors.systems import get_system


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


@pytest.mark.regression
def test_kuznetsov_seed_regression_against_stored_reference() -> None:
    expected = _read("validation/references/kuznetsov2017_expected.json")
    seed = find_lure_harmonic_seed(q=1.0, system=get_system("chua-nonsmooth").lure, nscan=10_000)
    assert abs(seed.omega - expected["omega0"]) < 1.0e-4
    assert abs(seed.gain - expected["k"]) < 1.0e-4
    assert abs(seed.amplitude - expected["a0"]) < 1.0e-3
    assert np.linalg.norm(seed.seed - np.asarray(expected["seed_plus"])) < 1.0e-3


@pytest.mark.regression
def test_integer_reference_matignon_eigenvalues_remain_stored() -> None:
    seed = _read("validation/reference_cases/chua_integer_q1/02_lure_df/chua_integer_seed_summary.json")
    eigenvalues = np.asarray(seed["eigP0"], dtype=float)
    assert eigenvalues.shape == (3, 2)
    assert eigenvalues[0, 0] == pytest.approx(-1.5385101632504523, abs=1.0e-10)
    assert abs(eigenvalues[1, 1]) == pytest.approx(seed["chosen_branch"]["omega0"], abs=1.0e-10)


@pytest.mark.regression
def test_fast_integrator_reference_states_remain_documented() -> None:
    summary = _read("validation/outputs/integrator_method_validation/integrator_method_validation_summary.json")
    assert summary["methods"]["ABM"]["status"] == "method_validated_against_exact_solution"
    assert summary["methods"]["EFORK3"]["status"] == "validated_elsewhere_against_published_errors"
    assert summary["methods"]["RK4"]["status"] == "method_validated_against_exact_and_solve_ivp"


@pytest.mark.regression
def test_phase_f_reference_states_remain_traceable() -> None:
    f4 = _read("validation/chaos_validation/lyapunov_methods/F4_internal_validation/f4_internal_validation_summary.json")
    f5 = _read("validation/chaos_validation/dynamics_diagnostics/f5_diagnostics_summary.json")
    f6 = _read("validation/chaos_validation/integrated_chaos_validator/integrated_chaos_summary.json")
    f7 = _read("validation/chaos_validation/method_comparison/method_comparison_summary.json")
    phase_f = _read("validation/chaos_validation/phase_F_closure/phase_F_closure_summary.json")
    assert f4["status"] == "f4_complete_with_documented_discrepancies"
    assert f5["final_f5_status"] == "f5_diagnostics_structured_outputs_ready"
    assert f6["status"] == "completed_finite_time_chaos_evidence_integration"
    assert f7["status"] == "completed_method_evidence_comparison"
    assert phase_f["status"] == "phase_F_frozen"
    assert phase_f["phase_F_frozen"] is True

