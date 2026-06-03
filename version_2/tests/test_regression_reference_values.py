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


@pytest.mark.regression
def test_kuznetsov_three_cases_are_readable() -> None:
    data = _read("docs/published_validation_data_extraction_v1.json")
    article = data["articles"]["kuznetsov2017_chua_integer_df"]
    cases = {c["case_id"]: c for c in article["published_cases"]}
    assert "kuznetsov2017_case_18_hidden_chaotic" in cases
    assert "kuznetsov2017_case_21_hidden_chaotic_branch" in cases
    assert "kuznetsov2017_case_21_hidden_periodic_branch" in cases


@pytest.mark.regression
def test_wu_arctan_parameter_equivalence() -> None:
    # m = 0.4, n = -1.1585 in Wu paper
    # Code uses a1 = m = 0.4 and a2 = n - m = -1.1585 - 0.4 = -1.5585
    system = get_system("fractional_chua_arctan_wu2023")
    params = system.parameters
    assert params["a1"] == 0.4
    assert params["a2"] == pytest.approx(-1.5585, abs=1e-6)


@pytest.mark.regression
def test_dk2018_validation_levels() -> None:
    data = _read("docs/published_validation_data_extraction_v1.json")
    benchmarks = data["articles"]["danca_kuznetsov2018_lyapunov_fo"]["benchmarks"]
    levels = {b["case_id"]: b["validation_level"] for b in benchmarks}
    assert levels["DK2018_RF_q0999"] == "quantitative"
    assert levels["DK2018_Lorenz_q0985"] == "quantitative"
    assert levels["DK2018_4D_nonsmooth_q098"] == "qualitative_or_experimental_only"


@pytest.mark.regression
def test_fischer_cloned_dynamics_system_structures() -> None:
    data = _read("docs/published_validation_data_extraction_v1.json")
    fischer = data["articles"]["fischer2020_cloned_dynamics"]
    assert "jerk_system" in fischer
    assert "financial_system" in fischer
    assert "four_wing_system" in fischer
    assert len(fischer["jerk_system"]["table4_lce_0_1"]) > 0
    assert len(fischer["financial_system"]["table5_lce_0_1"]) > 0
    assert len(fischer["four_wing_system"]["table6_lce_0_1"]) > 0


