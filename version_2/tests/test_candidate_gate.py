from __future__ import annotations

from copy import deepcopy

from hidden_attractors.verification.candidate_gate import (
    evaluate_candidate_gate,
    normalize_hiddenness_label,
)


def _complete_evidence(valid_run_metadata) -> dict:
    return {
        "run_metadata": valid_run_metadata,
        "equilibria": {"all_found": True, "max_residual": 1.0e-10},
        "matignon": {"all_classified": True, "q": 0.9998},
        "seed": {"localized": True, "method": "df_nyquist", "source": "published_reference"},
        "continuation": {"used": True, "eta_path": [0.0, 0.5, 1.0], "continuation_mode": "fractional", "memory_window_propagated": True, "final_eta": 1.0},
        "trajectory": {"bounded": True, "nontrivial": True, "finite_fraction": 1.0, "post_transient_length": 10_000},
        "robustness": {"tested_h": True, "tested_memory": True, "tested_t_final": True, "tested_integrator": True, "consistent": True},
        "hiddenness": {"tested_all_equilibria": True, "tested_radii": [1.0e-2, 1.0e-3], "required_radii": [1.0e-2, 1.0e-3], "target_hits_from_equilibria": 0, "basin_intersection_detected": False, "basin_controls_complete": True},
        "lyapunov": {"lambda_max": 0.15, "method_status": "internal_controls_passed"},
        "zero_one": {"K": 0.9},
        "spectrum": {"label": "broadband_spectrum"},
        "poincare": {"label": "complex_section"},
    }


def test_complete_zero_contact_evidence_promotes_sampled_hiddenness(valid_run_metadata) -> None:
    gate = evaluate_candidate_gate(_complete_evidence(valid_run_metadata))
    assert gate["verdict"] == "hiddenness_supported_under_tested_neighborhoods"
    assert gate["hiddenness_evidence_level"] == "hiddenness_supported_under_tested_neighborhoods"
    assert gate["evidence_level"] == "hiddenness_supported_under_tested_neighborhoods"
    assert gate["chaos_evidence_level"] == "strong_chaos_evidence"
    assert gate["promotion_allowed"] is True


def test_missing_robustness_is_compatible_only(valid_run_metadata) -> None:
    evidence = _complete_evidence(valid_run_metadata)
    evidence["robustness"]["tested_h"] = False
    assert evaluate_candidate_gate(evidence)["verdict"] == "compatible_with_hiddenness_under_tested_radii"


def test_target_hit_is_self_excited(valid_run_metadata) -> None:
    evidence = _complete_evidence(valid_run_metadata)
    evidence["hiddenness"]["target_hits_from_equilibria"] = 1
    assert evaluate_candidate_gate(evidence)["verdict"] == "self_excited_contact_detected"


def test_divergent_candidate_is_rejected(valid_run_metadata) -> None:
    evidence = _complete_evidence(valid_run_metadata)
    evidence["trajectory"]["bounded"] = False
    assert evaluate_candidate_gate(evidence)["verdict"] == "candidate_rejected"


def test_positive_lyapunov_high_zero_one_and_broadband_is_strong(valid_run_metadata) -> None:
    gate = evaluate_candidate_gate(_complete_evidence(valid_run_metadata))
    assert gate["chaos_evidence_level"] == "strong_chaos_evidence"


def test_positive_lyapunov_without_complements_is_supported(valid_run_metadata) -> None:
    evidence = _complete_evidence(valid_run_metadata)
    evidence["zero_one"] = {}
    evidence["spectrum"] = {}
    evidence["poincare"] = {}
    assert evaluate_candidate_gate(evidence)["chaos_evidence_level"] == "chaotic_dynamics_supported"


def test_nonpositive_lyapunov_and_periodic_psd_is_regular(valid_run_metadata) -> None:
    evidence = _complete_evidence(valid_run_metadata)
    evidence["lyapunov"]["lambda_max"] = -0.01
    evidence["zero_one"]["K"] = 0.1
    evidence["spectrum"]["label"] = "dominant_periodic_peak"
    assert evaluate_candidate_gate(evidence)["chaos_evidence_level"] == "regular_or_periodic_candidate"


def test_conflicting_diagnostics_are_inconclusive(valid_run_metadata) -> None:
    evidence = _complete_evidence(valid_run_metadata)
    evidence["zero_one"]["K"] = 0.1
    evidence["spectrum"]["label"] = "dominant_periodic_peak"
    assert evaluate_candidate_gate(evidence)["chaos_evidence_level"] == "chaos_evidence_inconclusive"


def test_incomplete_metadata_blocks_strong_hiddenness(valid_run_metadata) -> None:
    evidence = _complete_evidence(deepcopy(valid_run_metadata))
    evidence["run_metadata"]["software"]["git_commit"] = "unknown"
    gate = evaluate_candidate_gate(evidence)
    assert gate["verdict"] == "compatible_with_hiddenness_under_tested_radii"
    assert gate["promotion_allowed"] is False


def test_legacy_hiddenness_aliases() -> None:
    assert normalize_hiddenness_label("hidden_verified") == "hiddenness_supported_under_tested_neighborhoods"
    assert normalize_hiddenness_label("rejected_self_excited_contact") == "self_excited_contact_detected"
    assert normalize_hiddenness_label("compatible_in_all_tested_solver_memory_cases") == "compatible_with_hiddenness_under_tested_radii"
