"""Fast contracts for reusable integer hidden-chaos building blocks."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from hidden_attractors.reproducibility import validate_hiddenness_promotion_metadata
from hidden_attractors.systems.modified_van_der_pol_duffing import (
    mavpd_2023_system,
    mavpd_hopf_gamma_boundaries,
)
from hidden_attractors.verification.attractor_reference import (
    calibrate_attractor_reference,
    classify_cloud_against_reference,
)
from hidden_attractors.verification.candidate_gate import evaluate_candidate_gate
from hidden_attractors.verification.jacobian import compute_jacobian
from hidden_attractors.verification.stability import classify_equilibrium_stability
from hidden_attractors.workflows.integer_hidden_chaos import (
    IntegerHiddenChaosProbe,
    continue_integer_parameter_path,
    deterministic_unit_directions,
    equilibrium_stability_records,
    summarize_integer_hidden_chaos_controls,
)


def _probe(equilibrium: str) -> IntegerHiddenChaosProbe:
    return IntegerHiddenChaosProbe(
        sample_id=0,
        equilibrium=equilibrium,
        radius=1.0e-5,
        direction_id=1,
        sampling_mode="sphere",
        x0=np.zeros(3),
        status="ok",
        destination=f"equilibrium_{equilibrium}",
        target_classification="different_from_target_under_calibrated_cloud_test",
        target_distance_norm=2.0,
        target_hit=False,
        ambiguous=False,
        tail_span=0.0,
        closest_equilibrium=equilibrium,
        closest_equilibrium_distance=0.0,
        solver_metadata={},
    )


@pytest.mark.unit
def test_hiddenness_summary_cannot_claim_omitted_declared_equilibria() -> None:
    summary = summarize_integer_hidden_chaos_controls(
        [_probe("E0")],
        required_equilibrium_names=("E0",),
        declared_equilibrium_names=("E0", "E+", "E-"),
    )

    assert summary["tested_all_required_equilibria"] is True
    assert summary["tested_all_declared_equilibria"] is False
    assert summary["declared_equilibria"] == ["E0", "E+", "E-"]


@pytest.mark.unit
def test_calibrated_reference_accepts_replica_and_rejects_separated_cloud() -> None:
    angle = np.linspace(0.0, 2.0 * np.pi, 600, endpoint=False)
    first = np.column_stack((np.cos(angle), np.sin(angle)))
    second = np.column_stack((np.cos(angle + 0.013), np.sin(angle + 0.013)))
    third = np.column_stack((np.cos(angle + 0.027), np.sin(angle + 0.027)))
    separated = first + np.array([5.0, -3.0])
    calibration = calibrate_attractor_reference(
        [first, second], negative_control_clouds=[separated], safety_factor=2.0
    )

    assert calibration.status == "calibrated"
    assert classify_cloud_against_reference(third, [first, second], calibration)[
        "classification"
    ] == "same_attractor_under_calibrated_cloud_test"
    assert classify_cloud_against_reference(separated, [first, second], calibration)[
        "classification"
    ] == "different_from_target_under_calibrated_cloud_test"


@pytest.mark.unit
def test_three_dimensional_direction_contract_matches_existing_axes_and_fibonacci() -> None:
    directions = deterministic_unit_directions(3, 12)

    assert directions.shape == (12, 3)
    assert np.allclose(directions[:6], np.vstack((np.eye(3), -np.eye(3))))
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert directions[6] == pytest.approx([0.5527707983925666, 0.0, 5.0 / 6.0])


@pytest.mark.integration
def test_parameter_path_rebuilds_mavpd_lure_at_every_node() -> None:
    calls: list[dict[str, float]] = []

    def factory(parameters):
        calls.append(dict(parameters))
        return mavpd_2023_system(parameters)

    path = [
        {"gamma": 0.1, "delta": 100.0, "rho": 200.0, "xi": 3.1},
        {"gamma": 0.1, "delta": 100.0, "rho": 200.0, "xi": 3.09},
    ]
    steps = continue_integer_parameter_path(
        factory,
        path,
        [0.1, 0.0, 0.1],
        t_burn=0.02,
        t_keep=0.02,
        sample_step=0.01,
        max_step=0.005,
    )

    assert calls == path
    assert len(steps) == 2
    assert np.array_equal(steps[0].x_out, steps[1].x_in)
    assert mavpd_2023_system(path[0]).lure.matrix[1, 1] == pytest.approx(-3.1)
    assert mavpd_2023_system(path[1]).lure.matrix[1, 1] == pytest.approx(-3.09)


@pytest.mark.unit
def test_mavpd_equilibrium_stability_changes_across_selected_hopf_boundary() -> None:
    unstable = equilibrium_stability_records(
        mavpd_2023_system({"xi": 2.85, "gamma": 0.14})
    )
    selected = equilibrium_stability_records(
        mavpd_2023_system({"xi": 2.85, "gamma": 0.1538037983994911})
    )
    unstable_by_name = {row["equilibrium"]: row for row in unstable}
    selected_by_name = {row["equilibrium"]: row for row in selected}

    assert unstable_by_name["E+"]["stability"] == "unstable"
    assert selected_by_name["E+"]["stability"] == "locally_asymptotically_stable"
    assert selected_by_name["E-"]["stability"] == "locally_asymptotically_stable"
    assert selected_by_name["E0"]["stability"] == "unstable"


@pytest.mark.unit
def test_generic_verification_uses_registered_mavpd_jacobian_and_derived_boundary() -> None:
    system = mavpd_2023_system({"xi": 2.85, "gamma": 0.1538037983994911})
    point = np.array([0.2, -0.1, 0.4])

    assert np.array_equal(compute_jacobian(system, point), system.jacobian_matrix(point))
    assert max(mavpd_hopf_gamma_boundaries({"xi": 2.85})) == pytest.approx(
        0.1438037983994911
    )
    stability = classify_equilibrium_stability(system, system.equilibrium_points()["E+"])
    assert stability["stable"] is True
    assert stability["stability_class"] == "stable"


@pytest.mark.unit
def test_integer_metadata_and_real_diagnostic_shapes_enable_joint_gate(valid_run_metadata) -> None:
    metadata = deepcopy(valid_run_metadata)
    metadata["numerical_contract"].update(
        {
            "q": 1.0,
            "memory": {
                "mode": "not_applicable",
                "M": None,
                "memory_window_steps": None,
                "memory_window_time": None,
                "is_full_caputo": False,
            },
            "integrator": {"name": "DOP853", "backend": "python", "caputo": False},
        }
    )
    metadata["continuation"].update(
        {"used": True, "eta_path": [0.0, 1.0], "final_eta": 1.0, "continuation_mode": "integer", "memory_window_propagated": None}
    )
    assert validate_hiddenness_promotion_metadata(metadata) == []
    evidence = {
        "run_metadata": metadata,
        "equilibria": {"all_found": True, "max_residual": 1.0e-12},
        "matignon": {"all_classified": True, "q": 1.0},
        "seed": {"localized": True, "method": "describing_function", "source": "direct_integer_transfer"},
        "continuation": metadata["continuation"],
        "trajectory": {"bounded": True, "nontrivial": True, "finite_fraction": 1.0, "post_transient_rows": 10000},
        "robustness": {"tested_h": True, "tested_memory": False, "tested_t_final": True, "tested_integrator": True, "consistent": True},
        "hiddenness": {"tested_all_equilibria": True, "tested_radii": [1e-5, 1e-3, 1e-2], "required_radii": [1e-5, 1e-3, 1e-2], "coverage_by_equilibrium_radius_complete": True, "target_hits": 0, "basin_intersection_detected": False, "basin_controls_complete": True, "numerical_failures": 0},
        "lyapunov": {"exponents": [0.69, 0.0, -20.9], "method_status": "internal_controls_passed"},
        "zero_one": {"K_global_median": 0.91, "state_global": "zero_one_chaotic_candidate"},
        "spectrum": {"state_global": "broadband_spectrum"},
        "poincare": {"interpretation_label": "cloud_like"},
    }
    gate = evaluate_candidate_gate(evidence)

    assert gate["hiddenness_promotion_allowed"] is True
    assert gate["chaotic_hidden_promotion_allowed"] is True
    assert gate["hidden_chaos_status"] == "chaotic_hidden_under_tested_neighborhoods"
    assert gate["checked_conditions"]["robustness_memory_requirement_satisfied"] is True
    assert "robustness_tested_memory" not in gate["checked_conditions"]


@pytest.mark.unit
def test_periodic_hidden_candidate_is_not_promoted_as_hidden_chaos(valid_run_metadata) -> None:
    evidence = {
        "run_metadata": valid_run_metadata,
        "equilibria": {"all_found": True, "max_residual": 1.0e-12},
        "matignon": {"all_classified": True, "q": 0.9998},
        "seed": {"localized": True, "method": "df_nyquist", "source": "test"},
        "continuation": {"used": False},
        "trajectory": {"bounded": True, "nontrivial": True, "finite_fraction": 1.0, "post_transient_length": 1000},
        "robustness": {"tested_h": True, "tested_memory": True, "tested_t_final": True, "tested_integrator": True, "consistent": True},
        "hiddenness": {"tested_all_equilibria": True, "tested_radii": [1e-3], "required_radii": [1e-3], "coverage_by_equilibrium_radius_complete": True, "target_hits": 0, "basin_intersection_detected": False, "basin_controls_complete": True, "numerical_failures": 0},
        "lyapunov": {"lambda_max": -0.01, "method_status": "internal_controls_passed"},
        "zero_one": {"K": 0.05},
        "spectrum": {"label": "dominant_periodic_peak"},
        "poincare": {"label": "finite_set_like"},
    }
    gate = evaluate_candidate_gate(evidence)

    assert gate["hiddenness_promotion_allowed"] is True
    assert gate["chaotic_hidden_promotion_allowed"] is False
    assert gate["hidden_chaos_status"] == "hiddenness_only_not_chaotic"
