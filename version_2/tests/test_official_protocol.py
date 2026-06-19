from __future__ import annotations

import sys
import json
from pathlib import Path
import numpy as np
import pytest

# Add workspace root and version_2 to sys.path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))
if str(workspace_root) not in sys.path:
    sys.path.insert(1, str(workspace_root))

from hidden_attractors.workflows.danca_abm_sphere_controls import (
    FINITE_MEMORY_POLICY,
    FULL_HISTORY_POLICY,
    _solver_cases,
    make_parser as make_danca_sphere_parser,
)
from hidden_attractors.workflows.fractional_report_run import (
    HIDDENNESS_RADII,
    HIDDENNESS_SAMPLE_GROWTH_PER_RADIUS,
    HIDDENNESS_SAMPLES_PER_RADIUS,
)
from hidden_attractors.workflows.protocol import (
    OFFICIAL_STAGE_ORDER,
    ContinuationPlan,
    HiddennessTestResult,
    NumericalContract,
    SoftPrecheckResult,
    StageEnvelope,
    UnifiedSeedRecord,
    sample_uniform_ball,
)
from hidden_attractors.workflows.sphere_controls import make_parser as make_sphere_parser


def _contract() -> NumericalContract:
    return NumericalContract(
        q=0.9998,
        h=0.01,
        t_final=100.0,
        t_transient=20.0,
        backend="efork_c",
        memory_policy="finite_memory",
        memory_length=10.0,
        hiddenness_radii=(1.0e-4, 1.0e-3),
        samples_per_radius=100,
        sample_growth_per_radius=50,
        random_seed=123,
    )


def test_official_stage_order_and_uniform_envelope(valid_run_metadata) -> None:
    expected = (
        "numerical_contract",
        "algebraic_validation",
        "seed_generation",
        "soft_precheck",
        "continuation",
        "post_continuation_filter",
        "dynamic_reference",
        "robustness",
        "hiddenness_tests",
        "diagnostics",
    )
    assert OFFICIAL_STAGE_ORDER == expected
    envelope = StageEnvelope(
        stage="seed_generation",
        status="ok",
        system="chua-nonsmooth",
        numerical_contract=_contract().to_dict(),
        candidate_id="seed-1",
        verdict="seed_only",
        run_metadata=valid_run_metadata,
        metadata_validation_errors=[],
    )
    record = envelope.to_dict()
    assert envelope.validate() == []
    assert tuple(record) == (
        "schema_version",
        "protocol_version",
        "stage",
        "status",
        "candidate_id",
        "system",
        "numerical_contract",
        "inputs",
        "outputs",
        "metrics",
        "verdict",
        "files",
        "provenance",
        "run_metadata",
        "metadata_validation_errors",
        "state",
        "state_history",
        "evidence",
        "failed_requirements",
        "method_scope",
        "warnings",
        "literature_note",
        "attractor_status",
    )


def test_seed_families_apply_the_unified_constraints() -> None:
    centered = UnifiedSeedRecord(
        family="lure_classical_centered",
        centered_or_biased="centered",
        A=1.0,
        sigma0=0.0,
        omega=2.0,
        mu=1.0,
        theta=0.0,
        q=0.9998,
        harmonic_residual=1.0e-8,
        rho_H=0.1,
        x0=(1.0, 0.0, -1.0),
    )
    invalid = UnifiedSeedRecord(
        family="lure_classical_centered",
        centered_or_biased="centered",
        A=1.0,
        sigma0=0.1,
        omega=2.0,
        mu=2.0,
        theta=0.0,
        q=0.9998,
        harmonic_residual=1.0e-8,
        rho_H=0.1,
        x0=(1.0, 0.0, -1.0),
    )
    assert centered.validate() == []
    assert invalid.validate()


def test_periodic_soft_precheck_remains_admissible_for_continuation() -> None:
    result = SoftPrecheckResult.periodic("periodic-seed")
    assert result.label == "pre_continuation_periodic"
    assert result.admissible_for_continuation is True
    assert result.validate() == []


@pytest.mark.parametrize(
    "result",
    [
        SoftPrecheckResult("nan", "rejected_numerical_failure", False, False, immediate_numerical_failure=True),
        SoftPrecheckResult("div", "rejected_catastrophic_divergence", False, True, catastrophic_divergence=True),
        SoftPrecheckResult("dup", "rejected_exact_duplicate", False, True, exact_duplicate=True),
        SoftPrecheckResult("config", "rejected_invalid_configuration", False, True),
        SoftPrecheckResult("amp", "rejected_invalid_amplitude_frequency", False, True),
    ],
)
def test_hard_soft_precheck_failures_are_rejectable(result: SoftPrecheckResult) -> None:
    assert result.admissible_for_continuation is False
    assert result.validate() == []


def test_continuation_plan_publishes_lambda_and_keeps_legacy_mapping_as_metadata() -> None:
    plan = ContinuationPlan((0.0, 0.25, 1.0), {"internal_parameter": "eta"})
    assert plan.validate() == []
    assert plan.lambda_values[-1] == 1.0
    assert plan.mapping["internal_parameter"] == "eta"


def test_hiddenness_sampling_is_interior_to_equilibrium_balls() -> None:
    center = np.array([1.0, -2.0, 3.0])
    points = sample_uniform_ball(center, 0.1, 128, np.random.default_rng(7))
    distances = np.linalg.norm(points - center, axis=1)
    assert np.all(distances <= 0.1)
    assert np.any(distances < 0.09)


def test_sphere_controls_reach_danca_delta_with_requested_growth() -> None:
    expected_radii = [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 1.0e-2]
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    configured = json.loads((configs_dir / "unified_caputo_protocol.json").read_text(encoding="utf-8"))["numerical_contract"]

    assert configured["hiddenness_radii"] == expected_radii
    assert configured["samples_per_radius"] == 100
    assert configured["sample_growth_per_radius"] == 50
    assert HIDDENNESS_RADII == expected_radii
    assert HIDDENNESS_SAMPLES_PER_RADIUS == 100
    assert HIDDENNESS_SAMPLE_GROWTH_PER_RADIUS == 50

    for args in (make_sphere_parser().parse_args([]), make_danca_sphere_parser().parse_args([])):
        radii = [float(radius) for radius in args.radii.split(",")]
        assert radii == expected_radii
        assert args.samples_per_radius == 100
        assert args.sample_growth_per_radius == 50
        assert args.samples_per_radius + (len(radii) - 1) * args.sample_growth_per_radius == 350


def test_danca_control_matrix_declares_abm_and_efork_with_both_memory_policies() -> None:
    args = make_danca_sphere_parser().parse_args([])
    cases = _solver_cases(args)
    assert args.h == 0.01
    assert args.memory_length == 40.0
    assert [(case["solver"], case["history_policy"]) for case in cases] == [
        ("abm", FULL_HISTORY_POLICY),
        ("abm", FINITE_MEMORY_POLICY),
        ("efork3", FULL_HISTORY_POLICY),
        ("efork3", FINITE_MEMORY_POLICY),
    ]
    assert cases[0]["solver_case_id"] == "abm_full_history"
    assert cases[0]["reference_role"] == "single_seed_accreditation_and_comparison"
    assert all(case["memory_length"] == 40.0 for case in cases if case["history_policy"] == FINITE_MEMORY_POLICY)


def test_strong_hiddenness_label_requires_the_full_protocol(valid_run_metadata) -> None:
    incomplete = HiddennessTestResult(
        candidate_id="c1",
        tested_equilibria=("E0", "E+", "E-"),
        tested_radii=(1.0e-4,),
        neighborhood_sampling_mode="ball",
        target_contacts=0,
        numerical_failures=0,
        basin_planes=("xy_close",),
        reference_was_robust=True,
        final_label="hidden_verified_only_if_full_protocol_passed",
        run_metadata=valid_run_metadata,
        required_equilibria=("E0", "E+", "E-"),
        required_radii=(1.0e-4,),
    )
    complete = HiddennessTestResult(
        candidate_id="c1",
        tested_equilibria=("E0", "E+", "E-"),
        tested_radii=(1.0e-4,),
        neighborhood_sampling_mode="ball",
        target_contacts=0,
        numerical_failures=0,
        basin_planes=("xy_close", "xy_large", "xz_close", "xz_large", "yz_close", "yz_large"),
        reference_was_robust=True,
        final_label="hidden_verified_only_if_full_protocol_passed",
        run_metadata=valid_run_metadata,
        required_equilibria=("E0", "E+", "E-"),
        required_radii=(1.0e-4,),
    )
    assert incomplete.validate()
    assert complete.validate() == []


def test_strong_hiddenness_label_requires_every_declared_radius(valid_run_metadata) -> None:
    result = HiddennessTestResult(
        candidate_id="c1",
        tested_equilibria=("E0",),
        tested_radii=(1.0e-4,),
        neighborhood_sampling_mode="ball",
        target_contacts=0,
        numerical_failures=0,
        basin_planes=("xy_close", "xy_large", "xz_close", "xz_large", "yz_close", "yz_large"),
        reference_was_robust=True,
        final_label="hidden_verified",
        run_metadata=valid_run_metadata,
        required_equilibria=("E0",),
        required_radii=(1.0e-4, 1.0e-3),
    )
    assert result.promotion_verdict == "compatible_with_hiddenness"


def test_validation_contract_uses_only_the_official_stage_order() -> None:
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    contract = json.loads((configs_dir / "validation_contract.json").read_text(encoding="utf-8"))
    stages = tuple(stage["slug"] for stage in contract["stages"])
    assert stages == OFFICIAL_STAGE_ORDER
    assert set(contract["summary_required_fields"]) == {
        "schema_version",
        "protocol_version",
        "stage",
        "status",
        "system",
        "numerical_contract",
        "inputs",
        "outputs",
        "metrics",
        "verdict",
        "files",
        "provenance",
        "run_metadata",
        "metadata_validation_errors",
    }


def test_current_and_legacy_final_labels_are_separated() -> None:
    from hidden_attractors.workflows.protocol import CURRENT_FINAL_LABELS, LEGACY_FINAL_LABELS, FINAL_LABELS
    assert "hidden_verified" not in CURRENT_FINAL_LABELS
    assert "hidden_under_tested_neighborhoods" in CURRENT_FINAL_LABELS
    assert len(LEGACY_FINAL_LABELS) == 0
    assert len(FINAL_LABELS) == len(CURRENT_FINAL_LABELS)


def test_hiddenness_test_result_evidence_cases(valid_run_metadata) -> None:
    from copy import deepcopy
    res_complete = HiddennessTestResult(
        candidate_id="c_complete",
        tested_equilibria=("E0", "E+", "E-"),
        tested_radii=(1.0e-2, 1.0e-3),
        neighborhood_sampling_mode="ball",
        target_contacts=0,
        numerical_failures=0,
        basin_planes=("xy_close", "xy_large", "xz_close", "xz_large", "yz_close", "yz_large"),
        reference_was_robust=True,
        final_label="hidden_under_tested_neighborhoods",
        candidate_evidence={
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
        },
    )
    assert res_complete.promotion_verdict == "hidden_under_tested_neighborhoods"
    assert "candidate_evidence_missing_full_algebraic_payload" not in res_complete.promotion_gate.get("warnings", [])

    bad_metadata = deepcopy(valid_run_metadata)
    bad_metadata["software"]["git_commit"] = "unknown"
    res_fallback = HiddennessTestResult(
        candidate_id="c_fallback",
        tested_equilibria=("E0", "E+", "E-"),
        tested_radii=(1.0e-2, 1.0e-3),
        neighborhood_sampling_mode="ball",
        target_contacts=0,
        numerical_failures=0,
        basin_planes=("xy_close", "xy_large", "xz_close", "xz_large", "yz_close", "yz_large"),
        reference_was_robust=True,
        final_label="hidden_under_tested_neighborhoods",
        run_metadata=bad_metadata,
        required_equilibria=("E0", "E+", "E-"),
        required_radii=(1.0e-2, 1.0e-3),
    )
    assert res_fallback.promotion_verdict == "compatible_with_hiddenness"
    assert "candidate_evidence_missing_full_algebraic_payload" in res_fallback.promotion_gate["warnings"]

    res_fallback_none = HiddennessTestResult(
        candidate_id="c_fallback_none",
        tested_equilibria=("E0", "E+", "E-"),
        tested_radii=(1.0e-2, 1.0e-3),
        neighborhood_sampling_mode="ball",
        target_contacts=0,
        numerical_failures=0,
        basin_planes=("xy_close", "xy_large", "xz_close", "xz_large", "yz_close", "yz_large"),
        reference_was_robust=True,
        final_label="hidden_under_tested_neighborhoods",
        run_metadata=None,
        required_equilibria=("E0", "E+", "E-"),
        required_radii=(1.0e-2, 1.0e-3),
    )
    assert res_fallback_none.promotion_verdict == "inconclusive"
    assert "candidate_evidence_missing_full_algebraic_payload" in res_fallback_none.promotion_gate["warnings"]

    res_contacts = HiddennessTestResult(
        candidate_id="c_contacts",
        tested_equilibria=("E0", "E+", "E-"),
        tested_radii=(1.0e-2, 1.0e-3),
        neighborhood_sampling_mode="ball",
        target_contacts=5,
        numerical_failures=0,
        basin_planes=("xy_close", "xy_large", "xz_close", "xz_large", "yz_close", "yz_large"),
        reference_was_robust=True,
        final_label="hidden_under_tested_neighborhoods",
        run_metadata=valid_run_metadata,
        required_equilibria=("E0", "E+", "E-"),
        required_radii=(1.0e-2, 1.0e-3),
    )
    assert res_contacts.promotion_verdict == "self_excited"
