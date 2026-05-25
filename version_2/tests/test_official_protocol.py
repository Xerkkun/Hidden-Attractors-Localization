"""Contract tests for the single maintained Caputo methodology."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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


def test_official_stage_order_and_uniform_envelope() -> None:
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


def test_strong_hiddenness_label_requires_the_full_protocol() -> None:
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
    )
    assert incomplete.validate()
    assert complete.validate() == []


def test_validation_contract_uses_only_the_official_stage_order() -> None:
    contract = json.loads(Path("configs/validation_contract.json").read_text(encoding="utf-8"))
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
    }
