from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from hidden_attractors.seed_bank import SeedRecord, build_seed_bank
from hidden_attractors.systems.base import ChaoticSystem
from hidden_attractors.verification.destination_classifier import (
    DESTINATION_CLASSIFIER_ID,
    DESTINATION_CLASSIFIER_SCHEMA_VERSION,
    DestinationClassification,
    DestinationClassifierContract,
)
from hidden_attractors.workflows.geometric_topological_campaign import (
    CampaignManifest,
    initialize_campaign_artifacts,
)
from hidden_attractors.workflows.geometric_topological_pilot import (
    CaseDefinition,
    CaseRunContract,
    PhysicalBudget,
    SolverSpec,
    TrajectoryResult,
    append_case_seed_bank,
    append_trajectory_result,
    compute_central_symmetry_diagnostics,
    evaluate_case_evidence,
    run_and_classify_seed,
)


def _linear_case(tmp_path: Path) -> CaseDefinition:
    system = ChaoticSystem(
        name="linear-pilot",
        dimension=1,
        rhs=lambda state, _parameters: -np.asarray(state),
        equilibria=lambda _parameters: {"E0": np.zeros(1)},
        jacobian=lambda _state, _parameters: np.array([[-1.0]]),
    )
    record = SeedRecord(
        seed_id="linear-seed",
        system_id=system.name,
        route="manual",
        state=(1.0,),
    )
    contract = CaseRunContract(
        case_id="linear-case",
        system_id=system.name,
        parameter_set_id="linear-default",
        order_kind="integer",
        q=1.0,
        parameters={},
        tau_star=0.02,
        tau_star_basis="synthetic unit-test characteristic time",
        coordinate_scale=(1.0,),
        primary_solver=SolverSpec(
            "dop853-test", "dop853", rtol=1.0e-10, atol=1.0e-12
        ),
        secondary_solver=SolverSpec("rk4-test", "rk4", step_factor=0.5),
        classifier_overrides={"min_tail_samples": 16},
    )
    return CaseDefinition(
        contract=contract,
        system=system,
        bank=build_seed_bank((record,), coordinate_scale=(1.0,)),
        references={},
        equilibria=system.equilibrium_points(),
        budget_seed_ids={"B0": (record.seed_id,), "B1": (), "B2": ()},
    )


def _synthetic_result(
    case: CaseDefinition,
    run_root: Path,
    *,
    seed_id: str,
    budget_level: str,
    solver: SolverSpec,
    states: np.ndarray,
    case_id: str | None = None,
    system_id: str | None = None,
    label: str = "equilibrium",
    destination_id: str = "equilibrium:E0",
    observables: dict[str, object] | None = None,
) -> TrajectoryResult:
    """Build one persisted-looking result without invoking an integrator."""

    result_case_id = case_id or case.contract.case_id
    result_system_id = system_id or case.contract.system_id
    values = np.asarray(states, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    trajectory = run_root / "trajectories" / (
        f"{result_case_id}__{seed_id}__{budget_level}__{solver.solver_id}.csv"
    )
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    times = 0.01 * np.arange(values.shape[0], dtype=float)
    np.savetxt(
        trajectory,
        np.column_stack((times, values)),
        delimiter=",",
        header=",".join(("time", *(f"x{index + 1}" for index in range(values.shape[1])))),
        comments="",
    )
    classification = DestinationClassification(
        label=label,  # type: ignore[arg-type]
        destination_id=destination_id,
        subtype="synthetic-test",
        confidence=1.0,
        is_ambiguous=False,
        reasons=("synthetic regression fixture",),
        metrics={},
    )
    default_observables: dict[str, object] = {
        "tail_span_scaled": [1.0],
        "tail_centroid_scaled": [0.25],
        "tail_covariance_scaled": [[0.125]],
    }
    return TrajectoryResult(
        trajectory_id=f"{result_case_id}__{seed_id}__{budget_level}__{solver.solver_id}",
        case_id=result_case_id,
        system_id=result_system_id,
        seed_id=seed_id,
        budget_level=budget_level,
        integrator_id=solver.solver_id,
        integration_status="ok",
        classification=classification,
        trajectory_artifact=trajectory.relative_to(run_root).as_posix(),
        metadata_artifact="metadata/synthetic.json",
        trajectory_sha256="synthetic-sha256",
        physical_budget=case.contract.physical_budget(budget_level),
        solver_info={"family": solver.family},
        observables=observables or default_observables,
        elapsed_seconds=0.0,
        evidence_level="EV-TG2",
    )


def test_physical_budget_applies_declared_tau_star() -> None:
    from hidden_attractors.workflows.geometric_topological_campaign import DEFAULT_B0_B2_BUDGETS

    budget = PhysicalBudget.from_campaign(DEFAULT_B0_B2_BUDGETS[0], 0.1)
    assert budget.step == pytest.approx(0.002)
    assert budget.horizon == pytest.approx(5.0)
    assert budget.burn_time == pytest.approx(2.5)
    assert budget.nominal_steps == 2500


def test_case_contract_rejects_caputo_without_full_memory_reset() -> None:
    with pytest.raises(ValueError, match="full_memory_reset"):
        CaseRunContract(
            case_id="bad-caputo",
            system_id="bad",
            parameter_set_id="bad",
            order_kind="caputo",
            q=0.99,
            parameters={},
            tau_star=1.0,
            tau_star_basis="test",
            coordinate_scale=(1.0,),
            lower_terminal=0.0,
            initial_time=0.0,
            history_mode="caputo_reset",
        )


def test_append_case_bank_and_persist_classified_trajectory(tmp_path) -> None:
    case = _linear_case(tmp_path)
    manifest = CampaignManifest(campaign_id="pilot-test", cases=(case.contract.case_id,))
    paths = initialize_campaign_artifacts(manifest, root=tmp_path / "run")
    append_case_seed_bank(paths, manifest, case)

    result = run_and_classify_seed(
        case,
        case.record("linear-seed"),
        "B0",
        case.contract.primary_solver,
        run_root=paths.root,
    )
    append_trajectory_result(
        paths,
        result,
        campaign_id=manifest.campaign_id,
        contract=case.contract,
    )

    assert result.integration_status == "ok"
    assert (paths.root / result.trajectory_artifact).exists()
    assert (paths.root / result.metadata_artifact).exists()
    rows = paths.trajectory_metrics.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    with paths.trajectory_metrics.open(newline="", encoding="utf-8") as handle:
        metric_row = next(csv.DictReader(handle))
    metadata = json.loads((paths.root / result.metadata_artifact).read_text(encoding="utf-8"))
    assert metadata["trajectory_sha256"] == result.trajectory_sha256
    assert metadata["physical_budget"]["tau_star"] == pytest.approx(0.02)
    assert metric_row["classifier_id"] == DESTINATION_CLASSIFIER_ID
    assert metadata["classification"]["schema_version"] == (
        DESTINATION_CLASSIFIER_SCHEMA_VERSION
    )

    with pytest.raises(ValueError, match="already recorded"):
        append_trajectory_result(
            paths,
            result,
            campaign_id=manifest.campaign_id,
            contract=case.contract,
        )


def test_evidence_evaluation_isolated_by_case_before_key_indexing(tmp_path) -> None:
    case = _linear_case(tmp_path)
    target: list[TrajectoryResult] = []
    for level in ("B1", "B2"):
        for solver in (case.contract.primary_solver, case.contract.secondary_solver):
            target.append(
                _synthetic_result(
                    case,
                    tmp_path,
                    seed_id="linear-seed",
                    budget_level=level,
                    solver=solver,
                    states=np.array([[1.0], [0.75], [0.50]]),
                )
            )

    # These rows intentionally reuse every seed/budget/solver key.  If case
    # filtering happened after dictionary construction they would overwrite
    # the target rows and destroy the EV--TG3 decision.
    foreign: list[TrajectoryResult] = []
    for level in ("B1", "B2"):
        for solver in (case.contract.primary_solver, case.contract.secondary_solver):
            foreign.append(
                _synthetic_result(
                    case,
                    tmp_path,
                    seed_id="linear-seed",
                    budget_level=level,
                    solver=solver,
                    states=np.array([[9.0], [8.0], [7.0]]),
                    case_id="foreign-case",
                    system_id="foreign-system",
                    label="transient",
                    destination_id="transient:unsettled",
                    observables={
                        "tail_span_scaled": [99.0],
                        "tail_centroid_scaled": [99.0],
                        "tail_covariance_scaled": [[99.0]],
                    },
                )
            )

    decision = evaluate_case_evidence(case, [*target, *foreign], run_root=tmp_path)

    assert decision["evidence_level"] == "EV-TG3"
    assert "maximum evidence level" in decision["aggregation_policy"]
    assert [record["seed_id"] for record in decision["records"]] == ["linear-seed"]
    assert decision["records"][0]["horizon_destination_gate"] is True
    assert decision["records"][0]["numerical_convergence_gate"] is True


def test_evidence_evaluation_rejects_duplicate_key_within_case(tmp_path) -> None:
    case = _linear_case(tmp_path)
    result = _synthetic_result(
        case,
        tmp_path,
        seed_id="linear-seed",
        budget_level="B1",
        solver=case.contract.primary_solver,
        states=np.array([[1.0], [0.5]]),
    )

    with pytest.raises(ValueError, match="duplicate trajectory result within one case"):
        evaluate_case_evidence(case, [result, result], run_root=tmp_path)


def test_central_symmetry_diagnostic_is_exact_and_case_isolated(tmp_path) -> None:
    case = _linear_case(tmp_path)
    plus = np.array([[1.0], [0.5], [0.25]])
    target: list[TrajectoryResult] = []
    foreign: list[TrajectoryResult] = []
    for solver in (case.contract.primary_solver, case.contract.secondary_solver):
        target.extend(
            (
                _synthetic_result(
                    case,
                    tmp_path,
                    seed_id="plus",
                    budget_level="B2",
                    solver=solver,
                    states=plus,
                ),
                _synthetic_result(
                    case,
                    tmp_path,
                    seed_id="minus",
                    budget_level="B2",
                    solver=solver,
                    states=-plus,
                ),
            )
        )
        foreign.extend(
            (
                _synthetic_result(
                    case,
                    tmp_path,
                    seed_id="plus",
                    budget_level="B2",
                    solver=solver,
                    states=plus,
                    case_id="foreign-case",
                    system_id="foreign-system",
                ),
                _synthetic_result(
                    case,
                    tmp_path,
                    seed_id="minus",
                    budget_level="B2",
                    solver=solver,
                    states=plus,
                    case_id="foreign-case",
                    system_id="foreign-system",
                ),
            )
        )

    diagnostic = compute_central_symmetry_diagnostics(
        case,
        [*target, *foreign],
        run_root=tmp_path,
        pairs=(("plus", "minus"),),
        levels=("B2",),
    )

    assert diagnostic["all_pass_1e-10"] is True
    assert len(diagnostic["records"]) == 2
    assert all(
        record["maximum_scaled_odd_symmetry_residual"] == pytest.approx(0.0)
        for record in diagnostic["records"]
    )


def test_central_symmetry_diagnostic_detects_scaled_violation(tmp_path) -> None:
    case = _linear_case(tmp_path)
    solver = case.contract.primary_solver
    plus = np.array([[1.0], [0.5], [0.25]])
    minus = -plus
    minus[-1, 0] += 1.0e-6
    results = [
        _synthetic_result(
            case,
            tmp_path,
            seed_id="plus",
            budget_level="B0",
            solver=solver,
            states=plus,
        ),
        _synthetic_result(
            case,
            tmp_path,
            seed_id="minus",
            budget_level="B0",
            solver=solver,
            states=minus,
        ),
    ]

    diagnostic = compute_central_symmetry_diagnostics(
        case,
        results,
        run_root=tmp_path,
        pairs=(("plus", "minus"),),
        levels=("B0",),
    )

    assert diagnostic["all_pass_1e-10"] is False
    assert diagnostic["records"][0]["maximum_scaled_odd_symmetry_residual"] == pytest.approx(
        1.0e-6
    )
