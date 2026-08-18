from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from hidden_attractors.verification.edge_tracking import (
    EdgeDestination,
    EdgeTrackingConfig,
    ScaledEuclideanGeometry,
)
from hidden_attractors.workflows.geometric_topological_campaign import (
    DEFAULT_B0_B2_BUDGETS,
    EDGE_BRACKET_FIELDS,
    SEED_BANK_FIELDS,
    CampaignManifest,
    EdgeRunContext,
    initialize_campaign_artifacts,
    run_edge_tracking_and_record,
)


def test_campaign_initializer_creates_declared_pending_artifacts(tmp_path) -> None:
    manifest = CampaignManifest(
        campaign_id="tg-synthetic",
        cases=("synthetic-double-well",),
    )
    paths = initialize_campaign_artifacts(manifest, root=tmp_path / "validation-06")

    assert all(path.exists() for path in paths.required_files())
    payload = json.loads(paths.campaign_manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "initialized_dynamic_results_pending"
    assert payload["budget_contract"]["B0"]["nominal_steps"] == 2500
    assert payload["budget_contract"]["B1"]["nominal_steps"] == 10000
    assert payload["budget_contract"]["B2"]["nominal_steps"] == 40000
    outer = json.loads(paths.outer_enclosures.read_text(encoding="utf-8"))
    assert outer["status"] == "pending_TG7_not_executed"
    assert outer["records"] == []

    with paths.seed_bank.open(newline="", encoding="utf-8") as handle:
        seed_header = tuple(next(csv.reader(handle)))
        assert seed_header == SEED_BANK_FIELDS
        case_index = seed_header.index("case_id")
        assert seed_header[case_index - 1 : case_index + 2] == (
            "seed_id",
            "case_id",
            "system_id",
        )
    with paths.trajectory_metrics.open(newline="", encoding="utf-8") as handle:
        metric_header = tuple(next(csv.reader(handle)))
        assert metric_header[:4] == (
            "schema_version",
            "campaign_id",
            "case_id",
            "system_id",
        )
    with paths.edge_brackets.open(newline="", encoding="utf-8") as handle:
        assert tuple(next(csv.reader(handle))) == EDGE_BRACKET_FIELDS

    with pytest.raises(FileExistsError):
        initialize_campaign_artifacts(manifest, root=paths.root)


def test_edge_runner_records_confirmations_iterations_and_summary(tmp_path) -> None:
    paths = initialize_campaign_artifacts(
        CampaignManifest(campaign_id="tg-edge", cases=("synthetic-edge-case",)),
        root=tmp_path / "validation-06",
    )
    geometry = ScaledEuclideanGeometry((1.0,))

    def evaluator(state: np.ndarray, _context: object) -> EdgeDestination:
        return EdgeDestination.terminal("basin_a" if state[0] < 0.2 else "basin_b")

    context = EdgeRunContext(
        campaign_id="tg-edge",
        case_id="synthetic-edge-case",
        system_id="synthetic-threshold-model",
        contract_id="synthetic-q1-B0-B2",
        budget_level="B2",
        integrator_id="analytic-classifier",
        history_mode="not_applicable",
        classifier_id="threshold-v1",
        seed_left_id="left-001",
        seed_right_id="right-001",
    )
    result = run_edge_tracking_and_record(
        [-1.0],
        [1.0],
        evaluator=evaluator,
        geometry=geometry,
        bracket_id="edge-001",
        context=context,
        paths=paths,
        config=EdgeTrackingConfig(tolerance=1.0e-5),
    )

    assert result.status == "converged"
    with paths.edge_brackets.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert sum(row["record_type"] == "endpoint_confirmation" for row in rows) == 4
    assert sum(row["record_type"] == "iteration" for row in rows) == len(result.iterations)
    summaries = [row for row in rows if row["record_type"] == "summary"]
    assert len(summaries) == 1
    assert summaries[0]["result_status"] == "converged"
    assert summaries[0]["case_id"] == "synthetic-edge-case"
    assert summaries[0]["system_id"] == "synthetic-threshold-model"
    assert summaries[0]["method"] == "initial_data_boundary_bisection"
    assert summaries[0]["finite_resolution_only"] == "True"
    assert summaries[0]["evidence_level"] == "EV-TG2"
    confirmations = [row for row in rows if row["record_type"] == "endpoint_confirmation"]
    assert all(row["evaluation_metadata_json"] == "{}" for row in confirmations)
    assert [row["budget_level"] for row in confirmations] == ["B1", "B1", "B2", "B2"]

    with pytest.raises(ValueError, match="already recorded"):
        run_edge_tracking_and_record(
            [-1.0],
            [1.0],
            evaluator=evaluator,
            geometry=geometry,
            bracket_id="edge-001",
            context=context,
            paths=paths,
            config=EdgeTrackingConfig(tolerance=1.0e-5),
        )


def test_default_campaign_budgets_stop_at_b2() -> None:
    assert [budget.level for budget in DEFAULT_B0_B2_BUDGETS] == ["B0", "B1", "B2"]


def test_edge_record_rejects_case_budget_and_history_mismatches(tmp_path) -> None:
    paths = initialize_campaign_artifacts(
        CampaignManifest(campaign_id="tg-contract", cases=("declared-case",)),
        root=tmp_path / "validation-06",
    )
    geometry = ScaledEuclideanGeometry((1.0,))

    def evaluator(state: np.ndarray, _context: object) -> EdgeDestination:
        return EdgeDestination.terminal("left" if state[0] < 0.0 else "right")

    base = dict(
        campaign_id="tg-contract",
        case_id="declared-case",
        contract_id="contract-001",
        budget_level="B2",
        integrator_id="synthetic",
        history_mode="not_applicable",
        classifier_id="threshold-v1",
        seed_left_id="left",
        seed_right_id="right",
    )
    wrong_case = {**base, "case_id": "undeclared-case"}
    with pytest.raises(ValueError, match="case_id"):
        run_edge_tracking_and_record(
            [-1.0],
            [1.0],
            evaluator=evaluator,
            geometry=geometry,
            bracket_id="wrong-system",
            context=EdgeRunContext(system_id="model-a", **wrong_case),
            paths=paths,
        )

    wrong_budget = {**base, "budget_level": "B99"}
    with pytest.raises(ValueError, match="budget_level"):
        run_edge_tracking_and_record(
            [-1.0],
            [1.0],
            evaluator=evaluator,
            geometry=geometry,
            bracket_id="wrong-budget",
            context=EdgeRunContext(system_id="model-a", **wrong_budget),
            paths=paths,
        )

    wrong_history = {**base, "history_mode": "caputo_reset"}
    with pytest.raises(ValueError, match="history_mode"):
        run_edge_tracking_and_record(
            [-1.0],
            [1.0],
            evaluator=evaluator,
            geometry=geometry,
            bracket_id="wrong-history",
            context=EdgeRunContext(system_id="model-a", **wrong_history),
            paths=paths,
        )

    missing_case = {**base, "case_id": ""}
    with pytest.raises(ValueError, match="case_id"):
        EdgeRunContext(system_id="model-a", **missing_case)

    with pytest.raises(ValueError, match="not declared"):
        run_edge_tracking_and_record(
            [-1.0],
            [1.0],
            evaluator=evaluator,
            geometry=geometry,
            bracket_id="wrong-confirmation-budget",
            context=EdgeRunContext(system_id="model-a", **base),
            paths=paths,
            config=EdgeTrackingConfig(confirmation_levels=("B1", "B99")),
        )

    with pytest.raises(ValueError, match="not declared"):
        run_edge_tracking_and_record(
            [-1.0],
            [1.0],
            evaluator=evaluator,
            geometry=geometry,
            bracket_id="wrong-tracking-budget",
            context=EdgeRunContext(system_id="model-a", **base),
            paths=paths,
            config=EdgeTrackingConfig(tracking_level="B98"),
        )

    with paths.edge_brackets.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []


def test_empty_pre_case_id_edge_header_is_upgraded_on_first_record(tmp_path) -> None:
    paths = initialize_campaign_artifacts(
        CampaignManifest(campaign_id="tg-upgrade", cases=("case-001",)),
        root=tmp_path / "validation-06",
    )
    legacy_fields = tuple(field for field in EDGE_BRACKET_FIELDS if field != "case_id")
    with paths.edge_brackets.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(legacy_fields)

    def evaluator(state: np.ndarray, _context: object) -> EdgeDestination:
        return EdgeDestination.terminal("left" if state[0] < 0.0 else "right")

    run_edge_tracking_and_record(
        [-1.0],
        [1.0],
        evaluator=evaluator,
        geometry=ScaledEuclideanGeometry((1.0,)),
        bracket_id="upgrade-edge",
        context=EdgeRunContext(
            campaign_id="tg-upgrade",
            case_id="case-001",
            system_id="model-distinct-from-case",
            contract_id="contract-001",
            budget_level="B2",
            integrator_id="synthetic",
            history_mode="not_applicable",
            classifier_id="threshold-v1",
            seed_left_id="left",
            seed_right_id="right",
        ),
        paths=paths,
        config=EdgeTrackingConfig(tolerance=1.0e-5),
    )

    with paths.edge_brackets.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert "case_id" in rows[0]
    assert {row["case_id"] for row in rows} == {"case-001"}
