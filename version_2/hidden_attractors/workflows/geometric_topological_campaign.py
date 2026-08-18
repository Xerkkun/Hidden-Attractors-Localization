"""Artifacts and the minimal B0--B2 runner contract for the TG campaign.

Stability: experimental
    The artifact schemas are versioned, but the orchestration surface may
    expand when TG7 validated-enclosure support is implemented.

This module creates the digital evidence surface declared by the master
report.  Initialisation writes schemas and explicit ``pending`` payloads; it
does not manufacture dynamic results.  Edge histories can subsequently be
appended through :func:`run_edge_tracking_and_record`.

All recorded conclusions remain finite-time and finite-resolution numerical
evidence.  A completed edge bracket is not an isolating block, a Conley-index
calculation, a global hiddenness proof, or a Wada test.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..io import json_safe, read_csv_rows, write_csv, write_json
from ..paths import PROJECT_ROOT
from ..seed_bank import SEED_BANK_CSV_FIELDS
from ..verification.edge_tracking import (
    EdgeEvaluator,
    EdgeGeometry,
    EdgeTrackingConfig,
    EdgeTrackingResult,
    track_edge_bracket,
)


CAMPAIGN_SCHEMA_VERSION = "1.1"
CAMPAIGN_PROTOCOL_VERSION = "geometric_topological_TG0_TG8_v1"
DEFAULT_CAMPAIGN_ROOT = PROJECT_ROOT / "validation" / "06_geometric_topological_campaign"


_SEED_SYSTEM_INDEX = SEED_BANK_CSV_FIELDS.index("system_id")
SEED_BANK_FIELDS = (
    *SEED_BANK_CSV_FIELDS[:_SEED_SYSTEM_INDEX],
    "case_id",
    *SEED_BANK_CSV_FIELDS[_SEED_SYSTEM_INDEX:],
)

TRAJECTORY_METRIC_FIELDS = (
    "schema_version",
    "campaign_id",
    "case_id",
    "system_id",
    "seed_id",
    "contract_id",
    "budget_level",
    "integrator_id",
    "history_mode",
    "classifier_id",
    "window_start",
    "window_stop",
    "integration_status",
    "destination_label",
    "destination_subtype",
    "classification_confidence",
    "is_ambiguous",
    "metrics_json",
    "evidence_level",
    "trajectory_id",
    "destination_id",
    "trajectory_artifact",
    "metadata_artifact",
    "trajectory_sha256",
    "order_kind",
    "q",
    "tau_star",
    "step",
    "horizon",
    "n_samples",
)

EDGE_BRACKET_FIELDS = (
    "schema_version",
    "campaign_id",
    "bracket_id",
    "record_type",
    "case_id",
    "system_id",
    "contract_id",
    "budget_level",
    "integrator_id",
    "history_mode",
    "classifier_id",
    "seed_left_id",
    "seed_right_id",
    "method",
    "geometry",
    "data_semantics",
    "admissible_history_family_id",
    "iteration",
    "phase",
    "ambiguity_index",
    "coordinate",
    "destination_label",
    "destination_resolved",
    "evaluation_ok",
    "integration_status",
    "classification_reason",
    "evaluation_metadata_json",
    "width_before",
    "width_after",
    "left_state_after",
    "right_state_after",
    "ambiguous_streak",
    "iteration_event",
    "result_status",
    "stop_reason",
    "left_destination",
    "right_destination",
    "initial_width",
    "final_width",
    "candidate_state",
    "evaluation_trace_json",
    "evidence_level",
    "finite_resolution_only",
)


@dataclass(frozen=True)
class CampaignBudget:
    """Dimensionless integration budget from the declared TG protocol."""

    level: str
    normalized_step: float
    normalized_horizon: float
    transient_fraction: float
    seed_scope: str
    purpose: str

    def __post_init__(self) -> None:
        if not str(self.level).strip():
            raise ValueError("budget level must be non-empty.")
        if not np.isfinite(self.normalized_step) or self.normalized_step <= 0.0:
            raise ValueError("normalized_step must be finite and positive.")
        if not np.isfinite(self.normalized_horizon) or self.normalized_horizon <= 0.0:
            raise ValueError("normalized_horizon must be finite and positive.")
        if not np.isfinite(self.transient_fraction) or not 0.0 <= self.transient_fraction < 1.0:
            raise ValueError("transient_fraction must lie in [0, 1).")
        if not str(self.seed_scope).strip() or not str(self.purpose).strip():
            raise ValueError("budget seed_scope and purpose must be non-empty.")

    @property
    def nominal_steps(self) -> int:
        return int(round(self.normalized_horizon / self.normalized_step))


DEFAULT_B0_B2_BUDGETS = (
    CampaignBudget(
        "B0",
        2.0e-2,
        50.0,
        0.5,
        "minimum_seed_bank",
        "screen_obvious errors, event failures, and escapes",
    ),
    CampaignBudget(
        "B1",
        1.0e-2,
        100.0,
        0.5,
        "complete_initial_seed_bank",
        "first destination assignment",
    ),
    CampaignBudget(
        "B2",
        5.0e-3,
        200.0,
        0.6,
        "ambiguous_and_edge_seeds",
        "classification and observable convergence",
    ),
)


@dataclass(frozen=True)
class CampaignManifest:
    """Minimal campaign-level record written before any dynamic run."""

    campaign_id: str
    cases: tuple[str, ...] = ()
    budgets: tuple[CampaignBudget, ...] = DEFAULT_B0_B2_BUDGETS
    status: str = "initialized_dynamic_results_pending"
    schema_version: str = CAMPAIGN_SCHEMA_VERSION
    protocol_version: str = CAMPAIGN_PROTOCOL_VERSION
    claims_scope: str = (
        "finite-time finite-resolution numerical localization; edge records currently "
        "cover initial-data boundary bisection only, with no global hiddenness, Wada, "
        "isolating-block, or Conley-index claim"
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.campaign_id).strip():
            raise ValueError("campaign_id must be non-empty.")
        case_names = tuple(str(case).strip() for case in self.cases)
        if any(not case for case in case_names):
            raise ValueError("campaign cases must have non-empty identifiers.")
        if not self.budgets:
            raise ValueError("campaign must declare at least one budget.")
        levels = [budget.level for budget in self.budgets]
        if len(set(levels)) != len(levels):
            raise ValueError("campaign budget levels must be unique.")
        if not str(self.status).strip() or not str(self.claims_scope).strip():
            raise ValueError("campaign status and claims_scope must be non-empty.")
        object.__setattr__(self, "cases", case_names)

    def to_jsonable(self) -> dict[str, Any]:
        payload = json_safe(asdict(self))
        payload["case_id_semantics"] = (
            "cases contains experimental case identifiers; each artifact records "
            "system_id separately as the dynamical-model identifier"
        )
        payload["artifacts"] = {
            "seed_bank": "seed_bank.csv",
            "trajectory_metrics": "trajectory_metrics.csv",
            "edge_brackets": "edge_brackets.csv",
            "outer_enclosures": "outer_enclosures.json",
            "evidence_decisions": "evidence_decisions.json",
        }
        payload["budget_contract"] = {
            budget.level: {
                "normalized_step": budget.normalized_step,
                "normalized_horizon": budget.normalized_horizon,
                "transient_fraction": budget.transient_fraction,
                "nominal_steps": budget.nominal_steps,
                "seed_scope": budget.seed_scope,
                "purpose": budget.purpose,
            }
            for budget in self.budgets
        }
        payload["status_policy"] = (
            "campaign_manifest is an immutable initialization contract; partial edge or "
            "trajectory records do not promote campaign status, and TG decisions belong "
            "in evidence_decisions.json"
        )
        return payload


@dataclass(frozen=True)
class CampaignArtifactPaths:
    """Canonical paths under one campaign evidence directory."""

    root: Path
    campaign_manifest: Path
    seed_bank: Path
    trajectory_metrics: Path
    edge_brackets: Path
    outer_enclosures: Path
    evidence_decisions: Path

    @classmethod
    def under(cls, root: str | Path = DEFAULT_CAMPAIGN_ROOT) -> "CampaignArtifactPaths":
        base = Path(root)
        return cls(
            root=base,
            campaign_manifest=base / "campaign_manifest.json",
            seed_bank=base / "seed_bank.csv",
            trajectory_metrics=base / "trajectory_metrics.csv",
            edge_brackets=base / "edge_brackets.csv",
            outer_enclosures=base / "outer_enclosures.json",
            evidence_decisions=base / "evidence_decisions.json",
        )

    def required_files(self) -> tuple[Path, ...]:
        return (
            self.campaign_manifest,
            self.seed_bank,
            self.trajectory_metrics,
            self.edge_brackets,
            self.outer_enclosures,
            self.evidence_decisions,
        )


@dataclass(frozen=True)
class EdgeRunContext:
    """Traceability fields common to every row of one edge result.

    The runner implemented by this module performs only finite-time
    initial-data boundary bisection.  Its conservative default is therefore
    EV--TG2; a geometrically narrow bracket is not, by itself, an EV--TG4
    dynamic edge-trajectory result.
    """

    campaign_id: str
    case_id: str
    system_id: str
    contract_id: str
    budget_level: str
    integrator_id: str
    history_mode: str
    classifier_id: str
    seed_left_id: str
    seed_right_id: str
    evidence_level: str = "EV-TG2"

    def __post_init__(self) -> None:
        required = {
            "campaign_id": self.campaign_id,
            "case_id": self.case_id,
            "system_id": self.system_id,
            "contract_id": self.contract_id,
            "budget_level": self.budget_level,
            "integrator_id": self.integrator_id,
            "history_mode": self.history_mode,
            "classifier_id": self.classifier_id,
            "seed_left_id": self.seed_left_id,
            "seed_right_id": self.seed_right_id,
            "evidence_level": self.evidence_level,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"edge run context requires non-empty fields: {missing}.")
        allowed_history_modes = {
            "not_applicable",
            "caputo_reset",
            "caputo_inherited_admissible_family",
        }
        if self.history_mode not in allowed_history_modes:
            raise ValueError(
                "history_mode must be 'not_applicable', 'caputo_reset', or "
                "'caputo_inherited_admissible_family'."
            )


def initialize_campaign_artifacts(
    manifest: CampaignManifest,
    *,
    root: str | Path = DEFAULT_CAMPAIGN_ROOT,
    overwrite: bool = False,
) -> CampaignArtifactPaths:
    """Create the six declared campaign artifacts without dynamic claims.

    Existing artifacts are protected by default.  ``overwrite=True`` is an
    explicit replacement of the entire six-file campaign surface and should
    be used only for a known disposable run directory.
    """

    paths = CampaignArtifactPaths.under(root)
    existing = [path for path in paths.required_files() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(path.name for path in existing)
        raise FileExistsError(f"campaign artifacts already exist: {joined}.")
    paths.root.mkdir(parents=True, exist_ok=True)
    write_csv(paths.seed_bank, (), fields=SEED_BANK_FIELDS)
    write_csv(paths.trajectory_metrics, (), fields=TRAJECTORY_METRIC_FIELDS)
    write_csv(paths.edge_brackets, (), fields=EDGE_BRACKET_FIELDS)
    write_json(
        paths.outer_enclosures,
        {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": manifest.campaign_id,
            "status": "pending_TG7_not_executed",
            "records": [],
            "claims_scope": "no outer enclosure, isolating block, or Conley index computed",
        },
    )
    write_json(
        paths.evidence_decisions,
        {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": manifest.campaign_id,
            "status": "pending_dynamic_execution",
            "records": [],
        },
    )
    # The manifest is the completion marker for this six-file initialisation.
    write_json(paths.campaign_manifest, manifest.to_jsonable())
    return paths


def _evaluation_payload(record: Any) -> dict[str, Any]:
    return {
        "phase": record.context.phase,
        "budget_level": record.context.budget_level,
        "iteration": record.context.iteration,
        "ambiguity_index": record.context.ambiguity_index,
        "coordinate": list(record.coordinate),
        "destination_label": record.outcome.label,
        "destination_resolved": record.outcome.resolved,
        "evaluation_ok": record.outcome.evaluation_ok,
        "integration_status": record.outcome.integration_status,
        "reason": record.outcome.reason,
        "metadata": record.outcome.metadata,
    }


def edge_result_rows(
    result: EdgeTrackingResult,
    context: EdgeRunContext,
) -> list[dict[str, Any]]:
    """Flatten confirmations, iterations, and summary into audit-ready rows."""

    common: dict[str, Any] = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": context.campaign_id,
        "bracket_id": result.bracket_id,
        "case_id": context.case_id,
        "system_id": context.system_id,
        "contract_id": context.contract_id,
        "budget_level": context.budget_level,
        "integrator_id": context.integrator_id,
        "history_mode": context.history_mode,
        "classifier_id": context.classifier_id,
        "seed_left_id": context.seed_left_id,
        "seed_right_id": context.seed_right_id,
        "method": result.method,
        "geometry": result.geometry,
        "data_semantics": result.data_semantics,
        "admissible_history_family_id": result.admissible_history_family_id or "",
        "evidence_level": context.evidence_level,
        "finite_resolution_only": result.finite_resolution_only,
    }
    rows: list[dict[str, Any]] = []
    for confirmation in result.confirmations:
        rows.append(
            {
                **common,
                "record_type": "endpoint_confirmation",
                "budget_level": confirmation.context.budget_level,
                "iteration": confirmation.context.iteration,
                "phase": confirmation.context.phase,
                "ambiguity_index": confirmation.context.ambiguity_index,
                "coordinate": confirmation.coordinate,
                "destination_label": confirmation.outcome.label,
                "destination_resolved": confirmation.outcome.resolved,
                "evaluation_ok": confirmation.outcome.evaluation_ok,
                "integration_status": confirmation.outcome.integration_status,
                "classification_reason": confirmation.outcome.reason,
                "evaluation_metadata_json": json.dumps(
                    json_safe(confirmation.outcome.metadata), sort_keys=True
                ),
            }
        )
    for iteration in result.iterations:
        rows.append(
            {
                **common,
                "record_type": "iteration",
                "budget_level": iteration.evaluations[0].context.budget_level,
                "iteration": iteration.iteration,
                "phase": "midpoint",
                "ambiguity_index": iteration.ambiguous_streak,
                "coordinate": iteration.midpoint,
                "destination_label": iteration.midpoint_outcome.label,
                "destination_resolved": iteration.midpoint_outcome.resolved,
                "evaluation_ok": iteration.midpoint_outcome.evaluation_ok,
                "integration_status": iteration.midpoint_outcome.integration_status,
                "classification_reason": iteration.midpoint_outcome.reason,
                "evaluation_metadata_json": json.dumps(
                    json_safe(iteration.midpoint_outcome.metadata), sort_keys=True
                ),
                "width_before": iteration.width_before,
                "width_after": iteration.width_after,
                "left_state_after": iteration.left_after,
                "right_state_after": iteration.right_after,
                "ambiguous_streak": iteration.ambiguous_streak,
                "iteration_event": iteration.event,
                "evaluation_trace_json": json.dumps(
                    json_safe([_evaluation_payload(record) for record in iteration.evaluations]),
                    sort_keys=True,
                ),
            }
        )
    rows.append(
        {
            **common,
            "record_type": "summary",
            "result_status": result.status,
            "stop_reason": result.stop_reason,
            "left_destination": result.left_destination,
            "right_destination": result.right_destination,
            "initial_width": result.initial_width,
            "final_width": result.final_width,
            "left_state_after": result.final_left,
            "right_state_after": result.final_right,
            "candidate_state": result.candidate,
        }
    )
    return rows


def append_edge_tracking_result(
    paths: CampaignArtifactPaths,
    result: EdgeTrackingResult,
    context: EdgeRunContext,
) -> None:
    """Append one complete edge result, rejecting duplicate bracket IDs."""

    used_budget_levels = {
        record.context.budget_level for record in result.confirmations
    }
    used_budget_levels.update(
        record.context.budget_level
        for iteration in result.iterations
        for record in iteration.evaluations
    )
    _validate_edge_record_context(
        paths,
        context,
        result.data_semantics,
        used_budget_levels=tuple(sorted(used_budget_levels)),
    )
    existing = read_csv_rows(paths.edge_brackets)
    if any(row.get("bracket_id") == result.bracket_id for row in existing):
        raise ValueError(f"edge bracket {result.bracket_id!r} is already recorded.")
    combined = [*existing, *edge_result_rows(result, context)]
    temporary = paths.edge_brackets.with_name(f".{paths.edge_brackets.name}.pending")
    try:
        write_csv(temporary, combined, EDGE_BRACKET_FIELDS)
        temporary.replace(paths.edge_brackets)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_edge_record_context(
    paths: CampaignArtifactPaths,
    context: EdgeRunContext,
    data_semantics: str,
    *,
    used_budget_levels: Sequence[str] = (),
) -> None:
    """Check manifest referential integrity before running or writing an edge."""

    if not paths.campaign_manifest.exists() or not paths.edge_brackets.exists():
        raise FileNotFoundError("initialise campaign artifacts before recording an edge result.")
    manifest = json.loads(paths.campaign_manifest.read_text(encoding="utf-8"))
    if str(manifest.get("campaign_id", "")) != context.campaign_id:
        raise ValueError("edge context campaign_id does not match campaign_manifest.json.")
    cases = tuple(str(case) for case in manifest.get("cases", ()))
    if context.case_id not in cases:
        raise ValueError("edge context case_id is not declared in campaign cases.")
    budget_levels = set(str(level) for level in manifest.get("budget_contract", {}))
    if context.budget_level not in budget_levels:
        raise ValueError("edge context budget_level is not declared in campaign manifest.")
    undeclared_levels = sorted(
        {str(level) for level in used_budget_levels if str(level) not in budget_levels}
    )
    if undeclared_levels:
        raise ValueError(
            "edge tracking uses budget levels not declared in campaign manifest: "
            f"{undeclared_levels}."
        )
    expected_history_mode = {
        "ode_initial_state": "not_applicable",
        "caputo_reset_initial_state": "caputo_reset",
        "admissible_history_family_parameter": "caputo_inherited_admissible_family",
    }.get(data_semantics)
    if expected_history_mode is None:
        raise ValueError(f"unsupported edge data semantics: {data_semantics!r}.")
    if context.history_mode != expected_history_mode:
        raise ValueError(
            "edge context history_mode is inconsistent with the edge data semantics."
        )


def run_edge_tracking_and_record(
    left: Sequence[float],
    right: Sequence[float],
    *,
    evaluator: EdgeEvaluator,
    geometry: EdgeGeometry,
    bracket_id: str,
    context: EdgeRunContext,
    paths: CampaignArtifactPaths,
    config: EdgeTrackingConfig | None = None,
) -> EdgeTrackingResult:
    """Execute one edge refinement and persist its complete finite history."""

    active_config = config or EdgeTrackingConfig()
    _validate_edge_record_context(
        paths,
        context,
        active_config.data_semantics,
        used_budget_levels=(
            *active_config.confirmation_levels,
            active_config.tracking_level,
        ),
    )
    result = track_edge_bracket(
        left,
        right,
        evaluator=evaluator,
        geometry=geometry,
        bracket_id=bracket_id,
        config=active_config,
    )
    append_edge_tracking_result(paths, result, context)
    return result


__all__ = [
    "CAMPAIGN_PROTOCOL_VERSION",
    "CAMPAIGN_SCHEMA_VERSION",
    "DEFAULT_B0_B2_BUDGETS",
    "DEFAULT_CAMPAIGN_ROOT",
    "EDGE_BRACKET_FIELDS",
    "SEED_BANK_FIELDS",
    "TRAJECTORY_METRIC_FIELDS",
    "CampaignArtifactPaths",
    "CampaignBudget",
    "CampaignManifest",
    "EdgeRunContext",
    "append_edge_tracking_result",
    "edge_result_rows",
    "initialize_campaign_artifacts",
    "run_edge_tracking_and_record",
]
