"""Executable B0--B2 pilot for the geometric--topological campaign.

Stability: experimental
    This module couples the unified seed bank, causal integrators, the finite-
    time destination classifier, and the initial-data boundary bisection.  It
    intentionally stops below set-oriented/Conley certification and below a
    global hiddenness claim.

The maintained pilot cases are the integer lead--lag PLL, integer MAVPD at
``xi=3.1``, and the Caputo Wu arctangent Chua system at ``q=.99``.  Every run
is written beneath a fresh ``validation/06_geometric_topological_campaign/
runs/<run_id>`` directory.  Algebraic seed provenance and dynamic evidence
are kept separate: a perpetual point is a localization proposal, never a
destination claim.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from ..integrations.fractional_c import fractional_integrate
from ..integrations.rk4 import rk4_integrate
from ..io import json_safe, read_csv_rows, write_csv, write_json
from ..paths import PROJECT_ROOT
from ..plotting.export import save_figure_pair_local
from ..seed_bank import (
    SeedBank,
    SeedRecord,
    SymmetryTransform,
    build_seed_bank,
    seed_bank_csv_rows,
)
from ..solvers.integer import dop853_q1_integrate
from ..systems import get_system
from ..systems.base import ChaoticSystem
from ..systems.modified_van_der_pol_duffing import mavpd_2023_system
from ..systems.pll_lead_lag import (
    pll_lead_lag_2015_system,
    pll_original_to_shifted,
)
from ..verification.destination_classifier import (
    DESTINATION_CLASSIFIER_ID,
    DestinationClassification,
    DestinationClassifierContract,
    classify_destination,
)
from ..verification.edge_tracking import (
    EdgeEvaluationContext,
    EdgeTrackingConfig,
    ScaledCylindricalGeometry,
    ScaledEuclideanGeometry,
    edge_destination_from_classification,
)
from .geometric_topological_campaign import (
    CAMPAIGN_SCHEMA_VERSION,
    DEFAULT_B0_B2_BUDGETS,
    SEED_BANK_FIELDS,
    TRAJECTORY_METRIC_FIELDS,
    CampaignArtifactPaths,
    CampaignBudget,
    CampaignManifest,
    EdgeRunContext,
    initialize_campaign_artifacts,
    run_edge_tracking_and_record,
)


PILOT_PROTOCOL_VERSION = "geometric_topological_initial_pilot_v1"
DEFAULT_RUN_ID = "tg_pll_mavpd_wu_b0_b2_20260812"
DEFAULT_RUNS_ROOT = (
    PROJECT_ROOT / "validation" / "06_geometric_topological_campaign" / "runs"
)


@dataclass(frozen=True)
class SolverSpec:
    """One solver contract retained with every trajectory."""

    solver_id: str
    family: str
    step_factor: float = 1.0
    rtol: float | None = None
    atol: float | None = None
    full_memory: bool = False
    use_native_backend: bool = False

    def __post_init__(self) -> None:
        if not str(self.solver_id).strip() or not str(self.family).strip():
            raise ValueError("solver_id and family must be non-empty.")
        factor = float(self.step_factor)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("step_factor must be finite and positive.")
        object.__setattr__(self, "step_factor", factor)
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(float(value)) or float(value) <= 0.0):
                raise ValueError(f"{name} must be finite and positive when supplied.")


@dataclass(frozen=True)
class PhysicalBudget:
    """Dimensionless campaign budget converted by a declared time scale."""

    level: str
    normalized_step: float
    normalized_horizon: float
    transient_fraction: float
    tau_star: float
    step: float
    horizon: float
    burn_time: float
    nominal_steps: int

    @classmethod
    def from_campaign(cls, budget: CampaignBudget, tau_star: float) -> "PhysicalBudget":
        scale = float(tau_star)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("tau_star must be finite and positive.")
        step = scale * budget.normalized_step
        horizon = scale * budget.normalized_horizon
        return cls(
            level=budget.level,
            normalized_step=budget.normalized_step,
            normalized_horizon=budget.normalized_horizon,
            transient_fraction=budget.transient_fraction,
            tau_star=scale,
            step=step,
            horizon=horizon,
            burn_time=budget.transient_fraction * horizon,
            nominal_steps=budget.nominal_steps,
        )


@dataclass(frozen=True)
class CaseRunContract:
    """Frozen mathematical/numerical contract for one pilot case."""

    case_id: str
    system_id: str
    parameter_set_id: str
    order_kind: str
    q: float
    parameters: Mapping[str, Any]
    tau_star: float
    tau_star_basis: str
    coordinate_scale: tuple[float, ...]
    periodic_coordinates: Mapping[int, float] = field(default_factory=dict)
    divergence_radius: float = 120.0
    divergence_radius_kind: str = "absolute"
    lower_terminal: float | None = None
    initial_time: float | None = None
    history_mode: str = "not_applicable"
    primary_solver: SolverSpec = field(
        default_factory=lambda: SolverSpec(
            "dop853_rtol1e-10_atol1e-12", "dop853", rtol=1.0e-10, atol=1.0e-12
        )
    )
    secondary_solver: SolverSpec = field(
        default_factory=lambda: SolverSpec("rk4_fixed_h_over_2", "rk4", step_factor=0.5)
    )
    classifier_overrides: Mapping[str, Any] = field(default_factory=dict)
    reference_artifacts: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("case_id", "system_id", "parameter_set_id", "tau_star_basis"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        if self.order_kind not in {"integer", "caputo"}:
            raise ValueError("order_kind must be 'integer' or 'caputo'.")
        q = float(self.q)
        if not np.isfinite(q) or not 0.0 < q <= 1.0:
            raise ValueError("q must satisfy 0 < q <= 1.")
        if self.order_kind == "integer" and not math.isclose(q, 1.0, abs_tol=1.0e-12):
            raise ValueError("integer contracts require q=1.")
        if self.order_kind == "caputo" and not q < 1.0:
            raise ValueError("this pilot's Caputo backend requires q<1.")
        object.__setattr__(self, "q", q)
        scale = tuple(float(value) for value in self.coordinate_scale)
        if not scale or any(not np.isfinite(value) or value <= 0.0 for value in scale):
            raise ValueError("coordinate_scale must contain finite positive values.")
        object.__setattr__(self, "coordinate_scale", scale)
        tau = float(self.tau_star)
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau_star must be finite and positive.")
        object.__setattr__(self, "tau_star", tau)
        periodic = {int(index): float(period) for index, period in self.periodic_coordinates.items()}
        for index, period in periodic.items():
            if index < 0 or index >= len(scale) or not np.isfinite(period) or period <= 0.0:
                raise ValueError("invalid periodic coordinate declaration.")
        object.__setattr__(self, "periodic_coordinates", periodic)
        if self.order_kind == "caputo":
            if self.lower_terminal is None or self.initial_time is None:
                raise ValueError("Caputo contracts require lower_terminal and initial_time.")
            if not math.isclose(
                float(self.lower_terminal), float(self.initial_time), rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError("reset Caputo pilot requires initial_time=lower_terminal.")
            if self.history_mode != "caputo_full_memory_reset":
                raise ValueError("Caputo pilot requires caputo_full_memory_reset history mode.")
        elif self.history_mode != "not_applicable":
            raise ValueError("integer contracts require history_mode='not_applicable'.")
        parameter_values: dict[str, Any] = {}
        for key, value in self.parameters.items():
            if isinstance(value, (np.floating, float, np.integer, int)) and not isinstance(value, bool):
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise ValueError("numeric case parameters must be finite.")
                parameter_values[str(key)] = numeric
            elif value is None or isinstance(value, (str, bool)):
                parameter_values[str(key)] = value
            else:
                raise TypeError("case parameters must be finite numbers, strings, Boolean, or null.")
        object.__setattr__(self, "parameters", parameter_values)

    @property
    def contract_id(self) -> str:
        return f"{self.case_id}_{PILOT_PROTOCOL_VERSION}"

    def physical_budget(self, level: str) -> PhysicalBudget:
        for budget in DEFAULT_B0_B2_BUDGETS:
            if budget.level == level:
                return PhysicalBudget.from_campaign(budget, self.tau_star)
        raise KeyError(f"unknown campaign budget level {level!r}.")

    def classifier(self, level: str) -> DestinationClassifierContract:
        budget = self.physical_budget(level)
        values: dict[str, Any] = {
            "burn_time": budget.burn_time,
            "order_kind": self.order_kind,
            "coordinate_scale": self.coordinate_scale,
            "periodic_coordinates": self.periodic_coordinates,
            "require_declared_scale": True,
            "min_tail_samples": 128,
            "divergence_radius": self.divergence_radius,
            "divergence_radius_kind": self.divergence_radius_kind,
        }
        values.update(dict(self.classifier_overrides))
        return DestinationClassifierContract(**values)

    def to_jsonable(self) -> dict[str, Any]:
        payload = json_safe(asdict(self))
        payload["contract_id"] = self.contract_id
        payload["protocol_version"] = PILOT_PROTOCOL_VERSION
        payload["physical_budgets"] = {
            level: json_safe(asdict(self.physical_budget(level))) for level in ("B0", "B1", "B2")
        }
        payload["claim_limit"] = (
            "finite-time finite-resolution localization; no global attraction, hiddenness, "
            "Wada, isolating-block, or Conley-index claim"
        )
        return payload


@dataclass(frozen=True)
class CaseDefinition:
    """Runtime case adapter and its frozen bank/selection policy."""

    contract: CaseRunContract
    system: ChaoticSystem
    bank: SeedBank
    references: Mapping[str, np.ndarray]
    equilibria: Mapping[str, np.ndarray]
    budget_seed_ids: Mapping[str, tuple[str, ...]]

    def record(self, seed_id: str) -> SeedRecord:
        for membership in self.bank.memberships:
            if membership.record.seed_id == seed_id:
                return membership.record
        raise KeyError(seed_id)


@dataclass(frozen=True)
class TrajectoryResult:
    """One persisted integration and classification record."""

    trajectory_id: str
    case_id: str
    system_id: str
    seed_id: str
    budget_level: str
    integrator_id: str
    integration_status: str
    classification: DestinationClassification
    trajectory_artifact: str
    metadata_artifact: str
    trajectory_sha256: str
    physical_budget: PhysicalBudget
    solver_info: Mapping[str, Any]
    observables: Mapping[str, Any]
    elapsed_seconds: float
    evidence_level: str


def _classification_from_mapping(values: Mapping[str, Any]) -> DestinationClassification:
    """Reconstruct a strict classifier result from persisted JSON/CSV data."""

    return DestinationClassification(
        label=str(values["label"]),  # type: ignore[arg-type]
        destination_id=str(values["destination_id"]),
        subtype=str(values.get("subtype", "")),
        confidence=float(values.get("confidence", 0.0)),
        is_ambiguous=bool(values.get("is_ambiguous", False)),
        reasons=tuple(str(item) for item in values.get("reasons", ())),
        metrics=dict(values.get("metrics", {})),
        evidence_status=str(values.get("evidence_status", "finite_time_destination_diagnostic")),
        scientific_warnings=tuple(str(item) for item in values.get("scientific_warnings", ())),
        schema_version=str(values.get("schema_version", "1.0")),
    )


def load_persisted_results(run_root: Path) -> list[TrajectoryResult]:
    """Load trajectory results after an interrupted resumable pilot run."""

    results: list[TrajectoryResult] = []
    metrics_path = run_root / "trajectory_metrics.csv"
    if not metrics_path.exists():
        return results
    for row in read_csv_rows(metrics_path):
        payload = json.loads(row.get("metrics_json", "{}") or "{}")
        classification_values = payload.get("classification")
        physical_values = payload.get("physical_budget")
        if not isinstance(classification_values, Mapping) or not isinstance(physical_values, Mapping):
            raise ValueError("persisted trajectory row lacks full classification or budget payload.")
        results.append(
            TrajectoryResult(
                trajectory_id=str(row.get("trajectory_id") or payload["trajectory_id"]),
                case_id=str(row["case_id"]),
                system_id=str(row["system_id"]),
                seed_id=str(row["seed_id"]),
                budget_level=str(row["budget_level"]),
                integrator_id=str(row["integrator_id"]),
                integration_status=str(row["integration_status"]),
                classification=_classification_from_mapping(classification_values),
                trajectory_artifact=str(row.get("trajectory_artifact") or payload["trajectory_artifact"]),
                metadata_artifact=str(row.get("metadata_artifact") or payload["metadata_artifact"]),
                trajectory_sha256=str(row.get("trajectory_sha256") or payload["trajectory_sha256"]),
                physical_budget=PhysicalBudget(**physical_values),
                solver_info=dict(payload.get("solver_info", {})),
                observables=dict(payload.get("observables", {})),
                elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
                evidence_level=str(row.get("evidence_level", "EV-TG1")),
            )
        )
    return results


def _load_reference_cloud(path: Path, *, max_points: int = 12000) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = list(data.dtype.names or ())
    columns = [name for name in names if name.lower() not in {"t", "time"}]
    if not columns:
        raise ValueError(f"reference cloud has no state columns: {path}.")
    cloud = np.column_stack([np.atleast_1d(data[name]).astype(float) for name in columns])
    cloud = cloud[np.all(np.isfinite(cloud), axis=1)]
    if cloud.shape[0] > max_points:
        indices = np.linspace(0, cloud.shape[0] - 1, max_points, dtype=int)
        cloud = cloud[indices]
    return cloud


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_trajectory(path: Path, times: np.ndarray, states: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.pending")
    header = ",".join(("time", *(f"x{index + 1}" for index in range(states.shape[1]))))
    try:
        np.savetxt(
            temporary,
            np.column_stack((times, states)),
            delimiter=",",
            header=header,
            comments="",
            fmt="%.17e",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _principal_states(states: np.ndarray, periodic: Mapping[int, float]) -> np.ndarray:
    result = np.asarray(states, dtype=float).copy()
    for index, period in periodic.items():
        result[:, index] = (result[:, index] + 0.5 * period) % period - 0.5 * period
    return result


def _tail_observables(
    times: np.ndarray,
    states: np.ndarray,
    *,
    burn_time: float,
    scale: Sequence[float],
    periodic: Mapping[int, float],
) -> dict[str, Any]:
    tail = states[times >= burn_time]
    if tail.shape[0] == 0:
        tail = states[-1:]
    principal = _principal_states(tail, periodic)
    scaled = principal / np.asarray(scale, dtype=float)
    covariance = np.cov(scaled, rowvar=False) if scaled.shape[0] > 1 else np.zeros((scaled.shape[1],) * 2)
    return {
        "n_tail_samples": int(tail.shape[0]),
        "tail_centroid_scaled": np.mean(scaled, axis=0).tolist(),
        "tail_span_scaled": np.ptp(scaled, axis=0).tolist(),
        "tail_covariance_scaled": np.asarray(covariance, dtype=float).tolist(),
        "tail_radius_max_scaled": float(np.max(np.linalg.norm(scaled, axis=1))),
        "final_state": states[-1].tolist(),
    }


def _trajectory_csv_row(
    result: TrajectoryResult,
    *,
    campaign_id: str,
    contract: CaseRunContract,
) -> dict[str, Any]:
    metrics_payload = {
        "classification": result.classification.to_dict(),
        "trajectory_id": result.trajectory_id,
        "destination_id": result.classification.destination_id,
        "trajectory_artifact": result.trajectory_artifact,
        "metadata_artifact": result.metadata_artifact,
        "trajectory_sha256": result.trajectory_sha256,
        "physical_budget": asdict(result.physical_budget),
        "solver_info": result.solver_info,
        "observables": result.observables,
        "elapsed_seconds": result.elapsed_seconds,
        "order_kind": contract.order_kind,
        "q": contract.q,
        "tau_star": contract.tau_star,
        "tau_star_basis": contract.tau_star_basis,
        "finite_time_only": True,
    }
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "case_id": result.case_id,
        "system_id": result.system_id,
        "seed_id": result.seed_id,
        "contract_id": contract.contract_id,
        "budget_level": result.budget_level,
        "integrator_id": result.integrator_id,
        "history_mode": contract.history_mode,
        "classifier_id": DESTINATION_CLASSIFIER_ID,
        "window_start": result.physical_budget.burn_time,
        "window_stop": result.physical_budget.horizon,
        "integration_status": result.integration_status,
        "destination_label": result.classification.label,
        "destination_subtype": result.classification.subtype,
        "classification_confidence": result.classification.confidence,
        "is_ambiguous": result.classification.is_ambiguous,
        "metrics_json": json.dumps(json_safe(metrics_payload), sort_keys=True),
        "evidence_level": result.evidence_level,
        "trajectory_id": result.trajectory_id,
        "destination_id": result.classification.destination_id,
        "trajectory_artifact": result.trajectory_artifact,
        "metadata_artifact": result.metadata_artifact,
        "trajectory_sha256": result.trajectory_sha256,
        "order_kind": contract.order_kind,
        "q": contract.q,
        "tau_star": contract.tau_star,
        "step": result.physical_budget.step,
        "horizon": result.physical_budget.horizon,
        "n_samples": result.classification.metrics.get("n_samples", ""),
    }


def append_case_seed_bank(
    paths: CampaignArtifactPaths,
    manifest: CampaignManifest,
    case: CaseDefinition,
) -> None:
    """Append one dimension-homogeneous bank to the campaign-wide CSV."""

    if case.contract.case_id not in manifest.cases:
        raise ValueError("case is not declared by the campaign manifest.")
    existing = read_csv_rows(paths.seed_bank)
    rows = []
    for row in seed_bank_csv_rows(case.bank):
        if row["system_id"] != case.contract.system_id:
            raise ValueError("seed system_id does not match its case contract.")
        rows.append({**row, "case_id": case.contract.case_id})
    keys = {(row.get("case_id", ""), row.get("seed_id", "")) for row in existing}
    duplicates = [row["seed_id"] for row in rows if (case.contract.case_id, row["seed_id"]) in keys]
    if duplicates:
        raise ValueError(f"duplicate campaign seed identifiers: {duplicates}.")
    temporary = paths.seed_bank.with_name(f".{paths.seed_bank.name}.pending")
    try:
        write_csv(temporary, [*existing, *rows], fields=SEED_BANK_FIELDS)
        temporary.replace(paths.seed_bank)
    finally:
        temporary.unlink(missing_ok=True)


def append_trajectory_result(
    paths: CampaignArtifactPaths,
    result: TrajectoryResult,
    *,
    campaign_id: str,
    contract: CaseRunContract,
) -> None:
    """Atomically append a result and reject duplicate trajectory identifiers."""

    existing = read_csv_rows(paths.trajectory_metrics)
    for row in existing:
        payload = json.loads(row.get("metrics_json", "{}") or "{}")
        if row.get("trajectory_id") == result.trajectory_id or payload.get("trajectory_id") == result.trajectory_id:
            raise ValueError(f"trajectory {result.trajectory_id!r} is already recorded.")
    combined = [*existing, _trajectory_csv_row(result, campaign_id=campaign_id, contract=contract)]
    temporary = paths.trajectory_metrics.with_name(f".{paths.trajectory_metrics.name}.pending")
    try:
        write_csv(temporary, combined, fields=TRAJECTORY_METRIC_FIELDS)
        temporary.replace(paths.trajectory_metrics)
    finally:
        temporary.unlink(missing_ok=True)


def _integrate(
    case: CaseDefinition,
    seed: SeedRecord,
    budget: PhysicalBudget,
    solver: SolverSpec,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    system = case.system
    rhs_state = lambda state: system.evaluate(np.asarray(state, dtype=float))
    if case.contract.order_kind == "integer":
        step = budget.step * solver.step_factor
        if solver.family == "dop853":
            trajectory, status = dop853_q1_integrate(
                rhs_state,
                np.asarray(seed.state, dtype=float),
                t_final=budget.horizon,
                h=budget.step,
                rtol=float(solver.rtol or 1.0e-10),
                atol=float(solver.atol or 1.0e-12),
                max_step=step,
                div_threshold=(
                    None if case.contract.periodic_coordinates else case.contract.divergence_radius
                ),
            )
        elif solver.family == "rk4":
            fixed_step = step
            nominal_steps = int(round(budget.horizon / fixed_step))
            if not np.isclose(
                nominal_steps * fixed_step,
                budget.horizon,
                rtol=0.0,
                atol=32.0 * np.finfo(float).eps * max(1.0, budget.horizon),
            ):
                raise ValueError("fixed-step RK4 requires horizon/step to be integral.")
            times, states, status, rk4_info = rk4_integrate(
                rhs_state,
                np.asarray(seed.state, dtype=float),
                h=fixed_step,
                N=nominal_steps,
                divergence_norm=(
                    np.inf if case.contract.periodic_coordinates else case.contract.divergence_radius
                ),
            )
            return times, states, status, {
                **json_safe(rk4_info),
                "configured_step": fixed_step,
                "independent_fixed_step": True,
            }
        else:
            raise ValueError(f"unsupported integer solver family {solver.family!r}.")
        return trajectory[:, 0], trajectory[:, 1:], status, {
            "family": solver.family,
            "output_step": float(trajectory[1, 0] - trajectory[0, 0]) if len(trajectory) > 1 else step,
            "configured_step": step,
            "rtol": solver.rtol,
            "atol": solver.atol,
            "independent_fixed_step": solver.family == "rk4",
        }

    if solver.family not in {"abm", "efork3"}:
        raise ValueError(f"unsupported Caputo solver family {solver.family!r}.")
    times, states, status, info = fractional_integrate(
        rhs=lambda _time, state: system.evaluate(np.asarray(state, dtype=float)),
        x0=np.asarray(seed.state, dtype=float),
        q=case.contract.q,
        h=budget.step * solver.step_factor,
        t_final=budget.horizon,
        method=solver.family,
        memory_mode="full",
        system=system,
        use_c_backend=solver.use_native_backend,
        divergence_norm=case.contract.divergence_radius,
        return_history=True,
        allow_python_fallback=False,
        early_stop_config={"enabled": False},
    )
    if solver.use_native_backend and not bool(info.get("used_c_backend")):
        raise RuntimeError("declared native Caputo backend was not used.")
    if solver.use_native_backend and info.get("rhs_source") != "compiled_c_registry":
        raise RuntimeError("declared compiled Wu RHS was not used by the Caputo backend.")
    return times, states, status, json_safe(info)


def run_and_classify_seed(
    case: CaseDefinition,
    seed: SeedRecord,
    level: str,
    solver: SolverSpec,
    *,
    run_root: Path,
) -> TrajectoryResult:
    """Integrate one seed, persist the causal trajectory, and classify it."""

    budget = case.contract.physical_budget(level)
    trajectory_id = f"{case.contract.case_id}__{seed.seed_id}__{level}__{solver.solver_id}"
    started = time.perf_counter()
    times, states, status, solver_info = _integrate(case, seed, budget, solver)
    elapsed = time.perf_counter() - started
    classifier = case.contract.classifier(level)
    classification = classify_destination(
        times,
        states,
        contract=classifier,
        equilibria=case.equilibria,
        references=case.references,
        integration_status=status,
    )
    observables = _tail_observables(
        times,
        states,
        burn_time=budget.burn_time,
        scale=case.contract.coordinate_scale,
        periodic=case.contract.periodic_coordinates,
    )
    trajectory_path = run_root / "trajectories" / f"{trajectory_id}.csv"
    metadata_path = run_root / "metadata" / f"{trajectory_id}.json"
    _atomic_write_trajectory(trajectory_path, times, states)
    digest = _hash_file(trajectory_path)
    evidence_level = "EV-TG2" if status == "ok" and np.all(np.isfinite(states)) else "EV-TG1"
    metadata = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "case_id": case.contract.case_id,
        "system_id": case.contract.system_id,
        "seed": seed.to_dict(),
        "contract_id": case.contract.contract_id,
        "physical_budget": asdict(budget),
        "solver": asdict(solver),
        "solver_info": solver_info,
        "classification": classification.to_dict(),
        "observables": observables,
        "trajectory_artifact": str(trajectory_path.relative_to(run_root)).replace("\\", "/"),
        "trajectory_sha256": digest,
        "elapsed_seconds": elapsed,
        "evidence_level": evidence_level,
        "claims_scope": "finite-time trajectory and destination diagnostic only",
    }
    write_json(metadata_path, metadata)
    return TrajectoryResult(
        trajectory_id=trajectory_id,
        case_id=case.contract.case_id,
        system_id=case.contract.system_id,
        seed_id=seed.seed_id,
        budget_level=level,
        integrator_id=solver.solver_id,
        integration_status=status,
        classification=classification,
        trajectory_artifact=str(trajectory_path.relative_to(run_root)).replace("\\", "/"),
        metadata_artifact=str(metadata_path.relative_to(run_root)).replace("\\", "/"),
        trajectory_sha256=digest,
        physical_budget=budget,
        solver_info=solver_info,
        observables=observables,
        elapsed_seconds=elapsed,
        evidence_level=evidence_level,
    )


def _short_window_error(
    left_path: Path,
    right_path: Path,
    contract: CaseRunContract,
) -> float:
    left = np.genfromtxt(left_path, delimiter=",", names=True)
    right = np.genfromtxt(right_path, delimiter=",", names=True)
    left_names = list(left.dtype.names or ())
    right_names = list(right.dtype.names or ())
    t_left = np.atleast_1d(left[left_names[0]]).astype(float)
    t_right = np.atleast_1d(right[right_names[0]]).astype(float)
    x_left = np.column_stack([np.atleast_1d(left[name]).astype(float) for name in left_names[1:]])
    x_right = np.column_stack([np.atleast_1d(right[name]).astype(float) for name in right_names[1:]])
    dimension = len(contract.coordinate_scale)
    if x_left.shape[1] != dimension or x_right.shape[1] != dimension:
        raise ValueError(
            "trajectory dimension does not match its case contract: "
            f"left={x_left.shape[1]}, right={x_right.shape[1]}, contract={dimension}."
        )
    stop = min(10.0 * contract.tau_star, float(t_left[-1]), float(t_right[-1]))
    mask = t_left <= stop + np.finfo(float).eps
    grid = t_left[mask]
    interpolated = np.column_stack(
        [np.interp(grid, t_right, x_right[:, index]) for index in range(x_right.shape[1])]
    )
    delta = x_left[mask] - interpolated
    for index, period in contract.periodic_coordinates.items():
        delta[:, index] = (delta[:, index] + 0.5 * period) % period - 0.5 * period
    return float(np.max(np.linalg.norm(delta / np.asarray(contract.coordinate_scale), axis=1)))


def _relative_observable_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    values: list[float] = []
    for key in ("tail_span_scaled", "tail_centroid_scaled"):
        a = np.asarray(left[key], dtype=float)
        b = np.asarray(right[key], dtype=float)
        denominator = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0)
        values.append(float(np.linalg.norm(a - b) / denominator))
    a_cov = np.asarray(left["tail_covariance_scaled"], dtype=float)
    b_cov = np.asarray(right["tail_covariance_scaled"], dtype=float)
    denominator = max(float(np.linalg.norm(a_cov)), float(np.linalg.norm(b_cov)), 1.0)
    values.append(float(np.linalg.norm(a_cov - b_cov) / denominator))
    return float(max(values))


def _index_results_for_case(
    case: CaseDefinition,
    results: Sequence[TrajectoryResult],
) -> dict[tuple[str, str, str], TrajectoryResult]:
    """Index only records belonging to ``case`` and reject ambiguous keys.

    Seed identifiers are not globally unique across a campaign.  Filtering by
    ``case_id`` must therefore happen before indexing, otherwise a record from
    another case with the same seed/budget/solver tuple could overwrite the
    evidence under evaluation.  Duplicate tuples inside one case are rejected
    because choosing the last row would make the decision order-dependent.
    """

    indexed: dict[tuple[str, str, str], TrajectoryResult] = {}
    for result in results:
        if result.case_id != case.contract.case_id:
            continue
        if result.system_id != case.contract.system_id:
            raise ValueError(
                "trajectory result system_id does not match its selected case: "
                f"case={case.contract.case_id!r}, expected={case.contract.system_id!r}, "
                f"received={result.system_id!r}."
            )
        key = (result.seed_id, result.budget_level, result.integrator_id)
        if key in indexed:
            raise ValueError(
                "duplicate trajectory result within one case for "
                f"seed/budget/integrator={key!r}."
            )
        indexed[key] = result
    return indexed


def evaluate_case_evidence(
    case: CaseDefinition,
    results: Sequence[TrajectoryResult],
    *,
    run_root: Path,
) -> dict[str, Any]:
    """Evaluate EV--TG2/3 gates without promoting finite evidence to topology."""

    indexed = _index_results_for_case(case, results)
    case_results = list(indexed.values())
    seed_records: list[dict[str, Any]] = []
    case_level = "EV-TG1"
    for seed_id in sorted({result.seed_id for result in case_results}):
        checks: list[dict[str, Any]] = []
        seed_level = "EV-TG1"
        for level in ("B0", "B1", "B2"):
            primary = indexed.get((seed_id, level, case.contract.primary_solver.solver_id))
            secondary = indexed.get((seed_id, level, case.contract.secondary_solver.solver_id))
            if primary is None or secondary is None:
                continue
            same_destination = (
                primary.classification.destination_id == secondary.classification.destination_id
            )
            resolved = all(
                result.classification.label not in {"transient", "ambiguous"}
                and not result.classification.is_ambiguous
                for result in (primary, secondary)
            )
            observable_difference = _relative_observable_difference(
                primary.observables, secondary.observables
            )
            short_error = _short_window_error(
                run_root / primary.trajectory_artifact,
                run_root / secondary.trajectory_artifact,
                case.contract,
            )
            checks.append(
                {
                    "budget_level": level,
                    "primary_destination": primary.classification.destination_id,
                    "secondary_destination": secondary.classification.destination_id,
                    "same_destination": same_destination,
                    "resolved_destination": resolved,
                    "short_window_scaled_error": short_error,
                    "observable_relative_difference": observable_difference,
                    "short_error_gate": short_error <= 1.0e-3,
                    "observable_gate": observable_difference <= 0.05,
                }
            )
            if primary.evidence_level == "EV-TG2" and secondary.evidence_level == "EV-TG2":
                seed_level = "EV-TG2"
        b1 = next((item for item in checks if item["budget_level"] == "B1"), None)
        b2 = next((item for item in checks if item["budget_level"] == "B2"), None)
        horizon_destination_gate = bool(
            b1
            and b2
            and b1["same_destination"]
            and b2["same_destination"]
            and b1["primary_destination"] == b2["primary_destination"]
            and b1["resolved_destination"]
            and b2["resolved_destination"]
        )
        primary_b1 = indexed.get((seed_id, "B1", case.contract.primary_solver.solver_id))
        primary_b2 = indexed.get((seed_id, "B2", case.contract.primary_solver.solver_id))
        secondary_b1 = indexed.get((seed_id, "B1", case.contract.secondary_solver.solver_id))
        secondary_b2 = indexed.get((seed_id, "B2", case.contract.secondary_solver.solver_id))
        horizon_observable_differences = {
            "primary": (
                _relative_observable_difference(primary_b1.observables, primary_b2.observables)
                if primary_b1 is not None and primary_b2 is not None
                else None
            ),
            "secondary": (
                _relative_observable_difference(secondary_b1.observables, secondary_b2.observables)
                if secondary_b1 is not None and secondary_b2 is not None
                else None
            ),
        }
        horizon_observable_gate = bool(
            horizon_observable_differences["primary"] is not None
            and horizon_observable_differences["secondary"] is not None
            and float(horizon_observable_differences["primary"]) <= 0.05
            and float(horizon_observable_differences["secondary"]) <= 0.05
        )
        numerical_gate = bool(
            b1
            and b2
            and b1["short_error_gate"]
            and b2["short_error_gate"]
            and b1["observable_gate"]
            and b2["observable_gate"]
            and horizon_observable_gate
        )
        if horizon_destination_gate and numerical_gate:
            seed_level = "EV-TG3"
        if seed_level == "EV-TG3":
            case_level = "EV-TG3"
        elif seed_level == "EV-TG2" and case_level == "EV-TG1":
            case_level = "EV-TG2"
        seed_records.append(
            {
                "seed_id": seed_id,
                "evidence_level": seed_level,
                "checks": checks,
                "horizon_destination_gate": horizon_destination_gate,
                "horizon_observable_differences": horizon_observable_differences,
                "horizon_observable_gate": horizon_observable_gate,
                "numerical_convergence_gate": numerical_gate,
            }
        )
    return {
        "case_id": case.contract.case_id,
        "system_id": case.contract.system_id,
        "evidence_level": case_level,
        "aggregation_policy": (
            "maximum evidence level attained by any evaluated seed route; "
            "this is not an all-seeds case certificate"
        ),
        "records": seed_records,
        "claim_limit": (
            "EV-TG4 is not assigned: no complete neighborhood survey, no three non-collinear "
            "dynamic edge brackets, and no evolved/rebracketed edge trajectory"
        ),
    }


def compute_central_symmetry_diagnostics(
    case: CaseDefinition,
    results: Sequence[TrajectoryResult],
    *,
    run_root: Path,
    pairs: Sequence[tuple[str, str]],
    levels: Sequence[str] = ("B0", "B2"),
) -> dict[str, Any]:
    """Check ``x_-(t)=-x_+(t)`` under identical solver contracts."""

    indexed = _index_results_for_case(case, results)
    records: list[dict[str, Any]] = []
    for plus_id, minus_id in pairs:
        for level in levels:
            for solver in _case_solver_specs(case):
                plus = indexed.get((plus_id, level, solver.solver_id))
                minus = indexed.get((minus_id, level, solver.solver_id))
                if plus is None or minus is None:
                    continue
                t_plus, x_plus = _load_result_trajectory(run_root, plus)
                t_minus, x_minus = _load_result_trajectory(run_root, minus)
                if t_plus.shape != t_minus.shape or not np.allclose(
                    t_plus, t_minus, rtol=0.0, atol=32.0 * np.finfo(float).eps
                ):
                    raise ValueError("central-symmetry trajectories must share a time grid.")
                residual = float(
                    np.max(
                        np.linalg.norm(
                            (x_plus + x_minus) / np.asarray(case.contract.coordinate_scale),
                            axis=1,
                        )
                    )
                )
                records.append(
                    {
                        "plus_seed_id": plus_id,
                        "minus_seed_id": minus_id,
                        "budget_level": level,
                        "integrator_id": solver.solver_id,
                        "maximum_scaled_odd_symmetry_residual": residual,
                        "passes_1e-10": residual <= 1.0e-10,
                    }
                )
    return {
        "symmetry": "central_inversion",
        "records": records,
        "all_pass_1e-10": bool(records) and all(record["passes_1e-10"] for record in records),
        "scope": "numerical covariance diagnostic on paired reset/initial-state trajectories",
    }


def _make_record(
    *,
    seed_id: str,
    system_id: str,
    route: str,
    state: Sequence[float],
    parameter_set_id: str,
    source_artifact: str,
    source_record_id: str,
    priority: int,
    order_kind: str = "integer",
    q: float = 1.0,
    lower_terminal: float | None = None,
    initial_time: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SeedRecord:
    return SeedRecord(
        seed_id=seed_id,
        system_id=system_id,
        route=route,  # type: ignore[arg-type]
        state=tuple(float(value) for value in state),
        order_kind=order_kind,  # type: ignore[arg-type]
        q=q,
        parameter_set_id=parameter_set_id,
        lower_terminal=lower_terminal,
        initial_time=initial_time,
        source_artifact=source_artifact,
        source_record_id=source_record_id,
        priority=priority,
        metadata={} if metadata is None else metadata,
    )


def build_pll_case(project_root: Path = PROJECT_ROOT, *, tau_star: float) -> CaseDefinition:
    system = pll_lead_lag_2015_system()
    parameters = dict(system.parameters)
    scale = (parameters["tau1"] / 2.0, float(np.pi))
    p_plus = pll_original_to_shifted(np.array([parameters["tau1"] / 2.0, np.pi / 2.0]))
    p_minus = pll_original_to_shifted(np.array([-parameters["tau1"] / 2.0, 3.0 * np.pi / 2.0]))
    separator = np.array([-0.010493481203325857, -0.7974826998815102])
    records = (
        _make_record(
            seed_id="pll_pp_plus",
            system_id=system.name,
            route="perpetual_point",
            state=p_plus,
            parameter_set_id="pll_bianchi2015",
            source_artifact="validation/06_geometric_topological_campaign/wolfram/geometric_topological_engine_validation_summary.json",
            source_record_id="PLL_PP_positive",
            priority=10,
        ),
        _make_record(
            seed_id="pll_pp_minus",
            system_id=system.name,
            route="perpetual_point",
            state=p_minus,
            parameter_set_id="pll_bianchi2015",
            source_artifact="validation/06_geometric_topological_campaign/wolfram/geometric_topological_engine_validation_summary.json",
            source_record_id="PLL_PP_negative",
            priority=11,
        ),
        _make_record(
            seed_id="pll_cycle_seed",
            system_id=system.name,
            route="continuation",
            state=(-0.011120535749959005, -0.7974826998815102),
            parameter_set_id="pll_bianchi2015",
            source_artifact="validation/reference_cases/pll_lead_lag_integer_q1_full/02_cycle_localization.json",
            source_record_id="stable_target_shifted_seed",
            priority=20,
        ),
        _make_record(
            seed_id="pll_separator_reference",
            system_id=system.name,
            route="continuation",
            state=separator,
            parameter_set_id="pll_bianchi2015",
            source_artifact="validation/reference_cases/pll_lead_lag_integer_q1_full/02_cycle_localization.json",
            source_record_id="unstable_cycle_shifted_seed",
            priority=30,
        ),
        _make_record(
            seed_id="pll_edge_left",
            system_id=system.name,
            route="edge_tracking",
            state=separator + np.array([-1.0e-5, 0.0]),
            parameter_set_id="pll_bianchi2015",
            source_artifact="geometric_topological_initial_pilot_v1",
            source_record_id="predeclared_separator_bracket_left",
            priority=40,
        ),
        _make_record(
            seed_id="pll_edge_right",
            system_id=system.name,
            route="edge_tracking",
            state=separator + np.array([1.0e-5, 0.0]),
            parameter_set_id="pll_bianchi2015",
            source_artifact="geometric_topological_initial_pilot_v1",
            source_record_id="predeclared_separator_bracket_right",
            priority=41,
        ),
    )
    bank = build_seed_bank(
        records,
        coordinate_scale=scale,
        periodic_coordinates={1: 2.0 * np.pi},
        absolute_tolerance=1.0e-10,
        relative_tolerance=1.0e-8,
    )
    reference_path = (
        project_root
        / "validation"
        / "reference_cases"
        / "pll_lead_lag_integer_q1_full"
        / "03_stable_reference_cycle.csv"
    )
    contract = CaseRunContract(
        case_id="pll_lead_lag_integer_q1_tg_pilot",
        system_id=system.name,
        parameter_set_id="pll_bianchi2015",
        order_kind="integer",
        q=1.0,
        parameters=parameters,
        tau_star=tau_star,
        tau_star_basis=(
            "minimum of inverse maximum nonzero spectral rate and the predeclared-bank "
            "scaled crossing time; the PLL perpetual-point crossing time is active"
        ),
        coordinate_scale=scale,
        periodic_coordinates={1: 2.0 * np.pi},
        divergence_radius=120.0,
        divergence_radius_kind="scaled",
        classifier_overrides={
            "equilibrium_tolerance": 1.0e-5,
            "reference_distance_max": 0.12,
            "reference_separation_margin": 0.03,
            "periodic_return_error_max": 0.12,
        },
        reference_artifacts={"running_cycle": str(reference_path.relative_to(project_root))},
    )
    return CaseDefinition(
        contract=contract,
        system=system,
        bank=bank,
        references={"running_cycle": _load_reference_cloud(reference_path)},
        equilibria=system.equilibrium_points(),
        budget_seed_ids={
            "B0": ("pll_pp_plus", "pll_pp_minus", "pll_cycle_seed", "pll_edge_left", "pll_edge_right"),
            "B1": tuple(record.seed_id for record in records),
            "B2": ("pll_pp_plus", "pll_pp_minus"),
        },
    )


def build_mavpd_case(project_root: Path = PROJECT_ROOT, *, tau_star: float) -> CaseDefinition:
    parameters = {"gamma": 0.1, "delta": 100.0, "rho": 200.0, "xi": 3.1}
    system = mavpd_2023_system(parameters)
    # Frozen rounded semi-amplitudes of the promoted outer reference cloud.
    scale = (0.6, 0.15, 2.4)
    gamma, delta, rho, xi = (parameters[name] for name in ("gamma", "delta", "rho", "xi"))
    u = math.sqrt(gamma / 3.0)
    v = 2.0 * delta * gamma * u / (3.0 * (rho - delta))
    pp = np.array([u, v, u - xi * v])
    df0 = np.array([0.4315886850827659, 0.01713475910344968, 0.8631773701655315])
    df1 = np.array([0.6849613617742147, 0.17252741732512492, 1.3699227235484286])
    records: tuple[SeedRecord, ...] = tuple(
        _make_record(
            seed_id=seed_id,
            system_id=system.name,
            route=route,
            state=state,
            parameter_set_id="mavpd_xi3p1",
            source_artifact=artifact,
            source_record_id=source_id,
            priority=priority,
            metadata={"symmetry": sign},
        )
        for seed_id, route, state, artifact, source_id, priority, sign in (
            (
                "mavpd_pp_plus",
                "perpetual_point",
                pp,
                "validation/06_geometric_topological_campaign/wolfram/geometric_topological_engine_validation_summary.json",
                "MAVPD_PP_positive_xi3p1",
                10,
                "+",
            ),
            (
                "mavpd_pp_minus",
                "perpetual_point",
                -pp,
                "validation/06_geometric_topological_campaign/wolfram/geometric_topological_engine_validation_summary.json",
                "MAVPD_PP_negative_xi3p1",
                11,
                "-",
            ),
            (
                "mavpd_df0_plus",
                "describing_function",
                df0,
                "validation/reference_cases/mavpd_integer_q1_audit/01_primary_direct_branches.json",
                "primary_branch_0_positive",
                20,
                "+",
            ),
            (
                "mavpd_df0_minus",
                "describing_function",
                -df0,
                "validation/reference_cases/mavpd_integer_q1_audit/01_primary_direct_branches.json",
                "primary_branch_0_negative",
                21,
                "-",
            ),
            (
                "mavpd_df1_plus",
                "describing_function",
                df1,
                "validation/reference_cases/mavpd_integer_q1_audit/01_primary_direct_branches.json",
                "primary_branch_1_positive",
                30,
                "+",
            ),
            (
                "mavpd_df1_minus",
                "describing_function",
                -df1,
                "validation/reference_cases/mavpd_integer_q1_audit/01_primary_direct_branches.json",
                "primary_branch_1_negative",
                31,
                "-",
            ),
        )
    )
    bank = build_seed_bank(
        records,
        coordinate_scale=scale,
        symmetries=(SymmetryTransform.inversion(3),),
        symmetry_group_is_complete=True,
        absolute_tolerance=1.0e-10,
        relative_tolerance=1.0e-8,
    )
    base = project_root / "validation" / "reference_cases" / "mavpd_integer_q1_audit"
    reference_paths = {
        "inner_positive": base / "primary_candidates" / "branch_0_phase_0.0pi.csv",
        "inner_negative": base / "primary_candidates" / "branch_0_phase_1.0pi.csv",
        "outer_cycle": base / "04_primary_outer_candidate.csv",
    }
    contract = CaseRunContract(
        case_id="mavpd_integer_q1_xi31_tg_pilot",
        system_id=system.name,
        parameter_set_id="mavpd_xi3p1",
        order_kind="integer",
        q=1.0,
        parameters=parameters,
        tau_star=tau_star,
        tau_star_basis=(
            "minimum of inverse maximum nonzero equilibrium spectral rate and the "
            "predeclared-bank scaled crossing time; the spectral time is active"
        ),
        coordinate_scale=scale,
        divergence_radius=120.0,
        divergence_radius_kind="scaled",
        classifier_overrides={
            "reference_distance_max": 0.16,
            "reference_separation_margin": 0.04,
        },
        reference_artifacts={
            name: str(path.relative_to(project_root)) for name, path in reference_paths.items()
        },
    )
    return CaseDefinition(
        contract=contract,
        system=system,
        bank=bank,
        references={name: _load_reference_cloud(path) for name, path in reference_paths.items()},
        equilibria=system.equilibrium_points(),
        budget_seed_ids={
            "B0": ("mavpd_pp_plus", "mavpd_df0_plus", "mavpd_df1_plus"),
            "B1": tuple(record.seed_id for record in records),
            "B2": (
                "mavpd_pp_plus",
                "mavpd_pp_minus",
                "mavpd_df1_plus",
                "mavpd_df1_minus",
            ),
        },
    )


def build_wu_case(project_root: Path = PROJECT_ROOT, *, tau_star: float) -> CaseDefinition:
    system = get_system("fractional_chua_arctan_wu2023")
    parameters = dict(system.parameters)
    system_id = system.name
    pp = np.array([0.3369817810952544663, 0.08157492826864398825, -0.2549832342782550564])
    fdf1 = np.array([0.901922573917310, 0.06817948792356945, -1.294468839020390])
    fdf2 = np.array([3.208954181208862, 2.218015570083137, -4.725989229133311])
    idf1 = np.array([0.879076675356472, 0.0554410403333561, -1.2559160831542])
    idf2 = np.array([4.191398823569484, 3.407138582076569, -6.004495052865678])
    published = np.array([13.8, 0.7093, -19.8768])
    specifications = (
        ("wu_fpp_plus", "fractional_perpetual_point", pp, "Chua_arctan_Wu_PP_positive", 10, "+"),
        ("wu_fpp_minus", "fractional_perpetual_point", -pp, "Chua_arctan_Wu_PP_negative", 11, "-"),
        ("wu_fdf1_plus", "describing_function", fdf1, "fractional_branch_1_positive", 20, "+"),
        ("wu_fdf1_minus", "describing_function", -fdf1, "fractional_branch_1_negative", 21, "-"),
        ("wu_fdf2_plus", "describing_function", fdf2, "fractional_branch_2_positive", 30, "+"),
        ("wu_fdf2_minus", "describing_function", -fdf2, "fractional_branch_2_negative", 31, "-"),
        ("wu_idf1_plus", "describing_function", idf1, "integer_transfer_branch_1_positive", 40, "+"),
        ("wu_idf1_minus", "describing_function", -idf1, "integer_transfer_branch_1_negative", 41, "-"),
        ("wu_idf2_plus", "describing_function", idf2, "integer_transfer_branch_2_positive", 50, "+"),
        ("wu_idf2_minus", "describing_function", -idf2, "integer_transfer_branch_2_negative", 51, "-"),
        ("wu_published_adm_plus", "imported", published, "published_ADM_positive", 60, "+"),
        ("wu_published_adm_minus", "imported", -published, "published_ADM_negative", 61, "-"),
    )
    records = tuple(
        _make_record(
            seed_id=seed_id,
            system_id=system_id,
            route=route,
            state=state,
            parameter_set_id="wu2023_q0p99",
            source_artifact=(
                "validation/06_geometric_topological_campaign/wolfram/topologia_geometria_hidden_validation.txt"
                if route == "fractional_perpetual_point"
                else "validation/outputs/wolfram/chua_fractional_arctan/chua_fractional_arctan_seed_data.json"
                if seed_id.startswith("wu_fdf")
                else "validation/reference_cases/fractional_chua_arctan_wu2023/02_lure_df/centered_seeds.json"
                if seed_id.startswith("wu_idf")
                else "validation/reference_cases/fractional_chua_arctan_wu2023/reproducibility.yaml"
            ),
            source_record_id=source_id,
            priority=priority,
            order_kind="caputo",
            q=0.99,
            lower_terminal=0.0,
            initial_time=0.0,
            metadata={
                "symmetry": sign,
                "initialization_semantics": "new Caputo reset at a=t0=0",
                "published_ADM_transfer_warning": seed_id.startswith("wu_published_adm"),
            },
        )
        for seed_id, route, state, source_id, priority, sign in specifications
    )
    scale = (14.0, 3.5, 20.0)
    bank = build_seed_bank(
        records,
        coordinate_scale=scale,
        symmetries=(SymmetryTransform.inversion(3),),
        symmetry_group_is_complete=True,
        absolute_tolerance=1.0e-10,
        relative_tolerance=1.0e-8,
    )
    contract = CaseRunContract(
        case_id="chua_arctan_wu_caputo_q099_tg_pilot",
        system_id=system_id,
        parameter_set_id="wu2023_q0p99",
        order_kind="caputo",
        q=0.99,
        parameters=parameters,
        tau_star=tau_star,
        tau_star_basis=(
            "minimum of max|lambda|^(-1/q) and the predeclared core-bank Caputo crossing "
            "time; the spectral time is active and all runs retain full memory"
        ),
        coordinate_scale=scale,
        divergence_radius=120.0,
        divergence_radius_kind="absolute",
        lower_terminal=0.0,
        initial_time=0.0,
        history_mode="caputo_full_memory_reset",
        primary_solver=SolverSpec(
            "abm_pece_full_memory_native", "abm", full_memory=True, use_native_backend=True
        ),
        secondary_solver=SolverSpec(
            "efork3_full_memory_native", "efork3", full_memory=True, use_native_backend=True
        ),
        classifier_overrides={
            "reference_distance_max": 0.16,
            "reference_separation_margin": 0.04,
        },
        reference_artifacts={},
    )
    return CaseDefinition(
        contract=contract,
        system=system,
        bank=bank,
        references={},
        equilibria=system.equilibrium_points(),
        budget_seed_ids={
            "B0": (
                "wu_fpp_plus",
                "wu_fpp_minus",
                "wu_fdf1_plus",
                "wu_fdf1_minus",
                "wu_fdf2_plus",
                "wu_fdf2_minus",
                "wu_published_adm_plus",
                "wu_published_adm_minus",
            ),
            "B1": (
                "wu_fpp_plus",
                "wu_fdf1_plus",
                "wu_fdf2_plus",
                "wu_idf1_plus",
                "wu_idf2_plus",
                "wu_published_adm_plus",
            ),
            "B2": (
                "wu_fpp_plus",
                "wu_fpp_minus",
                "wu_fdf2_plus",
                "wu_idf2_plus",
                "wu_published_adm_plus",
            ),
        },
    )


def _case_solver_specs(case: CaseDefinition) -> tuple[SolverSpec, SolverSpec]:
    return case.contract.primary_solver, case.contract.secondary_solver


def _edge_evaluator(case: CaseDefinition, run_root: Path) -> Callable[[np.ndarray, EdgeEvaluationContext], Any]:
    def evaluate(coordinate: np.ndarray, context: EdgeEvaluationContext) -> Any:
        pseudo_seed = _make_record(
            seed_id=f"edge_eval_{context.phase}_{context.iteration}_{context.ambiguity_index}",
            system_id=case.contract.system_id,
            route="edge_tracking",
            state=coordinate,
            parameter_set_id=case.contract.parameter_set_id,
            source_artifact="runtime_edge_evaluation",
            source_record_id=context.bracket_id,
            priority=100,
            order_kind=case.contract.order_kind,
            q=case.contract.q,
            lower_terminal=case.contract.lower_terminal,
            initial_time=case.contract.initial_time,
        )
        budget = case.contract.physical_budget(context.budget_level)
        times, states, status, info = _integrate(
            case, pseudo_seed, budget, case.contract.primary_solver
        )
        classification = classify_destination(
            times,
            states,
            contract=case.contract.classifier(context.budget_level),
            equilibria=case.equilibria,
            references=case.references,
            integration_status=status,
        )
        outcome = edge_destination_from_classification(
            classification,
            integration_status=status,
            evaluation_ok=(status == "ok" and np.all(np.isfinite(states))),
        )
        return replace(
            outcome,
            metadata={
                **dict(outcome.metadata),
                "solver_info": info,
                "physical_budget": asdict(budget),
                "finite_time_only": True,
            },
        )

    return evaluate


def _run_declared_edges(
    cases: Mapping[str, CaseDefinition],
    paths: CampaignArtifactPaths,
    *,
    campaign_id: str,
    run_root: Path,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    pll = cases["pll_lead_lag_integer_q1_tg_pilot"]
    geometry_pll = ScaledCylindricalGeometry(
        scale=pll.contract.coordinate_scale,
        periodic_indices=(1,),
        periods=(2.0 * np.pi,),
    )
    result = run_edge_tracking_and_record(
        pll.record("pll_edge_left").state,
        pll.record("pll_edge_right").state,
        evaluator=_edge_evaluator(pll, run_root),
        geometry=geometry_pll,
        bracket_id="pll_separator_initial_data_bracket",
        context=EdgeRunContext(
            campaign_id=campaign_id,
            case_id=pll.contract.case_id,
            system_id=pll.contract.system_id,
            contract_id=pll.contract.contract_id,
            budget_level="B2",
            integrator_id=pll.contract.primary_solver.solver_id,
            history_mode="not_applicable",
            classifier_id=DESTINATION_CLASSIFIER_ID,
            seed_left_id="pll_edge_left",
            seed_right_id="pll_edge_right",
            evidence_level="EV-TG2",
        ),
        paths=paths,
        config=EdgeTrackingConfig(
            tolerance=1.0e-8,
            max_iterations=24,
            confirmation_levels=("B1", "B2"),
            tracking_level="B2",
            data_semantics="ode_initial_state",
        ),
    )
    records["pll"] = json_safe(asdict(result))

    mavpd = cases["mavpd_integer_q1_xi31_tg_pilot"]
    geometry_mavpd = ScaledEuclideanGeometry(scale=mavpd.contract.coordinate_scale)
    result_mavpd = run_edge_tracking_and_record(
        mavpd.record("mavpd_pp_plus").state,
        mavpd.record("mavpd_df1_plus").state,
        evaluator=_edge_evaluator(mavpd, run_root),
        geometry=geometry_mavpd,
        bracket_id="mavpd_pp_to_df1_initial_data_bracket",
        context=EdgeRunContext(
            campaign_id=campaign_id,
            case_id=mavpd.contract.case_id,
            system_id=mavpd.contract.system_id,
            contract_id=mavpd.contract.contract_id,
            budget_level="B2",
            integrator_id=mavpd.contract.primary_solver.solver_id,
            history_mode="not_applicable",
            classifier_id=DESTINATION_CLASSIFIER_ID,
            seed_left_id="mavpd_pp_plus",
            seed_right_id="mavpd_df1_plus",
            evidence_level="EV-TG2",
        ),
        paths=paths,
        config=EdgeTrackingConfig(
            tolerance=1.0e-8,
            max_iterations=28,
            confirmation_levels=("B1", "B2"),
            tracking_level="B2",
            data_semantics="ode_initial_state",
        ),
    )
    records["mavpd"] = json_safe(asdict(result_mavpd))
    return records


def _load_result_trajectory(run_root: Path, result: TrajectoryResult) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(run_root / result.trajectory_artifact, delimiter=",", names=True)
    names = list(data.dtype.names or ())
    times = np.atleast_1d(data[names[0]]).astype(float)
    states = np.column_stack([np.atleast_1d(data[name]).astype(float) for name in names[1:]])
    return times, states


def _save_figure(fig: plt.Figure, output: Path, *, title: str, sources: Sequence[str]) -> dict[str, Any]:
    pdf, png = save_figure_pair_local(fig, output, dpi=220)
    plt.close(fig)
    return {
        "id": output.name,
        "title": title,
        "png": str(png.name),
        "pdf": str(pdf.name),
        "sources": list(sources),
        "claims_scope": "pedagogical diagram or finite-time numerical evidence",
    }


def render_campaign_route_figure(*, run_root: Path) -> dict[str, Any]:
    """Render only the conceptual route figure for report synchronization."""

    output = run_root / "figures"
    fig, ax = plt.subplots(figsize=(12.0, 3.5))
    ax.set_axis_off()
    boxes = (
        (0.015, "Geometría algebraica\n$J_f f=0$\n$\\det J_f=0$"),
        (0.265, "Banco unificado\nPP/FPP · DF\ncontinuación"),
        (0.515, "PVI causal\nODE o Caputo\nmemoria completa"),
        (0.765, "Destino y borde\nB0--B2\ndos solucionadores"),
    )
    for x, text_value in boxes:
        ax.add_patch(
            plt.Rectangle((x, 0.30), 0.20, 0.40, facecolor="#eaf2f8", edgecolor="#1f4e79", lw=1.5)
        )
        ax.text(x + 0.10, 0.5, text_value, ha="center", va="center", fontsize=8.5)
    for x in (0.22, 0.47, 0.72):
        ax.annotate("", xy=(x + 0.04, 0.5), xytext=(x, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.text(
        0.5,
        0.12,
        (
            "Cada flecha indica la configuración numérica, la procedencia y el nivel de evidencia; "
            "ninguna semilla prueba ocultedad."
        ),
        ha="center",
        fontsize=9,
    )
    return _save_figure(
        fig,
        output / "00_ruta_geometrico_topologica",
        title="Ruta complementaria geometrico-topologica",
        sources=("PILOT_PROTOCOL_VERSION",),
    )


def render_campaign_figures(
    cases: Mapping[str, CaseDefinition],
    results: Sequence[TrajectoryResult],
    *,
    run_root: Path,
    edge_records: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Render conceptual and finite-time result figures in PDF and PNG."""

    output = run_root / "figures"
    manifest: list[dict[str, Any]] = [render_campaign_route_figure(run_root=run_root)]

    by_key = {(r.case_id, r.seed_id, r.budget_level, r.integrator_id): r for r in results}
    pll = cases["pll_lead_lag_integer_q1_tg_pilot"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ref = pll.references["running_cycle"]
    axes[0].plot(ref[:, 0], (ref[:, 1] + np.pi) % (2.0 * np.pi) - np.pi, color="0.65", lw=0.8, label="ciclo de referencia")
    for seed_id, color in (("pll_pp_plus", "#d62728"), ("pll_pp_minus", "#1f77b4")):
        result = by_key.get((pll.contract.case_id, seed_id, "B2", pll.contract.primary_solver.solver_id))
        if result:
            _, states = _load_result_trajectory(run_root, result)
            axes[0].plot(states[:, 0], (states[:, 1] + np.pi) % (2.0 * np.pi) - np.pi, color=color, lw=0.9, label=seed_id)
        point = np.asarray(pll.record(seed_id).state)
        axes[0].scatter(point[0], (point[1] + np.pi) % (2.0 * np.pi) - np.pi, color=color, s=35, zorder=5)
    axes[0].set(xlabel="$u$", ylabel="$v$ (mod $2\\pi$)", title="PLL: rutas desde puntos perpetuos")
    axes[0].legend(fontsize=7)
    pll_edge = edge_records.get("pll", {})
    iterations = pll_edge.get("iterations", [])
    if iterations:
        axes[1].semilogy(
            [item["iteration"] for item in iterations],
            [item["width_after"] for item in iterations],
            marker="o",
            ms=3,
        )
        axes[1].axhline(1.0e-8, color="k", ls="--", lw=0.8, label="tolerancia")
        axes[1].set(xlabel="iteración", ylabel="ancho escalado", title="Bisección inicial del borde")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(fontsize=8)
    else:
        axes[1].set_axis_off()
        axes[1].text(
            0.5,
            0.70,
            "Bracket PLL rechazado antes de bisecar",
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
            transform=axes[1].transAxes,
        )
        axes[1].text(
            0.5,
            0.46,
            "B1: ambos extremos transitorios\n"
            "B2: extremo izquierdo transitorio; derecho $E_{focus}$\n\n"
            f"ancho conservado = {float(pll_edge.get('final_width', np.nan)):.3e}",
            ha="center",
            va="center",
            fontsize=11,
            transform=axes[1].transAxes,
        )
        axes[1].text(
            0.5,
            0.18,
            "Resultado informativo: hace falta ampliar el horizonte\n"
            "o escoger otro extremo corredor ya resuelto.",
            ha="center",
            va="center",
            fontsize=9,
            color="#7f1d1d",
            transform=axes[1].transAxes,
        )
    manifest.append(
        _save_figure(
            fig,
            output / "01_pll_pp_y_borde",
            title="PLL: puntos perpetuos y biseccion inicial de frontera",
            sources=("trajectory_metrics.csv", "edge_brackets.csv", "03_stable_reference_cycle.csv"),
        )
    )

    mavpd = cases["mavpd_integer_q1_xi31_tg_pilot"]
    fig = plt.figure(figsize=(11.5, 4.8), constrained_layout=True)
    ax3 = fig.add_subplot(121, projection="3d")
    for name, color in (("inner_positive", "#1f77b4"), ("inner_negative", "#17becf"), ("outer_cycle", "#7f7f7f")):
        cloud = mavpd.references[name][:: max(1, len(mavpd.references[name]) // 1200)]
        ax3.plot(cloud[:, 0], cloud[:, 1], cloud[:, 2], color=color, lw=0.45, alpha=0.65, label=name)
    for seed_id, marker, color in (("mavpd_pp_plus", "o", "#d62728"), ("mavpd_df1_plus", "^", "#2ca02c")):
        point = np.asarray(mavpd.record(seed_id).state)
        ax3.scatter(*point, marker=marker, color=color, s=45, label=seed_id)
    ax3.set(xlabel="$y_1$", ylabel="$y_2$", zlabel="$y_3$", title="MAVPD: banco y destinos de referencia")
    ax3.legend(fontsize=6)
    ax = fig.add_subplot(122)
    u_surface = math.sqrt(mavpd.contract.parameters["gamma"] / 3.0)
    ax.axvline(u_surface, color="#d62728", ls="--", label="$y_1=+\\sqrt{\\gamma/3}$")
    ax.axvline(-u_surface, color="#1f77b4", ls="--", label="$y_1=-\\sqrt{\\gamma/3}$")
    for seed_id in ("mavpd_pp_plus", "mavpd_pp_minus", "mavpd_df0_plus", "mavpd_df1_plus"):
        point = np.asarray(mavpd.record(seed_id).state)
        ax.scatter(point[0], point[2], s=38, label=seed_id)
    ax.set(xlabel="$y_1$", ylabel="$y_3$", title="Superficies críticas y semillas")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6)
    manifest.append(
        _save_figure(
            fig,
            output / "02_mavpd_pp_superficie_destinos",
            title="MAVPD: puntos perpetuos, superficie critica y destinos",
            sources=("trajectory_metrics.csv", "MAVPD reference clouds"),
        )
    )

    wu = cases["chua_arctan_wu_caputo_q099_tg_pilot"]
    fig = plt.figure(figsize=(11.5, 4.8))
    ax3 = fig.add_subplot(121, projection="3d")
    for method, color in ((wu.contract.primary_solver.solver_id, "#d62728"), (wu.contract.secondary_solver.solver_id, "#1f77b4")):
        result = by_key.get((wu.contract.case_id, "wu_fpp_plus", "B2", method))
        if result:
            times, states = _load_result_trajectory(run_root, result)
            mask = times >= result.physical_budget.burn_time
            tail = states[mask]
            ax3.plot(tail[:: max(1, len(tail) // 4000), 0], tail[:: max(1, len(tail) // 4000), 1], tail[:: max(1, len(tail) // 4000), 2], color=color, lw=0.65, label=method)
    pp_point = np.asarray(wu.record("wu_fpp_plus").state)
    ax3.scatter(*pp_point, color="k", s=45, marker="*", label="FPP-A/PP")
    ax3.set(xlabel="$x$", ylabel="$y$", zlabel="$z$", title="Wu $q=.99$: cola desde FPP-A")
    ax3.legend(fontsize=6)
    ax = fig.add_subplot(122)
    levels = ("B0", "B1", "B2")
    for method, marker, color in ((wu.contract.primary_solver.solver_id, "o", "#d62728"), (wu.contract.secondary_solver.solver_id, "s", "#1f77b4")):
        spans = []
        for level in levels:
            result = by_key.get((wu.contract.case_id, "wu_fpp_plus", level, method))
            spans.append(float(np.linalg.norm(result.observables["tail_span_scaled"])) if result else np.nan)
        ax.plot(levels, spans, marker=marker, color=color, label=method)
    ax.set(xlabel="presupuesto", ylabel="norma del span escalado", title="Convergencia finita de la cola")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    manifest.append(
        _save_figure(
            fig,
            output / "03_wu_fpp_caputo_b0_b2",
            title="Wu q=.99: FPP-A bajo memoria Caputo completa",
            sources=("trajectory_metrics.csv", "Caputo ABM and EFORK full-memory trajectories"),
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0), constrained_layout=True)
    label_codes = {"equilibrium": 0, "periodic": 1, "recurrent": 2, "transient": 3, "ambiguous": 4, "escape": 5}
    cmap = plt.get_cmap("tab10", 6)
    for ax, case in zip(axes, cases.values()):
        case_results = [result for result in results if result.case_id == case.contract.case_id]
        seed_ids = sorted({result.seed_id for result in case_results})
        columns = [(level, solver.solver_id) for level in ("B0", "B1", "B2") for solver in _case_solver_specs(case)]
        matrix = np.full((len(seed_ids), len(columns)), np.nan)
        for row, seed_id in enumerate(seed_ids):
            for column, (level, solver_id) in enumerate(columns):
                result = by_key.get((case.contract.case_id, seed_id, level, solver_id))
                if result:
                    matrix[row, column] = label_codes[result.classification.label]
        ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=5.5)
        short_labels = [
            seed.replace("pll_", "").replace("mavpd_", "").replace("wu_", "")
            for seed in seed_ids
        ]
        ax.set_yticks(range(len(seed_ids)), short_labels, fontsize=5)
        ax.set_xticks(range(len(columns)), [f"{level}\n{'P' if solver == case.contract.primary_solver.solver_id else 'S'}" for level, solver in columns], fontsize=7)
        ax.set_title(case.contract.case_id.replace("_tg_pilot", ""), fontsize=8)
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=cmap(code), markersize=8, label=label) for label, code in label_codes.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7)
    manifest.append(
        _save_figure(
            fig,
            output / "04_matriz_destinos_b0_b2",
            title="Matriz de destinos por semilla, presupuesto y solver",
            sources=("trajectory_metrics.csv",),
        )
    )

    write_json(
        run_root / "figures_manifest.json",
        {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "figures": manifest,
            "result_figures_are_finite_time_only": True,
        },
    )
    return manifest


def _write_generated_tex(
    run_root: Path,
    case_decisions: Mapping[str, Any],
    edge_records: Mapping[str, Any],
) -> Path:
    output = run_root / "report" / "tg_pilot_results.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    pll_edge = edge_records.get("pll", {})
    mavpd_edge = edge_records.get("mavpd", {})
    wu_decision = case_decisions.get("chua_arctan_wu_caputo_q099_tg_pilot", {})
    run_id_tex = run_root.name.replace("_", "\\_")
    pll_status_tex = str(pll_edge.get("status", "not_run")).replace("_", "\\_")
    mavpd_status_tex = str(mavpd_edge.get("status", "not_run")).replace("_", "\\_")
    lines = [
        "% Generated by hidden_attractors.workflows.geometric_topological_pilot",
        f"\\newcommand{{\\TGRunID}}{{\\texttt{{{run_id_tex}}}}}",
        f"\\newcommand{{\\TGPLLEvidence}}{{{case_decisions['pll_lead_lag_integer_q1_tg_pilot']['evidence_level']}}}",
        f"\\newcommand{{\\TGMAVPDEvidence}}{{{case_decisions['mavpd_integer_q1_xi31_tg_pilot']['evidence_level']}}}",
        f"\\newcommand{{\\TGWuEvidence}}{{{wu_decision.get('evidence_level', 'EV-TG1')}}}",
        f"\\newcommand{{\\TGPLLEdgeStatus}}{{\\texttt{{{pll_status_tex}}}}}",
        f"\\newcommand{{\\TGPLLEdgeWidth}}{{{float(pll_edge.get('final_width', float('nan'))):.6e}}}",
        f"\\newcommand{{\\TGMAVPDEdgeStatus}}{{\\texttt{{{mavpd_status_tex}}}}}",
        f"\\newcommand{{\\TGMAVPDEdgeWidth}}{{{float(mavpd_edge.get('final_width', float('nan'))):.6e}}}",
        "",
    ]
    temporary = output.with_name(f".{output.name}.pending")
    try:
        temporary.write_text("\n".join(lines), encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _default_tau_stars(overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    return {
        "pll": 7.3247672e-3,
        "mavpd": 4.262239251e-2,
        "wu": 2.704709949e-1,
        **dict(overrides or {}),
    }


def _build_default_cases(
    project_root: Path,
    tau_stars: Mapping[str, float] | None = None,
) -> tuple[CaseDefinition, ...]:
    tau = _default_tau_stars(tau_stars)
    return (
        build_pll_case(project_root, tau_star=tau["pll"]),
        build_mavpd_case(project_root, tau_star=tau["mavpd"]),
        build_wu_case(project_root, tau_star=tau["wu"]),
    )


def _finalize_pilot(
    *,
    run_root: Path,
    manifest: CampaignManifest,
    cases: Sequence[CaseDefinition],
    results: Sequence[TrajectoryResult],
) -> None:
    paths = CampaignArtifactPaths.under(run_root)
    case_map = {case.contract.case_id: case for case in cases}
    case_decisions = {
        case.contract.case_id: evaluate_case_evidence(case, results, run_root=run_root)
        for case in cases
    }
    case_by_id = {case.contract.case_id: case for case in cases}
    case_decisions["mavpd_integer_q1_xi31_tg_pilot"]["central_symmetry_diagnostic"] = (
        compute_central_symmetry_diagnostics(
            case_by_id["mavpd_integer_q1_xi31_tg_pilot"],
            results,
            run_root=run_root,
            pairs=(("mavpd_pp_plus", "mavpd_pp_minus"),),
            levels=("B2",),
        )
    )
    case_decisions["chua_arctan_wu_caputo_q099_tg_pilot"]["central_symmetry_diagnostic"] = (
        compute_central_symmetry_diagnostics(
            case_by_id["chua_arctan_wu_caputo_q099_tg_pilot"],
            results,
            run_root=run_root,
            pairs=(("wu_fpp_plus", "wu_fpp_minus"),),
            levels=("B0", "B2"),
        )
    )
    edge_records = _run_declared_edges(
        case_map,
        paths,
        campaign_id=manifest.campaign_id,
        run_root=run_root,
    )
    render_campaign_figures(case_map, results, run_root=run_root, edge_records=edge_records)
    generated_tex = _write_generated_tex(run_root, case_decisions, edge_records)
    decision_payload = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": manifest.campaign_id,
        "status": "initial_pilot_executed",
        "records": list(case_decisions.values()),
        "edge_records": edge_records,
        "generated_report_fragment": str(generated_tex.relative_to(run_root)).replace("\\", "/"),
        "maximum_claim": (
            "EV-TG3 only when both B1/B2 and both declared solvers pass destination and "
            "numerical convergence gates; EV-TG4/TG7 remain pending"
        ),
    }
    write_json(paths.evidence_decisions, decision_payload)
    write_json(
        run_root / "run_summary.json",
        {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": manifest.campaign_id,
            "n_seed_rows": len(read_csv_rows(paths.seed_bank)),
            "n_trajectory_rows": len(read_csv_rows(paths.trajectory_metrics)),
            "n_edge_rows": len(read_csv_rows(paths.edge_brackets)),
            "case_decisions": case_decisions,
            "edge_records": edge_records,
            "figures_manifest": "figures_manifest.json",
            "complete": True,
            "scientific_limit": (
                "finite-time pilot; no complete dynamic edge tracking, topological certificate, "
                "or global hiddenness proof"
            ),
        },
    )


def finalize_existing_pilot(
    run_root: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    tau_stars: Mapping[str, float] | None = None,
) -> Path:
    """Finalize a run whose trajectories are already fully persisted."""

    output = Path(run_root)
    manifest_path = output / "campaign_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("campaign_manifest.json is required to finalize a run.")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = _build_default_cases(Path(project_root), tau_stars)
    manifest = CampaignManifest(
        campaign_id=str(payload["campaign_id"]),
        cases=tuple(str(item) for item in payload.get("cases", ())),
        status=str(payload.get("status", "initial_pilot_execution_contract")),
        schema_version=str(payload.get("schema_version", CAMPAIGN_SCHEMA_VERSION)),
        protocol_version=str(payload.get("protocol_version", "geometric_topological_TG0_TG8_v1")),
        claims_scope=str(payload.get("claims_scope", CampaignManifest("placeholder").claims_scope)),
        metadata=dict(payload.get("metadata", {})),
    )
    results = load_persisted_results(output)
    expected = sum(
        len(case.budget_seed_ids[level]) * 2
        for case in cases
        for level in ("B0", "B1", "B2")
    )
    if len(results) != expected:
        raise RuntimeError(
            f"cannot finalize incomplete run: expected {expected} trajectory rows, found {len(results)}."
        )
    if read_csv_rows(output / "edge_brackets.csv"):
        raise RuntimeError("edge_brackets.csv is not empty; refusing to duplicate an existing edge run.")
    _finalize_pilot(run_root=output, manifest=manifest, cases=cases, results=results)
    return output


def resume_existing_pilot_trajectories(
    run_root: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    tau_stars: Mapping[str, float] | None = None,
) -> list[TrajectoryResult]:
    """Append only trajectory combinations missing from an initialized run."""

    output = Path(run_root)
    payload = json.loads((output / "campaign_manifest.json").read_text(encoding="utf-8"))
    manifest = CampaignManifest(
        campaign_id=str(payload["campaign_id"]),
        cases=tuple(str(item) for item in payload.get("cases", ())),
        status=str(payload.get("status", "initial_pilot_execution_contract")),
        schema_version=str(payload.get("schema_version", CAMPAIGN_SCHEMA_VERSION)),
        protocol_version=str(payload.get("protocol_version", "geometric_topological_TG0_TG8_v1")),
        claims_scope=str(payload.get("claims_scope", CampaignManifest("placeholder").claims_scope)),
        metadata=dict(payload.get("metadata", {})),
    )
    cases = _build_default_cases(Path(project_root), tau_stars)
    results = load_persisted_results(output)
    existing = {
        (result.case_id, result.seed_id, result.budget_level, result.integrator_id)
        for result in results
    }
    paths = CampaignArtifactPaths.under(output)
    for case in cases:
        for level in ("B0", "B1", "B2"):
            for seed_id in case.budget_seed_ids[level]:
                seed = case.record(seed_id)
                for solver in _case_solver_specs(case):
                    key = (case.contract.case_id, seed_id, level, solver.solver_id)
                    if key in existing:
                        continue
                    result = run_and_classify_seed(case, seed, level, solver, run_root=output)
                    append_trajectory_result(
                        paths,
                        result,
                        campaign_id=manifest.campaign_id,
                        contract=case.contract,
                    )
                    results.append(result)
                    existing.add(key)
    return results


def run_initial_pilot(
    *,
    run_id: str = DEFAULT_RUN_ID,
    output_root: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
    overwrite: bool = False,
    tau_stars: Mapping[str, float] | None = None,
) -> Path:
    """Execute the declared PLL/MAVPD/Wu B0--B2 pilot and its figures."""

    root = Path(project_root)
    run_root = Path(output_root) if output_root is not None else DEFAULT_RUNS_ROOT / run_id
    cases = _build_default_cases(root, tau_stars)
    case_map = {case.contract.case_id: case for case in cases}
    manifest = CampaignManifest(
        campaign_id=run_id,
        cases=tuple(case_map),
        status="initial_pilot_execution_contract",
        metadata={
            "pilot_protocol_version": PILOT_PROTOCOL_VERSION,
            "case_contracts": [case.contract.contract_id for case in cases],
            "dynamic_execution_requested": True,
            "edge_scope": "initial_data_boundary_bisection_steps_1_to_4_only",
        },
    )
    paths = initialize_campaign_artifacts(manifest, root=run_root, overwrite=overwrite)
    (run_root / "contracts").mkdir(parents=True, exist_ok=True)
    (run_root / "trajectories").mkdir(parents=True, exist_ok=True)
    (run_root / "metadata").mkdir(parents=True, exist_ok=True)
    for case in cases:
        write_json(run_root / "contracts" / f"{case.contract.contract_id}.json", case.contract.to_jsonable())
        append_case_seed_bank(paths, manifest, case)

    results: list[TrajectoryResult] = []
    for case in cases:
        for level in ("B0", "B1", "B2"):
            for seed_id in case.budget_seed_ids[level]:
                seed = case.record(seed_id)
                for solver in _case_solver_specs(case):
                    result = run_and_classify_seed(case, seed, level, solver, run_root=run_root)
                    append_trajectory_result(
                        paths,
                        result,
                        campaign_id=manifest.campaign_id,
                        contract=case.contract,
                    )
                    results.append(result)

    _finalize_pilot(run_root=run_root, manifest=manifest, cases=cases, results=results)
    return run_root


def _parse_tau(values: Iterable[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for item in values:
        name, separator, raw = str(item).partition("=")
        if not separator or name not in {"pll", "mavpd", "wu"}:
            raise ValueError("--tau entries must be pll=VALUE, mavpd=VALUE, or wu=VALUE.")
        parsed[name] = float(raw)
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--tau", action="append", default=[], metavar="CASE=VALUE")
    args = parser.parse_args(argv)
    if args.finalize_only:
        if args.output_root is None:
            parser.error("--finalize-only requires --output-root.")
        output = finalize_existing_pilot(
            args.output_root,
            project_root=args.project_root,
            tau_stars=_parse_tau(args.tau),
        )
    else:
        output = run_initial_pilot(
            run_id=args.run_id,
            output_root=args.output_root,
            project_root=args.project_root,
            overwrite=args.overwrite,
            tau_stars=_parse_tau(args.tau),
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_RUN_ID",
    "PILOT_PROTOCOL_VERSION",
    "CaseDefinition",
    "CaseRunContract",
    "PhysicalBudget",
    "SolverSpec",
    "TrajectoryResult",
    "append_case_seed_bank",
    "append_trajectory_result",
    "build_mavpd_case",
    "build_pll_case",
    "build_wu_case",
    "evaluate_case_evidence",
    "compute_central_symmetry_diagnostics",
    "finalize_existing_pilot",
    "load_persisted_results",
    "render_campaign_route_figure",
    "render_campaign_figures",
    "resume_existing_pilot_trajectories",
    "run_and_classify_seed",
    "run_initial_pilot",
]
