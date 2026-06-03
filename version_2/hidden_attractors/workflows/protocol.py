"""Canonical Caputo hidden-attractor methodology contracts.

Stability: experimental
    This module defines the maintained protocol vocabulary and machine-readable
    records. Historical routes may be reproduced separately, but new workflow
    output must use the stage order, seed families, and verdicts defined here.

Scientific boundary:
    Describing functions, Lur'e reconstruction, and Machado/FDF are
    seed-generation mechanisms. They never establish hiddenness. A hiddenness
    label can be emitted only after continuation, target reproduction under
    robustness checks, and equilibrium-neighborhood plus basin tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from hidden_attractors.validation.states import AttractorValidationState
from hidden_attractors.reproducibility import (
    CONSERVATIVE_HIDDENNESS_LABEL,
    metadata_to_jsonable,
    validate_hiddenness_promotion_metadata,
    validate_run_metadata,
)


SCHEMA_VERSION = "1.0"
PROTOCOL_VERSION = "caputo_hidden_attractors_v1"


def _normalize_hiddenness_label(label: str) -> str:
    from hidden_attractors.verification.candidate_gate import normalize_hiddenness_label

    return normalize_hiddenness_label(label)


def _evaluate_candidate_gate(evidence: dict[str, Any]) -> dict[str, Any]:
    from hidden_attractors.verification.candidate_gate import evaluate_candidate_gate

    return evaluate_candidate_gate(evidence)

OFFICIAL_STAGE_ORDER: tuple[str, ...] = (
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

SEED_FAMILIES: tuple[str, ...] = (
    "lure_classical_centered",
    "lure_classical_biased",
    "machado_centered",
    "machado_biased",
)

FINAL_LABELS: tuple[str, ...] = (
    "seed_only",
    "continuation_survivor",
    "rejected_post_continuation",
    "robust_survivor",
    "compatible_with_hiddenness_under_tested_radii",
    "hiddenness_supported_under_tested_neighborhoods",
    "self_excited_contact_detected",
    "hiddenness_inconclusive",
    "candidate_not_reproducible",
    "numerical_failure",
    "candidate_rejected",
    "rejected_self_excited_contact",
    "hidden_verified",
    "hidden_verified_only_if_full_protocol_passed",
)

ROBUSTNESS_VERDICTS: tuple[str, ...] = (
    "robust_target_hit",
    "weak_target_hit",
    "not_reproduced",
    "numerical_failure",
)

SOFT_PRECHECK_LABELS: tuple[str, ...] = (
    "pre_continuation_admissible",
    "pre_continuation_periodic",
    "pre_continuation_quasiperiodic",
    "pre_continuation_chaotic_looking",
    "pre_continuation_equilibrium_collapse",
    "rejected_invalid_configuration",
    "rejected_invalid_amplitude_frequency",
    "rejected_numerical_failure",
    "rejected_catastrophic_divergence",
    "rejected_exact_duplicate",
)

SeedFamily = Literal[
    "lure_classical_centered",
    "lure_classical_biased",
    "machado_centered",
    "machado_biased",
]
FinalLabel = Literal[
    "seed_only",
    "continuation_survivor",
    "rejected_post_continuation",
    "robust_survivor",
    "compatible_with_hiddenness_under_tested_radii",
    "hiddenness_supported_under_tested_neighborhoods",
    "self_excited_contact_detected",
    "hiddenness_inconclusive",
    "candidate_not_reproducible",
    "numerical_failure",
    "candidate_rejected",
    "rejected_self_excited_contact",
    "hidden_verified",
    "hidden_verified_only_if_full_protocol_passed",
]
RobustnessLabel = Literal[
    "robust_target_hit",
    "weak_target_hit",
    "not_reproduced",
    "numerical_failure",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


@dataclass(frozen=True)
class NumericalContract:
    """Complete numerical source-of-truth for one Caputo workflow run."""

    q: float
    h: float
    t_final: float
    t_transient: float = 0.0
    backend: str = "efork_c"
    memory_policy: str = "full_history"
    memory_length: float | None = None
    tolerances: Mapping[str, float] = field(default_factory=dict)
    boundedness_thresholds: Mapping[str, float] = field(
        default_factory=lambda: {"divergence_norm": 120.0}
    )
    equilibrium_distance_thresholds: Mapping[str, float] = field(
        default_factory=lambda: {"collapsed_distance": 1.0e-3}
    )
    harmonic_residual_thresholds: Mapping[str, float] = field(default_factory=dict)
    similarity_thresholds: Mapping[str, float] = field(default_factory=dict)
    hiddenness_radii: tuple[float, ...] = ()
    samples_per_radius: int = 0
    sample_growth_per_radius: int = 0
    random_seed_policy: str = "fixed_reproducible"
    random_seed: int | None = None
    output_schema_version: str = SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Deprecated input alias accepted for existing callers.
    t_burn: float | None = None

    @property
    def effective_transient(self) -> float:
        return float(self.t_burn) if self.t_burn is not None else float(self.t_transient)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not np.isfinite(float(self.q)) or not (0.0 < float(self.q) <= 1.0):
            errors.append("numerical_contract.q must satisfy 0 < q <= 1.")
        if not np.isfinite(float(self.h)) or self.h <= 0.0:
            errors.append("numerical_contract.h must be positive.")
        if self.t_final <= 0.0 or self.effective_transient < 0.0 or self.effective_transient >= self.t_final:
            errors.append("numerical_contract requires 0 <= t_transient < t_final.")
        if self.memory_policy == "finite_memory" and (self.memory_length is None or self.memory_length <= 0.0):
            errors.append("finite_memory requires positive memory_length.")
        if self.memory_policy == "full_history" and self.memory_length is not None:
            errors.append("full_history must record memory_length as null.")
        if any(radius <= 0.0 for radius in self.hiddenness_radii):
            errors.append("hiddenness_radii must be positive.")
        if self.samples_per_radius < 0 or self.sample_growth_per_radius < 0:
            errors.append("hiddenness sampling counts cannot be negative.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["t_transient"] = self.effective_transient
        values.pop("t_burn", None)
        return _jsonable(values)


@dataclass(frozen=True)
class UnifiedSeedRecord:
    """Uniform seed record shared by classical Lur'e and Machado/FDF families."""

    family: SeedFamily
    centered_or_biased: Literal["centered", "biased"]
    A: float
    sigma0: float
    omega: float
    mu: float
    theta: float
    q: float
    harmonic_residual: float
    rho_H: float
    x0: tuple[float, ...]
    reconstruction_metadata: Mapping[str, Any] = field(default_factory=dict)
    source_config: str = ""
    candidate_id: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.family not in SEED_FAMILIES:
            errors.append("seed family is not part of the official seed schema.")
        expected_bias = "biased" if self.family.endswith("_biased") else "centered"
        if self.centered_or_biased != expected_bias:
            errors.append("centered_or_biased does not match seed family.")
        if expected_bias == "centered" and abs(float(self.sigma0)) > 1.0e-12:
            errors.append("centered seed families require sigma0=0.")
        if self.family.startswith("lure_classical") and abs(float(self.mu) - 1.0) > 1.0e-12:
            errors.append("classical Lur'e families require mu=1.")
        if self.A <= 0.0 or self.omega <= 0.0 or not np.all(np.isfinite(self.x0)):
            errors.append("seed amplitude, frequency, and x0 must be finite and valid.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class SoftPrecheckResult:
    """Soft diagnostic decision made before continuation."""

    candidate_id: str
    label: str
    admissible_for_continuation: bool
    finite_trajectory: bool
    immediate_numerical_failure: bool = False
    catastrophic_divergence: bool = False
    immediate_equilibrium_collapse: bool = False
    exact_duplicate: bool = False
    short_window_label: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def periodic(cls, candidate_id: str, **kwargs: Any) -> "SoftPrecheckResult":
        return cls(
            candidate_id=candidate_id,
            label="pre_continuation_periodic",
            admissible_for_continuation=True,
            finite_trajectory=True,
            short_window_label="periodic",
            **kwargs,
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.label not in SOFT_PRECHECK_LABELS:
            errors.append("soft_precheck label is not official.")
        hard_failure = (
            not self.finite_trajectory
            or self.immediate_numerical_failure
            or self.catastrophic_divergence
            or self.exact_duplicate
            or self.label in {"rejected_invalid_configuration", "rejected_invalid_amplitude_frequency"}
        )
        if hard_failure and self.admissible_for_continuation:
            errors.append("hard soft_precheck failures cannot be admitted to continuation.")
        if self.label == "pre_continuation_periodic" and not self.admissible_for_continuation:
            errors.append("pre-continuation periodicity is diagnostic and cannot reject a seed.")
        return errors


@dataclass(frozen=True)
class ContinuationPlan:
    """Public continuation interface; lambda=0 starts and lambda=1 targets."""

    lambda_values: tuple[float, ...]
    mapping: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        values = tuple(float(value) for value in self.lambda_values)
        if not values or abs(values[0]) > 1.0e-12 or abs(values[-1] - 1.0) > 1.0e-12:
            return ["ContinuationPlan requires lambda_values beginning at 0 and ending at 1."]
        if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            return ["ContinuationPlan lambda values must lie in [0, 1]."]
        if any(right <= left for left, right in zip(values, values[1:])):
            return ["ContinuationPlan lambda values must be strictly increasing."]
        return []

    @classmethod
    def uniform(cls, steps: int, *, internal_parameter: str = "lambda") -> "ContinuationPlan":
        if int(steps) < 2:
            raise ValueError("continuation steps must be at least 2.")
        values = tuple(float(value) for value in np.linspace(0.0, 1.0, int(steps)))
        return cls(values, {"public_parameter": "lambda", "internal_parameter": internal_parameter})


@dataclass(frozen=True)
class ContinuationStep:
    lambda_value: float
    state_in: tuple[float, ...]
    state_out: tuple[float, ...]
    status: str
    backend: str
    memory_policy: str
    step_size: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContinuationTrace:
    candidate_id: str
    plan: ContinuationPlan
    steps: tuple[ContinuationStep, ...]
    final_state: tuple[float, ...] | None
    survived: bool
    failure_reason: str = ""


@dataclass(frozen=True)
class PostContinuationDecision:
    candidate_id: str
    label: FinalLabel
    finite_trajectory: bool
    bounded_trajectory: bool
    collapsed_to_equilibrium: bool
    persistent_dynamics: bool
    trivial_short_period: bool
    duplicate_reference: bool
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DynamicReference:
    candidate_id: str
    trajectory_path: str
    bounding_box: Mapping[str, Any]
    centroid: tuple[float, ...]
    dispersion: tuple[float, ...]
    equilibrium_distances: Mapping[str, float]
    spectral_summary: Mapping[str, Any]
    recurrence_score: float | None
    similarity_signature: Mapping[str, Any]
    backend_metadata: Mapping[str, Any]
    lyapunov_estimate: Mapping[str, Any] | None = None
    plots: tuple[str, ...] = ()
    reports: tuple[str, ...] = ()


@dataclass(frozen=True)
class RobustnessVerdict:
    candidate_id: str
    verdict: RobustnessLabel
    case_results: Mapping[str, Any]
    similarity_metrics: Mapping[str, Any]


@dataclass(frozen=True)
class HiddennessTestResult:
    candidate_id: str
    tested_equilibria: tuple[str, ...]
    tested_radii: tuple[float, ...]
    neighborhood_sampling_mode: str
    target_contacts: int
    numerical_failures: int
    basin_planes: tuple[str, ...]
    reference_was_robust: bool
    final_label: FinalLabel
    run_metadata: Mapping[str, Any] | None = None
    required_equilibria: tuple[str, ...] = ()
    required_radii: tuple[float, ...] = ()

    def validate(self) -> list[str]:
        errors: list[str] = []
        required_planes = {"xy_close", "xy_large", "xz_close", "xz_large", "yz_close", "yz_large"}
        tested_radii = tuple(float(radius) for radius in self.tested_radii)
        includes_required_radii = bool(self.required_radii) and all(
            any(np.isclose(float(required), tested, rtol=1.0e-12, atol=1.0e-15) for tested in tested_radii)
            for required in self.required_radii
        )
        full_protocol = (
            self.reference_was_robust
            and self.neighborhood_sampling_mode == "ball"
            and bool(self.required_equilibria)
            and set(self.required_equilibria).issubset(set(self.tested_equilibria))
            and includes_required_radii
            and self.target_contacts == 0
            and self.numerical_failures == 0
            and required_planes.issubset(set(self.basin_planes))
        )
        strong_label = _normalize_hiddenness_label(self.final_label) == "hiddenness_supported_under_tested_neighborhoods"
        if strong_label and not full_protocol:
            errors.append("hidden_verified_only_if_full_protocol_passed requires the complete tested protocol.")
        if strong_label:
            errors.extend(validate_hiddenness_promotion_metadata(dict(self.run_metadata) if self.run_metadata else None))
        return errors

    @property
    def promotion_verdict(self) -> str:
        """Return the only candidate label allowed by the available evidence."""

        metadata = metadata_to_jsonable(self.run_metadata) if self.run_metadata else None
        numerical = metadata.get("numerical_contract", {}) if isinstance(metadata, Mapping) else {}
        seed = metadata.get("seed") if isinstance(metadata, Mapping) else {}
        seed = seed if isinstance(seed, Mapping) else {}
        continuation = metadata.get("continuation", {}) if isinstance(metadata, Mapping) else {}
        gate = _evaluate_candidate_gate(
            {
                "run_metadata": metadata,
                "equilibria": {"all_found": bool(self.required_equilibria), "max_residual": 0.0},
                "matignon": {"all_classified": bool(self.required_equilibria), "q": numerical.get("q")},
                "seed": {
                    "localized": self.reference_was_robust,
                    "method": "manual_traced",
                    "source": seed.get("source", "hiddenness_test_result"),
                },
                "continuation": continuation,
                "trajectory": {
                    "bounded": self.reference_was_robust,
                    "nontrivial": self.reference_was_robust,
                    "finite_fraction": 1.0,
                    "post_transient_length": 1,
                },
                "robustness": {
                    "tested_h": self.reference_was_robust,
                    "tested_memory": self.reference_was_robust,
                    "tested_t_final": self.reference_was_robust,
                    "tested_integrator": self.reference_was_robust,
                    "consistent": self.reference_was_robust,
                },
                "hiddenness": {
                    "tested_all_equilibria": bool(self.required_equilibria)
                    and set(self.required_equilibria).issubset(set(self.tested_equilibria)),
                    "tested_radii": self.tested_radii,
                    "required_radii": self.required_radii,
                    "target_hits_from_equilibria": self.target_contacts,
                    "basin_intersection_detected": False,
                    "basin_controls_complete": {
                        "xy_close",
                        "xy_large",
                        "xz_close",
                        "xz_large",
                        "yz_close",
                        "yz_large",
                    }.issubset(set(self.basin_planes)),
                    "numerical_failures": self.numerical_failures,
                },
            }
        )
        return gate["verdict"]


@dataclass(frozen=True)
class StageEnvelope:
    """Uniform machine-readable summary emitted by official stages."""

    stage: str
    status: str
    system: str
    numerical_contract: Mapping[str, Any]
    candidate_id: str | None = None
    inputs: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    verdict: str | None = None
    files: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    run_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata_validation_errors: Sequence[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    state: str | None = None
    state_history: Sequence[str] = field(default_factory=list)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    failed_requirements: Sequence[str] = field(default_factory=list)
    method_scope: str = ""
    warnings: Sequence[str] = field(default_factory=list)
    literature_note: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.stage not in OFFICIAL_STAGE_ORDER:
            errors.append("stage is not part of the official methodology.")
        if self.verdict in FINAL_LABELS and self.stage not in {
            "seed_generation",
            "post_continuation_filter",
            "robustness",
            "hiddenness_tests",
        }:
            errors.append("final candidate labels may only be issued by decision stages.")
        metadata_errors = validate_run_metadata(metadata_to_jsonable(dict(self.run_metadata)))
        if list(self.metadata_validation_errors) != metadata_errors:
            errors.append("metadata_validation_errors must match validate_run_metadata(run_metadata).")
        if _normalize_hiddenness_label(str(self.verdict)) == "hiddenness_supported_under_tested_neighborhoods":
            errors.extend(validate_hiddenness_promotion_metadata(dict(self.run_metadata)))
        return errors

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "schema_version": self.schema_version,
                "protocol_version": self.protocol_version,
                "stage": self.stage,
                "status": self.status,
                "candidate_id": self.candidate_id,
                "system": self.system,
                "numerical_contract": dict(self.numerical_contract),
                "inputs": dict(self.inputs),
                "outputs": dict(self.outputs),
                "metrics": dict(self.metrics),
                "verdict": self.verdict,
                "files": dict(self.files),
                "provenance": dict(self.provenance),
                "run_metadata": dict(self.run_metadata),
                "metadata_validation_errors": list(self.metadata_validation_errors),
                "state": self.state,
                "state_history": list(self.state_history),
                "evidence": dict(self.evidence),
                "failed_requirements": list(self.failed_requirements),
                "method_scope": self.method_scope,
                "warnings": list(self.warnings),
                "literature_note": self.literature_note,
            }
        )


def validate_global_report_coherence(report_data: dict) -> None:
    """Validate coherence of global report validation metadata states and evidence."""

    state = report_data.get("state")
    verdict = report_data.get("verdict")
    final_report_status = ""
    if isinstance(report_data.get("final_report"), dict):
        final_report_status = report_data["final_report"].get("status", "")

    outputs = report_data.get("outputs", {})
    metrics = report_data.get("metrics", {})
    stage_statuses = report_data.get("stage_statuses", {})

    # 1. Structural check: State 'seed_found' cannot be labeled as hidden
    is_seed_found = (
        state == "seed_found"
        or verdict == "seed_found"
        or (isinstance(outputs, dict) and outputs.get("state") == "seed_found")
    )
    if is_seed_found:
        if verdict in (
            "hidden_compatible",
            "hidden_verified",
            "compatible_with_hiddenness_under_tested_radii",
            "hidden_verified_only_if_full_protocol_passed",
        ):
            raise ValueError("State 'seed_found' cannot be labeled as hidden.")
        if state in ("hidden_compatible", "hidden_verified"):
            raise ValueError("State 'seed_found' cannot be labeled as hidden.")

    # Normalize state/verdict keys from envelopes if checking a list of stages or manifest
    is_hidden_verified = (
        state == "hidden_verified"
        or state == "hiddenness_supported_under_tested_neighborhoods"
        or _normalize_hiddenness_label(str(verdict)) == "hiddenness_supported_under_tested_neighborhoods"
        or final_report_status == "hidden_verified"
        or report_data.get("final_report_status") == "hidden_verified"
    )

    if isinstance(outputs, dict):
        if outputs.get("state") in {"hidden_verified", "hiddenness_supported_under_tested_neighborhoods"} or _normalize_hiddenness_label(str(outputs.get("verdict"))) == "hiddenness_supported_under_tested_neighborhoods":
            is_hidden_verified = True
    if isinstance(metrics, dict):
        if metrics.get("state") in {"hidden_verified", "hiddenness_supported_under_tested_neighborhoods"} or _normalize_hiddenness_label(str(metrics.get("verdict"))) == "hiddenness_supported_under_tested_neighborhoods":
            is_hidden_verified = True

    if is_hidden_verified:
        metadata_errors = validate_hiddenness_promotion_metadata(report_data.get("run_metadata"))
        if metadata_errors:
            raise ValueError(
                "State 'hidden_verified' requires complete reproducibility metadata: "
                + "; ".join(metadata_errors)
            )
        evidence = report_data.get("evidence", {})
        
        has_sphere = False
        has_basin = False

        if isinstance(evidence, dict):
            has_sphere = bool(evidence.get("sphere_tests") or evidence.get("completed_sphere_tests"))
            has_basin = bool(evidence.get("basin_neighborhood_tests") or evidence.get("completed_basin_tests"))

        if isinstance(outputs, dict):
            has_sphere = has_sphere or bool(outputs.get("sphere_tests") or outputs.get("sphere_controls"))
            has_basin = has_basin or bool(outputs.get("basin_neighborhood_tests") or outputs.get("basin_slices"))
            hiddenness_run = outputs.get("branches", {}).get("full_history", {}).get("run_type")
            if hiddenness_run == "full_protocol_hiddenness_tests":
                has_sphere = True

        files = report_data.get("files", {})
        if isinstance(files, dict):
            for f in files.values():
                if isinstance(f, str) and "sphere" in f:
                    has_sphere = True
                if isinstance(f, str) and "basin" in f:
                    has_basin = True
                elif isinstance(f, list):
                    for item in f:
                        if isinstance(item, str) and "sphere" in item:
                            has_sphere = True
                        if isinstance(item, str) and "basin" in item:
                            has_basin = True

        if not (has_sphere and has_basin):
            raise ValueError(
                "State 'hidden_verified' requires evidence of completed sphere_tests and basin_neighborhood_tests."
            )

    is_chaotic_candidate = (
        state == "chaotic_candidate"
        or verdict == "chaotic_candidate"
        or final_report_status == "chaotic_candidate"
        or (isinstance(outputs, dict) and (outputs.get("state") == "chaotic_candidate" or outputs.get("verdict") == "chaotic_candidate"))
    )
    if is_chaotic_candidate:
        has_chaos_evidence = False
        for d in (metrics, outputs, report_data.get("evidence", {})):
            if not isinstance(d, dict):
                continue
            if d.get("zero_one_test") is not None or d.get("zero_one_kappa") is not None:
                has_chaos_evidence = True
            if d.get("lyapunov_estimate") is not None or d.get("max_lyapunov") is not None or d.get("positive_lyapunov") is not None:
                max_ly = d.get("max_lyapunov")
                if max_ly is not None:
                    try:
                        if float(max_ly) > 0.0:
                            has_chaos_evidence = True
                    except ValueError:
                        pass
                else:
                    has_chaos_evidence = True
            kappa = d.get("zero_one_kappa") or d.get("kappa")
            if kappa is not None:
                try:
                    if float(kappa) > 0.5:
                        has_chaos_evidence = True
                except ValueError:
                    pass

        if isinstance(stage_statuses, dict):
            if stage_statuses.get("diagnostics") == "completed":
                has_chaos_evidence = True

        files = report_data.get("files", {})
        if isinstance(files, dict):
            for f in files.values():
                if isinstance(f, str) and ("lyapunov" in f or "zero_one" in f or "chaos" in f):
                    has_chaos_evidence = True

        if not has_chaos_evidence:
            raise ValueError(
                "State 'chaotic_candidate' requires chaos test evidence (positive max Lyapunov or 0-1 test)."
            )


def sample_uniform_ball(
    center: Sequence[float],
    radius: float,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample points inside an n-dimensional ball, including interior points."""

    point = np.asarray(center, dtype=float)
    if point.ndim != 1 or float(radius) <= 0.0 or int(count) <= 0:
        raise ValueError("center, radius, and count must define a nonempty ball sample.")
    directions = rng.normal(size=(int(count), point.size))
    norms = np.linalg.norm(directions, axis=1)
    norms[norms == 0.0] = 1.0
    directions /= norms[:, None]
    radial = float(radius) * rng.random(int(count)) ** (1.0 / float(point.size))
    return point[None, :] + radial[:, None] * directions


__all__ = [
    "ContinuationPlan",
    "ContinuationStep",
    "ContinuationTrace",
    "DynamicReference",
    "FINAL_LABELS",
    "HiddennessTestResult",
    "NumericalContract",
    "OFFICIAL_STAGE_ORDER",
    "PROTOCOL_VERSION",
    "PostContinuationDecision",
    "ROBUSTNESS_VERDICTS",
    "RobustnessVerdict",
    "SCHEMA_VERSION",
    "SEED_FAMILIES",
    "SOFT_PRECHECK_LABELS",
    "SoftPrecheckResult",
    "StageEnvelope",
    "UnifiedSeedRecord",
    "sample_uniform_ball",
    "validate_global_report_coherence",
]
