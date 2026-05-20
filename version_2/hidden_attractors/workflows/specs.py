"""Reusable workflow input specifications for new systems.

System-specific scripts such as the Chua/Danca compatibility workflows may
still exist, but new installable workflows should accept or build one of these
specifications.  The intent is that CLI wrappers, notebooks, and legacy
migration code all record the same explicit inputs before running sphere
controls, basin cuts, strict refinement, continuation, or robustness checks.

Validity warning:
    A valid specification means the numerical experiment is reproducible and
    auditable.  It does not imply that the selected thresholds prove
    hiddenness, chaos, or global basin structure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..io import read_json, write_json


OrderKind = Literal["integer", "caputo", "liouville-weyl-seed", "external"]
MemoryPolicy = Literal["not_applicable", "full_history", "finite_memory", "external"]
SamplingMode = Literal["sphere", "ball", "axis", "eigen_directions", "grid"]
SeedPolicy = Literal["independent", "continuation", "final_state", "manual"]


@dataclass(frozen=True)
class IntegratorSpec:
    """Numerical solver contract shared by CLI and legacy wrappers.

    Required for new systems:
        - ``implementation``: import path, executable, or native backend name.
        - ``order_kind``: integer, Caputo, Weyl seed, or external.
        - ``q``: fractional order when ``order_kind`` is not integer.
        - ``h``, ``t_final``, ``t_burn``: time contract.
        - ``memory_policy`` and ``memory_length``: full history, finite memory,
          or not applicable.
    """

    implementation: str
    order_kind: OrderKind
    h: float
    t_final: float
    t_burn: float = 0.0
    q: float | None = None
    memory_policy: MemoryPolicy = "not_applicable"
    memory_length: float | None = None
    state_columns: tuple[str, ...] = ("x", "y", "z")
    time_column: str = "t"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return validation errors without running the solver."""

        errors: list[str] = []
        if not self.implementation:
            errors.append("integrator.implementation is required.")
        if self.h <= 0.0:
            errors.append("integrator.h must be positive.")
        if self.t_final <= 0.0:
            errors.append("integrator.t_final must be positive.")
        if self.t_burn < 0.0 or self.t_burn >= self.t_final:
            errors.append("integrator.t_burn must satisfy 0 <= t_burn < t_final.")
        if self.order_kind in {"caputo", "liouville-weyl-seed"}:
            if self.q is None or not (0.0 < float(self.q) <= 1.0):
                errors.append("fractional workflows require 0 < integrator.q <= 1.")
        if self.memory_policy == "finite_memory" and (self.memory_length is None or float(self.memory_length) <= 0.0):
            errors.append("finite-memory workflows require positive integrator.memory_length.")
        if self.memory_policy == "full_history" and self.memory_length is not None:
            errors.append("full-history workflows should not set memory_length; record full_history instead.")
        return errors


@dataclass(frozen=True)
class DestinationClassifierSpec:
    """Operational destination labels for basin and hiddenness workflows."""

    implementation: str
    target_positive_label: str = "target_positive"
    target_negative_label: str = "target_negative"
    infinity_label: str = "infinity"
    equilibrium_label_prefix: str = "equilibrium"
    unknown_label: str = "unknown"
    thresholds: Mapping[str, float] = field(default_factory=dict)
    notes: str = ""

    def validate(self) -> list[str]:
        if not self.implementation:
            return ["classifier.implementation is required."]
        return []


@dataclass(frozen=True)
class TargetReferenceSpec:
    """Candidate attractor reference used for target-hit and refinement logic."""

    candidate_id: str
    positive_seed: tuple[float, ...]
    positive_trajectory: str | None = None
    negative_seed_policy: str = "symmetry: negative_seed = -positive_seed"
    target_definition: str = "finite-time similarity to recorded target reference"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, *, dimension: int | None = None) -> list[str]:
        errors: list[str] = []
        if not self.candidate_id:
            errors.append("target_reference.candidate_id is required.")
        if dimension is not None and len(self.positive_seed) != int(dimension):
            errors.append(f"target_reference.positive_seed must have dimension {dimension}.")
        if self.positive_trajectory is None and not self.positive_seed:
            errors.append("target_reference requires either a positive seed or a trajectory file.")
        return errors


@dataclass(frozen=True)
class SphereControlSpec:
    """Equilibrium-neighborhood sphere sampling contract."""

    equilibria: tuple[str, ...]
    radii: tuple[float, ...]
    samples_per_radius: int
    sampling_mode: SamplingMode = "sphere"
    random_seed: int = 20260517
    source: str = "generated"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.equilibria:
            errors.append("sphere_controls.equilibria cannot be empty.")
        if not self.radii or any(float(r) <= 0.0 for r in self.radii):
            errors.append("sphere_controls.radii must be positive.")
        if int(self.samples_per_radius) <= 0:
            errors.append("sphere_controls.samples_per_radius must be positive.")
        return errors


@dataclass(frozen=True)
class BasinSliceSpec:
    """Initial-condition grid or section used by basin workflows."""

    varying_state_indices: tuple[int, int]
    limits: tuple[tuple[float, float], tuple[float, float]]
    grid_shape: tuple[int, int]
    fixed_state: tuple[float, ...]
    plane_label: str = "custom"
    source: str = "generated"

    def validate(self, *, dimension: int | None = None) -> list[str]:
        errors: list[str] = []
        if len(self.varying_state_indices) != 2:
            errors.append("basin.varying_state_indices must contain two indices.")
        if any(int(i) < 0 for i in self.varying_state_indices):
            errors.append("basin.varying_state_indices must be nonnegative.")
        if dimension is not None and len(self.fixed_state) != int(dimension):
            errors.append(f"basin.fixed_state must have dimension {dimension}.")
        nx, ny = self.grid_shape
        if int(nx) <= 0 or int(ny) <= 0:
            errors.append("basin.grid_shape values must be positive.")
        for lo, hi in self.limits:
            if not float(lo) < float(hi):
                errors.append("basin.limits must satisfy min < max.")
        return errors


@dataclass(frozen=True)
class StrictRefinementSpec:
    """Geometry thresholds for target-reference refinement."""

    max_score: float = 0.45
    max_cloud_norm: float = 0.35
    max_range_rel: float = 0.60
    max_fft_rel: float = 0.35
    max_section_norm: float = 0.50
    min_reference_margin: float = 0.10
    min_control_margin: float = 0.10
    negative_control_equilibria: tuple[str, ...] = ("E0", "E+", "E-")
    negative_control_radius: float = 1.0e-4
    controls_enabled: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        positive_thresholds = {
            "max_score": self.max_score,
            "max_cloud_norm": self.max_cloud_norm,
            "max_range_rel": self.max_range_rel,
            "max_fft_rel": self.max_fft_rel,
            "max_section_norm": self.max_section_norm,
            "min_reference_margin": self.min_reference_margin,
            "min_control_margin": self.min_control_margin,
        }
        for name, value in positive_thresholds.items():
            if float(value) < 0.0:
                errors.append(f"strict_refinement.{name} must be nonnegative.")
        if self.controls_enabled and (not self.negative_control_equilibria or self.negative_control_radius <= 0.0):
            errors.append("enabled negative controls require equilibria and a positive radius.")
        return errors


@dataclass(frozen=True)
class TrajectoryDiagnosticsSpec:
    """Tail window and observable contract for metrics, spectra, and sections."""

    retained_time_start: float | None = None
    retained_time_end: float | None = None
    observables: tuple[str, ...] = ("x", "y", "z")
    extrema_observable: str = "x"
    spectrum_observable: str = "x"
    section: Mapping[str, Any] = field(default_factory=dict)
    metrics: tuple[str, ...] = ("range", "variance", "fft", "cloud")

    def validate(self, *, integrator: IntegratorSpec | None = None) -> list[str]:
        """Validate the declared post-transient diagnostics window."""

        errors: list[str] = []
        if not self.observables:
            errors.append("trajectory_diagnostics.observables cannot be empty.")
        if self.retained_time_start is not None and self.retained_time_start < 0.0:
            errors.append("trajectory_diagnostics.retained_time_start must be nonnegative.")
        if self.retained_time_end is not None and self.retained_time_end <= 0.0:
            errors.append("trajectory_diagnostics.retained_time_end must be positive.")
        if (
            self.retained_time_start is not None
            and self.retained_time_end is not None
            and self.retained_time_start >= self.retained_time_end
        ):
            errors.append("trajectory_diagnostics retained window must satisfy start < end.")
        if integrator is not None:
            if self.retained_time_start is not None and self.retained_time_start < integrator.t_burn:
                errors.append("trajectory_diagnostics.retained_time_start should not precede integrator.t_burn.")
            if self.retained_time_end is not None and self.retained_time_end > integrator.t_final:
                errors.append("trajectory_diagnostics.retained_time_end cannot exceed integrator.t_final.")
        return errors


@dataclass(frozen=True)
class ParameterSweepSpec:
    """Parameter sweep contract for bifurcation and continuation-like runs."""

    parameter_name: str
    values: tuple[float, ...] = ()
    start: float | None = None
    stop: float | None = None
    count: int | None = None
    seed_policy: SeedPolicy = "independent"
    observable: str = "x"
    extractor: str = "local_extrema_after_burn"

    def validate(self) -> list[str]:
        """Validate the sweep axis without generating trajectories."""

        errors: list[str] = []
        if not self.parameter_name:
            errors.append("parameter_sweep.parameter_name is required.")
        has_values = bool(self.values)
        has_range = self.start is not None or self.stop is not None or self.count is not None
        if has_values and has_range:
            errors.append("parameter_sweep must use either values or start/stop/count, not both.")
        if not has_values and not has_range:
            errors.append("parameter_sweep requires values or start/stop/count.")
        if has_values and any(not np.isfinite(float(value)) for value in self.values):
            errors.append("parameter_sweep.values must be finite.")
        if has_range:
            if self.start is None or self.stop is None or self.count is None:
                errors.append("parameter_sweep range requires start, stop, and count.")
            elif not float(self.start) < float(self.stop):
                errors.append("parameter_sweep.start must be smaller than stop.")
            elif int(self.count) <= 1:
                errors.append("parameter_sweep.count must be greater than one.")
        return errors


@dataclass(frozen=True)
class RobustnessCaseSpec:
    """One controlled perturbation for robustness workflows."""

    case_id: str
    integrator_overrides: Mapping[str, Any] = field(default_factory=dict)
    parameter_overrides: Mapping[str, Any] = field(default_factory=dict)
    allowed_change: str = "documented numerical robustness perturbation"

    def validate(self) -> list[str]:
        """Validate that the robustness case is named and auditable."""

        if not self.case_id:
            return ["robustness_cases.case_id is required."]
        if not self.integrator_overrides and not self.parameter_overrides:
            return [f"robustness case {self.case_id} requires an integrator or parameter override."]
        return []


@dataclass(frozen=True)
class WorkflowInputSpec:
    """Single auditable input contract for reusable workflows."""

    system_name: str
    dimension: int
    parameters: Mapping[str, Any]
    integrator: IntegratorSpec
    classifier: DestinationClassifierSpec | None = None
    target_reference: TargetReferenceSpec | None = None
    sphere_controls: SphereControlSpec | None = None
    basin: BasinSliceSpec | None = None
    strict_refinement: StrictRefinementSpec | None = None
    trajectory_diagnostics: TrajectoryDiagnosticsSpec | None = None
    parameter_sweep: ParameterSweepSpec | None = None
    robustness_cases: tuple[RobustnessCaseSpec, ...] = ()
    notes: str = ""

    def validate_for(self, features: Sequence[str]) -> list[str]:
        """Validate only the pieces needed by the requested features."""

        errors: list[str] = []
        if not self.system_name:
            errors.append("system_name is required.")
        if int(self.dimension) <= 0:
            errors.append("dimension must be positive.")
        errors.extend(self.integrator.validate())
        needs_classifier = {"sphere-controls", "basin", "full-hiddenness-protocol"} & set(features)
        needs_reference = {"sphere-controls", "strict-refinement", "robustness", "full-hiddenness-protocol"} & set(features)
        if needs_classifier:
            if self.classifier is None:
                errors.append("classifier is required for sphere-controls, basin, or full-hiddenness-protocol workflows.")
            else:
                errors.extend(self.classifier.validate())
        if needs_reference:
            if self.target_reference is None:
                errors.append("target_reference is required for sphere-controls, strict-refinement, robustness, and full-hiddenness-protocol.")
            else:
                errors.extend(self.target_reference.validate(dimension=self.dimension))
        if {"sphere-controls", "full-hiddenness-protocol"} & set(features):
            if self.sphere_controls is None:
                errors.append("sphere_controls is required for sphere-controls or full-hiddenness-protocol workflows.")
            else:
                errors.extend(self.sphere_controls.validate())
        if "basin" in features:
            if self.basin is None:
                errors.append("basin is required for basin workflows.")
            else:
                errors.extend(self.basin.validate(dimension=self.dimension))
        if "strict-refinement" in features:
            if self.strict_refinement is None:
                errors.append("strict_refinement is required for strict-refinement workflows.")
            else:
                errors.extend(self.strict_refinement.validate())
        if {"trajectory-diagnostics", "robustness", "bifurcation", "full-hiddenness-protocol"} & set(features):
            if self.trajectory_diagnostics is None:
                errors.append("trajectory_diagnostics is required for diagnostics, robustness, bifurcation, and full-hiddenness-protocol.")
            else:
                errors.extend(self.trajectory_diagnostics.validate(integrator=self.integrator))
        if "bifurcation" in features:
            if self.parameter_sweep is None:
                errors.append("parameter_sweep is required for bifurcation workflows.")
            else:
                errors.extend(self.parameter_sweep.validate())
        if "robustness" in features:
            if not self.robustness_cases:
                errors.append("robustness_cases is required for robustness workflows.")
            for case in self.robustness_cases:
                errors.extend(case.validate())
        return errors

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, Mapping):
                return {str(key): convert(val) for key, val in value.items()}
            if hasattr(value, "item"):
                return value.item()
            return value

        return convert(asdict(self))


def write_workflow_spec(path: str | Path, spec: WorkflowInputSpec) -> None:
    """Persist a workflow specification next to run artifacts."""

    write_json(path, spec.to_jsonable())


def load_workflow_spec(path: str | Path) -> Mapping[str, Any]:
    """Load a JSON workflow spec for legacy adapters or CLIs."""

    return read_json(path)


def example_chua_fractional_spec() -> WorkflowInputSpec:
    """Return a minimal example spec for documentation and tests."""

    return WorkflowInputSpec(
        system_name="chua-piecewise",
        dimension=3,
        parameters={"model": "piecewise"},
        integrator=IntegratorSpec(
            implementation="hidden_attractors.native.FractionalChuaBackend.integrate_efork3",
            order_kind="caputo",
            q=0.9998,
            h=0.01,
            memory_policy="finite_memory",
            memory_length=10.0,
            t_final=1500.0,
            t_burn=100.0,
        ),
        classifier=DestinationClassifierSpec(
            implementation="hidden_attractors.native.BasinBackend.classify_point",
            thresholds={"divergence_norm": 120.0, "equilibrium_tol": 1.0e-3},
        ),
        target_reference=TargetReferenceSpec(
            candidate_id="example_candidate",
            positive_seed=(5.0, 0.0, -8.0),
        ),
        sphere_controls=SphereControlSpec(
            equilibria=("E0", "E+", "E-"),
            radii=(1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3),
            samples_per_radius=100,
        ),
        basin=BasinSliceSpec(
            varying_state_indices=(0, 1),
            limits=((0.0, 9.0), (-1.5, 1.5)),
            grid_shape=(160, 160),
            fixed_state=(5.0, 0.0, -8.0),
            plane_label="xy",
        ),
        strict_refinement=StrictRefinementSpec(),
        trajectory_diagnostics=TrajectoryDiagnosticsSpec(
            retained_time_start=100.0,
            retained_time_end=1500.0,
            observables=("x", "y", "z"),
        ),
        parameter_sweep=ParameterSweepSpec(
            parameter_name="alpha",
            start=8.0,
            stop=16.0,
            count=81,
            seed_policy="independent",
        ),
        robustness_cases=(
            RobustnessCaseSpec(
                case_id="h_half",
                integrator_overrides={"h": 0.005},
                allowed_change="step-size sensitivity check",
            ),
        ),
        notes="Template only; replace seed, parameters, and backend for a real run.",
    )


__all__ = [
    "BasinSliceSpec",
    "DestinationClassifierSpec",
    "IntegratorSpec",
    "MemoryPolicy",
    "OrderKind",
    "ParameterSweepSpec",
    "RobustnessCaseSpec",
    "SamplingMode",
    "SeedPolicy",
    "SphereControlSpec",
    "StrictRefinementSpec",
    "TargetReferenceSpec",
    "TrajectoryDiagnosticsSpec",
    "WorkflowInputSpec",
    "example_chua_fractional_spec",
    "load_workflow_spec",
    "write_workflow_spec",
]
