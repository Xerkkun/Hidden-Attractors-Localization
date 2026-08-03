"""Common sampled-trajectory and analysis-result contracts.

Stability: experimental

The contracts in this module are deliberately independent of a particular
integer or fractional solver.  They preserve the metadata that changes the
meaning of a trajectory diagnostic: sampling, projection, transient interval,
derivative definition, order specification, lower terminal/prehistory, memory
policy, and numerical tolerances.

``TrajectoryInput`` is not a claim that the supplied coordinates form a
complete state.  In particular, a finite projection of a fractional-order
trajectory generally omits its hereditary state.  ``AnalysisResult`` records a
finite numerical diagnostic and its provenance; it is not a chaos,
attraction, or hiddenness certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from operator import index as operator_index
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


TRAJECTORY_SYSTEM_KINDS = frozenset(
    {
        "integer_flow",
        "integer_map",
        "fractional_continuous",
        "fractional_difference",
        "sampled_data",
    }
)
TRAJECTORY_MEMORY_POLICIES = frozenset(
    {
        "not_applicable",
        "full_history",
        "finite_window",
        "restart",
        "fast_approximation",
    }
)
TRAJECTORY_TIME_COORDINATES = frozenset(
    {
        "physical_time",
        "iteration_index",
        "log_t_over_lower_terminal",
        "conformable_clock",
        "user_defined",
    }
)
ANALYSIS_BACKENDS = frozenset(
    {"hafo_python", "hafo_numba", "hafo_c", "julia", "external"}
)
ANALYSIS_STATUSES = frozenset(
    {
        "finite_numerical_diagnostic",
        "validated_reference",
        "experimental",
    }
)
FRACTIONAL_TRAJECTORY_WARNING = (
    "The sampled coordinates of a fractional-order system are not generally "
    "the complete hereditary state; the result describes only the supplied "
    "projection and recorded history policy."
)
PREHISTORY_KINDS = frozenset(
    {
        "not_applicable",
        "point_initial_value",
        "sampled",
        "analytic_reference",
        "unknown",
    }
)


def _as_real_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        raise TypeError(f"{name} must not have Boolean dtype.")
    if np.iscomplexobj(raw):
        raise TypeError(f"{name} must be real-valued, not complex.")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric array.") from exc
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _json_scalar(value: Any, *, path: str) -> Any:
    if isinstance(value, np.generic):
        return _json_scalar(value.item(), path=path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise TypeError(f"{path} contains a non-finite float.")
        return value
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}.")


def _freeze_json(value: Any, *, path: str) -> Any:
    """Detach JSON-oriented provenance and make containers immutable."""

    if isinstance(value, np.ndarray):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value.tolist())
        )
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key.")
            copied[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    return _json_scalar(value, path=path)


def _freeze_result_value(value: Any, *, path: str) -> Any:
    """Freeze result values while retaining numerical arrays as arrays."""

    if isinstance(value, np.ndarray):
        raw = np.asarray(value)
        if raw.dtype.kind == "b":
            result = np.array(raw, dtype=np.bool_, copy=True, order="C")
        elif np.issubdtype(raw.dtype, np.integer):
            result = np.array(raw, dtype=np.int64, copy=True, order="C")
        elif np.issubdtype(raw.dtype, np.floating):
            if not np.all(np.isfinite(raw)):
                raise TypeError(f"{path} contains non-finite values.")
            result = np.array(raw, dtype=np.float64, copy=True, order="C")
        else:
            raise TypeError(f"{path} contains an unsupported array dtype.")
        result.setflags(write=False)
        return result
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key.")
            copied[key] = _freeze_result_value(item, path=f"{path}.{key}")
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_result_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    return _json_scalar(value, path=path)


def _serializable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _uniform_sampling(t: np.ndarray) -> tuple[bool, float | None, float]:
    differences = np.diff(t)
    reference = float(differences[0])
    tolerance = max(
        128.0 * np.finfo(np.float64).eps * max(1.0, abs(reference), abs(float(t[-1]))),
        1.0e-12 * abs(reference),
    )
    uniform = bool(np.all(np.abs(differences - reference) <= tolerance))
    return uniform, reference if uniform else None, tolerance


@dataclass(frozen=True, slots=True)
class PrehistorySpec:
    """Explicit lower-terminal/prehistory provenance.

    This object only describes history supplied to or assumed by a numerical
    problem.  It does not make an existing solver consume a prehistory that the
    solver does not already support.
    """

    kind: str
    lower_terminal: float | None = None
    coverage: tuple[float, float] | None = None
    sample_times: Sequence[float] | np.ndarray | None = None
    sample_values: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None = None
    analytic_reference: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = _nonempty_text(self.kind, name="prehistory.kind").lower()
        if kind not in PREHISTORY_KINDS:
            allowed = ", ".join(sorted(PREHISTORY_KINDS))
            raise ValueError(f"prehistory.kind must be one of: {allowed}.")

        lower_terminal = self.lower_terminal
        if lower_terminal is not None:
            lower_terminal = float(lower_terminal)
            if not np.isfinite(lower_terminal):
                raise ValueError("prehistory.lower_terminal must be finite.")

        coverage: tuple[float, float] | None = None
        if self.coverage is not None:
            if len(self.coverage) != 2:
                raise ValueError("prehistory.coverage must contain two values.")
            start, stop = (float(value) for value in self.coverage)
            if not np.isfinite(start) or not np.isfinite(stop) or stop < start:
                raise ValueError(
                    "prehistory.coverage must be finite with stop >= start."
                )
            coverage = (start, stop)

        sample_times: np.ndarray | None = None
        sample_values: np.ndarray | None = None
        if self.sample_times is not None or self.sample_values is not None:
            if self.sample_times is None or self.sample_values is None:
                raise ValueError(
                    "prehistory.sample_times and sample_values must be supplied together."
                )
            sample_times = _as_real_array(
                self.sample_times,
                name="prehistory.sample_times",
                ndim=1,
            )
            if sample_times.size < 1 or (
                sample_times.size > 1 and np.any(np.diff(sample_times) <= 0.0)
            ):
                raise ValueError(
                    "prehistory.sample_times must be non-empty and strictly increasing."
                )
            raw_values = np.asarray(self.sample_values)
            if raw_values.ndim == 1:
                raw_values = raw_values[:, None]
            sample_values = _as_real_array(
                raw_values,
                name="prehistory.sample_values",
                ndim=2,
            )
            if sample_values.shape[0] != sample_times.size:
                raise ValueError(
                    "prehistory.sample_values must have one row per sample time."
                )

        analytic_reference = self.analytic_reference
        if analytic_reference is not None:
            analytic_reference = _nonempty_text(
                analytic_reference,
                name="prehistory.analytic_reference",
            )

        if kind == "not_applicable":
            if any(
                value is not None
                for value in (
                    lower_terminal,
                    coverage,
                    sample_times,
                    sample_values,
                    analytic_reference,
                )
            ):
                raise ValueError(
                    "prehistory.kind='not_applicable' cannot carry history data."
                )
        elif kind == "point_initial_value":
            if lower_terminal is None:
                raise ValueError(
                    "point_initial_value prehistory requires lower_terminal."
                )
            if sample_times is not None or analytic_reference is not None:
                raise ValueError(
                    "point_initial_value cannot carry sampled or analytic history."
                )
            if coverage is None:
                coverage = (lower_terminal, lower_terminal)
        elif kind == "sampled":
            if lower_terminal is None or sample_times is None or sample_values is None:
                raise ValueError(
                    "sampled prehistory requires lower_terminal, sample_times, and "
                    "sample_values."
                )
            if analytic_reference is not None:
                raise ValueError("sampled prehistory cannot carry analytic_reference.")
            if coverage is None:
                coverage = (float(sample_times[0]), float(sample_times[-1]))
            tolerance = 128.0 * np.finfo(float).eps * max(
                1.0,
                abs(coverage[0]),
                abs(coverage[1]),
            )
            if (
                abs(coverage[0] - float(sample_times[0])) > tolerance
                or abs(coverage[1] - float(sample_times[-1])) > tolerance
            ):
                raise ValueError(
                    "sampled prehistory coverage must match its first and last sample."
                )
            if lower_terminal > coverage[0]:
                raise ValueError(
                    "prehistory.lower_terminal cannot follow sampled coverage."
                )
        elif kind == "analytic_reference":
            if lower_terminal is None or coverage is None or analytic_reference is None:
                raise ValueError(
                    "analytic_reference prehistory requires lower_terminal, coverage, "
                    "and an importable/formula reference."
                )
            if sample_times is not None:
                raise ValueError(
                    "analytic_reference prehistory cannot also carry sampled history."
                )
            if lower_terminal > coverage[0]:
                raise ValueError(
                    "prehistory.lower_terminal cannot follow analytic coverage."
                )
        elif kind == "unknown" and (
            sample_times is not None or analytic_reference is not None
        ):
            raise ValueError(
                "unknown prehistory cannot carry data that would make its kind known."
            )

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "lower_terminal", lower_terminal)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "sample_times", sample_times)
        object.__setattr__(self, "sample_values", sample_values)
        object.__setattr__(self, "analytic_reference", analytic_reference)
        object.__setattr__(
            self,
            "metadata",
            _freeze_json(self.metadata, path="prehistory.metadata"),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PrehistorySpec":
        """Construct from a serialized mapping with strict field names."""

        allowed = {
            "kind",
            "lower_terminal",
            "coverage",
            "sample_times",
            "sample_values",
            "analytic_reference",
            "metadata",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                "Unknown prehistory field(s): "
                + ", ".join(sorted(str(value) for value in unknown))
            )
        return cls(**dict(values))

    def to_serializable(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "lower_terminal": self.lower_terminal,
            "coverage": None if self.coverage is None else list(self.coverage),
            "sample_times": (
                None if self.sample_times is None else self.sample_times.tolist()
            ),
            "sample_values": (
                None if self.sample_values is None else self.sample_values.tolist()
            ),
            "analytic_reference": self.analytic_reference,
            "metadata": _serializable(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TrajectoryInput:
    """Immutable sampled trajectory with solver and history provenance."""

    t: Sequence[float] | np.ndarray
    x: Sequence[float] | Sequence[Sequence[float]] | np.ndarray
    system_kind: str
    time_coordinate: str = "physical_time"
    sampled_uniformly: bool | None = None
    projection: Sequence[str] | None = None
    transient_interval: tuple[float, float] | None = None
    derivative_definition: str | None = None
    order: Any = None
    lower_terminal_and_prehistory: PrehistorySpec | Mapping[str, Any] = field(
        default_factory=lambda: PrehistorySpec(kind="not_applicable")
    )
    memory_policy: str = "not_applicable"
    solver_and_tolerances: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _uniform_step: float | None = field(init=False, repr=False)
    _sampling_tolerance: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        t = _as_real_array(self.t, name="t", ndim=1)
        if t.size < 2:
            raise ValueError("t must contain at least two samples.")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("t must be strictly increasing without duplicates.")

        raw_x = np.asarray(self.x)
        if raw_x.ndim == 1:
            raw_x = raw_x[:, None]
        x = _as_real_array(raw_x, name="x", ndim=2)
        if x.shape[0] != t.size or x.shape[1] < 1:
            raise ValueError(
                "x must have shape (len(t), dimension) with a non-empty dimension."
            )

        system_kind = _nonempty_text(self.system_kind, name="system_kind").lower()
        if system_kind not in TRAJECTORY_SYSTEM_KINDS:
            allowed = ", ".join(sorted(TRAJECTORY_SYSTEM_KINDS))
            raise ValueError(f"system_kind must be one of: {allowed}.")
        time_coordinate = _nonempty_text(
            self.time_coordinate,
            name="time_coordinate",
        ).lower()
        if time_coordinate not in TRAJECTORY_TIME_COORDINATES:
            allowed = ", ".join(sorted(TRAJECTORY_TIME_COORDINATES))
            raise ValueError(f"time_coordinate must be one of: {allowed}.")
        if system_kind == "integer_map" and time_coordinate == "physical_time":
            raise ValueError(
                "integer_map trajectories must declare iteration_index or another "
                "explicit discrete sampling coordinate."
            )

        detected_uniform, uniform_step, sampling_tolerance = _uniform_sampling(t)
        if self.sampled_uniformly is None:
            sampled_uniformly = detected_uniform
        elif isinstance(self.sampled_uniformly, (bool, np.bool_)):
            sampled_uniformly = bool(self.sampled_uniformly)
            if sampled_uniformly and not detected_uniform:
                raise ValueError(
                    "sampled_uniformly=True contradicts the supplied time grid."
                )
        else:
            raise TypeError("sampled_uniformly must be Boolean or None.")

        if self.projection is None:
            projection = tuple(f"x{index}" for index in range(x.shape[1]))
        else:
            projection = tuple(
                _nonempty_text(value, name=f"projection[{index}]")
                for index, value in enumerate(self.projection)
            )
            if len(projection) != x.shape[1]:
                raise ValueError("projection must contain one label per x column.")
            if len(set(projection)) != len(projection):
                raise ValueError("projection labels must be unique.")

        transient_interval: tuple[float, float] | None
        if self.transient_interval is None:
            transient_interval = None
        else:
            if len(self.transient_interval) != 2:
                raise ValueError("transient_interval must contain exactly two values.")
            start, stop = (float(value) for value in self.transient_interval)
            if not np.isfinite(start) or not np.isfinite(stop) or stop < start:
                raise ValueError(
                    "transient_interval must be finite with stop >= start."
                )
            transient_interval = (start, stop)

        derivative_definition = self.derivative_definition
        if derivative_definition is not None:
            derivative_definition = _nonempty_text(
                derivative_definition,
                name="derivative_definition",
            ).lower()

        memory_policy = _nonempty_text(
            self.memory_policy,
            name="memory_policy",
        ).lower()
        if memory_policy not in TRAJECTORY_MEMORY_POLICIES:
            allowed = ", ".join(sorted(TRAJECTORY_MEMORY_POLICIES))
            raise ValueError(f"memory_policy must be one of: {allowed}.")

        fractional = system_kind.startswith("fractional_")
        if fractional and derivative_definition is None:
            raise ValueError(
                "fractional trajectories require derivative_definition provenance."
            )
        if fractional and self.order is None:
            raise ValueError("fractional trajectories require an order specification.")
        if fractional and memory_policy == "not_applicable":
            raise ValueError(
                "fractional trajectories require an explicit hereditary memory_policy."
            )
        if not fractional and memory_policy != "not_applicable":
            raise ValueError(
                "integer or generic sampled trajectories must use "
                "memory_policy='not_applicable'."
            )

        order = _freeze_json(self.order, path="order")
        if isinstance(self.lower_terminal_and_prehistory, PrehistorySpec):
            lower_terminal_and_prehistory = self.lower_terminal_and_prehistory
        elif isinstance(self.lower_terminal_and_prehistory, Mapping):
            lower_terminal_and_prehistory = PrehistorySpec.from_mapping(
                self.lower_terminal_and_prehistory
            )
        else:
            raise TypeError(
                "lower_terminal_and_prehistory must be PrehistorySpec or a mapping."
            )
        if fractional and lower_terminal_and_prehistory.kind == "not_applicable":
            raise ValueError(
                "fractional trajectories cannot use prehistory.kind="
                "'not_applicable'; explicit prehistory provenance is required."
            )
        if not fractional and lower_terminal_and_prehistory.kind != "not_applicable":
            raise ValueError(
                "integer or generic sampled trajectories require "
                "prehistory.kind='not_applicable'."
            )
        solver_and_tolerances = _freeze_json(
            self.solver_and_tolerances,
            path="solver_and_tolerances",
        )
        metadata = _freeze_json(self.metadata, path="metadata")

        object.__setattr__(self, "t", t)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "system_kind", system_kind)
        object.__setattr__(self, "time_coordinate", time_coordinate)
        object.__setattr__(self, "sampled_uniformly", sampled_uniformly)
        object.__setattr__(self, "projection", projection)
        object.__setattr__(self, "transient_interval", transient_interval)
        object.__setattr__(self, "derivative_definition", derivative_definition)
        object.__setattr__(self, "order", order)
        object.__setattr__(
            self,
            "lower_terminal_and_prehistory",
            lower_terminal_and_prehistory,
        )
        object.__setattr__(self, "memory_policy", memory_policy)
        object.__setattr__(self, "solver_and_tolerances", solver_and_tolerances)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "_uniform_step", uniform_step)
        object.__setattr__(self, "_sampling_tolerance", sampling_tolerance)

    @classmethod
    def from_simulation_result(
        cls,
        result: Any,
        *,
        projection: Sequence[str] | None = None,
        transient_interval: tuple[float, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TrajectoryInput":
        """Adapt the public ``SimulationResult`` without mutating it.

        The adapter is intentionally duck-typed to avoid making the analysis
        contract depend on the simulation facade.  It preserves both physical
        sample times and, when present, the solver's integration-coordinate
        samples.  Describing a prehistory here does not enable unsupported
        lower-terminal behavior in a solver.
        """

        required = ("times", "states", "system_kind", "method", "metadata")
        missing = [name for name in required if not hasattr(result, name)]
        if missing:
            raise TypeError(
                "result is missing SimulationResult field(s): " + ", ".join(missing)
            )
        times = np.asarray(result.times)
        states = np.asarray(result.states)
        source_metadata = dict(result.metadata)
        fractional_problem_raw = source_metadata.get("fractional_problem")
        fractional_problem = (
            dict(fractional_problem_raw)
            if isinstance(fractional_problem_raw, Mapping)
            else None
        )

        solver_metadata: dict[str, Any] = {
            "method": str(result.method),
            "step_size": getattr(result, "step_size", None),
            "requested_steps": int(getattr(result, "requested_steps", 0)),
            "completed_steps": int(getattr(result, "completed_steps", 0)),
            "status": str(getattr(result, "status", "unknown")),
            "backend": getattr(result, "backend", None),
            "backend_info": dict(getattr(result, "backend_info", {}) or {}),
            "grid_coordinate": str(
                getattr(result, "grid_coordinate", "physical_time")
            ),
        }
        integrator_times = getattr(result, "integrator_times", None)
        if integrator_times is not None:
            solver_metadata["integration_coordinate_times"] = np.asarray(
                integrator_times,
                dtype=np.float64,
            )

        provenance: dict[str, Any] = {
            "simulation_status": str(getattr(result, "status", "unknown")),
            "system_name": str(getattr(result, "system_name", "unknown")),
            "system_parameters": dict(getattr(result, "parameters", {}) or {}),
        }
        if metadata:
            provenance.update(dict(metadata))

        if fractional_problem is not None:
            raw_kind = str(result.system_kind).strip().lower()
            system_kind = (
                "fractional_difference"
                if raw_kind in {"map", "fractional_difference"}
                else "fractional_continuous"
            )
            derivative = _nonempty_text(
                fractional_problem.get("derivative"),
                name="fractional_problem.derivative",
            )
            orders = fractional_problem.get("orders")
            if orders is None:
                raise ValueError("fractional_problem metadata is missing orders.")
            source_memory_policy = str(
                fractional_problem.get("memory_policy", "")
            ).strip().lower()
            memory_aliases = {
                "full_history": "full_history",
                "finite_window": "finite_window",
                "block_restart": "restart",
                "recursive_kernel": "fast_approximation",
                "fast_history": "fast_approximation",
            }
            try:
                memory_policy = memory_aliases[source_memory_policy]
            except KeyError as exc:
                raise ValueError(
                    "Unsupported fractional memory policy in SimulationResult: "
                    f"{source_memory_policy!r}."
                ) from exc
            lower_terminal_raw = fractional_problem.get("lower_terminal")
            if lower_terminal_raw is None:
                prehistory = PrehistorySpec(kind="unknown")
            else:
                lower_terminal = float(lower_terminal_raw)
                prehistory = PrehistorySpec(
                    kind="point_initial_value",
                    lower_terminal=lower_terminal,
                    metadata={
                        "initial_condition_kind": fractional_problem.get(
                            "initial_condition_kind",
                            "unknown",
                        ),
                        "history_window": fractional_problem.get("history_window"),
                        "source_memory_policy": source_memory_policy,
                        "description_only_not_solver_enablement": True,
                    },
                )
            solver_metadata["fractional_method_options"] = fractional_problem.get(
                "method_options",
                {},
            )
            solver_metadata["fractional_kernel_parameters"] = fractional_problem.get(
                "kernel_parameters",
                {},
            )
            solver_metadata["fractional_reference_keys"] = fractional_problem.get(
                "reference_keys",
                [],
            )
            time_coordinate = (
                "iteration_index"
                if system_kind == "fractional_difference"
                else "physical_time"
            )
        else:
            raw_kind = str(result.system_kind).strip().lower()
            if raw_kind in {"map", "integer_map"}:
                system_kind = "integer_map"
                time_coordinate = "iteration_index"
            elif raw_kind in {"flow", "integer_flow"}:
                system_kind = "integer_flow"
                time_coordinate = "physical_time"
            else:
                raise ValueError(
                    "SimulationResult system_kind must identify a flow or map."
                )
            derivative = None
            orders = None
            memory_policy = "not_applicable"
            prehistory = PrehistorySpec(kind="not_applicable")

        return cls(
            t=times,
            x=states,
            system_kind=system_kind,
            time_coordinate=time_coordinate,
            projection=projection,
            transient_interval=transient_interval,
            derivative_definition=derivative,
            order=orders,
            lower_terminal_and_prehistory=prehistory,
            memory_policy=memory_policy,
            solver_and_tolerances=solver_metadata,
            metadata=provenance,
        )

    @property
    def sample_count(self) -> int:
        return int(self.t.size)

    @property
    def dimension(self) -> int:
        return int(self.x.shape[1])

    @property
    def uniform_step(self) -> float | None:
        return self._uniform_step if self.sampled_uniformly else None

    @property
    def sampling_tolerance(self) -> float:
        return float(self._sampling_tolerance)

    @property
    def scientific_warnings(self) -> tuple[str, ...]:
        warnings: list[str] = []
        if self.system_kind.startswith("fractional_"):
            warnings.append(FRACTIONAL_TRAJECTORY_WARNING)
            if self.lower_terminal_and_prehistory.kind == "unknown":
                warnings.append(
                    "The fractional prehistory is explicitly unknown; diagnostics "
                    "cannot be compared as if the hereditary initialization matched."
                )
        if not self.sampled_uniformly:
            warnings.append(
                "The time grid is irregular; analyses that assume a constant "
                "sampling interval must reject or explicitly resample it."
            )
        return tuple(warnings)

    def component(self, component: int | str) -> np.ndarray:
        """Return a read-only detached scalar component by index or label."""

        if isinstance(component, str):
            try:
                index = self.projection.index(component)
            except ValueError as exc:
                raise KeyError(f"Unknown projection label: {component!r}.") from exc
        elif isinstance(component, (bool, np.bool_)):
            raise TypeError("component must be an integer index or projection label.")
        else:
            try:
                index = int(operator_index(component))
            except TypeError as exc:
                raise TypeError(
                    "component must be an integer index or projection label."
                ) from exc
        if index < 0 or index >= self.dimension:
            raise IndexError("component index is outside the trajectory dimension.")
        result = np.array(self.x[:, index], dtype=np.float64, copy=True)
        result.setflags(write=False)
        return result

    def to_serializable(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""

        return {
            "t": self.t.tolist(),
            "x": self.x.tolist(),
            "sampled_uniformly": bool(self.sampled_uniformly),
            "time_coordinate": self.time_coordinate,
            "projection": list(self.projection),
            "transient_interval": (
                None
                if self.transient_interval is None
                else list(self.transient_interval)
            ),
            "system_kind": self.system_kind,
            "derivative_definition": self.derivative_definition,
            "order": _serializable(self.order),
            "lower_terminal_and_prehistory": (
                self.lower_terminal_and_prehistory.to_serializable()
            ),
            "memory_policy": self.memory_policy,
            "solver_and_tolerances": _serializable(self.solver_and_tolerances),
            "metadata": _serializable(self.metadata),
        }

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 of samples and semantic metadata."""

        digest = hashlib.sha256()
        for array in (self.t, self.x):
            canonical = np.asarray(array, dtype="<f8", order="C")
            digest.update(str(canonical.shape).encode("ascii"))
            digest.update(canonical.tobytes(order="C"))
        metadata = self.to_serializable()
        metadata.pop("t")
        metadata.pop("x")
        digest.update(
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Backend-neutral result envelope for finite trajectory diagnostics."""

    method: str
    values: Mapping[str, Any]
    parameters: Mapping[str, Any]
    backend: str
    status: str
    trajectory_fingerprint: str
    package_versions_and_hashes: Mapping[str, Any] = field(default_factory=dict)
    warnings: Sequence[str] = field(default_factory=tuple)
    references: Sequence[str] = field(default_factory=tuple)
    evidence_scope: str = "finite_sample_empirical_trajectory_diagnostic"

    def __post_init__(self) -> None:
        method = _nonempty_text(self.method, name="method")
        backend = _nonempty_text(self.backend, name="backend").lower()
        if backend not in ANALYSIS_BACKENDS:
            allowed = ", ".join(sorted(ANALYSIS_BACKENDS))
            raise ValueError(f"backend must be one of: {allowed}.")
        status = _nonempty_text(self.status, name="status").lower()
        if status not in ANALYSIS_STATUSES:
            allowed = ", ".join(sorted(ANALYSIS_STATUSES))
            raise ValueError(f"status must be one of: {allowed}.")
        fingerprint = _nonempty_text(
            self.trajectory_fingerprint,
            name="trajectory_fingerprint",
        ).lower()
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("trajectory_fingerprint must be a SHA-256 hex digest.")
        evidence_scope = _nonempty_text(
            self.evidence_scope,
            name="evidence_scope",
        )
        warnings = tuple(
            _nonempty_text(value, name=f"warnings[{index}]")
            for index, value in enumerate(self.warnings)
        )
        references = tuple(
            _nonempty_text(value, name=f"references[{index}]")
            for index, value in enumerate(self.references)
        )
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "values", _freeze_result_value(self.values, path="values"))
        object.__setattr__(
            self,
            "parameters",
            _freeze_json(self.parameters, path="parameters"),
        )
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "trajectory_fingerprint", fingerprint)
        object.__setattr__(
            self,
            "package_versions_and_hashes",
            _freeze_json(
                self.package_versions_and_hashes,
                path="package_versions_and_hashes",
            ),
        )
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "evidence_scope", evidence_scope)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "values": _serializable(self.values),
            "parameters": _serializable(self.parameters),
            "backend": self.backend,
            "package_versions_and_hashes": _serializable(
                self.package_versions_and_hashes
            ),
            "status": self.status,
            "warnings": list(self.warnings),
            "references": list(self.references),
            "evidence_scope": self.evidence_scope,
            "trajectory_fingerprint": self.trajectory_fingerprint,
        }


__all__ = [
    "ANALYSIS_BACKENDS",
    "ANALYSIS_STATUSES",
    "FRACTIONAL_TRAJECTORY_WARNING",
    "PREHISTORY_KINDS",
    "TRAJECTORY_MEMORY_POLICIES",
    "TRAJECTORY_SYSTEM_KINDS",
    "TRAJECTORY_TIME_COORDINATES",
    "AnalysisResult",
    "PrehistorySpec",
    "TrajectoryInput",
]
