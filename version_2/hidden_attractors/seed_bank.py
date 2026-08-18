"""Unified, provenance-preserving seed bank for localization campaigns.

Stability: experimental
    This module unifies seeds produced by describing functions, continuation,
    perpetual-point constructions, critical surfaces, connecting curves, KCC
    rankings, eigendirections, and edge tracking.  A seed record is only an
    initialisation proposal; it is never evidence of attraction, chaos, or
    hiddenness.

The existing :class:`hidden_attractors.workflows.protocol.UnifiedSeedRecord`
is intentionally left unchanged because it is the official schema for the two
maintained classical Lur'e families.  :class:`SeedRecord` is a broader campaign
record and provides adapters for those legacy/current mappings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .io import read_csv_rows, read_json, write_csv, write_json


SEED_BANK_SCHEMA_VERSION = "1.0"

SEED_ROUTES = (
    "describing_function",
    "machado",
    "continuation",
    "perpetual_point",
    "fractional_perpetual_point",
    "critical_surface",
    "connecting_curve",
    "kcc",
    "eigen_direction",
    "edge_tracking",
    "manual",
    "imported",
)

ORDER_KINDS = ("integer", "caputo")
INITIALIZATION_KINDS = (
    "point_initial_value",
    "sampled_history",
    "analytic_history_reference",
    "continued_history",
)

SeedRoute = Literal[
    "describing_function",
    "machado",
    "continuation",
    "perpetual_point",
    "fractional_perpetual_point",
    "critical_surface",
    "connecting_curve",
    "kcc",
    "eigen_direction",
    "edge_tracking",
    "manual",
    "imported",
]
OrderKind = Literal["integer", "caputo"]
InitializationKind = Literal[
    "point_initial_value",
    "sampled_history",
    "analytic_history_reference",
    "continued_history",
]


def _nonempty(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string.")
    return text


def _finite_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite vector.")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("seed metadata cannot contain non-finite floats.")
        return value
    raise TypeError(
        "seed metadata contains an unsupported non-JSON value of type "
        f"{type(value).__name__}."
    )


def _route_from_mapping(values: Mapping[str, Any]) -> str:
    raw = str(values.get("route", values.get("family", "imported"))).strip().lower()
    aliases = {
        "lure_classical_centered": "describing_function",
        "lure_classical_biased": "describing_function",
        "machado_centered": "machado",
        "machado_biased": "machado",
        "df": "describing_function",
        "pp": "perpetual_point",
        "fpp": "fractional_perpetual_point",
        "surface": "critical_surface",
        "cc": "connecting_curve",
        "eigen": "eigen_direction",
        "edge": "edge_tracking",
    }
    return aliases.get(raw, raw if raw in SEED_ROUTES else "imported")


@dataclass(frozen=True, slots=True)
class SeedRecord:
    """One seed proposal with numerical and hereditary provenance.

    ``state`` is the point used to initialise an integer IVP or the value at
    the lower terminal of a Caputo IVP.  It is not, by itself, a complete
    continued Caputo history.  When ``initialization_kind`` is history based,
    ``history_reference`` and ``history_coverage`` identify that additional
    state.
    """

    seed_id: str
    system_id: str
    route: SeedRoute
    state: tuple[float, ...]
    order_kind: OrderKind = "integer"
    q: float = 1.0
    parameter_set_id: str = "default"
    initialization_kind: InitializationKind = "point_initial_value"
    lower_terminal: float | None = None
    initial_time: float | None = None
    history_reference: str | None = None
    history_coverage: tuple[float, float] | None = None
    source_artifact: str = ""
    source_record_id: str = ""
    generation_residual: float | None = None
    score: float | None = None
    priority: int = 100
    parent_seed_id: str | None = None
    transform_name: str = "identity"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed_id", _nonempty(self.seed_id, "seed_id"))
        object.__setattr__(self, "system_id", _nonempty(self.system_id, "system_id"))
        object.__setattr__(
            self,
            "parameter_set_id",
            _nonempty(self.parameter_set_id, "parameter_set_id"),
        )
        if self.route not in SEED_ROUTES:
            raise ValueError(f"route must be one of: {', '.join(SEED_ROUTES)}.")
        if self.order_kind not in ORDER_KINDS:
            raise ValueError("order_kind must be 'integer' or 'caputo'.")
        if self.initialization_kind not in INITIALIZATION_KINDS:
            raise ValueError(
                "initialization_kind must be point_initial_value or an explicit "
                "history-based kind."
            )
        state = _finite_tuple(self.state, "state")
        object.__setattr__(self, "state", state)
        q = float(self.q)
        if not math.isfinite(q) or not 0.0 < q <= 1.0:
            raise ValueError("q must satisfy 0 < q <= 1.")
        if self.order_kind == "integer" and not math.isclose(q, 1.0, abs_tol=1.0e-12):
            raise ValueError("integer seeds require q=1.")
        object.__setattr__(self, "q", q)
        if self.lower_terminal is not None:
            lower_terminal = float(self.lower_terminal)
            if not math.isfinite(lower_terminal):
                raise ValueError("lower_terminal must be finite when supplied.")
            object.__setattr__(self, "lower_terminal", lower_terminal)
        if self.initial_time is not None:
            initial_time = float(self.initial_time)
            if not math.isfinite(initial_time):
                raise ValueError("initial_time must be finite when supplied.")
            object.__setattr__(self, "initial_time", initial_time)
        coverage = self.history_coverage
        if coverage is not None:
            if len(coverage) != 2:
                raise ValueError("history_coverage must contain start and stop.")
            start, stop = (float(value) for value in coverage)
            if not math.isfinite(start) or not math.isfinite(stop) or stop < start:
                raise ValueError("history_coverage must be finite with stop >= start.")
            object.__setattr__(self, "history_coverage", (start, stop))
        history_based = self.initialization_kind != "point_initial_value"
        if history_based and (not self.history_reference or coverage is None):
            raise ValueError(
                "history-based seeds require history_reference and history_coverage."
            )
        if not history_based and (self.history_reference is not None or coverage is not None):
            raise ValueError(
                "point_initial_value seeds cannot carry history_reference or history_coverage."
            )
        if self.order_kind == "integer" and (
            history_based
            or self.lower_terminal is not None
            or self.initial_time is not None
            or self.history_reference is not None
            or coverage is not None
        ):
            raise ValueError("integer seeds cannot carry fractional prehistory fields.")
        if self.order_kind == "caputo" and self.lower_terminal is None:
            raise ValueError("Caputo seeds require an explicit lower_terminal.")
        if self.order_kind == "caputo" and self.initial_time is None:
            raise ValueError("Caputo seeds require an explicit initial_time.")
        if self.order_kind == "caputo":
            assert self.lower_terminal is not None and self.initial_time is not None
            if self.initial_time < self.lower_terminal:
                raise ValueError("Caputo initial_time must satisfy initial_time >= lower_terminal.")
            if not history_based and not math.isclose(
                self.initial_time,
                self.lower_terminal,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "a Caputo point_initial_value seed must start at lower_terminal; "
                    "later initial_time requires an explicit history-based kind."
                )
            if history_based:
                assert coverage is not None
                if coverage[0] > self.lower_terminal or coverage[1] < self.initial_time:
                    raise ValueError(
                        "history_coverage must start at/before lower_terminal and end "
                        "at/after initial_time."
                    )
        for name in ("generation_residual", "score"):
            value = getattr(self, name)
            if value is not None:
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"{name} must be finite when supplied.")
                object.__setattr__(self, name, numeric)
        if int(self.priority) < 0:
            raise ValueError("priority must be a non-negative integer.")
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "transform_name", _nonempty(self.transform_name, "transform_name"))
        # Validate JSON provenance without silently stringifying unsupported data.
        object.__setattr__(self, "metadata", _jsonable(dict(self.metadata)))

    @property
    def dimension(self) -> int:
        return len(self.state)

    def semantic_partition(self) -> tuple[Any, ...]:
        """Return fields that must agree before geometric deduplication."""

        return (
            self.system_id,
            self.parameter_set_id,
            self.order_kind,
            round(self.q, 14),
            self.initialization_kind,
            self.lower_terminal,
            self.initial_time,
            self.history_reference,
            self.history_coverage,
            self.dimension,
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        system_id: str | None = None,
        order_kind: OrderKind | None = None,
        parameter_set_id: str | None = None,
        lower_terminal: float | None = None,
        initial_time: float | None = None,
    ) -> "SeedRecord":
        """Adapt current/legacy seed dictionaries without losing source fields."""

        state_value = values.get(
            "state",
            values.get("x0", values.get("seed", values.get("robust_start"))),
        )
        if state_value is None:
            raise ValueError("seed mapping requires state, x0, seed, or robust_start.")
        seed_id = values.get("seed_id", values.get("candidate_id"))
        if seed_id is None:
            raise ValueError("seed mapping requires seed_id or candidate_id.")
        q = float(values.get("q", 1.0))
        mapped_order = values.get("order_kind")
        resolved_order = order_kind or (None if mapped_order is None else str(mapped_order))
        if resolved_order is None:
            resolved_order = "integer" if math.isclose(q, 1.0) else "caputo"
        family = str(values.get("family", ""))
        known = {
            "candidate_id",
            "seed_id",
            "system_id",
            "route",
            "family",
            "state",
            "x0",
            "seed",
            "robust_start",
            "order_kind",
            "q",
            "parameter_set_id",
            "initialization_kind",
            "lower_terminal",
            "initial_time",
            "current_time",
            "history_reference",
            "history_coverage",
            "source_artifact",
            "source_config",
            "source_record_id",
            "generation_residual",
            "harmonic_residual",
            "residual_abs",
            "score",
            "priority",
            "parent_seed_id",
            "transform_name",
            "metadata",
        }
        metadata = dict(values.get("metadata", {}))
        metadata.update({key: _jsonable(value) for key, value in values.items() if key not in known})
        if family:
            metadata.setdefault("source_family", family)
        residual = values.get("generation_residual", values.get("harmonic_residual", values.get("residual_abs")))
        return cls(
            seed_id=str(seed_id),
            system_id=str(system_id or values.get("system_id", "unspecified")),
            route=_route_from_mapping(values),
            state=tuple(float(value) for value in state_value),
            order_kind=resolved_order,
            q=q,
            parameter_set_id=str(parameter_set_id or values.get("parameter_set_id", "default")),
            initialization_kind=str(values.get("initialization_kind", "point_initial_value")),
            lower_terminal=values.get("lower_terminal", lower_terminal),
            initial_time=values.get(
                "initial_time",
                values.get("current_time", initial_time),
            ),
            history_reference=values.get("history_reference"),
            history_coverage=(
                None
                if values.get("history_coverage") is None
                else tuple(float(item) for item in values["history_coverage"])
            ),
            source_artifact=str(values.get("source_artifact", values.get("source_config", ""))),
            source_record_id=str(values.get("source_record_id", seed_id)),
            generation_residual=None if residual in {None, ""} else float(residual),
            score=None if values.get("score") in {None, ""} else float(values["score"]),
            priority=int(values.get("priority", 100)),
            parent_seed_id=values.get("parent_seed_id"),
            transform_name=str(values.get("transform_name", "identity")),
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class SymmetryTransform:
    """Declared affine state-space symmetry used to quotient a seed bank.

    The bank only applies the transform.  A separate mathematical/numerical
    check must establish ``F(Sx+c)=S F(x)`` for an affine transform ``Sx+c``.
    """

    name: str
    matrix: tuple[tuple[float, ...], ...]
    offset: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "symmetry name"))
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("symmetry matrix must be finite, square, and non-empty.")
        if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) <= np.finfo(float).eps:
            raise ValueError("symmetry matrix must be finite and invertible.")
        object.__setattr__(self, "matrix", tuple(tuple(float(v) for v in row) for row in matrix))
        if self.offset is None:
            offset = tuple(0.0 for _ in range(matrix.shape[0]))
        else:
            offset = _finite_tuple(self.offset, "symmetry offset")
            if len(offset) != matrix.shape[0]:
                raise ValueError("symmetry offset dimension must match its matrix.")
        object.__setattr__(self, "offset", offset)

    @property
    def dimension(self) -> int:
        return len(self.matrix)

    def apply(self, state: Sequence[float]) -> tuple[float, ...]:
        values = np.asarray(state, dtype=float)
        if values.shape != (self.dimension,):
            raise ValueError(
                f"symmetry {self.name!r} requires dimension {self.dimension}."
            )
        transformed = np.asarray(self.matrix) @ values + np.asarray(self.offset)
        return tuple(float(value) for value in transformed)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def inversion(cls, dimension: int) -> "SymmetryTransform":
        if int(dimension) < 1:
            raise ValueError("dimension must be positive.")
        matrix = -np.eye(int(dimension), dtype=float)
        return cls("inversion", tuple(tuple(float(v) for v in row) for row in matrix))


@dataclass(frozen=True, slots=True)
class SeedMembership:
    """Deduplication annotation for one original seed record."""

    record: SeedRecord
    representative_seed_id: str
    symmetry_group_id: str
    is_representative: bool
    duplicate_of: str | None
    matched_symmetry: str
    normalized_distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.record.to_dict(),
            "representative_seed_id": self.representative_seed_id,
            "symmetry_group_id": self.symmetry_group_id,
            "is_representative": self.is_representative,
            "duplicate_of": self.duplicate_of,
            "matched_symmetry": self.matched_symmetry,
            "normalized_distance": self.normalized_distance,
        }


@dataclass(frozen=True, slots=True)
class SeedBank:
    """Immutable result of scale-aware, symmetry-aware deduplication."""

    memberships: tuple[SeedMembership, ...]
    coordinate_scale: tuple[float, ...]
    absolute_tolerance: float
    relative_tolerance: float
    symmetries: tuple[SymmetryTransform, ...] = ()
    periodic_coordinates: Mapping[int, float] = field(default_factory=dict)
    symmetry_group_is_complete: bool = True
    schema_version: str = SEED_BANK_SCHEMA_VERSION

    @property
    def representatives(self) -> tuple[SeedRecord, ...]:
        return tuple(item.record for item in self.memberships if item.is_representative)

    @property
    def records(self) -> tuple[SeedRecord, ...]:
        return tuple(item.record for item in self.memberships)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "coordinate_scale": list(self.coordinate_scale),
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "symmetries": [symmetry.to_dict() for symmetry in self.symmetries],
            "periodic_coordinates": {
                str(index): float(period)
                for index, period in sorted(self.periodic_coordinates.items())
            },
            "symmetry_group_is_complete": self.symmetry_group_is_complete,
            "n_records": len(self.memberships),
            "n_representatives": len(self.representatives),
            "seeds": [membership.to_dict() for membership in self.memberships],
            "scientific_scope": (
                "seed_initialization_and_deduplication_only; not evidence of "
                "attraction, chaos, or hiddenness"
            ),
        }


def scaled_state_distance(
    left: Sequence[float],
    right: Sequence[float],
    coordinate_scale: Sequence[float],
    *,
    periodic_coordinates: Mapping[int, float] | None = None,
) -> float:
    """Return a scaled Euclidean distance, with optional wrapped coordinates."""

    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    scale = np.asarray(coordinate_scale, dtype=float)
    if a.shape != b.shape or scale.shape != a.shape:
        raise ValueError("states and coordinate_scale must have the same shape.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("states must be finite.")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("coordinate_scale must be finite and strictly positive.")
    delta = a - b
    for raw_index, raw_period in dict(periodic_coordinates or {}).items():
        index = int(raw_index)
        period = float(raw_period)
        if index < 0 or index >= delta.size:
            raise ValueError("periodic coordinate index is outside the state dimension.")
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("periodic coordinate periods must be finite and positive.")
        delta[index] = (delta[index] + 0.5 * period) % period - 0.5 * period
    return float(np.linalg.norm(delta / scale))


def _principal_state(
    state: Sequence[float],
    periodic_coordinates: Mapping[int, float],
) -> np.ndarray:
    values = np.asarray(state, dtype=float).copy()
    for index, period in periodic_coordinates.items():
        values[int(index)] = (values[int(index)] + 0.5 * float(period)) % float(period) - 0.5 * float(period)
    return values


def _priority_key(record: SeedRecord) -> tuple[Any, ...]:
    residual = record.generation_residual
    return (
        record.semantic_partition(),
        record.priority,
        float("inf") if residual is None else abs(float(residual)),
        record.seed_id,
    )


def _distance_to_orbit(
    state: tuple[float, ...],
    representative: tuple[float, ...],
    scale: tuple[float, ...],
    symmetries: Sequence[SymmetryTransform],
    periodic_coordinates: Mapping[int, float],
) -> tuple[float, str]:
    best = scaled_state_distance(
        state,
        representative,
        scale,
        periodic_coordinates=periodic_coordinates,
    )
    best_name = "identity"
    for symmetry in symmetries:
        if symmetry.dimension != len(state):
            continue
        distance = scaled_state_distance(
            symmetry.apply(state),
            representative,
            scale,
            periodic_coordinates=periodic_coordinates,
        )
        if distance < best:
            best = distance
            best_name = symmetry.name
    return best, best_name


def build_seed_bank(
    records: Sequence[SeedRecord],
    *,
    coordinate_scale: Sequence[float],
    symmetries: Sequence[SymmetryTransform] = (),
    periodic_coordinates: Mapping[int, float] | None = None,
    symmetry_group_is_complete: bool = False,
    absolute_tolerance: float = 1.0e-8,
    relative_tolerance: float = 1.0e-6,
) -> SeedBank:
    """Build a deterministic bank while preserving every input record.

    Deduplication is performed only inside an identical semantic partition
    (system, parameters, order, history contract, and dimension).  The first
    representative in priority/residual/identifier order is retained, and all
    duplicate routes remain as annotated members of its symmetry orbit.

    ``symmetry_group_is_complete=True`` is an assertion by the caller that the
    supplied transforms, together with the implicit identity, enumerate a
    complete finite group.  This function does not prove closure or covariance
    of the vector field.  History-based Caputo seeds are rejected when such
    transforms are requested because transforming a point does not transform
    its hereditary state.
    """

    if not records:
        raise ValueError("records cannot be empty.")
    identifiers = [record.seed_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("seed_id values must be unique.")
    dimensions = {record.dimension for record in records}
    if len(dimensions) != 1:
        raise ValueError("one SeedBank must contain a single state dimension.")
    dimension = next(iter(dimensions))
    scale = _finite_tuple(coordinate_scale, "coordinate_scale")
    if len(scale) != dimension or any(value <= 0.0 for value in scale):
        raise ValueError("coordinate_scale must be positive with state dimension.")
    atol = float(absolute_tolerance)
    rtol = float(relative_tolerance)
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0.0 or rtol < 0.0:
        raise ValueError("deduplication tolerances must be finite and non-negative.")
    periodic = {int(index): float(period) for index, period in dict(periodic_coordinates or {}).items()}
    for index, period in periodic.items():
        if index < 0 or index >= dimension:
            raise ValueError("periodic coordinate index is outside the state dimension.")
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("periodic coordinate periods must be finite and positive.")
    symmetry_tuple = tuple(symmetries)
    names = [symmetry.name for symmetry in symmetry_tuple]
    if len(set(names)) != len(names):
        raise ValueError("symmetry names must be unique.")
    for symmetry in symmetry_tuple:
        if symmetry.dimension != dimension:
            raise ValueError("every symmetry must match the bank state dimension.")
    if symmetry_tuple and not bool(symmetry_group_is_complete):
        raise ValueError(
            "symmetries must be declared as the complete finite group (identity is implicit); "
            "use periodic_coordinates for cylindrical translations."
        )
    if symmetry_tuple and any(
        record.initialization_kind != "point_initial_value" for record in records
    ):
        raise ValueError(
            "history-based Caputo seeds cannot be deduplicated by state symmetry "
            "without an explicitly transformed hereditary state/history reference."
        )

    ordered = sorted(tuple(records), key=_priority_key)
    representatives: list[tuple[SeedRecord, str]] = []
    annotations: dict[str, SeedMembership] = {}
    for record in ordered:
        match: tuple[SeedRecord, str, float, str] | None = None
        state_norm = float(
            np.linalg.norm(_principal_state(record.state, periodic) / np.asarray(scale))
        )
        for representative, group_id in representatives:
            if representative.semantic_partition() != record.semantic_partition():
                continue
            distance, symmetry_name = _distance_to_orbit(
                record.state,
                representative.state,
                scale,
                symmetry_tuple,
                periodic,
            )
            representative_norm = float(
                np.linalg.norm(
                    _principal_state(representative.state, periodic) / np.asarray(scale)
                )
            )
            tolerance = atol + rtol * max(1.0, state_norm, representative_norm)
            if distance <= tolerance and (match is None or distance < match[2]):
                match = (representative, group_id, distance, symmetry_name)
        if match is None:
            group_id = f"orbit_{len(representatives) + 1:04d}"
            representatives.append((record, group_id))
            annotations[record.seed_id] = SeedMembership(
                record=record,
                representative_seed_id=record.seed_id,
                symmetry_group_id=group_id,
                is_representative=True,
                duplicate_of=None,
                matched_symmetry="identity",
                normalized_distance=0.0,
            )
        else:
            representative, group_id, distance, symmetry_name = match
            annotations[record.seed_id] = SeedMembership(
                record=record,
                representative_seed_id=representative.seed_id,
                symmetry_group_id=group_id,
                is_representative=False,
                duplicate_of=representative.seed_id,
                matched_symmetry=symmetry_name,
                normalized_distance=float(distance),
            )
    grouped_memberships: list[SeedMembership] = []
    for representative, _group_id in representatives:
        grouped_memberships.append(annotations[representative.seed_id])
        grouped_memberships.extend(
            sorted(
                (
                    membership
                    for membership in annotations.values()
                    if not membership.is_representative
                    and membership.representative_seed_id == representative.seed_id
                ),
                key=lambda membership: membership.record.seed_id,
            )
        )
    memberships = tuple(grouped_memberships)
    return SeedBank(
        memberships=memberships,
        coordinate_scale=scale,
        absolute_tolerance=atol,
        relative_tolerance=rtol,
        symmetries=symmetry_tuple,
        periodic_coordinates=periodic,
        symmetry_group_is_complete=(not symmetry_tuple) or bool(symmetry_group_is_complete),
    )


SEED_BANK_CSV_FIELDS = (
    "schema_version",
    "coordinate_scale",
    "periodic_coordinates_json",
    "absolute_tolerance",
    "relative_tolerance",
    "symmetry_group_is_complete",
    "symmetries_json",
    "seed_id",
    "system_id",
    "parameter_set_id",
    "route",
    "order_kind",
    "q",
    "state",
    "initialization_kind",
    "lower_terminal",
    "initial_time",
    "history_reference",
    "history_start",
    "history_stop",
    "source_artifact",
    "source_record_id",
    "generation_residual",
    "score",
    "priority",
    "symmetry_group_id",
    "parent_seed_id",
    "transform_name",
    "matched_symmetry",
    "representative_seed_id",
    "is_representative",
    "duplicate_of",
    "normalized_distance",
    "metadata_json",
)


def seed_bank_csv_rows(bank: SeedBank) -> list[dict[str, Any]]:
    """Flatten bank memberships to stable, dimension-independent CSV rows."""

    rows: list[dict[str, Any]] = []
    for membership in bank.memberships:
        record = membership.record
        coverage = record.history_coverage
        rows.append(
            {
                "schema_version": bank.schema_version,
                "coordinate_scale": ";".join(
                    f"{value:.17g}" for value in bank.coordinate_scale
                ),
                "periodic_coordinates_json": json.dumps(
                    {str(index): period for index, period in bank.periodic_coordinates.items()},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "absolute_tolerance": bank.absolute_tolerance,
                "relative_tolerance": bank.relative_tolerance,
                "symmetry_group_is_complete": bank.symmetry_group_is_complete,
                "symmetries_json": json.dumps(
                    [symmetry.to_dict() for symmetry in bank.symmetries],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "seed_id": record.seed_id,
                "system_id": record.system_id,
                "parameter_set_id": record.parameter_set_id,
                "route": record.route,
                "order_kind": record.order_kind,
                "q": record.q,
                "state": ";".join(f"{value:.17g}" for value in record.state),
                "initialization_kind": record.initialization_kind,
                "lower_terminal": "" if record.lower_terminal is None else record.lower_terminal,
                "initial_time": "" if record.initial_time is None else record.initial_time,
                "history_reference": record.history_reference or "",
                "history_start": "" if coverage is None else coverage[0],
                "history_stop": "" if coverage is None else coverage[1],
                "source_artifact": record.source_artifact,
                "source_record_id": record.source_record_id,
                "generation_residual": (
                    "" if record.generation_residual is None else record.generation_residual
                ),
                "score": "" if record.score is None else record.score,
                "priority": record.priority,
                "symmetry_group_id": membership.symmetry_group_id,
                "parent_seed_id": record.parent_seed_id or "",
                "transform_name": record.transform_name,
                "matched_symmetry": membership.matched_symmetry,
                "representative_seed_id": membership.representative_seed_id,
                "is_representative": membership.is_representative,
                "duplicate_of": membership.duplicate_of or "",
                "normalized_distance": membership.normalized_distance,
                "metadata_json": json.dumps(record.metadata, sort_keys=True, separators=(",", ":")),
            }
        )
    return rows


def write_seed_bank(output_dir: str | Path, bank: SeedBank) -> dict[str, Path]:
    """Write ``seed_bank.json`` and ``seed_bank.csv`` with matching records."""

    directory = Path(output_dir)
    json_path = directory / "seed_bank.json"
    csv_path = directory / "seed_bank.csv"
    write_json(json_path, bank.to_dict())
    write_csv(csv_path, seed_bank_csv_rows(bank), SEED_BANK_CSV_FIELDS)
    return {"json": json_path, "csv": csv_path}


def _optional_float(value: Any) -> float | None:
    return None if value in {None, ""} else float(value)


def _optional_text(value: Any) -> str | None:
    return None if value in {None, ""} else str(value)


def _serialized_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"cannot parse Boolean seed-bank value: {value!r}.")


def _record_from_serialized(row: Mapping[str, Any]) -> SeedRecord:
    raw_state = row.get("state")
    if isinstance(raw_state, str):
        state = tuple(float(value) for value in raw_state.split(";") if value != "")
    else:
        state = tuple(float(value) for value in raw_state or ())
    raw_coverage = row.get("history_coverage")
    if raw_coverage is None:
        start = _optional_float(row.get("history_start"))
        stop = _optional_float(row.get("history_stop"))
        coverage = None if start is None and stop is None else (start, stop)
    else:
        coverage = tuple(float(value) for value in raw_coverage)
    if coverage is not None and (coverage[0] is None or coverage[1] is None):
        raise ValueError("serialized history coverage requires both endpoints.")
    raw_metadata = row.get("metadata", row.get("metadata_json", {}))
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else dict(raw_metadata)
    return SeedRecord(
        seed_id=str(row["seed_id"]),
        system_id=str(row["system_id"]),
        route=str(row["route"]),
        state=state,
        order_kind=str(row["order_kind"]),
        q=float(row["q"]),
        parameter_set_id=str(row["parameter_set_id"]),
        initialization_kind=str(row["initialization_kind"]),
        lower_terminal=_optional_float(row.get("lower_terminal")),
        initial_time=_optional_float(row.get("initial_time")),
        history_reference=_optional_text(row.get("history_reference")),
        history_coverage=None if coverage is None else (float(coverage[0]), float(coverage[1])),
        source_artifact=str(row.get("source_artifact", "")),
        source_record_id=str(row.get("source_record_id", "")),
        generation_residual=_optional_float(row.get("generation_residual")),
        score=_optional_float(row.get("score")),
        priority=int(row.get("priority", 100)),
        parent_seed_id=_optional_text(row.get("parent_seed_id")),
        transform_name=str(row.get("transform_name", "identity")),
        metadata=metadata,
    )


def _membership_from_serialized(row: Mapping[str, Any]) -> SeedMembership:
    record = _record_from_serialized(row)
    duplicate = _optional_text(row.get("duplicate_of"))
    return SeedMembership(
        record=record,
        representative_seed_id=str(row["representative_seed_id"]),
        symmetry_group_id=str(row["symmetry_group_id"]),
        is_representative=_serialized_bool(row["is_representative"]),
        duplicate_of=duplicate,
        matched_symmetry=str(row.get("matched_symmetry", "identity")),
        normalized_distance=float(row["normalized_distance"]),
    )


def load_seed_bank(path: str | Path) -> SeedBank:
    """Load a bank written by :func:`write_seed_bank` from JSON or CSV."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = read_json(source)
        rows = payload.get("seeds", [])
        if not rows:
            raise ValueError("seed-bank JSON contains no seed rows.")
        symmetries = tuple(
            SymmetryTransform(
                name=str(item["name"]),
                matrix=tuple(tuple(float(value) for value in row) for row in item["matrix"]),
                offset=tuple(float(value) for value in item["offset"]),
            )
            for item in payload.get("symmetries", [])
        )
        return SeedBank(
            memberships=tuple(_membership_from_serialized(row) for row in rows),
            coordinate_scale=tuple(float(value) for value in payload["coordinate_scale"]),
            absolute_tolerance=float(payload["absolute_tolerance"]),
            relative_tolerance=float(payload["relative_tolerance"]),
            symmetries=symmetries,
            periodic_coordinates={
                int(index): float(period)
                for index, period in payload.get("periodic_coordinates", {}).items()
            },
            symmetry_group_is_complete=bool(payload.get("symmetry_group_is_complete", False)),
            schema_version=str(payload.get("schema_version", SEED_BANK_SCHEMA_VERSION)),
        )
    if suffix == ".csv":
        rows = read_csv_rows(source)
        if not rows:
            raise ValueError("seed-bank CSV contains no seed rows.")
        configuration_fields = (
            "schema_version",
            "coordinate_scale",
            "periodic_coordinates_json",
            "absolute_tolerance",
            "relative_tolerance",
            "symmetry_group_is_complete",
            "symmetries_json",
        )
        first = rows[0]
        for row in rows[1:]:
            if any(row.get(field) != first.get(field) for field in configuration_fields):
                raise ValueError("seed-bank CSV repeats inconsistent bank configuration.")
        symmetry_payload = json.loads(first["symmetries_json"])
        symmetries = tuple(
            SymmetryTransform(
                name=str(item["name"]),
                matrix=tuple(tuple(float(value) for value in matrix_row) for matrix_row in item["matrix"]),
                offset=tuple(float(value) for value in item["offset"]),
            )
            for item in symmetry_payload
        )
        periodic_payload = json.loads(first["periodic_coordinates_json"])
        return SeedBank(
            memberships=tuple(_membership_from_serialized(row) for row in rows),
            coordinate_scale=tuple(
                float(value) for value in first["coordinate_scale"].split(";") if value != ""
            ),
            absolute_tolerance=float(first["absolute_tolerance"]),
            relative_tolerance=float(first["relative_tolerance"]),
            symmetries=symmetries,
            periodic_coordinates={
                int(index): float(period) for index, period in periodic_payload.items()
            },
            symmetry_group_is_complete=_serialized_bool(first["symmetry_group_is_complete"]),
            schema_version=str(first["schema_version"]),
        )
    raise ValueError("seed-bank path must end in .json or .csv.")


__all__ = [
    "INITIALIZATION_KINDS",
    "ORDER_KINDS",
    "SEED_BANK_CSV_FIELDS",
    "SEED_BANK_SCHEMA_VERSION",
    "SEED_ROUTES",
    "SeedBank",
    "SeedMembership",
    "SeedRecord",
    "SymmetryTransform",
    "build_seed_bank",
    "load_seed_bank",
    "scaled_state_distance",
    "seed_bank_csv_rows",
    "write_seed_bank",
]
