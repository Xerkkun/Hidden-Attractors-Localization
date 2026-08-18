"""Finite-resolution edge tracking between two classified destinations.

Stability: experimental
    The API is intended for the TG campaign and may grow as validated
    inherited-history and set-oriented workflows are added.

The routines in this module refine a one-dimensional bracket in an initial-
data space.  They deliberately separate geometry, trajectory evaluation, and
destination classification: an evaluator supplied by a workflow is
responsible for integrating the system and returning an :class:`EdgeDestination`.

This first implementation is the ``initial_data_boundary_bisection`` stage
(steps 1--4 of the report protocol).  Evolving a narrow pair along the edge,
re-bracketing before destination fall-off, extracting edge returns, and
requiring three non-collinear initial brackets are subsequent stages and are
not claimed by this module.

An edge-tracking result is numerical basin-boundary evidence at the declared
resolution.  It is not a proof that the limiting point belongs to a smooth
stable manifold, nor is it by itself a proof of hiddenness or Wada structure.
For Caputo problems, ``caputo_reset_initial_state`` means a new causal IVP at
the declared lower terminal.  Inherited-history coordinates require an
explicitly named admissible family.  The workflow evaluator remains
responsible for constructing and validating that family; the identifier is a
provenance guard, not a proof that arbitrary input vectors are valid histories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

import numpy as np


EdgeDataSemantics = Literal[
    "ode_initial_state",
    "caputo_reset_initial_state",
    "admissible_history_family_parameter",
]
EdgePhase = Literal[
    "confirm_left",
    "confirm_right",
    "midpoint",
    "ambiguous_left_probe",
    "ambiguous_right_probe",
]


class EdgeGeometry(Protocol):
    """Metric and shortest-chart interpolation used by edge tracking."""

    @property
    def name(self) -> str:
        """Stable geometry identifier retained in campaign artifacts."""

    @property
    def dimension(self) -> int:
        """Dimension of the coordinate vector accepted by the geometry."""

    def distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        """Return the declared dimensionless distance."""

    def midpoint(self, left: Sequence[float], right: Sequence[float]) -> np.ndarray:
        """Return the midpoint in the selected local chart."""


def _finite_vector(value: Sequence[float], dimension: int, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (int(dimension),) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape ({int(dimension)},).")
    return array


@dataclass(frozen=True)
class ScaledEuclideanGeometry:
    """Distance ``||S^-1 (x-y)||_2`` for a positive diagonal scale ``S``."""

    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.scale)
        if not values or not np.all(np.isfinite(values)) or any(value <= 0.0 for value in values):
            raise ValueError("scale must contain finite positive values.")
        object.__setattr__(self, "scale", values)

    @property
    def name(self) -> str:
        return "scaled_euclidean"

    @property
    def dimension(self) -> int:
        return len(self.scale)

    def distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        a = _finite_vector(left, self.dimension, name="left")
        b = _finite_vector(right, self.dimension, name="right")
        return float(np.linalg.norm((a - b) / np.asarray(self.scale, dtype=float)))

    def midpoint(self, left: Sequence[float], right: Sequence[float]) -> np.ndarray:
        a = _finite_vector(left, self.dimension, name="left")
        b = _finite_vector(right, self.dimension, name="right")
        return 0.5 * (a + b)


@dataclass(frozen=True)
class ScaledCylindricalGeometry:
    """Scaled product metric with one or more periodic coordinates.

    The midpoint is computed on the shortest lifted arc.  It is intentionally
    not wrapped back to a principal interval, because systems such as the PLL
    integrate phase on an unwrapped lift.  Exactly antipodal endpoints do not
    select a unique chart and are rejected.
    """

    scale: tuple[float, ...]
    periodic_indices: tuple[int, ...]
    periods: tuple[float, ...]
    antipodal_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        scales = tuple(float(value) for value in self.scale)
        indices = tuple(int(index) for index in self.periodic_indices)
        periods = tuple(float(value) for value in self.periods)
        if not scales or not np.all(np.isfinite(scales)) or any(value <= 0.0 for value in scales):
            raise ValueError("scale must contain finite positive values.")
        if not indices or len(indices) != len(periods):
            raise ValueError("periodic_indices and periods must be non-empty and have equal length.")
        if len(set(indices)) != len(indices) or any(index < 0 or index >= len(scales) for index in indices):
            raise ValueError("periodic_indices must be unique valid coordinate indices.")
        if not np.all(np.isfinite(periods)) or any(value <= 0.0 for value in periods):
            raise ValueError("periods must contain finite positive values.")
        tolerance = float(self.antipodal_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("antipodal_tolerance must be finite and non-negative.")
        object.__setattr__(self, "scale", scales)
        object.__setattr__(self, "periodic_indices", indices)
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "antipodal_tolerance", tolerance)

    @property
    def name(self) -> str:
        return "scaled_cylindrical"

    @property
    def dimension(self) -> int:
        return len(self.scale)

    def _shortest_difference(
        self,
        left: Sequence[float],
        right: Sequence[float],
        *,
        require_unique: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        a = _finite_vector(left, self.dimension, name="left")
        b = _finite_vector(right, self.dimension, name="right")
        delta = b - a
        for index, period in zip(self.periodic_indices, self.periods):
            wrapped = (delta[index] + 0.5 * period) % period - 0.5 * period
            if require_unique and np.isclose(
                abs(wrapped),
                0.5 * period,
                rtol=0.0,
                atol=self.antipodal_tolerance * max(1.0, period),
            ):
                raise ValueError(
                    "periodic endpoints are antipodal; select an explicit chart before interpolation."
                )
            delta[index] = wrapped
        return a, delta

    def distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        _, delta = self._shortest_difference(left, right, require_unique=False)
        return float(np.linalg.norm(delta / np.asarray(self.scale, dtype=float)))

    def midpoint(self, left: Sequence[float], right: Sequence[float]) -> np.ndarray:
        a, delta = self._shortest_difference(left, right, require_unique=True)
        return a + 0.5 * delta


@dataclass(frozen=True)
class EdgeDestination:
    """Outcome of integrating and classifying one edge-tracking datum."""

    label: str
    resolved: bool
    evaluation_ok: bool = True
    integration_status: str = "ok"
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.label).strip():
            raise ValueError("destination label must be non-empty.")
        if not str(self.integration_status).strip():
            raise ValueError("integration_status must be non-empty.")

    @classmethod
    def terminal(
        cls,
        label: str,
        *,
        integration_status: str = "ok",
        metadata: Mapping[str, Any] | None = None,
    ) -> "EdgeDestination":
        return cls(
            label=str(label),
            resolved=True,
            evaluation_ok=True,
            integration_status=str(integration_status),
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def ambiguous(
        cls,
        *,
        label: str = "ambiguous",
        reason: str = "classifier did not assign a terminal destination",
        metadata: Mapping[str, Any] | None = None,
    ) -> "EdgeDestination":
        return cls(
            label=str(label),
            resolved=False,
            evaluation_ok=True,
            integration_status="ok",
            reason=str(reason),
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def failure(
        cls,
        integration_status: str,
        *,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "EdgeDestination":
        return cls(
            label="numerical_failure",
            resolved=False,
            evaluation_ok=False,
            integration_status=str(integration_status),
            reason=str(reason),
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True)
class EdgeEvaluationContext:
    """Context sent to the workflow evaluator for an auditable decision."""

    bracket_id: str
    phase: EdgePhase
    budget_level: str
    iteration: int
    ambiguity_index: int = 0


@dataclass(frozen=True)
class EdgeEvaluationRecord:
    """One coordinate evaluation retained in the edge history."""

    context: EdgeEvaluationContext
    coordinate: tuple[float, ...]
    outcome: EdgeDestination


@dataclass(frozen=True)
class EdgeIteration:
    """One bracket update, including optional probes around an ambiguity."""

    iteration: int
    width_before: float
    width_after: float
    left_after: tuple[float, ...]
    right_after: tuple[float, ...]
    midpoint: tuple[float, ...]
    midpoint_outcome: EdgeDestination
    ambiguous_streak: int
    evaluations: tuple[EdgeEvaluationRecord, ...]
    event: str


@dataclass(frozen=True)
class EdgeTrackingConfig:
    """Numerical contract for one bisection/refinement run."""

    tolerance: float = 1.0e-8
    max_iterations: int = 80
    max_consecutive_ambiguous: int = 3
    confirmation_levels: tuple[str, ...] = ("B1", "B2")
    tracking_level: str = "B2"
    data_semantics: EdgeDataSemantics = "ode_initial_state"
    admissible_history_family_id: str | None = None

    def __post_init__(self) -> None:
        tolerance = float(self.tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be positive.")
        if int(self.max_consecutive_ambiguous) < 1:
            raise ValueError("max_consecutive_ambiguous must be positive.")
        levels = tuple(str(level).strip() for level in self.confirmation_levels)
        if not levels or any(not level for level in levels):
            raise ValueError("confirmation_levels must contain non-empty labels.")
        if not str(self.tracking_level).strip():
            raise ValueError("tracking_level must be non-empty.")
        allowed = {
            "ode_initial_state",
            "caputo_reset_initial_state",
            "admissible_history_family_parameter",
        }
        if self.data_semantics not in allowed:
            raise ValueError(f"unsupported data_semantics: {self.data_semantics!r}.")
        if (
            self.data_semantics == "admissible_history_family_parameter"
            and not str(self.admissible_history_family_id or "").strip()
        ):
            raise ValueError(
                "inherited-history edge tracking requires admissible_history_family_id; "
                "raw histories are not interpolated."
            )
        if (
            self.data_semantics != "admissible_history_family_parameter"
            and self.admissible_history_family_id is not None
        ):
            raise ValueError(
                "admissible_history_family_id is valid only for "
                "admissible_history_family_parameter semantics."
            )
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        object.__setattr__(
            self,
            "max_consecutive_ambiguous",
            int(self.max_consecutive_ambiguous),
        )
        object.__setattr__(self, "confirmation_levels", levels)


@dataclass(frozen=True)
class EdgeTrackingResult:
    """Complete finite-resolution bracket and its evaluation history."""

    bracket_id: str
    geometry: str
    data_semantics: EdgeDataSemantics
    admissible_history_family_id: str | None
    status: str
    stop_reason: str
    initial_left: tuple[float, ...]
    initial_right: tuple[float, ...]
    final_left: tuple[float, ...]
    final_right: tuple[float, ...]
    candidate: tuple[float, ...]
    left_destination: str
    right_destination: str
    initial_width: float
    final_width: float
    confirmations: tuple[EdgeEvaluationRecord, ...]
    iterations: tuple[EdgeIteration, ...]
    method: str = "initial_data_boundary_bisection"
    finite_resolution_only: bool = True

    @property
    def converged(self) -> bool:
        return self.status == "converged"


EdgeEvaluator = Callable[[np.ndarray, EdgeEvaluationContext], EdgeDestination]


def edge_destination_from_classification(
    classification: Any,
    *,
    integration_status: str = "ok",
    evaluation_ok: bool = True,
) -> EdgeDestination:
    """Adapt the unified destination-classifier result to edge tracking.

    The adapter accepts either a mapping or an object exposing ``label`` and
    ``is_ambiguous``.  ``transient`` is deliberately unresolved even when the
    classifier does not mark it ambiguous: a finite transient is not a basin
    destination and therefore cannot replace a bracket endpoint.
    """

    if isinstance(classification, EdgeDestination):
        return classification
    if isinstance(classification, Mapping):
        broad_label = classification.get("label")
        edge_label = (
            classification.get("edge_label")
            or classification.get("destination_id")
            or broad_label
        )
        ambiguous = classification.get("is_ambiguous", False)
        reasons = classification.get("reasons", ())
        confidence = classification.get("confidence")
        subtype = classification.get("subtype", "")
        evidence_status = classification.get("evidence_status", "")
        metrics = classification.get("metrics", {})
    else:
        broad_label = getattr(classification, "label", None)
        edge_label = (
            getattr(classification, "edge_label", None)
            or getattr(classification, "destination_id", None)
            or broad_label
        )
        ambiguous = getattr(classification, "is_ambiguous", False)
        reasons = getattr(classification, "reasons", ())
        confidence = getattr(classification, "confidence", None)
        subtype = getattr(classification, "subtype", "")
        evidence_status = getattr(classification, "evidence_status", "")
        metrics = getattr(classification, "metrics", {})
    if broad_label is None or not str(broad_label).strip():
        raise ValueError("classification must expose a non-empty label.")
    if edge_label is None or not str(edge_label).strip():
        raise ValueError("classification must expose a non-empty edge label.")
    broad_label_value = str(broad_label)
    label_value = str(edge_label)
    reason_values = (
        tuple(str(item) for item in reasons)
        if isinstance(reasons, (list, tuple))
        else (str(reasons),) if reasons else ()
    )
    metadata = {
        "classifier_confidence": confidence,
        "classifier_label": broad_label_value,
        "classifier_destination_id": label_value,
        "classifier_subtype": str(subtype),
        "classifier_evidence_status": str(evidence_status),
        "classifier_metrics": metrics,
        "classifier_reasons": reason_values,
    }
    if not bool(evaluation_ok):
        return EdgeDestination.failure(
            integration_status,
            reason="; ".join(reason_values) or "trajectory evaluation failed",
            metadata=metadata,
        )
    resolved = not bool(ambiguous) and broad_label_value not in {"ambiguous", "transient"}
    return EdgeDestination(
        label=label_value,
        resolved=resolved,
        evaluation_ok=True,
        integration_status=str(integration_status),
        reason="; ".join(reason_values),
        metadata=metadata,
    )


def _recorded_evaluation(
    evaluator: EdgeEvaluator,
    coordinate: np.ndarray,
    context: EdgeEvaluationContext,
) -> EdgeEvaluationRecord:
    try:
        outcome = evaluator(np.asarray(coordinate, dtype=float).copy(), context)
        if not isinstance(outcome, EdgeDestination):
            raise TypeError("edge evaluator must return EdgeDestination.")
    except (ArithmeticError, FloatingPointError, RuntimeError, TypeError, ValueError) as exc:
        outcome = EdgeDestination.failure(
            "evaluator_exception",
            reason=f"{type(exc).__name__}: {exc}",
        )
    return EdgeEvaluationRecord(
        context=context,
        coordinate=tuple(float(value) for value in coordinate),
        outcome=outcome,
    )


def _result(
    *,
    bracket_id: str,
    geometry: EdgeGeometry,
    config: EdgeTrackingConfig,
    status: str,
    stop_reason: str,
    initial_left: np.ndarray,
    initial_right: np.ndarray,
    final_left: np.ndarray,
    final_right: np.ndarray,
    left_destination: str,
    right_destination: str,
    confirmations: Sequence[EdgeEvaluationRecord],
    iterations: Sequence[EdgeIteration],
    initial_width: float,
    candidate: np.ndarray | None = None,
) -> EdgeTrackingResult:
    if candidate is None:
        try:
            candidate = geometry.midpoint(final_left, final_right)
        except ValueError:
            candidate = 0.5 * (final_left + final_right)
    return EdgeTrackingResult(
        bracket_id=bracket_id,
        geometry=geometry.name,
        data_semantics=config.data_semantics,
        admissible_history_family_id=config.admissible_history_family_id,
        status=status,
        stop_reason=stop_reason,
        initial_left=tuple(float(value) for value in initial_left),
        initial_right=tuple(float(value) for value in initial_right),
        final_left=tuple(float(value) for value in final_left),
        final_right=tuple(float(value) for value in final_right),
        candidate=tuple(float(value) for value in candidate),
        left_destination=left_destination,
        right_destination=right_destination,
        initial_width=float(initial_width),
        final_width=float(geometry.distance(final_left, final_right)),
        confirmations=tuple(confirmations),
        iterations=tuple(iterations),
    )


def _resolved_transition_pairs(
    points: Sequence[tuple[float, np.ndarray, EdgeDestination]],
    left_label: str,
    right_label: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    resolved = [item for item in points if item[2].evaluation_ok and item[2].resolved]
    transitions: list[tuple[np.ndarray, np.ndarray]] = []
    for (_, point_a, outcome_a), (_, point_b, outcome_b) in zip(resolved, resolved[1:]):
        if outcome_a.label == outcome_b.label:
            continue
        if {outcome_a.label, outcome_b.label} != {left_label, right_label}:
            continue
        if outcome_a.label == left_label:
            transitions.append((point_a, point_b))
        else:
            transitions.append((point_b, point_a))
    return transitions


def track_edge_bracket(
    left: Sequence[float],
    right: Sequence[float],
    *,
    evaluator: EdgeEvaluator,
    geometry: EdgeGeometry,
    bracket_id: str,
    config: EdgeTrackingConfig | None = None,
) -> EdgeTrackingResult:
    """Confirm and bisect an initial-data bracket between two destinations.

    Endpoints are independently classified at every level in
    ``confirmation_levels``.  Refinement starts only when each endpoint keeps
    one resolved label and the two labels differ.  An ambiguous midpoint is
    surrounded by quarter-point probes; three consecutive ambiguous midpoint
    refinements stop the run instead of forcing a destination.  More than one
    resolved transition on the segment also stops the run because ordinary
    bisection would then select a branch without a declared rule.

    This is only the initial bracket-refinement stage of dynamic edge
    tracking.  It does not evolve the final pair along a separatrix or compute
    an edge-return map.
    """

    cfg = config or EdgeTrackingConfig()
    identifier = str(bracket_id).strip()
    if not identifier:
        raise ValueError("bracket_id must be non-empty.")
    a0 = _finite_vector(left, geometry.dimension, name="left")
    b0 = _finite_vector(right, geometry.dimension, name="right")
    initial_width = float(geometry.distance(a0, b0))
    if initial_width <= 0.0:
        raise ValueError("edge bracket endpoints must be distinct in the declared geometry.")

    confirmations: list[EdgeEvaluationRecord] = []
    left_outcomes: list[EdgeDestination] = []
    right_outcomes: list[EdgeDestination] = []
    for level in cfg.confirmation_levels:
        left_record = _recorded_evaluation(
            evaluator,
            a0,
            EdgeEvaluationContext(identifier, "confirm_left", level, 0),
        )
        right_record = _recorded_evaluation(
            evaluator,
            b0,
            EdgeEvaluationContext(identifier, "confirm_right", level, 0),
        )
        confirmations.extend((left_record, right_record))
        left_outcomes.append(left_record.outcome)
        right_outcomes.append(right_record.outcome)

    all_endpoint_outcomes = left_outcomes + right_outcomes
    if any(not outcome.evaluation_ok for outcome in all_endpoint_outcomes):
        return _result(
            bracket_id=identifier,
            geometry=geometry,
            config=cfg,
            status="rejected_endpoint_failure",
            stop_reason="an endpoint integration or classification evaluation failed",
            initial_left=a0,
            initial_right=b0,
            final_left=a0,
            final_right=b0,
            left_destination="unresolved",
            right_destination="unresolved",
            confirmations=confirmations,
            iterations=(),
            initial_width=initial_width,
        )
    if any(not outcome.resolved for outcome in all_endpoint_outcomes):
        return _result(
            bracket_id=identifier,
            geometry=geometry,
            config=cfg,
            status="rejected_endpoint_unresolved",
            stop_reason="both endpoints must have terminal destinations at every confirmation level",
            initial_left=a0,
            initial_right=b0,
            final_left=a0,
            final_right=b0,
            left_destination="unresolved",
            right_destination="unresolved",
            confirmations=confirmations,
            iterations=(),
            initial_width=initial_width,
        )
    left_labels = {outcome.label for outcome in left_outcomes}
    right_labels = {outcome.label for outcome in right_outcomes}
    if len(left_labels) != 1 or len(right_labels) != 1:
        return _result(
            bracket_id=identifier,
            geometry=geometry,
            config=cfg,
            status="rejected_endpoint_inconsistent",
            stop_reason="an endpoint changed destination across confirmation levels",
            initial_left=a0,
            initial_right=b0,
            final_left=a0,
            final_right=b0,
            left_destination="inconsistent",
            right_destination="inconsistent",
            confirmations=confirmations,
            iterations=(),
            initial_width=initial_width,
        )
    left_label = next(iter(left_labels))
    right_label = next(iter(right_labels))
    if left_label == right_label:
        return _result(
            bracket_id=identifier,
            geometry=geometry,
            config=cfg,
            status="rejected_same_destination",
            stop_reason="confirmed endpoints do not delimit different destinations",
            initial_left=a0,
            initial_right=b0,
            final_left=a0,
            final_right=b0,
            left_destination=left_label,
            right_destination=right_label,
            confirmations=confirmations,
            iterations=(),
            initial_width=initial_width,
        )

    a = a0.copy()
    b = b0.copy()
    iterations: list[EdgeIteration] = []
    ambiguous_streak = 0
    candidate: np.ndarray | None = None

    for iteration_index in range(1, cfg.max_iterations + 1):
        width_before = float(geometry.distance(a, b))
        if width_before <= cfg.tolerance:
            return _result(
                bracket_id=identifier,
                geometry=geometry,
                config=cfg,
                status="converged",
                stop_reason="declared scaled-distance tolerance reached",
                initial_left=a0,
                initial_right=b0,
                final_left=a,
                final_right=b,
                left_destination=left_label,
                right_destination=right_label,
                confirmations=confirmations,
                iterations=iterations,
                initial_width=initial_width,
                candidate=candidate,
            )
        try:
            midpoint = geometry.midpoint(a, b)
        except ValueError as exc:
            return _result(
                bracket_id=identifier,
                geometry=geometry,
                config=cfg,
                status="rejected_geometry_ambiguous",
                stop_reason=str(exc),
                initial_left=a0,
                initial_right=b0,
                final_left=a,
                final_right=b,
                left_destination=left_label,
                right_destination=right_label,
                confirmations=confirmations,
                iterations=iterations,
                initial_width=initial_width,
                candidate=0.5 * (a + b),
            )
        candidate = midpoint.copy()
        midpoint_record = _recorded_evaluation(
            evaluator,
            midpoint,
            EdgeEvaluationContext(
                identifier,
                "midpoint",
                cfg.tracking_level,
                iteration_index,
                ambiguous_streak,
            ),
        )
        records: list[EdgeEvaluationRecord] = [midpoint_record]
        outcome = midpoint_record.outcome
        event = "resolved_midpoint"

        if not outcome.evaluation_ok:
            iterations.append(
                EdgeIteration(
                    iteration=iteration_index,
                    width_before=width_before,
                    width_after=width_before,
                    left_after=tuple(a),
                    right_after=tuple(b),
                    midpoint=tuple(midpoint),
                    midpoint_outcome=outcome,
                    ambiguous_streak=ambiguous_streak,
                    evaluations=tuple(records),
                    event="evaluation_failure",
                )
            )
            return _result(
                bracket_id=identifier,
                geometry=geometry,
                config=cfg,
                status="evaluation_failure",
                stop_reason="midpoint integration or classification evaluation failed",
                initial_left=a0,
                initial_right=b0,
                final_left=a,
                final_right=b,
                left_destination=left_label,
                right_destination=right_label,
                confirmations=confirmations,
                iterations=iterations,
                initial_width=initial_width,
                candidate=midpoint,
            )

        if outcome.resolved:
            ambiguous_streak = 0
            if outcome.label == left_label:
                a = midpoint
            elif outcome.label == right_label:
                b = midpoint
            else:
                event = "third_destination"
                iterations.append(
                    EdgeIteration(
                        iteration=iteration_index,
                        width_before=width_before,
                        width_after=width_before,
                        left_after=tuple(a),
                        right_after=tuple(b),
                        midpoint=tuple(midpoint),
                        midpoint_outcome=outcome,
                        ambiguous_streak=ambiguous_streak,
                        evaluations=tuple(records),
                        event=event,
                    )
                )
                return _result(
                    bracket_id=identifier,
                    geometry=geometry,
                    config=cfg,
                    status="third_destination",
                    stop_reason="a third resolved destination intersects the bracket segment",
                    initial_left=a0,
                    initial_right=b0,
                    final_left=a,
                    final_right=b,
                    left_destination=left_label,
                    right_destination=right_label,
                    confirmations=confirmations,
                    iterations=iterations,
                    initial_width=initial_width,
                    candidate=midpoint,
                )
        else:
            ambiguous_streak += 1
            event = "ambiguous_midpoint"
            if ambiguous_streak < cfg.max_consecutive_ambiguous:
                left_probe = geometry.midpoint(a, midpoint)
                right_probe = geometry.midpoint(midpoint, b)
                left_probe_record = _recorded_evaluation(
                    evaluator,
                    left_probe,
                    EdgeEvaluationContext(
                        identifier,
                        "ambiguous_left_probe",
                        cfg.tracking_level,
                        iteration_index,
                        ambiguous_streak,
                    ),
                )
                right_probe_record = _recorded_evaluation(
                    evaluator,
                    right_probe,
                    EdgeEvaluationContext(
                        identifier,
                        "ambiguous_right_probe",
                        cfg.tracking_level,
                        iteration_index,
                        ambiguous_streak,
                    ),
                )
                records.extend((left_probe_record, right_probe_record))
                probe_outcomes = (left_probe_record.outcome, right_probe_record.outcome)
                if any(not item.evaluation_ok for item in probe_outcomes):
                    event = "ambiguity_probe_failure"
                elif any(
                    item.resolved and item.label not in {left_label, right_label}
                    for item in probe_outcomes
                ):
                    event = "third_destination"
                else:
                    points = (
                        (0.0, a, EdgeDestination.terminal(left_label)),
                        (0.25, left_probe, left_probe_record.outcome),
                        (0.5, midpoint, outcome),
                        (0.75, right_probe, right_probe_record.outcome),
                        (1.0, b, EdgeDestination.terminal(right_label)),
                    )
                    transitions = _resolved_transition_pairs(points, left_label, right_label)
                    if len(transitions) > 1:
                        event = "multiple_transitions"
                    elif len(transitions) == 1:
                        a, b = (transitions[0][0].copy(), transitions[0][1].copy())
                        event = "ambiguous_midpoint_refined"

        width_after = float(geometry.distance(a, b))
        iterations.append(
            EdgeIteration(
                iteration=iteration_index,
                width_before=width_before,
                width_after=width_after,
                left_after=tuple(float(value) for value in a),
                right_after=tuple(float(value) for value in b),
                midpoint=tuple(float(value) for value in midpoint),
                midpoint_outcome=outcome,
                ambiguous_streak=ambiguous_streak,
                evaluations=tuple(records),
                event=event,
            )
        )

        if event in {"ambiguity_probe_failure", "third_destination", "multiple_transitions"}:
            reasons = {
                "ambiguity_probe_failure": "a probe around an ambiguous midpoint failed",
                "third_destination": "a third resolved destination intersects the bracket segment",
                "multiple_transitions": "the segment contains more than one resolved destination transition",
            }
            return _result(
                bracket_id=identifier,
                geometry=geometry,
                config=cfg,
                status=event,
                stop_reason=reasons[event],
                initial_left=a0,
                initial_right=b0,
                final_left=a,
                final_right=b,
                left_destination=left_label,
                right_destination=right_label,
                confirmations=confirmations,
                iterations=iterations,
                initial_width=initial_width,
                candidate=midpoint,
            )
        if ambiguous_streak >= cfg.max_consecutive_ambiguous:
            return _result(
                bracket_id=identifier,
                geometry=geometry,
                config=cfg,
                status="ambiguous_limit",
                stop_reason="classifier remained unresolved for the declared consecutive-refinement limit",
                initial_left=a0,
                initial_right=b0,
                final_left=a,
                final_right=b,
                left_destination=left_label,
                right_destination=right_label,
                confirmations=confirmations,
                iterations=iterations,
                initial_width=initial_width,
                candidate=midpoint,
            )
        if width_after <= cfg.tolerance:
            return _result(
                bracket_id=identifier,
                geometry=geometry,
                config=cfg,
                status="converged",
                stop_reason="declared scaled-distance tolerance reached",
                initial_left=a0,
                initial_right=b0,
                final_left=a,
                final_right=b,
                left_destination=left_label,
                right_destination=right_label,
                confirmations=confirmations,
                iterations=iterations,
                initial_width=initial_width,
                candidate=geometry.midpoint(a, b),
            )

    return _result(
        bracket_id=identifier,
        geometry=geometry,
        config=cfg,
        status="max_iterations",
        stop_reason="maximum number of bracket refinements reached",
        initial_left=a0,
        initial_right=b0,
        final_left=a,
        final_right=b,
        left_destination=left_label,
        right_destination=right_label,
        confirmations=confirmations,
        iterations=iterations,
        initial_width=initial_width,
    )


__all__ = [
    "EdgeDataSemantics",
    "EdgeDestination",
    "EdgeEvaluationContext",
    "EdgeEvaluationRecord",
    "EdgeEvaluator",
    "EdgeGeometry",
    "EdgeIteration",
    "EdgeTrackingConfig",
    "EdgeTrackingResult",
    "ScaledCylindricalGeometry",
    "ScaledEuclideanGeometry",
    "edge_destination_from_classification",
    "track_edge_bracket",
]
