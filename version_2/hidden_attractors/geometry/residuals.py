"""Residuals for critical surfaces, PP/FPP startup filters and connecting sets.

Stability: experimental
    The integer definitions are local differential geometry.  The fractional
    quantities are explicitly the two-term Picard--Caputo *reset-startup*
    construction; they are not a global perpetual-point theory on the Caputo
    history space and they do not replace causal trajectory verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gamma
from typing import Any, Mapping

import numpy as np

from ..systems.base import ChaoticSystem
from .core import DifferentialGeometryEvaluation, evaluate_differential_geometry
from .piecewise import PWLPartition


@dataclass(frozen=True)
class CriticalSurfaceValues:
    """Local residuals for velocity, acceleration and singular-Jacobian sets."""

    state: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jacobian_determinant: float
    region: str | None

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=float)
        velocity = np.asarray(self.velocity, dtype=float)
        acceleration = np.asarray(self.acceleration, dtype=float)
        if state.ndim != 1 or state.size == 0:
            raise ValueError("state must be a non-empty vector.")
        if velocity.shape != state.shape or acceleration.shape != state.shape:
            raise ValueError("velocity and acceleration must match the state shape.")
        if not all(np.all(np.isfinite(value)) for value in (state, velocity, acceleration)):
            raise ValueError("critical-surface arrays must contain only finite values.")
        if not np.isfinite(float(self.jacobian_determinant)):
            raise ValueError("jacobian_determinant must be finite.")
        if self.region is not None and not str(self.region):
            raise ValueError("region must be None or a non-empty name.")
        object.__setattr__(self, "state", state.copy())
        object.__setattr__(self, "velocity", velocity.copy())
        object.__setattr__(self, "acceleration", acceleration.copy())
        object.__setattr__(self, "jacobian_determinant", float(self.jacobian_determinant))

    def velocity_component(self, index: int) -> float:
        return float(self.velocity[int(index)])

    def acceleration_component(self, index: int) -> float:
        return float(self.acceleration[int(index)])


@dataclass(frozen=True)
class PerpetualPointEvaluation:
    """Integer PP residual with explicit exclusion of equilibria.

    ``residual_norm`` retains the raw norm ``||J_f f||_2`` for backwards
    compatibility.  Candidate acceptance uses ``normalized_residual``, which
    is the scale-controlled residual declared by the campaign protocol,

    ``||J_f f||_2 / (1 + ||J_f||_2 ||f||_2)``.

    Here the matrix two-norm is the induced spectral norm.  Keeping both
    values avoids silently changing the meaning of existing raw diagnostics.
    """

    state: np.ndarray
    field: np.ndarray
    acceleration: np.ndarray
    field_norm: float
    residual_norm: float
    jacobian_norm: float
    normalization_denominator: float
    normalized_residual: float
    speed_floor: float
    acceleration_tolerance: float
    is_candidate: bool
    region: str | None

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=float)
        field = np.asarray(self.field, dtype=float)
        acceleration = np.asarray(self.acceleration, dtype=float)
        if state.ndim != 1 or state.size == 0 or field.shape != state.shape or acceleration.shape != state.shape:
            raise ValueError("PP state, field, and acceleration must be same-size non-empty vectors.")
        if not all(np.all(np.isfinite(value)) for value in (state, field, acceleration)):
            raise ValueError("PP arrays must contain only finite values.")
        speed, tolerance = _positive_tolerances(self.speed_floor, self.acceleration_tolerance)
        field_norm = float(self.field_norm)
        raw_residual = float(self.residual_norm)
        jacobian_norm = float(self.jacobian_norm)
        denominator = float(self.normalization_denominator)
        normalized = float(self.normalized_residual)
        if not np.isfinite(field_norm) or field_norm < 0.0:
            raise ValueError("field_norm must be finite and nonnegative.")
        if not np.isfinite(raw_residual) or raw_residual < 0.0:
            raise ValueError("residual_norm must be finite and nonnegative.")
        if not np.isfinite(jacobian_norm) or jacobian_norm < 0.0:
            raise ValueError("jacobian_norm must be finite and nonnegative.")
        expected_field_norm = float(np.linalg.norm(field))
        expected_raw = float(np.linalg.norm(acceleration))
        expected_denominator = 1.0 + jacobian_norm * field_norm
        expected_normalized = raw_residual / expected_denominator
        if not np.isclose(field_norm, expected_field_norm, rtol=1.0e-12, atol=1.0e-15):
            raise ValueError("field_norm is inconsistent with the field vector.")
        if not np.isclose(raw_residual, expected_raw, rtol=1.0e-12, atol=1.0e-15):
            raise ValueError("residual_norm is inconsistent with the raw acceleration.")
        if not np.isclose(denominator, expected_denominator, rtol=1.0e-12, atol=1.0e-15):
            raise ValueError("normalization_denominator is inconsistent with the PP formula.")
        if not np.isclose(normalized, expected_normalized, rtol=1.0e-12, atol=1.0e-15):
            raise ValueError("normalized_residual is inconsistent with the PP formula.")
        expected = field_norm > speed and normalized <= tolerance
        if bool(self.is_candidate) != bool(expected):
            raise ValueError("is_candidate is inconsistent with the declared PP tolerances.")
        object.__setattr__(self, "state", state.copy())
        object.__setattr__(self, "field", field.copy())
        object.__setattr__(self, "acceleration", acceleration.copy())
        object.__setattr__(self, "field_norm", field_norm)
        object.__setattr__(self, "residual_norm", raw_residual)
        object.__setattr__(self, "jacobian_norm", jacobian_norm)
        object.__setattr__(self, "normalization_denominator", denominator)
        object.__setattr__(self, "normalized_residual", normalized)
        object.__setattr__(self, "speed_floor", speed)
        object.__setattr__(self, "acceleration_tolerance", tolerance)

    @property
    def equilibrium_excluded(self) -> bool:
        return self.field_norm <= self.speed_floor

    @property
    def raw_residual_norm(self) -> float:
        """Alias making the legacy meaning of ``residual_norm`` explicit."""

        return self.residual_norm

    @property
    def normalized_residual_tolerance(self) -> float:
        """Tolerance applied to :attr:`normalized_residual`.

        ``acceleration_tolerance`` is retained as the constructor field for
        compatibility with the first experimental API.
        """

        return self.acceleration_tolerance


@dataclass(frozen=True)
class PicardCaputoStartupEvaluation:
    """Two-term Picard--Caputo reset-startup residual in Euclidean coordinates.

    ``residual_norm`` is the raw norm of ``startup_acceleration`` and therefore
    retains its explicit Gamma-dependent coefficient.  In contrast,
    ``normalized_residual`` is the geometric PP residual
    ``||J_f f||_2 / (1 + ||J_f||_2 ||f||_2)``.  Candidate acceptance uses the
    latter so that the Picard coefficient is not mistaken for a new physical
    fractional acceleration criterion.
    """

    state: np.ndarray
    q: float
    field: np.ndarray
    jacobian_field: np.ndarray
    first_coefficient: np.ndarray
    second_coefficient: np.ndarray
    startup_acceleration: np.ndarray
    field_norm: float
    residual_norm: float
    jacobian_norm: float
    jacobian_field_norm: float
    normalization_denominator: float
    normalized_residual: float
    speed_floor: float
    acceleration_tolerance: float
    is_candidate: bool
    region: str | None
    evidence_scope: str = "reset_startup_filter_not_global_caputo_history_event"

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=float)
        arrays = tuple(
            np.asarray(value, dtype=float)
            for value in (
                self.field,
                self.jacobian_field,
                self.first_coefficient,
                self.second_coefficient,
                self.startup_acceleration,
            )
        )
        if state.ndim != 1 or state.size == 0 or any(value.shape != state.shape for value in arrays):
            raise ValueError("Picard--Caputo arrays must match a non-empty state vector.")
        if not np.all(np.isfinite(state)) or any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("Picard--Caputo arrays must contain only finite values.")
        order = _validated_fractional_order(self.q)
        speed, tolerance = _positive_tolerances(self.speed_floor, self.acceleration_tolerance)
        field_norm = float(self.field_norm)
        startup_norm = float(self.residual_norm)
        jacobian_norm = float(self.jacobian_norm)
        jacobian_field_norm = float(self.jacobian_field_norm)
        denominator = float(self.normalization_denominator)
        normalized = float(self.normalized_residual)
        if not np.isfinite(field_norm) or field_norm < 0.0:
            raise ValueError("field_norm must be finite and nonnegative.")
        if not np.isfinite(startup_norm) or startup_norm < 0.0:
            raise ValueError("residual_norm must be finite and nonnegative.")
        if not np.isfinite(jacobian_norm) or jacobian_norm < 0.0:
            raise ValueError("jacobian_norm must be finite and nonnegative.")
        if not np.isfinite(jacobian_field_norm) or jacobian_field_norm < 0.0:
            raise ValueError("jacobian_field_norm must be finite and nonnegative.")
        expected_field_norm = float(np.linalg.norm(arrays[0]))
        expected_jacobian_field_norm = float(np.linalg.norm(arrays[1]))
        expected_startup_norm = float(np.linalg.norm(arrays[4]))
        expected_denominator = 1.0 + jacobian_norm * field_norm
        expected_normalized = jacobian_field_norm / expected_denominator
        consistency = (
            (field_norm, expected_field_norm, "field_norm"),
            (jacobian_field_norm, expected_jacobian_field_norm, "jacobian_field_norm"),
            (startup_norm, expected_startup_norm, "residual_norm"),
            (denominator, expected_denominator, "normalization_denominator"),
            (normalized, expected_normalized, "normalized_residual"),
        )
        for actual, expected_value, name in consistency:
            if not np.isclose(actual, expected_value, rtol=1.0e-12, atol=1.0e-15):
                raise ValueError(f"{name} is inconsistent with the Picard--Caputo residual contract.")
        expected = field_norm > speed and normalized <= tolerance
        if bool(self.is_candidate) != bool(expected):
            raise ValueError("is_candidate is inconsistent with the declared startup tolerances.")
        if not self.evidence_scope:
            raise ValueError("evidence_scope cannot be empty.")
        object.__setattr__(self, "state", state.copy())
        for name, value in zip(
            ("field", "jacobian_field", "first_coefficient", "second_coefficient", "startup_acceleration"),
            arrays,
        ):
            object.__setattr__(self, name, value.copy())
        object.__setattr__(self, "q", order)
        object.__setattr__(self, "field_norm", field_norm)
        object.__setattr__(self, "residual_norm", startup_norm)
        object.__setattr__(self, "jacobian_norm", jacobian_norm)
        object.__setattr__(self, "jacobian_field_norm", jacobian_field_norm)
        object.__setattr__(self, "normalization_denominator", denominator)
        object.__setattr__(self, "normalized_residual", normalized)
        object.__setattr__(self, "speed_floor", speed)
        object.__setattr__(self, "acceleration_tolerance", tolerance)

    @property
    def equilibrium_excluded(self) -> bool:
        return self.field_norm <= self.speed_floor

    @property
    def startup_acceleration_norm(self) -> float:
        """Raw norm of the Gamma-dependent startup acceleration coefficient."""

        return self.residual_norm

    @property
    def normalized_residual_tolerance(self) -> float:
        """Tolerance applied to the geometric normalized residual."""

        return self.acceleration_tolerance


@dataclass(frozen=True)
class ConnectingCurveEvaluation:
    """Rank-one residual for ``rank[F, DF F] <= 1`` away from equilibria."""

    state: np.ndarray
    field: np.ndarray
    acceleration: np.ndarray
    minor_pairs: tuple[tuple[int, int], ...]
    minors: np.ndarray
    parallel_multiplier: float
    orthogonal_acceleration: np.ndarray
    normalized_residual: float
    field_norm: float
    speed_floor: float
    residual_tolerance: float
    is_candidate: bool
    region: str | None

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=float)
        field = np.asarray(self.field, dtype=float)
        acceleration = np.asarray(self.acceleration, dtype=float)
        minors = np.asarray(self.minors, dtype=float)
        orthogonal = np.asarray(self.orthogonal_acceleration, dtype=float)
        if state.ndim != 1 or state.size == 0 or field.shape != state.shape or acceleration.shape != state.shape:
            raise ValueError("connecting state, field, and acceleration must be same-size vectors.")
        if minors.shape != (len(self.minor_pairs),):
            raise ValueError("minors must match minor_pairs.")
        if orthogonal.shape != state.shape:
            raise ValueError("orthogonal_acceleration must match the state shape.")
        if not all(np.all(np.isfinite(value)) for value in (state, field, acceleration, minors)):
            raise ValueError("connecting state, field, acceleration, and minors must be finite.")
        dimension = int(state.size)
        expected_pairs = connecting_minor_pairs(dimension)
        if tuple(self.minor_pairs) != expected_pairs:
            raise ValueError("minor_pairs must use the declared lexicographic ordering.")
        speed, tolerance = _positive_tolerances(self.speed_floor, self.residual_tolerance)
        field_norm = float(self.field_norm)
        if not np.isfinite(field_norm) or field_norm < 0.0:
            raise ValueError("field_norm must be finite and nonnegative.")
        excluded = field_norm <= speed
        if excluded:
            if not np.isnan(float(self.parallel_multiplier)):
                raise ValueError("equilibrium-excluded connecting residual requires a NaN multiplier.")
            if not np.all(np.isnan(orthogonal)) or not np.isinf(float(self.normalized_residual)):
                raise ValueError("equilibrium-excluded connecting residual requires NaN/Inf sentinels.")
            expected_candidate = False
        else:
            if not np.isfinite(float(self.parallel_multiplier)) or not np.all(np.isfinite(orthogonal)):
                raise ValueError("non-equilibrium connecting quantities must be finite.")
            if not np.isfinite(float(self.normalized_residual)) or float(self.normalized_residual) < 0.0:
                raise ValueError("normalized_residual must be finite and nonnegative.")
            expected_candidate = float(self.normalized_residual) <= tolerance
        if bool(self.is_candidate) != bool(expected_candidate):
            raise ValueError("is_candidate is inconsistent with the connecting residual contract.")
        object.__setattr__(self, "state", state.copy())
        object.__setattr__(self, "field", field.copy())
        object.__setattr__(self, "acceleration", acceleration.copy())
        object.__setattr__(self, "minors", minors.copy())
        object.__setattr__(self, "orthogonal_acceleration", orthogonal.copy())
        object.__setattr__(self, "field_norm", field_norm)
        object.__setattr__(self, "speed_floor", speed)
        object.__setattr__(self, "residual_tolerance", tolerance)

    @property
    def equilibrium_excluded(self) -> bool:
        return self.field_norm <= self.speed_floor


def _positive_tolerances(speed_floor: float, residual_tolerance: float) -> tuple[float, float]:
    speed = float(speed_floor)
    residual = float(residual_tolerance)
    if not np.isfinite(speed) or speed < 0.0:
        raise ValueError("speed_floor must be finite and nonnegative.")
    if not np.isfinite(residual) or residual < 0.0:
        raise ValueError("residual tolerance must be finite and nonnegative.")
    return speed, residual


def _perpetual_residual_scalars(
    evaluation: DifferentialGeometryEvaluation,
) -> tuple[float, float, float, float]:
    """Return the raw and protocol-normalized PP residual scalars.

    The tuple is ``(raw_norm, jacobian_norm, denominator, normalized)`` with
    the induced matrix two-norm.  This is the exact residual used in the
    geometric-topological campaign protocol.
    """

    raw_norm = float(np.linalg.norm(evaluation.jacobian_field))
    jacobian_norm = float(np.linalg.norm(evaluation.jacobian, ord=2))
    denominator = 1.0 + jacobian_norm * evaluation.field_norm
    normalized = raw_norm / denominator
    return raw_norm, jacobian_norm, denominator, normalized


def normalized_perpetual_residual(
    evaluation: DifferentialGeometryEvaluation,
) -> float:
    """Return ``||J_f f||_2 / (1 + ||J_f||_2 ||f||_2)``.

    The matrix norm is the induced spectral two-norm.  This public helper is
    useful when a caller already has a shared differential-geometry
    evaluation and wants the exact campaign acceptance residual without
    recomputing the field or Jacobian.
    """

    if not isinstance(evaluation, DifferentialGeometryEvaluation):
        raise TypeError("evaluation must be a DifferentialGeometryEvaluation.")
    return _perpetual_residual_scalars(evaluation)[3]


def critical_surface_values(
    system: ChaoticSystem,
    state: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
    *,
    partition: PWLPartition | None = None,
    jacobian_mode: str = "analytic",
    relative_step: float | None = None,
) -> CriticalSurfaceValues:
    """Evaluate the residuals of ``S_i^v``, ``S_i^a`` and ``S_J``."""

    evaluation = evaluate_differential_geometry(
        system,
        state,
        parameters,
        partition=partition,
        jacobian_mode=jacobian_mode,
        relative_step=relative_step,
    )
    return CriticalSurfaceValues(
        state=evaluation.state.copy(),
        velocity=evaluation.field.copy(),
        acceleration=evaluation.jacobian_field.copy(),
        jacobian_determinant=evaluation.jacobian_determinant,
        region=evaluation.region,
    )


def perpetual_point_residual(
    system: ChaoticSystem,
    state: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
    *,
    speed_floor: float = 1.0e-10,
    acceleration_tolerance: float = 1.0e-8,
    partition: PWLPartition | None = None,
    jacobian_mode: str = "analytic",
    relative_step: float | None = None,
) -> PerpetualPointEvaluation:
    """Evaluate the PP conditions ``F != 0`` and ``DF F = 0``."""

    speed, tolerance = _positive_tolerances(speed_floor, acceleration_tolerance)
    evaluation = evaluate_differential_geometry(
        system,
        state,
        parameters,
        partition=partition,
        jacobian_mode=jacobian_mode,
        relative_step=relative_step,
    )
    field_norm = evaluation.field_norm
    residual_norm, jacobian_norm, denominator, normalized = _perpetual_residual_scalars(
        evaluation
    )
    return PerpetualPointEvaluation(
        state=evaluation.state.copy(),
        field=evaluation.field.copy(),
        acceleration=evaluation.jacobian_field.copy(),
        field_norm=field_norm,
        residual_norm=residual_norm,
        jacobian_norm=jacobian_norm,
        normalization_denominator=denominator,
        normalized_residual=normalized,
        speed_floor=speed,
        acceleration_tolerance=tolerance,
        is_candidate=bool(field_norm > speed and normalized <= tolerance),
        region=evaluation.region,
    )


def _validated_fractional_order(q: float) -> float:
    order = float(q)
    if not np.isfinite(order) or not 0.0 < order <= 1.0:
        raise ValueError("q must be finite and satisfy 0 < q <= 1.")
    return order


def picard_caputo_startup_from_evaluation(
    evaluation: DifferentialGeometryEvaluation,
    q: float,
    *,
    speed_floor: float = 1.0e-10,
    acceleration_tolerance: float = 1.0e-8,
) -> PicardCaputoStartupEvaluation:
    """Build the two-term reset-startup coefficients from a local evaluation.

    With ``s=(t-t0)^q`` the expansion is

    ``x = p + F(p)s/Gamma(q+1) + DF(p)F(p)s^2/Gamma(2q+1) + ...``.

    Therefore the Euclidean startup acceleration with respect to ``s`` is
    ``2 DF(p)F(p)/Gamma(2q+1)``.  Its zero set equals the integer PP residual,
    but its scale depends on ``q``.
    """

    order = _validated_fractional_order(q)
    speed, tolerance = _positive_tolerances(speed_floor, acceleration_tolerance)
    first = evaluation.field / gamma(order + 1.0)
    second = evaluation.jacobian_field / gamma(2.0 * order + 1.0)
    startup_acceleration = 2.0 * second
    field_norm = evaluation.field_norm
    residual_norm = float(np.linalg.norm(startup_acceleration))
    (
        jacobian_field_norm,
        jacobian_norm,
        denominator,
        normalized,
    ) = _perpetual_residual_scalars(evaluation)
    return PicardCaputoStartupEvaluation(
        state=evaluation.state.copy(),
        q=order,
        field=evaluation.field.copy(),
        jacobian_field=evaluation.jacobian_field.copy(),
        first_coefficient=first,
        second_coefficient=second,
        startup_acceleration=startup_acceleration,
        field_norm=field_norm,
        residual_norm=residual_norm,
        jacobian_norm=jacobian_norm,
        jacobian_field_norm=jacobian_field_norm,
        normalization_denominator=denominator,
        normalized_residual=normalized,
        speed_floor=speed,
        acceleration_tolerance=tolerance,
        is_candidate=bool(field_norm > speed and normalized <= tolerance),
        region=evaluation.region,
    )


def fractional_perpetual_startup_residual(
    system: ChaoticSystem,
    state: np.ndarray,
    q: float,
    parameters: Mapping[str, Any] | None = None,
    *,
    speed_floor: float = 1.0e-10,
    acceleration_tolerance: float = 1.0e-8,
    partition: PWLPartition | None = None,
    jacobian_mode: str = "analytic",
    relative_step: float | None = None,
) -> PicardCaputoStartupEvaluation:
    """Evaluate the proposed FPP--A reset-startup filter for scalar Caputo order."""

    evaluation = evaluate_differential_geometry(
        system,
        state,
        parameters,
        partition=partition,
        jacobian_mode=jacobian_mode,
        relative_step=relative_step,
    )
    return picard_caputo_startup_from_evaluation(
        evaluation,
        q,
        speed_floor=speed_floor,
        acceleration_tolerance=acceleration_tolerance,
    )


def connecting_minor_pairs(dimension: int) -> tuple[tuple[int, int], ...]:
    """Return lexicographically ordered pairs ``(i,j)`` with ``i<j``."""

    size = int(dimension)
    if size < 1:
        raise ValueError("dimension must be positive.")
    return tuple((i, j) for i in range(size) for j in range(i + 1, size))


def connecting_minors(field: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    """Return all minors ``F_i A_j - F_j A_i`` of ``[F, A]``."""

    velocity = np.asarray(field, dtype=float)
    accel = np.asarray(acceleration, dtype=float)
    if velocity.ndim != 1 or accel.shape != velocity.shape or velocity.size == 0:
        raise ValueError("field and acceleration must be same-size non-empty vectors.")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(accel)):
        raise ValueError("field and acceleration must contain only finite values.")
    return np.asarray(
        [velocity[i] * accel[j] - velocity[j] * accel[i] for i, j in connecting_minor_pairs(velocity.size)],
        dtype=float,
    )


def connecting_curve_from_evaluation(
    evaluation: DifferentialGeometryEvaluation,
    *,
    speed_floor: float = 1.0e-10,
    residual_tolerance: float = 1.0e-8,
) -> ConnectingCurveEvaluation:
    """Evaluate a scale-controlled connecting-set residual away from equilibria."""

    speed, tolerance = _positive_tolerances(speed_floor, residual_tolerance)
    field = evaluation.field
    acceleration = evaluation.jacobian_field
    pairs = connecting_minor_pairs(evaluation.dimension)
    minors = connecting_minors(field, acceleration)
    field_norm = evaluation.field_norm
    if field_norm <= speed:
        multiplier = float("nan")
        orthogonal = np.full_like(acceleration, np.nan)
        normalized = float("inf")
        candidate = False
    else:
        denominator = float(field @ field)
        multiplier = float(field @ acceleration) / denominator
        orthogonal = acceleration - multiplier * field
        normalized = float(np.linalg.norm(orthogonal) / (1.0 + np.linalg.norm(acceleration)))
        candidate = normalized <= tolerance
    return ConnectingCurveEvaluation(
        state=evaluation.state.copy(),
        field=field.copy(),
        acceleration=acceleration.copy(),
        minor_pairs=pairs,
        minors=minors,
        parallel_multiplier=multiplier,
        orthogonal_acceleration=orthogonal,
        normalized_residual=normalized,
        field_norm=field_norm,
        speed_floor=speed,
        residual_tolerance=tolerance,
        is_candidate=bool(candidate),
        region=evaluation.region,
    )


def connecting_curve_residual(
    system: ChaoticSystem,
    state: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
    *,
    speed_floor: float = 1.0e-10,
    residual_tolerance: float = 1.0e-8,
    partition: PWLPartition | None = None,
    jacobian_mode: str = "analytic",
    relative_step: float | None = None,
) -> ConnectingCurveEvaluation:
    """Evaluate the algebraic connecting-curve condition at one state."""

    evaluation = evaluate_differential_geometry(
        system,
        state,
        parameters,
        partition=partition,
        jacobian_mode=jacobian_mode,
        relative_step=relative_step,
    )
    return connecting_curve_from_evaluation(
        evaluation,
        speed_floor=speed_floor,
        residual_tolerance=residual_tolerance,
    )


__all__ = [
    "ConnectingCurveEvaluation",
    "CriticalSurfaceValues",
    "PerpetualPointEvaluation",
    "PicardCaputoStartupEvaluation",
    "connecting_curve_from_evaluation",
    "connecting_curve_residual",
    "connecting_minor_pairs",
    "connecting_minors",
    "critical_surface_values",
    "fractional_perpetual_startup_residual",
    "normalized_perpetual_residual",
    "perpetual_point_residual",
    "picard_caputo_startup_from_evaluation",
]
