"""Differential evaluations shared by geometric localization methods.

Stability: experimental
    The module evaluates ``F``, ``DF`` and ``DF F`` for autonomous flows.
    It does not integrate trajectories and does not infer chaos, attraction,
    basin membership, or hiddenness from these local quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import numpy as np

from ..systems.base import ChaoticSystem
from .piecewise import IncompatiblePartitionError, PWLPartition, registered_pwl_partition


JacobianMode = Literal["analytic", "finite_difference", "auto"]


@dataclass(frozen=True)
class DifferentialGeometryEvaluation:
    """One local evaluation of a smooth or regional-smooth vector field."""

    state: np.ndarray
    field: np.ndarray
    jacobian: np.ndarray
    jacobian_field: np.ndarray
    jacobian_determinant: float
    jacobian_source: str
    region: str | None

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=float)
        field = np.asarray(self.field, dtype=float)
        jacobian = np.asarray(self.jacobian, dtype=float)
        jacobian_field = np.asarray(self.jacobian_field, dtype=float)
        if state.ndim != 1 or state.size == 0:
            raise ValueError("state must be a non-empty vector.")
        dimension = int(state.size)
        if field.shape != (dimension,) or jacobian_field.shape != (dimension,):
            raise ValueError("field and jacobian_field must match the state dimension.")
        if jacobian.shape != (dimension, dimension):
            raise ValueError("jacobian must be square and match the state dimension.")
        if not all(np.all(np.isfinite(value)) for value in (state, field, jacobian, jacobian_field)):
            raise ValueError("differential-geometry arrays must contain only finite values.")
        if not np.isfinite(float(self.jacobian_determinant)):
            raise ValueError("jacobian_determinant must be finite.")
        if not self.jacobian_source:
            raise ValueError("jacobian_source cannot be empty.")
        if self.region is not None and not str(self.region):
            raise ValueError("region must be None or a non-empty name.")
        object.__setattr__(self, "state", state.copy())
        object.__setattr__(self, "field", field.copy())
        object.__setattr__(self, "jacobian", jacobian.copy())
        object.__setattr__(self, "jacobian_field", jacobian_field.copy())
        object.__setattr__(self, "jacobian_determinant", float(self.jacobian_determinant))

    @property
    def dimension(self) -> int:
        return int(self.state.size)

    @property
    def field_norm(self) -> float:
        return float(np.linalg.norm(self.field))

    @property
    def acceleration_norm(self) -> float:
        return float(np.linalg.norm(self.jacobian_field))


def _validated_flow_state(system: ChaoticSystem, state: np.ndarray) -> np.ndarray:
    if not isinstance(system, ChaoticSystem):
        raise TypeError("system must be a ChaoticSystem instance.")
    if not system.is_continuous:
        raise ValueError("geometric flow quantities require system.kind='flow'.")
    point = np.asarray(state, dtype=float)
    if point.shape != (system.dimension,):
        raise ValueError(f"{system.name} expects a state of shape ({system.dimension},).")
    if not np.all(np.isfinite(point)):
        raise ValueError("state must contain only finite values.")
    return point


def _resolved_partition(
    system: ChaoticSystem,
    parameters: Mapping[str, Any] | None,
    partition: PWLPartition | None,
) -> PWLPartition | None:
    return partition if partition is not None else registered_pwl_partition(system, parameters)


def _validated_partition_field(
    system: ChaoticSystem,
    state: np.ndarray,
    parameters: Mapping[str, Any] | None,
    partition: PWLPartition,
) -> np.ndarray:
    """Validate provenance and confirm that the partition does not replace ``system.rhs``."""

    if partition.dimension != system.dimension:
        raise IncompatiblePartitionError("piecewise partition dimension does not match the system.")
    partition.validate_system(system, parameters)
    regional = np.asarray(partition.field_at(state), dtype=float)
    registered = np.asarray(system.evaluate(state, parameters), dtype=float)
    if registered.shape != regional.shape or not np.all(np.isfinite(registered)):
        raise IncompatiblePartitionError("registered system field is incompatible or non-finite.")
    discrepancy = float(np.linalg.norm(regional - registered))
    scale = max(1.0, float(np.linalg.norm(regional)), float(np.linalg.norm(registered)))
    if discrepancy > float(partition.compatibility_tolerance) * scale:
        raise IncompatiblePartitionError(
            "partition field does not match the registered ChaoticSystem field at the requested state."
        )
    return regional


def evaluate_field(
    system: ChaoticSystem,
    state: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
    *,
    partition: PWLPartition | None = None,
) -> np.ndarray:
    """Evaluate the vector field, optionally through an exact PWL partition."""

    point = _validated_flow_state(system, state)
    active_partition = _resolved_partition(system, parameters, partition)
    if active_partition is not None:
        value = _validated_partition_field(system, point, parameters, active_partition)
    else:
        value = system.evaluate(point, parameters)
    value = np.asarray(value, dtype=float)
    if value.shape != (system.dimension,) or not np.all(np.isfinite(value)):
        raise ValueError("vector-field evaluation must be finite and match system dimension.")
    return value


def finite_difference_jacobian(
    field: Callable[[np.ndarray], np.ndarray],
    state: np.ndarray,
    *,
    relative_step: float | None = None,
) -> np.ndarray:
    """Return a scaled central-difference Jacobian.

    This fallback is intended for smooth fields without an analytic Jacobian.
    It must not be used across a non-smooth switching surface; use a declared
    :class:`~hidden_attractors.geometry.PWLPartition` there.
    """

    point = np.asarray(state, dtype=float)
    if point.ndim != 1 or point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError("state must be a finite non-empty vector.")
    step_scale = float(relative_step) if relative_step is not None else float(np.cbrt(np.finfo(float).eps))
    if not np.isfinite(step_scale) or step_scale <= 0.0:
        raise ValueError("relative_step must be positive and finite.")
    dimension = int(point.size)
    jacobian = np.empty((dimension, dimension), dtype=float)
    for column in range(dimension):
        step = step_scale * max(1.0, abs(float(point[column])))
        forward = point.copy()
        backward = point.copy()
        forward[column] += step
        backward[column] -= step
        f_forward = np.asarray(field(forward), dtype=float)
        f_backward = np.asarray(field(backward), dtype=float)
        if f_forward.shape != (dimension,) or f_backward.shape != (dimension,):
            raise ValueError("field returned an incompatible shape during differencing.")
        if not np.all(np.isfinite(f_forward)) or not np.all(np.isfinite(f_backward)):
            raise ValueError("field returned non-finite values during differencing.")
        jacobian[:, column] = (f_forward - f_backward) / (2.0 * step)
    return jacobian


def evaluate_jacobian(
    system: ChaoticSystem,
    state: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
    *,
    mode: JacobianMode = "analytic",
    relative_step: float | None = None,
    partition: PWLPartition | None = None,
) -> tuple[np.ndarray, str, str | None]:
    """Return ``(Jacobian, source, region)`` under an explicit method policy."""

    point = _validated_flow_state(system, state)
    if mode not in {"analytic", "finite_difference", "auto"}:
        raise ValueError("mode must be 'analytic', 'finite_difference', or 'auto'.")
    active_partition = _resolved_partition(system, parameters, partition)
    if active_partition is not None:
        _validated_partition_field(system, point, parameters, active_partition)
        region = active_partition.region_at(point)
        regional_jacobian = region.matrix.copy()
        if system.jacobian is not None:
            registered_jacobian = np.asarray(system.jacobian_matrix(point, parameters), dtype=float)
            if registered_jacobian.shape != regional_jacobian.shape or not np.all(
                np.isfinite(registered_jacobian)
            ):
                raise IncompatiblePartitionError(
                    "registered analytic Jacobian is incompatible or non-finite."
                )
            discrepancy = float(np.linalg.norm(regional_jacobian - registered_jacobian))
            scale = max(
                1.0,
                float(np.linalg.norm(regional_jacobian)),
                float(np.linalg.norm(registered_jacobian)),
            )
            if discrepancy > float(active_partition.compatibility_tolerance) * scale:
                raise IncompatiblePartitionError(
                    "regional Jacobian does not match the registered analytic Jacobian at the requested state."
                )
        return regional_jacobian, "regional_affine", region.name

    if mode in {"analytic", "auto"} and system.jacobian is not None:
        matrix = system.jacobian_matrix(point, parameters)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("analytic Jacobian returned non-finite values.")
        return matrix, "analytic", None
    if mode == "analytic":
        raise ValueError(
            f"{system.name} has no analytic Jacobian; select mode='finite_difference' or 'auto'."
        )
    matrix = finite_difference_jacobian(
        lambda value: system.evaluate(value, parameters),
        point,
        relative_step=relative_step,
    )
    return matrix, "finite_difference", None


def evaluate_differential_geometry(
    system: ChaoticSystem,
    state: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
    *,
    jacobian_mode: JacobianMode = "analytic",
    relative_step: float | None = None,
    partition: PWLPartition | None = None,
) -> DifferentialGeometryEvaluation:
    """Evaluate ``F(x)``, ``DF(x)`` and the flow acceleration ``DF(x)F(x)``."""

    point = _validated_flow_state(system, state)
    active_partition = _resolved_partition(system, parameters, partition)
    field = evaluate_field(system, point, parameters, partition=active_partition)
    jacobian, source, region = evaluate_jacobian(
        system,
        point,
        parameters,
        mode=jacobian_mode,
        relative_step=relative_step,
        partition=active_partition,
    )
    jacobian_field = jacobian @ field
    if not np.all(np.isfinite(jacobian_field)):
        raise ValueError("Jacobian-vector product returned non-finite values.")
    determinant = float(np.linalg.det(jacobian))
    if not np.isfinite(determinant):
        raise ValueError("Jacobian determinant is non-finite.")
    return DifferentialGeometryEvaluation(
        state=point.copy(),
        field=field.copy(),
        jacobian=jacobian.copy(),
        jacobian_field=jacobian_field,
        jacobian_determinant=determinant,
        jacobian_source=source,
        region=region,
    )


def jacobian_field_product(
    system: ChaoticSystem,
    state: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Convenience wrapper returning the integer flow acceleration ``DF F``."""

    return evaluate_differential_geometry(system, state, parameters, **kwargs).jacobian_field


__all__ = [
    "DifferentialGeometryEvaluation",
    "JacobianMode",
    "evaluate_differential_geometry",
    "evaluate_field",
    "evaluate_jacobian",
    "finite_difference_jacobian",
    "jacobian_field_product",
]
