"""Explicit affine equivariances for geometric seed reduction.

Stability: experimental
    Symmetries are declared, then checked numerically against a finite set of
    states.  Passing this validator is evidence for an implementation contract,
    not a symbolic proof of equivariance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..systems.base import ChaoticSystem


@dataclass(frozen=True)
class AffineSymmetry:
    """Affine state transform ``T(x)=M x+b`` for an autonomous flow."""

    name: str
    matrix: np.ndarray
    offset: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        offset = np.asarray(self.offset, dtype=float)
        if not self.name:
            raise ValueError("symmetry name cannot be empty.")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("symmetry matrix must be square.")
        dimension = int(matrix.shape[0])
        if offset.shape != (dimension,):
            raise ValueError(f"symmetry offset must have shape ({dimension},).")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset)):
            raise ValueError("symmetry coefficients must be finite.")
        if np.linalg.matrix_rank(matrix) != dimension:
            raise ValueError("symmetry matrix must be invertible.")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    def transform_state(self, state: np.ndarray) -> np.ndarray:
        point = np.asarray(state, dtype=float)
        if point.shape != (self.dimension,):
            raise ValueError(f"symmetry expects state shape ({self.dimension},).")
        return self.matrix @ point + self.offset

    def pushforward_vector(self, vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=float)
        if value.shape != (self.dimension,):
            raise ValueError(f"symmetry expects vector shape ({self.dimension},).")
        return self.matrix @ value

    def inverse(self, *, name: str | None = None) -> "AffineSymmetry":
        inverse_matrix = np.linalg.inv(self.matrix)
        return AffineSymmetry(
            name=name or f"{self.name}^-1",
            matrix=inverse_matrix,
            offset=-(inverse_matrix @ self.offset),
        )


@dataclass(frozen=True)
class SymmetryValidation:
    """Finite-sample residual report for ``F(Tx)=DT F(x)``."""

    symmetry_name: str
    sample_count: int
    passed: bool
    max_absolute_residual: float
    max_scaled_residual: float
    absolute_tolerance: float
    relative_tolerance: float
    residuals: tuple[float, ...]
    evidence_scope: str = "finite_sample_equivariance_check_not_symbolic_proof"

    def __post_init__(self) -> None:
        if not self.symmetry_name or not self.evidence_scope:
            raise ValueError("symmetry_name and evidence_scope cannot be empty.")
        count = int(self.sample_count)
        if count < 1 or count != self.sample_count:
            raise ValueError("sample_count must be a positive integer.")
        residuals = tuple(float(value) for value in self.residuals)
        if len(residuals) != count or any(not np.isfinite(value) or value < 0.0 for value in residuals):
            raise ValueError("residuals must be finite, nonnegative, and match sample_count.")
        scalars = (
            float(self.max_absolute_residual),
            float(self.max_scaled_residual),
            float(self.absolute_tolerance),
            float(self.relative_tolerance),
        )
        if any(not np.isfinite(value) or value < 0.0 for value in scalars):
            raise ValueError("symmetry residual summaries and tolerances must be finite and nonnegative.")
        if not np.isclose(scalars[0], max(residuals), rtol=1.0e-12, atol=1.0e-15):
            raise ValueError("max_absolute_residual is inconsistent with residuals.")
        object.__setattr__(self, "sample_count", count)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "max_absolute_residual", scalars[0])
        object.__setattr__(self, "max_scaled_residual", scalars[1])
        object.__setattr__(self, "absolute_tolerance", scalars[2])
        object.__setattr__(self, "relative_tolerance", scalars[3])


@dataclass(frozen=True)
class SymmetryImage:
    """One point generated from a source point and a declared transform."""

    state: np.ndarray
    source_index: int
    transform_name: str

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=float)
        source_index = int(self.source_index)
        if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
            raise ValueError("symmetry image state must be a finite non-empty vector.")
        if source_index < 0 or source_index != self.source_index:
            raise ValueError("source_index must be a nonnegative integer.")
        if not self.transform_name:
            raise ValueError("transform_name cannot be empty.")
        object.__setattr__(self, "state", state.copy())
        object.__setattr__(self, "source_index", source_index)


def identity_symmetry(dimension: int, *, name: str = "identity") -> AffineSymmetry:
    size = int(dimension)
    if size < 1 or size != dimension:
        raise ValueError("dimension must be positive.")
    return AffineSymmetry(name, np.eye(size, dtype=float), np.zeros(size, dtype=float))


def sign_flip_symmetry(signs: Sequence[float], *, name: str = "sign_flip") -> AffineSymmetry:
    diagonal = np.asarray(tuple(signs), dtype=float)
    if diagonal.ndim != 1 or diagonal.size == 0:
        raise ValueError("signs must be a non-empty sequence.")
    if not np.all(np.isin(diagonal, (-1.0, 1.0))):
        raise ValueError("every sign must be -1 or +1.")
    return AffineSymmetry(name, np.diag(diagonal), np.zeros(diagonal.size, dtype=float))


def translation_symmetry(offset: Sequence[float], *, name: str = "translation") -> AffineSymmetry:
    shift = np.asarray(tuple(offset), dtype=float)
    if shift.ndim != 1 or shift.size == 0 or not np.all(np.isfinite(shift)):
        raise ValueError("translation offset must be a finite non-empty sequence.")
    return AffineSymmetry(name, np.eye(shift.size, dtype=float), shift)


def validate_affine_symmetry(
    system: ChaoticSystem,
    symmetry: AffineSymmetry,
    sample_points: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
    *,
    absolute_tolerance: float = 1.0e-10,
    relative_tolerance: float = 1.0e-9,
) -> SymmetryValidation:
    """Check flow equivariance ``F(Tx)=M F(x)`` on declared sample points."""

    if not isinstance(system, ChaoticSystem) or not system.is_continuous:
        raise ValueError("affine flow-symmetry validation requires a ChaoticSystem flow.")
    if symmetry.dimension != system.dimension:
        raise ValueError("symmetry dimension does not match the system.")
    points = np.asarray(sample_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != system.dimension or points.shape[0] == 0:
        raise ValueError("sample_points must have shape (n_samples, system.dimension).")
    if not np.all(np.isfinite(points)):
        raise ValueError("sample_points must contain only finite values.")
    atol = float(absolute_tolerance)
    rtol = float(relative_tolerance)
    if atol < 0.0 or rtol < 0.0 or not np.isfinite(atol + rtol):
        raise ValueError("symmetry tolerances must be finite and nonnegative.")

    residuals: list[float] = []
    scaled_residuals: list[float] = []
    decisions: list[bool] = []
    for point in points:
        transformed = symmetry.transform_state(point)
        lhs = system.evaluate(transformed, parameters)
        rhs = symmetry.pushforward_vector(system.evaluate(point, parameters))
        if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
            raise ValueError("system returned non-finite values during symmetry validation.")
        residual = float(np.linalg.norm(lhs - rhs))
        scale = max(1.0, float(np.linalg.norm(lhs)), float(np.linalg.norm(rhs)))
        residuals.append(residual)
        scaled_residuals.append(residual / scale)
        decisions.append(residual <= atol + rtol * scale)
    return SymmetryValidation(
        symmetry_name=symmetry.name,
        sample_count=int(points.shape[0]),
        passed=bool(all(decisions)),
        max_absolute_residual=max(residuals),
        max_scaled_residual=max(scaled_residuals),
        absolute_tolerance=atol,
        relative_tolerance=rtol,
        residuals=tuple(residuals),
    )


def _append_if_new(
    images: list[SymmetryImage],
    candidate: SymmetryImage,
    *,
    tolerance: float,
) -> None:
    if all(float(np.linalg.norm(candidate.state - item.state)) > tolerance for item in images):
        images.append(candidate)


def generate_symmetry_images(
    points: np.ndarray,
    symmetries: Sequence[AffineSymmetry],
    *,
    include_identity: bool = True,
    include_inverses: bool = False,
    deduplication_tolerance: float = 1.0e-10,
) -> tuple[SymmetryImage, ...]:
    """Generate one-step symmetry images with provenance and deduplication.

    The operation is intentionally one-step.  It does not close the generated
    group, which prevents an unbounded orbit for translational symmetries such
    as the lifted PLL angle ``v -> v+2*pi``.
    """

    values = np.asarray(points, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("points must be a non-empty vector or two-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must contain only finite values.")
    tolerance = float(deduplication_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("deduplication_tolerance must be finite and nonnegative.")
    transforms = list(symmetries)
    if any(transform.dimension != values.shape[1] for transform in transforms):
        raise ValueError("all symmetries must match the point dimension.")
    if include_inverses:
        transforms.extend(transform.inverse() for transform in symmetries)

    images: list[SymmetryImage] = []
    if include_identity:
        for index, point in enumerate(values):
            _append_if_new(
                images,
                SymmetryImage(point.copy(), index, "identity"),
                tolerance=tolerance,
            )
    for transform in transforms:
        for index, point in enumerate(values):
            _append_if_new(
                images,
                SymmetryImage(transform.transform_state(point), index, transform.name),
                tolerance=tolerance,
            )
    return tuple(images)


__all__ = [
    "AffineSymmetry",
    "SymmetryImage",
    "SymmetryValidation",
    "generate_symmetry_images",
    "identity_symmetry",
    "sign_flip_symmetry",
    "translation_symmetry",
    "validate_affine_symmetry",
]
