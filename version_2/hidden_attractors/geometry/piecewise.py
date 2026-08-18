"""Piecewise-affine geometry with explicit switching-boundary semantics.

Stability: experimental
    These contracts support the geometric/topological campaign.  A regional
    Jacobian is returned only in the interior of one declared region.  On a
    switching surface the vector field may remain continuous, but the
    classical Jacobian is deliberately treated as non-unique.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np

from ..models.chua import chua_parameters, normalize_chua_model

if TYPE_CHECKING:
    from ..systems.base import ChaoticSystem


RegionStatus = Literal["interior", "switching", "outside", "overlap"]


class PiecewiseGeometryError(ValueError):
    """Base error for invalid or non-unique piecewise geometry."""


class NondifferentiablePointError(PiecewiseGeometryError):
    """Raised when a single classical Jacobian is requested on a switch."""


class DiscontinuousFieldError(PiecewiseGeometryError):
    """Raised when adjacent affine fields disagree on a switching surface."""


class IncompatiblePartitionError(PiecewiseGeometryError):
    """Raised when a PWL partition is not bound to the requested system."""


def _normalized_identifier(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


@dataclass(frozen=True)
class PWLSystemBinding:
    """Auditable binding between a partition and one system parameterization.

    ``parameters`` contains only coefficients that affect the regional affine
    field.  The fingerprint is derived from the normalized system IDs, model,
    and these exact values; validation also reports the mismatching key rather
    than relying on the digest alone.
    """

    system_names: tuple[str, ...]
    model: str | None
    parameters: tuple[tuple[str, float | int | str | bool | None], ...]
    coefficient_fingerprint: str

    def __post_init__(self) -> None:
        names = tuple(_normalized_identifier(name) for name in self.system_names)
        if not names or any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("system_names must contain unique non-empty identifiers.")
        model = None if self.model is None else _normalized_identifier(self.model)
        if self.model is not None and not model:
            raise ValueError("model must be None or a non-empty identifier.")
        normalized_parameters: list[tuple[str, float | int | str | bool | None]] = []
        seen: set[str] = set()
        for raw_key, raw_value in self.parameters:
            key = str(raw_key)
            if not key or key in seen:
                raise ValueError("binding parameter names must be unique and non-empty.")
            seen.add(key)
            value: float | int | str | bool | None
            if isinstance(raw_value, (np.floating, float)):
                value = float(raw_value)
                if not np.isfinite(value):
                    raise ValueError("numeric binding parameters must be finite.")
            elif isinstance(raw_value, (np.integer, int)) and not isinstance(raw_value, bool):
                value = int(raw_value)
            elif isinstance(raw_value, (str, bool)) or raw_value is None:
                value = raw_value
            else:
                raise TypeError("binding parameter values must be scalar JSON values.")
            normalized_parameters.append((key, value))
        coefficient_fingerprint = str(self.coefficient_fingerprint).strip().lower()
        if len(coefficient_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in coefficient_fingerprint
        ):
            raise ValueError("coefficient_fingerprint must be a 64-character hexadecimal digest.")
        object.__setattr__(self, "system_names", names)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "parameters", tuple(sorted(normalized_parameters)))
        object.__setattr__(self, "coefficient_fingerprint", coefficient_fingerprint)

    @property
    def parameter_fingerprint(self) -> str:
        payload = {
            "system_names": self.system_names,
            "model": self.model,
            "parameters": self.parameters,
            "coefficient_fingerprint": self.coefficient_fingerprint,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return sha256(canonical.encode("ascii")).hexdigest()

    def validate_system(
        self,
        system: "ChaoticSystem",
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        system_name = _normalized_identifier(system.name)
        if system_name not in self.system_names:
            raise IncompatiblePartitionError(
                f"partition is bound to {self.system_names}, not system '{system.name}'."
            )
        effective = dict(system.parameters)
        if parameters:
            effective.update(parameters)
        if self.model is not None:
            raw_model = effective.get("model", "")
            if self.model in {"nonsmooth", "arctan"}:
                try:
                    actual_model = normalize_chua_model(raw_model)
                except (TypeError, ValueError):
                    actual_model = _normalized_identifier(raw_model)
            else:
                actual_model = _normalized_identifier(raw_model)
            if actual_model != self.model:
                raise IncompatiblePartitionError(
                    f"partition model '{self.model}' does not match system model '{actual_model}'."
                )
        for key, expected in self.parameters:
            if key not in effective:
                raise IncompatiblePartitionError(
                    f"system is missing partition parameter '{key}'."
                )
            actual = effective[key]
            if isinstance(expected, (float, int)) and not isinstance(expected, bool):
                try:
                    actual_numeric = float(actual)
                except (TypeError, ValueError) as exc:
                    raise IncompatiblePartitionError(
                        f"partition parameter '{key}' is not numeric in the system."
                    ) from exc
                matches = np.isfinite(actual_numeric) and actual_numeric == float(expected)
            else:
                matches = actual == expected
            if not matches:
                raise IncompatiblePartitionError(
                    f"partition parameter '{key}'={expected!r} does not match system value {actual!r}; "
                    "rebuild the partition for the active parameters."
                )


@dataclass(frozen=True)
class AffineHalfSpace:
    """Closed half-space ``normal @ x + offset >= 0``.

    Strict region interiors use a positive tolerance.  Closed membership is
    used only to identify the regions adjacent to a switching surface.
    """

    normal: np.ndarray
    offset: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=float)
        if normal.ndim != 1 or normal.size == 0:
            raise ValueError("half-space normal must be a non-empty vector.")
        if not np.all(np.isfinite(normal)) or not np.isfinite(float(self.offset)):
            raise ValueError("half-space coefficients must be finite.")
        if float(np.linalg.norm(normal)) == 0.0:
            raise ValueError("half-space normal must be nonzero.")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "offset", float(self.offset))

    @property
    def dimension(self) -> int:
        return int(self.normal.size)

    def margin(self, state: np.ndarray) -> float:
        point = np.asarray(state, dtype=float)
        if point.shape != (self.dimension,):
            raise ValueError(f"half-space expects state shape ({self.dimension},).")
        if not np.all(np.isfinite(point)):
            raise ValueError("state must contain only finite values.")
        return float(self.normal @ point + self.offset)


@dataclass(frozen=True)
class SwitchingSurface:
    """Affine switching surface ``normal @ x + offset = 0``."""

    name: str
    normal: np.ndarray
    offset: float = 0.0

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=float)
        if not self.name:
            raise ValueError("switching surface name cannot be empty.")
        if normal.ndim != 1 or normal.size == 0:
            raise ValueError("switching surface normal must be a non-empty vector.")
        if not np.all(np.isfinite(normal)) or not np.isfinite(float(self.offset)):
            raise ValueError("switching surface coefficients must be finite.")
        if float(np.linalg.norm(normal)) == 0.0:
            raise ValueError("switching surface normal must be nonzero.")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "offset", float(self.offset))

    @property
    def dimension(self) -> int:
        return int(self.normal.size)

    def residual(self, state: np.ndarray) -> float:
        point = np.asarray(state, dtype=float)
        if point.shape != (self.dimension,):
            raise ValueError(f"switching surface expects state shape ({self.dimension},).")
        if not np.all(np.isfinite(point)):
            raise ValueError("state must contain only finite values.")
        return float(self.normal @ point + self.offset)


@dataclass(frozen=True)
class AffineRegion:
    """One affine branch ``F(x)=A x+b`` and its half-space constraints."""

    name: str
    matrix: np.ndarray
    offset: np.ndarray
    constraints: tuple[AffineHalfSpace, ...]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        offset = np.asarray(self.offset, dtype=float)
        if not self.name:
            raise ValueError("region name cannot be empty.")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("region matrix must be square.")
        dimension = int(matrix.shape[0])
        if offset.shape != (dimension,):
            raise ValueError(f"region offset must have shape ({dimension},).")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset)):
            raise ValueError("region affine coefficients must be finite.")
        if not self.constraints:
            raise ValueError("region must declare at least one half-space constraint.")
        if any(constraint.dimension != dimension for constraint in self.constraints):
            raise ValueError("region constraints must match the affine-field dimension.")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    def field(self, state: np.ndarray) -> np.ndarray:
        point = np.asarray(state, dtype=float)
        if point.shape != (self.dimension,):
            raise ValueError(f"region expects state shape ({self.dimension},).")
        if not np.all(np.isfinite(point)):
            raise ValueError("state must contain only finite values.")
        return self.matrix @ point + self.offset

    def contains_closed(self, state: np.ndarray, *, tolerance: float) -> bool:
        return all(constraint.margin(state) >= -float(tolerance) for constraint in self.constraints)

    def contains_interior(self, state: np.ndarray, *, tolerance: float) -> bool:
        return all(constraint.margin(state) > float(tolerance) for constraint in self.constraints)


def _coefficient_fingerprint(
    regions: tuple[AffineRegion, ...],
    switching_surfaces: tuple[SwitchingSurface, ...],
) -> str:
    """Hash every coefficient that defines the regional field and partition."""

    payload = {
        "regions": [
            {
                "name": region.name,
                "matrix": region.matrix.tolist(),
                "offset": region.offset.tolist(),
                "constraints": [
                    {
                        "name": constraint.name,
                        "normal": constraint.normal.tolist(),
                        "offset": constraint.offset,
                    }
                    for constraint in region.constraints
                ],
            }
            for region in regions
        ],
        "switching_surfaces": [
            {
                "name": surface.name,
                "normal": surface.normal.tolist(),
                "offset": surface.offset,
            }
            for surface in switching_surfaces
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(canonical.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class RegionResolution:
    """Classification of one state relative to a piecewise partition."""

    status: RegionStatus
    region_names: tuple[str, ...]
    switching_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {"interior", "switching", "outside", "overlap"}:
            raise ValueError("invalid region-resolution status.")
        if any(not str(name) for name in self.region_names + self.switching_names):
            raise ValueError("region and switching names must be non-empty.")
        if self.status == "interior" and len(self.region_names) != 1:
            raise ValueError("interior resolution requires exactly one region.")
        if self.status == "switching" and not self.switching_names:
            raise ValueError("switching resolution requires a switching-surface name.")
        if self.status == "outside" and (self.region_names or self.switching_names):
            raise ValueError("outside resolution cannot name regions or switches.")
        if self.status == "overlap" and len(self.region_names) < 2:
            raise ValueError("overlap resolution requires at least two regions.")

    @property
    def is_differentiable(self) -> bool:
        return self.status == "interior" and len(self.region_names) == 1


@dataclass(frozen=True)
class PWLPartition:
    """Finite piecewise-affine partition used for exact regional geometry."""

    name: str
    regions: tuple[AffineRegion, ...]
    switching_surfaces: tuple[SwitchingSurface, ...]
    binding: PWLSystemBinding | None = None
    tolerance: float = 1.0e-12
    continuity_tolerance: float = 1.0e-10
    compatibility_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("partition name cannot be empty.")
        if not self.regions:
            raise ValueError("partition must contain at least one region.")
        dimension = self.regions[0].dimension
        if any(region.dimension != dimension for region in self.regions):
            raise ValueError("all regions must have the same dimension.")
        if any(surface.dimension != dimension for surface in self.switching_surfaces):
            raise ValueError("all switching surfaces must match the region dimension.")
        names = [region.name for region in self.regions]
        if len(names) != len(set(names)):
            raise ValueError("region names must be unique.")
        surface_names = [surface.name for surface in self.switching_surfaces]
        if len(surface_names) != len(set(surface_names)):
            raise ValueError("switching surface names must be unique.")
        if self.binding is not None and not isinstance(self.binding, PWLSystemBinding):
            raise TypeError("binding must be a PWLSystemBinding or None.")
        if (
            not np.isfinite(float(self.tolerance))
            or not np.isfinite(float(self.continuity_tolerance))
            or not np.isfinite(float(self.compatibility_tolerance))
            or self.tolerance < 0.0
            or self.continuity_tolerance < 0.0
            or self.compatibility_tolerance < 0.0
        ):
            raise ValueError("partition tolerances must be nonnegative.")

    @property
    def dimension(self) -> int:
        return self.regions[0].dimension

    @property
    def coefficient_fingerprint(self) -> str:
        """Digest of matrices, offsets, half-spaces and switching surfaces."""

        return _coefficient_fingerprint(self.regions, self.switching_surfaces)

    def _state(self, state: np.ndarray) -> np.ndarray:
        point = np.asarray(state, dtype=float)
        if point.shape != (self.dimension,):
            raise ValueError(f"{self.name} expects state shape ({self.dimension},).")
        if not np.all(np.isfinite(point)):
            raise ValueError("state must contain only finite values.")
        return point

    def switching_residuals(self, state: np.ndarray) -> dict[str, float]:
        point = self._state(state)
        return {surface.name: surface.residual(point) for surface in self.switching_surfaces}

    def validate_system(
        self,
        system: "ChaoticSystem",
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        """Reject use through the system API when provenance is absent or incompatible."""

        if self.binding is None:
            raise IncompatiblePartitionError(
                f"partition '{self.name}' has no system binding and cannot replace a ChaoticSystem field."
            )
        self.binding.validate_system(system, parameters)
        if self.coefficient_fingerprint != self.binding.coefficient_fingerprint:
            raise IncompatiblePartitionError(
                "partition regional coefficients do not match their bound coefficient fingerprint."
            )

    def resolve(self, state: np.ndarray) -> RegionResolution:
        point = self._state(state)
        switch_values = self.switching_residuals(point)
        switching = tuple(
            name for name, value in switch_values.items() if abs(value) <= self.tolerance
        )
        closed = tuple(
            region.name
            for region in self.regions
            if region.contains_closed(point, tolerance=self.tolerance)
        )
        interior = tuple(
            region.name
            for region in self.regions
            if region.contains_interior(point, tolerance=self.tolerance)
        )
        if switching:
            return RegionResolution("switching", closed, switching)
        if len(interior) == 1:
            return RegionResolution("interior", interior, ())
        if len(interior) > 1:
            return RegionResolution("overlap", interior, ())
        if len(closed) > 1:
            return RegionResolution("overlap", closed, ())
        if len(closed) == 1:
            return RegionResolution("interior", closed, ())
        return RegionResolution("outside", (), ())

    def _region_by_name(self, name: str) -> AffineRegion:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(name)

    def region_at(self, state: np.ndarray) -> AffineRegion:
        resolution = self.resolve(state)
        if resolution.status == "switching":
            switches = ", ".join(resolution.switching_names)
            raise NondifferentiablePointError(
                f"{self.name} has no unique classical Jacobian on switching surface(s): {switches}."
            )
        if not resolution.is_differentiable:
            raise PiecewiseGeometryError(
                f"state is not in exactly one region of {self.name}: {resolution.status}."
            )
        return self._region_by_name(resolution.region_names[0])

    def jacobian_at(self, state: np.ndarray) -> np.ndarray:
        """Return the exact affine Jacobian in a unique region interior."""

        return self.region_at(state).matrix.copy()

    def adjacent_jacobians(self, state: np.ndarray) -> dict[str, np.ndarray]:
        """Return every regional limiting Jacobian adjacent to *state*."""

        resolution = self.resolve(state)
        if resolution.status == "outside":
            raise PiecewiseGeometryError(f"state lies outside partition {self.name}.")
        return {
            name: self._region_by_name(name).matrix.copy()
            for name in resolution.region_names
        }

    def field_at(self, state: np.ndarray) -> np.ndarray:
        """Evaluate a continuous PWL field, including on a switching surface.

        On a switch, all adjacent affine formulas are checked.  A discontinuous
        field is rejected instead of silently choosing one side.
        """

        point = self._state(state)
        resolution = self.resolve(point)
        if resolution.status == "outside":
            raise PiecewiseGeometryError(f"state lies outside partition {self.name}.")
        if resolution.status == "overlap":
            raise PiecewiseGeometryError(f"ambiguous overlapping regions in {self.name}.")
        if not resolution.region_names:
            raise PiecewiseGeometryError(f"no adjacent region found in {self.name}.")
        values = [self._region_by_name(name).field(point) for name in resolution.region_names]
        reference = values[0]
        scale = max(1.0, float(np.linalg.norm(reference)))
        for value in values[1:]:
            if float(np.linalg.norm(value - reference)) > self.continuity_tolerance * scale:
                raise DiscontinuousFieldError(
                    f"adjacent affine branches of {self.name} disagree at the switching state."
                )
        return reference.copy()


def chua_nonsmooth_partition(
    parameters: Mapping[str, Any] | None = None,
    *,
    tolerance: float = 1.0e-12,
) -> PWLPartition:
    """Return the exact three-region PWL partition of non-smooth Chua."""

    supplied = dict(parameters or {})
    if "model" in supplied and normalize_chua_model(supplied["model"]) != "nonsmooth":
        raise ValueError("chua_nonsmooth_partition requires the non-smooth Chua model.")
    allowed = {"alpha", "beta", "gamma", "m0", "m1", "a1", "a2", "rho"}
    values = {key: value for key, value in supplied.items() if key in allowed}
    params = chua_parameters(model="nonsmooth", **values)

    def matrix(slope: float) -> np.ndarray:
        return np.array(
            [
                [-params.alpha * (1.0 + slope), params.alpha, 0.0],
                [1.0, -1.0, 1.0],
                [0.0, -params.beta, -params.gamma],
            ],
            dtype=float,
        )

    difference = float(params.m0 - params.m1)
    zero = np.zeros(3, dtype=float)
    left = AffineRegion(
        name="left",
        matrix=matrix(params.m1),
        offset=np.array([params.alpha * difference, 0.0, 0.0], dtype=float),
        constraints=(
            AffineHalfSpace(np.array([-1.0, 0.0, 0.0]), -1.0, "x<=-1"),
        ),
    )
    inner = AffineRegion(
        name="inner",
        matrix=matrix(params.m0),
        offset=zero,
        constraints=(
            AffineHalfSpace(np.array([1.0, 0.0, 0.0]), 1.0, "x>=-1"),
            AffineHalfSpace(np.array([-1.0, 0.0, 0.0]), 1.0, "x<=1"),
        ),
    )
    right = AffineRegion(
        name="right",
        matrix=matrix(params.m1),
        offset=np.array([-params.alpha * difference, 0.0, 0.0], dtype=float),
        constraints=(
            AffineHalfSpace(np.array([1.0, 0.0, 0.0]), -1.0, "x>=1"),
        ),
    )
    regions = (left, inner, right)
    switching_surfaces = (
        SwitchingSurface("x=-1", np.array([1.0, 0.0, 0.0]), 1.0),
        SwitchingSurface("x=+1", np.array([1.0, 0.0, 0.0]), -1.0),
    )
    return PWLPartition(
        name="chua-nonsmooth-three-region",
        regions=regions,
        switching_surfaces=switching_surfaces,
        binding=PWLSystemBinding(
            system_names=("chua-nonsmooth",),
            model="nonsmooth",
            parameters=(
                ("alpha", params.alpha),
                ("beta", params.beta),
                ("gamma", params.gamma),
                ("m0", params.m0),
                ("m1", params.m1),
            ),
            coefficient_fingerprint=_coefficient_fingerprint(regions, switching_surfaces),
        ),
        tolerance=float(tolerance),
    )


def registered_pwl_partition(
    system: "ChaoticSystem",
    parameters: Mapping[str, Any] | None = None,
) -> PWLPartition | None:
    """Return the maintained PWL partition automatically known for *system*.

    This registry is intentionally conservative: only the canonical non-smooth
    Chua system is auto-associated.  Unknown piecewise systems must supply an
    explicitly bound :class:`PWLPartition`.
    """

    system_name = _normalized_identifier(system.name)
    effective = dict(system.parameters)
    if parameters:
        effective.update(parameters)
    if system_name == "chua-nonsmooth" and normalize_chua_model(effective.get("model")) == "nonsmooth":
        return chua_nonsmooth_partition(effective)
    return None


__all__ = [
    "AffineHalfSpace",
    "AffineRegion",
    "DiscontinuousFieldError",
    "IncompatiblePartitionError",
    "NondifferentiablePointError",
    "PWLPartition",
    "PWLSystemBinding",
    "PiecewiseGeometryError",
    "RegionResolution",
    "RegionStatus",
    "SwitchingSurface",
    "chua_nonsmooth_partition",
    "registered_pwl_partition",
]
