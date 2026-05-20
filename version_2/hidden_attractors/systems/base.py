"""System-definition contracts for reusable chaotic-system workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from .lure import LureSystem
    from ..workflows.contracts import FullWorkflowContract


State = Sequence[float] | np.ndarray
RhsFunction = Callable[[State, Mapping[str, Any]], State]
EquilibriaFunction = Callable[[Mapping[str, Any]], Mapping[str, State]]
JacobianFunction = Callable[[State, Mapping[str, Any]], np.ndarray]


@dataclass(frozen=True)
class ChaoticSystem:
    """Definition of a dynamical system that can enter package workflows."""

    name: str
    dimension: int
    rhs: RhsFunction
    parameters: Mapping[str, Any] = field(default_factory=dict)
    equilibria: EquilibriaFunction | None = None
    jacobian: JacobianFunction | None = None
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    workflows: Mapping[str, str] = field(default_factory=dict)
    lure: "LureSystem | None" = None
    full_workflow: "FullWorkflowContract | None" = None

    def evaluate(self, state: State, parameters: Mapping[str, Any] | None = None) -> np.ndarray:
        """Evaluate the system vector field at ``state``."""

        x = np.asarray(state, dtype=float)
        if x.shape != (self.dimension,):
            raise ValueError(f"{self.name} expects a state of shape ({self.dimension},).")
        merged = dict(self.parameters)
        if parameters:
            merged.update(parameters)
        out = np.asarray(self.rhs(x, merged), dtype=float)
        if out.shape != (self.dimension,):
            raise ValueError(f"{self.name} rhs returned shape {out.shape}, expected ({self.dimension},).")
        return out

    def equilibrium_points(self, parameters: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
        """Return known equilibria for the system, if an equilibrium provider exists."""

        if self.equilibria is None:
            return {}
        merged = dict(self.parameters)
        if parameters:
            merged.update(parameters)
        return {str(k): np.asarray(v, dtype=float) for k, v in self.equilibria(merged).items()}

    def jacobian_matrix(self, state: State, parameters: Mapping[str, Any] | None = None) -> np.ndarray:
        """Evaluate an analytic Jacobian supplied by the system definition."""

        if self.jacobian is None:
            raise ValueError(f"{self.name} does not define an analytic Jacobian.")
        x = np.asarray(state, dtype=float)
        if x.shape != (self.dimension,):
            raise ValueError(f"{self.name} expects a state of shape ({self.dimension},).")
        merged = dict(self.parameters)
        if parameters:
            merged.update(parameters)
        out = np.asarray(self.jacobian(x, merged), dtype=float)
        if out.shape != (self.dimension, self.dimension):
            raise ValueError(f"{self.name} jacobian returned shape {out.shape}, expected ({self.dimension}, {self.dimension}).")
        return out


class SystemRegistry:
    """Mutable registry for built-in and user-defined chaotic systems."""

    def __init__(self) -> None:
        self._systems: MutableMapping[str, ChaoticSystem] = {}

    @staticmethod
    def normalize_name(name: str) -> str:
        return str(name).strip().lower().replace("_", "-")

    def register(self, system: ChaoticSystem, *, replace: bool = False) -> ChaoticSystem:
        key = self.normalize_name(system.name)
        if key in self._systems and not replace:
            raise ValueError(f"System already registered: {key}")
        self._systems[key] = system
        return system

    def get(self, name: str) -> ChaoticSystem:
        key = self.normalize_name(name)
        try:
            return self._systems[key]
        except KeyError as exc:
            known = ", ".join(self.list_names())
            raise KeyError(f"Unknown system '{name}'. Known systems: {known}") from exc

    def list_names(self) -> list[str]:
        return sorted(self._systems)

    def values(self) -> list[ChaoticSystem]:
        return [self._systems[name] for name in self.list_names()]


_REGISTRY = SystemRegistry()


def register_system(system: ChaoticSystem, *, replace: bool = False) -> ChaoticSystem:
    """Register a built-in or user-defined system."""

    return _REGISTRY.register(system, replace=replace)


def get_system(name: str) -> ChaoticSystem:
    """Return a registered system by name."""

    return _REGISTRY.get(name)


def list_systems() -> list[str]:
    """Return all registered system names."""

    return _REGISTRY.list_names()
