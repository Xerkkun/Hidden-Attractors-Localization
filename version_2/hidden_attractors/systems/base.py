"""System-definition contracts for reusable chaotic-system workflows.

Stability: stable
    System dataclasses, registry API (``register_system``, ``get_system``,
    ``list_systems``), and capability checks.  Signatures are fixed.
"""

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
    """Definition of a dynamical system that can enter package workflows.

    Attributes
    ----------
    name : str
        Unique registry identifier (case-insensitive, hyphens normalised).
    dimension : int
        State-space dimension.
    rhs : callable
        Vector field ``F(state, params) -> state``.
    parameters : Mapping[str, Any], default {}
        Default parameter dict passed to *rhs* and *equilibria*.
    equilibria : callable or None, default None
        Returns ``{label: state}`` for the active parameter set.
    jacobian : callable or None, default None
        Analytic Jacobian ``J(state, params) -> (n, n) array``.
    description : str, default ''
        Human-readable one-line description shown by the CLI.
    tags : tuple[str, ...], default ()
        Searchable labels (e.g. ``'fractional'``, ``'hidden'``).
    workflows : Mapping[str, str], default {}
        Maps workflow names to CLI command strings.
    lure : LureSystem or None, default None
        Lur'e decomposition required for DF seed generation.
    full_workflow : FullWorkflowContract or None, default None
        Complete hiddenness-protocol contract for this system.

    Examples
    --------
    >>> from hidden_attractors.systems import get_system
    >>> sys = get_system('chua-fractional')
    >>> sys.dimension
    3
    """

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
        """Evaluate the system vector field at *state*.

        Parameters
        ----------
        state : array-like, shape (dimension,)
            Current system state.
        parameters : Mapping[str, Any] or None, default None
            Override parameters merged on top of ``self.parameters``.

        Returns
        -------
        dxdt : np.ndarray, shape (dimension,)
            Right-hand side of the ODE / FDE.

        Raises
        ------
        ValueError
            If *state* does not have shape ``(dimension,)`` or if the
            returned vector has an unexpected shape.

        Examples
        --------
        >>> import numpy as np
        >>> from hidden_attractors.systems import get_system
        >>> sys = get_system('chua-fractional')
        >>> sys.evaluate(np.zeros(3))
        array([0., 0., 0.])
        """

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
        """Return known equilibria for the system.

        Parameters
        ----------
        parameters : Mapping[str, Any] or None, default None
            Override parameters merged on top of ``self.parameters``.

        Returns
        -------
        equilibria : dict[str, np.ndarray]
            Maps label strings to state vectors.  Empty dict if the system
            does not define an equilibrium provider.

        Examples
        --------
        >>> from hidden_attractors.systems import get_system
        >>> eq = get_system('chua-fractional').equilibrium_points()
        >>> list(eq.keys())
        ['E0', 'E+', 'E-']
        """

        if self.equilibria is None:
            return {}
        merged = dict(self.parameters)
        if parameters:
            merged.update(parameters)
        return {str(k): np.asarray(v, dtype=float) for k, v in self.equilibria(merged).items()}

    def jacobian_matrix(self, state: State, parameters: Mapping[str, Any] | None = None) -> np.ndarray:
        """Evaluate the analytic Jacobian at *state*.

        Parameters
        ----------
        state : array-like, shape (dimension,)
            Point at which the Jacobian is evaluated.
        parameters : Mapping[str, Any] or None, default None
            Override parameters merged on top of ``self.parameters``.

        Returns
        -------
        J : np.ndarray, shape (dimension, dimension)
            Jacobian matrix ``∂F/∂x``.

        Raises
        ------
        ValueError
            If ``self.jacobian`` is ``None`` or if *state* has wrong shape.
        """

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
        """Return a lower-case, stripped, underscore-free registry key."""
        return str(name).strip().lower().replace("_", "-")

    def register(self, system: ChaoticSystem, *, replace: bool = False) -> ChaoticSystem:
        """Add *system* to the registry.

        Parameters
        ----------
        system : ChaoticSystem
            System definition to register.
        replace : bool, default False
            If ``False`` (default), raise :exc:`ValueError` when a system
            with the same name already exists.

        Returns
        -------
        system : ChaoticSystem
            The same object, for chaining.

        Raises
        ------
        ValueError
            If the system name is already registered and *replace* is ``False``.
        """
        key = self.normalize_name(system.name)
        if key in self._systems and not replace:
            raise ValueError(f"System already registered: {key}")
        self._systems[key] = system
        return system

    def get(self, name: str) -> ChaoticSystem:
        """Return a registered system by name.

        Parameters
        ----------
        name : str
            Registry key (case-insensitive, hyphens and underscores interchangeable).

        Returns
        -------
        system : ChaoticSystem

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        key = self.normalize_name(name)
        try:
            return self._systems[key]
        except KeyError as exc:
            known = ", ".join(self.list_names())
            raise KeyError(f"Unknown system '{name}'. Known systems: {known}") from exc

    def list_names(self) -> list[str]:
        """Return all registered system names in alphabetical order."""
        return sorted(self._systems)

    def values(self) -> list[ChaoticSystem]:
        """Return all registered systems in alphabetical name order."""
        return [self._systems[name] for name in self.list_names()]


_REGISTRY = SystemRegistry()
_SYSTEM_ALIASES = {"chua-piecewise": "chua-nonsmooth"}


def register_system(system: ChaoticSystem, *, replace: bool = False) -> ChaoticSystem:
    """Register a built-in or user-defined chaotic system.

    Parameters
    ----------
    system : ChaoticSystem
        The system definition to add to the global registry.
    replace : bool, default False
        Allow overwriting an existing entry with the same name.

    Returns
    -------
    system : ChaoticSystem
        The registered object (same instance), for chaining.

    Raises
    ------
    ValueError
        If the system is already registered and *replace* is ``False``.

    Examples
    --------
    >>> from hidden_attractors.systems.base import ChaoticSystem, register_system
    >>> import numpy as np
    >>> sys = ChaoticSystem(name='my-system', dimension=2,
    ...     rhs=lambda s, p: np.zeros(2))
    >>> register_system(sys, replace=True)  # doctest: +ELLIPSIS
    ChaoticSystem(name='my-system', ...)
    """

    return _REGISTRY.register(system, replace=replace)


def get_system(name: str) -> ChaoticSystem:
    """Return a registered system by name.

    Parameters
    ----------
    name : str
        Registry key (case-insensitive).

    Returns
    -------
    system : ChaoticSystem

    Raises
    ------
    KeyError
        If *name* is not in the registry.

    Examples
    --------
    >>> from hidden_attractors.systems import get_system
    >>> get_system('chua-fractional').dimension
    3
    """

    key = SystemRegistry.normalize_name(name)
    return _REGISTRY.get(_SYSTEM_ALIASES.get(key, key))


def list_systems() -> list[str]:
    """Return all registered system names in alphabetical order.

    Returns
    -------
    names : list[str]
        Sorted list of registry keys.

    Examples
    --------
    >>> from hidden_attractors.systems import list_systems
    >>> isinstance(list_systems(), list)
    True
    """

    return _REGISTRY.list_names()
