"""Contracts for reusable Lur'e-form chaotic systems.

Stability: stable
    System dataclasses, registry API, and capability checks.  Signatures
    are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ScalarNonlinearity = Callable[[float], float]
DescribingFunction = Callable[[float], complex | float]
MachadoDescribingFunction = Callable[[float, float], complex | float]
GainCompatibility = Callable[[float], bool]
AmplitudeSolver = Callable[[float], float]


@dataclass(frozen=True)
class LureSystem:
    """Representation of the fractional Lur'e system ``D^q x = Ax + b·ψ(c^T x)``.

    The dataclass stores only the linear split and the scalar nonlinear
    response.  Describing-function workflows use the optional gain and
    amplitude callbacks when a closed-form or numerical relation is known.

    Attributes
    ----------
    name : str
        Human-readable identifier.
    matrix : np.ndarray, shape (n, n)
        Linear part ``A`` of the Lur'e representation.
    input_vector : np.ndarray, shape (n,)
        Input direction ``b`` (column vector).
    output_vector : np.ndarray, shape (n,)
        Output direction ``c`` (row vector); the feedback is ``σ = c^T x``.
    nonlinearity : callable
        Scalar nonlinearity ``ψ(σ) -> float``.
    describing_function : callable
        Classical describing function ``N(A) -> complex | float``.
    machado_describing_function : callable
        Machado-family describing function ``N_μ(A, μ) -> complex | float``.
    gain_compatible : callable or None, default None
        Returns ``True`` if a given gain can be produced by the DF model.
        If ``None``, all gains are accepted.
    amplitude_from_gain : callable or None, default None
        Closed-form amplitude solver ``k -> A`` for the classical DF.
        If ``None``, numerical bisection is used by the seed generators.
    description : str, default ''
        One-line description for the CLI and documentation.

    Notes
    -----
    This dataclass enforces that *matrix* is square and that *input_vector*
    and *output_vector* have shapes consistent with the matrix dimension.

    Examples
    --------
    >>> from hidden_attractors.systems import get_system
    >>> sys = get_system('chua-fractional')
    >>> lure = sys.lure
    >>> lure.dimension
    3
    """

    name: str
    matrix: np.ndarray
    input_vector: np.ndarray
    output_vector: np.ndarray
    nonlinearity: ScalarNonlinearity
    describing_function: DescribingFunction
    machado_describing_function: MachadoDescribingFunction
    gain_compatible: GainCompatibility | None = None
    amplitude_from_gain: AmplitudeSolver | None = None
    description: str = ""

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        input_vector = np.asarray(self.input_vector, dtype=float)
        output_vector = np.asarray(self.output_vector, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square.")
        dimension = matrix.shape[0]
        if input_vector.shape != (dimension,):
            raise ValueError(f"input_vector must have shape ({dimension},).")
        if output_vector.shape != (dimension,):
            raise ValueError(f"output_vector must have shape ({dimension},).")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "input_vector", input_vector)
        object.__setattr__(self, "output_vector", output_vector)

    @property
    def dimension(self) -> int:
        """State dimension ``n`` of the Lur'e representation.

        Returns
        -------
        n : int
            Number of state variables.
        """

        return int(self.matrix.shape[0])

    def sigma(self, state: np.ndarray) -> float:
        """Return the scalar feedback coordinate ``σ = c^T x``.

        Parameters
        ----------
        state : np.ndarray, shape (dimension,)
            Current system state.

        Returns
        -------
        sigma : float
            Output coordinate used as input to the nonlinearity.

        Raises
        ------
        ValueError
            If *state* does not have shape ``(dimension,)``.
        """

        x = np.asarray(state, dtype=float)
        if x.shape != (self.dimension,):
            raise ValueError(f"{self.name} expects a state of shape ({self.dimension},).")
        return float(self.output_vector @ x)

    def evaluate(self, state: np.ndarray) -> np.ndarray:
        """Evaluate the Lur'e vector field ``Ax + b·ψ(c^T x)``.

        Parameters
        ----------
        state : np.ndarray, shape (dimension,)
            Current system state.

        Returns
        -------
        dxdt : np.ndarray, shape (dimension,)
            Right-hand side of the fractional / integer ODE.
        """

        x = np.asarray(state, dtype=float)
        sigma = self.sigma(x)
        return self.matrix @ x + self.input_vector * float(self.nonlinearity(sigma))

    def is_gain_compatible(self, gain: float) -> bool:
        """Return whether *gain* can be produced by the describing-function model.

        Parameters
        ----------
        gain : float
            Describing-function gain ``k = -1 / Re(W_q(iω))``.

        Returns
        -------
        compatible : bool
            ``True`` when ``gain_compatible`` is ``None`` (all gains
            accepted) or when ``self.gain_compatible(gain)`` is truthy.
        """

        if self.gain_compatible is None:
            return True
        return bool(self.gain_compatible(float(gain)))

    def solve_amplitude(self, gain: float) -> float:
        """Solve the classical describing-function amplitude relation for *gain*.

        Parameters
        ----------
        gain : float
            Target describing-function gain ``k``.

        Returns
        -------
        amplitude : float
            Oscillation amplitude ``A`` satisfying ``N(A) = k``.

        Raises
        ------
        RuntimeError
            If ``self.amplitude_from_gain`` is ``None`` (no closed-form
            solver has been supplied).
        """

        if self.amplitude_from_gain is None:
            raise RuntimeError(f"{self.name} does not define amplitude_from_gain.")
        return float(self.amplitude_from_gain(float(gain)))


__all__ = [
    "AmplitudeSolver",
    "DescribingFunction",
    "GainCompatibility",
    "LureSystem",
    "MachadoDescribingFunction",
    "ScalarNonlinearity",
]
