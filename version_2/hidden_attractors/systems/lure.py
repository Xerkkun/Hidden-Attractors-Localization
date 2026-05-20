"""Contracts for reusable Lur'e-form chaotic systems."""

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
    """Representation of ``D^q x = A x + b psi(c^T x)``.

    The object is intentionally small: it stores the linear Lur'e split and the
    scalar nonlinear response.  Describing-function workflows can use optional
    gain and amplitude callbacks when a closed-form or numerical relation is
    known for the selected nonlinearity.
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
        """State dimension of the Lur'e representation."""

        return int(self.matrix.shape[0])

    def sigma(self, state: np.ndarray) -> float:
        """Return the scalar feedback coordinate ``c^T x``."""

        x = np.asarray(state, dtype=float)
        if x.shape != (self.dimension,):
            raise ValueError(f"{self.name} expects a state of shape ({self.dimension},).")
        return float(self.output_vector @ x)

    def evaluate(self, state: np.ndarray) -> np.ndarray:
        """Evaluate the Lur'e vector field."""

        x = np.asarray(state, dtype=float)
        sigma = self.sigma(x)
        return self.matrix @ x + self.input_vector * float(self.nonlinearity(sigma))

    def is_gain_compatible(self, gain: float) -> bool:
        """Return whether ``gain`` can be produced by the available DF model."""

        if self.gain_compatible is None:
            return True
        return bool(self.gain_compatible(float(gain)))

    def solve_amplitude(self, gain: float) -> float:
        """Solve the amplitude relation for ``gain`` if the system provides one."""

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
