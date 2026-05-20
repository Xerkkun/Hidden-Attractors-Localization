"""Finite-memory history windows for EFORK/Caputo workflow stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FractionalHistory:
    """Discrete finite-memory window transported between continuation stages."""

    t_window: np.ndarray
    x_window: np.ndarray
    q: float
    h: float
    memory_length: float
    f_window: np.ndarray | None = None

    @property
    def memory_points(self) -> int:
        """Number of stored samples in the history window."""

        return int(self.x_window.shape[0])

    @property
    def dimension(self) -> int:
        """State dimension stored in the history window."""

        return int(self.x_window.shape[1])

    def as_efork_history(self) -> np.ndarray:
        """Return columns ``t,state...`` accepted by continuation wrappers."""

        return np.column_stack([self.t_window, self.x_window]).astype(float)

    @classmethod
    def from_trajectory(
        cls,
        trajectory: np.ndarray,
        *,
        q: float,
        h: float,
        memory_length: float,
        rhs: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> "FractionalHistory":
        """Extract the last ``ceil(memory_length / h)+1`` samples.

        ``trajectory`` must contain columns ``t,state...``.  Times are shifted so
        the final sample is at ``t=0``; this is the convention used by the
        migrated continuation routines.
        """

        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim != 2 or traj.shape[1] < 2 or traj.shape[0] < 1:
            raise ValueError("trajectory must have shape (N, d+1) with columns t,state....")
        h_value = float(h)
        memory = float(memory_length)
        if h_value <= 0.0:
            raise ValueError("h must be positive.")
        if memory <= 0.0:
            raise ValueError("memory_length must be positive.")

        points = max(1, int(np.ceil(memory / h_value))) + 1
        window = traj[-min(traj.shape[0], points):].copy()
        window[:, 0] -= window[-1, 0]
        f_window = None
        if rhs is not None:
            try:
                f_window = np.vstack([rhs(row[1:]) for row in window]).astype(float)
            except Exception:
                f_window = None
        return cls(
            t_window=window[:, 0].copy(),
            x_window=window[:, 1:].copy(),
            f_window=f_window,
            q=float(q),
            h=h_value,
            memory_length=memory,
        )


__all__ = ["FractionalHistory"]
