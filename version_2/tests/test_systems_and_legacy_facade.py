"""Tests for the installable system registry."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from hidden_attractors.systems import ChaoticSystem, get_system, list_systems, register_system


def test_builtin_chua_system_is_registered() -> None:
    assert "chua-nonsmooth" in list_systems()
    assert "chua-piecewise" not in list_systems()

    system = get_system("chua-nonsmooth")
    rhs = system.evaluate([0.0, 0.0, 0.0])
    equilibria = system.equilibrium_points()

    assert rhs.shape == (3,)
    assert set(equilibria) == {"E0", "E+", "E-"}
    assert "full" in system.workflows
    assert get_system("chua-piecewise") is system


def test_user_system_can_be_registered() -> None:
    def rhs(state: np.ndarray, p: Mapping[str, Any]) -> np.ndarray:
        return np.array([p["a"] * state[0]], dtype=float)

    register_system(
        ChaoticSystem(name="toy-1d", dimension=1, rhs=rhs, parameters={"a": -2.0}),
        replace=True,
    )

    assert get_system("toy-1d").evaluate([3.0]).tolist() == [-6.0]
