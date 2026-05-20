"""Tests for the installable system registry and legacy facade."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from hidden_attractors.legacy import legacy_script_path, legacy_script_names, run_legacy_script
from hidden_attractors.systems import ChaoticSystem, get_system, list_systems, register_system


def test_builtin_chua_system_is_registered() -> None:
    assert "chua-piecewise" in list_systems()

    system = get_system("chua-piecewise")
    rhs = system.evaluate([0.0, 0.0, 0.0])
    equilibria = system.equilibrium_points()

    assert rhs.shape == (3,)
    assert set(equilibria) == {"E0", "E+", "E-"}
    assert "full" in system.workflows


def test_user_system_can_be_registered() -> None:
    def rhs(state: np.ndarray, p: Mapping[str, Any]) -> np.ndarray:
        return np.array([p["a"] * state[0]], dtype=float)

    register_system(
        ChaoticSystem(name="toy-1d", dimension=1, rhs=rhs, parameters={"a": -2.0}),
        replace=True,
    )

    assert get_system("toy-1d").evaluate([3.0]).tolist() == [-6.0]


def test_legacy_facade_exposes_packaged_scripts() -> None:
    assert "nyquist-pipeline" in legacy_script_names()
    assert legacy_script_path("nyquist-pipeline").exists()

    proc = run_legacy_script("nyquist-pipeline", ["--help"])
    assert proc.returncode == 0
