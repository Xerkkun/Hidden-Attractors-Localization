"""Regression checks for the Wu2023 ADM reproduction integrator."""

from __future__ import annotations

import numpy as np

from hidden_attractors.integrations.adm_wu2023 import adm_wu2023_integrate


def test_adm_wu2023_uses_nonlinear_n_parameter_not_loop_index() -> None:
    base = {
        "alpha": 8.4562,
        "beta": 12.0732,
        "gamma": 0.0052,
        "m": 0.4,
        "n": -1.1585,
    }
    changed = {**base, "n": -0.8}
    x0 = np.array([13.8, 0.7093, -19.8768], dtype=float)

    _, states_base, status_base, _ = adm_wu2023_integrate(base, x0, q=0.99, h=0.01, N=5)
    _, states_changed, status_changed, _ = adm_wu2023_integrate(changed, x0, q=0.99, h=0.01, N=5)

    assert status_base == "ok"
    assert status_changed == "ok"
    assert not np.allclose(states_base, states_changed)


def test_adm_wu2023_accepts_m_n_without_a2_alias() -> None:
    params = {
        "alpha": 8.4562,
        "beta": 12.0732,
        "gamma": 0.0052,
        "m": -0.2,
        "n": -1.0,
    }
    x0 = np.array([0.5, 0.0, -0.5], dtype=float)

    times, states, status, _ = adm_wu2023_integrate(params, x0, q=0.99, h=0.005, N=3)

    assert status == "ok"
    assert times.shape == (4,)
    assert states.shape == (4, 3)
    assert np.all(np.isfinite(states))
