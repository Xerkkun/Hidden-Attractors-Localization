"""Regression checks for the Wu2023 ADM reproduction integrator."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from hidden_attractors.integrations.adm_wu2023 import (
    adm_wu2023_integrate,
    rhs_chua_arctan,
)
from hidden_attractors.models.chua import chua_parameters, rhs_arctan


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


def test_adm_arctan_rhs_matches_project_model_for_nonunit_rho() -> None:
    params = {
        "alpha": 21.849356906616716,
        "beta": 19.081840840860202,
        "gamma": 0.007378011979156531,
        "a1": 0.04228979343578827,
        "a2": -3.3367815123026694,
        "rho": 1.7984259332820332,
    }
    model = chua_parameters(model="arctan", **params)
    state = np.array([0.73, -0.12, 0.31], dtype=float)

    np.testing.assert_allclose(
        rhs_chua_arctan(state, params),
        rhs_arctan(state, model),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_adm_q1_recovers_fourth_order_taylor_convergence() -> None:
    params = {
        "alpha": 8.4562,
        "beta": 12.0732,
        "gamma": 0.0052,
        "a1": 0.4,
        "a2": -1.5585,
        "rho": 1.3,
    }
    x0 = np.array([0.2, -0.1, 0.05], dtype=float)
    reference = solve_ivp(
        lambda _t, state: rhs_chua_arctan(state, params),
        (0.0, 0.1),
        x0,
        method="DOP853",
        rtol=1.0e-13,
        atol=1.0e-15,
    ).y[:, -1]

    _, coarse, status_coarse, _ = adm_wu2023_integrate(
        params, x0, q=1.0, h=0.01, N=10
    )
    _, fine, status_fine, _ = adm_wu2023_integrate(
        params, x0, q=1.0, h=0.005, N=20
    )

    assert status_coarse == status_fine == "ok"
    error_coarse = np.linalg.norm(coarse[-1] - reference)
    error_fine = np.linalg.norm(fine[-1] - reference)
    assert error_fine < error_coarse / 10.0
