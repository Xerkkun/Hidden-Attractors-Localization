"""Short numerical diagnostics for the block-restarted component-wise ABM."""

from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.analysis import compute_cloned_dynamics_spectrum
from hidden_attractors.integrations.abm_fractional import integrate_fractional_abm


def test_abm_q1_linear_matches_short_exact_exponential() -> None:
    rate = -0.4
    times, states, status = integrate_fractional_abm(
        lambda x: rate * x,
        np.asarray([1.0]),
        orders=[1.0],
        h=0.01,
        n_steps=100,
    )
    assert status == "ok"
    np.testing.assert_allclose(states[:, 0], np.exp(rate * times), atol=2e-5)


def test_abm_fractional_scalar_decay_is_finite_and_monotone() -> None:
    _, states, status = integrate_fractional_abm(
        lambda x: -x,
        np.asarray([1.0]),
        orders=[0.8],
        h=0.01,
        n_steps=100,
    )
    assert status == "ok"
    assert np.all(np.isfinite(states))
    assert np.all(np.diff(states[:, 0]) < 0.0)
    assert 0.0 < states[-1, 0] < states[0, 0]


def test_abm_incommensurate_components_decay_with_component_orders() -> None:
    _, states, status = integrate_fractional_abm(
        lambda x: -x,
        np.asarray([1.0, 1.0, 1.0]),
        orders=[0.6, 0.8, 1.0],
        h=0.01,
        n_steps=100,
    )
    assert status == "ok"
    assert np.all(np.isfinite(states))
    assert np.all(states[-1] < states[0])
    assert np.all(states[-1] > 0.0)
    assert len({round(value, 8) for value in states[-1]}) == 3


def test_cloned_dynamics_diagnostic_gs_classical_and_q1_rk4_reference_run() -> None:
    common = {
        "rhs": lambda x: -x,
        "x0": np.asarray([1.0, 1.0]),
        "orders": [1.0],
        "h": 0.01,
        "t_clone": 0.1,
        "n_clones": 2,
        "k_blocks": 2,
        "delta": 1e-3,
    }
    gs_result = compute_cloned_dynamics_spectrum(**common, method="gs_classical")
    rk4_result = compute_cloned_dynamics_spectrum(
        **common,
        integration_mode="integer_rk4_reference",
    )
    assert gs_result.status == "ok"
    assert gs_result.method_metadata["orthonormalization"] == "gs_classical"
    assert rk4_result.status == "ok"
    assert rk4_result.method_metadata["integration_mode"] == "integer_rk4_reference"


def test_cloned_dynamics_q1_rk4_reference_rejects_fractional_orders() -> None:
    with pytest.raises(ValueError, match="only when all orders are q=1"):
        compute_cloned_dynamics_spectrum(
            lambda x: -x,
            np.asarray([1.0]),
            orders=[0.9],
            h=0.01,
            t_clone=0.1,
            n_clones=1,
            k_blocks=1,
            delta=1e-3,
            integration_mode="integer_rk4_reference",
        )
