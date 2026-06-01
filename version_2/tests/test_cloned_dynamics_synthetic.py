"""Fast synthetic checks for F3 cloned dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.analysis import (
    compute_cloned_dynamics_spectrum,
    compute_lyapunov_spectrum,
)


def _run_linear(delta: float = 1e-3):
    matrix = np.diag([0.1, -0.2, -0.5])
    return compute_cloned_dynamics_spectrum(
        lambda x: matrix @ x,
        np.ones(3),
        orders=[1.0],
        h=0.01,
        t_clone=0.5,
        n_clones=3,
        k_blocks=8,
        delta=delta,
    )


def test_integer_diagonal_linear_spectrum() -> None:
    result = _run_linear()
    assert result.status == "ok"
    np.testing.assert_allclose(result.exponents, [0.1, -0.2, -0.5], atol=0.05)
    assert np.array_equal(np.sign(result.exponents), [1, -1, -1])


def test_fractional_stable_system_is_bounded_and_nonpositive() -> None:
    result = compute_cloned_dynamics_spectrum(
        lambda x: -x,
        np.ones(2),
        orders=[0.9],
        h=0.02,
        t_clone=0.2,
        n_clones=2,
        k_blocks=5,
        delta=1e-3,
    )
    assert result.status == "ok"
    assert result.bounded_trajectory is True
    assert np.all(np.isfinite(result.exponents))
    assert np.all(result.exponents <= 0.0)


def test_small_delta_scale_preserves_sign_pattern() -> None:
    first = _run_linear(1e-3)
    second = _run_linear(1e-4)
    assert np.array_equal(np.sign(first.exponents), np.sign(second.exponents))
    np.testing.assert_allclose(first.exponents, second.exponents, atol=0.05)


def test_no_jacobian_required_and_incommensurate_orders_run() -> None:
    result = compute_cloned_dynamics_spectrum(
        lambda x: -x,
        np.ones(2),
        orders=[0.9, 1.0],
        h=0.02,
        t_clone=0.1,
        n_clones=2,
        k_blocks=2,
        delta=1e-3,
    )
    assert result.status == "ok"
    assert result.jacobian_required is False
    assert result.method_metadata["order_class"] == "incommensurate_fractional"


def test_api_dispatches_cloned_dynamics_without_jacobian() -> None:
    summary = compute_lyapunov_spectrum(
        rhs=lambda x: -x,
        x0=np.ones(2),
        q=0.9,
        orders=[0.9],
        method="fractional_cloned_dynamics_abm_gs_published",
        h=0.02,
        t_final=0.4,
        t_clone=0.2,
        k_blocks=2,
        delta=1e-3,
    )
    assert summary.result.status == "ok"
    assert summary.result.method_id == "fractional_cloned_dynamics_abm_gs_published"
    assert "cloned_dynamics_no_jacobian_required" in summary.warnings


@pytest.mark.parametrize("orders", [[0.0], [1.1], [0.9, 0.8, 0.7]])
def test_bad_orders_are_rejected(orders: list[float]) -> None:
    with pytest.raises(ValueError):
        compute_cloned_dynamics_spectrum(
            lambda x: -x,
            np.ones(2),
            orders=orders,
            h=0.02,
            t_clone=0.1,
            n_clones=2,
            k_blocks=2,
            delta=1e-3,
        )
