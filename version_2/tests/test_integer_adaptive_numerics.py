"""Focused tests for adaptive q=1 integration and variational QR."""

from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.analysis import (
    AdaptiveLyapunovResult,
    integer_dop853_variational_qr,
    integer_system_dop853_variational_qr,
)
from hidden_attractors.solvers import dop853_q1_integrate
from hidden_attractors.systems.base import ChaoticSystem


def test_dop853_q1_integrate_reports_uniform_samples_and_exact_final_time() -> None:
    trajectory, status = dop853_q1_integrate(
        lambda state: -state,
        np.array([1.0, 2.0]),
        t_final=0.23,
        h=0.1,
        rtol=1.0e-11,
        atol=1.0e-13,
    )

    assert status == "ok"
    assert trajectory.shape == (4, 3)
    assert np.allclose(trajectory[:, 0], [0.0, 0.1, 0.2, 0.23])
    assert np.allclose(trajectory[-1, 1:], np.array([1.0, 2.0]) * np.exp(-0.23))


def test_dop853_q1_integrate_stops_at_norm_divergence_event() -> None:
    trajectory, status = dop853_q1_integrate(
        lambda state: np.ones_like(state),
        np.array([0.0]),
        t_final=1.0,
        h=0.2,
        div_threshold=0.35,
        max_step=0.05,
    )

    assert status == "diverged"
    assert trajectory[-1, 0] == pytest.approx(0.35, abs=1.0e-9)
    assert trajectory[-1, 1] == pytest.approx(0.35, abs=1.0e-9)


def test_dop853_q1_integrate_preserves_status_contract_for_runtime_error() -> None:
    trajectory, status = dop853_q1_integrate(
        lambda _state: np.zeros(2),
        np.array([1.0]),
        t_final=1.0,
        h=0.1,
    )

    assert trajectory.shape == (1, 2)
    assert status.startswith("solver_exception:")


def test_variational_qr_is_dimension_generic_and_allows_unaligned_burn() -> None:
    matrix = np.diag([-0.25, -0.5, -1.0, -2.0])
    result = integer_dop853_variational_qr(
        lambda state: matrix @ state,
        lambda _state: matrix,
        np.ones(4),
        t_burn=0.17,
        t_accumulate=1.03,
        qr_interval=0.2,
        max_step=0.05,
        rtol=1.0e-11,
        atol=1.0e-13,
    )

    assert isinstance(result, AdaptiveLyapunovResult)
    assert result.status == "ok"
    assert result.error_message is None
    assert result.accumulated_time == pytest.approx(1.03)
    assert result.times[-1] == pytest.approx(1.03)
    assert result.convergence.shape == (6, 4)
    assert np.allclose(result.exponents, np.diag(matrix), atol=2.0e-10)
    assert result.final_state.shape == (4,)
    assert result.metadata["t_burn_completed"] == pytest.approx(0.17)
    assert result.metadata["qr_segments"] == 6
    assert result.metadata["jacobian_source"] == "analytic"


def test_variational_qr_can_use_finite_difference_jacobian() -> None:
    matrix = np.diag([-1.0, -3.0])
    result = integer_dop853_variational_qr(
        lambda state: matrix @ state,
        None,
        np.ones(2),
        t_accumulate=0.6,
        qr_interval=0.2,
        max_step=0.05,
        jacobian_eps=1.0e-5,
    )

    assert result.status == "ok"
    assert np.allclose(result.exponents, np.diag(matrix), atol=2.0e-8)
    assert result.metadata["jacobian_source"] == "central_finite_difference"


def test_variational_qr_returns_structured_invalid_jacobian_error() -> None:
    result = integer_dop853_variational_qr(
        lambda state: -state,
        lambda _state: np.eye(3),
        np.ones(2),
        t_accumulate=0.5,
    )

    assert result.status == "invalid_jacobian"
    assert result.error_message is not None
    assert "expected a finite (2, 2) matrix" in result.error_message
    assert np.all(np.isnan(result.exponents))


def test_system_wrapper_uses_declared_analytic_jacobian() -> None:
    matrix = np.diag([-0.75, -1.25])
    system = ChaoticSystem(
        name="stable-linear-adaptive-test",
        dimension=2,
        rhs=lambda state, _parameters: matrix @ state,
        jacobian=lambda _state, _parameters: matrix,
    )
    result = integer_system_dop853_variational_qr(
        system,
        np.ones(2),
        t_burn=0.13,
        t_accumulate=0.7,
        qr_interval=0.2,
        max_step=0.05,
    )

    assert result.status == "ok"
    assert np.allclose(result.exponents, np.diag(matrix), atol=2.0e-8)
    assert result.metadata["jacobian_source"] == "analytic"


def test_adaptive_q1_routines_reject_fractional_or_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="valid only for q=1"):
        integer_dop853_variational_qr(
            lambda state: -state,
            lambda _state: np.array([[-1.0]]),
            np.ones(1),
            t_accumulate=1.0,
            q=0.99,
        )

    fractional_system = ChaoticSystem(
        name="fractional-wrapper-rejection-test",
        dimension=1,
        rhs=lambda state, _parameters: -np.asarray(state),
        metadata={"q": 0.95},
    )
    with pytest.raises(ValueError, match="valid only for q=1"):
        integer_system_dop853_variational_qr(
            fractional_system,
            np.ones(1),
            t_accumulate=1.0,
        )

    integer_system = ChaoticSystem(
        name="integer-wrapper-q-override-test",
        dimension=1,
        rhs=lambda state, _parameters: -np.asarray(state),
    )
    with pytest.raises(ValueError, match="valid only for q=1"):
        integer_system_dop853_variational_qr(
            integer_system,
            np.ones(1),
            t_accumulate=1.0,
            q=0.9,
        )

    with pytest.raises(ValueError, match="qr_interval"):
        integer_dop853_variational_qr(
            lambda state: -state,
            None,
            np.ones(1),
            t_accumulate=1.0,
            qr_interval=0.0,
        )
