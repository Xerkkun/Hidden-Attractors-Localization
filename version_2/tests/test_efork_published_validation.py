"""Reproduction tests for the published three-stage EFORK examples."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hidden_attractors.solvers import efork3_caputo_integrate, efork_q1_step


def _mittag_leffler(alpha: float, beta: float, z: float) -> float:
    total = 0.0
    for index in range(300):
        term = z**index / math.gamma(alpha * index + beta)
        total += term
        if abs(term) < 1.0e-16:
            break
    return total


def _example_1_error(alpha: float, n_steps: int) -> float:
    rhs = lambda t, y: -y + t ** (4.0 - alpha) / math.gamma(5.0 - alpha)
    times, states = efork3_caputo_integrate(
        rhs,
        np.array([0.0]),
        alpha=alpha,
        h=1.0 / n_steps,
        t_final=1.0,
    )
    exact = times[-1] ** 4 * _mittag_leffler(alpha, 5.0, -(times[-1] ** alpha))
    return abs(float(states[-1, 0]) - exact)


def _example_2_error(alpha: float, n_steps: int) -> float:
    rhs = lambda t, y: (
        2.0 * t ** (2.0 - alpha) / math.gamma(3.0 - alpha)
        - t ** (1.0 - alpha) / math.gamma(2.0 - alpha)
        - y
        + t**2
        - t
    )
    _, states = efork3_caputo_integrate(
        rhs,
        np.array([0.0]),
        alpha=alpha,
        h=1.0 / n_steps,
        t_final=1.0,
    )
    return abs(float(states[-1, 0]))


@pytest.mark.parametrize(
    ("calculator", "alpha", "published"),
    [
        (_example_1_error, 0.25, [9.94252e-4, 5.54011e-4, 3.13499e-4, 1.79258e-4, 1.03255e-4]),
        (_example_1_error, 0.50, [7.45694e-5, 2.46986e-5, 8.26771e-6, 2.79911e-6, 9.57367e-7]),
        (_example_2_error, 0.25, [9.90939e-3, 5.54249e-3, 3.14342e-3, 1.79955e-3, 1.03718e-3]),
        (_example_2_error, 0.50, [5.79341e-4, 1.96590e-4, 6.68302e-5, 2.28624e-5, 7.87606e-6]),
    ],
)
def test_efork3_reproduces_published_terminal_errors(calculator, alpha: float, published: list[float]) -> None:
    computed = [calculator(alpha, n_steps) for n_steps in (40, 80, 160, 320, 640)]
    assert np.allclose(computed, published, atol=6.0e-9, rtol=0.0)


def test_q1_step_uses_published_k3_stage_order() -> None:
    rhs = lambda y: np.array([y[0] ** 2 + 0.25])
    state = np.array([0.3])
    h = 0.1
    k1 = h * rhs(state)
    k2 = h * rhs(state + 0.5 * k1)
    k3 = h * rhs(state + 0.5 * k1 - 0.25 * k2)
    expected = state + (2.0 / 3.0) * k1 + (5.0 / 3.0) * k2 - (4.0 / 3.0) * k3
    assert np.allclose(efork_q1_step(rhs, state, h), expected, atol=1.0e-15, rtol=0.0)


def test_native_fractional_backend_matches_reference_k3_history_stages() -> None:
    from hidden_attractors.models import chua_nonsmooth_parameters, rhs_nonsmooth
    from hidden_attractors.native.backends import FractionalChuaBackend

    try:
        backend = FractionalChuaBackend.build(output_name="chua_frac_backend_efork_stage_test")
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"Native compiler unavailable: {exc}")

    params = chua_nonsmooth_parameters()
    state0 = np.array([0.31, -0.08, 0.12])
    q = 0.8
    h = 0.02
    t_final = 0.20
    native = backend.integrate_efork3(state0, q=q, h=h, Lm=t_final, t_final=t_final)
    times, states = efork3_caputo_integrate(
        lambda _time, state: rhs_nonsmooth(state, params),
        state0,
        alpha=q,
        h=h,
        t_final=t_final,
    )
    reference = np.column_stack((times, states))
    assert np.allclose(native, reference, atol=2.0e-12, rtol=0.0)


def test_native_continuation_carries_memory_with_corrected_backend() -> None:
    from hidden_attractors.native.backends import FractionalChuaBackend

    try:
        backend = FractionalChuaBackend.build(output_name="chua_frac_backend_continuation_test")
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"Native compiler unavailable: {exc}")
    result = backend.continue_efork3(
        [0.31, -0.08, 0.12],
        eps_values=[0.0, 0.5, 1.0],
        q=0.8,
        k=0.2,
        h=0.02,
        Lm=0.20,
        t_transient=0.10,
        t_keep=0.10,
        t_observe=0.10,
    )
    assert result["trajectories"].shape == (3, 6, 4)
    assert result["observation"].shape == (6, 4)
    assert result["history_in_counts"][0] == 0
    assert result["history_in_counts"][1] > 0
    assert np.all(np.isfinite(result["x_out"]))
    assert np.all(np.isfinite(result["observation"]))


def test_native_abm_full_history_matches_diethelm_reference() -> None:
    from hidden_attractors.models import chua_nonsmooth_parameters, rhs_nonsmooth
    from hidden_attractors.native.backends import FullHistoryABMBackend

    try:
        backend = FullHistoryABMBackend.build(output_name="chua_abm_full_history_test")
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"Native compiler unavailable: {exc}")

    params = chua_nonsmooth_parameters()
    y0 = np.array([0.31, -0.08, 0.12], dtype=float)
    q = 0.9998
    h = 0.01
    t_final = 0.10
    nsteps = int(math.ceil(t_final / h))
    states = np.zeros((nsteps + 1, 3), dtype=float)
    f_hist = np.zeros_like(states)
    states[0] = y0
    f_hist[0] = rhs_nonsmooth(y0, params)
    powers = np.arange(nsteps + 2, dtype=float)
    pow_q = powers**q
    pow_q1 = powers ** (q + 1.0)
    pred_scale = h**q / math.gamma(q + 1.0)
    corr_scale = h**q / math.gamma(q + 2.0)
    for i in range(nsteps):
        weights = pow_q[1 : i + 2][::-1] - pow_q[0 : i + 1][::-1]
        predictor = y0 + pred_scale * (weights @ f_hist[: i + 1])
        fp = rhs_nonsmooth(predictor, params)
        if i == 0:
            a_weights = np.array([q], dtype=float)
        else:
            r = np.arange(i, 0, -1, dtype=int)
            middle = pow_q1[r + 1] + pow_q1[r - 1] - 2.0 * pow_q1[r]
            a0 = pow_q1[i] - (float(i) - q) * pow_q[i + 1]
            a_weights = np.concatenate(([a0], middle))
        states[i + 1] = y0 + corr_scale * ((a_weights @ f_hist[: i + 1]) + fp)
        f_hist[i + 1] = rhs_nonsmooth(states[i + 1], params)

    native = backend.integrate(y0, q=q, h=h, t_final=t_final)
    assert np.allclose(native[:, 1:4], states, atol=2.0e-12, rtol=0.0)


def test_native_abm_truncated_reduces_to_full_history_when_window_covers_horizon() -> None:
    from hidden_attractors.native.backends import FullHistoryABMBackend

    try:
        backend = FullHistoryABMBackend.build(output_name="chua_abm_truncated_contract_test")
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"Native compiler unavailable: {exc}")

    y0 = np.array([0.31, -0.08, 0.12], dtype=float)
    full = backend.integrate(y0, q=0.9998, h=0.01, t_final=0.20)
    no_cut = backend.integrate_truncated(y0, q=0.9998, h=0.01, Lm=0.20, t_final=0.20)
    cut = backend.integrate_truncated(y0, q=0.9998, h=0.01, Lm=0.05, t_final=0.20)

    assert np.allclose(no_cut, full, atol=2.0e-12, rtol=0.0)
    assert np.all(np.isfinite(cut))
    assert not np.allclose(cut[:, 1:4], full[:, 1:4], atol=2.0e-12, rtol=0.0)


def test_hiddenness_c_backends_use_published_k3_stage_order() -> None:
    from pathlib import Path

    root = Path(__file__).parents[1]
    native_fractional = (root / "hidden_attractors" / "native" / "csrc" / "chua_frac_backend_lib.c").read_text(encoding="utf-8")
    assert "c.a31 * k1x + c.a32 * k2x" in native_fractional
    assert "c.a31 * k2x + c.a32 * k1x" not in native_fractional
    for source in (
        root / "hidden_attractors" / "native" / "csrc" / "chua_hidden_backend.c",
        root / "tools" / "legacy" / "chua_hidden_backend.c",
    ):
        text = source.read_text(encoding="utf-8")
        assert "coef.a31*K1x+coef.a32*K2x" in text
        assert "coef.a31*K2x+coef.a32*K1x" not in text
        assert "exchange its stored coefficients" not in text
