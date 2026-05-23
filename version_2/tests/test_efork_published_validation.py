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
