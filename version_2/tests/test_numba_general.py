from __future__ import annotations

import numpy as np
import pytest
from numba import njit

from hidden_attractors.integrations.numba_general import (
    integrate_rk4_numba,
    iterate_map_numba,
)


@njit
def _linear_rhs(time, state, parameters):
    return -parameters[0] * state + 0.0 * time


@njit
def _logistic_map(iteration, state, parameters):
    return np.array(
        [parameters[0] * state[0] * (1.0 - state[0]) + 0.0 * iteration]
    )


def test_numba_rk4_matches_linear_analytic_solution() -> None:
    result = integrate_rk4_numba(
        _linear_rhs,
        [1.0],
        [2.0],
        step=0.001,
        n_steps=1000,
    )
    assert result.status == "ok"
    assert result.backend == "numba"
    assert result.states[-1, 0] == pytest.approx(np.exp(-2.0), rel=2e-10)
    assert result.times[-1] == pytest.approx(1.0)


def test_numba_map_matches_direct_logistic_iterations_after_discard() -> None:
    result = iterate_map_numba(
        _logistic_map,
        [0.2],
        [3.7],
        n_steps=8,
        discard=5,
    )
    state = 0.2
    expected = []
    for iteration in range(13):
        if iteration == 5:
            expected.append(state)
        state = 3.7 * state * (1.0 - state)
        if 5 <= iteration < 13:
            expected.append(state)
    assert result.kind == "map"
    assert np.allclose(result.states[:, 0], expected)
    assert np.array_equal(result.times, np.arange(5, 14, dtype=float))


def test_numba_backend_rejects_uncompiled_callable() -> None:
    def python_rhs(time, state, parameters):
        return state

    with pytest.raises(TypeError, match="numba.njit"):
        integrate_rk4_numba(python_rhs, [1.0], step=0.1, n_steps=2)
