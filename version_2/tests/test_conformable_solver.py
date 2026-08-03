from __future__ import annotations

import numpy as np
import pytest
from numba import njit

from hidden_attractors.fractional import (
    FractionalProblem,
    conformable_clock_from_time,
    integrate_conformable_rk4,
    physical_times_from_conformable_clock,
    solve_fractional_problem,
)


@njit
def _constant_numba_rhs(time, state, parameters):
    return np.ones_like(state) * parameters[0] + 0.0 * time


def _constant_python_rhs(time, state, parameters):
    return np.ones_like(state) * parameters["forcing"] + 0.0 * time


def test_conformable_clock_round_trip_is_explicit() -> None:
    physical = np.array([2.0, 2.25, 3.0, 6.0])
    clock = conformable_clock_from_time(physical, 0.5, 2.0)
    recovered = physical_times_from_conformable_clock(clock, 0.5, 2.0)
    np.testing.assert_allclose(recovered, physical, rtol=0.0, atol=2.0e-15)
    assert clock[0] == 0.0


def test_conformable_rk4_matches_manufactured_constant_clock_flow() -> None:
    common = dict(
        initial_state=[3.0],
        order=0.5,
        lower_terminal=2.0,
        upper_terminal=3.0,
        clock_step=0.01,
        divergence_norm=None,
    )
    accelerated = integrate_conformable_rk4(
        _constant_numba_rhs,
        parameters=[1.0],
        **common,
    )
    reference = integrate_conformable_rk4(
        _constant_python_rhs,
        parameters={"forcing": 1.0},
        use_acceleration=False,
        **common,
    )
    assert accelerated.status == reference.status == "ok"
    assert accelerated.backend == "numba_rk4_conformable_clock"
    assert reference.backend == "python_rk4_conformable_clock"
    np.testing.assert_allclose(accelerated.clock_times, np.arange(201) * 0.01)
    np.testing.assert_allclose(accelerated.states, reference.states, atol=2.0e-14)
    assert accelerated.states[-1, 0] == pytest.approx(5.0, abs=1.0e-13)
    assert accelerated.times[-1] == pytest.approx(3.0)
    assert not np.allclose(np.diff(accelerated.times), np.diff(accelerated.times)[0])
    assert accelerated.memory_policy == "none_local_operator"


def test_fractional_problem_dispatches_conformable_clock() -> None:
    with pytest.raises(ValueError, match="grid_coordinate"):
        FractionalProblem(
            "conformable",
            "conformable_rk4_clock",
            0.5,
            [3.0],
            0.01,
            (2.0, 3.0),
            memory_policy="none",
            allow_experimental=True,
        )
    with pytest.raises(ValueError, match="single conformable clock"):
        FractionalProblem(
            "conformable",
            "conformable_rk4_clock",
            [0.5, 0.75],
            [0.0, 0.0],
            0.01,
            (2.0, 3.0),
            grid_coordinate="conformable_clock",
            memory_policy="none",
            allow_experimental=True,
        )

    problem = FractionalProblem(
        "conformable",
        "conformable_rk4_clock",
        0.5,
        [3.0],
        0.01,
        (2.0, 3.0),
        grid_coordinate="conformable_clock",
        memory_policy="none",
        allow_experimental=True,
    )
    result = solve_fractional_problem(
        problem,
        _constant_numba_rhs,
        [1.0],
        divergence_norm=None,
    )
    assert problem.n_steps == 200
    assert result.status == "ok"
    assert result.backend == "numba_rk4_conformable_clock"
    assert result.states[-1, 0] == pytest.approx(5.0, abs=1.0e-13)
    np.testing.assert_allclose(result.coordinate_times, np.arange(201) * 0.01)
    assert result.metadata["memory_policy"] == "none"
    assert result.metadata["backend_info"]["memory_semantics"] == (
        "none_local_time_reparametrization"
    )


def test_conformable_solver_can_require_acceleration() -> None:
    with pytest.raises(RuntimeError, match="no Numba conformable backend"):
        integrate_conformable_rk4(
            _constant_python_rhs,
            [0.0],
            0.5,
            {"forcing": 1.0},
            lower_terminal=0.0,
            upper_terminal=1.0,
            clock_step=0.1,
            allow_python_fallback=False,
        )
