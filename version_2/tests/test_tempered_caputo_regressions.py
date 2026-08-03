from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.fractional import (
    FractionalProblem,
    integrate_tempered_caputo_abm,
    solve_fractional_problem,
)


def _zero_rhs(time: float, state: np.ndarray) -> np.ndarray:
    del time
    return np.zeros_like(state)


def _constant_rhs(
    time: float,
    state: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    del time
    return np.ones_like(state) * parameters[0]


def _tempered_problem_arguments(**updates: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "derivative": "tempered_caputo",
        "method": "tempered_caputo_abm_pece_transform",
        "orders": 0.6,
        "initial_state": [0.0],
        "step": 0.1,
        "t_span": (1.0, 1.3),
        "kernel_parameters": {"tempering": 0.2},
        "allow_experimental": True,
    }
    arguments.update(updates)
    return arguments


def test_physical_divergence_stops_before_a_later_rhs_failure() -> None:
    seen_times: list[float] = []

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        del state
        seen_times.append(float(time))
        if time > 0.025:
            raise RuntimeError("rhs-evaluated-after-physical-divergence")
        return np.array([5.0])

    result = integrate_tempered_caputo_abm(
        rhs,
        [0.0],
        0.5,
        tempering=0.4,
        lower_terminal=0.0,
        upper_terminal=0.2,
        step=0.01,
        use_acceleration=False,
        divergence_norm=0.2,
    )

    assert result.status == "diverged"
    assert len(result.times) == len(result.states) == 2
    assert result.times[-1] == pytest.approx(0.01)
    assert np.linalg.norm(result.states[-2]) <= 0.2
    assert np.linalg.norm(result.states[-1]) > 0.2
    assert max(seen_times) <= result.times[-1]


def test_initial_physical_divergence_returns_without_evaluating_rhs() -> None:
    calls = 0

    def rhs_must_not_run(time: float, state: np.ndarray) -> np.ndarray:
        nonlocal calls
        del time, state
        calls += 1
        raise AssertionError("RHS must not run for an already divergent initial state")

    result = integrate_tempered_caputo_abm(
        rhs_must_not_run,
        [2.0],
        0.5,
        tempering=0.4,
        lower_terminal=0.0,
        upper_terminal=0.2,
        step=0.01,
        use_acceleration=False,
        divergence_norm=1.0,
    )

    assert result.status == "diverged"
    assert calls == 0
    np.testing.assert_array_equal(result.times, np.array([0.0]))
    np.testing.assert_array_equal(result.states, np.array([[2.0]]))
    np.testing.assert_array_equal(result.transformed_states, np.array([[2.0]]))
    assert result.actual_upper_terminal == 0.0
    assert result.solver_info["n_steps_completed"] == 0


def test_nonfinite_solution_takes_precedence_over_divergence() -> None:
    maximum = np.finfo(np.float64).max

    def finite_but_overflowing_rhs(time: float, state: np.ndarray) -> np.ndarray:
        del time
        return np.full_like(state, maximum)

    with np.errstate(over="ignore", invalid="ignore"):
        result = integrate_tempered_caputo_abm(
            finite_but_overflowing_rhs,
            [0.0],
            0.5,
            tempering=0.0,
            lower_terminal=0.0,
            upper_terminal=0.2,
            step=0.1,
            use_acceleration=False,
            divergence_norm=1.0,
        )

    assert result.status == "nonfinite_solution"
    assert result.solver_info["underlying_status"] == "nonfinite_solution"


def test_truncated_trajectory_metadata_counts_only_returned_samples() -> None:
    result = integrate_tempered_caputo_abm(
        _constant_rhs,
        [0.0],
        0.5,
        np.array([5.0]),
        tempering=0.4,
        lower_terminal=0.0,
        upper_terminal=0.2,
        step=0.01,
        use_acceleration=False,
        divergence_norm=0.2,
    )
    info = result.solver_info

    assert result.status == "diverged"
    assert len(result.times) < result.n_steps_requested + 1
    assert info["n_steps"] == len(result.times) - 1
    assert info["n_steps_completed"] == len(result.times) - 1
    assert info["n_samples"] == len(result.times)
    assert info["n_samples_returned"] == len(result.times)


@pytest.mark.parametrize("history_window", [2.5, 3.0, np.float64(4.0)])
def test_fractional_problem_rejects_floating_history_window(
    history_window: float,
) -> None:
    arguments = _tempered_problem_arguments(
        memory_policy="finite_window",
        history_window=history_window,
    )
    with pytest.raises((TypeError, ValueError), match="history_window|integer"):
        FractionalProblem(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_opt_in", ["false", "true", 0, 1, None])
def test_fractional_problem_rejects_nonboolean_experimental_opt_in(
    invalid_opt_in: object,
) -> None:
    arguments = _tempered_problem_arguments(allow_experimental=invalid_opt_in)
    with pytest.raises(TypeError, match="allow_experimental|Boolean"):
        FractionalProblem(**arguments)  # type: ignore[arg-type]


def test_from_mapping_does_not_coerce_string_false_into_experimental_opt_in() -> None:
    mapping = _tempered_problem_arguments(allow_experimental="false")
    with pytest.raises(TypeError, match="allow_experimental|Boolean"):
        FractionalProblem.from_mapping(mapping)


def test_zero_tempering_matches_caputo_dispatcher_on_roundoff_sensitive_grid() -> None:
    common = {
        "orders": 0.6,
        "initial_state": [0.2],
        "step": 0.1,
        "t_span": (1.0, 1.3),
    }
    caputo = FractionalProblem(
        derivative="caputo",
        method="caputo_abm_pece",
        **common,
    )
    tempered = FractionalProblem(
        derivative="tempered_caputo",
        method="tempered_caputo_abm_pece_transform",
        kernel_parameters={"tempering": 0.0},
        allow_experimental=True,
        **common,
    )

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        return np.array([0.3 * time - 0.2 * state[0]])

    caputo_result = solve_fractional_problem(
        caputo,
        rhs,
        use_acceleration=False,
        divergence_norm=None,
    )
    tempered_result = solve_fractional_problem(
        tempered,
        rhs,
        use_acceleration=False,
        divergence_norm=None,
    )

    assert caputo.n_steps == tempered.n_steps == 3
    np.testing.assert_array_equal(caputo_result.times, tempered_result.times)
    np.testing.assert_array_equal(caputo_result.states, tempered_result.states)
    assert caputo_result.times[-1] == pytest.approx(1.3)
    assert len(caputo_result.times) == caputo.n_steps + 1


def test_finite_window_records_exact_sliding_restart_semantics() -> None:
    result = integrate_tempered_caputo_abm(
        _zero_rhs,
        [0.4],
        0.6,
        tempering=0.3,
        lower_terminal=0.0,
        upper_terminal=0.2,
        step=0.01,
        memory_policy="finite_window",
        history_window=5,
        use_acceleration=False,
        divergence_norm=None,
    )

    assert result.memory_policy == "finite_window"
    assert result.history_window == 5
    assert result.solver_info["window_semantics"] == (
        "sliding_restart_in_physical_state_with_tempered_anchor"
    )


def test_native_physical_kernel_avoids_transformed_norm_overflow() -> None:
    arguments = dict(
        rhs=_constant_rhs,
        initial_state=[0.1],
        order=0.5,
        parameters=np.array([1.0]),
        tempering=360.0,
        lower_terminal=0.0,
        upper_terminal=1.0,
        step=0.01,
        divergence_norm=None,
    )
    reference = integrate_tempered_caputo_abm(
        **arguments,
        use_acceleration=False,
    )
    try:
        accelerated = integrate_tempered_caputo_abm(
            **arguments,
            use_acceleration=True,
            allow_python_fallback=False,
        )
    except RuntimeError as exc:
        pytest.skip(f"Native tempered backend unavailable: {exc}")

    assert accelerated.backend == "native_c_tempered_abm_physical"
    assert accelerated.status == reference.status == "ok"
    np.testing.assert_array_equal(accelerated.times, reference.times)
    np.testing.assert_allclose(accelerated.states, reference.states, rtol=2e-13, atol=2e-14)
    assert np.all(np.isfinite(accelerated.states))
    assert accelerated.solver_info["n_samples"] == len(accelerated.times)
