from __future__ import annotations

from math import gamma

import numpy as np
import pytest

from hidden_attractors.fractional import (
    CAPUTO_HADAMARD_INITIAL_CONDITION,
    FractionalProblem,
    integrate_caputo_hadamard_abm,
    solve_fractional_problem,
)
from hidden_attractors.integrations.abm import caputo_abm_integrate
from hidden_attractors.integrations.fractional_c import GeneralFractionalCBackend


def _constant_rhs(time: float, state: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    del time
    return np.ones_like(state) * parameters[0]


def test_constant_forcing_matches_caputo_hadamard_closed_form() -> None:
    order = 0.63
    lower = 2.0
    result = integrate_caputo_hadamard_abm(
        _constant_rhs,
        [1.25, -0.5],
        order,
        np.array([0.8]),
        lower_terminal=lower,
        upper_terminal=lower * np.e,
        log_step=0.005,
        use_acceleration=False,
        divergence_norm=None,
    )
    expected = np.array([1.25, -0.5]) + 0.8 / gamma(order + 1.0)
    np.testing.assert_allclose(result.states[-1], expected, rtol=2.0e-12, atol=2.0e-12)
    assert result.status == "ok"
    assert result.backend == "python_numpy_abm"
    assert result.log_times[-1] == pytest.approx(1.0)
    assert result.times[-1] == pytest.approx(lower * np.e)


def test_manufactured_log_power_uses_physical_rhs_time() -> None:
    order = 0.58
    power = 2.0
    lower = 1.5
    coefficient = gamma(power + 1.0) / gamma(power + 1.0 - order)
    seen_times: list[float] = []

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        seen_times.append(float(time))
        u = np.log(time / lower)
        return np.full_like(state, coefficient * u ** (power - order))

    result = integrate_caputo_hadamard_abm(
        rhs,
        [3.0],
        order,
        lower_terminal=lower,
        upper_terminal=lower * np.e,
        log_step=0.0025,
        use_acceleration=False,
        divergence_norm=None,
    )
    assert result.states[-1, 0] == pytest.approx(4.0, rel=4.0e-5)
    assert min(seen_times) == pytest.approx(lower)
    assert max(seen_times) == pytest.approx(lower * np.e)
    assert result.rhs_time_coordinate == "physical_time_t_equals_a_exp_u"
    assert not result.physical_sampling_uniform


def test_wrapper_matches_canonical_caputo_abm_after_log_transform() -> None:
    lower = 0.8
    order = 0.71
    step = 0.01

    def physical_rhs(time: float, state: np.ndarray) -> np.ndarray:
        return np.array([0.2 * time - 0.1 * state[0]])

    wrapped = integrate_caputo_hadamard_abm(
        physical_rhs,
        [0.4],
        order,
        lower_terminal=lower,
        upper_terminal=lower * np.e,
        log_step=step,
        use_acceleration=False,
        divergence_norm=None,
    )

    def transformed_rhs(log_time: float, state: np.ndarray) -> np.ndarray:
        return physical_rhs(lower * np.exp(log_time), state)

    direct_times, direct_states, direct_status = caputo_abm_integrate(
        transformed_rhs,
        np.array([0.4]),
        order,
        step,
        float(np.nextafter(1.0, -np.inf)),
        divergence_norm=None,
        memory_mode="full",
        use_c_backend=False,
        early_stop_config={"enabled": False},
    )
    assert wrapped.status == direct_status
    np.testing.assert_array_equal(wrapped.log_times, direct_times)
    np.testing.assert_allclose(
        wrapped.states, direct_states, rtol=3.0e-15, atol=3.0e-15
    )


def test_accelerated_dispatch_agrees_with_reference_path() -> None:
    arguments = dict(
        rhs=_constant_rhs,
        initial_state=[0.3, -0.2],
        order=0.67,
        parameters=np.array([0.15]),
        lower_terminal=1.0,
        upper_terminal=float(np.exp(0.4)),
        log_step=0.01,
        divergence_norm=None,
    )
    try:
        GeneralFractionalCBackend.get_instance()
        native_available = True
    except Exception:
        native_available = False

    reference = integrate_caputo_hadamard_abm(**arguments, use_acceleration=False)
    accelerated = integrate_caputo_hadamard_abm(**arguments, use_acceleration=True)
    np.testing.assert_allclose(
        accelerated.states, reference.states, rtol=5.0e-12, atol=5.0e-12
    )
    np.testing.assert_allclose(accelerated.times, reference.times, rtol=0.0, atol=0.0)
    if native_available:
        assert accelerated.backend == "c_abm_with_python_time_transform"
        assert accelerated.solver_info["used_c_backend"] is True
        assert "c_backend_error" not in accelerated.solver_info
    else:
        assert accelerated.backend == "python_numpy_abm"
        assert accelerated.solver_info["used_c_backend"] is False
        assert "c_backend_error" in accelerated.solver_info


def test_two_argument_rhs_keeps_physical_time_when_parameters_are_present() -> None:
    seen_times: list[object] = []

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        seen_times.append(time)
        return np.ones_like(state)

    result = integrate_caputo_hadamard_abm(
        rhs,
        [0.0],
        0.5,
        parameters={"unused": True},
        lower_terminal=1.0,
        upper_terminal=float(np.exp(0.1)),
        log_step=0.01,
        use_acceleration=False,
        divergence_norm=None,
    )
    assert result.status == "ok"
    assert seen_times
    assert all(isinstance(value, float) for value in seen_times)
    assert seen_times[0] == pytest.approx(1.0)


def test_internal_rhs_typeerror_is_not_reinterpreted_as_another_signature() -> None:
    calls = 0

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        nonlocal calls
        del time, state
        calls += 1
        raise TypeError("intentional-rhs-typeerror")

    with pytest.raises(TypeError, match="intentional-rhs-typeerror"):
        integrate_caputo_hadamard_abm(
            rhs,
            [0.0],
            0.5,
            lower_terminal=1.0,
            upper_terminal=float(np.exp(0.1)),
            log_step=0.01,
            use_acceleration=False,
        )
    assert calls == 1


def test_c_callback_propagates_late_rhs_shape_failure() -> None:
    probe = integrate_caputo_hadamard_abm(
        lambda time, state: np.ones_like(state),
        [0.0],
        0.5,
        lower_terminal=1.0,
        upper_terminal=float(np.exp(0.03)),
        log_step=0.01,
        use_acceleration=True,
        divergence_norm=None,
    )
    if probe.backend != "c_abm_with_python_time_transform":
        pytest.skip("The native C callback path is unavailable on this host.")

    def changing_shape_rhs(time: float, state: np.ndarray) -> np.ndarray:
        if time > np.exp(0.015):
            return np.empty(0)
        return np.ones_like(state)

    with pytest.raises(ValueError, match="shape"):
        integrate_caputo_hadamard_abm(
            changing_shape_rhs,
            [0.0],
            0.5,
            lower_terminal=1.0,
            upper_terminal=float(np.exp(0.05)),
            log_step=0.01,
            use_acceleration=True,
            divergence_norm=None,
        )


def test_c_callback_propagates_late_internal_typeerror() -> None:
    probe = integrate_caputo_hadamard_abm(
        lambda time, state: np.ones_like(state),
        [0.0],
        0.5,
        lower_terminal=1.0,
        upper_terminal=float(np.exp(0.03)),
        log_step=0.01,
        use_acceleration=True,
        divergence_norm=None,
    )
    if probe.backend != "c_abm_with_python_time_transform":
        pytest.skip("The native C callback path is unavailable on this host.")

    def failing_rhs(time: float, state: np.ndarray) -> np.ndarray:
        if time > np.exp(0.015):
            raise TypeError("late-callback-typeerror")
        return np.ones_like(state)

    with pytest.raises(TypeError, match="late-callback-typeerror"):
        integrate_caputo_hadamard_abm(
            failing_rhs,
            [0.0],
            0.5,
            lower_terminal=1.0,
            upper_terminal=float(np.exp(0.05)),
            log_step=0.01,
            use_acceleration=True,
            divergence_norm=None,
        )


def test_structured_result_records_grid_memory_and_evidence_scope() -> None:
    result = integrate_caputo_hadamard_abm(
        _constant_rhs,
        [0.0],
        0.5,
        np.array([1.0]),
        lower_terminal=3.0,
        upper_terminal=3.0 * np.exp(0.2),
        log_step=0.01,
        use_acceleration=False,
    )
    assert result.method == "caputo_hadamard_abm_pece"
    assert result.grid_coordinate == "uniform_log_t_over_lower_terminal"
    assert result.memory_policy == "full_history"
    assert result.initial_condition_semantics == CAPUTO_HADAMARD_INITIAL_CONDITION
    assert result.n_steps_requested == 20
    assert result.scope == "finite_numerical_trajectory_only"
    assert result.trajectory.shape == (21, 2)
    assert result.solver_info["n_steps"] == 20
    assert result.solver_info["n_steps_completed"] == 20
    assert result.solver_info["n_samples"] == 21
    assert result.solver_info["n_samples_returned"] == 21
    assert "https://doi.org/10.1016/j.aml.2021.107366" in result.references
    assert "https://doi.org/10.3390/math9212728" in result.references
    assert (
        "https://doi.org/10.1023/B:NUMA.0000027736.85078.be"
        in result.references
    )


def test_adjacent_large_terminals_preserve_log_grid_and_exact_initial_time() -> None:
    lower = 1.0e300
    upper = float(np.nextafter(lower, np.inf))
    log_step = float(np.log1p((upper - lower) / lower))
    seen_times: list[float] = []

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        seen_times.append(time)
        return np.ones_like(state)

    result = integrate_caputo_hadamard_abm(
        rhs,
        [0.0],
        0.5,
        lower_terminal=lower,
        upper_terminal=upper,
        log_step=log_step,
        use_acceleration=False,
        divergence_norm=None,
    )
    assert result.status == "ok"
    assert result.n_steps_requested == 1
    assert seen_times[0] == lower
    np.testing.assert_array_equal(result.times, np.asarray([lower, upper]))

    problem = FractionalProblem(
        "caputo_hadamard",
        "caputo_hadamard_abm_pece",
        0.5,
        [0.0],
        log_step,
        (lower, upper),
        grid_coordinate="log_t_over_lower_terminal",
        allow_experimental=True,
    )
    assert problem.n_steps == 1
    assert problem.coordinate_duration == log_step

    with pytest.raises(ValueError, match="integer"):
        integrate_caputo_hadamard_abm(
            rhs,
            [0.0],
            0.5,
            lower_terminal=lower,
            upper_terminal=upper,
            log_step=1.25 * log_step,
            use_acceleration=False,
        )


def test_wide_terminal_ratio_maps_to_finite_increasing_physical_times() -> None:
    lower = 1.0e-300
    upper = 1.0e300
    log_duration = float(np.log(upper) - np.log(lower))
    result = integrate_caputo_hadamard_abm(
        lambda time, state: np.ones_like(state),
        [0.0],
        0.5,
        lower_terminal=lower,
        upper_terminal=upper,
        log_step=log_duration / 4.0,
        use_acceleration=False,
        divergence_norm=None,
    )
    assert result.status == "ok"
    assert np.all(np.isfinite(result.times))
    assert np.all(np.diff(result.times) > 0.0)
    assert result.times[0] == lower
    assert result.times[-1] == pytest.approx(upper, rel=3.0e-13)


@pytest.mark.parametrize(
    ("updates", "error", "match"),
    [
        ({"order": 0.0}, ValueError, "strictly"),
        ({"order": 1.0}, ValueError, "strictly"),
        ({"order": True}, TypeError, "order"),
        ({"lower_terminal": 0.0}, ValueError, "strictly positive"),
        ({"upper_terminal": 1.0}, ValueError, "greater"),
        ({"log_step": 0.0}, ValueError, "positive"),
        ({"log_step": True}, TypeError, "log_step"),
        ({"initial_condition_semantics": "classical"}, ValueError, "requires"),
        ({"use_acceleration": 1}, TypeError, "Boolean"),
    ],
)
def test_invalid_solver_contract_is_rejected(
    updates: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "rhs": _constant_rhs,
        "initial_state": [0.0],
        "order": 0.5,
        "parameters": np.array([1.0]),
        "lower_terminal": 1.0,
        "upper_terminal": float(np.e),
        "log_step": 0.1,
        "use_acceleration": False,
    }
    arguments.update(updates)
    with pytest.raises(error, match=match):
        integrate_caputo_hadamard_abm(**arguments)  # type: ignore[arg-type]


def test_nonintegral_log_grid_and_bad_rhs_are_rejected() -> None:
    with pytest.raises(ValueError, match="integer"):
        integrate_caputo_hadamard_abm(
            _constant_rhs,
            [0.0],
            0.5,
            np.array([1.0]),
            lower_terminal=1.0,
            upper_terminal=2.0,
            log_step=0.1,
            use_acceleration=False,
        )

    def wrong_shape(time: float, state: np.ndarray) -> np.ndarray:
        del time, state
        return np.ones(2)

    with pytest.raises(ValueError, match="shape"):
        integrate_caputo_hadamard_abm(
            wrong_shape,
            [0.0],
            0.5,
            lower_terminal=1.0,
            upper_terminal=float(np.e),
            log_step=0.1,
            use_acceleration=False,
        )


def test_fractional_problem_requires_explicit_log_coordinate_and_dispatches() -> None:
    with pytest.raises(ValueError, match="grid_coordinate"):
        FractionalProblem(
            "caputo_hadamard",
            "caputo_hadamard_abm_pece",
            0.6,
            [0.0],
            0.01,
            (1.0, float(np.e)),
            allow_experimental=True,
        )

    problem = FractionalProblem(
        "caputo_hadamard",
        "caputo_hadamard_abm_pece",
        0.6,
        [0.0],
        0.01,
        (1.0, float(np.e)),
        grid_coordinate="log_t_over_lower_terminal",
        allow_experimental=True,
    )
    assert problem.n_steps == 100
    assert problem.duration == pytest.approx(np.e - 1.0)
    assert problem.coordinate_duration == pytest.approx(1.0)
    assert FractionalProblem.from_mapping(problem.to_mapping()) == problem
    result = solve_fractional_problem(
        problem,
        _constant_rhs,
        np.array([1.0]),
        use_acceleration=False,
        divergence_norm=None,
    )
    assert result.status == "ok"
    assert result.states[-1, 0] == pytest.approx(1.0 / gamma(1.6), rel=2.0e-12)
    np.testing.assert_allclose(result.coordinate_times, np.arange(101) * 0.01)
    assert result.metadata["grid_coordinate"] == "log_t_over_lower_terminal"


def test_caputo_hadamard_problem_rejects_near_but_distinct_component_orders() -> None:
    with pytest.raises(ValueError, match="componentwise"):
        FractionalProblem(
            "caputo_hadamard",
            "caputo_hadamard_abm_pece",
            [0.6, np.nextafter(0.6, np.inf)],
            [0.0, 0.0],
            0.01,
            (1.0, float(np.e)),
            grid_coordinate="log_t_over_lower_terminal",
            allow_experimental=True,
        )
