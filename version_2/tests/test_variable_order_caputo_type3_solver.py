from __future__ import annotations

from math import gamma
import warnings

import numpy as np
import pytest

import hidden_attractors.fractional.variable_order_caputo_type3 as vo_type3_module
from hidden_attractors.fractional import (
    FractionalProblem,
    VariableOrderInitialCompatibilityWarning,
    integrate_variable_order_caputo_type3_l1,
    solve_fractional_problem,
    variable_order_l1_weight,
)


def _zero_rhs(time: float, state: np.ndarray) -> np.ndarray:
    del time
    return np.zeros_like(state)


def _constant_order(
    time: float,
    state: np.ndarray,
    parameters: np.ndarray,
) -> float:
    del time, state
    return float(parameters[0])


def _constant_six_order(time: float, state: np.ndarray) -> float:
    del time, state
    return 0.6


def _affine_order(
    time: float,
    state: np.ndarray,
    parameters: np.ndarray,
) -> float:
    del state
    return float(parameters[0] + parameters[1] * (time - parameters[2]))


def _linear_rhs(
    time: float,
    state: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    del time
    return parameters[3] - parameters[4] * state


def _constant_alpha_linear_l1_reference(
    *,
    initial_value: float,
    order: float,
    step: float,
    n_steps: int,
    forcing: float,
    damping: float,
) -> np.ndarray:
    """Independent scalar L1 recurrence for f=c-mu*u."""

    states = np.empty(n_steps + 1, dtype=np.float64)
    states[0] = initial_value
    scale = gamma(2.0 - order) * step**order
    for n in range(1, n_steps + 1):
        memory = 0.0
        for k in range(n - 1):
            lag = n - k - 1
            weight = (lag + 1.0) ** (1.0 - order) - lag ** (1.0 - order)
            memory += weight * (states[k + 1] - states[k])
        base = states[n - 1] - memory
        states[n] = (base + scale * forcing) / (1.0 + scale * damping)
    return states


def _solver_arguments(**updates: object) -> dict[str, object]:
    parameters = np.array([0.6, 0.0, 0.0, 0.4, 0.3])
    arguments: dict[str, object] = {
        "rhs": _linear_rhs,
        "initial_state": [0.2],
        "order_function": _constant_order,
        "parameters": parameters,
        "step": 0.02,
        "n_steps": 10,
        "lower_terminal": 0.0,
        "order_function_name": "constant-alpha-0.6",
        "corrector_atol": 1.0e-12,
        "corrector_rtol": 1.0e-10,
        "corrector_max_iterations": 100,
        "on_nonconvergence": "raise",
        "initial_regularity": "nonsmooth",
        "compatibility_tolerance": 1.0e-12,
        "use_acceleration": False,
        "divergence_norm": None,
    }
    arguments.update(updates)
    return arguments


def _problem_arguments(**updates: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "derivative": "caputo_variable_type3",
        "method": "vo_caputo_type3_l1",
        "orders": 0.6,
        "initial_state": [0.2],
        "step": 0.02,
        "t_span": (0.0, 0.2),
        "kernel_parameters": {
            "order_function": _constant_six_order,
            "order_function_name": "constant-alpha-0.6",
        },
        "method_options": {
            "corrector_atol": 1.0e-12,
            "corrector_rtol": 1.0e-10,
            "corrector_max_iterations": 100,
            "on_nonconvergence": "raise",
            "initial_regularity": "nonsmooth",
            "compatibility_tolerance": 1.0e-12,
        },
        "allow_experimental": True,
    }
    arguments.update(updates)
    return arguments


@pytest.mark.parametrize("order", [0.05, 0.37, 0.63, 0.95])
def test_l1_weights_match_direct_formula_at_moderate_lags(order: float) -> None:
    for lag in range(65):
        actual = variable_order_l1_weight(order, lag)
        expected = (lag + 1.0) ** (1.0 - order) - lag ** (1.0 - order)
        assert actual == pytest.approx(expected, rel=2.0e-14, abs=0.0)


def test_l1_weight_avoids_large_lag_cancellation() -> None:
    order = 0.73
    lag = 10**12
    exponent = 1.0 - order
    stable_reference = lag**exponent * np.expm1(exponent * np.log1p(1.0 / lag))
    actual = variable_order_l1_weight(order, lag)

    assert actual > 0.0
    assert actual == pytest.approx(stable_reference, rel=3.0e-15, abs=0.0)


def test_l1_weight_remains_positive_next_to_order_one() -> None:
    order = np.nextafter(1.0, 0.0)
    exponent = 1.0 - order
    stable_reference = np.expm1(exponent * np.log(2.0))

    actual = variable_order_l1_weight(order, 1)

    assert actual > 0.0
    assert actual == pytest.approx(stable_reference, rel=3.0e-15, abs=0.0)


def test_variable_order_quadratic_manufactured_solution() -> None:
    lower = 0.4
    initial = 1.25
    power = 2.0
    parameters = np.array([0.52, 0.2, lower])

    def order_function(
        time: float,
        state: np.ndarray,
        values: np.ndarray,
    ) -> float:
        del state
        return float(values[0] + values[1] * (time - values[2]))

    def manufactured_rhs(
        time: float,
        state: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        del state
        tau = time - values[2]
        alpha = values[0] + values[1] * tau
        coefficient = gamma(power + 1.0) / gamma(power + 1.0 - alpha)
        return np.array([coefficient * tau ** (power - alpha)])

    result = integrate_variable_order_caputo_type3_l1(
        rhs=manufactured_rhs,
        initial_state=[initial],
        order_function=order_function,
        parameters=parameters,
        step=0.005,
        n_steps=100,
        lower_terminal=lower,
        order_function_name="alpha(t)=0.52+0.2(t-a)",
        initial_regularity="smooth",
        use_acceleration=False,
        divergence_norm=None,
    )
    exact = initial + (result.times - lower) ** power

    assert result.status == "ok"
    np.testing.assert_allclose(result.states[:, 0], exact, rtol=0.0, atol=1.5e-3)
    np.testing.assert_allclose(
        result.orders,
        parameters[0] + parameters[1] * (result.times - lower),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_constant_order_reduction_matches_independent_implicit_l1_recurrence() -> None:
    order = 0.63
    step = 0.02
    n_steps = 20
    initial = 0.2
    forcing = 0.4
    damping = 0.3
    parameters = np.array([order, 0.0, 0.0, forcing, damping])
    result = integrate_variable_order_caputo_type3_l1(
        **_solver_arguments(
            initial_state=[initial],
            parameters=parameters,
            step=step,
            n_steps=n_steps,
        )
    )
    expected = _constant_alpha_linear_l1_reference(
        initial_value=initial,
        order=order,
        step=step,
        n_steps=n_steps,
        forcing=forcing,
        damping=damping,
    )

    assert result.status == "ok"
    np.testing.assert_allclose(result.states[:, 0], expected, rtol=2.0e-11, atol=2.0e-13)
    np.testing.assert_array_equal(result.orders, np.full(n_steps + 1, order))


def test_numba_history_backend_matches_python_without_fastmath() -> None:
    parameters = np.array([0.55, 0.25, 0.0, 0.35, 0.2])
    arguments = _solver_arguments(
        order_function=_affine_order,
        order_function_name="affine-order",
        parameters=parameters,
        step=0.01,
        n_steps=30,
    )
    python_arguments = dict(arguments)
    python_arguments["use_acceleration"] = False
    numba_arguments = dict(arguments)
    numba_arguments["use_acceleration"] = True
    python_result = integrate_variable_order_caputo_type3_l1(**python_arguments)
    numba_result = integrate_variable_order_caputo_type3_l1(**numba_arguments)

    assert "python" in python_result.backend
    assert "numba" in numba_result.backend
    assert python_result.status == numba_result.status == "ok"
    np.testing.assert_array_equal(numba_result.times, python_result.times)
    np.testing.assert_array_equal(numba_result.orders, python_result.orders)
    np.testing.assert_allclose(
        numba_result.states,
        python_result.states,
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    assert python_result.solver_info["numba_history_requested"] is False
    assert python_result.solver_info["numba_history_attempted"] is False
    assert python_result.solver_info["used_numba_history"] is False
    assert python_result.solver_info["numba_fallback_used"] is False
    assert python_result.solver_info["history_backend"] == "python"
    assert numba_result.solver_info["numba_history_requested"] is True
    assert numba_result.solver_info["numba_history_attempted"] is True
    assert numba_result.solver_info["used_numba_history"] is True
    assert numba_result.solver_info["numba_history_steps"] == 30
    assert numba_result.solver_info["numba_fallback_used"] is False
    assert numba_result.solver_info["history_backend"] == "numba"


def test_partial_numba_fallback_records_requested_attempted_actual_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def flaky_history(
        states: np.ndarray,
        output_index: int,
        order: float,
    ) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced-variable-order-history-fallback")
        return vo_type3_module._history_sum_python(states, output_index, order)

    monkeypatch.setattr(vo_type3_module, "_history_sum_numba", flaky_history)
    result = integrate_variable_order_caputo_type3_l1(
        **_solver_arguments(n_steps=4, use_acceleration=True)
    )

    assert result.status == "ok"
    assert result.backend == "numba_then_python_history_python_picard"
    assert result.solver_info["numba_history_requested"] is True
    assert result.solver_info["numba_history_attempted"] is True
    assert result.solver_info["used_numba_history"] is True
    assert result.solver_info["numba_history_steps"] == 1
    assert result.solver_info["numba_fallback_used"] is True
    assert "forced-variable-order-history-fallback" in str(
        result.solver_info["numba_fallback_error"]
    )
    assert result.solver_info["history_evaluations"] == 4
    assert result.solver_info["history_backend"] == "numba_then_python"


def test_initial_divergence_stops_before_rhs_or_history_evaluation() -> None:
    calls = 0

    def forbidden_rhs(time: float, state: np.ndarray) -> np.ndarray:
        nonlocal calls
        del time, state
        calls += 1
        raise AssertionError("RHS must not run after initial divergence")

    result = integrate_variable_order_caputo_type3_l1(
        rhs=forbidden_rhs,
        initial_state=[2.0],
        order_function=_constant_six_order,
        step=0.1,
        n_steps=3,
        divergence_norm=1.0,
        use_acceleration=True,
    )

    assert result.status == "diverged"
    assert calls == 0
    np.testing.assert_array_equal(result.times, np.array([0.0]))
    np.testing.assert_array_equal(result.states, np.array([[2.0]]))
    assert result.solver_info["n_steps_completed"] == 0
    assert result.solver_info["initial_compatibility_residual"] is None
    assert result.solver_info["termination_step"] == 0
    assert result.solver_info["termination_time"] == pytest.approx(0.0)
    assert result.solver_info["numba_history_requested"] is True
    assert result.solver_info["numba_history_attempted"] is False
    assert result.solver_info["used_numba_history"] is False
    assert result.solver_info["numba_fallback_used"] is False
    assert result.solver_info["history_evaluations"] == 0
    assert result.solver_info["history_backend"] == "not_executed"


def test_nonfinite_predictor_and_corrector_are_not_returned_or_counted() -> None:
    largest = np.finfo(np.float64).max

    def predictor_overflow_rhs(time: float, state: np.ndarray) -> np.ndarray:
        del time
        return np.full_like(state, largest)

    def corrector_overflow_rhs(time: float, state: np.ndarray) -> np.ndarray:
        del time
        value = 1.0 if state[0] == 0.0 else largest
        return np.full_like(state, value)

    for rhs, expected_stage in (
        (predictor_overflow_rhs, "predictor"),
        (corrector_overflow_rhs, "corrector"),
    ):
        with np.errstate(over="ignore", invalid="ignore"):
            result = integrate_variable_order_caputo_type3_l1(
                rhs=rhs,
                initial_state=[0.0],
                order_function=_constant_six_order,
                step=4.0,
                n_steps=2,
                initial_regularity="nonsmooth",
                divergence_norm=None,
                use_acceleration=False,
            )

        assert result.status == "nonfinite_solution"
        assert np.all(np.isfinite(result.states))
        np.testing.assert_array_equal(result.times, np.array([0.0]))
        np.testing.assert_array_equal(result.states, np.array([[0.0]]))
        assert result.solver_info["n_steps_completed"] == 0
        assert result.solver_info["n_samples_returned"] == 1
        assert result.solver_info["termination_step"] == 1
        assert result.solver_info["termination_time"] == pytest.approx(4.0)
        assert result.solver_info["nonfinite_stage"] == expected_stage


def test_picard_corrector_converges_to_implicit_l1_solution() -> None:
    order = 0.6
    parameters = np.array([order, 0.0, 0.0, 0.5, 0.8])
    result = integrate_variable_order_caputo_type3_l1(
        **_solver_arguments(
            parameters=parameters,
            step=0.025,
            n_steps=8,
            corrector_atol=1.0e-13,
            corrector_rtol=1.0e-12,
        )
    )
    expected = _constant_alpha_linear_l1_reference(
        initial_value=0.2,
        order=order,
        step=0.025,
        n_steps=8,
        forcing=0.5,
        damping=0.8,
    )

    assert result.status == "ok"
    np.testing.assert_allclose(result.states[:, 0], expected, rtol=3.0e-12, atol=3.0e-13)
    assert result.solver_info["corrector"] == "picard"
    assert result.solver_info["max_corrector_iterations_used"] >= 2
    for index in range(1, len(result.times)):
        tolerance = 1.0e-13 + 1.0e-12 * np.linalg.norm(result.states[index])
        assert result.corrector_residuals[index] <= tolerance


def _noncontractive_rhs(time: float, state: np.ndarray) -> np.ndarray:
    del time
    return 1.0 + 10.0 * state


def test_picard_nonconvergence_raise_policy_propagates() -> None:
    with pytest.raises(RuntimeError, match="corrector|converg"):
        integrate_variable_order_caputo_type3_l1(
            rhs=_noncontractive_rhs,
            initial_state=[0.0],
            order_function=_constant_six_order,
            step=0.1,
            n_steps=2,
            lower_terminal=0.0,
            order_function_name="constant-alpha-0.6",
            corrector_atol=1.0e-15,
            corrector_rtol=1.0e-15,
            corrector_max_iterations=2,
            on_nonconvergence="raise",
            initial_regularity="nonsmooth",
            use_acceleration=False,
        )


def test_picard_nonconvergence_return_policy_is_structured_and_truncated() -> None:
    result = integrate_variable_order_caputo_type3_l1(
        rhs=_noncontractive_rhs,
        initial_state=[0.0],
        order_function=_constant_six_order,
        step=0.1,
        n_steps=3,
        lower_terminal=0.0,
        order_function_name="constant-alpha-0.6",
        corrector_atol=1.0e-15,
        corrector_rtol=1.0e-15,
        corrector_max_iterations=2,
        on_nonconvergence="return",
        initial_regularity="nonsmooth",
        use_acceleration=False,
    )

    assert result.status == "corrector_nonconvergence"
    assert result.solver_info["nonconverged_step"] == 1
    assert result.solver_info["failure_time"] == pytest.approx(0.1)
    assert result.solver_info["failure_iterations"] == 2
    assert result.solver_info["failure_residual"] is not None
    assert result.solver_info["max_corrector_iterations_used"] == 2
    assert result.solver_info["max_corrector_residual"] == pytest.approx(
        result.solver_info["failure_residual"]
    )
    assert result.solver_info["n_steps_completed"] == len(result.times) - 1
    assert len(result.times) < 4


def test_incompatible_initial_rhs_warns_only_for_declared_smooth_solution() -> None:
    def incompatible_rhs(time: float, state: np.ndarray) -> np.ndarray:
        del time
        return np.ones_like(state)

    arguments = dict(
        rhs=incompatible_rhs,
        initial_state=[0.0],
        order_function=_constant_six_order,
        step=0.02,
        n_steps=2,
        lower_terminal=0.0,
        order_function_name="constant-alpha-0.6",
        compatibility_tolerance=1.0e-12,
        use_acceleration=False,
        divergence_norm=None,
    )
    with pytest.warns(VariableOrderInitialCompatibilityWarning, match="compatib"):
        smooth = integrate_variable_order_caputo_type3_l1(
            **arguments,
            initial_regularity="smooth",
        )
    assert smooth.status == "ok"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nonsmooth = integrate_variable_order_caputo_type3_l1(
            **arguments,
            initial_regularity="nonsmooth",
        )
    assert nonsmooth.status == "ok"
    assert not any(
        isinstance(item.message, VariableOrderInitialCompatibilityWarning)
        for item in caught
    )


def test_smooth_compatibility_respects_tolerance_without_warning() -> None:
    def negligible_rhs(time: float, state: np.ndarray) -> np.ndarray:
        del time
        return np.full_like(state, 1.0e-14)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = integrate_variable_order_caputo_type3_l1(
            rhs=negligible_rhs,
            initial_state=[0.0],
            order_function=_constant_six_order,
            step=0.02,
            n_steps=2,
            lower_terminal=0.0,
            order_function_name="constant-alpha-0.6",
            initial_regularity="smooth",
            compatibility_tolerance=1.0e-12,
            use_acceleration=False,
            divergence_norm=None,
        )
    assert result.status == "ok"
    assert not any(
        isinstance(item.message, VariableOrderInitialCompatibilityWarning)
        for item in caught
    )


@pytest.mark.parametrize(
    ("invalid_order", "error"),
    [
        (0.0, ValueError),
        (1.0, ValueError),
        (-0.1, ValueError),
        (np.nan, ValueError),
        (True, TypeError),
        (0.5 + 0.1j, TypeError),
    ],
)
def test_invalid_order_function_values_are_rejected(
    invalid_order: object,
    error: type[Exception],
) -> None:
    def bad_order(time: float, state: np.ndarray) -> object:
        del time, state
        return invalid_order

    with pytest.raises(error, match="order|alpha|real"):
        integrate_variable_order_caputo_type3_l1(
            **_solver_arguments(order_function=bad_order)
        )


def test_order_function_is_validated_at_every_output_time() -> None:
    def later_invalid_order(time: float, state: np.ndarray) -> float:
        del state
        return 0.6 if time < 0.05 else 1.0

    with pytest.raises(ValueError, match="order|alpha"):
        integrate_variable_order_caputo_type3_l1(
            **_solver_arguments(
                order_function=later_invalid_order,
                step=0.02,
                n_steps=5,
            )
        )


def test_order_function_receives_detached_fixed_initial_state_context() -> None:
    observed: list[float] = []

    def fixed_schedule(time: float, state: np.ndarray) -> float:
        del time
        observed.append(float(state[0]))
        state[0] = 999.0
        return 0.6

    result = integrate_variable_order_caputo_type3_l1(
        rhs=_zero_rhs,
        initial_state=[0.25],
        order_function=fixed_schedule,
        step=0.02,
        n_steps=4,
        initial_regularity="smooth",
        use_acceleration=False,
        divergence_norm=None,
    )

    assert result.status == "ok"
    assert observed == [0.25] * 5
    np.testing.assert_array_equal(result.states[:, 0], np.full(5, 0.25))


@pytest.mark.parametrize(
    "bad_rhs",
    [
        lambda time, state: np.ones(state.size + 1),
        lambda time, state: np.full_like(state, np.nan),
        lambda time, state: np.full(state.shape, 1.0j),
    ],
)
def test_invalid_rhs_shape_finiteness_and_reality_are_rejected(bad_rhs) -> None:
    with pytest.raises((TypeError, ValueError), match="rhs|shape|real|finite"):
        integrate_variable_order_caputo_type3_l1(
            **_solver_arguments(rhs=bad_rhs, parameters=None)
        )


def test_internal_rhs_typeerror_is_not_reinterpreted_as_another_signature() -> None:
    calls = 0

    def broken_rhs(time: float, state: np.ndarray) -> np.ndarray:
        nonlocal calls
        del time, state
        calls += 1
        raise TypeError("variable-order-internal-typeerror")

    with pytest.raises(TypeError, match="variable-order-internal-typeerror"):
        integrate_variable_order_caputo_type3_l1(
            **_solver_arguments(rhs=broken_rhs, parameters=None)
        )
    assert calls == 1


@pytest.mark.parametrize(
    ("updates", "error", "match"),
    [
        ({"rhs": None}, TypeError, "rhs|callable"),
        ({"order_function": None}, TypeError, "order_function|callable"),
        ({"initial_state": []}, ValueError, "initial_state|finite"),
        ({"initial_state": [np.nan]}, ValueError, "initial_state|finite"),
        ({"step": 0.0}, ValueError, "step|positive"),
        ({"step": True}, TypeError, "step|real"),
        ({"n_steps": 0}, ValueError, "n_steps|integer"),
        ({"n_steps": 2.5}, ValueError, "n_steps|integer"),
        ({"n_steps": True}, ValueError, "n_steps|integer"),
        ({"lower_terminal": np.inf}, ValueError, "lower_terminal|finite"),
        ({"order_function_name": ""}, ValueError, "order_function_name"),
        ({"corrector_atol": -1.0}, ValueError, "corrector_atol"),
        ({"corrector_rtol": -1.0}, ValueError, "corrector_rtol"),
        ({"corrector_max_iterations": 0}, ValueError, "corrector_max_iterations"),
        ({"corrector_max_iterations": 2.5}, ValueError, "corrector_max_iterations"),
        ({"on_nonconvergence": "ignore"}, ValueError, "on_nonconvergence"),
        ({"initial_regularity": "analytic"}, ValueError, "initial_regularity"),
        ({"compatibility_tolerance": -1.0}, ValueError, "compatibility_tolerance"),
        ({"use_acceleration": 1}, TypeError, "use_acceleration|Boolean"),
        ({"divergence_norm": 0.0}, ValueError, "divergence_norm|positive"),
    ],
)
def test_invalid_solver_grid_and_options_are_rejected(
    updates: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        integrate_variable_order_caputo_type3_l1(
            **_solver_arguments(**updates)
        )


@pytest.mark.parametrize(
    ("order", "lag", "error"),
    [
        (0.0, 1, ValueError),
        (1.0, 1, ValueError),
        (True, 1, TypeError),
        (0.6, -1, ValueError),
        (0.6, 1.5, ValueError),
        (0.6, True, ValueError),
    ],
)
def test_invalid_l1_weight_contract_is_rejected(
    order: object,
    lag: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="order|lag|integer"):
        variable_order_l1_weight(order, lag)  # type: ignore[arg-type]


def test_structured_result_records_type3_method_orders_and_corrector_metadata() -> None:
    parameters = np.array([0.55, 0.2, 1.0, 0.0, 0.0])
    result = integrate_variable_order_caputo_type3_l1(
        **_solver_arguments(
            rhs=_zero_rhs,
            order_function=_affine_order,
            parameters=parameters,
            lower_terminal=1.0,
            step=0.02,
            n_steps=5,
            order_function_name="alpha(t)=0.55+0.2(t-1)",
            initial_state=[0.7, -0.2],
            initial_regularity="smooth",
        )
    )

    assert result.status == "ok"
    assert "python" in result.backend
    assert result.method == "vo_caputo_type3_l1"
    assert result.memory_policy == "full_history"
    assert result.grid_coordinate == "physical_time"
    assert result.order_function_name == "alpha(t)=0.55+0.2(t-1)"
    assert result.initial_regularity == "smooth"
    assert result.scope == "finite_numerical_trajectory_only"
    np.testing.assert_array_equal(result.trajectory[:, 0], result.times)
    np.testing.assert_array_equal(result.trajectory[:, 1:], result.states)
    np.testing.assert_array_equal(
        result.states,
        np.repeat(np.array([[0.7, -0.2]]), 6, axis=0),
    )
    assert result.solver_info["history_complexity"] == "O(N^2)"
    assert result.solver_info["corrector"] == "picard"
    assert result.solver_info["n_steps_completed"] == 5
    assert result.solver_info["order_min"] == pytest.approx(np.min(result.orders))
    assert result.solver_info["order_max"] == pytest.approx(np.max(result.orders))
    assert result.solver_info["initial_compatibility_residual"] == 0.0


def test_fractional_problem_dispatcher_consumes_all_variable_order_options() -> None:
    problem = FractionalProblem(**_problem_arguments())  # type: ignore[arg-type]
    parameters = np.array([0.6, 0.0, 0.0, 0.4, 0.3])
    dispatched = solve_fractional_problem(
        problem,
        _linear_rhs,
        parameters,
        use_acceleration=False,
        divergence_norm=None,
    )
    direct = integrate_variable_order_caputo_type3_l1(
        **_solver_arguments(parameters=parameters)
    )

    assert dispatched.status == direct.status == "ok"
    np.testing.assert_array_equal(dispatched.times, direct.times)
    np.testing.assert_array_equal(dispatched.states, direct.states)
    assert dispatched.metadata["derivative"] == "caputo_variable_type3"
    assert dispatched.metadata["method"] == "vo_caputo_type3_l1"
    assert dispatched.metadata["order_mode"] == "variable"
    assert dispatched.metadata["kernel_parameters"]["order_function_name"] == (
        "constant-alpha-0.6"
    )
    assert dispatched.metadata["method_options"] == dict(problem.method_options)
    assert dispatched.metadata["backend_info"]["order_function_name"] == (
        "constant-alpha-0.6"
    )
    np.testing.assert_array_equal(
        dispatched.metadata["backend_info"]["evaluated_orders"],
        direct.orders,
    )
    assert dispatched.metadata["backend_info"]["evaluated_order_samples"] == (
        direct.orders.size
    )
    np.testing.assert_array_equal(dispatched.metadata["rhs_parameters"], parameters)
    assert dispatched.metadata["backend_info"]["corrector_atol"] == pytest.approx(
        1.0e-12
    )
    assert dispatched.metadata["backend_info"]["corrector_rtol"] == pytest.approx(
        1.0e-10
    )
    assert dispatched.metadata["backend_info"]["corrector_max_iterations"] == 100
    assert dispatched.metadata["backend_info"]["on_nonconvergence"] == "raise"
    assert dispatched.metadata["backend_info"]["initial_regularity"] == "nonsmooth"
    assert dispatched.metadata["backend_info"]["compatibility_tolerance"] == pytest.approx(
        1.0e-12
    )
    assert dispatched.metadata["kernel_parameters"]["order_function"] == {
        "binding": "runtime_callable_not_serialized",
        "name": "constant-alpha-0.6",
    }


def test_dispatcher_rejects_nominal_initial_order_mismatch() -> None:
    problem = FractionalProblem(
        **_problem_arguments(orders=0.7)  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="declared_initial_order|does not match"):
        solve_fractional_problem(
            problem,
            _linear_rhs,
            np.array([0.6, 0.0, 0.0, 0.4, 0.3]),
            use_acceleration=False,
            divergence_norm=None,
        )


def test_dispatcher_rejects_unconsumed_method_option() -> None:
    options = dict(_problem_arguments()["method_options"])  # type: ignore[arg-type]
    options["silent_numerical_change"] = True
    problem = FractionalProblem(
        **_problem_arguments(method_options=options)  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="unsupported method_options|silent_numerical_change"):
        solve_fractional_problem(
            problem,
            _linear_rhs,
            np.array([0.6, 0.0, 0.0, 0.4, 0.3]),
            use_acceleration=False,
        )


def test_dispatcher_rejects_unconsumed_kernel_parameter() -> None:
    kernel = dict(_problem_arguments()["kernel_parameters"])  # type: ignore[arg-type]
    kernel["silent_kernel_change"] = True
    problem = FractionalProblem(
        **_problem_arguments(kernel_parameters=kernel)  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="unsupported kernel_parameters|silent_kernel_change"):
        solve_fractional_problem(
            problem,
            _linear_rhs,
            np.array([0.6, 0.0, 0.0, 0.4, 0.3]),
            use_acceleration=False,
        )


@pytest.mark.parametrize(
    "kernel_parameters",
    [
        {"order_function_name": "missing-callable"},
        {"order_function": "not-callable", "order_function_name": "bad"},
        {"order_function": _constant_order},
        {"order_function": _constant_order, "order_function_name": ""},
    ],
)
def test_fractional_problem_requires_named_callable_order_function(
    kernel_parameters: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError), match="order_function|callable|name"):
        FractionalProblem(
            **_problem_arguments(  # type: ignore[arg-type]
                kernel_parameters=kernel_parameters,
            )
        )
