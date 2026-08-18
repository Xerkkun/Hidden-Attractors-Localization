from __future__ import annotations

from math import gamma

import numpy as np
import pytest
from numba import njit

from hidden_attractors.fractional.abc_solver import (
    ABC_PREDICTOR_CORRECTOR_REFERENCES,
    abc_linear_product_weights,
    integrate_abc_predictor_corrector,
)
from hidden_attractors.fractional.atangana_baleanu import (
    atangana_baleanu_normalization,
)
from hidden_attractors.fractional.problem import (
    FractionalProblem,
    solve_fractional_problem,
)


def _zero_rhs(time: float, state: np.ndarray) -> np.ndarray:
    del time
    return np.zeros_like(state)


def _quadratic_rhs(
    time: float,
    state: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    tau = time - parameters[0]
    return np.full_like(state, parameters[1] * tau * tau)


@njit
def _quadratic_numba_rhs(time, state, parameters):
    tau = time - parameters[0]
    values = np.empty_like(state)
    for component in range(state.size):
        values[component] = parameters[1] * tau * tau
    return values


@njit
def _linear_time_numba_rhs(time, state, parameters):
    values = np.empty_like(state)
    for component in range(state.size):
        values[component] = parameters[0] * time
    return values


def _abc_power_solution(
    times: np.ndarray,
    *,
    initial_value: float,
    order: float,
    power: float,
    amplitude: float = 1.0,
    lower_terminal: float = 0.0,
    normalization: float = 1.0,
) -> np.ndarray:
    tau = times - lower_terminal
    local = (1.0 - order) * amplitude * tau**power / normalization
    memory = (
        order
        * amplitude
        * gamma(power + 1.0)
        * tau ** (power + order)
        / (normalization * gamma(power + order + 1.0))
    )
    return initial_value + local + memory


def test_zero_rhs_preserves_state_and_records_finite_trajectory_contract() -> None:
    initial = np.array([1.25, -0.75])
    result = integrate_abc_predictor_corrector(
        _zero_rhs,
        initial,
        0.63,
        step=0.05,
        n_steps=6,
        lower_terminal=1.5,
        use_acceleration=False,
        divergence_norm=None,
    )

    assert result.status == "ok"
    assert result.backend == "python_abc_pcm_full_history"
    np.testing.assert_allclose(
        result.states,
        np.repeat(initial[None, :], 7, axis=0),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(result.times, 1.5 + 0.05 * np.arange(7))
    np.testing.assert_allclose(result.trajectory[:, 0], result.times)
    np.testing.assert_allclose(result.trajectory[:, 1:], result.states)
    assert result.compatibility_residual == 0.0
    assert result.memory_policy == "full_history"
    assert result.solver_info["history_complexity"] == "O(N^2)"
    assert result.solver_info["fast_soe_used"] is False
    assert result.solver_info["n_steps_completed"] == 6
    assert result.scope == "finite_numerical_trajectory_only"
    assert set(ABC_PREDICTOR_CORRECTOR_REFERENCES) <= set(result.references)


def test_public_linear_product_weights_preserve_constant_interval_integral() -> None:
    order = 0.63
    step = 0.04
    theta0, theta1 = abc_linear_product_weights(order, step, 12)
    lags = np.arange(1, 13, dtype=float)
    expected = (
        step**order
        * (lags**order - (lags - 1.0) ** order)
        / order
    )

    assert theta0.shape == theta1.shape == (13,)
    assert theta0[0] == theta1[0] == 0.0
    np.testing.assert_allclose(theta0[1:] + theta1[1:], expected, rtol=2e-14)
    assert np.all(theta0[1:] >= 0.0)
    assert np.all(theta1[1:] >= 0.0)


def test_incompatible_classical_initial_value_is_rejected() -> None:
    def incompatible_rhs(time: float, state: np.ndarray) -> np.ndarray:
        del time
        return np.ones_like(state) * 2.0e-5

    with pytest.raises(ValueError, match=r"compatibility requires f\(lower_terminal, x0\)=0"):
        integrate_abc_predictor_corrector(
            incompatible_rhs,
            [0.0],
            0.7,
            step=0.1,
            n_steps=2,
            compatibility_tolerance=1.0e-6,
            use_acceleration=False,
        )


def test_quadratic_manufactured_solution_matches_abc_volterra_equation() -> None:
    order = 0.7
    lower_terminal = 0.4
    initial_value = 1.2
    parameters = np.array([lower_terminal, 1.0])
    result = integrate_abc_predictor_corrector(
        _quadratic_rhs,
        [initial_value],
        order,
        parameters,
        step=0.05,
        n_steps=20,
        lower_terminal=lower_terminal,
        use_acceleration=False,
        divergence_norm=None,
    )
    exact = _abc_power_solution(
        result.times,
        initial_value=initial_value,
        order=order,
        power=2.0,
        lower_terminal=lower_terminal,
    )

    assert result.status == "ok"
    np.testing.assert_allclose(result.states[:, 0], exact, rtol=0.0, atol=3.3e-4)


def test_quadratic_manufactured_solution_converges_at_second_order() -> None:
    order = 0.7
    initial_value = 1.2
    exact_final = _abc_power_solution(
        np.array([1.0]),
        initial_value=initial_value,
        order=order,
        power=2.0,
    )[0]
    errors: list[float] = []
    for step in (0.1, 0.05, 0.025):
        result = integrate_abc_predictor_corrector(
            _quadratic_rhs,
            [initial_value],
            order,
            np.array([0.0, 1.0]),
            step=step,
            n_steps=round(1.0 / step),
            use_acceleration=False,
            divergence_norm=None,
        )
        errors.append(abs(result.states[-1, 0] - exact_final))

    assert errors[0] > errors[1] > errors[2] > 0.0
    observed_orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert np.all(observed_orders > 1.95)
    assert np.all(observed_orders < 2.05)


def test_numba_and_python_paths_are_numerically_identical() -> None:
    parameters = np.array([0.2, 0.75])
    common = dict(
        initial_state=[1.0, -2.0],
        order=0.65,
        parameters=parameters,
        step=0.025,
        n_steps=20,
        lower_terminal=0.2,
        divergence_norm=None,
    )
    accelerated = integrate_abc_predictor_corrector(
        _quadratic_numba_rhs,
        use_acceleration=True,
        **common,
    )
    reference = integrate_abc_predictor_corrector(
        _quadratic_rhs,
        use_acceleration=False,
        **common,
    )

    assert accelerated.status == reference.status == "ok"
    assert accelerated.backend == "numba_abc_pcm_full_history"
    assert reference.backend == "python_abc_pcm_full_history"
    assert accelerated.solver_info["used_numba_backend"] is True
    assert reference.solver_info["used_numba_backend"] is False
    np.testing.assert_array_equal(accelerated.times, reference.times)
    np.testing.assert_allclose(
        accelerated.states,
        reference.states,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_explicit_normalization_is_used_and_reported() -> None:
    order = 0.7
    normalization = atangana_baleanu_normalization(order)
    parameters = np.array([0.0, 1.0])
    result = integrate_abc_predictor_corrector(
        _quadratic_rhs,
        [1.2],
        order,
        parameters,
        step=0.025,
        n_steps=40,
        normalization=atangana_baleanu_normalization,
        normalization_name="B(alpha)=1-alpha+alpha/Gamma(alpha)",
        use_acceleration=False,
        divergence_norm=None,
    )
    numeric_normalization = integrate_abc_predictor_corrector(
        _quadratic_rhs,
        [1.2],
        order,
        parameters,
        step=0.025,
        n_steps=40,
        normalization=normalization,
        use_acceleration=False,
        divergence_norm=None,
    )
    exact_final = _abc_power_solution(
        np.array([1.0]),
        initial_value=1.2,
        order=order,
        power=2.0,
        normalization=normalization,
    )[0]

    assert result.normalization_value == pytest.approx(normalization)
    assert result.normalization_description == (
        "B(alpha)=1-alpha+alpha/Gamma(alpha)"
    )
    assert result.states[-1, 0] == pytest.approx(exact_final, abs=1.2e-4)
    np.testing.assert_allclose(result.states, numeric_normalization.states)


def test_python_divergence_returns_only_the_computed_prefix() -> None:
    def growing_rhs(time: float, state: np.ndarray) -> np.ndarray:
        return np.full_like(state, 10.0 * time)

    result = integrate_abc_predictor_corrector(
        growing_rhs,
        [0.0],
        0.6,
        step=0.1,
        n_steps=5,
        use_acceleration=False,
        divergence_norm=0.1,
    )

    assert result.status == "diverged"
    assert result.times.shape == (2,)
    assert result.states.shape == (2, 1)
    assert result.times[-1] == pytest.approx(0.1)
    assert np.linalg.norm(result.states[-1]) > 0.1
    assert result.solver_info["n_steps_completed"] == 1


def test_numba_divergence_returns_only_the_computed_prefix() -> None:
    result = integrate_abc_predictor_corrector(
        _linear_time_numba_rhs,
        [0.0],
        0.6,
        np.array([10.0]),
        step=0.1,
        n_steps=5,
        use_acceleration=True,
        divergence_norm=0.1,
    )

    assert result.status == "diverged"
    assert result.times.shape == (2,)
    assert result.states.shape == (2, 1)
    assert result.solver_info["n_steps_completed"] == 1


def test_startup_failure_is_explicit_and_returns_only_initial_data() -> None:
    def expansive_startup_rhs(time: float, state: np.ndarray) -> np.ndarray:
        return np.array([100.0 * time * state[0]])

    result = integrate_abc_predictor_corrector(
        expansive_startup_rhs,
        [1.0],
        0.6,
        step=0.1,
        n_steps=5,
        startup_max_iterations=3,
        use_acceleration=False,
        divergence_norm=None,
    )

    assert result.status == "startup_no_convergence"
    np.testing.assert_array_equal(result.times, np.array([0.0]))
    np.testing.assert_array_equal(result.states, np.array([[1.0]]))
    assert result.startup_iterations == 3
    assert result.solver_info["n_steps_completed"] == 0


@pytest.mark.parametrize(
    ("updates", "error", "match"),
    [
        ({"rhs": None}, TypeError, "callable"),
        ({"initial_state": []}, ValueError, "initial_state"),
        ({"initial_state": [np.nan]}, ValueError, "initial_state"),
        ({"order": 0.0}, ValueError, "0 < order < 1"),
        ({"order": 1.0}, ValueError, "0 < order < 1"),
        ({"order": np.nan}, ValueError, "order must be finite"),
        ({"order": True}, TypeError, "order must be a real scalar"),
        ({"step": 0.0}, ValueError, "step must be finite and positive"),
        ({"step": True}, TypeError, "step must be a real scalar"),
        ({"lower_terminal": np.inf}, ValueError, "lower_terminal must be finite"),
        ({"n_steps": 0}, ValueError, "integer >= 1"),
        ({"n_steps": True}, ValueError, "integer >= 1"),
        (
            {"compatibility_tolerance": 0.0},
            ValueError,
            "compatibility_tolerance must be finite and positive",
        ),
        (
            {"startup_tolerance": 0.0},
            ValueError,
            "startup_tolerance must be finite and positive",
        ),
        ({"startup_max_iterations": 0}, ValueError, "integer >= 1"),
        ({"startup_max_iterations": True}, ValueError, "integer >= 1"),
        ({"normalization": 0.0}, ValueError, "finite positive"),
        ({"normalization": True}, ValueError, "positive scalar or callable"),
        ({"normalization": np.nan}, ValueError, "finite positive"),
        ({"divergence_norm": 0.0}, ValueError, "finite and positive"),
        ({"divergence_norm": np.nan}, ValueError, "finite and positive"),
    ],
)
def test_invalid_solver_arguments_are_rejected(
    updates: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "rhs": _zero_rhs,
        "initial_state": [0.0],
        "order": 0.6,
        "step": 0.1,
        "n_steps": 2,
        "use_acceleration": False,
    }
    arguments.update(updates)
    with pytest.raises(error, match=match):
        integrate_abc_predictor_corrector(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "value"),
    [("n_steps", 1.5), ("startup_max_iterations", 1.5)],
)
def test_nonintegral_iteration_counts_are_rejected(name: str, value: float) -> None:
    arguments: dict[str, object] = {
        "rhs": _zero_rhs,
        "initial_state": [0.0],
        "order": 0.6,
        "step": 0.1,
        "n_steps": 2,
        "startup_max_iterations": 10,
        "use_acceleration": False,
    }
    arguments[name] = value
    with pytest.raises(ValueError, match="integer >= 1"):
        integrate_abc_predictor_corrector(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_rhs",
    [
        lambda time, state: np.ones(state.size + 1),
        lambda time, state: np.full_like(state, np.nan),
    ],
)
def test_invalid_initial_rhs_is_rejected(bad_rhs) -> None:
    with pytest.raises(ValueError, match="rhs at the lower terminal"):
        integrate_abc_predictor_corrector(
            bad_rhs,
            [0.0],
            0.6,
            step=0.1,
            n_steps=2,
            use_acceleration=False,
        )


def test_python_rhs_can_require_acceleration_without_silent_fallback() -> None:
    with pytest.raises(RuntimeError, match="no Numba ABC backend"):
        integrate_abc_predictor_corrector(
            _zero_rhs,
            [0.0],
            0.6,
            step=0.1,
            n_steps=2,
            use_acceleration=True,
            allow_python_fallback=False,
        )


def test_numba_rhs_requires_a_finite_numeric_parameter_vector() -> None:
    with pytest.raises(TypeError, match="numeric parameter vector"):
        integrate_abc_predictor_corrector(
            _quadratic_numba_rhs,
            [0.0],
            0.6,
            {"lower_terminal": 0.0, "amplitude": 1.0},
            step=0.1,
            n_steps=2,
        )
    with pytest.raises(ValueError, match="finite values"):
        integrate_abc_predictor_corrector(
            _quadratic_numba_rhs,
            [0.0],
            0.6,
            [0.0, np.nan],
            step=0.1,
            n_steps=2,
        )


def test_fractional_problem_dispatches_abc_and_consumes_declared_options() -> None:
    problem = FractionalProblem(
        derivative="atangana_baleanu_caputo",
        method="abc_predictor_corrector",
        orders=0.7,
        initial_state=[1.2],
        step=0.05,
        t_span=(0.0, 0.2),
        memory_policy="full_history",
        kernel_parameters={"normalization": 1.0, "normalization_name": "B=1"},
        method_options={
            "compatibility_tolerance": 1.0e-13,
            "startup_tolerance": 1.0e-13,
            "startup_max_iterations": 25,
        },
        allow_experimental=True,
    )
    result = solve_fractional_problem(
        problem,
        _quadratic_rhs,
        np.array([0.0, 1.0]),
        use_acceleration=False,
        divergence_norm=None,
    )

    assert result.status == "ok"
    assert result.backend == "python_abc_pcm_full_history"
    assert result.metadata["backend_info"]["normalization_description"] == "B=1"
    assert result.metadata["backend_info"]["startup_max_iterations"] == 25
    assert result.metadata["kernel_parameters"]["normalization"] == 1.0
    assert result.metadata["method_options"]["startup_tolerance"] == 1.0e-13


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("method_options", {"ignored_option": 1}, "unsupported method_options"),
        ("kernel_parameters", {"ignored_kernel": 1}, "unsupported kernel_parameters"),
    ],
)
def test_fractional_problem_rejects_unknown_abc_settings(
    field: str,
    value: dict[str, object],
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "derivative": "atangana_baleanu_caputo",
        "method": "abc_predictor_corrector",
        "orders": 0.7,
        "initial_state": [0.0],
        "step": 0.1,
        "t_span": (0.0, 0.2),
        "memory_policy": "full_history",
        "allow_experimental": True,
    }
    arguments[field] = value
    problem = FractionalProblem(**arguments)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=match):
        solve_fractional_problem(problem, _zero_rhs, use_acceleration=False)


def test_fractional_problem_accepts_promoted_abc_without_opt_in() -> None:
    problem = FractionalProblem(
        derivative="atangana_baleanu_caputo",
        method="abc_predictor_corrector",
        orders=0.7,
        initial_state=[0.0],
        step=0.1,
        t_span=(0.0, 0.2),
        memory_policy="full_history",
    )
    result = solve_fractional_problem(problem, _zero_rhs, use_acceleration=False)
    assert result.status == "ok"
