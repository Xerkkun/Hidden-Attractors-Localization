from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gamma

from hidden_attractors.fractional.sampled_operators import (
    OPERATOR_ONLY_INITIAL_CONDITION,
    conformable_khalil_derivative,
    riemann_liouville_gl_derivative,
    tempered_grunwald_letnikov_derivative,
    variable_order_grunwald_letnikov_derivative,
)
from hidden_attractors.fractional.contracts import (
    get_fractional_derivative,
    get_fractional_method,
)


def _uniform_grid(step: float = 0.001, n_steps: int = 1000, start: float = 0.0):
    return start + np.arange(n_steps + 1, dtype=float) * step


def test_rl_gl_constant_and_power_match_analytic_terminal_values() -> None:
    times = _uniform_grid()
    constant = riemann_liouville_gl_derivative(
        np.ones_like(times),
        times,
        0.5,
        lower_terminal=0.0,
        initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION,
    )
    quadratic = riemann_liouville_gl_derivative(
        times**2,
        times,
        0.5,
        lower_terminal=0.0,
        initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION,
    )

    assert constant.values[-1] == pytest.approx(1.0 / gamma(0.5), rel=3e-4)
    expected_quadratic = gamma(3.0) / gamma(2.5)
    assert quadratic.values[-1] == pytest.approx(expected_quadratic, rel=8e-4)
    assert constant.initial_condition_semantics == OPERATOR_ONLY_INITIAL_CONDITION
    assert constant.lower_terminal == 0.0


def test_rl_shifted_terminal_and_q_one_backward_difference() -> None:
    lower_terminal = 2.5
    times = _uniform_grid(step=0.01, n_steps=100, start=lower_terminal)
    samples = times - lower_terminal
    result = riemann_liouville_gl_derivative(
        samples,
        times,
        1.0,
        lower_terminal=lower_terminal,
        initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION,
    )

    assert result.values[0] == pytest.approx(0.0, abs=0.0)
    assert np.allclose(result.values[1:], 1.0, rtol=0.0, atol=5e-14)


def test_rl_contract_rejects_implicit_terminal_or_ivp_semantics() -> None:
    times = _uniform_grid(step=0.1, n_steps=10)
    with pytest.raises(ValueError, match=r"times\[0\] must equal lower_terminal"):
        riemann_liouville_gl_derivative(
            np.ones_like(times),
            times,
            0.5,
            lower_terminal=-1.0,
            initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION,
        )
    with pytest.raises(ValueError, match="not an IVP solver"):
        riemann_liouville_gl_derivative(
            np.ones_like(times),
            times,
            0.5,
            lower_terminal=0.0,
            initial_condition_semantics="classical_state_value",
        )


def test_rl_numba_and_python_paths_match_componentwise() -> None:
    times = _uniform_grid(step=0.005, n_steps=200)
    samples = np.column_stack((times, 1.0 + times**2))
    kwargs = {
        "lower_terminal": 0.0,
        "initial_condition_semantics": OPERATOR_ONLY_INITIAL_CONDITION,
    }
    accelerated = riemann_liouville_gl_derivative(
        samples, times, [0.4, 0.8], backend="numba", **kwargs
    )
    reference = riemann_liouville_gl_derivative(
        samples, times, [0.4, 0.8], backend="python", **kwargs
    )

    assert accelerated.backend == "numba"
    assert reference.backend == "python"
    assert np.allclose(accelerated.values, reference.values, rtol=2e-14, atol=2e-14)


def test_tempered_lambda_zero_reduces_exactly_to_rl_gl() -> None:
    times = _uniform_grid(step=0.004, n_steps=250)
    samples = np.column_stack((1.0 + times, times**2))
    kwargs = {
        "lower_terminal": 0.0,
        "initial_condition_semantics": OPERATOR_ONLY_INITIAL_CONDITION,
        "backend": "numba",
    }
    rl = riemann_liouville_gl_derivative(samples, times, [0.5, 0.75], **kwargs)
    tempered = tempered_grunwald_letnikov_derivative(
        samples,
        times,
        [0.5, 0.75],
        tempering=0.0,
        **kwargs,
    )

    assert np.array_equal(tempered.values, rl.values)
    assert tempered.derivative == "tempered_riemann_liouville"
    assert tempered.parameters["tempering"] == 0.0


def test_tempered_exponential_power_matches_conjugated_rl_identity() -> None:
    times = _uniform_grid()
    order = 0.5
    tempering = 0.7
    samples = np.exp(-tempering * times) * times**2
    result = tempered_grunwald_letnikov_derivative(
        samples,
        times,
        order,
        tempering=tempering,
        lower_terminal=0.0,
        initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION,
    )
    expected = np.exp(-tempering) * gamma(3.0) / gamma(2.5)

    assert result.values[-1] == pytest.approx(expected, rel=8e-4)
    assert result.convention == "exponential_conjugation_uncorrected"


def test_tempered_q_one_annihilates_matching_exponential_discretely() -> None:
    times = _uniform_grid(step=0.01, n_steps=100)
    tempering = 1.25
    samples = np.exp(-tempering * times)
    result = tempered_grunwald_letnikov_derivative(
        samples,
        times,
        1.0,
        tempering=tempering,
        lower_terminal=0.0,
        initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION,
    )

    assert np.allclose(result.values[1:], 0.0, rtol=0.0, atol=3e-14)


def test_tempered_numba_python_parity_and_parameter_validation() -> None:
    times = _uniform_grid(step=0.005, n_steps=200)
    samples = 1.0 + np.sin(times)
    kwargs = {
        "tempering": 0.35,
        "lower_terminal": 0.0,
        "initial_condition_semantics": OPERATOR_ONLY_INITIAL_CONDITION,
    }
    accelerated = tempered_grunwald_letnikov_derivative(
        samples, times, 0.65, backend="numba", **kwargs
    )
    reference = tempered_grunwald_letnikov_derivative(
        samples, times, 0.65, backend="python", **kwargs
    )
    assert np.allclose(accelerated.values, reference.values, rtol=2e-14, atol=2e-14)

    with pytest.raises(ValueError, match="non-negative"):
        tempered_grunwald_letnikov_derivative(
            samples,
            times,
            0.65,
            tempering=-0.1,
            lower_terminal=0.0,
            initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION,
        )


def test_variable_order_constant_case_is_identical_to_rl_gl() -> None:
    times = _uniform_grid(step=0.005, n_steps=200)
    samples = 1.0 + times**2
    kwargs = {
        "lower_terminal": 0.0,
        "initial_condition_semantics": OPERATOR_ONLY_INITIAL_CONDITION,
        "backend": "numba",
    }
    fixed = riemann_liouville_gl_derivative(samples, times, 0.625, **kwargs)
    variable = variable_order_grunwald_letnikov_derivative(
        samples, times, np.full(times.size, 0.625), **kwargs
    )

    assert np.array_equal(variable.values, fixed.values)
    assert variable.orders.shape == (times.size, 1)
    assert "output_time" in variable.convention


def test_variable_order_q_one_and_numba_python_parity() -> None:
    times = _uniform_grid(step=0.01, n_steps=100)
    samples = times.copy()
    orders = 0.35 + 0.65 * times
    accelerated = variable_order_grunwald_letnikov_derivative(
        samples,
        times,
        orders,
        lower_terminal=0.0,
        initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION,
        backend="numba",
    )
    reference = variable_order_grunwald_letnikov_derivative(
        samples,
        times,
        orders,
        lower_terminal=0.0,
        initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION,
        backend="python",
    )

    assert np.allclose(accelerated.values, reference.values, rtol=2e-14, atol=2e-14)
    assert accelerated.values[-1] == pytest.approx(1.0, abs=3e-14)


def test_variable_order_rejects_ambiguous_component_only_vector() -> None:
    times = _uniform_grid(step=0.1, n_steps=10)
    samples = np.column_stack((times, times**2))
    with pytest.raises(ValueError, match="Variable orders"):
        variable_order_grunwald_letnikov_derivative(
            samples,
            times,
            [0.4, 0.8],
            lower_terminal=0.0,
            initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION,
        )


def test_conformable_power_identity_and_q_one_limit() -> None:
    times = _uniform_grid(step=0.01, n_steps=100)
    ordinary_derivative = 3.0 * times**2
    fractional = conformable_khalil_derivative(
        times,
        ordinary_derivative,
        0.6,
        terminal_policy="bounded_derivative_zero",
    )
    integer = conformable_khalil_derivative(
        times,
        ordinary_derivative,
        1.0,
    )

    assert np.allclose(fractional.values, 3.0 * times**2.4, atol=2e-14)
    assert np.array_equal(integer.values, ordinary_derivative)
    assert fractional.initial_condition_semantics == "local_operator_no_memory_no_ivp"


def test_conformable_terminal_policy_handles_singular_derivative_limit() -> None:
    order = 0.6
    times = np.array([0.0, 0.25, 1.0])
    ordinary_derivative = np.array(
        [np.inf, order * times[1] ** (order - 1.0), order]
    )
    with pytest.raises(ValueError, match="terminal_policy"):
        conformable_khalil_derivative(times, ordinary_derivative, order)

    result = conformable_khalil_derivative(
        times,
        ordinary_derivative,
        order,
        terminal_policy="provided",
        terminal_value=order,
    )
    assert np.allclose(result.values, order)


def test_conformable_accepts_callable_component_rhs_and_matches_python() -> None:
    times = np.linspace(0.1, 1.0, 50)

    def ordinary_rhs(time: float) -> np.ndarray:
        return np.array([2.0 * time, 3.0 * time**2])

    accelerated = conformable_khalil_derivative(
        times, ordinary_rhs, [0.5, 0.75], backend="numba"
    )
    reference = conformable_khalil_derivative(
        times, ordinary_rhs, [0.5, 0.75], backend="python"
    )
    expected = np.column_stack((2.0 * times**1.5, 3.0 * times**2.25))

    assert np.allclose(accelerated.values, expected, rtol=2e-15, atol=2e-15)
    assert np.array_equal(accelerated.values, reference.values)


def test_conformable_shifted_terminal_is_explicitly_labelled() -> None:
    times = np.array([2.0, 2.5, 3.0])
    derivative = np.array([0.0, 1.0, 2.0])
    result = conformable_khalil_derivative(
        times,
        derivative,
        0.5,
        lower_terminal=2.0,
        terminal_policy="bounded_derivative_zero",
    )

    assert result.convention == "shifted_khalil"
    assert np.allclose(result.values, [0.0, np.sqrt(0.5), 2.0])


def test_additional_operator_families_are_public_and_registry_anchored() -> None:
    from hidden_attractors import fractional

    assert get_fractional_derivative(
        "tempered_riemann_liouville"
    ).implementation_status == "experimental"
    assert get_fractional_method(
        "variable_order_gl_direct"
    ).execution_kind == "sampled_operator"
    assert get_fractional_method(
        "conformable_sampled_local"
    ).execution_kind == "sampled_operator"
    assert (
        fractional.tempered_grunwald_letnikov_derivative
        is tempered_grunwald_letnikov_derivative
    )
