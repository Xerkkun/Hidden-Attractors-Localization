from __future__ import annotations

import numpy as np
import pytest
from numba import njit
from scipy.special import gamma

from hidden_attractors.fractional import (
    get_fractional_derivative,
    get_fractional_method,
    grunwald_letnikov_derivative,
    grunwald_letnikov_weights,
    integrate_gl_explicit,
    integrate_gl_explicit_numba,
    list_fractional_derivatives,
    normalize_fractional_orders,
    validate_fractional_method,
)


@njit
def _constant_fractional_rhs(time, state, parameters):
    return np.ones_like(state) * parameters[0] + 0.0 * time


@njit
def _linear_fractional_rhs(time, state, parameters):
    return parameters[0] * state + 0.0 * time


def test_registry_does_not_equate_fractional_with_caputo() -> None:
    names = {item.name for item in list_fractional_derivatives()}
    assert {
        "caputo",
        "grunwald_letnikov",
        "riemann_liouville",
        "caputo_fabrizio",
        "atangana_baleanu_caputo",
        "tempered_caputo",
        "variable_order_caputo",
        "distributed_order",
    } <= names
    assert get_fractional_derivative("caputo_fabrizio").kernel_family == "nonsingular_exponential"
    assert (
        get_fractional_derivative("atangana_baleanu_caputo").kernel_family
        == "nonsingular_mittag_leffler"
    )


def test_method_contract_distinguishes_experimental_abc_from_planned_fast_soe() -> None:
    with pytest.raises(ValueError, match="not registered"):
        validate_fractional_method("atangana_baleanu_caputo", "caputo_abm_pece")
    with pytest.raises(NotImplementedError, match="not implemented"):
        validate_fractional_method(
            "atangana_baleanu_caputo",
            "abc_predictor_corrector",
        )
    experimental = validate_fractional_method(
        "atangana_baleanu_caputo",
        "abc_predictor_corrector",
        require_implemented=False,
    )
    assert experimental.implementation_status == "experimental"
    assert experimental.supported_combinations == (
        ("atangana_baleanu_caputo", "commensurate", "full_history"),
    )
    planned = validate_fractional_method(
        "atangana_baleanu_caputo",
        "abc_fast_soe_predictor_corrector",
        memory_policy="fast_history",
        require_implemented=False,
    )
    assert planned.implementation_status == "planned"


def test_lubich_cq_contract_is_an_experimental_sampled_operator() -> None:
    method = validate_fractional_method(
        "riemann_liouville",
        "convolution_quadrature",
        require_implemented=False,
    )
    assert method.implementation_status == "experimental"
    assert method.execution_kind == "sampled_operator"
    assert "tempered_caputo" not in method.derivative_families


@pytest.mark.parametrize(
    "derivative", ["tempered_riemann_liouville", "tempered_caputo"]
)
@pytest.mark.parametrize("order_mode", ["commensurate", "componentwise"])
def test_tempered_cq_contract_exposes_only_the_executable_batch_lane(
    derivative: str,
    order_mode: str,
) -> None:
    method = validate_fractional_method(
        derivative,
        "tempered_convolution_quadrature",
        order_mode=order_mode,
        memory_policy="full_history",
        require_implemented=False,
    )
    assert method.implementation_status == "experimental"
    assert method.execution_kind == "sampled_operator"
    assert method.supports_combination(derivative, order_mode, "full_history")
    assert "fast_history" not in method.memory_policies
    assert "delta(exp(-lambda*h)*z)**q" in method.accuracy_note


def test_tempered_fast_history_is_executable_while_symbol_shift_stays_planned() -> None:
    fast = get_fractional_method("tempered_fast_multistep_history")
    shifted = get_fractional_method("tempered_symbol_shift_cq")
    assert fast.implementation_status == "experimental"
    assert fast.execution_kind == "sampled_operator"
    assert fast.memory_policies == ("fast_history",)
    assert "GNGF2" in fast.accuracy_note
    assert "not CQ/FDE error" in fast.accuracy_note
    assert shifted.implementation_status == "planned"
    assert shifted.memory_policies == ("full_history",)
    assert "not a backend" in shifted.accuracy_note


def test_registry_records_exact_solver_combinations() -> None:
    caputo = get_fractional_derivative("caputo")
    assert "gl_explicit_discrete" in caputo.compatible_methods

    abm = get_fractional_method("caputo_abm_pece")
    assert set(abm.supported_combinations) == {
        ("caputo", "commensurate", "full_history"),
        ("caputo", "commensurate", "finite_window"),
        ("caputo", "componentwise", "block_restart"),
    }

    efork = get_fractional_method("efork3")
    assert efork.memory_policies == ("full_history", "finite_window")
    assert set(efork.supported_combinations) == {
        ("caputo", "commensurate", "full_history"),
        ("caputo", "commensurate", "finite_window"),
    }


@pytest.mark.parametrize(
    ("order_mode", "memory_policy"),
    [
        ("commensurate", "full_history"),
        ("commensurate", "finite_window"),
        ("componentwise", "block_restart"),
    ],
)
def test_caputo_abm_accepts_only_dispatched_exact_combinations(
    order_mode: str,
    memory_policy: str,
) -> None:
    method = validate_fractional_method(
        "caputo",
        "caputo_abm_pece",
        order_mode=order_mode,
        memory_policy=memory_policy,
    )
    assert method.supports_combination("caputo", order_mode, memory_policy)


@pytest.mark.parametrize(
    ("order_mode", "memory_policy"),
    [
        ("commensurate", "block_restart"),
        ("componentwise", "full_history"),
        ("componentwise", "finite_window"),
    ],
)
def test_caputo_abm_rejects_undispatched_cartesian_combinations(
    order_mode: str,
    memory_policy: str,
) -> None:
    with pytest.raises(ValueError, match="exact combination"):
        validate_fractional_method(
            "caputo",
            "caputo_abm_pece",
            order_mode=order_mode,
            memory_policy=memory_policy,
        )


def test_efork_contract_does_not_advertise_restart_or_block_restart() -> None:
    for memory_policy in ("restart", "block_restart"):
        with pytest.raises(ValueError, match="memory policy"):
            validate_fractional_method(
                "caputo",
                "efork3",
                memory_policy=memory_policy,
            )


@pytest.mark.parametrize("derivative", ["caputo", "grunwald_letnikov"])
@pytest.mark.parametrize("order_mode", ["commensurate", "componentwise"])
@pytest.mark.parametrize("memory_policy", ["full_history", "finite_window"])
def test_explicit_gl_contract_matches_all_dispatched_combinations(
    derivative: str,
    order_mode: str,
    memory_policy: str,
) -> None:
    method = validate_fractional_method(
        derivative,
        "gl_explicit_discrete",
        order_mode=order_mode,
        memory_policy=memory_policy,
        require_implemented=False,
    )
    assert method.supports_combination(derivative, order_mode, memory_policy)


def test_component_orders_are_normalized_and_validated() -> None:
    assert np.array_equal(normalize_fractional_orders(0.7, 3), [0.7, 0.7, 0.7])
    assert np.array_equal(normalize_fractional_orders([0.5, 0.8], 2), [0.5, 0.8])
    with pytest.raises(ValueError, match="received 2"):
        normalize_fractional_orders([0.5, 0.8], 3)
    with pytest.raises(ValueError, match="lie in"):
        normalize_fractional_orders([0.0, 0.8], 2)


def test_gl_weights_match_binomial_recurrence() -> None:
    weights = grunwald_letnikov_weights(0.5, 4)
    assert np.allclose(weights, [1.0, -0.5, -0.125, -0.0625])


def test_caputo_shifted_gl_derivative_matches_polynomial_solution() -> None:
    step = 0.001
    times = np.arange(1001, dtype=float) * step
    result = grunwald_letnikov_derivative(
        times**2,
        step,
        0.5,
        definition="caputo_shifted",
    )
    expected = gamma(3.0) / gamma(2.5) * times[-1] ** 1.5
    assert result.values[0] == pytest.approx(0.0, abs=1e-15)
    assert result.values[-1] == pytest.approx(expected, rel=8e-4)
    assert result.memory_policy == "full_history"


def test_raw_gl_constant_approximates_riemann_liouville_value() -> None:
    step = 0.001
    samples = np.ones(1001, dtype=float)
    result = grunwald_letnikov_derivative(
        samples,
        step,
        0.5,
        definition="riemann_liouville_gl",
    )
    expected_at_one = 1.0 / gamma(0.5)
    assert result.values[-1] == pytest.approx(expected_at_one, rel=3e-4)


def test_gl_supports_componentwise_orders_and_finite_memory_metadata() -> None:
    step = 0.002
    times = np.arange(501, dtype=float) * step
    samples = np.column_stack((times, times**2))
    result = grunwald_letnikov_derivative(
        samples,
        step,
        [0.5, 0.75],
        definition="caputo_shifted",
        history_window=100,
    )
    assert result.values.shape == samples.shape
    assert np.array_equal(result.orders, [0.5, 0.75])
    assert result.memory_policy == "finite_window"
    assert result.history_window == 100


def test_explicit_gl_solver_matches_constant_caputo_forcing() -> None:
    step = 0.001
    result = integrate_gl_explicit_numba(
        _constant_fractional_rhs,
        [2.0],
        0.5,
        [1.0],
        step=step,
        n_steps=1000,
        initialization="caputo_shifted",
    )
    expected = 2.0 + 1.0 / gamma(1.5)
    assert result.states[-1, 0] == pytest.approx(expected, rel=4e-4)
    assert result.derivative == "caputo"
    assert result.method == "gl_explicit_discrete"


def test_raw_discrete_gl_initialization_is_labelled_and_q1_reduces_to_euler() -> None:
    result = integrate_gl_explicit_numba(
        _linear_fractional_rhs,
        [1.0],
        1.0,
        [-1.0],
        step=0.001,
        n_steps=1000,
        initialization="discrete_gl",
    )
    assert result.initialization == "discrete_gl"
    assert result.derivative == "grunwald_letnikov"
    assert result.states[-1, 0] == pytest.approx((1.0 - 0.001) ** 1000)


@pytest.mark.parametrize("accelerated", [False, True])
def test_explicit_gl_divergence_limit_stops_both_backends(accelerated: bool) -> None:
    rhs = _constant_fractional_rhs if accelerated else (
        lambda time, state, parameters: np.ones_like(state) * parameters[0]
        + 0.0 * time
    )
    result = integrate_gl_explicit(
        rhs,
        [0.0],
        1.0,
        [1.0],
        step=0.1,
        n_steps=10,
        use_acceleration=accelerated,
        divergence_norm=0.25,
    )
    assert result.status == "diverged"
    assert result.divergence_norm == pytest.approx(0.25)
    assert result.times.size == result.states.shape[0] == 4
    assert result.states[-1, 0] == pytest.approx(0.3)


def test_explicit_gl_rejects_invalid_divergence_limit() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        integrate_gl_explicit_numba(
            _constant_fractional_rhs,
            [0.0],
            1.0,
            [1.0],
            step=0.1,
            n_steps=2,
            divergence_norm=float("inf"),
        )
