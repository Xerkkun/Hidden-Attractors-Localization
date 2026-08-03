"""Edge and finite-fixture checks for tempered recurrent fast history.

These tests compare sampled operators and finite manufactured fixtures only.
They do not claim a general convergence rate or an FDE-solver theorem.
"""

from __future__ import annotations

from math import gamma

import numpy as np
import pytest

from hidden_attractors.fractional.tempered_convolution_quadrature import (
    TEMPERED_CAPUTO_INITIAL_CONDITION,
    TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
)
from hidden_attractors.fractional.tempered_fast_history import (
    tempered_fast_multistep_history,
)


def _base_weights(order: float, count: int, method: str) -> np.ndarray:
    gl = np.empty(count, dtype=np.float64)
    gl[0] = 1.0
    for lag in range(1, count):
        gl[lag] = ((lag - 1.0 - order) / lag) * gl[lag - 1]
    if method == "fbdf1":
        return gl
    if method != "gngf2":
        raise ValueError(f"unsupported test method {method!r}")
    weights = (1.0 + 0.5 * order) * gl
    if count > 1:
        weights[1:] -= 0.5 * order * gl[:-1]
    return weights


def _direct_tempered(
    samples: np.ndarray,
    orders: float | np.ndarray,
    tempering: float | np.ndarray,
    step: float,
    method: str,
    definition: str,
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[:, None]
    normalized_orders = np.broadcast_to(
        np.asarray(orders, dtype=np.float64).reshape(-1),
        (values.shape[1],),
    )
    normalized_tempering = np.broadcast_to(
        np.asarray(tempering, dtype=np.float64).reshape(-1),
        (values.shape[1],),
    )
    output = np.empty_like(values)
    lags = np.arange(values.shape[0], dtype=np.float64)
    for component, order in enumerate(normalized_orders):
        weights = _base_weights(float(order), values.shape[0], method)
        with np.errstate(under="ignore"):
            damping = np.exp(
                -float(normalized_tempering[component]) * step * lags
            )
        applied = np.convolve(
            values[:, component], weights * damping
        )[: values.shape[0]]
        if definition == "tempered_caputo":
            applied -= (
                values[0, component] * damping * np.cumsum(weights)
            )
            applied[0] = 0.0
        output[:, component] = step ** (-float(order)) * applied
    return output[:, 0] if was_vector else output


@pytest.mark.parametrize(
    ("count", "expected_quadrature_points"),
    [(5, 0), (6, None)],
)
def test_local_to_compressed_boundary_has_no_index_shift(
    count: int,
    expected_quadrature_points: int | None,
) -> None:
    local_steps = 4
    step = 0.08
    order = 0.57
    tempering = 0.35
    samples = 0.4 + np.arange(count, dtype=np.float64) ** 1.2
    result = tempered_fast_multistep_history(
        samples,
        order,
        tempering=tempering,
        multistep_method="fbdf1",
        definition="tempered_riemann_liouville",
        step=step,
        initial_condition_semantics=(
            TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION
        ),
        local_history_steps=local_steps,
        relative_tolerance=1.0e-10,
        backend="python",
    )
    direct = _direct_tempered(
        samples,
        order,
        tempering,
        step,
        "fbdf1",
        "tempered_riemann_liouville",
    )

    np.testing.assert_allclose(result.values, direct, atol=2.0e-12, rtol=2.0e-12)
    if expected_quadrature_points is None:
        assert result.quadrature_points >= 17
    else:
        assert result.quadrature_points == expected_quadrature_points


@pytest.mark.parametrize("impulse_index", [0, 16])
def test_old_history_impulses_cover_first_and_last_compressed_sources(
    impulse_index: int,
) -> None:
    count = 22
    local_steps = 4
    final_index = count - 1
    order = 0.63
    step = 0.04
    tempering = 0.45
    samples = np.zeros(count, dtype=np.float64)
    samples[impulse_index] = 1.0
    result = tempered_fast_multistep_history(
        samples,
        order,
        tempering=tempering,
        multistep_method="fbdf1",
        definition="tempered_riemann_liouville",
        step=step,
        initial_condition_semantics=(
            TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION
        ),
        local_history_steps=local_steps,
        relative_tolerance=1.0e-11,
        backend="python",
    )
    lag = final_index - impulse_index
    exact_weight = _base_weights(order, count, "fbdf1")[lag]
    expected_final = (
        step**(-order)
        * np.exp(-tempering * step * lag)
        * exact_weight
    )

    observed = abs(float(result.values[final_index]) - expected_final)
    roundoff = (
        128.0
        * np.finfo(np.float64).eps
        * max(1.0, abs(expected_final), abs(float(result.values[final_index])))
    )
    assert observed <= float(result.operator_absolute_error_bound[0]) + roundoff
    assert expected_final != 0.0


def test_step_scaling_is_h_to_minus_q_when_sigma_h_is_fixed() -> None:
    count = 84
    order = 0.58
    first_step = 0.02
    second_step = 0.05
    first_tempering = 0.7
    second_tempering = first_tempering * first_step / second_step
    indices = np.arange(count, dtype=np.float64)
    samples = 0.6 + np.sin(0.09 * indices) + 0.003 * indices**1.4
    common = dict(
        samples=samples,
        orders=order,
        multistep_method="gngf2",
        definition="tempered_riemann_liouville",
        initial_condition_semantics=(
            TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION
        ),
        local_history_steps=8,
        relative_tolerance=1.0e-10,
        backend="python",
    )
    first = tempered_fast_multistep_history(
        **common,
        tempering=first_tempering,
        step=first_step,
    )
    second = tempered_fast_multistep_history(
        **common,
        tempering=second_tempering,
        step=second_step,
        quadrature_points=first.quadrature_points,
    )
    expected_scale = (second_step / first_step) ** (-order)

    np.testing.assert_allclose(
        second.values,
        expected_scale * first.values,
        atol=3.0e-12,
        rtol=3.0e-12,
    )


@pytest.mark.parametrize(
    ("order", "method"),
    [(1.0e-6, "fbdf1"), (1.0 - 1.0e-6, "gngf2")],
)
def test_orders_near_open_and_integer_endpoints_remain_finite(
    order: float,
    method: str,
) -> None:
    count = 72
    step = 0.025
    tempering = 0.2
    indices = np.arange(count, dtype=np.float64)
    samples = 0.25 + np.cos(0.08 * indices) + 0.001 * indices**1.7
    result = tempered_fast_multistep_history(
        samples,
        order,
        tempering=tempering,
        multistep_method=method,
        definition="tempered_riemann_liouville",
        step=step,
        initial_condition_semantics=(
            TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION
        ),
        local_history_steps=8,
        relative_tolerance=1.0e-7,
        backend="python",
    )
    direct = _direct_tempered(
        samples,
        order,
        tempering,
        step,
        method,
        "tempered_riemann_liouville",
    )
    observed = float(np.max(np.abs(result.values - direct)))

    assert np.all(np.isfinite(result.values))
    assert np.all(result.l1_relative_weight_error <= 1.0e-7)
    assert observed <= float(result.operator_absolute_error_bound[0]) + 2.0e-10


@pytest.mark.parametrize(
    "definition",
    ["tempered_riemann_liouville", "tempered_caputo"],
)
def test_strong_tempering_underflow_is_safe(definition: str) -> None:
    count = 36
    order = 0.62
    step = 0.01
    tempering = 1.0e6
    samples = np.linspace(0.8, 1.7, count)
    token = (
        TEMPERED_CAPUTO_INITIAL_CONDITION
        if definition == "tempered_caputo"
        else TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION
    )
    result = tempered_fast_multistep_history(
        samples,
        order,
        tempering=tempering,
        multistep_method="gngf2",
        definition=definition,
        step=step,
        initial_condition_semantics=token,
        local_history_steps=4,
        relative_tolerance=1.0e-9,
        backend="python",
    )
    direct = _direct_tempered(
        samples,
        order,
        tempering,
        step,
        "gngf2",
        definition,
    )

    np.testing.assert_allclose(result.values, direct, atol=2.0e-12, rtol=2.0e-12)
    assert np.all(np.isfinite(result.values))
    assert np.all(result.local_tempered_weights[1:] == 0.0)
    assert np.all(result.final_history_state == 0.0)
    assert not result.positive_exponential_materialized


def test_gngf2_componentwise_with_nonzero_lower_terminal() -> None:
    count = 78
    step = 1.0 / 64.0
    lower_terminal = -1.5
    elapsed = step * np.arange(count, dtype=np.float64)
    times = lower_terminal + elapsed
    orders = np.array([0.34, 0.72, 1.0])
    tempering = np.array([0.0, 0.3, 0.8])
    samples = np.column_stack(
        (
            0.4 + np.sin(0.7 * elapsed),
            1.1 + elapsed**1.6,
            np.cos(0.5 * elapsed) + 0.2 * elapsed,
        )
    )
    result = tempered_fast_multistep_history(
        samples,
        orders,
        tempering=tempering,
        multistep_method="gngf2",
        definition="tempered_riemann_liouville",
        times=times,
        lower_terminal=lower_terminal,
        initial_condition_semantics=(
            TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION
        ),
        local_history_steps=8,
        relative_tolerance=1.0e-9,
        backend="numba",
    )
    direct = _direct_tempered(
        samples,
        orders,
        tempering,
        step,
        "gngf2",
        "tempered_riemann_liouville",
    )
    origin_result = tempered_fast_multistep_history(
        samples,
        orders,
        tempering=tempering,
        multistep_method="gngf2",
        definition="tempered_riemann_liouville",
        step=step,
        lower_terminal=0.0,
        initial_condition_semantics=(
            TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION
        ),
        local_history_steps=8,
        quadrature_points=result.quadrature_points,
        relative_tolerance=1.0e-9,
        backend="numba",
    )

    np.testing.assert_allclose(result.values, direct, atol=3.0e-10, rtol=3.0e-12)
    np.testing.assert_allclose(
        result.values, origin_result.values, atol=2.0e-13, rtol=2.0e-13
    )
    np.testing.assert_allclose(result.times, times, atol=0.0, rtol=0.0)
    assert result.lower_terminal == lower_terminal
    assert np.array_equal(result.orders, orders)
    assert np.all(result.quadrature_nodes[:, 2] == 0.0)
    assert np.all(result.quadrature_weights[:, 2] == 0.0)
    assert result.l1_relative_weight_error[2] == 0.0


def test_explicit_quadrature_rejects_insufficient_points() -> None:
    with pytest.raises(RuntimeError, match="quadrature_points does not satisfy"):
        tempered_fast_multistep_history(
            np.linspace(0.5, 1.5, 96),
            0.6,
            tempering=0.25,
            multistep_method="fbdf1",
            definition="tempered_riemann_liouville",
            step=0.02,
            initial_condition_semantics=(
                TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION
            ),
            local_history_steps=4,
            quadrature_points=17,
            relative_tolerance=1.0e-12,
            backend="python",
        )


def test_automatic_quadrature_reports_cap_exhaustion() -> None:
    with pytest.raises(RuntimeError, match="before max_quadrature_points"):
        tempered_fast_multistep_history(
            np.linspace(0.5, 1.5, 112),
            0.6,
            tempering=0.25,
            multistep_method="fbdf1",
            definition="tempered_riemann_liouville",
            step=0.02,
            initial_condition_semantics=(
                TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION
            ),
            local_history_steps=4,
            relative_tolerance=1.0e-15,
            max_quadrature_points=65,
            backend="python",
        )


def test_automatic_quadrature_rejects_cap_below_initial_grid() -> None:
    with pytest.raises(ValueError, match="at least 65"):
        tempered_fast_multistep_history(
            np.linspace(0.5, 1.5, 72),
            0.6,
            tempering=0.25,
            multistep_method="fbdf1",
            definition="tempered_riemann_liouville",
            step=0.02,
            initial_condition_semantics=(
                TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION
            ),
            local_history_steps=4,
            max_quadrature_points=64,
            backend="python",
        )


def test_caputo_compression_bound_covers_direct_difference() -> None:
    count = 126
    step = 0.015
    order = 0.47
    tempering = 0.4
    elapsed = step * np.arange(count, dtype=np.float64)
    samples = 1.3 + np.sin(0.8 * elapsed) + 0.15 * elapsed**1.8
    result = tempered_fast_multistep_history(
        samples,
        order,
        tempering=tempering,
        multistep_method="gngf2",
        definition="tempered_caputo",
        step=step,
        initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
        local_history_steps=8,
        relative_tolerance=1.0e-8,
        backend="python",
    )
    direct = _direct_tempered(
        samples,
        order,
        tempering,
        step,
        "gngf2",
        "tempered_caputo",
    )
    observed = float(np.max(np.abs(result.values - direct)))

    assert observed <= float(result.operator_absolute_error_bound[0]) + 5.0e-12
    assert result.caputo_initial_correction.startswith("exact_")


def test_finite_manufactured_refinement_reduces_endpoint_error() -> None:
    """Check one smooth finite fixture without asserting a universal rate."""

    order = 0.63
    tempering = 0.35
    power = 4.0
    errors: list[float] = []
    for count in (65, 129):
        elapsed = np.linspace(0.0, 1.0, count)
        step = float(elapsed[1] - elapsed[0])
        samples = np.exp(-tempering * elapsed) * (1.0 + elapsed**power)
        result = tempered_fast_multistep_history(
            samples,
            order,
            tempering=tempering,
            multistep_method="gngf2",
            definition="tempered_caputo",
            step=step,
            initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
            local_history_steps=8,
            relative_tolerance=1.0e-11,
            backend="python",
        )
        direct = _direct_tempered(
            samples,
            order,
            tempering,
            step,
            "gngf2",
            "tempered_caputo",
        )
        compression_error = float(np.max(np.abs(result.values - direct)))
        roundoff = (
            256.0
            * np.finfo(np.float64).eps
            * max(1.0, float(np.max(np.abs(direct))))
        )
        assert compression_error <= (
            float(result.operator_absolute_error_bound[0]) + roundoff
        )
        exact_endpoint = (
            np.exp(-tempering)
            * gamma(power + 1.0)
            / gamma(power + 1.0 - order)
        )
        errors.append(abs(float(result.values[-1]) - exact_endpoint))

    assert np.all(np.isfinite(errors))
    assert errors[1] < errors[0]
