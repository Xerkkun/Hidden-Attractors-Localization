from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors import fractional
from hidden_attractors.capabilities import get_capability
from hidden_attractors.fractional.contracts import get_fractional_method
from hidden_attractors.fractional.tempered_convolution_quadrature import (
    TEMPERED_CAPUTO_INITIAL_CONDITION,
    TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
    tempered_convolution_quadrature,
)
from hidden_attractors.fractional.tempered_fast_history import (
    TemperedFastHistoryResult,
    tempered_fast_multistep_history,
)


def _gngf2_weights(order: float, count: int) -> np.ndarray:
    gl = np.empty(count, dtype=np.float64)
    gl[0] = 1.0
    for lag in range(1, count):
        gl[lag] = ((lag - 1.0 - order) / lag) * gl[lag - 1]
    weights = (1.0 + 0.5 * order) * gl
    if count > 1:
        weights[1:] -= 0.5 * order * gl[:-1]
    return weights


def _direct_gngf2(
    samples: np.ndarray,
    order: float,
    tempering: float,
    step: float,
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    weights = _gngf2_weights(order, values.size)
    weights *= np.exp(-tempering * step * np.arange(values.size))
    weights *= step ** (-order)
    return np.array(
        [np.dot(weights[: n + 1], values[n::-1]) for n in range(values.size)]
    )


def test_public_api_contract_and_capability_are_executable() -> None:
    assert (
        fractional.tempered_fast_multistep_history
        is tempered_fast_multistep_history
    )
    assert fractional.TemperedFastHistoryResult is TemperedFastHistoryResult
    method = get_fractional_method("tempered_fast_multistep_history")
    assert method.implementation_status == "implemented"
    assert method.execution_kind == "sampled_operator"
    capability = get_capability("tempered_fast_multistep_history")
    assert capability.fractional_status == "implemented"
    assert capability.backend == "numba/python"


def test_fbdf1_componentwise_matches_direct_cq_and_python_numba() -> None:
    count = 240
    step = 0.015
    times = step * np.arange(count)
    samples = np.column_stack(
        (
            0.3 + np.sin(0.7 * times) + 0.04 * times**2,
            np.cos(1.1 * times) + 0.1 * times**1.4,
        )
    )
    common = dict(
        samples=samples,
        orders=np.array([0.42, 0.73]),
        tempering=np.array([0.0, 0.35]),
        multistep_method="fbdf1",
        definition="tempered_riemann_liouville",
        step=step,
        initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        relative_tolerance=1.0e-9,
    )
    python_result = tempered_fast_multistep_history(**common, backend="python")
    numba_result = tempered_fast_multistep_history(
        **common,
        backend="numba",
        quadrature_points=python_result.quadrature_points,
    )
    direct = tempered_convolution_quadrature(
        samples,
        np.array([0.42, 0.73]),
        tempering=np.array([0.0, 0.35]),
        bdf_order=1,
        definition="tempered_riemann_liouville",
        step=step,
        initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend="numba",
    )

    np.testing.assert_allclose(python_result.values, numba_result.values, atol=2e-13)
    np.testing.assert_allclose(numba_result.values, direct.values, atol=2e-10)
    assert python_result.quadrature_points >= 17
    assert np.all(python_result.l1_relative_weight_error <= 1.0e-9)
    assert python_result.compression_tolerance_satisfied
    assert not python_result.positive_exponential_materialized


def test_gngf2_matches_independent_direct_convolution() -> None:
    count = 280
    step = 0.02
    order = 0.67
    tempering = 0.28
    times = step * np.arange(count)
    samples = 0.6 + np.cos(0.4 * times) + times**1.8
    result = tempered_fast_multistep_history(
        samples,
        order,
        tempering=tempering,
        multistep_method="gngf2",
        definition="tempered_riemann_liouville",
        step=step,
        initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        relative_tolerance=1.0e-10,
        backend="numba",
    )
    direct = _direct_gngf2(samples, order, tempering, step)

    np.testing.assert_allclose(result.values, direct, atol=3e-10, rtol=2e-12)
    assert result.multistep_method == "gngf2"
    assert result.formal_order == 2
    assert "q*(1-z)/2" in result.generating_formula
    assert result.active_working_memory.startswith("O(d*(Q+n0))")
    assert result.quadrature_nodes.shape == (
        result.quadrature_points,
        1,
    )


@pytest.mark.parametrize("method", ["fbdf1", "gngf2"])
def test_conjugated_caputo_anchor_is_removed_with_exact_partial_sum(
    method: str,
) -> None:
    count = 220
    step = 0.0125
    tempering = 0.8
    times = step * np.arange(count)
    samples = 2.75 * np.exp(-tempering * times)
    result = tempered_fast_multistep_history(
        samples,
        0.61,
        tempering=tempering,
        multistep_method=method,
        definition="tempered_caputo",
        step=step,
        initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
        relative_tolerance=1.0e-9,
        backend="numba",
    )

    assert result.values[0] == 0.0
    assert np.max(np.abs(result.values)) < 2.0e-10
    assert result.caputo_initial_correction.startswith("exact_")


@pytest.mark.parametrize(
    ("method", "bdf_order"),
    [("fbdf1", 1), ("gngf2", 2)],
)
def test_integer_order_limit_is_exact_local_multistep_formula(
    method: str,
    bdf_order: int,
) -> None:
    count = 130
    step = 0.025
    tempering = 0.45
    times = step * np.arange(count)
    samples = np.sin(times) + 0.2 * times**2
    fast = tempered_fast_multistep_history(
        samples,
        1.0,
        tempering=tempering,
        multistep_method=method,
        definition="tempered_riemann_liouville",
        step=step,
        initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend="numba",
    )
    direct = tempered_convolution_quadrature(
        samples,
        1.0,
        tempering=tempering,
        bdf_order=bdf_order,
        definition="tempered_riemann_liouville",
        step=step,
        initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend="numba",
    )

    np.testing.assert_allclose(fast.values, direct.values, atol=2e-12, rtol=2e-12)
    assert fast.quadrature_points == 0
    assert fast.final_history_state.shape == (0, 1)
    assert np.array_equal(fast.l1_relative_weight_error, np.zeros(1))


def test_short_trajectory_uses_only_exact_local_history() -> None:
    samples = np.array([1.0, 1.2, 1.5, 1.9])
    result = tempered_fast_multistep_history(
        samples,
        0.55,
        tempering=0.2,
        multistep_method="fbdf1",
        definition="tempered_riemann_liouville",
        step=0.1,
        initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        local_history_steps=50,
        backend="python",
    )
    direct = tempered_convolution_quadrature(
        samples,
        0.55,
        tempering=0.2,
        bdf_order=1,
        definition="tempered_riemann_liouville",
        step=0.1,
        initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend="python",
    )

    np.testing.assert_array_equal(result.values, direct.values)
    assert result.local_history_steps == samples.size - 1
    assert result.quadrature_points == 0


def test_reported_operator_bound_covers_compression_up_to_roundoff() -> None:
    count = 190
    step = 0.01
    times = step * np.arange(count)
    samples = 0.4 + np.sin(times) + times**1.3
    fast = tempered_fast_multistep_history(
        samples,
        0.48,
        tempering=0.3,
        multistep_method="fbdf1",
        definition="tempered_riemann_liouville",
        step=step,
        initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        relative_tolerance=1.0e-8,
        backend="python",
    )
    direct = tempered_convolution_quadrature(
        samples,
        0.48,
        tempering=0.3,
        bdf_order=1,
        definition="tempered_riemann_liouville",
        step=step,
        initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend="python",
    )
    observed = float(np.max(np.abs(fast.values - direct.values)))
    assert observed <= float(fast.operator_absolute_error_bound[0]) + 3.0e-12


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"multistep_method": "bdf2"}, ValueError, "multistep_method"),
        ({"backend": "fft"}, ValueError, "backend"),
        ({"local_history_steps": 1}, ValueError, "local_history_steps"),
        ({"quadrature_points": 16}, ValueError, "quadrature_points"),
        ({"relative_tolerance": 1.0}, ValueError, "relative_tolerance"),
        ({"tail_cutoff": 0.0}, ValueError, "tail_cutoff"),
    ],
)
def test_invalid_fast_history_contracts_are_rejected(
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    arguments = dict(
        samples=np.linspace(0.0, 1.0, 80),
        orders=0.6,
        tempering=0.2,
        multistep_method="fbdf1",
        definition="tempered_riemann_liouville",
        step=0.01,
        initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend="python",
    )
    arguments.update(kwargs)
    with pytest.raises(error, match=match):
        tempered_fast_multistep_history(**arguments)


def test_automatic_quadrature_never_exceeds_a_declared_small_cap() -> None:
    with pytest.raises(ValueError, match="at least 65"):
        tempered_fast_multistep_history(
            np.linspace(0.0, 1.0, 80),
            0.6,
            tempering=0.2,
            multistep_method="fbdf1",
            definition="tempered_riemann_liouville",
            step=0.01,
            initial_condition_semantics=(
                TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION
            ),
            max_quadrature_points=64,
            backend="python",
        )


def test_initial_condition_token_and_uniform_grid_remain_explicit() -> None:
    samples = np.linspace(1.0, 2.0, 90)
    with pytest.raises(ValueError, match="requires"):
        tempered_fast_multistep_history(
            samples,
            0.5,
            tempering=0.1,
            definition="tempered_caputo",
            step=0.01,
            initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        )
    times = 0.01 * np.arange(samples.size)
    times[40] += 1.0e-4
    with pytest.raises(ValueError, match="uniform grid"):
        tempered_fast_multistep_history(
            samples,
            0.5,
            tempering=0.1,
            definition="tempered_riemann_liouville",
            times=times,
            initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        )
