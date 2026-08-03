from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from math import gamma

import numpy as np
import pytest

from hidden_attractors.fractional.convolution_quadrature import (
    CAPUTO_SHIFTED_INITIAL_CONDITION,
    RL_OPERATOR_ONLY_INITIAL_CONDITION,
    lubich_bdf_weights,
    lubich_convolution_quadrature,
)
from hidden_attractors.fractional.grunwald_letnikov import (
    grunwald_letnikov_weights,
)


def _half_binomial_weights(count: int) -> list[Fraction]:
    """Exact rational coefficients of ``(1-z)**(1/2)``."""

    result = [Fraction(1)]
    order = Fraction(1, 2)
    for k in range(1, count):
        result.append(result[-1] * Fraction(k - 1 - order, k))
    return result


def _independent_half_order_bdf2_weights(count: int) -> np.ndarray:
    """Expand sqrt((3/2)(1-z)(1-z/3)) with 60-digit arithmetic."""

    first_factor = _half_binomial_weights(count)
    second_factor = [value / (3**k) for k, value in enumerate(first_factor)]
    rational_coefficients: list[Fraction] = []
    for k in range(count):
        rational_coefficients.append(
            sum(
                (first_factor[j] * second_factor[k - j] for j in range(k + 1)),
                start=Fraction(0),
            )
        )
    with localcontext() as context:
        context.prec = 60
        scale = (Decimal(3) / Decimal(2)).sqrt()
        return np.array(
            [
                float(scale * Decimal(value.numerator) / Decimal(value.denominator))
                for value in rational_coefficients
            ]
        )


@pytest.mark.parametrize("order", [0.17, 0.5, 0.83, 1.0])
@pytest.mark.parametrize("count", [0, 1, 2, 19, 128])
def test_bdf1_weights_are_exactly_the_canonical_gl_weights(
    order: float,
    count: int,
) -> None:
    actual = lubich_bdf_weights(order, count, bdf_order=1)
    expected = grunwald_letnikov_weights(order, count)
    assert np.array_equal(actual, expected)


def test_bdf2_weights_match_independent_high_precision_factor_expansion() -> None:
    actual = lubich_bdf_weights(0.5, 12, bdf_order=2)
    expected = _independent_half_order_bdf2_weights(12)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-15, atol=1.0e-16)


@pytest.mark.parametrize("bdf_order", [1, 2])
def test_q_one_is_the_ordinary_bdf_with_explicit_truncated_startup(
    bdf_order: int,
) -> None:
    samples = np.array([2.0, 3.0, 5.0, 8.0, 13.0])
    step = 0.25
    result = lubich_convolution_quadrature(
        samples,
        1.0,
        bdf_order=bdf_order,
        definition="riemann_liouville",
        step=step,
        initial_condition_semantics=RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend="python",
    )
    if bdf_order == 1:
        expected = np.array(
            [samples[0], *(samples[1:] - samples[:-1])]
        ) / step
        np.testing.assert_array_equal(result.weights[:, 0], [1.0, -1.0, 0, 0, 0])
    else:
        padded = np.pad(samples, (2, 0))
        expected = np.array(
            [
                (1.5 * padded[n + 2] - 2.0 * padded[n + 1] + 0.5 * padded[n])
                / step
                for n in range(samples.size)
            ]
        )
        np.testing.assert_array_equal(
            result.weights[:, 0], [1.5, -2.0, 0.5, 0.0, 0.0]
        )
    np.testing.assert_allclose(result.values, expected, rtol=0.0, atol=0.0)
    assert result.startup_convention == (
        "terminal_truncated_history_no_prehistory_extrapolation"
    )
    assert result.starting_corrections == "none_implemented"


def test_q_one_caputo_shift_exposes_bdf2_startup_semantics() -> None:
    samples = np.array([7.0, 9.0, 12.0, 16.0])
    result = lubich_convolution_quadrature(
        samples,
        1.0,
        bdf_order=2,
        definition="caputo_shifted",
        step=0.5,
        initial_condition_semantics=CAPUTO_SHIFTED_INITIAL_CONDITION,
        backend="python",
    )
    shifted = samples - samples[0]
    expected = np.array(
        [
            1.5 * shifted[0],
            1.5 * shifted[1] - 2.0 * shifted[0],
            1.5 * shifted[2] - 2.0 * shifted[1] + 0.5 * shifted[0],
            1.5 * shifted[3] - 2.0 * shifted[2] + 0.5 * shifted[1],
        ]
    ) / 0.5
    np.testing.assert_array_equal(result.values, expected)


@pytest.mark.parametrize("backend", ["python", "numba", "fft"])
@pytest.mark.parametrize("bdf_order", [1, 2])
def test_caputo_shifted_constant_is_exactly_zero_for_all_backends(
    backend: str,
    bdf_order: int,
) -> None:
    samples = np.tile(np.array([[2.5, -7.25]]), (43, 1))
    result = lubich_convolution_quadrature(
        samples,
        [0.31, 0.88],
        bdf_order=bdf_order,
        definition="caputo_shifted",
        step=0.01,
        initial_condition_semantics=CAPUTO_SHIFTED_INITIAL_CONDITION,
        backend=backend,
    )
    np.testing.assert_array_equal(result.values, np.zeros_like(samples))


@pytest.mark.parametrize(
    ("bdf_order", "minimum_ratio"),
    [(1, 1.7), (2, 3.2)],
)
def test_monomial_endpoint_converges_at_expected_bdf_rate(
    bdf_order: int,
    minimum_ratio: float,
) -> None:
    order = 0.63
    power = 3
    exact = gamma(power + 1) / gamma(power + 1 - order)
    errors = []
    for intervals in (80, 160, 320):
        times = np.linspace(0.0, 1.0, intervals + 1)
        result = lubich_convolution_quadrature(
            times**power,
            order,
            bdf_order=bdf_order,
            definition="caputo_shifted",
            times=times,
            lower_terminal=0.0,
            initial_condition_semantics=CAPUTO_SHIFTED_INITIAL_CONDITION,
            backend="numba",
        )
        errors.append(abs(float(result.values[-1]) - exact))
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[1] > minimum_ratio
    assert errors[1] / errors[2] > minimum_ratio


@pytest.mark.parametrize("bdf_order", [1, 2])
@pytest.mark.parametrize("definition", ["riemann_liouville", "caputo_shifted"])
def test_numba_fft_and_python_backends_agree(
    bdf_order: int,
    definition: str,
) -> None:
    rng = np.random.default_rng(20260803)
    samples = rng.normal(size=(73, 3))
    semantics = (
        RL_OPERATOR_ONLY_INITIAL_CONDITION
        if definition == "riemann_liouville"
        else CAPUTO_SHIFTED_INITIAL_CONDITION
    )
    arguments = dict(
        samples=samples,
        orders=[0.21, 0.56, 0.91],
        bdf_order=bdf_order,
        definition=definition,
        step=0.037,
        lower_terminal=-0.4,
        initial_condition_semantics=semantics,
    )
    python_result = lubich_convolution_quadrature(**arguments, backend="python")
    numba_result = lubich_convolution_quadrature(**arguments, backend="numba")
    fft_result = lubich_convolution_quadrature(**arguments, backend="fft")
    np.testing.assert_allclose(
        numba_result.values, python_result.values, rtol=2.0e-14, atol=2.0e-14
    )
    np.testing.assert_allclose(
        fft_result.values, python_result.values, rtol=3.0e-13, atol=3.0e-13
    )
    np.testing.assert_array_equal(numba_result.weights, python_result.weights)
    np.testing.assert_array_equal(fft_result.weights, python_result.weights)


def test_structured_result_reports_formula_cost_scope_and_sources() -> None:
    result = lubich_convolution_quadrature(
        np.arange(17.0),
        0.4,
        bdf_order=2,
        definition="riemann_liouville",
        step=0.125,
        lower_terminal=2.0,
        initial_condition_semantics=RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend="fft",
    )
    assert result.delta_formula == "delta(z) = 3/2 - 2*z + z^2/2"
    assert result.weights.shape == (17, 1)
    assert result.times[0] == 2.0
    assert "log(N)" in result.time_complexity
    assert result.working_memory.startswith("O(d*N)")
    assert result.scope == "sampled_fractional_operator_only_not_an_fde_solver"
    assert result.initial_condition_semantics == RL_OPERATOR_ONLY_INITIAL_CONDITION
    assert result.starting_corrections == "none_implemented"
    assert "https://doi.org/10.1137/0517050" in result.references
    assert (
        "https://doi.org/10.1023/B:BITN.0000046813.23911.2D"
        in result.references
    )
    assert "https://doi.org/10.1137/17M1118816" in result.references


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bdf_order": 3}, "bdf_order"),
        ({"bdf_order": True}, "Boolean"),
        ({"backend": "auto"}, "backend"),
        ({"orders": 0.0}, "orders"),
        ({"orders": 1.1}, "orders"),
        ({"definition": "riesz"}, "definition"),
        ({"initial_condition_semantics": "classical_point_value"}, "requires"),
    ],
)
def test_invalid_contract_values_are_rejected(
    kwargs: dict[str, object],
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "samples": np.arange(5.0),
        "orders": 0.5,
        "bdf_order": 1,
        "definition": "riemann_liouville",
        "step": 0.1,
        "initial_condition_semantics": RL_OPERATOR_ONLY_INITIAL_CONDITION,
        "backend": "python",
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        lubich_convolution_quadrature(**arguments)


def test_grid_and_sample_validation_is_strict() -> None:
    base = dict(
        samples=np.arange(4.0),
        orders=0.5,
        definition="riemann_liouville",
        initial_condition_semantics=RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend="python",
    )
    with pytest.raises(ValueError, match="exactly one"):
        lubich_convolution_quadrature(**base, step=0.1, times=np.arange(4) * 0.1)
    with pytest.raises(ValueError, match="uniform"):
        lubich_convolution_quadrature(
            **base, times=np.array([0.0, 0.1, 0.21, 0.3])
        )
    with pytest.raises(ValueError, match=r"times\[0\]"):
        lubich_convolution_quadrature(
            **base, times=np.array([1.0, 1.1, 1.2, 1.3]), lower_terminal=0.0
        )
    with pytest.raises(ValueError, match="finite"):
        lubich_convolution_quadrature(
            **{**base, "samples": np.array([0.0, np.nan])}, step=0.1
        )
    with pytest.raises(ValueError, match="at least two"):
        lubich_convolution_quadrature(
            **{**base, "samples": np.array([1.0])}, times=np.array([0.0])
        )


def test_component_order_length_must_match_dimension() -> None:
    with pytest.raises(ValueError, match="orders"):
        lubich_convolution_quadrature(
            np.ones((8, 2)),
            [0.3, 0.5, 0.7],
            definition="caputo_shifted",
            step=0.1,
            initial_condition_semantics=CAPUTO_SHIFTED_INITIAL_CONDITION,
            backend="numba",
        )


@pytest.mark.parametrize("count", [True, 2.9, "2"])
def test_weight_count_requires_a_real_integer(count: object) -> None:
    with pytest.raises(TypeError, match="count"):
        lubich_bdf_weights(0.5, count)  # type: ignore[arg-type]


def test_complex_inputs_are_rejected_instead_of_silently_truncated() -> None:
    with pytest.raises(TypeError, match="real-valued"):
        lubich_convolution_quadrature(
            np.array([1.0 + 2.0j, 2.0 + 3.0j]),
            0.5,
            definition="riemann_liouville",
            step=0.1,
            initial_condition_semantics=RL_OPERATOR_ONLY_INITIAL_CONDITION,
        )
    with pytest.raises(TypeError, match="real"):
        lubich_bdf_weights(True, 3)


def test_large_terminal_does_not_hide_nonuniform_or_unrepresentable_steps() -> None:
    terminal = 1.0e16
    with pytest.raises(ValueError, match="uniform"):
        lubich_convolution_quadrature(
            np.arange(3.0),
            0.5,
            definition="riemann_liouville",
            times=np.array([terminal, terminal + 2.0, terminal + 6.0]),
            lower_terminal=terminal,
            initial_condition_semantics=RL_OPERATOR_ONLY_INITIAL_CONDITION,
        )
    with pytest.raises(ValueError, match="representable"):
        lubich_convolution_quadrature(
            np.arange(3.0),
            0.5,
            definition="riemann_liouville",
            step=0.1,
            lower_terminal=terminal,
            initial_condition_semantics=RL_OPERATOR_ONLY_INITIAL_CONDITION,
        )


def test_shifted_uniform_grid_remains_valid_when_step_is_representable() -> None:
    terminal = 1.0e12
    times = terminal + 0.25 * np.arange(5)
    from_times = lubich_convolution_quadrature(
        np.arange(5.0),
        0.5,
        definition="riemann_liouville",
        times=times,
        lower_terminal=terminal,
        initial_condition_semantics=RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend="python",
    )
    from_step = lubich_convolution_quadrature(
        np.arange(5.0),
        0.5,
        definition="riemann_liouville",
        step=0.25,
        lower_terminal=terminal,
        initial_condition_semantics=RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend="python",
    )
    np.testing.assert_array_equal(from_times.values, from_step.values)
