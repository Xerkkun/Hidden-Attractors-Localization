from __future__ import annotations

from math import gamma

import numpy as np
import pytest

import hidden_attractors.fractional as fractional
from hidden_attractors.fractional.convolution_quadrature import (
    CAPUTO_SHIFTED_INITIAL_CONDITION,
    RL_OPERATOR_ONLY_INITIAL_CONDITION,
    lubich_convolution_quadrature,
)
from hidden_attractors.fractional.sampled_operators import (
    OPERATOR_ONLY_INITIAL_CONDITION,
    tempered_grunwald_letnikov_derivative,
)
from hidden_attractors.fractional.tempered_convolution_quadrature import (
    TEMPERED_CAPUTO_INITIAL_CONDITION,
    TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
    tempered_convolution_quadrature,
)


def _semantics(definition: str) -> tuple[str, str, str]:
    if definition == "tempered_riemann_liouville":
        return (
            TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
            "riemann_liouville",
            RL_OPERATOR_ONLY_INITIAL_CONDITION,
        )
    return (
        TEMPERED_CAPUTO_INITIAL_CONDITION,
        "caputo_shifted",
        CAPUTO_SHIFTED_INITIAL_CONDITION,
    )


def test_fractional_facade_exports_tempered_cq_contract() -> None:
    assert (
        fractional.tempered_convolution_quadrature
        is tempered_convolution_quadrature
    )
    assert (
        fractional.TEMPERED_CAPUTO_INITIAL_CONDITION
        == TEMPERED_CAPUTO_INITIAL_CONDITION
    )
    assert (
        fractional.TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION
        == TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION
    )
    assert fractional.TemperedConvolutionQuadratureResult.__name__ == (
        "TemperedConvolutionQuadratureResult"
    )


@pytest.mark.parametrize(
    "definition", ["tempered_riemann_liouville", "tempered_caputo"]
)
@pytest.mark.parametrize("bdf_order", [1, 2])
@pytest.mark.parametrize("backend", ["python", "numba", "fft"])
def test_zero_tempering_is_exact_canonical_cq_reduction(
    definition: str,
    bdf_order: int,
    backend: str,
) -> None:
    rng = np.random.default_rng(20260803)
    samples = rng.normal(size=(47, 2))
    token, base_definition, base_token = _semantics(definition)
    result = tempered_convolution_quadrature(
        samples,
        [0.37, 0.81],
        tempering=[0.0, -0.0],
        bdf_order=bdf_order,
        definition=definition,
        step=0.025,
        lower_terminal=-1.2,
        initial_condition_semantics=token,
        backend=backend,
    )
    reference = lubich_convolution_quadrature(
        samples,
        [0.37, 0.81],
        bdf_order=bdf_order,
        definition=base_definition,
        step=0.025,
        lower_terminal=-1.2,
        initial_condition_semantics=base_token,
        backend=backend,
    )

    np.testing.assert_array_equal(result.values, reference.values)
    np.testing.assert_array_equal(result.base_weights, reference.weights)
    np.testing.assert_array_equal(result.weights, reference.weights)
    np.testing.assert_array_equal(result.tempering, [0.0, 0.0])
    assert result.positive_exponential_materialized is False
    assert result.damping_underflowed is False


@pytest.mark.parametrize(
    "definition", ["tempered_riemann_liouville", "tempered_caputo"]
)
@pytest.mark.parametrize("backend", ["python", "numba", "fft"])
def test_discrete_exponential_conjugation_identity_is_exactly_the_contract(
    definition: str,
    backend: str,
) -> None:
    rng = np.random.default_rng(9217)
    lower_terminal = 1.4
    step = 0.02
    tau = step * np.arange(65, dtype=np.float64)
    times = lower_terminal + tau
    samples = rng.normal(size=(tau.size, 2))
    orders = np.array([0.41, 0.77])
    tempering = np.array([0.35, 1.1])
    token, base_definition, base_token = _semantics(definition)

    result = tempered_convolution_quadrature(
        samples,
        orders,
        tempering=tempering,
        bdf_order=2,
        definition=definition,
        times=times,
        lower_terminal=lower_terminal,
        initial_condition_semantics=token,
        backend=backend,
    )
    positive_factors = np.exp(tau[:, None] * tempering[None, :])
    conjugated = lubich_convolution_quadrature(
        positive_factors * samples,
        orders,
        bdf_order=2,
        definition=base_definition,
        times=times,
        lower_terminal=lower_terminal,
        initial_condition_semantics=base_token,
        backend=backend,
    )
    expected = conjugated.values / positive_factors
    expected_weights = result.base_weights * np.exp(
        -tau[:, None] * tempering[None, :]
    )

    np.testing.assert_allclose(result.values, expected, rtol=3.0e-13, atol=3.0e-13)
    np.testing.assert_allclose(
        result.weights, expected_weights, rtol=3.0e-15, atol=0.0
    )


@pytest.mark.parametrize(
    ("bdf_order", "minimum_ratio"), [(1, 1.7), (2, 3.2)]
)
def test_manufactured_tempered_power_converges_at_the_bdf_rate(
    bdf_order: int,
    minimum_ratio: float,
) -> None:
    order = 0.62
    tempering = 0.7
    power = 3
    coefficient = gamma(power + 1) / gamma(power + 1 - order)
    exact_endpoint = np.exp(-tempering) * coefficient
    errors: list[float] = []
    for intervals in (80, 160, 320):
        times = np.linspace(0.0, 1.0, intervals + 1)
        samples = np.exp(-tempering * times) * times**power
        result = tempered_convolution_quadrature(
            samples,
            order,
            tempering=tempering,
            bdf_order=bdf_order,
            definition="tempered_caputo",
            times=times,
            lower_terminal=0.0,
            initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
            backend="numba",
        )
        errors.append(abs(float(result.values[-1]) - exact_endpoint))

    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[1] > minimum_ratio
    assert errors[1] / errors[2] > minimum_ratio


@pytest.mark.parametrize("bdf_order", [1, 2])
@pytest.mark.parametrize(
    "definition", ["tempered_riemann_liouville", "tempered_caputo"]
)
def test_q_one_uses_exact_terminal_truncated_bdf_startup(
    bdf_order: int,
    definition: str,
) -> None:
    step = 0.1
    tempering = 0.8
    damping = np.exp(-tempering * step * np.arange(5, dtype=np.float64))
    samples = np.array([2.0, -0.5, 1.3, 2.2, -0.1])
    token, _, _ = _semantics(definition)
    result = tempered_convolution_quadrature(
        samples,
        1.0,
        tempering=tempering,
        bdf_order=bdf_order,
        definition=definition,
        step=step,
        lower_terminal=2.3,
        initial_condition_semantics=token,
        backend="python",
    )

    base_weights = (
        np.array([1.0, -1.0, 0.0, 0.0, 0.0])
        if bdf_order == 1
        else np.array([1.5, -2.0, 0.5, 0.0, 0.0])
    )
    damped_weights = base_weights * damping
    expected = np.empty_like(samples)
    partial_sums = np.cumsum(base_weights)
    for n in range(samples.size):
        unscaled = sum(
            damped_weights[lag] * samples[n - lag] for lag in range(n + 1)
        )
        if definition == "tempered_caputo":
            unscaled -= samples[0] * damping[n] * partial_sums[n]
        expected[n] = unscaled / step

    np.testing.assert_allclose(result.values, expected, rtol=0.0, atol=2.0e-14)
    np.testing.assert_array_equal(result.base_weights[:, 0], base_weights)
    if definition == "tempered_caputo":
        assert result.values[0] == 0.0
        # BDF2 is deliberately not replaced by a BDF1 start at n=1.
        if bdf_order == 2:
            transformed_1 = np.exp(tempering * step) * samples[1]
            expected_start = (
                np.exp(-tempering * step)
                * 1.5
                * (transformed_1 - samples[0])
                / step
            )
            assert result.values[1] == pytest.approx(expected_start)


def test_componentwise_orders_tempering_and_backends_agree() -> None:
    rng = np.random.default_rng(4711)
    samples = rng.normal(size=(97, 3))
    arguments = dict(
        samples=samples,
        orders=[0.29, 0.64, 1.0],
        tempering=[0.0, 0.45, 2.3],
        bdf_order=2,
        definition="tempered_caputo",
        step=0.013,
        lower_terminal=-0.7,
        initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
    )
    python_result = tempered_convolution_quadrature(**arguments, backend="python")
    numba_result = tempered_convolution_quadrature(**arguments, backend="numba")
    fft_result = tempered_convolution_quadrature(**arguments, backend="fft")

    np.testing.assert_allclose(
        numba_result.values, python_result.values, rtol=2.0e-13, atol=2.0e-13
    )
    np.testing.assert_allclose(
        fft_result.values, python_result.values, rtol=3.0e-12, atol=3.0e-12
    )
    np.testing.assert_array_equal(numba_result.orders, [0.29, 0.64, 1.0])
    np.testing.assert_array_equal(numba_result.tempering, [0.0, 0.45, 2.3])


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_bdf1_tempered_rl_matches_existing_tempered_gl_operator(
    backend: str,
) -> None:
    rng = np.random.default_rng(8931)
    lower_terminal = -0.6
    times = lower_terminal + 0.017 * np.arange(83, dtype=np.float64)
    samples = rng.normal(size=(times.size, 2))
    orders = [0.34, 0.79]
    tempering = 0.57
    cq = tempered_convolution_quadrature(
        samples,
        orders,
        tempering=tempering,
        bdf_order=1,
        definition="tempered_riemann_liouville",
        times=times,
        lower_terminal=lower_terminal,
        initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend=backend,
    )
    gl = tempered_grunwald_letnikov_derivative(
        samples,
        times,
        orders,
        tempering=tempering,
        lower_terminal=lower_terminal,
        initial_condition_semantics=OPERATOR_ONLY_INITIAL_CONDITION,
        backend=backend,
    )
    np.testing.assert_allclose(cq.values, gl.values, rtol=3.0e-14, atol=3.0e-14)


@pytest.mark.parametrize("bdf_order", [1, 2])
@pytest.mark.parametrize("backend", ["python", "numba", "fft"])
def test_tempered_caputo_annihilates_its_exponential_anchor(
    bdf_order: int,
    backend: str,
) -> None:
    step = 0.031
    tempering = 0.83
    elapsed = step * np.arange(71, dtype=np.float64)
    samples = 2.75 * np.exp(-tempering * elapsed)
    result = tempered_convolution_quadrature(
        samples,
        0.61,
        tempering=tempering,
        bdf_order=bdf_order,
        definition="tempered_caputo",
        step=step,
        initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
        backend=backend,
    )
    np.testing.assert_allclose(result.values, 0.0, rtol=0.0, atol=2.0e-12)


@pytest.mark.parametrize("bdf_order", [1, 2])
def test_tempered_caputo_does_not_annihilate_a_physical_constant(
    bdf_order: int,
) -> None:
    result = tempered_convolution_quadrature(
        np.full(8, 2.0),
        1.0,
        tempering=0.9,
        bdf_order=bdf_order,
        definition="tempered_caputo",
        step=0.05,
        initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
        backend="python",
    )
    assert result.values[0] == 0.0
    assert result.values[1] > 0.0


def test_huge_finite_tempering_underflows_safely_without_positive_exponential() -> None:
    result = tempered_convolution_quadrature(
        np.linspace(0.0, 1.0, 9),
        0.55,
        tempering=np.finfo(np.float64).max,
        bdf_order=2,
        definition="tempered_caputo",
        step=0.25,
        lower_terminal=0.0,
        initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
        backend="fft",
    )

    assert np.all(np.isfinite(result.values))
    assert result.damping_underflowed is True
    assert result.positive_exponential_materialized is False
    np.testing.assert_array_equal(result.weights[1:, 0], 0.0)


@pytest.mark.parametrize(
    ("tempering", "error", "match"),
    [
        (True, TypeError, "Boolean"),
        (1.0 + 2.0j, TypeError, "complex"),
        ("0.4", TypeError, "numeric"),
        (np.nan, ValueError, "finite and non-negative"),
        (np.inf, ValueError, "finite and non-negative"),
        (-0.1, ValueError, "finite and non-negative"),
        ([0.1, 0.2, 0.3], ValueError, "one value or 2 values"),
        (np.array([[0.1, 0.2]]), ValueError, "one-dimensional"),
    ],
)
def test_tempering_validation_is_strict(
    tempering: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        tempered_convolution_quadrature(
            np.ones((5, 2)),
            [0.4, 0.8],
            tempering=tempering,  # type: ignore[arg-type]
            definition="tempered_riemann_liouville",
            step=0.1,
            initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        )


@pytest.mark.parametrize("orders", [True, [False, True], 0.4 + 0.2j, "0.5"])
def test_order_validation_rejects_boolean_complex_and_text(orders: object) -> None:
    with pytest.raises(TypeError, match="orders"):
        tempered_convolution_quadrature(
            np.arange(5.0),
            orders,  # type: ignore[arg-type]
            tempering=0.2,
            definition="tempered_riemann_liouville",
            step=0.1,
            initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        )


@pytest.mark.parametrize(
    ("samples", "match"),
    [
        (np.array([True, False]), "Boolean"),
        (np.array(["1.0", "2.0"]), "numeric"),
        (np.array([1.0 + 2.0j, 3.0 + 4.0j]), "complex"),
    ],
)
def test_samples_are_strictly_real_numeric(samples: np.ndarray, match: str) -> None:
    with pytest.raises(TypeError, match=match):
        tempered_convolution_quadrature(
            samples,
            0.5,
            tempering=0.2,
            definition="tempered_riemann_liouville",
            step=0.1,
            initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        )


def test_component_parameters_must_be_scalar_or_one_dimensional() -> None:
    with pytest.raises(ValueError, match="orders.*one-dimensional"):
        tempered_convolution_quadrature(
            np.ones((5, 2)),
            np.array([[0.4, 0.8]]),
            tempering=[0.2, 0.3],
            definition="tempered_riemann_liouville",
            step=0.1,
            initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        )


def test_definition_and_initial_condition_tokens_are_not_interchangeable() -> None:
    base = dict(samples=np.arange(5.0), orders=0.5, tempering=0.2, step=0.1)
    with pytest.raises(ValueError, match="definition"):
        tempered_convolution_quadrature(
            **base,
            definition="caputo",
            initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
        )
    with pytest.raises(ValueError, match="requires"):
        tempered_convolution_quadrature(
            **base,
            definition="tempered_caputo",
            initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        )
    with pytest.raises(ValueError, match="requires"):
        tempered_convolution_quadrature(
            **base,
            definition="tempered_riemann_liouville",
            initial_condition_semantics=RL_OPERATOR_ONLY_INITIAL_CONDITION,
        )


def test_uniform_grid_lower_terminal_and_backend_validation_are_strict() -> None:
    base = dict(
        samples=np.arange(4.0),
        orders=0.5,
        tempering=0.2,
        definition="tempered_riemann_liouville",
        lower_terminal=1.0,
        initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
    )
    with pytest.raises(ValueError, match="exactly one"):
        tempered_convolution_quadrature(
            **base, step=0.1, times=np.array([1.0, 1.1, 1.2, 1.3])
        )
    with pytest.raises(ValueError, match="uniform"):
        tempered_convolution_quadrature(
            **base, times=np.array([1.0, 1.1, 1.21, 1.3])
        )
    with pytest.raises(ValueError, match=r"times\[0\]"):
        tempered_convolution_quadrature(
            **base, times=np.array([0.0, 0.1, 0.2, 0.3])
        )
    with pytest.raises(TypeError, match="lower_terminal"):
        tempered_convolution_quadrature(
            **{**base, "lower_terminal": True}, step=0.1
        )
    with pytest.raises(TypeError, match="step"):
        tempered_convolution_quadrature(**base, step=True)
    with pytest.raises(ValueError, match="backend"):
        tempered_convolution_quadrature(**base, step=0.1, backend="auto")


def test_scale_and_history_overflow_raise_clear_errors() -> None:
    base = dict(
        orders=1.0,
        tempering=0.3,
        definition="tempered_riemann_liouville",
        lower_terminal=0.0,
        initial_condition_semantics=TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend="python",
    )
    with pytest.raises(ValueError, match="overflows float64"):
        tempered_convolution_quadrature(
            np.array([0.0, 1.0]),
            **base,
            step=np.nextafter(0.0, 1.0),
        )
    with pytest.raises(ValueError, match="convolution overflowed"):
        tempered_convolution_quadrature(
            np.full(4, np.finfo(np.float64).max),
            **base,
            bdf_order=2,
            step=0.1,
        )


def test_result_records_operator_scope_formula_cost_and_sources() -> None:
    result = tempered_convolution_quadrature(
        np.arange(32.0),
        0.5,
        tempering=0.7,
        bdf_order=2,
        definition="tempered_caputo",
        step=0.04,
        lower_terminal=3.0,
        initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
        backend="fft",
    )

    assert result.scope == "sampled_fractional_operator_only_not_an_fde_solver"
    assert result.status == "finite_numerical_diagnostic"
    assert "exp(-lambda*h)" in result.generating_formula
    assert result.tempering_convention == "unnormalized_exponential_conjugation"
    assert "no_minus_lambda_power_q_times_x" in result.normalization_correction
    assert "not_(delta(z)/h+lambda)^q" in result.normalization_correction
    assert "exp(-lambda*n*h)" in result.caputo_initial_correction
    assert "log(N)" in result.time_complexity
    assert result.starting_corrections == "none_implemented"
    assert result.startup_convention.startswith("terminal_truncated")
    assert "https://doi.org/10.1016/j.jcp.2014.04.024" in result.references
    assert "https://doi.org/10.1137/0517050" in result.references
    assert "https://doi.org/10.1051/m2an/2014037" in result.references
    assert "https://doi.org/10.1137/18M1230153" in result.references
