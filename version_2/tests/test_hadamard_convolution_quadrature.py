from __future__ import annotations

from math import gamma

import numpy as np
import pytest

from hidden_attractors.fractional import (
    CAPUTO_HADAMARD_INITIAL_CONDITION,
    HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION,
    FractionalProblem,
    get_fractional_derivative,
    get_fractional_method,
    hadamard_convolution_quadrature,
)
from hidden_attractors.fractional.convolution_quadrature import (
    CAPUTO_SHIFTED_INITIAL_CONDITION,
    RL_OPERATOR_ONLY_INITIAL_CONDITION,
    lubich_convolution_quadrature,
)


def test_registry_keeps_hadamard_families_and_sampled_method_explicit() -> None:
    raw = get_fractional_derivative("hadamard_riemann_liouville")
    shifted = get_fractional_derivative("caputo_hadamard")
    method = get_fractional_method("hadamard_convolution_quadrature")
    assert raw.kernel_family == "singular_logarithmic_power_law"
    assert shifted.kernel_family == "singular_logarithmic_power_law"
    assert method.derivative_families == (
        "hadamard_riemann_liouville",
        "caputo_hadamard",
    )
    assert method.execution_kind == "sampled_operator"
    assert method.implementation_status == "experimental"


@pytest.mark.parametrize("definition", ["hadamard_riemann_liouville", "caputo_hadamard"])
@pytest.mark.parametrize("bdf_order", [1, 2])
@pytest.mark.parametrize("backend", ["python", "numba", "fft"])
def test_logarithmic_transformation_matches_canonical_cq(
    definition: str,
    bdf_order: int,
    backend: str,
) -> None:
    rng = np.random.default_rng(20260803)
    samples = rng.normal(size=(61, 2))
    semantics = (
        HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION
        if definition == "hadamard_riemann_liouville"
        else CAPUTO_HADAMARD_INITIAL_CONDITION
    )
    result = hadamard_convolution_quadrature(
        samples,
        [0.37, 0.81],
        bdf_order=bdf_order,
        definition=definition,
        log_step=0.025,
        lower_terminal=2.5,
        initial_condition_semantics=semantics,
        backend=backend,
    )
    reference = lubich_convolution_quadrature(
        samples,
        [0.37, 0.81],
        bdf_order=bdf_order,
        definition=(
            "riemann_liouville"
            if definition == "hadamard_riemann_liouville"
            else "caputo_shifted"
        ),
        times=result.log_times,
        lower_terminal=0.0,
        initial_condition_semantics=(
            RL_OPERATOR_ONLY_INITIAL_CONDITION
            if definition == "hadamard_riemann_liouville"
            else CAPUTO_SHIFTED_INITIAL_CONDITION
        ),
        backend=backend,
    )
    np.testing.assert_allclose(result.values, reference.values, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(result.weights, reference.weights)
    np.testing.assert_allclose(
        result.times,
        2.5 * np.exp(result.log_times),
        rtol=3.0e-15,
        atol=0.0,
    )


@pytest.mark.parametrize("backend", ["python", "numba", "fft"])
@pytest.mark.parametrize("bdf_order", [1, 2])
def test_caputo_hadamard_constant_is_exactly_zero(
    backend: str,
    bdf_order: int,
) -> None:
    samples = np.full((37, 2), [4.0, -3.0])
    result = hadamard_convolution_quadrature(
        samples,
        [0.25, 0.9],
        bdf_order=bdf_order,
        definition="caputo_hadamard",
        log_step=0.04,
        lower_terminal=1.2,
        initial_condition_semantics=CAPUTO_HADAMARD_INITIAL_CONDITION,
        backend=backend,
    )
    np.testing.assert_array_equal(result.values, np.zeros_like(samples))


@pytest.mark.parametrize(("bdf_order", "minimum_ratio"), [(1, 1.7), (2, 3.2)])
def test_log_power_endpoint_converges_at_the_expected_bdf_rate(
    bdf_order: int,
    minimum_ratio: float,
) -> None:
    order = 0.62
    power = 3
    exact = gamma(power + 1) / gamma(power + 1 - order)
    errors: list[float] = []
    for intervals in (80, 160, 320):
        log_times = np.linspace(0.0, 1.0, intervals + 1)
        physical_times = 1.7 * np.exp(log_times)
        result = hadamard_convolution_quadrature(
            log_times**power,
            order,
            bdf_order=bdf_order,
            definition="caputo_hadamard",
            times=physical_times,
            lower_terminal=1.7,
            initial_condition_semantics=CAPUTO_HADAMARD_INITIAL_CONDITION,
            backend="numba",
        )
        errors.append(abs(float(result.values[-1]) - exact))
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[1] > minimum_ratio
    assert errors[1] / errors[2] > minimum_ratio


def test_raw_hadamard_constant_approximates_analytic_endpoint() -> None:
    order = 0.43
    result = hadamard_convolution_quadrature(
        np.ones(2049),
        order,
        definition="hadamard_riemann_liouville",
        log_step=1.0 / 2048.0,
        lower_terminal=0.75,
        initial_condition_semantics=HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend="fft",
    )
    expected = 1.0 / gamma(1.0 - order)
    assert result.values[-1] == pytest.approx(expected, rel=8.0e-4)
    assert result.values[0] > 0.0


def test_integer_limit_is_the_dilation_derivative_in_log_time() -> None:
    log_step = 0.1
    u = log_step * np.arange(8, dtype=float)
    samples = 3.0 + u**2
    result = hadamard_convolution_quadrature(
        samples,
        1.0,
        definition="caputo_hadamard",
        log_step=log_step,
        lower_terminal=2.0,
        initial_condition_semantics=CAPUTO_HADAMARD_INITIAL_CONDITION,
        backend="python",
    )
    shifted = samples - samples[0]
    expected = np.r_[0.0, np.diff(shifted) / log_step]
    np.testing.assert_allclose(result.values, expected, rtol=0.0, atol=1.0e-14)


def test_result_reports_physical_grid_cost_scope_and_sources() -> None:
    result = hadamard_convolution_quadrature(
        np.arange(16.0),
        0.5,
        bdf_order=2,
        definition="hadamard_riemann_liouville",
        log_step=0.03,
        lower_terminal=4.0,
        initial_condition_semantics=HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION,
        backend="fft",
    )
    assert result.times[0] == 4.0
    assert result.log_times[0] == 0.0
    assert result.grid_kind == "exponential_uniform_in_log_t_over_a"
    assert "t*d/dt" in result.transformation
    assert "log(N)" in result.time_complexity
    assert result.scope == "sampled_fractional_operator_only_not_an_fde_solver"
    assert result.starting_corrections == "none_implemented"
    assert "https://doi.org/10.1186/1687-1847-2012-142" in result.references
    assert "https://doi.org/10.1016/j.cnsns.2024.108221" in result.references


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"lower_terminal": 0.0}, ValueError, "lower_terminal > 0"),
        ({"lower_terminal": -1.0}, ValueError, "lower_terminal > 0"),
        ({"lower_terminal": True}, TypeError, "positive real"),
        ({"log_step": 0.0}, ValueError, "log_step"),
        ({"log_step": True}, TypeError, "log_step"),
        ({"definition": "caputo"}, ValueError, "definition"),
        (
            {"initial_condition_semantics": CAPUTO_HADAMARD_INITIAL_CONDITION},
            ValueError,
            "requires",
        ),
    ],
)
def test_invalid_contract_values_are_rejected(
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "samples": np.arange(5.0),
        "orders": 0.5,
        "definition": "hadamard_riemann_liouville",
        "log_step": 0.1,
        "lower_terminal": 1.0,
        "initial_condition_semantics": HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION,
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=match):
        hadamard_convolution_quadrature(**arguments)  # type: ignore[arg-type]


def test_physical_grid_validation_is_strict() -> None:
    base = dict(
        samples=np.arange(4.0),
        orders=0.5,
        definition="hadamard_riemann_liouville",
        lower_terminal=2.0,
        initial_condition_semantics=HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION,
    )
    with pytest.raises(ValueError, match="exactly one"):
        hadamard_convolution_quadrature(
            **base,
            log_step=0.1,
            times=2.0 * np.exp(0.1 * np.arange(4)),
        )
    with pytest.raises(ValueError, match="uniform in log"):
        hadamard_convolution_quadrature(
            **base,
            times=2.0 * np.exp([0.0, 0.1, 0.22, 0.3]),
        )
    with pytest.raises(ValueError, match=r"times\[0\]"):
        hadamard_convolution_quadrature(
            **base,
            times=3.0 * np.exp(0.1 * np.arange(4)),
        )
    with pytest.raises(ValueError, match="finite, strictly increasing"):
        hadamard_convolution_quadrature(**base, log_step=1000.0)
    with pytest.raises(ValueError, match="finite, strictly increasing"):
        hadamard_convolution_quadrature(**base, log_step=1.0e-20)


def test_fractional_problem_records_but_does_not_solve_hadamard_cq() -> None:
    problem = FractionalProblem(
        "caputo_hadamard",
        "hadamard_convolution_quadrature",
        0.7,
        [1.0],
        0.1,
        (1.0, 2.0),
        allow_experimental=True,
    )
    assert problem.initial_condition_kind == "classical"
    assert "yin_zhang_liu_li2024" in problem.reference_keys
    with pytest.raises(NotImplementedError, match="sampled operator"):
        problem.validate_executable()
    with pytest.raises(ValueError, match="lower_terminal > 0"):
        FractionalProblem(
            "caputo_hadamard",
            "hadamard_convolution_quadrature",
            0.7,
            [1.0],
            0.1,
            (0.0, 1.0),
            allow_experimental=True,
        )
