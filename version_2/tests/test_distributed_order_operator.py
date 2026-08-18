from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.fractional.distributed_order import (
    DistributedOrderDerivativeResult,
    distributed_order_gl_derivative,
)
from hidden_attractors.fractional.grunwald_letnikov import (
    grunwald_letnikov_derivative,
)
from hidden_attractors.fractional.contracts import (
    get_fractional_derivative,
    get_fractional_method,
)


def _samples(count: int = 48) -> tuple[float, np.ndarray, np.ndarray]:
    step = 0.025
    times = np.arange(count, dtype=np.float64) * step
    values = 0.4 + times + 0.3 * times**2
    return step, times, values


@pytest.mark.scientific_contract
def test_delta_order_reproduces_public_gl_operator_exactly() -> None:
    step, _, values = _samples()
    distributed = distributed_order_gl_derivative(
        values,
        step,
        [0.63],
        [1.0],
        definition="riemann_liouville_gl",
        backend="numba",
    )
    single = grunwald_letnikov_derivative(
        values,
        step,
        0.63,
        definition="riemann_liouville_gl",
    )

    assert np.array_equal(distributed.values, single.values)
    assert distributed.mass == 1.0
    assert distributed.approximation == (
        "double_discretization_order_quadrature_and_time_gl"
    )


@pytest.mark.scientific_contract
def test_linear_combination_matches_sum_of_public_gl_operators() -> None:
    step, _, values = _samples()
    nodes = np.array([0.25, 0.7, 1.0])
    masses = np.array([0.2, 0.5, 0.3])
    result = distributed_order_gl_derivative(
        values,
        step,
        nodes,
        masses,
        definition="caputo_shifted",
        backend="python",
    )
    expected = sum(
        mass
        * grunwald_letnikov_derivative(
            values,
            step,
            order,
            definition="caputo_shifted",
        ).values
        for order, mass in zip(nodes, masses, strict=True)
    )

    assert np.allclose(result.values, expected, rtol=2e-15, atol=2e-15)
    assert result.method == "distributed_order_gl_direct_python_reference"


@pytest.mark.scientific_contract
def test_caputo_shifted_distributed_operator_annihilates_constant() -> None:
    result = distributed_order_gl_derivative(
        np.full(35, 4.25),
        0.02,
        [0.1, 0.55, 1.0],
        [0.2, 0.3, 0.5],
        definition="caputo_shifted",
    )

    assert np.array_equal(result.values, np.zeros(35))


@pytest.mark.scientific_contract
def test_order_one_reduces_to_backward_difference_for_shifted_linear_data() -> None:
    step = 0.05
    times = np.arange(20, dtype=np.float64) * step
    result = distributed_order_gl_derivative(
        3.0 + 2.5 * times,
        step,
        [1.0],
        [1.0],
        definition="caputo_shifted",
    )

    assert result.values[0] == 0.0
    assert np.allclose(result.values[1:], 2.5, rtol=0.0, atol=2e-14)


@pytest.mark.scientific_contract
def test_multicomponent_output_and_metadata_are_structured() -> None:
    step, times, values = _samples()
    samples = np.column_stack((values, np.sin(times), times**3))
    result = distributed_order_gl_derivative(
        samples,
        step,
        [0.3, 0.8],
        [0.4, 0.6],
        lower_terminal=-1.25,
    )

    assert isinstance(result, DistributedOrderDerivativeResult)
    assert result.values.shape == samples.shape
    assert result.lower_terminal == -1.25
    assert result.grid_convention == "t_n=lower_terminal+n*step"
    assert result.memory_policy == "full_history"
    assert "no order-by-time derivative tensor" in result.working_memory


@pytest.mark.scientific_contract
def test_finite_window_matches_weighted_public_window_operators() -> None:
    step, _, values = _samples(60)
    nodes = [0.4, 0.9]
    masses = [0.35, 0.65]
    result = distributed_order_gl_derivative(
        values,
        step,
        nodes,
        masses,
        definition="grunwald_letnikov",
        history_window=7,
    )
    expected = sum(
        mass
        * grunwald_letnikov_derivative(
            values,
            step,
            order,
            definition="grunwald_letnikov",
            history_window=7,
        ).values
        for order, mass in zip(nodes, masses, strict=True)
    )

    assert np.allclose(result.values, expected, rtol=2e-15, atol=2e-15)
    assert result.memory_policy == "finite_window"
    assert result.history_window == 7
    assert "history_window" in result.complexity


@pytest.mark.scientific_contract
def test_quadrature_density_and_unit_mass_have_explicit_accounting() -> None:
    step, _, values = _samples()
    result = distributed_order_gl_derivative(
        values,
        step,
        [0.2, 0.6, 0.95],
        [0.1, 0.2, 0.1],
        density_values=[2.0, 3.0, 4.0],
        weight_semantics="nonnegative_quadrature_density",
        normalization="unit_mass",
    )

    assert result.raw_mass == pytest.approx(1.2)
    assert result.raw_l1_norm == pytest.approx(1.2)
    assert result.mass == pytest.approx(1.0)
    assert result.l1_norm == pytest.approx(1.0)
    assert np.allclose(result.effective_weights, [1 / 6, 1 / 2, 1 / 3])
    assert np.array_equal(result.density_values, [2.0, 3.0, 4.0])


@pytest.mark.scientific_contract
def test_signed_weights_require_declared_semantics_and_record_norm() -> None:
    step, _, values = _samples()
    with pytest.raises(ValueError, match="signed semantics"):
        distributed_order_gl_derivative(values, step, [0.3, 0.8], [1.0, -0.25])

    result = distributed_order_gl_derivative(
        values,
        step,
        [0.3, 0.8],
        [1.0, -0.25],
        weight_semantics="signed_mass",
    )
    assert result.raw_mass == pytest.approx(0.75)
    assert result.raw_l1_norm == pytest.approx(1.25)
    assert result.weight_semantics == "signed_mass"


@pytest.mark.scientific_contract
def test_signed_zero_mass_cannot_be_unit_mass_normalized() -> None:
    step, _, values = _samples()
    with pytest.raises(ValueError, match="weights cancel"):
        distributed_order_gl_derivative(
            values,
            step,
            [0.2, 0.9],
            [1.0, -1.0],
            weight_semantics="signed_mass",
            normalization="unit_mass",
        )


@pytest.mark.scientific_contract
@pytest.mark.parametrize(
    ("nodes", "weights", "message"),
    [
        ([0.0, 0.5], [0.5, 0.5], "lie in"),
        ([0.5, 1.01], [0.5, 0.5], "lie in"),
        ([0.5, np.nan], [0.5, 0.5], "finite"),
        ([0.5, 0.8], [1.0], "one value per"),
        ([0.5], [0.0], "non-zero mass norm"),
    ],
)
def test_order_nodes_and_weights_are_validated(nodes, weights, message) -> None:
    with pytest.raises(ValueError, match=message):
        distributed_order_gl_derivative(np.arange(6.0), 0.1, nodes, weights)


@pytest.mark.scientific_contract
def test_density_semantics_are_not_inferred() -> None:
    step, _, values = _samples()
    with pytest.raises(ValueError, match="required"):
        distributed_order_gl_derivative(
            values,
            step,
            [0.2, 0.7],
            [0.5, 0.5],
            weight_semantics="nonnegative_quadrature_density",
        )
    with pytest.raises(ValueError, match="not used"):
        distributed_order_gl_derivative(
            values,
            step,
            [0.2, 0.7],
            [0.5, 0.5],
            density_values=[1.0, 1.0],
            weight_semantics="nonnegative_mass",
        )


@pytest.mark.scientific_contract
def test_history_window_and_backend_validation() -> None:
    step, _, values = _samples()
    for invalid in (0, -1, 1.5, True, np.nan, np.inf, "not-a-window"):
        with pytest.raises(ValueError, match="positive integer"):
            distributed_order_gl_derivative(
                values,
                step,
                [0.5],
                [1.0],
                history_window=invalid,
            )
    with pytest.raises(ValueError, match="backend"):
        distributed_order_gl_derivative(
            values,
            step,
            [0.5],
            [1.0],
            backend="gpu",
        )


@pytest.mark.scientific_contract
@pytest.mark.parametrize("definition", ["grunwald_letnikov", "caputo_shifted"])
@pytest.mark.parametrize("history_window", [None, 9])
def test_numba_python_parity(definition: str, history_window: int | None) -> None:
    step, times, values = _samples(55)
    samples = np.column_stack((values, np.cos(1.7 * times)))
    arguments = dict(
        samples=samples,
        step=step,
        order_nodes=[0.15, 0.48, 0.83, 1.0],
        order_weights=[0.1, 0.25, 0.4, 0.25],
        definition=definition,
        history_window=history_window,
    )
    numba_result = distributed_order_gl_derivative(**arguments, backend="numba")
    python_result = distributed_order_gl_derivative(**arguments, backend="python")

    assert np.array_equal(numba_result.values, python_result.values)
    assert np.array_equal(numba_result.effective_weights, python_result.effective_weights)
    assert numba_result.backend == "numba"
    assert python_result.backend == "python"


def test_distributed_operator_is_public_but_not_registered_as_an_fde_solver() -> None:
    from hidden_attractors import fractional

    derivative = get_fractional_derivative("distributed_order")
    method = get_fractional_method("distributed_order_gl_direct")
    assert derivative.implementation_status == "implemented"
    assert method.execution_kind == "sampled_operator"
    assert fractional.distributed_order_gl_derivative is distributed_order_gl_derivative
