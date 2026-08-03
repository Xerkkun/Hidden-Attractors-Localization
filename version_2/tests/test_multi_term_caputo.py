"""Contract tests for the semantic multi-term Caputo L1 facade."""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np
import pytest

import hidden_attractors.fractional.multi_term_caputo as multi_term_module
from hidden_attractors.fractional.distributed_order_caputo_solver import (
    integrate_distributed_order_caputo_l1,
)
from hidden_attractors.fractional.multi_term_caputo import (
    MULTI_TERM_CAPUTO_L1_REFERENCES,
    MULTI_TERM_CAPUTO_SCISPACE_PAPER_IDS,
    MultiTermCaputoResult,
    canonicalize_multi_term_caputo_terms,
    integrate_multi_term_caputo_l1,
)


def _affine_rhs(
    time: float,
    state: np.ndarray,
    parameters: tuple[float, float],
) -> np.ndarray:
    del time
    slope, offset = parameters
    return slope * state + offset


def _facade_arguments(**updates: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "rhs": _affine_rhs,
        "initial_state": np.array([0.25, -0.1]),
        "parameters": (-0.2, 0.05),
        "orders": [0.8, 0.3, 0.3, 0.6],
        "coefficients": [0.6, 0.15, 0.25, 0.0],
        "step": 0.02,
        "n_steps": 12,
        "initial_regularity": "nonsmooth",
        "use_acceleration": False,
        "divergence_norm": None,
    }
    arguments.update(updates)
    return arguments


def test_canonicalization_preserves_provenance_without_mutating_inputs() -> None:
    orders = np.array([0.8, 0.3, 0.3, 0.6])
    coefficients = np.array([0.6, 0.15, 0.25, 0.0])
    original_orders = orders.copy()
    original_coefficients = coefficients.copy()

    terms = canonicalize_multi_term_caputo_terms(orders, coefficients)

    np.testing.assert_array_equal(orders, original_orders)
    np.testing.assert_array_equal(coefficients, original_coefficients)
    np.testing.assert_array_equal(terms.original_orders, original_orders)
    np.testing.assert_array_equal(terms.original_coefficients, original_coefficients)
    np.testing.assert_array_equal(terms.orders, [0.3, 0.8])
    np.testing.assert_array_equal(terms.coefficients, [0.4, 0.6])
    assert terms.source_indices == ((1, 2), (0,))
    assert terms.zero_coefficient_indices == (3,)
    assert terms.original_term_count == 4
    assert terms.term_count == 2
    assert terms.zero_terms_removed == 1
    assert terms.duplicate_terms_coalesced == 1
    assert terms.coefficient_sum == 1.0
    assert not terms.orders.flags.writeable
    assert not terms.coefficients.flags.writeable


def test_coefficient_accumulation_uses_fsum_and_is_not_normalized() -> None:
    terms = canonicalize_multi_term_caputo_terms(
        [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0],
        [0.4, 0.2, 0.5, 0.75],
    )

    np.testing.assert_array_equal(
        terms.coefficients,
        [0.4, math.fsum([0.2, 0.5]), 0.75],
    )
    assert terms.coefficient_sum == math.fsum([0.4, 0.2, 0.5, 0.75])
    assert terms.coefficient_sum == pytest.approx(1.85)
    assert terms.normalization == "none"


def test_nearby_orders_are_not_coalesced_by_a_tolerance() -> None:
    order = 0.4
    next_order = np.nextafter(order, 1.0)
    terms = canonicalize_multi_term_caputo_terms(
        [next_order, order],
        [0.25, 0.75],
    )

    assert terms.term_count == 2
    assert terms.duplicate_terms_coalesced == 0
    assert terms.orders[0] == order
    assert terms.orders[1] == next_order


@pytest.mark.parametrize(
    ("orders", "coefficients", "error", "match"),
    [
        ([0.3], [0.5, 0.5], ValueError, "one value per order"),
        ([], [], ValueError, "non-empty"),
        ([[0.3]], [[1.0]], ValueError, "one-dimensional"),
        ([0.0], [1.0], ValueError, r"\(0, 1\]"),
        ([1.01], [1.0], ValueError, r"\(0, 1\]"),
        ([np.nan], [1.0], ValueError, "finite"),
        ([0.3], [np.inf], ValueError, "finite"),
        ([0.3], [-0.1], ValueError, "nonnegative"),
        ([0.3], [0.0], ValueError, "positive"),
        ([True], [1.0], TypeError, "non-Boolean"),
        ([0.3], [True], TypeError, "non-Boolean"),
        ([0.3 + 0.1j], [1.0], TypeError, "real"),
        ([0.3], [1.0 + 0.1j], TypeError, "real"),
        ([0.3, 0.8], [1.0e308, 1.0e308], ValueError, "overflows"),
    ],
)
def test_invalid_term_contracts_are_rejected(
    orders: object,
    coefficients: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        canonicalize_multi_term_caputo_terms(orders, coefficients)


def test_zero_coefficient_policy_is_explicit() -> None:
    with pytest.raises(ValueError, match="zero_coefficient_policy='raise'"):
        canonicalize_multi_term_caputo_terms(
            [0.3, 0.8],
            [0.0, 1.0],
            zero_coefficient_policy="raise",
        )
    with pytest.raises(ValueError, match="drop.*raise"):
        canonicalize_multi_term_caputo_terms(
            [0.3],
            [1.0],
            zero_coefficient_policy="keep",
        )


def test_facade_is_exactly_the_canonical_distributed_order_call() -> None:
    facade = integrate_multi_term_caputo_l1(**_facade_arguments())
    direct = integrate_distributed_order_caputo_l1(
        _affine_rhs,
        np.array([0.25, -0.1]),
        (-0.2, 0.05),
        order_nodes=[0.3, 0.8],
        order_weights=[0.4, 0.6],
        step=0.02,
        n_steps=12,
        lower_terminal=0.0,
        weight_semantics="nonnegative_mass",
        normalization="none",
        order_quadrature_name="finite_atomic_multi_term_caputo_equation",
        initial_regularity="nonsmooth",
        use_acceleration=False,
        divergence_norm=None,
    )

    np.testing.assert_array_equal(facade.times, direct.times)
    np.testing.assert_array_equal(facade.states, direct.states)
    np.testing.assert_array_equal(
        facade.combined_l1_kernel,
        direct.combined_l1_kernel,
    )
    assert facade.distributed_result.method == "distributed_order_caputo_l1"
    assert facade.method == "multi_term_caputo_l1"


def test_facade_calls_the_existing_solver_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    original = multi_term_module.integrate_distributed_order_caputo_l1

    def spy(*args: object, **kwargs: object):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        multi_term_module,
        "integrate_distributed_order_caputo_l1",
        spy,
    )
    result = integrate_multi_term_caputo_l1(**_facade_arguments(n_steps=2))

    assert result.status == "ok"
    assert len(calls) == 1
    call = calls[0]
    np.testing.assert_array_equal(call["order_nodes"], [0.3, 0.8])
    np.testing.assert_array_equal(call["order_weights"], [0.4, 0.6])
    assert call["weight_semantics"] == "nonnegative_mass"
    assert call["density_values"] is None
    assert call["normalization"] == "none"
    assert call["order_quadrature_name"] == (
        "finite_atomic_multi_term_caputo_equation"
    )


def test_nonunit_coefficients_remain_equation_coefficients() -> None:
    result = integrate_multi_term_caputo_l1(
        **_facade_arguments(
            orders=[1.0 / 3.0, 2.0 / 3.0, 1.0],
            coefficients=[0.4, 0.7, 0.75],
        )
    )

    np.testing.assert_array_equal(result.coefficients, [0.4, 0.7, 0.75])
    np.testing.assert_array_equal(
        result.distributed_result.effective_weights,
        [0.4, 0.7, 0.75],
    )
    assert result.solver_info["coefficient_sum"] == pytest.approx(1.85)
    assert result.solver_info["coefficient_normalization"] == "none"
    assert result.solver_info["continuous_order_quadrature_used"] is False
    assert result.solver_info["continuous_order_density_inferred"] is False


def test_alpha_one_term_uses_the_exact_backward_euler_branch() -> None:
    coefficient = 2.5
    step = 0.1
    n_steps = 8

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        del time
        return -state

    result = integrate_multi_term_caputo_l1(
        rhs,
        [1.0],
        orders=[1.0, 1.0],
        coefficients=[1.0, 1.5],
        step=step,
        n_steps=n_steps,
        initial_regularity="smooth",
        use_acceleration=False,
        divergence_norm=None,
    )
    expected = np.power(1.0 + step / coefficient, -np.arange(n_steps + 1))

    np.testing.assert_allclose(result.states[:, 0], expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(result.orders, [1.0])
    np.testing.assert_array_equal(result.coefficients, [coefficient])
    assert result.solver_info["alpha_one_handling"] == "exact_backward_euler_limit"


def test_numba_and_python_paths_agree_after_canonicalization() -> None:
    python_result = integrate_multi_term_caputo_l1(**_facade_arguments())
    numba_result = integrate_multi_term_caputo_l1(
        **_facade_arguments(use_acceleration=True)
    )

    np.testing.assert_allclose(
        numba_result.states,
        python_result.states,
        rtol=5e-14,
        atol=5e-14,
    )
    np.testing.assert_allclose(
        numba_result.combined_l1_kernel,
        python_result.combined_l1_kernel,
        rtol=2e-15,
        atol=2e-15,
    )
    assert numba_result.solver_info["numba_requested"] is True
    assert python_result.solver_info["numba_requested"] is False


def test_result_metadata_is_frozen_and_semantically_distinct() -> None:
    result = integrate_multi_term_caputo_l1(**_facade_arguments(n_steps=2))

    assert isinstance(result, MultiTermCaputoResult)
    assert isinstance(result.solver_info, MappingProxyType)
    assert result.definition == "caputo_multi_term_finite_sum"
    assert result.measure_kind == "finite_discrete_atomic_order_measure"
    assert result.normalization == "none"
    assert result.scope == "finite_numerical_trajectory_only"
    assert result.solver_info["implementation_reuse"] == (
        "distributed_order_combined_l1_kernel_without_solver_reconstruction"
    )
    assert result.solver_info["underlying_definition"] == (
        "caputo_distributed_order_discrete_measure"
    )
    assert result.solver_info["underlying_method"] == (
        "distributed_order_caputo_l1"
    )
    assert result.solver_info["original_term_count"] == 4
    assert result.solver_info["canonical_term_count"] == 2
    assert result.solver_info["zero_terms_removed"] == 1
    assert result.solver_info["duplicate_terms_coalesced"] == 1
    with pytest.raises(TypeError):
        result.solver_info["definition"] = "changed"  # type: ignore[index]


def test_scientific_evidence_identifiers_are_embedded_in_result() -> None:
    result = integrate_multi_term_caputo_l1(**_facade_arguments(n_steps=2))

    assert len(MULTI_TERM_CAPUTO_L1_REFERENCES) == 4
    assert all(reference.startswith("https://doi.org/") for reference in result.references)
    assert set(result.solver_info["scispace_paper_ids"]) == set(
        MULTI_TERM_CAPUTO_SCISPACE_PAPER_IDS
    )
    assert set(result.solver_info["reference_urls"]) == set(
        MULTI_TERM_CAPUTO_L1_REFERENCES
    )
