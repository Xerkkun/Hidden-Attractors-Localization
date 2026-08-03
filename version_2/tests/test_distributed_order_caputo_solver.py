from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from scipy.special import gamma

import hidden_attractors.fractional.distributed_order_caputo_solver as solver_module
from hidden_attractors.fractional import FractionalProblem, solve_fractional_problem
from hidden_attractors.fractional.contracts import (
    get_fractional_derivative,
    get_fractional_method,
)
from hidden_attractors.fractional.distributed_order_caputo_solver import (
    DistributedOrderCaputoResult,
    DistributedOrderCorrectorError,
    DistributedOrderInitialCompatibilityWarning,
    distributed_order_l1_weight,
    integrate_distributed_order_caputo_l1,
)
from hidden_attractors.fractional.variable_order_caputo_type3 import (
    integrate_variable_order_caputo_type3_l1,
)


def _zero_rhs(time: float, state: np.ndarray) -> np.ndarray:
    del time
    return np.zeros_like(state)


def _constant_rhs(
    time: float,
    state: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    del time
    return np.full_like(state, parameters[0])


def _scalar_affine_rhs(
    time: float,
    state: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    del time
    forcing, damping = parameters
    return forcing - damping * state


def _matrix_affine_rhs(
    time: float,
    state: np.ndarray,
    parameters: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    del time
    matrix, forcing = parameters
    return matrix @ state + forcing


def _solver_arguments(**updates: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "rhs": _scalar_affine_rhs,
        "initial_state": [0.2],
        "parameters": np.array([0.4, 0.3]),
        "order_nodes": [0.25, 0.63, 0.9],
        "order_weights": [0.2, 0.5, 0.3],
        "step": 0.02,
        "n_steps": 12,
        "lower_terminal": 0.0,
        "weight_semantics": "nonnegative_mass",
        "normalization": "none",
        "order_quadrature_name": "three-node-test-rule",
        "corrector_atol": 1.0e-13,
        "corrector_rtol": 1.0e-12,
        "corrector_max_iterations": 100,
        "on_nonconvergence": "raise",
        "initial_regularity": "nonsmooth",
        "compatibility_tolerance": 1.0e-12,
        "use_acceleration": False,
        "divergence_norm": None,
    }
    arguments.update(updates)
    return arguments


def _independent_l1_weight(order: float, lag: int) -> float:
    if lag == 0:
        return 1.0
    if order == 1.0:
        return 0.0
    exponent = 1.0 - order
    return float((lag + 1.0) ** exponent - lag**exponent)


def _independent_combined_kernel(
    order_nodes: np.ndarray,
    effective_weights: np.ndarray,
    step: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = (
        effective_weights
        * np.power(step, -order_nodes)
        / gamma(2.0 - order_nodes)
    )
    kernel = np.zeros(n_steps, dtype=np.float64)
    for order, coefficient in zip(order_nodes, coefficients, strict=True):
        for lag in range(n_steps):
            kernel[lag] += float(coefficient) * _independent_l1_weight(
                float(order), lag
            )
    return coefficients, kernel


def _independent_linear_l1_reference(
    *,
    initial_state: np.ndarray,
    matrix: np.ndarray,
    forcing: Callable[[float], np.ndarray],
    lower_terminal: float,
    step: float,
    n_steps: int,
    order_nodes: np.ndarray,
    effective_weights: np.ndarray,
) -> np.ndarray:
    """Independent direct L1 recurrence for f(t, x) = M x + g(t)."""

    _, kernel = _independent_combined_kernel(
        order_nodes,
        effective_weights,
        step,
        n_steps,
    )
    current_coefficient = float(kernel[0])
    dimension = initial_state.size
    states = np.empty((n_steps + 1, dimension), dtype=np.float64)
    states[0] = initial_state
    system_matrix = current_coefficient * np.eye(dimension) - matrix
    for output_index in range(1, n_steps + 1):
        history = np.zeros(dimension, dtype=np.float64)
        for history_index in range(output_index - 1):
            lag = output_index - history_index - 1
            history += kernel[lag] * (
                states[history_index + 1] - states[history_index]
            )
        time = lower_terminal + output_index * step
        right_hand_side = (
            current_coefficient * states[output_index - 1]
            - history
            + forcing(time)
        )
        states[output_index] = np.linalg.solve(system_matrix, right_hand_side)
    return states


@pytest.mark.scientific_contract
def test_combined_kernel_matches_independent_sum_over_orders() -> None:
    nodes = np.array([0.21, 0.63, 0.94])
    masses = np.array([0.15, 0.55, 0.30])
    result = integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            rhs=_zero_rhs,
            parameters=None,
            initial_state=[0.7, -0.2],
            order_nodes=nodes,
            order_weights=masses,
            n_steps=16,
            use_acceleration=False,
        )
    )
    expected_coefficients, expected_kernel = _independent_combined_kernel(
        nodes,
        masses,
        0.02,
        16,
    )

    np.testing.assert_allclose(
        result.l1_coefficients,
        expected_coefficients,
        rtol=3.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.combined_l1_kernel,
        expected_kernel,
        rtol=3.0e-15,
        atol=3.0e-15,
    )
    assert result.combined_l1_kernel[0] == pytest.approx(
        np.sum(expected_coefficients), rel=3.0e-15
    )
    assert result.solver_info["total_structural_complexity"] == (
        "O(R*N + N^2*d)"
    )
    assert result.solver_info["naive_order_time_state_tensor_avoided"] is True


@pytest.mark.scientific_contract
def test_single_order_reduces_to_independent_implicit_l1_recurrence() -> None:
    order = 0.63
    step = 0.02
    n_steps = 20
    initial = np.array([0.2])
    forcing = 0.4
    damping = 0.3
    result = integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            initial_state=initial,
            parameters=np.array([forcing, damping]),
            order_nodes=[order],
            order_weights=[1.0],
            step=step,
            n_steps=n_steps,
        )
    )
    expected = _independent_linear_l1_reference(
        initial_state=initial,
        matrix=np.array([[-damping]]),
        forcing=lambda time: np.array([forcing]),
        lower_terminal=0.0,
        step=step,
        n_steps=n_steps,
        order_nodes=np.array([order]),
        effective_weights=np.array([1.0]),
    )

    assert result.status == "ok"
    np.testing.assert_allclose(result.states, expected, rtol=2.0e-11, atol=2.0e-13)


@pytest.mark.scientific_contract
def test_single_order_matches_constant_type3_l1_public_solver() -> None:
    order = 0.58
    parameters = np.array([0.35, 0.2])
    distributed = integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            parameters=parameters,
            order_nodes=[order],
            order_weights=[1.0],
            step=0.015,
            n_steps=18,
        )
    )
    variable = integrate_variable_order_caputo_type3_l1(
        _scalar_affine_rhs,
        [0.2],
        parameters,
        step=0.015,
        n_steps=18,
        order_function=lambda time: order,
        order_function_name="constant-alpha-0.58",
        corrector_atol=1.0e-13,
        corrector_rtol=1.0e-12,
        corrector_max_iterations=100,
        initial_regularity="nonsmooth",
        use_acceleration=False,
        divergence_norm=None,
    )

    np.testing.assert_array_equal(distributed.times, variable.times)
    np.testing.assert_allclose(
        distributed.states,
        variable.states,
        rtol=3.0e-12,
        atol=3.0e-13,
    )


@pytest.mark.scientific_contract
def test_order_one_is_exact_implicit_backward_euler_limit() -> None:
    step = 0.05
    n_steps = 15
    initial = 0.2
    forcing = 0.4
    damping = 0.3
    result = integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            initial_state=[initial],
            parameters=np.array([forcing, damping]),
            order_nodes=[1.0],
            order_weights=[1.0],
            step=step,
            n_steps=n_steps,
        )
    )
    expected = np.empty(n_steps + 1)
    expected[0] = initial
    for index in range(1, n_steps + 1):
        expected[index] = (expected[index - 1] + step * forcing) / (
            1.0 + step * damping
        )

    np.testing.assert_allclose(result.states[:, 0], expected, rtol=3.0e-12, atol=3.0e-13)
    np.testing.assert_array_equal(
        result.combined_l1_kernel,
        np.concatenate(([1.0 / step], np.zeros(n_steps - 1))),
    )
    assert distributed_order_l1_weight(1.0, 0) == 1.0
    assert distributed_order_l1_weight(1.0, 1) == 0.0
    assert result.solver_info["alpha_one_handling"] == "exact_backward_euler_limit"


@pytest.mark.scientific_contract
def test_zero_rhs_preserves_multicomponent_state_exactly() -> None:
    initial = np.array([0.7, -0.2, 1.1])
    result = integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            rhs=_zero_rhs,
            parameters=None,
            initial_state=initial,
            n_steps=14,
            initial_regularity="smooth",
        )
    )

    np.testing.assert_array_equal(
        result.states,
        np.repeat(initial[None, :], 15, axis=0),
    )
    assert result.status == "ok"


@pytest.mark.scientific_contract
def test_multinode_quadratic_manufactured_solution() -> None:
    lower = 0.4
    nodes = np.array([0.25, 0.6, 0.9])
    masses = np.array([0.2, 0.5, 0.3])
    initial = np.array([1.25, -0.3])
    amplitude = np.array([1.0, 0.4])

    def manufactured_rhs(time: float, state: np.ndarray) -> np.ndarray:
        del state
        tau = max(0.0, time - lower)
        distributed_derivative = sum(
            mass
            * gamma(3.0)
            / gamma(3.0 - order)
            * tau ** (2.0 - order)
            for order, mass in zip(nodes, masses, strict=True)
        )
        return amplitude * distributed_derivative

    result = integrate_distributed_order_caputo_l1(
        manufactured_rhs,
        initial,
        order_nodes=nodes,
        order_weights=masses,
        step=0.005,
        n_steps=100,
        lower_terminal=lower,
        order_quadrature_name="manufactured-three-node-rule",
        initial_regularity="smooth",
        use_acceleration=False,
        divergence_norm=None,
    )
    exact = initial + (result.times - lower)[:, None] ** 2 * amplitude

    assert result.status == "ok"
    np.testing.assert_allclose(result.states, exact, rtol=0.0, atol=2.5e-3)
    assert result.solver_info["order_quadrature_error_estimated"] is False
    assert result.solver_info["time_discretization_error_estimated"] is False


@pytest.mark.scientific_contract
def test_manufactured_solution_refines_with_fixed_order_rule() -> None:
    lower = 0.3
    duration = 0.5
    nodes = np.array([0.3, 0.65, 0.9])
    masses = np.array([0.25, 0.5, 0.25])

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        del state
        tau = max(0.0, time - lower)
        value = sum(
            mass * gamma(3.0) / gamma(3.0 - order) * tau ** (2.0 - order)
            for order, mass in zip(nodes, masses, strict=True)
        )
        return np.array([value])

    errors: list[float] = []
    for step in (0.02, 0.01):
        result = integrate_distributed_order_caputo_l1(
            rhs,
            [0.4],
            order_nodes=nodes,
            order_weights=masses,
            step=step,
            n_steps=int(round(duration / step)),
            lower_terminal=lower,
            initial_regularity="smooth",
            use_acceleration=False,
            divergence_norm=None,
        )
        exact_final = 0.4 + duration**2
        errors.append(abs(float(result.states[-1, 0]) - exact_final))

    assert errors[1] < 0.75 * errors[0]


@pytest.mark.scientific_contract
def test_coupled_linear_system_matches_independent_matrix_recurrence() -> None:
    nodes = np.array([0.35, 0.8])
    masses = np.array([0.4, 0.6])
    matrix = np.array([[-0.3, 0.05], [-0.02, -0.4]])
    forcing = np.array([0.15, -0.05])
    initial = np.array([0.2, -0.1])
    step = 0.015
    n_steps = 15
    result = integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            rhs=_matrix_affine_rhs,
            parameters=(matrix, forcing),
            initial_state=initial,
            order_nodes=nodes,
            order_weights=masses,
            step=step,
            n_steps=n_steps,
        )
    )
    expected = _independent_linear_l1_reference(
        initial_state=initial,
        matrix=matrix,
        forcing=lambda time: forcing,
        lower_terminal=0.0,
        step=step,
        n_steps=n_steps,
        order_nodes=nodes,
        effective_weights=masses,
    )

    np.testing.assert_allclose(result.states, expected, rtol=3.0e-12, atol=3.0e-13)


@pytest.mark.scientific_contract
def test_duplicate_zero_weight_and_permuted_nodes_preserve_the_rule() -> None:
    common = _solver_arguments(n_steps=10)
    duplicated = integrate_distributed_order_caputo_l1(
        **dict(
            common,
            order_nodes=[0.3, 0.3, 0.8, 0.6],
            order_weights=[0.15, 0.25, 0.6, 0.0],
        )
    )
    coalesced = integrate_distributed_order_caputo_l1(
        **dict(
            common,
            order_nodes=[0.3, 0.8],
            order_weights=[0.4, 0.6],
        )
    )
    permuted = integrate_distributed_order_caputo_l1(
        **dict(
            common,
            order_nodes=[0.8, 0.3],
            order_weights=[0.6, 0.4],
        )
    )

    np.testing.assert_allclose(duplicated.states, coalesced.states, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(permuted.states, coalesced.states, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(
        duplicated.combined_l1_kernel,
        coalesced.combined_l1_kernel,
        rtol=3e-15,
        atol=3e-15,
    )


@pytest.mark.scientific_contract
def test_scaling_order_measure_and_rhs_together_preserves_trajectory() -> None:
    scale = 7.5
    base = integrate_distributed_order_caputo_l1(**_solver_arguments(n_steps=15))

    def scaled_rhs(
        time: float,
        state: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:
        return scale * _scalar_affine_rhs(time, state, parameters)

    scaled = integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            rhs=scaled_rhs,
            order_weights=np.array([0.2, 0.5, 0.3]) * scale,
            n_steps=15,
        )
    )

    np.testing.assert_allclose(scaled.states, base.states, rtol=3e-13, atol=3e-14)


@pytest.mark.scientific_contract
def test_density_and_unit_mass_accounting_match_explicit_masses() -> None:
    nodes = [0.2, 0.6, 0.95]
    quadrature_weights = np.array([0.1, 0.2, 0.1])
    density = np.array([2.0, 3.0, 4.0])
    raw_effective = quadrature_weights * density
    normalized_effective = raw_effective / np.sum(raw_effective)
    density_result = integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            order_nodes=nodes,
            order_weights=quadrature_weights,
            weight_semantics="nonnegative_quadrature_density",
            density_values=density,
            normalization="unit_mass",
        )
    )
    explicit_result = integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            order_nodes=nodes,
            order_weights=normalized_effective,
            weight_semantics="nonnegative_mass",
            normalization="none",
        )
    )

    assert density_result.raw_mass == pytest.approx(1.2)
    assert density_result.raw_l1_norm == pytest.approx(1.2)
    assert density_result.mass == pytest.approx(1.0)
    assert density_result.l1_norm == pytest.approx(1.0)
    np.testing.assert_allclose(
        density_result.effective_weights,
        normalized_effective,
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        density_result.states,
        explicit_result.states,
        rtol=3.0e-14,
        atol=3.0e-14,
    )


@pytest.mark.scientific_contract
def test_numba_and_python_combined_history_paths_agree() -> None:
    arguments = _solver_arguments(n_steps=24)
    python_result = integrate_distributed_order_caputo_l1(**arguments)
    accelerated_arguments = dict(arguments)
    accelerated_arguments["use_acceleration"] = True
    numba_result = integrate_distributed_order_caputo_l1(**accelerated_arguments)

    assert python_result.status == numba_result.status == "ok"
    assert "python" in python_result.backend
    assert "numba" in numba_result.backend
    np.testing.assert_array_equal(numba_result.times, python_result.times)
    np.testing.assert_allclose(
        numba_result.combined_l1_kernel,
        python_result.combined_l1_kernel,
        rtol=3.0e-15,
        atol=3.0e-15,
    )
    np.testing.assert_allclose(
        numba_result.states,
        python_result.states,
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    assert numba_result.solver_info["used_numba_history"] is True


def test_numba_kernel_fallback_is_reported_and_can_be_disabled(monkeypatch) -> None:
    def broken_kernel(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic-numba-kernel-failure")

    monkeypatch.setattr(solver_module, "_combined_kernel_numba", broken_kernel)
    fallback = integrate_distributed_order_caputo_l1(
        **_solver_arguments(use_acceleration=True, allow_python_fallback=True)
    )

    assert fallback.status == "ok"
    assert fallback.backend == "python_numpy_combined_l1_picard"
    assert fallback.solver_info["numba_fallback_occurred"] is True
    assert "synthetic-numba-kernel-failure" in str(
        fallback.solver_info["numba_fallback_error"]
    )
    with pytest.raises(RuntimeError, match="synthetic-numba-kernel-failure"):
        integrate_distributed_order_caputo_l1(
            **_solver_arguments(
                use_acceleration=True,
                allow_python_fallback=False,
            )
        )


@pytest.mark.parametrize(
    "nodes",
    [
        [],
        [[0.5]],
        [0.0],
        [-0.1],
        [1.01],
        [np.nan],
        [np.inf],
        [0.5 + 0.1j],
        [True],
    ],
)
def test_invalid_order_nodes_are_rejected(nodes: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        integrate_distributed_order_caputo_l1(
            **_solver_arguments(order_nodes=nodes, order_weights=[1.0])
        )


@pytest.mark.parametrize(
    "weights",
    [
        [],
        [[1.0, 0.0]],
        [1.0],
        [0.0, 0.0, 0.0],
        [0.2, np.nan, 0.8],
        [0.2, np.inf, 0.8],
        [0.2, -0.1, 0.9],
        [True, False, False],
    ],
)
def test_invalid_order_weights_are_rejected(weights: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        integrate_distributed_order_caputo_l1(
            **_solver_arguments(order_weights=weights)
        )


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"weight_semantics": "signed_mass"}, "nonnegative"),
        (
            {
                "weight_semantics": "nonnegative_quadrature_density",
                "density_values": None,
            },
            "required",
        ),
        ({"density_values": [1.0, 1.0, 1.0]}, "not used"),
        (
            {
                "weight_semantics": "nonnegative_quadrature_density",
                "density_values": [1.0, -1.0, 1.0],
            },
            "Negative|signed",
        ),
        ({"normalization": "probability"}, "normalization"),
        ({"order_quadrature_name": ""}, "order_quadrature_name"),
    ],
)
def test_measure_semantics_are_explicit_and_validated(
    updates: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        integrate_distributed_order_caputo_l1(**_solver_arguments(**updates))


@pytest.mark.parametrize(
    ("updates", "error", "match"),
    [
        ({"rhs": None}, TypeError, "rhs|callable"),
        ({"initial_state": []}, ValueError, "initial_state"),
        ({"initial_state": [np.nan]}, ValueError, "initial_state"),
        ({"initial_state": [1.0j]}, TypeError, "initial_state|real"),
        ({"step": 0.0}, ValueError, "step|positive"),
        ({"step": True}, TypeError, "step|real"),
        ({"n_steps": 0}, ValueError, "n_steps"),
        ({"n_steps": 2.5}, ValueError, "n_steps|integer"),
        ({"n_steps": True}, ValueError, "n_steps|integer"),
        ({"lower_terminal": np.inf}, ValueError, "lower_terminal|finite"),
        ({"corrector_atol": -1.0}, ValueError, "corrector_atol"),
        ({"corrector_rtol": -1.0}, ValueError, "corrector_rtol"),
        (
            {"corrector_atol": 0.0, "corrector_rtol": 0.0},
            ValueError,
            "must not both be zero",
        ),
        ({"corrector_max_iterations": 0}, ValueError, "corrector_max_iterations"),
        ({"on_nonconvergence": "ignore"}, ValueError, "on_nonconvergence"),
        ({"initial_regularity": "analytic"}, ValueError, "initial_regularity"),
        ({"compatibility_tolerance": -1.0}, ValueError, "compatibility_tolerance"),
        ({"use_acceleration": 1}, TypeError, "use_acceleration|Boolean"),
        ({"allow_python_fallback": 1}, TypeError, "allow_python_fallback|Boolean"),
        ({"divergence_norm": 0.0}, ValueError, "divergence_norm|positive"),
    ],
)
def test_invalid_grid_state_and_corrector_options_are_rejected(
    updates: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        integrate_distributed_order_caputo_l1(**_solver_arguments(**updates))


@pytest.mark.parametrize(
    ("order", "lag", "error"),
    [
        (0.0, 1, ValueError),
        (1.01, 1, ValueError),
        (True, 1, TypeError),
        (0.6, -1, ValueError),
        (0.6, 1.5, ValueError),
        (0.6, True, ValueError),
    ],
)
def test_invalid_public_l1_weight_contract_is_rejected(
    order: object,
    lag: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error, match="order|lag|integer"):
        distributed_order_l1_weight(order, lag)  # type: ignore[arg-type]


def test_picard_corrector_matches_linear_oracle_and_reports_residuals() -> None:
    result = integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            order_nodes=[0.6],
            order_weights=[1.0],
            parameters=np.array([0.5, 0.8]),
            step=0.025,
            n_steps=8,
        )
    )
    expected = _independent_linear_l1_reference(
        initial_state=np.array([0.2]),
        matrix=np.array([[-0.8]]),
        forcing=lambda time: np.array([0.5]),
        lower_terminal=0.0,
        step=0.025,
        n_steps=8,
        order_nodes=np.array([0.6]),
        effective_weights=np.array([1.0]),
    )

    np.testing.assert_allclose(result.states, expected, rtol=3e-12, atol=3e-13)
    assert result.solver_info["corrector"] == "picard"
    assert result.solver_info["max_corrector_iterations_used"] >= 2
    assert np.all(np.isfinite(result.corrector_residuals[1:]))


def _noncontractive_rhs(time: float, state: np.ndarray) -> np.ndarray:
    del time
    return 1.0 + 10.0 * state


def test_picard_nonconvergence_raise_and_return_policies_are_explicit() -> None:
    arguments = _solver_arguments(
        rhs=_noncontractive_rhs,
        parameters=None,
        initial_state=[0.0],
        order_nodes=[0.6],
        order_weights=[1.0],
        step=0.1,
        n_steps=3,
        corrector_atol=1.0e-15,
        corrector_rtol=1.0e-15,
        corrector_max_iterations=2,
    )
    with pytest.raises(DistributedOrderCorrectorError, match="corrector|failed"):
        integrate_distributed_order_caputo_l1(**arguments)

    arguments["on_nonconvergence"] = "return"
    result = integrate_distributed_order_caputo_l1(**arguments)
    assert result.status == "corrector_nonconvergence"
    assert result.solver_info["nonconverged_step"] == 1
    assert result.solver_info["failure_time"] == pytest.approx(0.1)
    assert result.solver_info["failure_iterations"] == 2
    assert result.solver_info["failure_residual"] is not None
    assert result.solver_info["n_steps_completed"] == len(result.times) - 1
    assert len(result.times) < 4


def test_smooth_initial_compatibility_warning_applies_only_below_order_one() -> None:
    def incompatible_rhs(time: float, state: np.ndarray) -> np.ndarray:
        del time
        return np.ones_like(state)

    below_one = _solver_arguments(
        rhs=incompatible_rhs,
        parameters=None,
        initial_state=[0.0],
        order_nodes=[0.4, 0.8],
        order_weights=[0.5, 0.5],
        n_steps=2,
        initial_regularity="smooth",
    )
    with pytest.warns(DistributedOrderInitialCompatibilityWarning, match="lower|terminal"):
        fractional_result = integrate_distributed_order_caputo_l1(**below_one)
    assert fractional_result.solver_info["initial_compatibility_check_applies"] is True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        order_one_result = integrate_distributed_order_caputo_l1(
            **dict(
                below_one,
                order_nodes=[0.4, 1.0],
                order_weights=[0.5, 0.5],
            )
        )
    assert not any(
        isinstance(item.message, DistributedOrderInitialCompatibilityWarning)
        for item in caught
    )
    assert order_one_result.solver_info["initial_compatibility_check_applies"] is False
    assert order_one_result.solver_info["alpha_one_effective_mass"] == pytest.approx(0.5)


def test_nonsmooth_start_and_compatible_smooth_start_do_not_warn() -> None:
    def tiny_rhs(time: float, state: np.ndarray) -> np.ndarray:
        del time
        return np.full_like(state, 1.0e-14)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nonsmooth = integrate_distributed_order_caputo_l1(
            **_solver_arguments(
                rhs=_constant_rhs,
                parameters=np.array([1.0]),
                initial_state=[0.0],
                n_steps=2,
                initial_regularity="nonsmooth",
            )
        )
        smooth = integrate_distributed_order_caputo_l1(
            **_solver_arguments(
                rhs=tiny_rhs,
                parameters=None,
                initial_state=[0.0],
                n_steps=2,
                initial_regularity="smooth",
                compatibility_tolerance=1.0e-12,
            )
        )
    assert nonsmooth.status == smooth.status == "ok"
    assert not any(
        isinstance(item.message, DistributedOrderInitialCompatibilityWarning)
        for item in caught
    )


def test_initial_divergence_returns_without_evaluating_rhs() -> None:
    calls = 0

    def rhs_must_not_run(time: float, state: np.ndarray) -> np.ndarray:
        nonlocal calls
        del time, state
        calls += 1
        raise AssertionError("RHS must not run for an already divergent state")

    result = integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            rhs=rhs_must_not_run,
            parameters=None,
            initial_state=[2.0],
            divergence_norm=1.0,
        )
    )

    assert result.status == "diverged"
    assert calls == 0
    np.testing.assert_array_equal(result.times, np.array([0.0]))
    np.testing.assert_array_equal(result.states, np.array([[2.0]]))
    assert result.actual_upper_terminal == 0.0
    assert result.solver_info["n_steps_completed"] == 0


def test_physical_divergence_stops_before_later_rhs_evaluation() -> None:
    seen_times: list[float] = []

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        del state
        seen_times.append(float(time))
        if time > 0.015:
            raise RuntimeError("rhs-evaluated-after-physical-divergence")
        return np.array([5.0])

    result = integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            rhs=rhs,
            parameters=None,
            initial_state=[0.0],
            order_nodes=[0.5],
            order_weights=[1.0],
            step=0.01,
            n_steps=20,
            divergence_norm=0.2,
        )
    )

    assert result.status == "diverged"
    assert len(result.times) == len(result.states) == 2
    assert result.times[-1] == pytest.approx(0.01)
    assert np.linalg.norm(result.states[-2]) <= 0.2
    assert np.linalg.norm(result.states[-1]) > 0.2
    assert max(seen_times) <= result.times[-1]


def test_nonfinite_solution_takes_precedence_over_divergence() -> None:
    maximum = np.finfo(np.float64).max

    def finite_overflowing_rhs(time: float, state: np.ndarray) -> np.ndarray:
        del time
        return np.full_like(state, maximum)

    with np.errstate(over="ignore", invalid="ignore"):
        result = integrate_distributed_order_caputo_l1(
            **_solver_arguments(
                rhs=finite_overflowing_rhs,
                parameters=None,
                initial_state=[0.0],
                order_nodes=[0.5],
                order_weights=[1.0e-3],
                step=0.1,
                n_steps=2,
                divergence_norm=1.0,
            )
        )

    assert result.status == "nonfinite_solution"
    assert result.solver_info["failure_residual_nonfinite"] is True
    assert result.solver_info["failure_time"] == pytest.approx(0.1)


def test_large_finite_vector_norm_does_not_overflow() -> None:
    initial = np.array([1.0e308, 1.0e308])
    result = integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            rhs=_zero_rhs,
            parameters=None,
            initial_state=initial,
            n_steps=2,
            divergence_norm=1.7e308,
        )
    )

    assert result.status == "ok"
    assert np.all(np.isfinite(result.states))


@pytest.mark.parametrize(
    "bad_rhs",
    [
        lambda time, state: np.ones(state.size + 1),
        lambda time, state: np.full_like(state, np.nan),
        lambda time, state: np.full(state.shape, 1.0j),
    ],
)
def test_invalid_rhs_shape_finiteness_and_reality_are_rejected(bad_rhs) -> None:
    with pytest.raises((TypeError, ValueError), match="rhs|shape|real|finite"):
        integrate_distributed_order_caputo_l1(
            **_solver_arguments(rhs=bad_rhs, parameters=None)
        )


def test_internal_rhs_typeerror_is_not_reinterpreted_as_another_signature() -> None:
    calls = 0

    def broken_rhs(time: float, state: np.ndarray) -> np.ndarray:
        nonlocal calls
        del time, state
        calls += 1
        raise TypeError("distributed-order-internal-typeerror")

    with pytest.raises(TypeError, match="distributed-order-internal-typeerror"):
        integrate_distributed_order_caputo_l1(
            **_solver_arguments(rhs=broken_rhs, parameters=None)
        )
    assert calls == 1


def test_structured_result_and_truncation_metadata_are_consistent() -> None:
    result = integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            rhs=_constant_rhs,
            parameters=np.array([5.0]),
            initial_state=[0.0],
            order_nodes=[0.5],
            order_weights=[1.0],
            step=0.01,
            n_steps=20,
            divergence_norm=0.2,
        )
    )

    assert isinstance(result, DistributedOrderCaputoResult)
    assert result.method == "distributed_order_caputo_l1"
    assert result.memory_policy == "full_history"
    assert result.grid_coordinate == "physical_time"
    assert result.scope == "finite_numerical_trajectory_only"
    np.testing.assert_array_equal(result.trajectory[:, 0], result.times)
    np.testing.assert_array_equal(result.trajectory[:, 1:], result.states)
    assert result.solver_info["n_steps"] == len(result.times) - 1
    assert result.solver_info["n_steps_completed"] == len(result.times) - 1
    assert result.solver_info["n_samples"] == len(result.times)
    assert result.solver_info["n_samples_returned"] == len(result.times)


def _problem_arguments(**updates: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "derivative": "caputo_distributed_order",
        "method": "distributed_order_caputo_l1",
        "orders": [0.25, 0.7, 0.95],
        "initial_state": [0.2, -0.1],
        "step": 0.02,
        "t_span": (0.4, 0.6),
        "kernel_parameters": {
            "order_weights": [0.2, 0.5, 0.3],
            "weight_semantics": "nonnegative_mass",
            "normalization": "none",
            "order_quadrature_name": "problem-three-node-rule",
        },
        "method_options": {
            "corrector_atol": 1.0e-13,
            "corrector_rtol": 1.0e-12,
            "corrector_max_iterations": 100,
            "on_nonconvergence": "raise",
            "initial_regularity": "nonsmooth",
            "compatibility_tolerance": 1.0e-12,
        },
        "allow_experimental": True,
        "problem_id": "distributed-order-contract",
    }
    arguments.update(updates)
    return arguments


def test_fractional_problem_uses_order_nodes_independently_of_state_dimension() -> None:
    problem = FractionalProblem(**_problem_arguments())  # type: ignore[arg-type]
    derivative = get_fractional_derivative("caputo_distributed_order")
    method = get_fractional_method("distributed_order_caputo_l1")

    assert problem.dimension == 2
    assert len(problem.orders) == 3
    assert problem.order_mode == "distributed"
    assert problem.initial_condition_kind == "classical"
    assert derivative.implementation_status == "experimental"
    assert method.execution_kind == "solver"
    assert method.supports_combination(
        "caputo_distributed_order",
        "distributed",
        "full_history",
    )
    assert FractionalProblem.from_mapping(problem.to_mapping()) == problem


def test_fractional_problem_dispatch_matches_direct_solver_and_metadata() -> None:
    problem = FractionalProblem(**_problem_arguments())  # type: ignore[arg-type]
    matrix = np.array([[-0.3, 0.05], [-0.02, -0.4]])
    forcing = np.array([0.15, -0.05])
    parameters = (matrix, forcing)
    dispatched = solve_fractional_problem(
        problem,
        _matrix_affine_rhs,
        parameters,
        use_acceleration=False,
        divergence_norm=None,
    )
    direct = integrate_distributed_order_caputo_l1(
        _matrix_affine_rhs,
        problem.initial_state,
        parameters,
        order_nodes=problem.orders,
        order_weights=problem.kernel_parameters["order_weights"],
        step=problem.step,
        n_steps=problem.n_steps,
        lower_terminal=float(problem.lower_terminal),
        weight_semantics=str(problem.kernel_parameters["weight_semantics"]),
        normalization=str(problem.kernel_parameters["normalization"]),
        order_quadrature_name=str(
            problem.kernel_parameters["order_quadrature_name"]
        ),
        corrector_atol=problem.method_options["corrector_atol"],
        corrector_rtol=problem.method_options["corrector_rtol"],
        corrector_max_iterations=problem.method_options[
            "corrector_max_iterations"
        ],
        on_nonconvergence=str(problem.method_options["on_nonconvergence"]),
        initial_regularity=str(problem.method_options["initial_regularity"]),
        compatibility_tolerance=problem.method_options[
            "compatibility_tolerance"
        ],
        use_acceleration=False,
        divergence_norm=None,
    )

    assert dispatched.status == direct.status == "ok"
    np.testing.assert_array_equal(dispatched.times, direct.times)
    np.testing.assert_array_equal(dispatched.states, direct.states)
    assert dispatched.metadata["derivative"] == "caputo_distributed_order"
    assert dispatched.metadata["method"] == "distributed_order_caputo_l1"
    assert dispatched.metadata["order_mode"] == "distributed"
    assert dispatched.metadata["orders"] == [0.25, 0.7, 0.95]
    assert dispatched.metadata["initial_condition_kind"] == "classical"
    assert dispatched.metadata["kernel_parameters"] == dict(
        problem.kernel_parameters
    )
    assert dispatched.metadata["method_options"] == dict(problem.method_options)
    assert dispatched.metadata["backend_info"]["effective_order_weights"] == [
        0.2,
        0.5,
        0.3,
    ]
    assert dispatched.metadata["backend_info"]["order_quadrature"] == (
        "problem-three-node-rule"
    )
    assert dispatched.metadata["claims"] == "finite_numerical_trajectory_only"


def test_fractional_problem_requires_experimental_opt_in_at_execution() -> None:
    blocked = FractionalProblem(
        **_problem_arguments(allow_experimental=False)  # type: ignore[arg-type]
    )
    with pytest.raises(PermissionError, match="allow_experimental"):
        solve_fractional_problem(
            blocked,
            _zero_rhs,
            use_acceleration=False,
            divergence_norm=None,
        )


def test_fractional_problem_rejects_unconsumed_options_before_execution() -> None:
    method_options = dict(_problem_arguments()["method_options"])  # type: ignore[arg-type]
    method_options["silent_numerical_change"] = True
    method_problem = FractionalProblem(
        **_problem_arguments(method_options=method_options)  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="unsupported method_options|silent_numerical_change"):
        solve_fractional_problem(method_problem, _zero_rhs, use_acceleration=False)

    kernel_parameters = dict(_problem_arguments()["kernel_parameters"])  # type: ignore[arg-type]
    kernel_parameters["silent_kernel_change"] = True
    kernel_problem = FractionalProblem(
        **_problem_arguments(kernel_parameters=kernel_parameters)  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="unsupported kernel_parameters|silent_kernel_change"):
        solve_fractional_problem(kernel_problem, _zero_rhs, use_acceleration=False)


@pytest.mark.parametrize(
    ("updates", "error", "match"),
    [
        (
            {"kernel_parameters": {}},
            ValueError,
            "order_weights",
        ),
        (
            {
                "kernel_parameters": {
                    "order_weights": [0.2, 0.5, 0.3],
                    "weight_semantics": "signed_mass",
                }
            },
            ValueError,
            "nonnegative",
        ),
        (
            {"orders": [True]},
            TypeError,
            "orders|real|Boolean",
        ),
        (
            {
                "memory_policy": "finite_window",
                "history_window": 4,
            },
            ValueError,
            "memory policy|full_history|finite_window",
        ),
        (
            {"method": "distributed_order_gl_direct"},
            ValueError,
            "not registered|derivative|Method",
        ),
        (
            {"allow_experimental": "true"},
            TypeError,
            "allow_experimental|Boolean",
        ),
    ],
)
def test_fractional_problem_rejects_invalid_distributed_contracts(
    updates: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        FractionalProblem(**_problem_arguments(**updates))  # type: ignore[arg-type]


def test_solver_does_not_mutate_order_or_state_inputs() -> None:
    initial = np.array([0.2, -0.1])
    nodes = np.array([0.3, 0.8])
    weights = np.array([0.4, 0.6])
    initial_copy = initial.copy()
    nodes_copy = nodes.copy()
    weights_copy = weights.copy()
    integrate_distributed_order_caputo_l1(
        **_solver_arguments(
            rhs=_zero_rhs,
            parameters=None,
            initial_state=initial,
            order_nodes=nodes,
            order_weights=weights,
            n_steps=4,
        )
    )

    np.testing.assert_array_equal(initial, initial_copy)
    np.testing.assert_array_equal(nodes, nodes_copy)
    np.testing.assert_array_equal(weights, weights_copy)
