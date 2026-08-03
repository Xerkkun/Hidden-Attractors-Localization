from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors import (
    PUBLIC_API_EXPERIMENTAL,
    SimulationResult,
    simulate_fractional,
)
from hidden_attractors.fractional import FractionalProblem
from hidden_attractors.simulation import simulate_fractional as facade_function
from hidden_attractors.systems import ChaoticSystem


@pytest.mark.integration
def test_fractional_facade_preserves_hadamard_coordinates_and_contract() -> None:
    lower_terminal = 2.0
    log_step = 0.1
    n_steps = 3
    upper_terminal = lower_terminal * np.exp(n_steps * log_step)
    system = ChaoticSystem(
        name="toolbox-hadamard-constant",
        dimension=1,
        rhs=lambda state, parameters: np.asarray(
            [parameters["forcing"] + 0.0 * state[0]], dtype=float
        ),
        parameters={"forcing": 0.25},
        initial_state=(0.0,),
    )
    contract = FractionalProblem(
        derivative="caputo_hadamard",
        method="caputo_hadamard_abm_pece",
        orders=0.5,
        initial_state=(0.0,),
        step=log_step,
        t_span=(lower_terminal, upper_terminal),
        grid_coordinate="log_t_over_lower_terminal",
        memory_policy="full_history",
        allow_experimental=True,
        problem_id="toolbox-hadamard-bridge",
    )

    result = simulate_fractional(
        system,
        contract.to_mapping(),
        parameters={"forcing": 0.75},
        use_acceleration=False,
        allow_python_fallback=True,
        divergence_norm=None,
    )

    expected_coordinate = log_step * np.arange(n_steps + 1, dtype=float)
    expected_physical = lower_terminal * np.exp(expected_coordinate)
    assert result.status == "ok"
    assert result.times == pytest.approx(expected_physical)
    assert result.coordinate_times == pytest.approx(expected_coordinate)
    assert result.integrator_times == pytest.approx(expected_coordinate)
    assert result.trajectory[:, 0] == pytest.approx(expected_physical)
    assert result.grid_coordinate == "log_t_over_lower_terminal"
    assert result.system_name == system.name
    assert result.system_kind == "flow"
    assert result.method == contract.method
    assert result.parameters == {"forcing": 0.75}
    assert result.step_size == log_step
    assert result.requested_steps == n_steps
    assert result.completed_steps == n_steps

    metadata = result.metadata
    assert metadata["derivative"] == "caputo_hadamard"
    assert metadata["orders"] == [0.5]
    assert metadata["memory_policy"] == "full_history"
    assert metadata["history_window"] is None
    assert metadata["grid_coordinate"] == "log_t_over_lower_terminal"
    assert metadata["backend"] == result.backend
    assert metadata["backend_info"] == result.backend_info
    assert metadata["fractional_problem"] == contract.as_metadata()
    assert metadata["system_parameters"] == {"forcing": 0.75}
    assert metadata["simulation_facade"] == "simulate_fractional"
    assert metadata["time_coordinates"] == {
        "physical_field": "times",
        "integrator_field": "coordinate_times",
        "integrator_coordinate": "log_t_over_lower_terminal",
    }
    assert metadata["claims"] == "finite_numerical_trajectory_only"


@pytest.mark.integration
def test_fractional_facade_executes_distributed_order_nodes_independent_of_dimension() -> None:
    system = ChaoticSystem(
        name="toolbox-distributed-order-zero",
        dimension=2,
        rhs=lambda state, _parameters: np.zeros_like(state, dtype=float),
        initial_state=(0.4, -0.7),
    )
    contract = FractionalProblem(
        derivative="caputo_distributed_order",
        method="distributed_order_caputo_l1",
        orders=[0.3, 0.65, 1.0],
        initial_state=system.initial_state,
        step=0.05,
        t_span=(0.0, 0.2),
        memory_policy="full_history",
        kernel_parameters={
            "order_weights": [0.2, 0.5, 0.3],
            "weight_semantics": "nonnegative_mass",
            "order_quadrature_name": "toolbox-three-atom",
        },
        allow_experimental=True,
    )

    result = simulate_fractional(
        system,
        contract.to_mapping(),
        use_acceleration=False,
        divergence_norm=None,
    )

    assert result.status == "ok"
    np.testing.assert_array_equal(
        result.states,
        np.repeat(np.asarray([[0.4, -0.7]]), 5, axis=0),
    )
    assert result.metadata["order_mode"] == "distributed"
    assert result.metadata["orders"] == [0.3, 0.65, 1.0]
    assert result.metadata["backend_info"]["effective_order_weights"] == [
        0.2,
        0.5,
        0.3,
    ]
    assert result.metadata["fractional_problem"] == contract.as_metadata()


@pytest.mark.unit
def test_fractional_facade_is_public_and_simulation_result_is_compatible() -> None:
    legacy = SimulationResult(
        times=np.asarray([0.0, 0.1]),
        states=np.asarray([[1.0], [0.9]]),
        status="ok",
        system_name="legacy-constructor",
        system_kind="flow",
        method="rk4",
    )

    assert simulate_fractional is facade_function
    assert "simulate_fractional" in PUBLIC_API_EXPERIMENTAL
    assert legacy.coordinate_times is legacy.times
    assert legacy.grid_coordinate == "physical_time"
    assert legacy.backend is None
    assert legacy.backend_info == {}


@pytest.mark.unit
def test_fractional_facade_rejects_nonproblem_input() -> None:
    system = ChaoticSystem(
        name="invalid-problem-guard",
        dimension=1,
        rhs=lambda state, _parameters: np.asarray([state[0]], dtype=float),
        initial_state=(1.0,),
    )
    with pytest.raises(TypeError, match="FractionalProblem or mapping"):
        simulate_fractional(system, object())
