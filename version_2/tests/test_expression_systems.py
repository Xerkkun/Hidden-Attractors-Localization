from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors import (
    ExpressionSystemDefinition,
    ExpressionValidationError,
    compile_expression_system,
    simulate,
)


def lorenz_definition() -> ExpressionSystemDefinition:
    return ExpressionSystemDefinition(
        name="lorenz-no-code",
        kind="flow",
        variables=("x", "y", "z"),
        parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        equations=("sigma*(y-x)", "x*(rho-z)-y", "x*y-beta*z"),
        initial_state=(0.1, 0.1, 0.1),
    )


def test_expression_system_evaluates_without_python_eval() -> None:
    system = compile_expression_system(lorenz_definition())
    value = system.evaluate([1.0, 1.0, 1.0])
    assert np.allclose(value, [0.0, 26.0, -5.0 / 3.0])
    assert system.kind == "flow"
    assert system.state_names == ("x", "y", "z")


def test_expression_system_rejects_attribute_access_and_imports() -> None:
    with pytest.raises(ExpressionValidationError):
        ExpressionSystemDefinition(
            name="unsafe",
            variables=("x",),
            equations=("__import__('os').system('echo unsafe')",),
            initial_state=(1.0,),
        )


def test_structured_flow_simulation() -> None:
    system = compile_expression_system(lorenz_definition())
    result = simulate(system, duration=0.05, step_size=0.01, method="rk4")
    assert result.status == "ok"
    assert result.states.shape == (6, 3)
    assert result.trajectory.shape == (6, 4)
    assert result.metadata["claims"] == "trajectory_only"


def test_structured_discrete_map_simulation() -> None:
    definition = ExpressionSystemDefinition(
        name="logistic-no-code",
        kind="map",
        variables=("x",),
        parameters={"r": 4.0},
        equations=("r*x*(1-x)",),
        initial_state=(0.2,),
    )
    result = simulate(compile_expression_system(definition), iterations=3)
    assert result.status == "ok"
    assert np.allclose(result.states[:, 0], [0.2, 0.64, 0.9216, 0.28901376])
    assert result.method == "map_iteration"


def test_definition_round_trip_mapping() -> None:
    original = lorenz_definition()
    restored = ExpressionSystemDefinition.from_mapping(original.to_mapping())
    assert restored == original
