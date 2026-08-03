from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.simulation import simulate
from hidden_attractors.systems import ChaoticSystem


def test_flow_parameter_override_changes_trajectory_and_metadata() -> None:
    system = ChaoticSystem(
        name="linear-override-contract",
        dimension=1,
        rhs=lambda state, parameters: np.array(
            [parameters["rate"] * state[0]], dtype=float
        ),
        parameters={"rate": -1.0},
        initial_state=(1.0,),
    )

    result = simulate(
        system,
        parameters={"rate": -2.0},
        duration=0.1,
        step_size=0.001,
        method="rk4",
        use_acceleration=True,
    )

    assert result.status == "ok"
    assert result.parameters == {"rate": -2.0}
    assert result.states[-1, 0] == pytest.approx(np.exp(-0.2), rel=2.0e-10)
    assert result.states[-1, 0] != pytest.approx(np.exp(-0.1), rel=1.0e-5)
