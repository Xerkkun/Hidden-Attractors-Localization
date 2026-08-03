#!/usr/bin/env python
"""Run one finite Chua trajectory with the experimental Caputo--Hadamard solver.

The q=1 comparison uses the logarithmic coordinate ``u=log(t/a)``.  The output
is a numerical usage example and does not certify chaos, attraction or
hiddenness.
"""

from __future__ import annotations

import json
from math import exp
from typing import Any

import numpy as np

from hidden_attractors.fractional import FractionalProblem, solve_fractional_system
from hidden_attractors.solvers.integer import dop853_q1_integrate
from hidden_attractors.systems.builtins import chua_system


def run_example(
    *,
    order: float = 0.98,
    log_duration: float = 0.25,
    log_step: float = 0.0025,
    use_acceleration: bool = False,
) -> dict[str, Any]:
    """Return a JSON-compatible integer/fractional Chua comparison record."""

    lower_terminal = 1.0
    upper_terminal = lower_terminal * exp(log_duration)
    initial_state = np.array([0.1, 0.0, 0.0])
    system = chua_system("nonsmooth")
    problem = FractionalProblem(
        derivative="caputo_hadamard",
        method="caputo_hadamard_abm_pece",
        orders=order,
        initial_state=initial_state,
        step=log_step,
        t_span=(lower_terminal, upper_terminal),
        grid_coordinate="log_t_over_lower_terminal",
        lower_terminal=lower_terminal,
        memory_policy="full_history",
        allow_experimental=True,
        problem_id="caputo-hadamard-chua-example",
    )
    fractional = solve_fractional_system(
        problem,
        system,
        use_acceleration=use_acceleration,
        divergence_norm=120.0,
    )

    integer, integer_status = dop853_q1_integrate(
        lambda state: system.evaluate(state),
        initial_state,
        t_final=log_duration,
        h=log_step,
        max_step=log_step,
        div_threshold=120.0,
    )
    return {
        "system": system.name,
        "coordinate": "u=log(t/lower_terminal)",
        "fractional": {
            "order": order,
            "status": fractional.status,
            "backend": fractional.backend,
            "samples": int(fractional.times.size),
            "physical_time_interval": [
                float(fractional.times[0]),
                float(fractional.times[-1]),
            ],
            "final_state": fractional.states[-1].tolist(),
            "metadata": dict(fractional.metadata),
        },
        "integer_q1_log_coordinate_reference": {
            "status": integer_status,
            "samples": int(integer.shape[0]),
            "log_time_interval": [float(integer[0, 0]), float(integer[-1, 0])],
            "final_state": integer[-1, 1:].tolist(),
        },
        "claims": (
            "Finite trajectory comparison only; no chaos, attraction, basin, "
            "or hiddenness conclusion."
        ),
    }


def main() -> None:
    print(json.dumps(run_example(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
