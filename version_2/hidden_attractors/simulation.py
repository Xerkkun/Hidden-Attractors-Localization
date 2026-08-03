"""Structured simulation facade for continuous flows and discrete maps.

Stability: experimental

Trajectory generation is deliberately separated from claims about chaos,
attraction, or hiddenness. The result contract is suitable for graphical
clients and reproducible scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .fractional.problem import FractionalProblem, solve_fractional_system
from .integrations.selector import integrate
from .systems import ChaoticSystem, get_system


@dataclass(frozen=True)
class SimulationResult:
    """Trajectory plus the numerical contract used to create it."""

    times: np.ndarray
    states: np.ndarray
    status: str
    system_name: str
    system_kind: str
    method: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    step_size: float | None = None
    requested_steps: int = 0
    completed_steps: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    integrator_times: np.ndarray | None = None
    grid_coordinate: str = "physical_time"
    backend: str | None = None
    backend_info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def trajectory(self) -> np.ndarray:
        """Return the conventional ``t,state...`` matrix."""

        return np.column_stack((self.times, self.states))

    @property
    def coordinate_times(self) -> np.ndarray:
        """Return the numerical integrator coordinate.

        Existing integer/map results default to physical time. Fractional
        results explicitly retain the selected solver clock, including
        logarithmic ``log(t/a)`` and conformable ``(t-a)^q/q`` grids.
        """

        if self.integrator_times is None:
            return self.times
        return self.integrator_times


def _resolve_initial(system: ChaoticSystem, initial_state: Sequence[float] | np.ndarray | None) -> np.ndarray:
    raw = initial_state if initial_state is not None else system.initial_state
    if raw is None or len(raw) == 0:
        raise ValueError("initial_state is required because the system has no default.")
    state = np.asarray(raw, dtype=float)
    if state.shape != (system.dimension,) or not np.all(np.isfinite(state)):
        raise ValueError(f"initial_state must be finite with shape ({system.dimension},).")
    return state


def simulate(
    system: ChaoticSystem | str,
    *,
    initial_state: Sequence[float] | np.ndarray | None = None,
    parameters: Mapping[str, Any] | None = None,
    step_size: float = 0.01,
    duration: float = 10.0,
    iterations: int | None = None,
    method: str = "rk4",
    divergence_norm: float | None = 1.0e6,
    use_acceleration: bool = True,
) -> SimulationResult:
    """Simulate a registered or constructed flow/map with one result contract."""

    model = get_system(system) if isinstance(system, str) else system
    if not isinstance(model, ChaoticSystem):
        raise TypeError("system must be a ChaoticSystem or registered system name.")
    state0 = _resolve_initial(model, initial_state)
    active_parameters = dict(model.parameters)
    if parameters:
        active_parameters.update(parameters)
    active_model = replace(model, parameters=active_parameters)
    threshold = None if divergence_norm is None else float(divergence_norm)
    if threshold is not None and (not np.isfinite(threshold) or threshold <= 0.0):
        raise ValueError("divergence_norm must be finite and positive, or None.")

    if model.kind == "map":
        count = int(iterations if iterations is not None else round(float(duration)))
        if count < 1:
            raise ValueError("iterations must be at least 1 for a map.")
        states = np.empty((count + 1, model.dimension), dtype=float)
        states[0] = state0
        status = "ok"
        completed = 0
        for index in range(count):
            try:
                next_state = model.evaluate(states[index], active_parameters)
            except (ArithmeticError, FloatingPointError, OverflowError, ValueError) as exc:
                status = f"map_exception:{exc}"
                break
            states[index + 1] = next_state
            completed = index + 1
            if not np.all(np.isfinite(next_state)):
                status = "nonfinite_solution"
                break
            if threshold is not None and float(np.linalg.norm(next_state)) >= threshold:
                status = "diverged"
                break
        return SimulationResult(
            times=np.arange(completed + 1, dtype=float),
            states=states[: completed + 1],
            status=status,
            system_name=model.name,
            system_kind=model.kind,
            method="map_iteration",
            parameters=active_parameters,
            requested_steps=count,
            completed_steps=completed,
            metadata={"claims": "trajectory_only", "acceleration_requested": False},
        )

    step = float(step_size)
    final_time = float(duration)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_size must be finite and positive.")
    if not np.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("duration must be finite and positive.")

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        return active_model.evaluate(state)

    times, states, status = integrate(
        rhs=rhs,
        x0=state0,
        q=1.0,
        h=step,
        t_final=final_time,
        integrator=method,
        divergence_norm=threshold,
        system=active_model,
        use_c_backend=bool(use_acceleration),
        allow_python_fallback=True,
        early_stop_config={"enabled": False},
    )
    requested = int(np.ceil(final_time / step))
    return SimulationResult(
        times=np.asarray(times, dtype=float),
        states=np.asarray(states, dtype=float),
        status=str(status),
        system_name=model.name,
        system_kind=model.kind,
        method=str(method),
        parameters=active_parameters,
        step_size=step,
        requested_steps=requested,
        completed_steps=max(0, len(times) - 1),
        metadata={"claims": "trajectory_only", "acceleration_requested": bool(use_acceleration)},
    )


def simulate_fractional(
    system: ChaoticSystem | str,
    problem: FractionalProblem | Mapping[str, Any],
    parameters: Mapping[str, Any] | None = None,
    *,
    use_acceleration: bool = True,
    allow_python_fallback: bool = True,
    divergence_norm: float | None = 120.0,
) -> SimulationResult:
    """Simulate a fractional flow through the public structured facade.

    ``times`` always contains physical time.  ``coordinate_times`` exposes the
    actual fixed-step coordinate, including ``log(t/a)`` for Caputo--Hadamard
    and ``(t-a)^q/q`` for conformable dynamics.
    The complete :class:`FractionalProblem` contract and the solver's backend
    metadata are retained for Toolbox Chaos and reproducible clients.

    The returned trajectory is finite numerical evidence only; this facade
    does not infer chaos, attraction, stability, or hiddenness.
    """

    if isinstance(problem, FractionalProblem):
        contract = problem
    elif isinstance(problem, Mapping):
        contract = FractionalProblem.from_mapping(problem)
    else:
        raise TypeError("problem must be a FractionalProblem or mapping.")

    solution = solve_fractional_system(
        contract,
        system,
        parameters,
        use_acceleration=bool(use_acceleration),
        allow_python_fallback=bool(allow_python_fallback),
        divergence_norm=divergence_norm,
    )
    physical_times = np.asarray(solution.times, dtype=float)
    coordinate_times = np.asarray(solution.coordinate_times, dtype=float)
    states = np.asarray(solution.states, dtype=float)
    metadata = dict(solution.metadata)
    backend_info = dict(metadata.get("backend_info", {}))
    active_parameters = dict(metadata.get("system_parameters", {}))
    metadata.update(
        {
            "simulation_facade": "simulate_fractional",
            "fractional_problem": contract.as_metadata(),
            "time_coordinates": {
                "physical_field": "times",
                "integrator_field": "coordinate_times",
                "integrator_coordinate": contract.grid_coordinate,
            },
        }
    )
    return SimulationResult(
        times=physical_times,
        states=states,
        status=solution.status,
        system_name=str(metadata["system_name"]),
        system_kind=str(metadata["system_kind"]),
        method=contract.method,
        parameters=active_parameters,
        step_size=contract.step,
        requested_steps=contract.n_steps,
        completed_steps=max(0, physical_times.size - 1),
        metadata=metadata,
        integrator_times=coordinate_times,
        grid_coordinate=contract.grid_coordinate,
        backend=solution.backend,
        backend_info=backend_info,
    )


__all__ = ["SimulationResult", "simulate", "simulate_fractional"]
