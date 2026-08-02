"""Structured simulation facade for continuous flows and discrete maps.

Stability: experimental

Trajectory generation is deliberately separated from claims about chaos,
attraction, or hiddenness. The result contract is suitable for graphical
clients and reproducible scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

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

    @property
    def trajectory(self) -> np.ndarray:
        """Return the conventional ``t,state...`` matrix."""

        return np.column_stack((self.times, self.states))


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
        return model.evaluate(state, active_parameters)

    times, states, status = integrate(
        rhs=rhs,
        x0=state0,
        q=1.0,
        h=step,
        t_final=final_time,
        integrator=method,
        divergence_norm=threshold,
        system=model,
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


__all__ = ["SimulationResult", "simulate"]
