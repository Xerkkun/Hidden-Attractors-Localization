"""Reusable integer-order continuation and sampled hidden-chaos controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..analysis.trajectory import sample_rows
from ..solvers.integer import dop853_q1_integrate
from ..systems.base import ChaoticSystem
from ..verification.attractor_reference import (
    AttractorReferenceCalibration,
    classify_cloud_against_reference,
)
from .protocol import sample_uniform_ball


SystemFactory = Callable[[Mapping[str, Any]], ChaoticSystem]


@dataclass(frozen=True)
class IntegerParameterContinuationStep:
    """One state-transport node with a freshly reconstructed system."""

    node_index: int
    node_id: str
    parameters: Mapping[str, Any]
    x_in: np.ndarray
    x_out: np.ndarray
    trajectory: np.ndarray
    status: str
    solver_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class IntegerHiddenChaosProbe:
    """One finite equilibrium-neighborhood probe against a chaotic reference."""

    sample_id: int
    equilibrium: str
    radius: float
    direction_id: int
    sampling_mode: str
    x0: np.ndarray
    status: str
    destination: str
    target_classification: str
    target_distance_norm: float
    target_hit: bool
    ambiguous: bool
    tail_span: float
    closest_equilibrium: str
    closest_equilibrium_distance: float
    solver_metadata: Mapping[str, Any]


def _validate_complete_parameters(system: ChaoticSystem, parameters: Mapping[str, Any]) -> None:
    declared = dict(system.parameters)
    supplied = dict(parameters)
    if set(declared) != set(supplied):
        raise ValueError("each continuation node must declare the complete system parameter set.")
    for key in supplied:
        if isinstance(declared[key], (int, float)) and not np.isclose(
            float(declared[key]), float(supplied[key]), rtol=0.0, atol=0.0
        ):
            raise RuntimeError(f"system_factory did not apply parameter {key!r} at this node.")


def continue_integer_parameter_path(
    system_factory: SystemFactory,
    parameter_path: Sequence[Mapping[str, Any]],
    x0: Sequence[float],
    *,
    t_burn: float,
    t_keep: float,
    sample_step: float,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    max_step: float = 0.02,
    div_threshold: float | None = None,
) -> list[IntegerParameterContinuationStep]:
    """Transport a state while rebuilding equations and Lur'e data per node."""

    if not parameter_path:
        raise ValueError("parameter_path cannot be empty.")
    state = np.asarray(x0, dtype=float).copy()
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        raise ValueError("x0 must be a finite one-dimensional state.")
    steps: list[IntegerParameterContinuationStep] = []
    for index, raw_parameters in enumerate(parameter_path):
        parameters = dict(raw_parameters)
        system = system_factory(parameters)
        _validate_complete_parameters(system, parameters)
        if system.dimension != state.size:
            raise ValueError("system dimension changed along the continuation path.")
        if system.lure is not None:
            residual = np.linalg.norm(system.evaluate(state) - system.lure.evaluate(state))
            if not np.isfinite(residual) or residual > 1.0e-9:
                raise RuntimeError("reconstructed Lur'e declaration does not match the node equations.")
        x_in = state.copy()
        burn, burn_status = dop853_q1_integrate(
            lambda value: system.evaluate(value),
            state,
            t_final=float(t_burn),
            h=float(sample_step),
            rtol=float(rtol),
            atol=float(atol),
            max_step=float(max_step),
            div_threshold=div_threshold,
        )
        if burn_status != "ok":
            kept = burn
            kept_status = burn_status
        else:
            kept, kept_status = dop853_q1_integrate(
                lambda value: system.evaluate(value),
                burn[-1, 1:],
                t_final=float(t_keep),
                h=float(sample_step),
                rtol=float(rtol),
                atol=float(atol),
                max_step=float(max_step),
                div_threshold=div_threshold,
            )
        state = kept[-1, 1:].copy()
        steps.append(
            IntegerParameterContinuationStep(
                node_index=index,
                node_id=f"node_{index:04d}",
                parameters=parameters,
                x_in=x_in,
                x_out=state.copy(),
                trajectory=kept,
                status=kept_status,
                solver_metadata={
                    "method": "DOP853",
                    "rtol": float(rtol),
                    "atol": float(atol),
                    "max_step": float(max_step),
                    "sample_step": float(sample_step),
                    "t_burn": float(t_burn),
                    "t_keep": float(t_keep),
                    "burn_status": burn_status,
                    "keep_status": kept_status,
                },
            )
        )
        if kept_status != "ok":
            break
    return steps


def deterministic_unit_directions(dimension: int, count: int = 12) -> np.ndarray:
    """Return deterministic sphere directions; the 3-D contract matches the audit."""

    n = int(dimension)
    m = int(count)
    if n < 1 or m < 2 * n:
        raise ValueError("count must be at least twice the positive dimension.")
    axes = np.vstack((np.eye(n), -np.eye(n)))
    if m == 2 * n:
        return axes
    if n == 3:
        extra_count = m - 6
        golden = np.pi * (3.0 - np.sqrt(5.0))
        extra = []
        for index in range(extra_count):
            z = 1.0 - (2.0 * index + 1.0) / extra_count
            radius = np.sqrt(max(0.0, 1.0 - z * z))
            angle = golden * index
            extra.append([radius * np.cos(angle), radius * np.sin(angle), z])
        return np.vstack((axes, np.asarray(extra, dtype=float)))
    rng = np.random.default_rng(20260802)
    extra = rng.normal(size=(m - 2 * n, n))
    extra /= np.linalg.norm(extra, axis=1)[:, None]
    return np.vstack((axes, extra))


def equilibrium_stability_records(
    system: ChaoticSystem,
    *,
    tolerance: float = 1.0e-8,
) -> list[dict[str, Any]]:
    """Classify q=1 equilibria from the registered analytic Jacobian."""

    records: list[dict[str, Any]] = []
    for name, equilibrium in system.equilibrium_points().items():
        eigenvalues = np.linalg.eigvals(system.jacobian_matrix(equilibrium))
        spectral_abscissa = float(np.max(np.real(eigenvalues)))
        if spectral_abscissa < -float(tolerance):
            stability = "locally_asymptotically_stable"
        elif spectral_abscissa > float(tolerance):
            stability = "unstable"
        else:
            stability = "marginal_or_inconclusive"
        records.append(
            {
                "equilibrium": name,
                "state": equilibrium,
                "rhs_residual": float(np.linalg.norm(system.evaluate(equilibrium))),
                "eigenvalues": eigenvalues,
                "spectral_abscissa": spectral_abscissa,
                "stability": stability,
            }
        )
    return records


def run_integer_hidden_chaos_controls(
    system: ChaoticSystem,
    reference_clouds: Sequence[np.ndarray],
    calibration: AttractorReferenceCalibration,
    *,
    equilibrium_names: Sequence[str] | None = None,
    radii: Sequence[float] = (1.0e-5, 1.0e-3, 1.0e-2),
    directions: np.ndarray | None = None,
    samples_per_radius: int = 12,
    sampling_mode: str = "sphere",
    random_seed: int = 20260802,
    t_burn: float = 300.0,
    t_keep: float = 120.0,
    sample_step: float = 0.05,
    rtol: float = 2.0e-10,
    atol: float = 2.0e-12,
    max_step: float = 0.02,
    div_threshold: float = 50.0,
    equilibrium_tol: float = 1.0e-6,
    equilibrium_tail_span_tol: float = 1.0e-5,
    max_cloud_points: int = 1000,
) -> list[IntegerHiddenChaosProbe]:
    """Probe all equilibrium neighborhoods against a calibrated chaotic cloud.

    The result is finite sampled-neighborhood evidence.  It is not a global
    proof that the basin is disjoint from every equilibrium neighborhood.
    """

    all_equilibria = system.equilibrium_points()
    if not all_equilibria:
        raise ValueError("system must declare all equilibria.")
    equilibria = all_equilibria
    if equilibrium_names is not None:
        requested = tuple(str(name) for name in equilibrium_names)
        missing = [name for name in requested if name not in equilibria]
        if missing:
            raise ValueError(f"unknown equilibrium names: {missing}.")
        equilibria = {name: equilibria[name] for name in requested}
    if sampling_mode not in {"sphere", "ball"}:
        raise ValueError("sampling_mode must be 'sphere' or 'ball'.")
    rng = np.random.default_rng(int(random_seed))
    direction_values = (
        deterministic_unit_directions(system.dimension, int(samples_per_radius))
        if directions is None
        else np.asarray(directions, dtype=float)
    )
    if direction_values.shape != (int(samples_per_radius), system.dimension):
        raise ValueError("directions must have shape (samples_per_radius, system.dimension).")
    norms = np.linalg.norm(direction_values, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("directions must be finite and nonzero.")
    direction_values = direction_values / norms[:, None]

    probes: list[IntegerHiddenChaosProbe] = []
    sample_id = 0
    for equilibrium_name, equilibrium in equilibria.items():
        center = np.asarray(equilibrium, dtype=float)
        for radius in radii:
            radius_value = float(radius)
            if radius_value <= 0.0 or not np.isfinite(radius_value):
                raise ValueError("radii must be finite and positive.")
            if sampling_mode == "sphere":
                points = center + radius_value * direction_values
            else:
                points = sample_uniform_ball(center, radius_value, int(samples_per_radius), rng)
            for direction_id, x0 in enumerate(points, start=1):
                trajectory, status = dop853_q1_integrate(
                    lambda value: system.evaluate(value),
                    x0,
                    t_final=float(t_burn) + float(t_keep),
                    h=float(sample_step),
                    rtol=float(rtol),
                    atol=float(atol),
                    max_step=float(max_step),
                    div_threshold=float(div_threshold),
                )
                tail = trajectory[trajectory[:, 0] >= float(t_burn), 1:]
                tail = sample_rows(tail, int(max_cloud_points))
                tail_span = float(np.linalg.norm(np.ptp(tail, axis=0))) if tail.size else float("nan")
                final_state = trajectory[-1, 1:]
                distances = {
                    name: float(np.linalg.norm(final_state - value))
                    for name, value in all_equilibria.items()
                }
                closest = min(distances, key=distances.get)
                closest_distance = distances[closest]
                if status != "ok" or tail.size == 0 or not np.all(np.isfinite(tail)):
                    destination = "numerical_failure"
                    target = {
                        "classification": "inconclusive",
                        "distance_norm": float("nan"),
                    }
                elif closest_distance <= float(equilibrium_tol) and tail_span <= float(equilibrium_tail_span_tol):
                    destination = f"equilibrium_{closest}"
                    target = classify_cloud_against_reference(tail, reference_clouds, calibration)
                else:
                    target = classify_cloud_against_reference(tail, reference_clouds, calibration)
                    destination = str(target["classification"])
                classification = str(target["classification"])
                probes.append(
                    IntegerHiddenChaosProbe(
                        sample_id=sample_id,
                        equilibrium=equilibrium_name,
                        radius=radius_value,
                        direction_id=direction_id,
                        sampling_mode=sampling_mode,
                        x0=np.asarray(x0, dtype=float),
                        status=status,
                        destination=destination,
                        target_classification=classification,
                        target_distance_norm=float(target.get("distance_norm", float("nan"))),
                        target_hit=classification == "same_attractor_under_calibrated_cloud_test",
                        ambiguous=classification == "inconclusive",
                        tail_span=tail_span,
                        closest_equilibrium=closest,
                        closest_equilibrium_distance=closest_distance,
                        solver_metadata={
                            "method": "DOP853",
                            "rtol": float(rtol),
                            "atol": float(atol),
                            "max_step": float(max_step),
                            "sample_step": float(sample_step),
                            "t_burn": float(t_burn),
                            "t_keep": float(t_keep),
                        },
                    )
                )
                sample_id += 1
    return probes


def summarize_integer_hidden_chaos_controls(
    probes: Sequence[IntegerHiddenChaosProbe],
    *,
    required_equilibrium_names: Sequence[str] | None = None,
    declared_equilibrium_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize sampled hiddenness without converting it into global proof."""

    by_equilibrium: dict[str, dict[str, int]] = {}
    target_hits = 0
    ambiguous = 0
    failures = 0
    for probe in probes:
        row = by_equilibrium.setdefault(
            probe.equilibrium,
            {"n": 0, "target_hits": 0, "ambiguous": 0, "numerical_failures": 0, "equilibrium_destinations": 0},
        )
        row["n"] += 1
        row["target_hits"] += int(probe.target_hit)
        row["ambiguous"] += int(probe.ambiguous)
        row["numerical_failures"] += int(probe.status != "ok")
        row["equilibrium_destinations"] += int(probe.destination.startswith("equilibrium_"))
        target_hits += int(probe.target_hit)
        ambiguous += int(probe.ambiguous)
        failures += int(probe.status != "ok")
    tested_names = tuple(by_equilibrium)
    required_names = tuple(dict.fromkeys(required_equilibrium_names or tested_names))
    declared_names = tuple(dict.fromkeys(declared_equilibrium_names or ()))
    tested_all_required = bool(required_names) and all(
        name in by_equilibrium and by_equilibrium[name]["n"] > 0
        for name in required_names
    )
    tested_all_declared = bool(declared_names) and all(
        name in by_equilibrium and by_equilibrium[name]["n"] > 0
        for name in declared_names
    )
    complete = (
        bool(probes)
        and tested_all_required
        and target_hits == 0
        and ambiguous == 0
        and failures == 0
    )
    return {
        "n_probes": len(probes),
        "target_hits": target_hits,
        "ambiguous": ambiguous,
        "numerical_failures": failures,
        "required_equilibria": list(required_names),
        "declared_equilibria": list(declared_names),
        "tested_equilibria": list(tested_names),
        "tested_all_required_equilibria": tested_all_required,
        "tested_all_declared_equilibria": tested_all_declared,
        "by_equilibrium": by_equilibrium,
        "sampled_hiddenness_status": (
            "hidden_under_tested_neighborhoods"
            if complete
            else "self_excited_under_tested_neighborhoods"
            if target_hits > 0
            else "inconclusive"
        ),
        "finite_sample_only": True,
        "global_hiddenness_proved": False,
    }


__all__ = [
    "IntegerHiddenChaosProbe",
    "IntegerParameterContinuationStep",
    "continue_integer_parameter_path",
    "deterministic_unit_directions",
    "equilibrium_stability_records",
    "run_integer_hidden_chaos_controls",
    "summarize_integer_hidden_chaos_controls",
]
