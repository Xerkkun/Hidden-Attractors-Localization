"""Reproducible A1 localization for the integer lead-lag PLL.

The direct scalar Lur'e route is evaluated first and rejected from the exact
sign of the transfer imaginary part.  The alternative route uses the exact
Andronov phase transformation, a one-turn return map, and continuation of the
known running solution at zero loop gain.  No published target initial state
enters the localization functions in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.spatial import cKDTree

from ..seed_generation import find_integer_lure_omega_gain_candidates_direct
from ..systems.base import ChaoticSystem
from ..systems.pll_lead_lag import (
    pll_lead_lag_parameters,
    pll_original_to_shifted,
    pll_shifted_to_original,
    wrap_pll_angle,
)


@dataclass(frozen=True)
class PllReturnEvaluation:
    """One complete positive-winding evaluation of the Andronov map."""

    loop_gain: float
    initial_velocity: float
    return_velocity: float
    period: float
    multiplier: float

    @property
    def residual(self) -> float:
        return float(self.return_velocity - self.initial_velocity)


@dataclass(frozen=True)
class PllRunningCycle:
    """Fixed point of the one-turn return map."""

    loop_gain: float
    section_velocity: float
    section_x: float
    period: float
    multiplier: float
    return_residual: float
    stability: str


@dataclass(frozen=True)
class PllHiddennessProbe:
    """One finite cylinder-neighborhood trajectory classification."""

    sample_id: int
    equilibrium: str
    radius: float
    initial_state: np.ndarray
    scaled_distance_from_equilibrium: float
    status: str
    final_class: str
    target_hit: bool
    winding_tail: float
    cloud_distance_p90: float
    focus_distance_final: float
    saddle_distance_final: float


@dataclass(frozen=True)
class PllTrajectoryClassification:
    """Cylinder-aware destination classification for one PLL trajectory."""

    final_class: str
    target_hit: bool
    winding_tail: float
    cloud_distance_p90: float
    focus_distance_final: float
    saddle_distance_final: float


def pll_direct_route_diagnostic(system: ChaoticSystem) -> dict[str, Any]:
    """Execute and analytically reject the grid-free direct transfer route."""

    if system.lure is None:
        raise ValueError("PLL direct-route diagnostics require a scalar Lur'e declaration.")
    p = pll_lead_lag_parameters(system.parameters)
    candidates = find_integer_lure_omega_gain_candidates_direct(
        system.lure,
        wmin=0.0,
        wmax=float("inf"),
        compatible_only=False,
    )
    alpha = 1.0 / p["total_time_constant"]
    n_coefficient = p["tau2"] * p["loop_gain"] / (2.0 * p["total_time_constant"])
    delta = p["loop_gain"] / (2.0 * p["total_time_constant"])
    if candidates:
        raise RuntimeError("the PLL transfer unexpectedly produced a real-axis frequency crossing.")
    return {
        "route": "direct_integer_transfer_no_grid",
        "transfer_standard": "-L*(1+tau2*s)/(2*s*(1+(tau1+tau2)*s))",
        "transfer_code_convention": "+L*(1+tau2*s)/(2*s*(1+(tau1+tau2)*s))",
        "closure_conventions": {
            "standard": "1-k*G_standard=0",
            "code": "1+k*W_code=0",
        },
        "positive_imaginary_numerator": {
            "omega_coefficient": float(delta * alpha),
            "omega_cubed_coefficient": float(n_coefficient),
        },
        "analytic_condition": (
            "Im G_kappa(i*omega)=omega*(delta*alpha+n*omega^2)/"
            "((kappa*delta-omega^2)^2+(alpha+kappa*n)^2*omega^2)>0 for omega>0"
        ),
        "direct_candidates": [],
        "decision": "rejected_no_positive_imaginary_part_root",
        "failure_scope": "bounded_static_real_describing_function",
        "frequency_scan_used": False,
        "published_initial_conditions_used": False,
    }


def _base_values(parameters: Mapping[str, Any] | None) -> dict[str, float]:
    target = pll_lead_lag_parameters(parameters)
    return {
        "tau1": target["tau1"],
        "tau2": target["tau2"],
        "total": target["total_time_constant"],
        "omega_delta": target["omega_delta"],
        "target_loop_gain": target["loop_gain"],
    }


def pll_andronov_return(
    initial_velocity: float,
    loop_gain: float,
    parameters: Mapping[str, Any] | None = None,
    *,
    rtol: float = 2.0e-11,
    atol: float = 2.0e-13,
    max_step_angle: float = 0.02,
    minimum_velocity: float = 1.0e-8,
) -> PllReturnEvaluation | None:
    """Integrate one positive phase winding with ``theta`` as independent variable.

    ``None`` means that the trajectory lost positive winding before reaching
    ``theta=2*pi``.  The third augmented equation integrates the exact
    derivative of the scalar return map, so the reported multiplier is not a
    finite-difference estimate.
    """

    velocity0 = float(initial_velocity)
    gain = float(loop_gain)
    if velocity0 <= minimum_velocity or not np.isfinite(velocity0):
        return None
    if gain < 0.0 or not np.isfinite(gain):
        raise ValueError("loop_gain must be finite and nonnegative during continuation.")
    base = _base_values(parameters)
    total = base["total"]
    omega_delta = base["omega_delta"]
    tau2 = base["tau2"]

    def augmented_rhs(theta: float, state: np.ndarray) -> np.ndarray:
        velocity, _clock, sensitivity = state
        forcing = omega_delta - 0.5 * gain * np.sin(theta)
        damping = 1.0 + 0.5 * tau2 * gain * np.cos(theta)
        return np.array(
            [
                (forcing - damping * velocity) / (total * velocity),
                1.0 / velocity,
                -forcing * sensitivity / (total * velocity * velocity),
            ],
            dtype=float,
        )

    def stopped(_theta: float, state: np.ndarray) -> float:
        return float(state[0] - minimum_velocity)

    stopped.direction = -1
    stopped.terminal = True
    result = solve_ivp(
        augmented_rhs,
        (0.0, 2.0 * np.pi),
        np.array([velocity0, 0.0, 1.0], dtype=float),
        events=stopped,
        method="DOP853",
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step_angle),
    )
    if result.status == 1 or result.t[-1] < 2.0 * np.pi - 1.0e-9:
        return None
    return PllReturnEvaluation(
        loop_gain=gain,
        initial_velocity=velocity0,
        return_velocity=float(result.y[0, -1]),
        period=float(result.y[1, -1]),
        multiplier=float(result.y[2, -1]),
    )


def _return_residual(
    velocity: float,
    loop_gain: float,
    parameters: Mapping[str, Any] | None,
    **return_options: float,
) -> float:
    evaluation = pll_andronov_return(
        velocity, loop_gain, parameters, **return_options
    )
    if evaluation is None:
        return float("nan")
    return evaluation.residual


def _cycle_from_velocity(
    velocity: float,
    loop_gain: float,
    parameters: Mapping[str, Any] | None,
    **return_options: float,
) -> PllRunningCycle:
    evaluation = pll_andronov_return(
        velocity, loop_gain, parameters, **return_options
    )
    if evaluation is None:
        raise RuntimeError("the requested section velocity did not complete one winding.")
    base = _base_values(parameters)
    if loop_gain == 0.0:
        alpha = 1.0 / base["total"]
        input_gain = base["tau1"] / (2.0 * base["total"])
        section_x = -input_gain * base["omega_delta"] / (
            alpha * alpha + base["omega_delta"] * base["omega_delta"]
        )
    else:
        section_x = base["total"] * (base["omega_delta"] - velocity) / loop_gain
    multiplier = float(evaluation.multiplier)
    return PllRunningCycle(
        loop_gain=float(loop_gain),
        section_velocity=float(velocity),
        section_x=float(section_x),
        period=float(evaluation.period),
        multiplier=multiplier,
        return_residual=float(evaluation.residual),
        stability="stable" if abs(multiplier) < 1.0 else "unstable",
    )


def exact_zero_gain_running_cycle(
    parameters: Mapping[str, Any] | None = None,
) -> PllRunningCycle:
    """Return the analytic running orbit used to start loop-gain continuation."""

    base = _base_values(parameters)
    omega_delta = base["omega_delta"]
    total = base["total"]
    cycle = _cycle_from_velocity(omega_delta, 0.0, parameters)
    expected_period = 2.0 * np.pi / omega_delta
    expected_multiplier = float(np.exp(-expected_period / total))
    if abs(cycle.period - expected_period) > 2.0e-10:
        raise RuntimeError("zero-gain Andronov period does not match the analytic orbit.")
    if abs(cycle.multiplier - expected_multiplier) > 2.0e-9:
        raise RuntimeError("zero-gain return multiplier does not match the analytic orbit.")
    return cycle


def _bracket_root_near_previous(
    previous_velocity: float,
    loop_gain: float,
    parameters: Mapping[str, Any] | None,
    *,
    samples: int = 96,
    **return_options: float,
) -> tuple[float, float]:
    lower = max(1.0e-5, 0.45 * previous_velocity)
    upper = max(lower + 1.0, 1.55 * previous_velocity)
    grid = np.linspace(lower, upper, int(samples))
    values = [
        _return_residual(value, loop_gain, parameters, **return_options)
        for value in grid
    ]
    candidates: list[tuple[float, float]] = []
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue
        if f_left == 0.0 or f_left * f_right < 0.0:
            candidates.append((float(left), float(right)))
    if not candidates:
        raise RuntimeError(f"failed to bracket a running-cycle return root at L={loop_gain:g}.")
    return min(candidates, key=lambda pair: abs(0.5 * sum(pair) - previous_velocity))


def continue_pll_running_cycle(
    loop_gain_values: Sequence[float],
    parameters: Mapping[str, Any] | None = None,
    *,
    root_tolerance: float = 1.0e-11,
    **return_options: float,
) -> list[PllRunningCycle]:
    """Continue the analytic zero-gain running orbit over a gain schedule."""

    schedule = np.asarray(loop_gain_values, dtype=float)
    if schedule.ndim != 1 or len(schedule) < 2:
        raise ValueError("loop_gain_values must be a one-dimensional schedule with at least two points.")
    if abs(float(schedule[0])) > 1.0e-15 or np.any(np.diff(schedule) <= 0.0):
        raise ValueError("the continuation schedule must start at zero and increase strictly.")
    target = _base_values(parameters)["target_loop_gain"]
    if float(schedule[-1]) > target + 1.0e-12:
        raise ValueError("the continuation schedule exceeds the configured target loop gain.")

    cycles = [exact_zero_gain_running_cycle(parameters)]
    previous = cycles[0].section_velocity
    for loop_gain in schedule[1:]:
        gain = float(loop_gain)

        def residual(value: float) -> float:
            result = _return_residual(value, gain, parameters, **return_options)
            if not np.isfinite(result):
                raise ValueError("section velocity did not complete one winding")
            return result

        root = None
        try:
            solution = root_scalar(
                residual,
                x0=max(1.0e-6, previous * 0.999),
                x1=previous * 1.001 + 1.0e-6,
                method="secant",
                xtol=float(root_tolerance),
                rtol=float(root_tolerance),
                maxiter=40,
            )
            if solution.converged and solution.root > 0.0:
                root = float(solution.root)
        except (RuntimeError, ValueError, ZeroDivisionError):
            root = None
        if root is None:
            bracket = _bracket_root_near_previous(
                previous, gain, parameters, **return_options
            )
            root = float(
                root_scalar(
                    lambda value: _return_residual(
                        value, gain, parameters, **return_options
                    ),
                    bracket=bracket,
                    method="brentq",
                    xtol=float(root_tolerance),
                    rtol=float(root_tolerance),
                ).root
            )
        cycle = _cycle_from_velocity(root, gain, parameters, **return_options)
        cycles.append(cycle)
        previous = cycle.section_velocity
    return cycles


def minimum_full_winding_velocity(
    loop_gain: float,
    parameters: Mapping[str, Any] | None = None,
    *,
    bisection_iterations: int = 45,
    **return_options: float,
) -> float:
    """Locate the lower boundary of section states completing one winding."""

    base = _base_values(parameters)
    lower = 1.0e-7
    upper = max(base["omega_delta"], 1.0)
    while pll_andronov_return(upper, loop_gain, parameters, **return_options) is None:
        upper *= 1.5
        if upper > 1.0e6:
            raise RuntimeError("failed to find any full positive-winding section state.")
    if pll_andronov_return(lower, loop_gain, parameters, **return_options) is not None:
        return lower
    for _ in range(int(bisection_iterations)):
        midpoint = 0.5 * (lower + upper)
        if pll_andronov_return(midpoint, loop_gain, parameters, **return_options) is None:
            lower = midpoint
        else:
            upper = midpoint
    return float(upper)


def find_pll_unstable_separator(
    stable_cycle: PllRunningCycle,
    parameters: Mapping[str, Any] | None = None,
    *,
    bracket_samples: int = 160,
    root_tolerance: float = 1.0e-11,
    **return_options: float,
) -> PllRunningCycle:
    """Find the lower unstable running-cycle root without a published IC."""

    gain = float(stable_cycle.loop_gain)
    threshold = minimum_full_winding_velocity(
        gain, parameters, **return_options
    )
    lower = threshold + max(1.0e-7, 1.0e-8 * threshold)
    upper = stable_cycle.section_velocity - max(
        1.0e-7, 1.0e-8 * stable_cycle.section_velocity
    )
    if upper <= lower:
        raise RuntimeError("no interval remains below the stable running cycle.")
    fractions = np.linspace(0.0, 1.0, int(bracket_samples)) ** 2
    grid = lower + (upper - lower) * fractions
    values = [
        _return_residual(value, gain, parameters, **return_options)
        for value in grid
    ]
    bracket = None
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if not np.isfinite(f_left) or not np.isfinite(f_right):
            continue
        if f_left == 0.0 or f_left * f_right < 0.0:
            bracket = (float(left), float(right))
            break
    if bracket is None:
        raise RuntimeError("failed to bracket the unstable running-cycle separator.")
    root = float(
        root_scalar(
            lambda value: _return_residual(
                value, gain, parameters, **return_options
            ),
            bracket=bracket,
            method="brentq",
            xtol=float(root_tolerance),
            rtol=float(root_tolerance),
        ).root
    )
    cycle = _cycle_from_velocity(root, gain, parameters, **return_options)
    if cycle.stability != "unstable":
        raise RuntimeError("the lower return-map root is not an unstable separator.")
    return cycle


def pll_cycle_shifted_seed(
    cycle: PllRunningCycle, parameters: Mapping[str, Any] | None = None
) -> np.ndarray:
    """Return the cycle section point in registered shifted coordinates."""

    return pll_original_to_shifted(
        np.array([cycle.section_x, 0.0], dtype=float), parameters
    )


def integrate_pll_shifted(
    system: ChaoticSystem,
    initial_state: Sequence[float],
    *,
    t_final: float,
    output_step: float = 0.002,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-13,
    max_step: float = 0.002,
) -> tuple[np.ndarray, str]:
    """Integrate the registered shifted PLL and return ``[t,u,v]`` rows."""

    duration = float(t_final)
    if duration <= 0.0 or output_step <= 0.0:
        raise ValueError("t_final and output_step must be positive.")
    samples = max(2, int(np.ceil(duration / float(output_step))) + 1)
    times = np.linspace(0.0, duration, samples)
    result = solve_ivp(
        lambda _time, state: system.evaluate(state),
        (0.0, duration),
        np.asarray(initial_state, dtype=float),
        method="DOP853",
        t_eval=times,
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )
    trajectory = np.column_stack((result.t, result.y.T))
    return trajectory, "ok" if result.success and len(result.t) == len(times) else "failed"


def pll_reference_cycle_trajectory(
    system: ChaoticSystem,
    cycle: PllRunningCycle,
    *,
    points: int = 2001,
    rtol: float = 2.0e-12,
    atol: float = 2.0e-14,
    max_step: float = 2.0e-4,
) -> np.ndarray:
    """Integrate exactly one period of a return-map cycle in shifted coordinates."""

    seed = pll_cycle_shifted_seed(cycle, system.parameters)
    times = np.linspace(0.0, cycle.period, int(points))
    result = solve_ivp(
        lambda _time, state: system.evaluate(state),
        (0.0, cycle.period),
        seed,
        method="DOP853",
        t_eval=times,
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )
    if not result.success or len(result.t) != len(times):
        raise RuntimeError("failed to integrate the PLL reference cycle.")
    return np.column_stack((result.t, result.y.T))


def pll_cylinder_distance(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    parameters: Mapping[str, Any] | None = None,
) -> float:
    """Return the explicit dimensionless distance on ``R x S1``."""

    p = pll_lead_lag_parameters(parameters)
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != (2,) or b.shape != (2,):
        raise ValueError("cylinder distance expects two states of shape (2,).")
    scale_u = p["tau1"] / 2.0
    scale_v = np.pi
    return float(
        np.hypot((a[0] - b[0]) / scale_u, wrap_pll_angle(a[1] - b[1]) / scale_v)
    )


def _candidate_cloud_tree(
    reference_trajectory: np.ndarray,
    parameters: Mapping[str, Any] | None,
) -> cKDTree:
    p = pll_lead_lag_parameters(parameters)
    data = np.asarray(reference_trajectory, dtype=float)
    scale_u = p["tau1"] / 2.0
    scale_v = np.pi
    base = np.column_stack(
        (data[:, 1] / scale_u, wrap_pll_angle(data[:, 2]) / scale_v)
    )
    return cKDTree(
        np.vstack(
            (
                base,
                base + np.array([0.0, 2.0]),
                base - np.array([0.0, 2.0]),
            )
        )
    )


def classify_pll_trajectory(
    system: ChaoticSystem,
    trajectory: np.ndarray,
    reference_trajectory: np.ndarray,
    *,
    tail_duration: float,
    target_cloud_tolerance: float,
    equilibrium_tolerance: float,
    minimum_tail_windings: float,
) -> PllTrajectoryClassification:
    """Classify a completed shifted trajectory with the cylinder contract."""

    data = np.asarray(trajectory, dtype=float)
    reference = np.asarray(reference_trajectory, dtype=float)
    if data.ndim != 2 or data.shape[1] != 3 or len(data) < 2:
        raise ValueError("trajectory must contain [t,u,v] rows.")
    if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 2:
        raise ValueError("reference_trajectory must contain [t,u,v] rows.")
    if tail_duration <= 0.0 or tail_duration >= data[-1, 0] - data[0, 0]:
        raise ValueError("tail_duration must be positive and shorter than the trajectory.")
    equilibria = system.equilibrium_points()
    p = pll_lead_lag_parameters(system.parameters)
    scale = np.array([p["tau1"] / 2.0, np.pi], dtype=float)
    tree = _candidate_cloud_tree(reference, system.parameters)
    tail = data[:, 0] >= data[-1, 0] - float(tail_duration)
    tail_rows = data[tail]
    query = np.column_stack(
        (
            tail_rows[::5, 1] / scale[0],
            wrap_pll_angle(tail_rows[::5, 2]) / scale[1],
        )
    )
    distances, _ = tree.query(query, k=1)
    cloud_p90 = float(np.percentile(distances, 90.0))
    winding = float((tail_rows[-1, 2] - tail_rows[0, 2]) / (2.0 * np.pi))
    focus_distance = pll_cylinder_distance(
        data[-1, 1:], equilibria["E_focus"], system.parameters
    )
    saddle_distance = pll_cylinder_distance(
        data[-1, 1:], equilibria["E_saddle"], system.parameters
    )
    if abs(winding) >= minimum_tail_windings and cloud_p90 <= target_cloud_tolerance:
        final_class = "target_cycle"
    elif focus_distance <= equilibrium_tolerance:
        final_class = "equilibrium_E_focus"
    elif saddle_distance <= equilibrium_tolerance:
        final_class = "equilibrium_E_saddle"
    else:
        final_class = "other_or_unresolved"
    return PllTrajectoryClassification(
        final_class=final_class,
        target_hit=final_class == "target_cycle",
        winding_tail=winding,
        cloud_distance_p90=cloud_p90,
        focus_distance_final=focus_distance,
        saddle_distance_final=saddle_distance,
    )


def run_pll_cylindrical_hiddenness_controls(
    system: ChaoticSystem,
    reference_trajectory: np.ndarray,
    *,
    radii: Sequence[float],
    samples_per_radius: int,
    random_seed: int,
    t_final: float,
    tail_duration: float,
    output_step: float,
    max_step: float,
    target_cloud_tolerance: float,
    equilibrium_tolerance: float,
    minimum_tail_windings: float,
) -> list[PllHiddennessProbe]:
    """Sample scaled cylinder balls around both principal equilibria."""

    if samples_per_radius < 1:
        raise ValueError("samples_per_radius must be positive.")
    if tail_duration <= 0.0 or tail_duration >= t_final:
        raise ValueError("tail_duration must lie strictly between zero and t_final.")
    equilibria = system.equilibrium_points()
    if set(equilibria) != {"E_focus", "E_saddle"}:
        raise ValueError("the PLL cylinder protocol requires focus and saddle representatives.")
    p = pll_lead_lag_parameters(system.parameters)
    scale = np.array([p["tau1"] / 2.0, np.pi], dtype=float)
    rng = np.random.default_rng(int(random_seed))
    probes: list[PllHiddennessProbe] = []
    sample_id = 0
    for equilibrium_name, equilibrium in equilibria.items():
        for radius_value in radii:
            radius = float(radius_value)
            if radius <= 0.0:
                raise ValueError("hiddenness radii must be positive.")
            for _ in range(int(samples_per_radius)):
                angle = rng.uniform(0.0, 2.0 * np.pi)
                radial = radius * np.sqrt(rng.uniform())
                normalized_offset = radial * np.array([np.cos(angle), np.sin(angle)])
                initial_state = equilibrium + scale * normalized_offset
                trajectory, status = integrate_pll_shifted(
                    system,
                    initial_state,
                    t_final=float(t_final),
                    output_step=float(output_step),
                    max_step=float(max_step),
                )
                if status != "ok":
                    final_class = "numerical_failure"
                else:
                    classification = classify_pll_trajectory(
                        system,
                        trajectory,
                        reference_trajectory,
                        tail_duration=float(tail_duration),
                        target_cloud_tolerance=float(target_cloud_tolerance),
                        equilibrium_tolerance=float(equilibrium_tolerance),
                        minimum_tail_windings=float(minimum_tail_windings),
                    )
                    final_class = classification.final_class
                if status != "ok":
                    winding = float("nan")
                    cloud_p90 = float("nan")
                    focus_distance = float("nan")
                    saddle_distance = float("nan")
                else:
                    winding = classification.winding_tail
                    cloud_p90 = classification.cloud_distance_p90
                    focus_distance = classification.focus_distance_final
                    saddle_distance = classification.saddle_distance_final
                probes.append(
                    PllHiddennessProbe(
                        sample_id=sample_id,
                        equilibrium=equilibrium_name,
                        radius=radius,
                        initial_state=np.asarray(initial_state, dtype=float),
                        scaled_distance_from_equilibrium=float(np.linalg.norm(normalized_offset)),
                        status=status,
                        final_class=final_class,
                        target_hit=final_class == "target_cycle",
                        winding_tail=winding,
                        cloud_distance_p90=cloud_p90,
                        focus_distance_final=focus_distance,
                        saddle_distance_final=saddle_distance,
                    )
                )
                sample_id += 1
    return probes


def summarize_pll_hiddenness(probes: Sequence[PllHiddennessProbe]) -> dict[str, Any]:
    """Summarize finite cylinder controls without upgrading them to a proof."""

    records = list(probes)
    by_equilibrium: dict[str, dict[str, int]] = {}
    for probe in records:
        summary = by_equilibrium.setdefault(
            probe.equilibrium,
            {"n": 0, "target_hits": 0, "focus_hits": 0, "unresolved": 0, "failures": 0},
        )
        summary["n"] += 1
        summary["target_hits"] += int(probe.target_hit)
        summary["focus_hits"] += int(probe.final_class == "equilibrium_E_focus")
        summary["unresolved"] += int(probe.final_class == "other_or_unresolved")
        summary["failures"] += int(probe.final_class == "numerical_failure")
    target_hits = sum(int(probe.target_hit) for probe in records)
    failures = sum(int(probe.final_class == "numerical_failure") for probe in records)
    unresolved = sum(int(probe.final_class == "other_or_unresolved") for probe in records)
    return {
        "n_probes": len(records),
        "target_hits": target_hits,
        "numerical_failures": failures,
        "unresolved": unresolved,
        "hidden_candidate_allowed": target_hits == 0 and failures == 0 and unresolved == 0,
        "metric": "scaled_R_times_S1_nearest_neighbor_p90",
        "interpretation": "finite_equilibrium_neighborhood_test_not_global_proof",
        "by_equilibrium": by_equilibrium,
    }


def published_pll_initial_condition_to_shifted(
    x0: float,
    theta0: float,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Convert a published point only for post-localization regression."""

    return pll_original_to_shifted(np.array([x0, theta0], dtype=float), parameters)


def pll_section_velocity_from_shifted(
    state: Sequence[float], parameters: Mapping[str, Any] | None = None
) -> float:
    """Return ``theta_dot`` for a shifted state."""

    p = pll_lead_lag_parameters(parameters)
    original = pll_shifted_to_original(np.asarray(state, dtype=float), p)
    return float(
        p["omega_delta"]
        - p["loop_gain"] * original[0] / p["total_time_constant"]
        - p["tau2"]
        * p["loop_gain"]
        * np.sin(original[1])
        / (2.0 * p["total_time_constant"])
    )


__all__ = [
    "PllHiddennessProbe",
    "PllReturnEvaluation",
    "PllRunningCycle",
    "PllTrajectoryClassification",
    "classify_pll_trajectory",
    "continue_pll_running_cycle",
    "exact_zero_gain_running_cycle",
    "find_pll_unstable_separator",
    "integrate_pll_shifted",
    "minimum_full_winding_velocity",
    "pll_andronov_return",
    "pll_cylinder_distance",
    "pll_cycle_shifted_seed",
    "pll_direct_route_diagnostic",
    "pll_reference_cycle_trajectory",
    "pll_section_velocity_from_shifted",
    "published_pll_initial_condition_to_shifted",
    "run_pll_cylindrical_hiddenness_controls",
    "summarize_pll_hiddenness",
]
