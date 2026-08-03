#!/usr/bin/env python3
"""Reproduce the integer Kalman--Fitts non-Chua hidden limit cycle."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent
VERSION2 = EXAMPLE_DIR.parents[1]
ROOT = VERSION2.parent
for path in (VERSION2, ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from hidden_attractors import get_system
from hidden_attractors.analysis import integer_system_lyapunov_exponents
from hidden_attractors.plotting import (
    plot_integer_hiddenness_controls,
    plot_lyapunov_convergence,
    plot_phase_projections,
    plot_phase_space,
    plot_time_series,
    plot_trajectory_spectra,
)
from hidden_attractors.plotting.export import intercept_and_export_path
from hidden_attractors.seed_generation import (
    find_integer_lure_omega_gain_candidates_direct,
    lure_transfer_function,
)
from hidden_attractors.workflows import ContinuationPlan
from hidden_attractors.workflows.integer_lure import (
    final_integer_lure_attractor,
    integer_lure_seed,
    run_integer_lure_hiddenness_controls,
    summarize_integer_hiddenness_controls,
)
from hidden_attractors.workflows.switching_lure import (
    SwitchingMapSeed,
    continue_integer_lure_nonlinearity,
    find_sign_switching_cycle_seed,
    sign_nonlinearity,
)


CONFIG_PATH = EXAMPLE_DIR / "reproducibility.yaml"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_trajectory(path: Path, trajectory: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, trajectory, delimiter=",", header="t,x1,x2,x3,x4", comments="")


def load_config(*, config_path: Path = CONFIG_PATH, quick: bool = False) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if quick:
        cfg["continuation"]["lambda_values"] = [0.0, 0.25, 0.5, 0.75, 1.0]
        cfg["continuation"]["t_transient"] = 15.0
        cfg["continuation"]["t_keep"] = 5.0
        cfg["final_simulation"]["t_burn"] = 100.0
        cfg["final_simulation"]["t_keep"] = 80.0
        cfg["hiddenness"]["radii"] = [1.0e-4]
        cfg["hiddenness"]["samples_per_radius"] = 2
        cfg["hiddenness"]["t_final"] = 20.0
        cfg["hiddenness"]["t_burn"] = 5.0
        cfg["hiddenness"]["max_cloud_points"] = 100
        cfg["lyapunov"]["enabled"] = False
    return cfg


def output_dir(cfg: dict[str, Any]) -> Path:
    return VERSION2 / cfg["outputs"]["output_dir"]


def figure_dir(cfg: dict[str, Any]) -> Path:
    return VERSION2 / cfg["outputs"]["figures_dir"]


def build_system(cfg: dict[str, Any]):
    system = get_system(cfg["system"]["system_id"])
    configured = {key: float(value) for key, value in cfg["system"]["parameters"].items()}
    registered = {key: float(system.parameters[key]) for key in configured}
    if configured != registered:
        raise ValueError("this maintained example requires the registered 2019 parameter set")
    return system


def validate_system_declaration(system) -> dict[str, Any]:
    if system.lure is None:
        raise ValueError("the example requires an explicit scalar Lur'e declaration")
    samples = (
        np.zeros(4),
        np.array([0.37, -0.21, 0.58, -0.14]),
        np.array([-2.0, 0.4, -0.8, 0.3]),
    )
    lure_residuals = [
        float(np.linalg.norm(system.evaluate(state) - system.lure.evaluate(state)))
        for state in samples
    ]
    point = samples[1]
    delta = 1.0e-7
    basis = np.eye(system.dimension)
    finite_difference = np.column_stack(
        [
            (system.evaluate(point + delta * direction) - system.evaluate(point - delta * direction))
            / (2.0 * delta)
            for direction in basis
        ]
    )
    jacobian_residual = float(np.linalg.norm(system.jacobian_matrix(point) - finite_difference))
    equilibrium_rows = []
    for name, state in system.equilibrium_points().items():
        jacobian = system.jacobian_matrix(state)
        equilibrium_rows.append(
            {
                "name": name,
                "state": state,
                "rhs_residual": float(np.linalg.norm(system.evaluate(state))),
                "jacobian": jacobian,
                "eigenvalues": np.linalg.eigvals(jacobian),
            }
        )
    if max(lure_residuals) > 1.0e-12:
        raise RuntimeError("the Lur'e declaration does not reproduce the registered equations")
    if jacobian_residual > 2.0e-7:
        raise RuntimeError("the analytic Jacobian does not match finite differences")
    if max(row["rhs_residual"] for row in equilibrium_rows) > 1.0e-12:
        raise RuntimeError("a registered equilibrium does not satisfy the equations")
    return {
        "system": system.name,
        "dimension": system.dimension,
        "parameters": dict(system.parameters),
        "reference": dict(system.reference),
        "lure": {
            "matrix": system.lure.matrix,
            "input_vector": system.lure.input_vector,
            "output_vector": system.lure.output_vector,
            "transfer_convention": "c^T (P - s I)^(-1) b",
            "form": "exact_scalar",
        },
        "lure_field_residuals": lure_residuals,
        "max_lure_field_residual": max(lure_residuals),
        "jacobian_finite_difference_residual": jacobian_residual,
        "equilibria": equilibrium_rows,
        "status": "passed",
    }


def run_contract(cfg: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    system = context.setdefault("system", build_system(cfg))
    result = validate_system_declaration(system)
    _write_json(output_dir(cfg) / "00_system_contract.json", result)
    context["contract"] = result
    return result


def run_direct_route(cfg: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    system = context.setdefault("system", build_system(cfg))
    if "contract" not in context:
        run_contract(cfg, context)
    direct_cfg = cfg["direct_route"]
    all_pairs = find_integer_lure_omega_gain_candidates_direct(
        system.lure,
        wmin=float(direct_cfg["omega_min"]),
        wmax=float(direct_cfg["omega_max"]),
        compatible_only=False,
    )
    compatible_pairs = find_integer_lure_omega_gain_candidates_direct(
        system.lure,
        wmin=float(direct_cfg["omega_min"]),
        wmax=float(direct_cfg["omega_max"]),
        compatible_only=True,
    )
    failure = None
    try:
        integer_lure_seed(
            system,
            wmin=float(direct_cfg["omega_min"]),
            wmax=float(direct_cfg["omega_max"]),
        )
    except (RuntimeError, ValueError, IndexError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    p = system.parameters
    theoretical_omega = float(
        np.sqrt(float(p["beta"]) ** 2 + 0.5 * (float(p["m1"]) ** 2 + float(p["m2"]) ** 2))
    )
    theoretical_gain = -float(
        4.0 * float(p["beta"]) ** 2
        + (float(p["m1"]) ** 2 - float(p["m2"]) ** 2) ** 2
        / (2.0 * (2.0 * float(p["beta"]) ** 2 + float(p["m1"]) ** 2 + float(p["m2"]) ** 2))
    )
    if len(all_pairs) != 1 or compatible_pairs or failure is None:
        raise RuntimeError("the direct route did not expose the expected DF sign incompatibility")
    omega, gain = all_pairs[0]
    transfer = lure_transfer_function(omega, 1.0, system.lure)
    result = {
        "route": "direct_integer_transfer_no_grid",
        "all_transfer_candidates": all_pairs,
        "describing_function_compatible_candidates": compatible_pairs,
        "omega": omega,
        "gain": gain,
        "transfer_value": transfer,
        "nyquist_closure_residual": abs(1.0 + gain * transfer),
        "theoretical_omega": theoretical_omega,
        "theoretical_gain": theoretical_gain,
        "theoretical_differences": {
            "omega": abs(omega - theoretical_omega),
            "gain": abs(gain - theoretical_gain),
        },
        "tanh_df_range": {"lower_open": 0.0, "upper_open": 1.0 / float(p["epsilon"])},
        "direct_seed_failure": failure,
        "decision": "direct_route_rejected_gain_sign_incompatible_use_published_alternative",
        "frequency_scan_used": False,
    }
    _write_json(output_dir(cfg) / "01_direct_route_diagnostic.json", result)
    context["direct"] = result
    return result


def run_switching_seed(cfg: dict[str, Any], context: dict[str, Any]) -> SwitchingMapSeed:
    system = context.setdefault("system", build_system(cfg))
    if "direct" not in context:
        run_direct_route(cfg, context)
    source_cfg = cfg["switching_seed"]
    seed = find_sign_switching_cycle_seed(
        system,
        source_cfg["initial_section_state"],
        max_crossings=int(source_cfg["max_crossings"]),
        max_return_period=int(source_cfg["max_return_period"]),
        convergence_window=int(source_cfg["convergence_window"]),
        convergence_tolerance=float(source_cfg["convergence_tolerance"]),
        bracket_step=float(source_cfg["bracket_step"]),
        max_crossing_time=float(source_cfg["max_crossing_time"]),
        root_tolerance=float(source_cfg["root_tolerance"]),
    )
    result = {
        "method": seed.method,
        "source_nonlinearity": "sign(c^T x)",
        "initial_section_state": seed.initial_section_state,
        "generated_seed": seed.seed,
        "return_period": seed.return_period,
        "iterations": seed.iterations,
        "convergence_error": seed.convergence_error,
        "crossing_times_tail": seed.crossing_times[-2 * seed.return_period :],
        "crossing_states_tail": seed.crossing_states[-2 * seed.return_period :],
        "published_target_seed_used": False,
    }
    _write_json(output_dir(cfg) / "02_switching_seed.json", result)
    context["switching_seed"] = seed
    return seed


def run_continuation(cfg: dict[str, Any], context: dict[str, Any]):
    system = context.setdefault("system", build_system(cfg))
    seed = context.get("switching_seed") or run_switching_seed(cfg, context)
    cont_cfg = cfg["continuation"]
    plan = ContinuationPlan(
        tuple(float(value) for value in cont_cfg["lambda_values"]),
        {
            "internal_parameter": "mu",
            "source": "sign",
            "target": "tanh(sigma/epsilon)",
            "reference": "Kuznetsov et al. 2019 footnote 5",
        },
    )
    steps = continue_integer_lure_nonlinearity(
        system,
        seed,
        sign_nonlinearity,
        plan=plan,
        t_transient=float(cont_cfg["t_transient"]),
        t_keep=float(cont_cfg["t_keep"]),
        h=float(cont_cfg["h"]),
        div_threshold=float(cont_cfg["div_threshold"]),
    )
    rows = []
    for index, step in enumerate(steps):
        trajectory_path = output_dir(cfg) / "continuation_steps" / f"lambda_{index:03d}.csv"
        _write_trajectory(trajectory_path, step.trajectory)
        rows.append(
            {
                "step": index,
                "lambda_value": step.lambda_value,
                "x_in": step.x_in,
                "x_out": step.x_out,
                "status": step.status,
                "trajectory": str(trajectory_path.relative_to(output_dir(cfg))).replace("\\", "/"),
            }
        )
    if not steps or steps[-1].status != "ok" or abs(steps[-1].lambda_value - 1.0) > 1.0e-12:
        raise RuntimeError("the sign-to-tanh continuation did not reach the target system")
    _write_json(
        output_dir(cfg) / "03_continuation_trace.json",
        {
            "route": cont_cfg["route"],
            "frequency_scan_used": False,
            "steps": rows,
        },
    )
    context["continuation_steps"] = steps
    return steps


def _same_direction_crossings(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(trajectory, dtype=float)
    values = data[:, 3]
    indices = np.where((values[:-1] < 0.0) & (values[1:] >= 0.0))[0]
    times: list[float] = []
    states: list[np.ndarray] = []
    for index in indices:
        fraction = -values[index] / (values[index + 1] - values[index])
        row = data[index] + fraction * (data[index + 1] - data[index])
        times.append(float(row[0]))
        states.append(row[1:])
    return np.asarray(times, dtype=float), np.asarray(states, dtype=float)


def run_verification(cfg: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    system = context.setdefault("system", build_system(cfg))
    steps = context.get("continuation_steps") or run_continuation(cfg, context)
    final_cfg = cfg["final_simulation"]
    target_seed, trajectory, status = final_integer_lure_attractor(
        system,
        steps[-1].x_out,
        t_burn=float(final_cfg["t_burn"]),
        t_keep=float(final_cfg["t_keep"]),
        h=float(final_cfg["h"]),
        div_threshold=float(final_cfg["div_threshold"]),
    )
    _write_trajectory(output_dir(cfg) / "04_final_attractor.csv", trajectory)
    crossing_times, crossing_states = _same_direction_crossings(trajectory)
    if len(crossing_times) < 5:
        raise RuntimeError("too few Poincare returns were retained for periodicity verification")
    periods = np.diff(crossing_times[-20:])
    recurrence = np.linalg.norm(np.diff(crossing_states[-20:], axis=0), axis=1)
    published_cfg = cfg["published_regression"]
    published_point = np.asarray(published_cfg["third_cycle_point"], dtype=float)
    distances = np.linalg.norm(trajectory[:, 1:] - published_point, axis=1)
    published_distance = float(np.min(distances))

    hidden_cfg = cfg["hiddenness"]
    probes = run_integer_lure_hiddenness_controls(
        system,
        trajectory,
        radii=tuple(float(value) for value in hidden_cfg["radii"]),
        samples_per_radius=int(hidden_cfg["samples_per_radius"]),
        t_final=float(hidden_cfg["t_final"]),
        t_burn=float(hidden_cfg["t_burn"]),
        h=float(hidden_cfg["h"]),
        div_threshold=float(hidden_cfg["div_threshold"]),
        equilibrium_tol=float(hidden_cfg["equilibrium_tol"]),
        target_cloud_tol=float(hidden_cfg["target_cloud_tol"]),
        max_cloud_points=int(hidden_cfg["max_cloud_points"]),
        random_seed=int(hidden_cfg["random_seed"]),
        sampling_mode=str(hidden_cfg["sampling_mode"]),
    )
    hidden_summary = summarize_integer_hiddenness_controls(probes)
    probe_rows = [
        {
            "sample_id": probe.sample_id,
            "equilibrium": probe.equilibrium,
            "radius": probe.radius,
            "x0": probe.x0,
            "distance_from_equilibrium": probe.distance_from_equilibrium,
            "status": probe.status,
            "final_class": probe.final_class,
            "target_hit": probe.target_hit,
            "cloud_distance_norm": probe.cloud_distance_norm,
        }
        for probe in probes
    ]
    initial_csv = output_dir(cfg) / "05_hiddenness_initial_conditions.csv"
    initial_csv.parent.mkdir(parents=True, exist_ok=True)
    with initial_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "equilibrium", "radius", "x1", "x2", "x3", "x4"])
        for probe in probes:
            writer.writerow([probe.sample_id, probe.equilibrium, probe.radius, *probe.x0.tolist()])

    lyapunov = None
    lyap_cfg = cfg["lyapunov"]
    if bool(lyap_cfg.get("enabled", False)):
        lyapunov = integer_system_lyapunov_exponents(
            system,
            target_seed,
            h=float(lyap_cfg["h"]),
            t_final=float(lyap_cfg["t_final"]),
            t_burn=float(lyap_cfg["t_burn"]),
            reorthonormalize_every=int(lyap_cfg["reorthonormalize_every"]),
            div_threshold=float(lyap_cfg["div_threshold"]),
        )
    full_run = not bool(context.get("quick", False))
    regression_passed = bool(
        (not full_run)
        or published_distance <= float(published_cfg["point_distance_tolerance"])
    )
    if status != "ok" or hidden_summary["target_hits"] != 0 or not regression_passed:
        raise RuntimeError("the target, hiddenness, or published-point regression contract failed")
    result = {
        "final_status": status,
        "target_seed": target_seed,
        "candidate_type": "stable_periodic_limit_cycle",
        "poincare": {
            "n_same_direction_returns": len(crossing_times),
            "period_mean": float(np.mean(periods)),
            "period_std": float(np.std(periods)),
            "return_state_error_max": float(np.max(recurrence)),
            "return_states_tail": crossing_states[-5:],
        },
        "published_regression": {
            "role": published_cfg["role"],
            "point": published_point,
            "minimum_trajectory_distance": published_distance,
            "tolerance": float(published_cfg["point_distance_tolerance"]),
            "passed": regression_passed,
        },
        "hiddenness": {
            "summary": hidden_summary,
            "probes": probe_rows,
            "interpretation": "finite_equilibrium_neighborhood_test_not_global_proof",
        },
        "lyapunov": None
        if lyapunov is None
        else {
            "status": lyapunov.status,
            "exponents": lyapunov.exponents,
            "interpretation": "finite_time_limit_cycle_diagnostic_expected_one_near_zero_exponent",
        },
    }
    _write_json(output_dir(cfg) / "05_verification_summary.json", result)
    context.update(
        {
            "target_seed": target_seed,
            "trajectory": trajectory,
            "probes": probes,
            "lyapunov": lyapunov,
            "verification": result,
        }
    )
    return result


def _plot_direct_incompatibility(system, direct: dict[str, Any], path: Path) -> None:
    omega0 = float(direct["omega"])
    frequencies = np.linspace(max(0.02, 0.25 * omega0), 1.8 * omega0, 700)
    values = np.asarray([lure_transfer_function(omega, 1.0, system.lure) for omega in frequencies])
    amplitudes = np.logspace(-4, 2, 500)
    dfs = np.asarray([float(system.lure.describing_function(amplitude)) for amplitude in amplitudes])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(frequencies, values.imag, color="#1d4ed8", label="Im W(i omega)")
    axes[0].axhline(0.0, color="#6b7280", lw=0.8)
    axes[0].axvline(omega0, color="#dc2626", ls="--", label=f"direct root {omega0:.6f}")
    axes[0].set_xlabel("omega")
    axes[0].set_ylabel("imaginary transfer component")
    axes[0].set_title("Diagnostic visualization only")
    axes[0].legend(fontsize=8)
    axes[1].semilogx(amplitudes, dfs, color="#15803d", label="tanh describing function")
    axes[1].axhline(float(direct["gain"]), color="#dc2626", ls="--", label=f"required k={direct['gain']:.6f}")
    axes[1].set_xlabel("amplitude")
    axes[1].set_ylabel("gain")
    axes[1].set_title("No amplitude closure: positive DF vs negative k")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    intercept_and_export_path(fig, path, "lure_transfer")
    plt.close(fig)


def _plot_switching_convergence(seed: SwitchingMapSeed, path: Path) -> None:
    period = seed.return_period
    errors = np.full(seed.iterations, np.nan)
    for index in range(period, seed.iterations):
        errors[index] = np.linalg.norm(seed.crossing_states[index] - seed.crossing_states[index - period])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.semilogy(np.arange(seed.iterations), np.maximum(errors, 1.0e-16), color="#7c3aed")
    ax.axhline(seed.convergence_error, color="#dc2626", ls="--", lw=0.8)
    ax.set_xlabel("switching-map iteration")
    ax.set_ylabel(f"period-{period} return error")
    ax.set_title("Generated sign-cycle seed convergence")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    intercept_and_export_path(fig, path, "poincare")
    plt.close(fig)


def _plot_continuation(steps, path: Path) -> None:
    values = np.asarray([step.lambda_value for step in steps])
    states = np.asarray([step.x_out for step in steps])
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for index in range(states.shape[1]):
        ax.plot(values, states[:, index], marker="o", ms=2.4, lw=0.8, label=f"x{index + 1}")
    ax.set_xlabel("lambda: sign to tanh")
    ax.set_ylabel("terminal state")
    ax.set_title("Nonlinearity continuation trace")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    intercept_and_export_path(fig, path, "continuation")
    plt.close(fig)


def run_figures(cfg: dict[str, Any], context: dict[str, Any]) -> None:
    system = context.setdefault("system", build_system(cfg))
    direct = context.get("direct") or run_direct_route(cfg, context)
    seed = context.get("switching_seed") or run_switching_seed(cfg, context)
    steps = context.get("continuation_steps") or run_continuation(cfg, context)
    if "verification" not in context:
        run_verification(cfg, context)
    trajectory = context["trajectory"]
    probes = context["probes"]
    figures = figure_dir(cfg)
    figures.mkdir(parents=True, exist_ok=True)
    _plot_direct_incompatibility(system, direct, figures / "01_direct_route_incompatibility.png")
    _plot_switching_convergence(seed, figures / "02_switching_map_convergence.png")
    _plot_continuation(steps, figures / "03_sign_to_tanh_continuation.png")
    plot_phase_space(
        trajectory,
        figures / "04_hidden_limit_cycle_3d.png",
        title="Kalman--Fitts hidden limit cycle (x1,x2,x3)",
    )
    plot_phase_projections(
        trajectory,
        figures / "05_hidden_limit_cycle_projections.png",
        title="Kalman--Fitts phase projections",
    )
    plot_time_series(
        trajectory,
        figures / "06_hidden_limit_cycle_timeseries.png",
        columns=(1, 2, 3, 4),
        title="Kalman--Fitts target trajectory",
    )
    plot_integer_hiddenness_controls(
        trajectory,
        probes,
        figures / "07_equilibrium_neighborhood_controls.png",
        title="Finite hiddenness controls around the stable equilibrium",
    )
    plot_trajectory_spectra(trajectory, figures, method="fft", prefix="08_kalman_fitts")
    plot_trajectory_spectra(trajectory, figures, method="psd", prefix="09_kalman_fitts")
    if context.get("lyapunov") is not None:
        plot_lyapunov_convergence(
            context["lyapunov"], figures / "10_lyapunov_convergence.png"
        )


def run_selected(
    steps: list[str],
    *,
    quick: bool = False,
    output_override: Path | None = None,
    config_path: Path = CONFIG_PATH,
) -> None:
    cfg = load_config(config_path=config_path, quick=quick)
    if output_override is not None:
        destination = output_override.resolve()
        cfg["outputs"]["output_dir"] = str(destination)
        cfg["outputs"]["figures_dir"] = str(destination / "figures")
    context: dict[str, Any] = {"quick": quick}
    operations = {
        "contract": run_contract,
        "direct": run_direct_route,
        "source": run_switching_seed,
        "continuation": run_continuation,
        "verification": run_verification,
        "figures": run_figures,
    }
    timings: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for step in steps:
        started = time.perf_counter()
        operations[step](cfg, context)
        timings.append({"phase": step, "seconds": time.perf_counter() - started})
    total_seconds = time.perf_counter() - total_started
    _write_json(
        output_dir(cfg) / "run_manifest.json",
        {
            "case_id": cfg["case_id"],
            "steps": steps,
            "quick": quick,
            "direct_route_decision": context.get("direct", {}).get("decision"),
            "alternative_route": cfg["continuation"]["route"],
            "published_target_seed_used": False,
            "frequency_scan_used": False,
            "phase_timings": timings,
            "total_seconds": total_seconds,
            "python": sys.version,
            "platform": platform.platform(),
        },
    )
    print(f"total_seconds={total_seconds:.9f}")
    print(f"output_dir={output_dir(cfg)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["contract", "direct", "source", "continuation", "verification", "figures"],
        default=["contract", "direct", "source", "continuation", "verification", "figures"],
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_selected(
        args.steps,
        quick=args.quick,
        output_override=args.output_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
