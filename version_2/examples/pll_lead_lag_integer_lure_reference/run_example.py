#!/usr/bin/env python3
"""Reproduce the integer two-phase lead-lag PLL hidden running cycle."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict
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
from hidden_attractors.plotting.export import intercept_and_export_path
from hidden_attractors.seed_generation import lure_transfer_function
from hidden_attractors.systems.pll_lead_lag import (
    pll_lead_lag_parameters,
    pll_lure_transfer,
    pll_original_rhs,
    pll_original_to_shifted,
    pll_shifted_sine_harmonics,
    pll_shifted_to_original,
    wrap_pll_angle,
)
from hidden_attractors.workflows.integer_lure import integer_lure_seed
from hidden_attractors.workflows.pll_lead_lag import (
    PllRunningCycle,
    classify_pll_trajectory,
    continue_pll_running_cycle,
    find_pll_unstable_separator,
    integrate_pll_shifted,
    pll_cylinder_distance,
    pll_cycle_shifted_seed,
    pll_direct_route_diagnostic,
    pll_reference_cycle_trajectory,
    published_pll_initial_condition_to_shifted,
    run_pll_cylindrical_hiddenness_controls,
    summarize_pll_hiddenness,
)


CONFIG_PATH = EXAMPLE_DIR / "reproducibility.yaml"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, PllRunningCycle):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_trajectory(path: Path, trajectory: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, trajectory, delimiter=",", header="t,u,v", comments="")


def load_config(*, config_path: Path = CONFIG_PATH, quick: bool = False) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["runtime"] = {
        "quick": bool(quick),
        "continuation_steps": int(
            cfg["continuation"]["quick_steps"] if quick else cfg["continuation"]["steps"]
        ),
    }
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
        raise ValueError("this maintained example requires the registered 2015 parameter set")
    return system


def validate_system_declaration(system) -> dict[str, Any]:
    if system.lure is None:
        raise ValueError("the PLL example requires an explicit scalar Lur'e declaration")
    samples = (
        np.zeros(2),
        np.array([0.0037, -0.21]),
        np.array([-0.012, 1.4]),
    )
    lure_residuals = [
        float(np.linalg.norm(system.evaluate(state) - system.lure.evaluate(state)))
        for state in samples
    ]
    point = samples[1]
    delta = 1.0e-7
    finite_difference = np.column_stack(
        [
            (system.evaluate(point + delta * direction) - system.evaluate(point - delta * direction))
            / (2.0 * delta)
            for direction in np.eye(2)
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
    original = np.array([0.007, 0.43], dtype=float)
    shifted = pll_original_to_shifted(original, system.parameters)
    original_field = pll_original_rhs(original, system.parameters)
    shifted_field = system.evaluate(shifted)
    coordinate_field_residual = float(np.linalg.norm(original_field - shifted_field))
    spectral_point = complex(-0.3, 17.0)
    declared_transfer = pll_lure_transfer(
        spectral_point, system.parameters, convention="code"
    )
    matrix_transfer = complex(
        system.lure.output_vector
        @ np.linalg.solve(
            system.lure.matrix - spectral_point * np.eye(2),
            system.lure.input_vector,
        )
    )
    transfer_residual = abs(declared_transfer - matrix_transfer)
    if max(lure_residuals) > 1.0e-12:
        raise RuntimeError("the shifted Lur'e declaration does not reproduce the PLL equations")
    if jacobian_residual > 3.0e-7:
        raise RuntimeError("the analytic PLL Jacobian does not match finite differences")
    if max(row["rhs_residual"] for row in equilibrium_rows) > 1.0e-11:
        raise RuntimeError("a principal cylinder equilibrium does not satisfy the equations")
    if coordinate_field_residual > 1.0e-11 or transfer_residual > 1.0e-10:
        raise RuntimeError("the T1 coordinate or transfer declaration is inconsistent")
    p = pll_lead_lag_parameters(system.parameters)
    return {
        "system": system.name,
        "dimension": system.dimension,
        "parameters": dict(system.parameters),
        "derived": p,
        "reference": dict(system.reference),
        "state_coordinates": system.metadata["state_coordinates"],
        "state_space": system.metadata["state_space"],
        "cylinder_scales": system.metadata["cylinder_scales"],
        "lure": {
            "matrix": system.lure.matrix,
            "input_vector": system.lure.input_vector,
            "output_vector": system.lure.output_vector,
            "nonlinearity": "sin(theta_focus+v)-sin(theta_focus)",
            "transfer_standard": "c^T(sI-A)^(-1)b",
            "transfer_code": "c^T(A-sI)^(-1)b",
            "form": "T1_exact_scalar_after_locked_equilibrium_shift",
        },
        "centered_df_at_amplitude_1": pll_shifted_sine_harmonics(1.0, system.parameters),
        "max_lure_field_residual": max(lure_residuals),
        "jacobian_finite_difference_residual": jacobian_residual,
        "coordinate_field_residual": coordinate_field_residual,
        "transfer_residual": transfer_residual,
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
    result = pll_direct_route_diagnostic(system)
    failure = None
    try:
        integer_lure_seed(
            system,
            wmin=0.0,
            wmax=float("inf"),
        )
    except (RuntimeError, ValueError, IndexError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    if failure is None or result["decision"] != cfg["direct_route"]["expected_decision"]:
        raise RuntimeError("the direct PLL route did not stop at the expected analytic obstruction")
    result["direct_seed_failure"] = failure
    result["fallback_route"] = cfg["direct_route"]["fallback_route"]
    _write_json(output_dir(cfg) / "01_direct_route_diagnostic.json", result)
    context["direct"] = result
    return result


def _return_options(cfg: dict[str, Any]) -> dict[str, float]:
    source = cfg["andronov_return"]
    return {
        "rtol": float(source["rtol"]),
        "atol": float(source["atol"]),
        "max_step_angle": float(source["max_step_angle"]),
        "minimum_velocity": float(source["minimum_velocity"]),
    }


def run_localization(cfg: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    system = context.setdefault("system", build_system(cfg))
    if "direct" not in context:
        run_direct_route(cfg, context)
    continuation_cfg = cfg["continuation"]
    schedule = np.linspace(
        float(continuation_cfg["start_loop_gain"]),
        float(continuation_cfg["stop_loop_gain"]),
        int(cfg["runtime"]["continuation_steps"]),
    )
    options = _return_options(cfg)
    cycles = continue_pll_running_cycle(
        schedule,
        system.parameters,
        root_tolerance=float(cfg["andronov_return"]["root_tolerance"]),
        **options,
    )
    stable_cycle = cycles[-1]
    separator = find_pll_unstable_separator(
        stable_cycle,
        system.parameters,
        bracket_samples=int(cfg["andronov_return"]["separator_bracket_samples"]),
        root_tolerance=float(cfg["andronov_return"]["root_tolerance"]),
        **options,
    )
    trace_path = output_dir(cfg) / "02_loop_gain_continuation.csv"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "loop_gain",
                "section_velocity",
                "section_x",
                "period",
                "multiplier",
                "return_residual",
                "stability",
            ]
        )
        for index, cycle in enumerate(cycles):
            writer.writerow(
                [
                    index,
                    cycle.loop_gain,
                    cycle.section_velocity,
                    cycle.section_x,
                    cycle.period,
                    cycle.multiplier,
                    cycle.return_residual,
                    cycle.stability,
                ]
            )
    result = {
        "route": continuation_cfg["route"],
        "schedule": {
            "start": float(schedule[0]),
            "stop": float(schedule[-1]),
            "steps": len(schedule),
        },
        "analytic_zero_gain_cycle": asdict(cycles[0]),
        "stable_target_cycle": asdict(stable_cycle),
        "unstable_separator": asdict(separator),
        "stable_shifted_seed": pll_cycle_shifted_seed(stable_cycle, system.parameters),
        "separator_shifted_seed": pll_cycle_shifted_seed(separator, system.parameters),
        "published_initial_conditions_used": False,
        "frequency_scan_used": False,
    }
    _write_json(output_dir(cfg) / "02_cycle_localization.json", result)
    context.update(
        {
            "continuation_cycles": cycles,
            "stable_cycle": stable_cycle,
            "separator": separator,
            "localization": result,
        }
    )
    return result


def _write_probe_csv(path: Path, probes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "equilibrium",
                "radius_scaled",
                "u0",
                "v0",
                "scaled_distance_from_equilibrium",
                "status",
                "final_class",
                "target_hit",
                "winding_tail",
                "cloud_distance_p90",
                "focus_distance_final",
                "saddle_distance_final",
            ]
        )
        for probe in probes:
            writer.writerow(
                [
                    probe.sample_id,
                    probe.equilibrium,
                    probe.radius,
                    *probe.initial_state.tolist(),
                    probe.scaled_distance_from_equilibrium,
                    probe.status,
                    probe.final_class,
                    probe.target_hit,
                    probe.winding_tail,
                    probe.cloud_distance_p90,
                    probe.focus_distance_final,
                    probe.saddle_distance_final,
                ]
            )


def run_verification(cfg: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    system = context.setdefault("system", build_system(cfg))
    if "localization" not in context:
        run_localization(cfg, context)
    stable_cycle = context["stable_cycle"]
    separator = context["separator"]
    reference_cfg = cfg["reference_cycle"]
    stable_reference = pll_reference_cycle_trajectory(
        system,
        stable_cycle,
        points=int(reference_cfg["points"]),
        rtol=float(reference_cfg["rtol"]),
        atol=float(reference_cfg["atol"]),
        max_step=float(reference_cfg["max_step"]),
    )
    separator_reference = pll_reference_cycle_trajectory(
        system,
        separator,
        points=int(reference_cfg["points"]),
        rtol=float(reference_cfg["rtol"]),
        atol=float(reference_cfg["atol"]),
        max_step=float(reference_cfg["max_step"]),
    )
    stable_return_error = pll_cylinder_distance(
        stable_reference[0, 1:], stable_reference[-1, 1:], system.parameters
    )
    separator_return_error = pll_cylinder_distance(
        separator_reference[0, 1:], separator_reference[-1, 1:], system.parameters
    )
    if max(stable_return_error, separator_return_error) > float(reference_cfg["return_tolerance"]):
        raise RuntimeError("a return-map cycle failed the direct one-period integration check")
    _write_trajectory(output_dir(cfg) / "03_stable_reference_cycle.csv", stable_reference)
    _write_trajectory(output_dir(cfg) / "03_unstable_separator_cycle.csv", separator_reference)

    hidden_cfg = cfg["hiddenness"]
    candidate, candidate_status = integrate_pll_shifted(
        system,
        pll_cycle_shifted_seed(stable_cycle, system.parameters),
        t_final=float(hidden_cfg["t_final"]),
        output_step=float(hidden_cfg["output_step"]),
        max_step=float(hidden_cfg["max_step"]),
    )
    candidate_class = classify_pll_trajectory(
        system,
        candidate,
        stable_reference,
        tail_duration=float(hidden_cfg["tail_duration"]),
        target_cloud_tolerance=float(hidden_cfg["target_cloud_tolerance"]),
        equilibrium_tolerance=float(hidden_cfg["equilibrium_tolerance"]),
        minimum_tail_windings=float(hidden_cfg["minimum_tail_windings"]),
    )
    _write_trajectory(output_dir(cfg) / "04_final_attractor.csv", candidate)

    probes = run_pll_cylindrical_hiddenness_controls(
        system,
        stable_reference,
        radii=tuple(float(value) for value in hidden_cfg["radii"]),
        samples_per_radius=int(hidden_cfg["samples_per_radius"]),
        random_seed=int(hidden_cfg["random_seed"]),
        t_final=float(hidden_cfg["t_final"]),
        tail_duration=float(hidden_cfg["tail_duration"]),
        output_step=float(hidden_cfg["output_step"]),
        max_step=float(hidden_cfg["max_step"]),
        target_cloud_tolerance=float(hidden_cfg["target_cloud_tolerance"]),
        equilibrium_tolerance=float(hidden_cfg["equilibrium_tolerance"]),
        minimum_tail_windings=float(hidden_cfg["minimum_tail_windings"]),
    )
    hidden_summary = summarize_pll_hiddenness(probes)
    _write_probe_csv(output_dir(cfg) / "05_hiddenness_probes.csv", probes)

    regression_rows = []
    for case in cfg["published_regression"]["cases"]:
        initial_state = published_pll_initial_condition_to_shifted(
            float(case["x0"]), float(case["theta0"]), system.parameters
        )
        trajectory, status = integrate_pll_shifted(
            system,
            initial_state,
            t_final=float(hidden_cfg["t_final"]),
            output_step=float(hidden_cfg["output_step"]),
            max_step=float(hidden_cfg["max_step"]),
        )
        classification = classify_pll_trajectory(
            system,
            trajectory,
            stable_reference,
            tail_duration=float(hidden_cfg["tail_duration"]),
            target_cloud_tolerance=float(hidden_cfg["target_cloud_tolerance"]),
            equilibrium_tolerance=float(hidden_cfg["equilibrium_tolerance"]),
            minimum_tail_windings=float(hidden_cfg["minimum_tail_windings"]),
        )
        regression_rows.append(
            {
                "x0": float(case["x0"]),
                "theta0": float(case["theta0"]),
                "shifted_initial_state": initial_state,
                "expected": str(case["expected"]),
                "observed": classification.final_class,
                "passed": status == "ok" and classification.final_class == str(case["expected"]),
                "classification": asdict(classification),
            }
        )

    expected_count = int(hidden_cfg["expected_probe_count"])
    if len(probes) != expected_count:
        raise RuntimeError(f"expected exactly {expected_count} cylinder probes")
    if candidate_status != "ok" or not candidate_class.target_hit:
        raise RuntimeError("the generated return-map seed did not reproduce the target cycle")
    if not hidden_summary["hidden_candidate_allowed"]:
        raise RuntimeError("the finite cylinder hiddenness contract found a target contact or failure")
    if not all(row["passed"] for row in regression_rows):
        raise RuntimeError("a post-derivation published initial-condition regression failed")
    result = {
        "final_status": "ok",
        "candidate_type": "stable_periodic_running_cycle_on_R_times_S1",
        "stable_cycle": asdict(stable_cycle),
        "unstable_separator": asdict(separator),
        "one_period_checks": {
            "stable_cylinder_return_error": stable_return_error,
            "separator_cylinder_return_error": separator_return_error,
        },
        "candidate_classification": asdict(candidate_class),
        "hiddenness": {
            "summary": hidden_summary,
            "radii": hidden_cfg["radii"],
            "samples_per_radius_per_equilibrium": hidden_cfg["samples_per_radius"],
            "equilibria": hidden_cfg["required_equilibrium_classes"],
            "metric": hidden_cfg["metric"],
            "interpretation": "hiddenness_supported_under_tested_cylindrical_neighborhoods_not_global_proof",
        },
        "published_regression": {
            "role": cfg["published_regression"]["role"],
            "cases": regression_rows,
        },
    }
    _write_json(output_dir(cfg) / "06_verification_summary.json", result)
    context.update(
        {
            "stable_reference": stable_reference,
            "separator_reference": separator_reference,
            "candidate": candidate,
            "probes": probes,
            "verification": result,
        }
    )
    return result


def _export(fig, path: Path, kind: str, system) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    intercept_and_export_path(
        fig,
        path,
        kind,
        {
            "source_script": "examples/pll_lead_lag_integer_lure_reference/run_example.py",
            "system_id": system.name,
            "q": 1.0,
            "integrator": "DOP853",
            "memory_mode": "none",
            "parameters": dict(system.parameters),
        },
    )
    plt.close(fig)


def run_figures(cfg: dict[str, Any], context: dict[str, Any]) -> None:
    system = context.setdefault("system", build_system(cfg))
    direct = context.get("direct") or run_direct_route(cfg, context)
    if "verification" not in context:
        run_verification(cfg, context)
    figures = figure_dir(cfg)

    frequencies = np.logspace(-2.0, 4.0, 900)
    standard = np.asarray(
        [pll_lure_transfer(1j * value, system.parameters, convention="standard") for value in frequencies]
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.loglog(frequencies, standard.imag, color="#1d4ed8", label="Im G(i omega) > 0")
    ax.set_xlabel("omega")
    ax.set_ylabel("imaginary transfer component")
    ax.text(
        0.03,
        0.07,
        "Visualization after analytic rejection; no search grid used",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.legend(fontsize=8)
    _export(fig, figures / "01_direct_transfer_obstruction.png", "lure_transfer", system)

    cycles = context["continuation_cycles"]
    gains = np.asarray([cycle.loop_gain for cycle in cycles])
    velocities = np.asarray([cycle.section_velocity for cycle in cycles])
    section_x = np.asarray([cycle.section_x for cycle in cycles])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    axes[0].plot(gains, velocities, color="#7c3aed", marker="o", ms=2.2, lw=1.0)
    axes[0].set_xlabel("loop gain L")
    axes[0].set_ylabel("section velocity")
    axes[1].plot(gains, section_x, color="#15803d", marker="o", ms=2.2, lw=1.0)
    axes[1].set_xlabel("loop gain L")
    axes[1].set_ylabel("section x at theta=0")
    fig.tight_layout()
    _export(fig, figures / "02_loop_gain_continuation.png", "continuation", system)

    p = pll_lead_lag_parameters(system.parameters)
    stable = context["stable_reference"]
    separator = context["separator_reference"]
    stable_original = np.asarray(
        [pll_shifted_to_original(row[1:], system.parameters) for row in stable]
    )
    separator_original = np.asarray(
        [pll_shifted_to_original(row[1:], system.parameters) for row in separator]
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(
        np.mod(stable_original[:-1, 1], 2.0 * np.pi),
        stable_original[:-1, 0],
        color="#1d4ed8",
        lw=1.8,
        label="stable running cycle",
    )
    ax.plot(
        np.mod(separator_original[:-1, 1], 2.0 * np.pi),
        separator_original[:-1, 0],
        color="#dc2626",
        lw=1.4,
        ls="--",
        label="unstable separator",
    )
    ax.scatter(
        [p["theta_focus"], p["theta_saddle"]],
        [p["x_equilibrium"], p["x_equilibrium"]],
        c=["#15803d", "#111827"],
        s=34,
        label="locked equilibria",
        zorder=5,
    )
    ax.set_xlabel("theta_delta mod 2 pi")
    ax.set_ylabel("loop-filter state x")
    ax.legend(fontsize=8)
    _export(fig, figures / "03_phase_cylinder_cycles.png", "poincare", system)

    probes = context["probes"]
    fig, ax = plt.subplots(figsize=(7.3, 4.7))
    for equilibrium, color in (("E_focus", "#15803d"), ("E_saddle", "#f59e0b")):
        selected = np.asarray(
            [probe.initial_state for probe in probes if probe.equilibrium == equilibrium]
        )
        ax.scatter(
            wrap_pll_angle(selected[:, 1]),
            selected[:, 0],
            s=12,
            alpha=0.65,
            color=color,
            label=f"48 probes: {equilibrium}",
        )
    ax.scatter([0.0, p["saddle_offset"]], [0.0, 0.0], c="#111827", s=28, zorder=5)
    ax.set_xlabel("wrapped shifted phase v")
    ax.set_ylabel("shifted filter state u")
    ax.legend(fontsize=8)
    _export(fig, figures / "04_cylindrical_hiddenness_probes.png", "hiddenness", system)

    # Record explicitly that the frequency grid above was only a plot aid.
    direct["plot_grid_role"] = "visualization_only_after_analytic_rejection"


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
        "localization": run_localization,
        "verification": run_verification,
        "figures": run_figures,
    }
    timings = []
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
            "continuation_steps": cfg["runtime"]["continuation_steps"],
            "direct_route_decision": context.get("direct", {}).get("decision"),
            "alternative_route": cfg["continuation"]["route"],
            "published_initial_conditions_used_for_seed": False,
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
        choices=["contract", "direct", "localization", "verification", "figures"],
        default=["contract", "direct", "localization", "verification", "figures"],
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
