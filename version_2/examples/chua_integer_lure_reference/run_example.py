#!/usr/bin/env python3
"""Reproducible integer-order Chua Lur'e validation example."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    plot_integer_lure_continuation,
    plot_lyapunov_convergence,
    plot_lure_nyquist_describing_function,
    plot_lure_transfer_components,
    plot_phase_projections,
    plot_phase_space,
    plot_trajectory_spectra,
)
from hidden_attractors.seed_generation.core import HarmonicSeed
from hidden_attractors.seed_generation.lure import lure_transfer_function
from hidden_attractors.workflows.integer_lure import (
    IntegerHiddennessProbe,
    continue_integer_lure_seed,
    final_integer_lure_attractor,
    integer_lure_seed,
    run_integer_lure_hiddenness_controls,
    summarize_integer_hiddenness_controls,
)

CONFIG_PATH = EXAMPLE_DIR / "reproducibility.yaml"
REFERENCE_PATH = VERSION2 / "validation" / "references" / "kuznetsov2017_expected.json"


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
    np.savetxt(path, trajectory, delimiter=",", header="t,x,y,z", comments="")


def _seed_payload(seed: HarmonicSeed) -> dict[str, Any]:
    return {
        "seed": seed.seed,
        "eigenvector": [[float(z.real), float(z.imag)] for z in seed.eigenvector],
        "matched_eigenvalue": [float(seed.matched_eigenvalue.real), float(seed.matched_eigenvalue.imag)],
        "omega": seed.omega,
        "gain": seed.gain,
        "amplitude": seed.amplitude,
        "branch_index": seed.branch_index,
        "method": seed.method,
        "mu": seed.mu,
        "search_route": seed.search_route,
        "interpretation": "describing_function_seed_only",
    }


def load_config(
    *,
    config_path: Path = CONFIG_PATH,
    quick: bool = False,
) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if quick:
        cfg["continuation"]["lambda_values"] = [0.0, 0.25, 0.5, 0.75, 1.0]
        cfg["continuation"]["t_transient"] = 2.0
        cfg["continuation"]["t_keep"] = 2.0
        cfg["final_simulation"]["t_burn"] = 2.0
        cfg["final_simulation"]["t_keep"] = 4.0
        cfg["hiddenness"]["radii"] = [1.0e-4]
        cfg["hiddenness"]["samples_per_radius"] = 2
        cfg["hiddenness"]["t_final"] = 2.0
        cfg["hiddenness"]["t_burn"] = 0.5
        cfg["hiddenness"]["target_cloud_tol"] = 0.2
        cfg["lyapunov"]["t_final"] = 1.0
        cfg["lyapunov"]["t_burn"] = 0.2
    return cfg


def output_dir(cfg: dict[str, Any]) -> Path:
    return VERSION2 / cfg["outputs"]["output_dir"]


def figure_dir(cfg: dict[str, Any]) -> Path:
    return VERSION2 / cfg["outputs"]["figures_dir"]


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(VERSION2).as_posix()
    except ValueError:
        return str(path)


def build_system(cfg: dict[str, Any]):
    system = get_system(cfg["system"]["system_id"])
    merged = dict(system.parameters)
    merged.update(cfg["system"]["parameters"])
    return replace(system, parameters=merged)


def validate_system_declaration(system) -> dict[str, Any]:
    """Validate equations, Jacobian, equilibria, and the declared Lur'e split."""

    if system.lure is None:
        raise ValueError("the integer reference requires an explicit Lur'e declaration")
    sample_states = (
        np.array([0.0, 0.0, 0.0]),
        np.array([0.37, -0.21, 0.58]),
        np.array([2.0, -0.4, 0.8]),
    )
    lure_residuals = [
        float(np.linalg.norm(system.evaluate(state) - system.lure.evaluate(state)))
        for state in sample_states
    ]
    equilibria = system.equilibrium_points()
    equilibrium_rows = []
    for name, state in equilibria.items():
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

    point = sample_states[1]
    epsilon = 1.0e-7
    finite_difference = np.column_stack(
        [
            (
                system.evaluate(point + epsilon * np.eye(system.dimension)[index])
                - system.evaluate(point - epsilon * np.eye(system.dimension)[index])
            )
            / (2.0 * epsilon)
            for index in range(system.dimension)
        ]
    )
    jacobian_residual = float(
        np.linalg.norm(system.jacobian_matrix(point) - finite_difference)
    )
    if max(lure_residuals) > 1.0e-12:
        raise RuntimeError("declared Lur'e field does not reproduce the registered equations")
    if max(row["rhs_residual"] for row in equilibrium_rows) > 1.0e-10:
        raise RuntimeError("registered equilibrium does not satisfy the equations")
    if jacobian_residual > 1.0e-7:
        raise RuntimeError("analytic Jacobian does not match finite differences")
    return {
        "system": system.name,
        "parameters": dict(system.parameters),
        "lure": {
            "matrix": system.lure.matrix,
            "input_vector": system.lure.input_vector,
            "output_vector": system.lure.output_vector,
            "transfer_convention": "c^T (P - s I)^(-1) b",
        },
        "lure_field_residuals": lure_residuals,
        "max_lure_field_residual": max(lure_residuals),
        "equilibria": equilibrium_rows,
        "jacobian_finite_difference_residual": jacobian_residual,
        "status": "passed",
    }


def run_search(cfg: dict[str, Any], context: dict[str, Any]) -> HarmonicSeed:
    system = context.setdefault("system", build_system(cfg))
    declaration = validate_system_declaration(system)
    _write_json(output_dir(cfg) / "00_system_contract.json", declaration)
    seed_cfg = cfg["seed_search"]
    requested_fallback = seed_cfg.get("fallback_route")
    seed = integer_lure_seed(
        system,
        branch_index=int(seed_cfg["branch_index"]),
        method=str(seed_cfg["method"]),
        wmin=float(seed_cfg["omega_min"]),
        wmax=float(seed_cfg["omega_max"]),
        search_route=str(seed_cfg.get("route", "direct_integer_transfer")),
        fallback_route=requested_fallback,
    )
    transfer_value = lure_transfer_function(seed.omega, 1.0, system.lure)
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    payload = _seed_payload(seed)
    payload.update(
        {
            "transfer_value": transfer_value,
            "nyquist_closure_residual": abs(1.0 + seed.gain * transfer_value),
            "describing_function_residual": abs(
                float(system.lure.describing_function(seed.amplitude)) - seed.gain
            ),
            "reference_role": "post_derivation_regression_only_not_seed_input",
            "reference_sources": [
                "validation/reference_cases/chua_integer_q1/08_literature_comparison/sources/chua_entero_algebraico_sin_numericos.wl",
                "validation/reference_cases/chua_integer_q1/08_literature_comparison/sources/verifica_chua_entero.m",
            ],
            "reference_differences": {
                "omega0": abs(seed.omega - float(reference["omega0"])),
                "k": abs(seed.gain - float(reference["k"])),
                "a0": abs(seed.amplitude - float(reference["a0"])),
                "seed_max": float(
                    np.max(np.abs(seed.seed - np.asarray(reference["seed_plus"], dtype=float)))
                ),
            },
        }
    )
    if max(payload["reference_differences"].values()) > 1.0e-8:
        raise RuntimeError("direct Python seed does not reproduce the Mathematica/MATLAB reference")
    _write_json(output_dir(cfg) / "01_seed_report.json", payload)
    context["seed"] = seed
    return seed


def run_continuation(cfg: dict[str, Any], context: dict[str, Any]):
    system = context.setdefault("system", build_system(cfg))
    seed = context.get("seed") or run_search(cfg, context)
    cont_cfg = cfg["continuation"]
    steps = continue_integer_lure_seed(
        system,
        seed,
        eps_values=tuple(float(value) for value in cont_cfg["lambda_values"]),
        t_transient=float(cont_cfg["t_transient"]),
        t_keep=float(cont_cfg["t_keep"]),
        h=float(cont_cfg["h"]),
        div_threshold=float(cont_cfg["div_threshold"]),
    )
    rows = []
    for index, step in enumerate(steps):
        _write_trajectory(output_dir(cfg) / "continuation_steps" / f"continuation_lambda_{index:03d}.csv", step.trajectory)
        rows.append(
            {
                "step_idx": index,
                "lambda_value": step.lambda_value,
                "x_in": step.x_in,
                "x_out": step.x_out,
                "status": step.status,
                "trajectory": f"continuation_steps/continuation_lambda_{index:03d}.csv",
            }
        )
    _write_json(output_dir(cfg) / "02_continuation_trace.json", rows)
    context["continuation_steps"] = steps
    return steps


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
    _write_trajectory(output_dir(cfg) / "03_final_attractor.csv", trajectory)
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
        random_seed=int(hidden_cfg["random_seed"]),
        sampling_mode=str(hidden_cfg["sampling_mode"]),
    )
    probe_rows = [
        {
            "equilibrium": probe.equilibrium,
            "radius": probe.radius,
            "sample_id": probe.sample_id,
            "x0": probe.x0,
            "status": probe.status,
            "final_class": probe.final_class,
            "target_hit": probe.target_hit,
            "cloud_distance_norm": probe.cloud_distance_norm,
            "distance_from_equilibrium": probe.distance_from_equilibrium,
        }
        for probe in probes
    ]
    probe_csv = output_dir(cfg) / "03_hiddenness_initial_conditions.csv"
    probe_csv.parent.mkdir(parents=True, exist_ok=True)
    with probe_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "equilibrium", "radius", "x0", "y0", "z0"])
        for probe in probes:
            writer.writerow(
                [
                    probe.sample_id,
                    probe.equilibrium,
                    probe.radius,
                    *np.asarray(probe.x0, dtype=float).tolist(),
                ]
            )
    _write_json(output_dir(cfg) / "03_hiddenness_summary.json", {
        "target_seed": target_seed,
        "final_status": status,
        "summary": summarize_integer_hiddenness_controls(probes),
        "probes": probe_rows,
        "interpretation": "finite_neighborhood_test_not_global_proof",
    })
    context.update({"target_seed": target_seed, "trajectory": trajectory, "final_status": status, "probes": probes})
    return context


def run_figures(cfg: dict[str, Any], context: dict[str, Any]) -> None:
    system = context.setdefault("system", build_system(cfg))
    seed = context.get("seed") or run_search(cfg, context)
    steps = context.get("continuation_steps") or run_continuation(cfg, context)
    if "trajectory" not in context or "probes" not in context:
        run_verification(cfg, context)
    trajectory = context["trajectory"]
    probes = context["probes"]
    figures = figure_dir(cfg)
    figures.mkdir(parents=True, exist_ok=True)

    plot_lure_nyquist_describing_function(system.lure, seed, figures / "integer_lure_nyquist.png", q=1.0)
    plot_lure_transfer_components(system.lure, seed, figures / "integer_lure_transfer_components.png", q=1.0)
    plot_integer_lure_continuation(steps, figures / "integer_lure_continuation.png")
    plot_phase_space(trajectory, figures / "integer_lure_attractor.png", title="Integer Chua Lure reference")
    plot_phase_projections(trajectory, figures / "integer_lure_projections.png", title="Integer Chua Lure projections")
    plot_integer_hiddenness_controls(trajectory, probes, figures / "integer_lure_hiddenness_controls.png")
    plot_trajectory_spectra(trajectory, figures, method="fft", prefix="integer_lure")
    plot_trajectory_spectra(trajectory, figures, method="psd", prefix="integer_lure")

    if cfg.get("lyapunov", {}).get("enabled", False):
        lyap_cfg = cfg["lyapunov"]
        lyap = integer_system_lyapunov_exponents(
            system,
            context["target_seed"],
            h=float(lyap_cfg["h"]),
            t_final=float(lyap_cfg["t_final"]),
            t_burn=float(lyap_cfg["t_burn"]),
            reorthonormalize_every=int(lyap_cfg["reorthonormalize_every"]),
            div_threshold=float(lyap_cfg["div_threshold"]),
        )
        _write_json(output_dir(cfg) / "04_lyapunov_summary.json", {
            "status": lyap.status,
            "exponents": lyap.exponents,
            "interpretation": "finite_time_diagnostic_not_hiddenness_proof",
        })
        plot_lyapunov_convergence(lyap, figures / "integer_lure_lyapunov_convergence.png")


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
    context: dict[str, Any] = {}
    phase_timings: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    operations = {
        "search": run_search,
        "continuation": run_continuation,
        "verification": run_verification,
        "figures": run_figures,
    }
    for step in steps:
        started = time.perf_counter()
        operations[step](cfg, context)
        phase_timings.append(
            {"phase": step, "seconds": time.perf_counter() - started}
        )
    total_seconds = time.perf_counter() - total_started
    _write_json(output_dir(cfg) / "run_manifest.json", {
        "case_id": cfg["case_id"],
        "steps": steps,
        "quick": quick,
        "output_dir": _display_path(output_dir(cfg)),
        "figures_dir": _display_path(figure_dir(cfg)),
        "seed_route_requested": cfg["seed_search"]["route"],
        "seed_route_executed": getattr(context.get("seed"), "search_route", None),
        "phase_timings": phase_timings,
        "total_seconds": total_seconds,
        "python": sys.version,
        "platform": platform.platform(),
    })
    print(f"total_seconds={total_seconds:.9f}")
    print(f"output_dir={output_dir(cfg)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["search", "continuation", "verification", "figures"],
        default=["search", "continuation", "verification", "figures"],
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="YAML configuration for the integer reference workflow.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write all sample products to this directory.",
    )
    args = parser.parse_args()
    run_selected(
        args.steps,
        quick=args.quick,
        output_override=args.output_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()

