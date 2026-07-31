"""Generate one reviewable real-system example for every plotting callable.

Every numerical curve is computed from the registered nonsmooth Chua circuit
model.  The script integrates the model, sweeps a physical model parameter,
runs the public Lur'e continuation and equilibrium-neighbourhood controls, and
calculates spectra and finite-time Lyapunov estimates through library APIs.
It never fabricates parametric curves, basin labels, or convergence histories.

These are reproducible numerical examples, not new validation evidence.  A
figure alone does not certify chaos, hiddenness, asymptotic stability, or
solver convergence.

Run from ``version_2``:

    python figure_scripts/generate_plot_catalog_examples.py
    python figure_scripts/generate_plot_catalog_examples.py --only plot_phase_space
    python figure_scripts/generate_plot_catalog_examples.py --list
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


PUBLIC_GRAPH_CALLABLES = (
    "plot_integer_hiddenness_controls",
    "plot_integer_lure_continuation",
    "plot_lyapunov_convergence",
    "plot_lure_nyquist_describing_function",
    "plot_lure_transfer_components",
    "plot_bifurcation_diagram",
    "plot_phase_projections",
    "plot_phase_space",
    "plot_spectrum",
    "plot_time_series",
    "plot_trajectory_spectra",
    "plot_trajectory_overlay",
    "plot_basin_slices",
    "plot_basin_slice_file",
    "plot_matignon_equilibria",
    "plot_nyquist_transfer",
    "plot_describing_function",
    "plot_harmonic_residual_map",
    "plot_continuation_eta",
    "plot_continuation_first_last_comparison",
    "plot_continuation_timeseries_comparison",
    "plot_continuation_progression",
    "plot_continuation_tracking",
    "plot_attractor_trajectories",
    "plot_flexible_attractor_and_projections",
    "plot_timeseries_data",
    "plot_neighborhood_control_spheres",
    "plot_sphere_test_results",
    "render_attractor",
    "render_basin",
    "render_nyquist",
    "render_matignon",
    "render_all_plots",
)


@dataclass
class ExampleContext:
    plotting: object
    raw_root: Path
    trajectory: np.ndarray
    trajectory_alt: np.ndarray
    equilibria: dict[str, np.ndarray]
    basin_u: np.ndarray
    basin_v: np.ndarray
    basin_grid: np.ndarray
    renderer_basin_grid: np.ndarray
    bifurcation_points: list
    spectrum: object
    lyapunov_result: object
    continuation_steps: list
    integer_steps: list
    integer_probes: list
    probe_results: list[dict]
    sphere_runs: list[dict]
    config: dict
    system: object
    lure_system_owner: object
    lure_seed: object
    lure_candidates: list[tuple[float, float, float]]
    omega_grid: np.ndarray
    transfer_values: np.ndarray
    describing_values: np.ndarray
    eigenvalues: np.ndarray
    provenance: dict[str, object]

    def raw_dir(self, function_name: str) -> Path:
        path = self.raw_root / function_name
        path.mkdir(parents=True, exist_ok=True)
        return path


def _integrate_registered_chua(
    system: object,
    x0: np.ndarray,
    *,
    h: float,
    t_final: float,
) -> np.ndarray:
    """Integrate the registered Chua model with the public q=1 selector."""

    from hidden_attractors.integrations.selector import integrate

    times, states, status = integrate(
        system.evaluate,
        np.asarray(x0, dtype=float),
        q=1.0,
        h=float(h),
        t_final=float(t_final),
        integrator="efork_q1",
        system=system,
        use_c_backend=True,
        allow_python_fallback=True,
        divergence_norm=120.0,
        early_stop_config={"enabled": False},
    )
    if status != "ok":
        raise RuntimeError(f"registered Chua integration ended with status={status!r}")
    return np.column_stack((times, states))


def _real_chua_basin_slice(
    system: object,
    target_trajectory: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Compute a finite-time xy outcome grid by integrating every initial state."""

    from hidden_attractors.analysis.trajectory import cloud_median_distance
    from hidden_attractors.integrations.selector import integrate

    grid_axis = np.linspace(-2.0, 2.0, 31)
    classes = np.empty((grid_axis.size, grid_axis.size), dtype=int)
    equilibria = system.equilibrium_points()
    target_tail = np.asarray(target_trajectory, dtype=float)
    target_tail = target_tail[target_tail[:, 0] >= 40.0, 1:]
    target_cloud = target_tail[
        np.linspace(0, len(target_tail) - 1, min(201, len(target_tail)), dtype=int)
    ]
    target_scale = max(float(np.linalg.norm(np.ptp(target_cloud, axis=0))), 1.0e-12)

    for i, x0_value in enumerate(grid_axis):
        for j, y0_value in enumerate(grid_axis):
            times, states, status = integrate(
                system.evaluate,
                np.array([x0_value, y0_value, 0.0], dtype=float),
                q=1.0,
                h=0.02,
                t_final=35.0,
                integrator="efork_q1",
                system=system,
                use_c_backend=True,
                allow_python_fallback=True,
                divergence_norm=120.0,
                early_stop_config={"enabled": False},
            )
            if "diverged" in status:
                classes[i, j] = 3
                continue
            if status != "ok" or states.size == 0 or not np.all(np.isfinite(states)):
                classes[i, j] = 4
                continue

            tail = states[times >= 17.5]
            final_state = tail[-1]
            tail_span = float(np.linalg.norm(np.ptp(tail, axis=0)))
            nearest_equilibrium = min(
                float(np.linalg.norm(final_state - equilibrium))
                for equilibrium in equilibria.values()
            )
            if nearest_equilibrium <= 0.35 or tail_span < 1.0:
                classes[i, j] = 0
                continue

            probe_cloud = tail[
                np.linspace(0, len(tail) - 1, min(101, len(tail)), dtype=int)
            ]
            normalized_cloud_distance = (
                cloud_median_distance(probe_cloud, target_cloud) / target_scale
            )
            classes[i, j] = 1 if normalized_cloud_distance <= 0.10 else 2

    counts = {
        str(class_id): int(np.count_nonzero(classes == class_id))
        for class_id in range(5)
    }
    return (
        grid_axis,
        grid_axis.copy(),
        classes,
        {
            "plane": "xy",
            "fixed_z": 0.0,
            "grid_shape": list(classes.shape),
            "axis_range": [-2.0, 2.0],
            "integrator": "efork_q1",
            "q": 1.0,
            "h": 0.02,
            "t_final": 35.0,
            "tail_start": 17.5,
            "equilibrium_distance_threshold": 0.35,
            "tail_span_threshold": 1.0,
            "target_cloud_normalized_distance_threshold": 0.10,
            "class_counts": counts,
            "classification_scope": "finite_time_numerical_example",
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_catalog_outputs(
    paths: list[Path],
    destination: Path,
    *,
    manifest_path_base: Path,
) -> list[dict[str, object]]:
    existing = [path for path in paths if path.exists() and path.suffix.lower() == ".png"]
    if not existing:
        raise FileNotFoundError(
            "The plotting callable returned without producing the expected PNG. "
            f"Checked: {[str(path) for path in paths]}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    for stale in destination.parent.glob(f"{destination.stem}__*.png"):
        stale.unlink()

    copied: list[dict[str, object]] = []
    for output_index, source in enumerate(existing, start=1):
        if output_index == 1:
            target = destination
        else:
            target = destination.with_name(
                f"{destination.stem}__{output_index:02d}_{source.stem}.png"
            )
        shutil.copy2(source, target)
        copied.append(
            {
                "output_index": output_index,
                "source_name": source.name,
                "catalog_png": str(target.relative_to(manifest_path_base)).replace(
                    "\\", "/"
                ),
                "sha256": _sha256(target),
            }
        )
    return copied


def _build_context(plotting: object, raw_root: Path) -> ExampleContext:
    from hidden_attractors import (
        bifurcation_points_from_trajectories,
        get_system,
        integer_system_lyapunov_exponents,
    )
    from hidden_attractors.analysis.spectral import fft_spectrum
    from hidden_attractors.integrations.rk4 import rk4_integrate
    from hidden_attractors.lure.transfer import W_eval
    from hidden_attractors.workflows.integer_lure import (
        continue_integer_lure_seed,
        integer_lure_seed,
        run_integer_lure_hiddenness_controls,
    )
    from hidden_attractors.workflows.protocol import ContinuationPlan

    system = get_system("chua-nonsmooth")
    equilibria = system.equilibrium_points()
    lure_seed = integer_lure_seed(system, nscan=4000, wmax=50.0)
    integer_steps = continue_integer_lure_seed(
        system,
        lure_seed,
        plan=ContinuationPlan.uniform(5, internal_parameter="epsilon"),
        t_transient=12.0,
        t_keep=24.0,
        h=0.02,
        div_threshold=120.0,
    )
    if len(integer_steps) != 5 or any(step.status != "ok" for step in integer_steps):
        raise RuntimeError(
            "The real Chua Lur'e continuation did not complete its five steps."
        )

    trajectory = _integrate_registered_chua(
        system,
        np.asarray(integer_steps[-1].x_out, dtype=float),
        h=0.01,
        t_final=100.0,
    )
    trajectory_alt = _integrate_registered_chua(
        system,
        np.asarray(lure_seed.seed, dtype=float)
        + np.array([0.12, -0.04, 0.08], dtype=float),
        h=0.01,
        t_final=100.0,
    )

    basin_u, basin_v, basin_grid, basin_provenance = _real_chua_basin_slice(
        system,
        trajectory,
    )
    # The legacy plot_* basin functions use [stable,target,other,diverged,failure].
    # render_basin uses [target,stable,diverged,other]; normalize explicitly.
    renderer_basin_grid = np.full_like(basin_grid, 3)
    renderer_basin_grid[basin_grid == 0] = 1
    renderer_basin_grid[basin_grid == 1] = 0
    renderer_basin_grid[basin_grid == 2] = 3
    renderer_basin_grid[basin_grid == 3] = 2
    renderer_basin_grid[basin_grid == 4] = 3

    bifurcation_scans: list[tuple[float, np.ndarray]] = []
    beta_values = np.linspace(11.5, 14.5, 31)
    for beta in beta_values:
        parameterized_system = dataclasses.replace(
            system,
            parameters=dict(system.parameters, beta=float(beta)),
        )

        def parameterized_rhs(*args, current_system=parameterized_system):
            return current_system.evaluate(np.asarray(args[-1], dtype=float))

        times, states, status, _info = rk4_integrate(
            parameterized_rhs,
            np.asarray(lure_seed.seed, dtype=float),
            h=0.01,
            N=5000,
            divergence_norm=120.0,
        )
        if status == "ok":
            bifurcation_scans.append(
                (float(beta), np.column_stack((times, states)))
            )
    bifurcation_points = bifurcation_points_from_trajectories(
        bifurcation_scans,
        observable="x",
        t_start=25.0,
        mode="maxima",
        max_points_per_parameter=80,
    )
    if not bifurcation_points:
        raise RuntimeError("The real Chua beta sweep produced no local maxima.")

    spectrum = fft_spectrum(
        trajectory[trajectory[:, 0] >= 40.0, 1],
        h=0.01,
        component=0,
    )
    lyapunov_result = integer_system_lyapunov_exponents(
        system,
        np.asarray(integer_steps[-1].x_out, dtype=float),
        h=0.01,
        t_final=30.0,
        t_burn=5.0,
        reorthonormalize_every=10,
        div_threshold=120.0,
    )
    if lyapunov_result.status != "ok":
        raise RuntimeError(
            "The real Chua Lyapunov calculation ended with "
            f"status={lyapunov_result.status!r}."
        )

    continuation_steps = [
        {
            "lambda_value": float(step.lambda_value),
            "x_in": np.asarray(step.x_in, dtype=float),
            "x_out": np.asarray(step.x_out, dtype=float),
            "x_out_norm": float(np.linalg.norm(step.x_out)),
            "trajectory": np.asarray(step.trajectory, dtype=float),
            "status": str(step.status),
        }
        for step in integer_steps
    ]

    integer_probes = run_integer_lure_hiddenness_controls(
        system,
        trajectory,
        equilibria=equilibria,
        radii=(0.08,),
        samples_per_radius=4,
        t_final=30.0,
        t_burn=15.0,
        h=0.02,
        div_threshold=120.0,
        equilibrium_tol=0.35,
        target_cloud_tol=0.05,
        max_cloud_points=400,
        random_seed=20260729,
    )

    def probe_destination(probe: object) -> str:
        if probe.target_hit:
            return "target_attractor"
        if str(probe.final_class).startswith("equilibrium_"):
            return "stable_equilibrium"
        if probe.status != "ok":
            return "divergence" if "diverg" in probe.status else "numerical_failure"
        if probe.final_class == "diverged":
            return "divergence"
        return "other_attractor"

    probe_results = [
        {
            "x0": np.asarray(probe.x0, dtype=float),
            "destination": probe_destination(probe),
            "trajectory": np.asarray(probe.trajectory, dtype=float),
        }
        for probe in integer_probes
    ]
    sphere_runs = [
        record
        for record, probe in zip(probe_results, integer_probes)
        if probe.equilibrium == "E0" and abs(float(probe.radius) - 0.08) <= 1.0e-12
    ]

    lure_system_owner = system
    lure_candidates = [(float(lure_seed.amplitude), float(lure_seed.omega), float(lure_seed.gain))]
    omega_grid = np.linspace(0.05, 8.0, 500)
    transfer_values = np.asarray(
        W_eval(
            omega_grid,
            1.0,
            "integer",
            lure_system_owner.lure.matrix,
            lure_system_owner.lure.input_vector,
            lure_system_owner.lure.output_vector,
            transfer_convention="opposite_sign",
        ),
        dtype=complex,
    )
    amplitudes = np.linspace(1.001, 12.0, omega_grid.size)
    describing_values = np.asarray(
        [lure_system_owner.lure.describing_function(float(value)) for value in amplitudes],
        dtype=complex,
    )

    config = {
        "system_id": "chua-nonsmooth",
        "h": 0.01,
        "q": 1.0,
        "t_burn": 40.0,
        "t_final": 100.0,
        "final_simulation": {"t_burn": 40.0},
        "attractor_plots": {
            "line_width": 0.7,
            "point_size": 0.0,
            "include_equilibria": True,
        },
        "timeseries_max_points": 3000,
        "amplitude_min": 1.001,
        "amplitude_max": 12.0,
        "omega_min": 0.05,
        "omega_max": 8.0,
        "transfer_mode": "integer",
        "transfer_convention": "opposite_sign",
        "harmonic_condition": "1_plus_WN",
        "system_params": dict(system.parameters),
        "integrator": "efork_q1",
        "memory_mode": "not_applicable_integer_q1",
        "classification_criteria": basin_provenance,
        "basin_plane": "xy",
        "fixed_variables": {"z": 0.0},
    }

    eigenvalues = np.concatenate(
        [
            np.linalg.eigvals(system.jacobian_matrix(equilibrium))
            for equilibrium in equilibria.values()
        ]
    )
    provenance: dict[str, object] = {
        "case_id": "chua_nonsmooth_real_system_catalog_20260729",
        "source_kind": "canonical_library_reintegration",
        "system_id": system.name,
        "system_description": system.description,
        "parameters": dict(system.parameters),
        "trajectory": {
            "integrator": "efork_q1",
            "q": 1.0,
            "h": 0.01,
            "t_final": 100.0,
            "initial_condition": [
                float(value) for value in integer_steps[-1].x_out
            ],
        },
        "continuation": {
            "method": "integer_lure_lambda_continuation",
            "lambda_values": [
                float(step.lambda_value) for step in integer_steps
            ],
            "t_transient": 12.0,
            "t_keep": 24.0,
            "h": 0.02,
        },
        "bifurcation": {
            "parameter": "beta",
            "range": [11.5, 14.5],
            "n_parameter_values": int(beta_values.size),
            "integrator": "rk4",
            "h": 0.01,
            "t_final": 50.0,
            "t_start": 25.0,
            "observable": "x",
            "extraction": "local_maxima",
            "successful_parameter_values": len(bifurcation_scans),
        },
        "lyapunov": {
            "method": lyapunov_result.method_id,
            "q": 1.0,
            "h": 0.01,
            "t_burn": 5.0,
            "t_final": 30.0,
            "status": lyapunov_result.status,
        },
        "hiddenness_controls": {
            "radii": [0.08],
            "samples_per_radius_per_equilibrium": 4,
            "random_seed": 20260729,
            "t_final": 30.0,
            "t_burn": 15.0,
            "h": 0.02,
        },
        "lure_frequency_domain": {
            "transfer_convention": "opposite_sign",
            "transfer_definition": "W_code(s)=c^T(P-sI)^(-1)b",
            "harmonic_condition": "1_plus_WN",
            "closure": "W_code(i*omega_0)=-1/N(A_0)",
        },
        "basin": basin_provenance,
        "claim_scope": (
            "reproducible_real_system_numerical_example_not_new_validation_evidence"
        ),
    }
    return ExampleContext(
        plotting=plotting,
        raw_root=raw_root,
        trajectory=trajectory,
        trajectory_alt=trajectory_alt,
        equilibria=equilibria,
        basin_u=basin_u,
        basin_v=basin_v,
        basin_grid=basin_grid,
        renderer_basin_grid=renderer_basin_grid,
        bifurcation_points=bifurcation_points,
        spectrum=spectrum,
        lyapunov_result=lyapunov_result,
        continuation_steps=continuation_steps,
        integer_steps=integer_steps,
        integer_probes=integer_probes,
        probe_results=probe_results,
        sphere_runs=sphere_runs,
        config=config,
        system=system,
        lure_system_owner=lure_system_owner,
        lure_seed=lure_seed,
        lure_candidates=lure_candidates,
        omega_grid=omega_grid,
        transfer_values=transfer_values,
        describing_values=describing_values,
        eigenvalues=eigenvalues,
        provenance=provenance,
    )


def _runners(context: ExampleContext) -> dict[str, Callable[[], list[Path]]]:
    p = context.plotting

    def out(name: str, filename: str) -> Path:
        return context.raw_dir(name) / filename

    def plot_integer_hiddenness_controls() -> list[Path]:
        path = out("plot_integer_hiddenness_controls", "integer_hiddenness_controls.png")
        p.plot_integer_hiddenness_controls(context.trajectory, context.integer_probes, path)
        return [path]

    def plot_integer_lure_continuation() -> list[Path]:
        path = out("plot_integer_lure_continuation", "integer_lure_continuation.png")
        p.plot_integer_lure_continuation(context.integer_steps, path)
        return [path]

    def plot_lyapunov_convergence() -> list[Path]:
        path = out("plot_lyapunov_convergence", "lyapunov_convergence.png")
        p.plot_lyapunov_convergence(context.lyapunov_result, path)
        return [path]

    def plot_lure_nyquist_describing_function() -> list[Path]:
        path = out("plot_lure_nyquist_describing_function", "lure_nyquist_df.png")
        p.plot_lure_nyquist_describing_function(
            context.lure_system_owner.lure,
            context.lure_seed,
            path,
            q=1.0,
        )
        return [path]

    def plot_lure_transfer_components() -> list[Path]:
        path = out("plot_lure_transfer_components", "lure_transfer_components.png")
        p.plot_lure_transfer_components(
            context.lure_system_owner.lure,
            context.lure_seed,
            path,
            q=1.0,
            nscan=1500,
        )
        return [path]

    def plot_bifurcation_diagram() -> list[Path]:
        path = out("plot_bifurcation_diagram", "bifurcation_diagram.png")
        p.plot_bifurcation_diagram(
            context.bifurcation_points,
            path,
            parameter_label=r"Chua parameter $\beta$",
            observable_label="local maxima of x",
            title="Nonsmooth Chua: finite-time beta sweep",
        )
        return [path]

    def plot_phase_projections() -> list[Path]:
        path = out("plot_phase_projections", "phase_projections.png")
        p.plot_phase_projections(context.trajectory, path)
        return [path]

    def plot_phase_space() -> list[Path]:
        path = out("plot_phase_space", "phase_space.png")
        p.plot_phase_space(context.trajectory, path, dims=("x", "y", "z"))
        return [path]

    def plot_spectrum() -> list[Path]:
        path = out("plot_spectrum", "spectrum.png")
        p.plot_spectrum(context.spectrum, path, x_units="Hz")
        return [path]

    def plot_time_series() -> list[Path]:
        path = out("plot_time_series", "time_series.png")
        p.plot_time_series(context.trajectory, path, columns=("x", "y", "z"))
        return [path]

    def plot_trajectory_spectra() -> list[Path]:
        directory = context.raw_dir("plot_trajectory_spectra")
        return [
            Path(path)
            for path in p.plot_trajectory_spectra(
                context.trajectory,
                directory,
                method="fft",
                prefix="catalog",
            )
        ]

    def plot_trajectory_overlay() -> list[Path]:
        path = out("plot_trajectory_overlay", "trajectory_overlay.png")
        p.plot_trajectory_overlay(
            [context.trajectory, context.trajectory_alt],
            ["continuation endpoint", "perturbed harmonic seed"],
            title="Nonsmooth Chua q=1 trajectories",
            output_path=path,
        )
        return [path]

    def plot_basin_slices() -> list[Path]:
        directory = context.raw_dir("plot_basin_slices")
        result = p.plot_basin_slices(
            {"xy": (context.basin_u, context.basin_v, context.basin_grid)},
            "chua-nonsmooth-q1",
            directory,
        )
        return [Path(path) for path in result.values()]

    def plot_basin_slice_file() -> list[Path]:
        directory = context.raw_dir("plot_basin_slice_file")
        path = p.plot_basin_slice_file(
            "xy",
            context.basin_u,
            context.basin_v,
            context.basin_grid,
            "E0",
            "chua-nonsmooth-q1",
            directory,
        )
        return [Path(path)]

    def plot_matignon_equilibria() -> list[Path]:
        directory = context.raw_dir("plot_matignon_equilibria")
        path = p.plot_matignon_equilibria(
            context.system,
            context.equilibria,
            0.9998,
            directory,
        )
        return [Path(path)]

    def plot_nyquist_transfer() -> list[Path]:
        directory = context.raw_dir("plot_nyquist_transfer")
        config = dict(context.config, system_id="chua-nonsmooth")
        p.plot_nyquist_transfer(
            context.omega_grid,
            context.transfer_values,
            context.lure_candidates,
            config,
            directory,
        )
        return [
            directory / "figures" / "transfer_nyquist.png",
            directory / "figures" / "fig01b_nyquist_zoom_x.png",
            directory / "figures" / "transfer_real_imag.png",
        ]

    def plot_describing_function() -> list[Path]:
        directory = context.raw_dir("plot_describing_function")
        config = dict(context.config, system_id="chua-nonsmooth")
        p.plot_describing_function(
            context.lure_system_owner,
            context.lure_candidates,
            config,
            directory,
        )
        return [directory / "figures" / "describing_function.png"]

    def plot_harmonic_residual_map() -> list[Path]:
        directory = context.raw_dir("plot_harmonic_residual_map")
        config = dict(context.config, system_id="chua-nonsmooth")
        p.plot_harmonic_residual_map(
            context.lure_system_owner,
            context.lure_candidates,
            config,
            directory,
        )
        return [directory / "figures" / "harmonic_residual_map.png"]

    def plot_continuation_eta() -> list[Path]:
        directory = context.raw_dir("plot_continuation_eta")
        p.plot_continuation_eta(context.continuation_steps, context.config, directory)
        return [
            directory / "figures" / "continuation_norm_vs_eta.png",
            directory / "figures" / "continuation_amplitude_vs_eta.png",
            directory / "figures" / "continuation_first_last_comparison.png",
            directory / "figures" / "continuation_first_last_projections.png",
            directory / "figures" / "continuation_timeseries_comparison_x.png",
            directory / "figures" / "continuation_progression.png",
        ]

    def plot_continuation_first_last_comparison() -> list[Path]:
        directory = context.raw_dir("plot_continuation_first_last_comparison")
        p.plot_continuation_first_last_comparison(
            context.continuation_steps,
            context.config,
            directory,
        )
        return [
            directory / "figures" / "continuation_first_last_comparison.png",
            directory / "figures" / "continuation_first_last_projections.png",
        ]

    def plot_continuation_timeseries_comparison() -> list[Path]:
        directory = context.raw_dir("plot_continuation_timeseries_comparison")
        p.plot_continuation_timeseries_comparison(
            context.continuation_steps,
            context.config,
            directory,
        )
        return [directory / "figures" / "continuation_timeseries_comparison_x.png"]

    def plot_continuation_progression() -> list[Path]:
        directory = context.raw_dir("plot_continuation_progression")
        p.plot_continuation_progression(context.continuation_steps, context.config, directory)
        return [directory / "figures" / "continuation_progression.png"]

    def plot_continuation_tracking() -> list[Path]:
        directory = context.raw_dir("plot_continuation_tracking")
        p.plot_continuation_tracking(context.continuation_steps, context.config, directory)
        return [
            directory / "figures" / "continuation_tracking_norm.png",
            directory / "figures" / "continuation_tracking_status.png",
        ]

    def plot_attractor_trajectories() -> list[Path]:
        directory = context.raw_dir("plot_attractor_trajectories")
        p.plot_attractor_trajectories(
            context.trajectory,
            context.equilibria,
            context.config,
            directory,
        )
        return [
            directory / "figures" / "attractor_3d.png",
            directory / "figures" / "attractor_xy.png",
            directory / "figures" / "attractor_xz.png",
            directory / "figures" / "attractor_yz.png",
        ]

    def plot_flexible_attractor_and_projections() -> list[Path]:
        directory = context.raw_dir("plot_flexible_attractor_and_projections")
        p.plot_flexible_attractor_and_projections(
            context.trajectory,
            context.equilibria,
            context.config,
            directory,
            "catalog_flexible",
        )
        return [
            directory / "figures" / "catalog_flexible_3d.png",
            directory / "figures" / "catalog_flexible_xy.png",
            directory / "figures" / "catalog_flexible_xz.png",
            directory / "figures" / "catalog_flexible_yz.png",
        ]

    def plot_timeseries_data() -> list[Path]:
        directory = context.raw_dir("plot_timeseries_data")
        p.plot_timeseries_data(
            context.trajectory,
            context.config,
            directory,
            "catalog_series",
        )
        return [
            directory / "figures" / "catalog_series_timeseries_x.png",
            directory / "figures" / "catalog_series_timeseries_y.png",
            directory / "figures" / "catalog_series_timeseries_z.png",
            directory / "figures" / "catalog_series_timeseries_xyz.png",
        ]

    def plot_neighborhood_control_spheres() -> list[Path]:
        directory = context.raw_dir("plot_neighborhood_control_spheres")
        p.plot_neighborhood_control_spheres(
            context.trajectory,
            context.probe_results,
            context.equilibria,
            context.config,
            directory,
        )
        return [directory / "figures" / "fig05b_hiddenness_overview.png"]

    def plot_sphere_test_results() -> list[Path]:
        directory = context.raw_dir("plot_sphere_test_results")
        path = p.plot_sphere_test_results(
            "E0",
            context.equilibria["E0"],
            0.08,
            context.sphere_runs,
            directory,
        )
        return [Path(path)]

    def render_attractor() -> list[Path]:
        result = p.render_attractor(
            context.trajectory,
            context.equilibria,
            context.config,
            run_id="catalog_render_attractor",
        )
        return [
            Path(result["3d_png"]),
            Path(result["xy_png"]),
            Path(result["xz_png"]),
            Path(result["yz_png"]),
        ]

    def render_basin() -> list[Path]:
        _pdf, png = p.render_basin(
            context.basin_u,
            context.basin_v,
            context.renderer_basin_grid,
            context.config,
            run_id="catalog_render_basin",
        )
        return [Path(png)]

    def render_nyquist() -> list[Path]:
        _pdf, png = p.render_nyquist(
            context.omega_grid,
            context.transfer_values,
            context.describing_values,
            context.lure_candidates,
            dict(context.config, system_id="chua-nonsmooth"),
            run_id="catalog_render_nyquist",
        )
        return [Path(png)]

    def render_matignon() -> list[Path]:
        _pdf, png = p.render_matignon(
            context.eigenvalues,
            0.9998,
            context.config,
            run_id="catalog_render_matignon",
        )
        return [Path(png)]

    def render_all_plots() -> list[Path]:
        result = p.render_all_plots(
            trajectory=context.trajectory,
            equilibria=context.equilibria,
            basin_grid=context.renderer_basin_grid,
            grid_x=context.basin_u,
            grid_y=context.basin_v,
            freqs=context.omega_grid,
            w_evals=context.transfer_values,
            n_evals=context.describing_values,
            candidates=context.lure_candidates,
            eigenvalues=context.eigenvalues,
            config=context.config,
            run_id="catalog_render_all",
        )
        return [
            Path(result["attractor"]["3d_png"]),
            Path(result["attractor"]["xy_png"]),
            Path(result["attractor"]["xz_png"]),
            Path(result["attractor"]["yz_png"]),
            Path(result["basin"]["png"]),
            Path(result["nyquist"]["png"]),
            Path(result["matignon"]["png"]),
        ]

    runners = {name: locals()[name] for name in PUBLIC_GRAPH_CALLABLES}
    return runners


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory; defaults to docs/assets/generated_plot_catalog.",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=PUBLIC_GRAPH_CALLABLES,
        help="Generate only one named public callable; may be repeated.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the 33 public plotting callables and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.list:
        for name in PUBLIC_GRAPH_CALLABLES:
            print(name)
        return 0

    project_root = Path(__file__).resolve().parents[1]
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "docs" / "assets" / "generated_plot_catalog"
    )
    examples_dir = output_root / "examples"
    output_root.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)
    try:
        output_root.relative_to(project_root)
        manifest_path_base = project_root
        manifest_path_base_label = "project_root"
    except ValueError:
        manifest_path_base = output_root
        manifest_path_base_label = "output_root"
    sys.path.insert(0, str(project_root))

    requested = list(args.only) if args.only else list(PUBLIC_GRAPH_CALLABLES)
    results: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="hidden-attractors-plot-catalog-") as temporary:
        temporary_root = Path(temporary)
        os.environ["HIDDEN_ATTRACTORS_OUTPUT_DIR"] = str(temporary_root / "runtime_outputs")
        os.environ.setdefault("MPLBACKEND", "Agg")

        import matplotlib.pyplot as plt
        import hidden_attractors.plotting as plotting

        exported_public = tuple(
            name
            for name in plotting.__all__
            if name.startswith("plot_") or name.startswith("render_")
        )
        if exported_public != PUBLIC_GRAPH_CALLABLES:
            missing = sorted(set(exported_public) - set(PUBLIC_GRAPH_CALLABLES))
            stale = sorted(set(PUBLIC_GRAPH_CALLABLES) - set(exported_public))
            raise RuntimeError(
                "The plotting public API changed. "
                f"Uncatalogued exports={missing}; stale catalog entries={stale}."
            )

        context = _build_context(plotting, temporary_root / "raw")
        runners = _runners(context)
        for index, name in enumerate(PUBLIC_GRAPH_CALLABLES, start=1):
            if name not in requested:
                continue
            target = examples_dir / f"{index:02d}_{name}.png"
            function = getattr(plotting, name)
            if name in {
                "plot_lure_nyquist_describing_function",
                "plot_lure_transfer_components",
                "plot_nyquist_transfer",
                "plot_describing_function",
                "plot_harmonic_residual_map",
                "render_nyquist",
            }:
                input_bundle = "computed_chua_nonsmooth_lure_algebra"
            elif name in {"plot_basin_slices", "plot_basin_slice_file", "render_basin"}:
                input_bundle = "computed_chua_nonsmooth_q1_finite_time_basin"
            elif name in {"plot_matignon_equilibria", "render_matignon"}:
                input_bundle = "computed_chua_nonsmooth_equilibrium_jacobians"
            elif name == "plot_bifurcation_diagram":
                input_bundle = "computed_chua_nonsmooth_beta_sweep"
            elif name == "plot_lyapunov_convergence":
                input_bundle = "computed_chua_nonsmooth_q1_lyapunov"
            elif name in {
                "plot_integer_hiddenness_controls",
                "plot_neighborhood_control_spheres",
                "plot_sphere_test_results",
            }:
                input_bundle = "computed_chua_nonsmooth_q1_neighborhood_controls"
            elif name in {
                "plot_integer_lure_continuation",
                "plot_continuation_eta",
                "plot_continuation_first_last_comparison",
                "plot_continuation_timeseries_comparison",
                "plot_continuation_progression",
                "plot_continuation_tracking",
            }:
                input_bundle = "computed_chua_nonsmooth_q1_lure_continuation"
            elif name == "render_all_plots":
                input_bundle = "computed_chua_nonsmooth_complete_context"
            else:
                input_bundle = "computed_chua_nonsmooth_q1_trajectory"
            row: dict[str, object] = {
                "index": index,
                "function": name,
                "signature": f"{name}{inspect.signature(function)}",
                "generator_command": (
                    f"python figure_scripts/generate_plot_catalog_examples.py --only {name}"
                ),
                "representative_png": str(
                    target.relative_to(manifest_path_base)
                ).replace("\\", "/"),
                "data_policy": "real_named_system_numerical_output",
                "input_bundle": input_bundle,
                "system_id": "chua-nonsmooth",
                "claim_scope": (
                    "reproducible_numerical_example_not_new_validation_evidence"
                ),
            }
            try:
                produced = runners[name]()
                copied_outputs = _copy_catalog_outputs(
                    produced,
                    target,
                    manifest_path_base=manifest_path_base,
                )
                row["produced_png_names"] = [
                    output["source_name"] for output in copied_outputs
                ]
                row["produced_outputs"] = copied_outputs
                row["representative_sha256"] = copied_outputs[0]["sha256"]
                row["status"] = "ok"
                if name == "plot_continuation_eta":
                    row["implementation_note"] = (
                        "Produces norm-vs-lambda and amplitude-vs-lambda panels, then "
                        "delegates the first/last, time-series, and progression views."
                    )
                elif name == "plot_continuation_tracking":
                    row["implementation_note"] = (
                        "Produces both norm-vs-lambda and status-per-step panels."
                    )
                elif name == "render_all_plots":
                    row["implementation_note"] = (
                        "Orchestrator: produces attractor, basin, Nyquist, and "
                        "Matignon figures; the first output is representative."
                    )
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()
                print(f"FAILED {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            finally:
                plt.close("all")
            results.append(row)

    manifest_path = output_root / "catalog_results.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scope": "public plot_ and render_ exports from hidden_attractors.plotting",
                "scientific_evidence": False,
                "input_data_are_real_numerical_outputs": True,
                "catalog_plot_alone_certifies_scientific_claim": False,
                "warning": (
                    "Every input is computed from the registered nonsmooth Chua system. "
                    "These reproducible numerical examples must not be treated as new "
                    "validation or as stand-alone certification of chaos, hiddenness, "
                    "asymptotic stability, or solver convergence."
                ),
                "numerical_input_provenance": context.provenance,
                "representative_path_base": manifest_path_base_label,
                "total_public_callables": len(PUBLIC_GRAPH_CALLABLES),
                "requested": len(requested),
                "successful": sum(row["status"] == "ok" for row in results),
                "failed": sum(row["status"] == "failed" for row in results),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    failures = [row for row in results if row["status"] == "failed"]
    print(
        f"plot catalog: {len(results) - len(failures)}/{len(results)} successful; "
        f"manifest={manifest_path}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
