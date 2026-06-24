import argparse
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..systems import get_system
from ..lure.transfer import W_eval
from ..verification.stability import classify_equilibrium_stability
from .biased_chua import build_hiddenness_heatmap
from .export import export_figure
from .style import apply_axes_style, apply_library_style


VERSION2_ROOT = Path(__file__).resolve().parents[2]
OBSOLETE_C590_FIGURE_IDS = (
    "chua_frac_arctan_c590_fig03abc_projections",
    "chua_frac_arctan_c590_fig12_lyapunov_exploratory",
    "chua_frac_arctan_c590_fig16_integrator_audit",
    "chua_frac_ns_biased_fig03abc_projections",
)

def first_harmonic_reconstruction(traj: np.ndarray, tail_fraction: float = 0.85) -> np.ndarray:
    """Helper to perform first harmonic (linearized) reconstruction of the attractor trajectory."""
    n0 = int((1.0 - tail_fraction) * len(traj))
    tail = traj[n0:, 1:4]
    tail = tail[np.all(np.isfinite(tail), axis=1)]
    n = len(tail)
    if n < 32:
        return traj[n0:, :]
    centered = tail - tail.mean(axis=0)
    fft_x = np.fft.rfft(centered[:, 0])
    k = int(np.argmax(np.abs(fft_x[1:])) + 1)
    coeffs = np.fft.rfft(centered, axis=0)
    keep = np.zeros_like(coeffs)
    keep[0, :] = coeffs[0, :]
    keep[k, :] = coeffs[k, :]
    recon = np.fft.irfft(keep, n=n, axis=0) + tail.mean(axis=0)
    t = traj[n0:n0+n, 0:1]
    return np.hstack([t, recon])

def downsample(arr: np.ndarray, max_points: int) -> np.ndarray:
    if len(arr) <= max_points:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_points).astype(int)
    return arr[idx]

def save_and_close(fig, path: Path):
    """Saves the figure in both PNG and PDF formats and closes the plot."""
    path_png = path.with_suffix(".png")
    path_pdf = path.with_suffix(".pdf")
    
    path_png.parent.mkdir(parents=True, exist_ok=True)
    
    from .export import intercept_and_export_path
    intercept_and_export_path(fig, str(path_png), "publication")
    pass
    plt.close(fig)
    print(f"[Publication Figures] Saved: {path.name}.png and {path.name}.pdf")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve_publication_input(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (base / path, VERSION2_ROOT / path, VERSION2_ROOT.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (base / path).resolve()


def _source_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(VERSION2_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_csv_trajectory(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = tuple(data.dtype.names or ())
    if not names:
        raise ValueError(f"trajectory CSV has no named columns: {path}")
    t_name = "t" if "t" in names else names[0]
    coordinate_names = [name for name in ("x", "y", "z") if name in names]
    if len(coordinate_names) < 3:
        coordinate_names = [name for name in ("x0", "x1", "x2") if name in names]
    if len(coordinate_names) < 3:
        raise ValueError(f"trajectory CSV lacks three state columns: {path}")
    return np.column_stack(
        [np.atleast_1d(data[t_name])]
        + [np.atleast_1d(data[name]) for name in coordinate_names[:3]]
    ).astype(float)


def _candidate_metadata(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    figure_id: str,
    sources: list[Path],
    *,
    kind: str,
) -> dict[str, Any]:
    contract = summary["numerical_contract"]
    return {
        "caption_key": f"fig_{figure_id}",
        "source_script": "hidden_attractors/plotting/generate_publication_figures.py",
        "source_function": figure_id,
        "data_sources": [_source_label(path) for path in sources],
        "system_id": "chua_fractional_arctan",
        "q": float(summary["q"]),
        "parameters": summary["parameters"],
        "integrator": contract["integrator"],
        "memory_mode": contract["memory_mode"],
        "t_final": contract["target_t_final"],
        "t_burn": contract["target_t_burn"],
        "scientific_status": summary["status"],
        "candidate_id": summary["candidate_id"],
        "kind": kind,
    }


def _export_candidate_figure(
    fig: plt.Figure,
    figure_id: str,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    sources: list[Path],
    *,
    kind: str,
) -> None:
    export_figure(
        fig,
        figure_id,
        kind,
        _candidate_metadata(summary, manifest, figure_id, sources, kind=kind),
        run_id=str(manifest["run_id"]),
        report_targets=list(manifest.get("report_targets", ["df_nc_chua"])),
    )
    plt.close(fig)


def _candidate_tail(
    summary: dict[str, Any],
    times: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    burn = float(summary["numerical_contract"]["target_t_burn"])
    return states[times >= burn]


def _prune_obsolete_candidate_figures() -> None:
    """Remove report assets superseded by the consolidated c590 suite."""
    from .manifest import load_manifest, save_manifest

    library_root = VERSION2_ROOT / "library_figures"
    for figure_id in OBSOLETE_C590_FIGURE_IDS:
        for directory in (
            library_root / "current" / "pdf",
            library_root / "current" / "png",
            library_root / "by_report" / "df_nc_chua" / "pdf",
            library_root / "by_report" / "df_nc_chua" / "png",
        ):
            for suffix in (".pdf", ".png"):
                path = directory / f"{figure_id}{suffix}"
                if path.exists():
                    path.unlink()
    obsolete = set(OBSOLETE_C590_FIGURE_IDS)
    save_manifest(
        [entry for entry in load_manifest() if entry.get("figure_id") not in obsolete]
    )


def _plot_candidate_seed(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    times: np.ndarray,
    states: np.ndarray,
    target_path: Path,
) -> None:
    prefix = str(manifest["figure_prefix"])
    tail = downsample(_candidate_tail(summary, times, states), 14000)
    seed = np.asarray(summary["seed"], dtype=float)
    equilibria = summary["hiddenness_evidence"]["equilibria"]
    fig = plt.figure(figsize=(6.2, 5.0), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tail[:, 0], tail[:, 1], tail[:, 2], color="#0f766e", lw=0.35, alpha=0.8)
    ax.scatter(*seed, color="#dc2626", marker="*", s=105, label="Semilla de búsqueda", zorder=8)
    for name, point in equilibria.items():
        point_array = np.asarray(point, dtype=float)
        ax.scatter(*point_array, marker="x", s=46, label=name)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.legend(loc="best", fontsize=7)
    apply_axes_style(ax, is_3d=True)
    fig.tight_layout()
    _export_candidate_figure(
        fig,
        f"{prefix}_fig00_initial_seed",
        summary,
        manifest,
        [target_path],
        kind="initial_seed",
    )


def _plot_candidate_transfer(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    seed_path: Path,
) -> None:
    from ..models.chua import chua_parameters
    from ..seed_generation.chua_arctan_wu2023 import transfer_function_arctan_wu2023

    prefix = str(manifest["figure_prefix"])
    seed_report = _read_json(seed_path)
    branch = seed_report["branches"][0]
    omega0 = float(branch["omega"])
    gain = float(branch["k"])
    params = chua_parameters(model="arctan", **summary["parameters"])
    omega = np.linspace(0.05, 10.0, 1800)
    values = np.asarray(
        [
            transfer_function_arctan_wu2023(
                value,
                q=float(summary["q"]),
                params=params,
                transfer_mode=str(manifest.get("transfer_mode", "fractional_spectral")),
            )
            for value in omega
        ],
        dtype=complex,
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True, dpi=300)
    axes[0].plot(omega, values.real, color="#2563eb", lw=1.0)
    axes[0].axhline(1.0 / gain, color="#64748b", ls="--", lw=0.8, label="$1/k$")
    axes[0].axvline(omega0, color="#dc2626", lw=1.0, label=r"$\omega_0$")
    axes[0].scatter([omega0], [1.0 / gain], color="#dc2626", s=28, zorder=5)
    axes[0].set_ylabel(r"$\operatorname{Re} W_q(i\omega)$")
    axes[0].legend(loc="best", fontsize=8)
    apply_axes_style(axes[0], grid=True)
    axes[1].plot(omega, values.imag, color="#0891b2", lw=1.0)
    axes[1].axhline(0.0, color="#64748b", ls="--", lw=0.8)
    axes[1].axvline(omega0, color="#dc2626", lw=1.0, label=fr"$\omega_0={omega0:.4f}$")
    axes[1].scatter([omega0], [0.0], color="#dc2626", s=28, zorder=5)
    axes[1].set_xlabel(r"$\omega$ [rad/s]")
    axes[1].set_ylabel(r"$\operatorname{Im} W_q(i\omega)$")
    axes[1].legend(loc="best", fontsize=8)
    apply_axes_style(axes[1], grid=True)
    fig.tight_layout()
    _export_candidate_figure(
        fig,
        f"{prefix}_fig01_transfer_components",
        summary,
        manifest,
        [seed_path],
        kind="transfer_components",
    )


def _continuation_trajectories(continuation_dir: Path) -> list[tuple[float, Path, np.ndarray]]:
    trace = _read_json(continuation_dir / "continuation_trace.json")
    output: list[tuple[float, Path, np.ndarray]] = []
    for row in trace:
        path = continuation_dir / "continuation_steps" / f"continuation_eta_{int(row['step_idx']):03d}.csv"
        if path.exists():
            output.append((float(row["lambda_value"]), path, _load_csv_trajectory(path)))
    return output


def _plot_candidate_linear_and_continuation(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    times: np.ndarray,
    states: np.ndarray,
    target_path: Path,
    continuation_dir: Path,
) -> None:
    prefix = str(manifest["figure_prefix"])
    trajectories = _continuation_trajectories(continuation_dir)
    if not trajectories:
        return
    eta0, eta0_path, eta0_traj = trajectories[0]
    eta1, eta1_path, eta1_traj = trajectories[-1]
    candidate = downsample(_candidate_tail(summary, times, states), 7000)

    fig = plt.figure(figsize=(11.2, 4.0), dpi=300)
    panels = (
        (eta0_traj[:, 1:4], "#2563eb", fr"Linealizado, $\eta={eta0:g}$"),
        (eta1_traj[:, 1:4], "#dc2626", fr"No lineal, $\eta={eta1:g}$"),
        (candidate, "#0f766e", "Candidato c590"),
    )
    for index, (cloud, color, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        reduced = downsample(cloud, 3500)
        ax.plot(reduced[:, 0], reduced[:, 1], reduced[:, 2], color=color, lw=0.45)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        apply_axes_style(ax, is_3d=True)
    fig.tight_layout()
    _export_candidate_figure(
        fig,
        f"{prefix}_fig02a_linearized_vs_original",
        summary,
        manifest,
        [eta0_path, eta1_path, target_path],
        kind="linearized_vs_original",
    )

    fig = plt.figure(figsize=(7.0, 5.7), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    color_map = plt.get_cmap("viridis")
    path_points = []
    for eta, path, trajectory in trajectories:
        reduced = downsample(trajectory[:, 1:4], 1000)
        ax.plot(
            reduced[:, 0],
            reduced[:, 1],
            reduced[:, 2],
            color=color_map(eta),
            lw=0.35,
            alpha=0.6,
        )
        path_points.append(trajectory[-1, 1:4])
    path_array = np.asarray(path_points)
    ax.plot(
        path_array[:, 0],
        path_array[:, 1],
        path_array[:, 2],
        color="#111827",
        ls="--",
        marker="o",
        markersize=3.5,
        lw=1.0,
        label="Recorrido de condiciones iniciales",
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.legend(loc="best", fontsize=7)
    apply_axes_style(ax, is_3d=True)
    scalar = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(0.0, 1.0))
    colorbar = fig.colorbar(scalar, ax=ax, shrink=0.65, pad=0.08)
    colorbar.set_label(r"$\eta$")
    fig.tight_layout()
    _export_candidate_figure(
        fig,
        f"{prefix}_fig02b_continuation_path",
        summary,
        manifest,
        [path for _, path, _ in trajectories],
        kind="continuation",
    )


def _plot_candidate_dynamics(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    times: np.ndarray,
    states: np.ndarray,
    target_path: Path,
) -> None:
    prefix = str(manifest["figure_prefix"])
    tail = downsample(_candidate_tail(summary, times, states), 16000)
    fig = plt.figure(figsize=(5.6, 4.6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tail[:, 0], tail[:, 1], tail[:, 2], color="#0f766e", lw=0.35)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    apply_axes_style(ax, is_3d=True)
    fig.tight_layout()
    _export_candidate_figure(
        fig,
        f"{prefix}_fig03_attractor",
        summary,
        manifest,
        [target_path],
        kind="attractor_3d",
    )

    mask = times >= max(float(times[-1]) - 50.0, 0.0)
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 5.4), sharex=True, dpi=300)
    for index, (ax, label, color) in enumerate(
        zip(axes, ("$x(t)$", "$y(t)$", "$z(t)$"), ("#0f766e", "#2563eb", "#9333ea"))
    ):
        ax.plot(times[mask], states[mask, index], color=color, lw=0.5)
        ax.set_ylabel(label)
        apply_axes_style(ax, grid=True)
    axes[-1].set_xlabel("$t$")
    fig.tight_layout()
    _export_candidate_figure(
        fig,
        f"{prefix}_fig04_timeseries",
        summary,
        manifest,
        [target_path],
        kind="time_series",
    )


def _plot_candidate_fft(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    times: np.ndarray,
    states: np.ndarray,
    target_path: Path,
    spectral_path: Path,
) -> None:
    prefix = str(manifest["figure_prefix"])
    burn = float(summary["numerical_contract"]["target_t_burn"])
    tail = states[times >= burn]
    h = float(np.median(np.diff(times)))
    omega0 = float(manifest["omega0"])
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True, dpi=300)
    for index, (ax, label, color) in enumerate(
        zip(axes, ("$x$", "$y$", "$z$"), ("#0f766e", "#2563eb", "#9333ea"))
    ):
        signal = tail[:, index] - float(np.mean(tail[:, index]))
        amplitude = np.abs(np.fft.rfft(signal))
        amplitude /= max(float(np.max(amplitude)), 1.0e-15)
        omega = 2.0 * np.pi * np.fft.rfftfreq(len(signal), d=h)
        keep = omega <= 10.0
        ax.plot(omega[keep], amplitude[keep], color=color, lw=0.65)
        ax.axvline(omega0, color="#dc2626", lw=1.0, label=r"$\omega_0$")
        ax.set_ylabel(f"{label}\nnormalizada")
        ax.set_ylim(0.0, 1.05)
        apply_axes_style(ax, grid=True)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel(r"$\omega$ [rad/s]")
    axes[-1].set_xlim(0.0, 10.0)
    fig.tight_layout()
    _export_candidate_figure(
        fig,
        f"{prefix}_fig11_fft",
        summary,
        manifest,
        [target_path, spectral_path],
        kind="fft",
    )


def _plot_candidate_lyapunov(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    evidence_path: Path,
    convergence_path: Path,
) -> None:
    prefix = str(manifest["figure_prefix"])
    evidence = _read_json(evidence_path)
    convergence = np.load(convergence_path)
    rows = (
        ("variational", "Variacional ABM--QR"),
        ("cloned", "Dinámica clonada ABM--GS"),
    )
    colors = ("#dc2626", "#2563eb", "#9333ea")
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6), dpi=300)
    for row_index, (key, method_label) in enumerate(rows):
        method_times = np.asarray(convergence[f"{key}_times"], dtype=float)
        method_values = np.asarray(convergence[f"{key}_convergence"], dtype=float)
        for exponent_index in range(3):
            ax = axes[row_index, exponent_index]
            ax.plot(method_times, method_values[:, exponent_index], color=colors[exponent_index], lw=0.9)
            ax.axhline(0.0, color="#64748b", ls="--", lw=0.6)
            ax.set_title(fr"{method_label}: $\lambda_{exponent_index + 1}$", fontsize=8.5)
            ax.set_xlabel(r"$t_{\mathrm{eval}}$")
            ax.set_ylabel(fr"$\lambda_{exponent_index + 1}(t)$")
            apply_axes_style(ax, grid=True)
    fig.tight_layout()
    _export_candidate_figure(
        fig,
        f"{prefix}_fig12_lyapunov_two_methods",
        summary,
        manifest,
        [evidence_path, convergence_path],
        kind="lyapunov_convergence",
    )


def _plot_candidate_spheres_and_heatmap(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    matrix_path: Path,
) -> None:
    prefix = str(manifest["figure_prefix"])
    matrix = _read_json(matrix_path)
    radii = sorted(float(value) for value in matrix["radii"])
    slugs = {"E0": "E0", "E+": "Ep", "E-": "Em"}
    for equilibrium in ("E0", "E+", "E-"):
        center = np.asarray(matrix["equilibria"][equilibrium], dtype=float)
        fig = plt.figure(figsize=(8.0, 6.8), dpi=300)
        for panel, radius in enumerate(radii, start=1):
            ax = fig.add_subplot(2, 2, panel, projection="3d")
            rows = [
                row
                for row in matrix["rows"]
                if row["equilibrium"] == equilibrium
                and np.isclose(float(row["radius"]), radius)
            ]
            directions = np.asarray([row["direction_vector"] for row in rows], dtype=float)
            points = center + radius * directions
            contacts = np.asarray([bool(row["contact"]) for row in rows], dtype=bool)
            u = np.linspace(0.0, 2.0 * np.pi, 28)
            v = np.linspace(0.0, np.pi, 15)
            x_surface = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y_surface = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z_surface = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(
                x_surface,
                y_surface,
                z_surface,
                rstride=3,
                cstride=2,
                color="#94a3b8",
                linewidth=0.25,
                alpha=0.24,
            )
            ax.scatter(
                points[~contacts, 0],
                points[~contacts, 1],
                points[~contacts, 2],
                s=15,
                color="#2563eb",
                alpha=0.78,
                label="sin contacto",
            )
            if np.any(contacts):
                ax.scatter(
                    points[contacts, 0],
                    points[contacts, 1],
                    points[contacts, 2],
                    s=20,
                    color="#dc2626",
                    label="TARGET",
                )
            ax.scatter(*center, marker="x", s=40, color="#111827")
            ax.text2D(
                0.03,
                0.93,
                fr"$r={radius:.0e}$, $N={len(rows)}$",
                transform=ax.transAxes,
                fontsize=8,
            )
            limit = 1.15 * radius
            ax.set_xlim(center[0] - limit, center[0] + limit)
            ax.set_ylim(center[1] - limit, center[1] + limit)
            ax.set_zlim(center[2] - limit, center[2] + limit)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_zlabel("$z$")
            apply_axes_style(ax, is_3d=True)
        fig.tight_layout()
        _export_candidate_figure(
            fig,
            f"{prefix}_fig13_{slugs[equilibrium]}_hiddenness_spherical_3d",
            summary,
            manifest,
            [matrix_path],
            kind="hiddenness_spheres",
        )

    heatmap_records = []
    for equilibrium in ("E0", "E+", "E-"):
        for radius in radii:
            rows = [
                row
                for row in matrix["rows"]
                if row["equilibrium"] == equilibrium
                and np.isclose(float(row["radius"]), radius)
            ]
            heatmap_records.append(
                {
                    "equilibrium": equilibrium,
                    "radius": radius,
                    "samples": len(rows),
                    "TARGET": sum(bool(row["contact"]) for row in rows),
                }
            )
    fig, _ = build_hiddenness_heatmap(heatmap_records, radii)
    _export_candidate_figure(
        fig,
        f"{prefix}_fig14_hiddenness_contact_heatmap",
        summary,
        manifest,
        [matrix_path],
        kind="hiddenness_heatmap",
    )


def _export_report_heatmap(
    figure_id: str,
    records: list[dict[str, Any]],
    sources: list[Path],
    *,
    q: float,
    system_id: str,
    status: str,
    wide: bool = False,
) -> None:
    radii = sorted({float(row["radius"]) for row in records})
    fig, _ = build_hiddenness_heatmap(records, radii)
    if wide:
        fig.set_size_inches(9.2, 3.6)
        fig.tight_layout()
    metadata = {
        "caption_key": f"fig_{figure_id}",
        "source_script": "hidden_attractors/plotting/generate_publication_figures.py",
        "source_function": "generate_comparison_report_heatmaps",
        "data_sources": [_source_label(path) for path in sources],
        "system_id": system_id,
        "q": q,
        "scientific_status": status,
        "kind": "hiddenness_heatmap",
    }
    export_figure(
        fig,
        figure_id,
        "hiddenness_heatmap",
        metadata,
        run_id="df_nc_chua_heatmap_unification_20260624",
        report_targets=["df_nc_chua"],
    )
    plt.close(fig)


def generate_comparison_report_heatmaps() -> None:
    """Regenerate report heatmaps 24, 33 and 37 with one visual contract."""
    centered_path = (
        VERSION2_ROOT
        / "validation"
        / "outputs"
        / "candidate_chaos_hiddenness"
        / "danca2017_chua_fractional_saturation_candidate"
        / "report"
        / "candidate_chaos_hiddenness_summary.json"
    )
    inputs_path = VERSION2_ROOT / "configs" / "report_heatmap_inputs.json"
    centered = _read_json(centered_path)
    centered_records = [
        {
            "equilibrium": row["equilibrium"],
            "radius": float(row["radius"]),
            "samples": int(row["samples"]),
            "TARGET": int(row["target_hits"]),
        }
        for row in centered["hiddenness"]["decisions"]
    ]
    _export_report_heatmap(
        "chua_frac_ns_fig14_hiddenness_contact_heatmap",
        centered_records,
        [centered_path],
        q=float(centered["parameters"]["q"]),
        system_id="chua_fractional_saturation",
        status="self_excited_contact_detected",
    )

    inputs = _read_json(inputs_path)
    local = inputs["biased_local"]
    local_records = [
        {
            "equilibrium": equilibrium,
            "radius": float(radius),
            "samples": int(samples),
            "TARGET": int(local["target_hits"]),
        }
        for equilibrium in local["equilibria"]
        for radius, samples in zip(local["radii"], local["samples_per_radius"])
    ]
    _export_report_heatmap(
        "chua_frac_ns_biased_fig14_hiddenness_contact_heatmap",
        local_records,
        [inputs_path],
        q=0.9998,
        system_id="chua_fractional_saturation_biased",
        status="compatible_with_hiddenness_under_finite_surface_test",
    )
    _export_report_heatmap(
        "chua_frac_ns_biased_fig15_extended_heatmap",
        list(inputs["biased_extended"]["records"]),
        [inputs_path],
        q=0.9998,
        system_id="chua_fractional_saturation_biased",
        status="report_table_transcription_raw_artifact_missing",
        wide=True,
    )


def generate_biased_report_dynamics() -> None:
    """Regenerate the biased attractor as 3D-only and its normalized FFT."""
    output_root = (
        VERSION2_ROOT.parent
        / "outputs"
        / "example_chua_nonsmooth_biased_hidden_attractor"
        / "step2_biased_df"
    )
    trajectory_path = (
        output_root
        / "trajectories"
        / "biased_q9998_m1_m1p1468_m0_m0p1768_branch_1_c_2p776_trajectory.csv"
    )
    branch_path = output_root / "affine_continuation_summary.csv"
    trajectory = _load_csv_trajectory(trajectory_path)
    cloud = downsample(trajectory[:, 1:4], 16000)
    common_metadata = {
        "source_script": "hidden_attractors/plotting/generate_publication_figures.py",
        "source_function": "generate_biased_report_dynamics",
        "data_sources": [_source_label(trajectory_path), _source_label(branch_path)],
        "system_id": "chua_fractional_saturation_biased",
        "q": 0.9998,
        "parameters": {"m0": -0.1768, "m1": -1.1468},
        "scientific_status": "finite_time_candidate",
    }

    fig = plt.figure(figsize=(5.6, 4.6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(cloud[:, 0], cloud[:, 1], cloud[:, 2], color="#0f766e", lw=0.35)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    apply_axes_style(ax, is_3d=True)
    fig.tight_layout()
    export_figure(
        fig,
        "chua_frac_ns_biased_fig03_attractor",
        "attractor_3d",
        {
            **common_metadata,
            "caption_key": "fig_chua_frac_ns_biased_fig03_attractor",
            "kind": "attractor_3d",
        },
        run_id="df_nc_chua_biased_dynamics_20260624",
        report_targets=["df_nc_chua"],
    )
    plt.close(fig)
    _prune_obsolete_candidate_figures()


def generate_centered_report_fft() -> None:
    """Regenerate the centered control FFT with normalized amplitude and omega0."""
    output_root = (
        VERSION2_ROOT.parent
        / "outputs"
        / "example_chua_nonsmooth_biased_hidden_attractor"
        / "step2_biased_df"
    )
    trajectory_path = (
        output_root
        / "trajectories"
        / "biased_q9998_m1_m1p2000_m0_m0p2000_branch_0_c_centered_like_trajectory.csv"
    )
    classification_path = output_root / "final_classification.csv"
    trajectory = _load_csv_trajectory(trajectory_path)
    h = float(np.median(np.diff(trajectory[:, 0])))
    signal = trajectory[:, 1] - float(np.mean(trajectory[:, 1]))
    amplitude = np.abs(np.fft.rfft(signal))
    amplitude /= max(float(np.max(amplitude)), 1.0e-15)
    omega = 2.0 * np.pi * np.fft.rfftfreq(len(signal), d=h)
    omega0 = 2.04028605107949
    keep = omega <= 10.0

    fig, ax = plt.subplots(figsize=(6.8, 3.7), dpi=300)
    ax.plot(omega[keep], amplitude[keep], color="#111827", lw=0.7)
    ax.axvline(omega0, color="#dc2626", lw=1.0, label=fr"$\omega_0={omega0:.4f}$")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel("Amplitud normalizada")
    ax.legend(loc="upper right", fontsize=8)
    apply_axes_style(ax, grid=True)
    fig.tight_layout()
    export_figure(
        fig,
        "chua_frac_ns_fig11a_fft_x",
        "fft",
        {
            "caption_key": "fig_chua_frac_ns_fig11a_fft_x",
            "source_script": "hidden_attractors/plotting/generate_publication_figures.py",
            "source_function": "generate_centered_report_fft",
            "data_sources": [
                _source_label(trajectory_path),
                _source_label(classification_path),
            ],
            "system_id": "chua_fractional_saturation",
            "q": 0.9998,
            "parameters": {"m0": -0.2, "m1": -1.2},
            "scientific_status": "self_excited_contact_detected",
            "omega0": omega0,
            "kind": "fft",
        },
        run_id="df_nc_chua_centered_fft_20260624",
        report_targets=["df_nc_chua"],
    )
    plt.close(fig)

    h = float(np.median(np.diff(trajectory[:, 0])))
    signal = trajectory[:, 1] - float(np.mean(trajectory[:, 1]))
    amplitude = np.abs(np.fft.rfft(signal))
    amplitude /= max(float(np.max(amplitude)), 1.0e-15)
    omega = 2.0 * np.pi * np.fft.rfftfreq(len(signal), d=h)
    omega0 = 2.04028605107949
    keep = omega <= 10.0
    fig, ax = plt.subplots(figsize=(6.8, 3.7), dpi=300)
    ax.plot(omega[keep], amplitude[keep], color="#0f766e", lw=0.7)
    ax.axvline(omega0, color="#dc2626", lw=1.0, label=fr"$\omega_0={omega0:.4f}$")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel("Amplitud normalizada")
    ax.legend(loc="upper right", fontsize=8)
    apply_axes_style(ax, grid=True)
    fig.tight_layout()
    export_figure(
        fig,
        "chua_frac_ns_biased_fig11_fft",
        "fft",
        {
            **common_metadata,
            "caption_key": "fig_chua_frac_ns_biased_fig11_fft",
            "omega0": omega0,
            "kind": "fft",
        },
        run_id="df_nc_chua_biased_dynamics_20260624",
        report_targets=["df_nc_chua"],
    )
    plt.close(fig)


def generate_candidate_publication_figures(candidate_dir: str | Path) -> None:
    """Generate the complete c590-style paper suite from one candidate bundle."""
    directory = Path(candidate_dir).resolve()
    manifest_path = directory / "publication_figure_inputs.json"
    manifest = _read_json(manifest_path)
    summary_path = _resolve_publication_input(directory, manifest["candidate_summary"])
    matrix_path = _resolve_publication_input(directory, manifest["hiddenness_matrix"])
    continuation_dir = _resolve_publication_input(directory, manifest["continuation_dir"])
    target_path = directory / "target.npz"
    seed_path = directory / "df_fractional_spectral_seed.json"
    spectral_path = directory / "spectral_periodicity_audit.json"
    lyapunov_path = directory / "lyapunov_two_method.json"
    convergence_path = directory / "lyapunov_two_method_convergence.npz"
    required = (
        summary_path,
        matrix_path,
        continuation_dir / "continuation_trace.json",
        target_path,
        seed_path,
        spectral_path,
        lyapunov_path,
        convergence_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"candidate publication inputs are missing: {missing}")

    summary = _read_json(summary_path)
    target = np.load(target_path)
    times = np.asarray(target["times"], dtype=float)
    states = np.asarray(target["states"], dtype=float)
    apply_library_style()
    _plot_candidate_seed(summary, manifest, times, states, target_path)
    _plot_candidate_transfer(summary, manifest, seed_path)
    _plot_candidate_linear_and_continuation(
        summary,
        manifest,
        times,
        states,
        target_path,
        continuation_dir,
    )
    _plot_candidate_dynamics(summary, manifest, times, states, target_path)
    _plot_candidate_fft(summary, manifest, times, states, target_path, spectral_path)
    _plot_candidate_lyapunov(summary, manifest, lyapunov_path, convergence_path)
    _plot_candidate_spheres_and_heatmap(summary, manifest, matrix_path)
    generate_comparison_report_heatmaps()
    generate_centered_report_fft()
    generate_biased_report_dynamics()
    _prune_obsolete_candidate_figures()
    print(f"[Publication Figures] Candidate suite completed from {directory}")


def generate_all_publication_figures(output_dir: str, config: Dict[str, Any]) -> None:
    """
    Core post-processor that parses raw data and configuration from a workflow run
    and produces vector PDF + high-resolution PNG figures.
    """
    out_dir_path = Path(output_dir)
    if (out_dir_path / "publication_figure_inputs.json").exists():
        generate_candidate_publication_figures(out_dir_path)
        return
    fig_dir = out_dir_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = out_dir_path / "summary.json"
    effective_cfg_path = out_dir_path / "effective_config.json"
    
    if not summary_path.exists() or not effective_cfg_path.exists():
        print(f"[Publication Figures] WARNING: Missing summary/config in {output_dir}. Skipping generation.")
        return
        
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(effective_cfg_path, "r", encoding="utf-8") as f:
        eff_config = json.load(f)
        
    system_id = eff_config.get("system_id")
    q = eff_config.get("q", 1.0)
    
    system_params = {
        "alpha": eff_config.get("alpha", 8.4562),
        "beta": eff_config.get("beta", 12.0732),
        "gamma": eff_config.get("gamma", 0.0052),
        "m0": eff_config.get("m0", -0.1768),
        "m1": eff_config.get("m1", -1.1468),
        "q": q
    }
    if "chua_fractional_arctan" in system_id:
        system_params["m"] = eff_config.get("m", 1.5)
        system_params["n"] = eff_config.get("n", 10.0)
    elif "polynomial" in system_id:
        system_params["coeff"] = eff_config.get("coeff", 1.0)
        
    # Translate old system IDs if needed
    name_map = {
        "chua_piecewise": "chua-nonsmooth",
        "chua_integer_saturation": "chua-nonsmooth",
        "chua_fractional_saturation": "chua-nonsmooth",
        "chua_integer_arctan": "chua-arctan",
        "chua_fractional_arctan": "chua-arctan",
        "chua_arctan_wu2023": "fractional-chua-arctan-wu2023",
    }
    normalized_sys_id = name_map.get(system_id, system_id)
    system = get_system(normalized_sys_id)
    
    # Inject overrides as attributes and parameters
    merged_params = dict(system.parameters)
    merged_params.update(system_params)
    object.__setattr__(system, "parameters", merged_params)
    object.__setattr__(system, "q", q)
    for k, v in merged_params.items():
        try:
            object.__setattr__(system, k, v)
        except Exception:
            pass
            
    if system.lure is not None:
        object.__setattr__(system, "P", system.lure.matrix)
        object.__setattr__(system, "b", system.lure.input_vector)
        object.__setattr__(system, "r", system.lure.output_vector)
        object.__setattr__(system, "describing_function", system.lure.describing_function)
        
    omega0 = summary.get("omega0")
    a0 = summary.get("amplitude_a0")
    k = summary.get("k")
    
    # 2. FIG02: Continuation plots
    trace_json_path = out_dir_path / "continuation_trace.json"
    if trace_json_path.exists():
        with open(trace_json_path, "r", encoding="utf-8") as f:
            trace = json.load(f)
            
        etas = [s["lambda_value"] for s in trace]
        x_out_norms = [s["x_out_norm"] for s in trace]
        
        steps_coords = []
        for s in trace:
            step_idx = s["step_idx"]
            step_csv = out_dir_path / "continuation_steps" / f"continuation_eta_{step_idx:03d}.csv"
            if step_csv.exists():
                with open(step_csv, "r", encoding="utf-8") as sf:
                    rows = list(csv.DictReader(sf))
                    if rows:
                        last_row = rows[-1]
                        keys = list(last_row.keys())
                        c_keys = [k for k in ["x0", "x1", "x2"] if k in keys]
                        if not c_keys:
                            c_keys = [k for k in ["x", "y", "z"] if k in keys]
                        if len(c_keys) >= 3:
                            steps_coords.append((s["lambda_value"], float(last_row[c_keys[0]]), float(last_row[c_keys[1]]), float(last_row[c_keys[2]])))
                            
        if steps_coords:
            steps_coords = sorted(steps_coords, key=lambda x: x[0])
            
        first_step_csv = out_dir_path / "continuation_steps" / "continuation_eta_000.csv"
        csv_files = sorted((out_dir_path / "continuation_steps").glob("continuation_eta_*.csv"))
        if first_step_csv.exists() and len(csv_files) >= 2:
            last_step_csv = csv_files[-1]
            
            def load_traj_coords(csv_path: Path) -> np.ndarray:
                with csv_path.open("r", newline="", encoding="utf-8") as sf:
                    rows = list(csv.DictReader(sf))
                    keys = list(rows[0].keys())
                    c_keys = [k for k in ["x0", "x1", "x2"] if k in keys]
                    if not c_keys:
                        c_keys = [k for k in ["x", "y", "z"] if k in keys]
                    return np.array([[float(r[c_keys[0]]), float(r[c_keys[1]]), float(r[c_keys[2]])] for r in rows])
                    
            try:
                first_traj = load_traj_coords(first_step_csv)
                last_traj = load_traj_coords(last_step_csv)
                
                fig2d = plt.figure(figsize=(8.0, 7.0), dpi=300)
                ax2d = fig2d.add_subplot(111, projection="3d")
                
                first_small = downsample(first_traj, 1500)
                last_small = downsample(last_traj, 2000)
                
                ax2d.plot(first_small[:, 0], first_small[:, 1], first_small[:, 2], color="blue", linewidth=1.8, label="Linearized Attractor ($\\eta=0.0$)")
                ax2d.plot(last_small[:, 0], last_small[:, 1], last_small[:, 2], color="red", linewidth=1.8, label="Nonlinear Attractor ($\\eta=1.0$)")
                
                if steps_coords:
                    path_arr = np.array([[x[1], x[2], x[3]] for x in steps_coords])
                    ax2d.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2], "k--", marker="o", markersize=4, linewidth=1.2, label="Continuation path")
                    
                ax2d.set_title("Numerical Continuation: Trajectory Evolution", fontsize=11, fontweight="bold", pad=12)
                ax2d.set_xlabel("x")
                ax2d.set_ylabel("y")
                ax2d.set_zlabel("z")
                ax2d.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
                save_and_close(fig2d, fig_dir / "fig02d_continuation_story")
            except Exception as e:
                print(f"[Publication Figures] Failed to render fig02d: {e}")
                
    # 3. FIG03: Final Attractor Trajectory and projections
    final_attractor_csv = out_dir_path / "final_attractor.csv"
    if final_attractor_csv.exists():
        with final_attractor_csv.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            traj_data = np.array([[float(r["t"]), float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
            
        x_col = traj_data[:, 1]
        y_col = traj_data[:, 2]
        z_col = traj_data[:, 3]
        
        fig3 = plt.figure(figsize=(8.0, 7.0), dpi=300)
        ax3 = fig3.add_subplot(111, projection="3d")
        ax3.plot(x_col, y_col, z_col, color="#ef4444", linewidth=1.0, alpha=0.85, label="Final Nonlinear Attractor")
        ax3.scatter([x_col[-1]], [y_col[-1]], [z_col[-1]], color="black", s=45, zorder=5, label="Endpoint")
        ax3.set_title("Final Attractor Trajectory (3D)", fontsize=11, fontweight="bold", pad=12)
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        ax3.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
        save_and_close(fig3, fig_dir / "fig03_final_attractor")
        
        fig3a, ax3a = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        ax3a.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
        ax3a.plot(x_col, y_col, color="#ef4444", linewidth=0.9, alpha=0.85)
        ax3a.set_title("Final Attractor XY Projection", fontsize=11, fontweight="bold", pad=12)
        ax3a.set_xlabel("x")
        ax3a.set_ylabel("y")
        save_and_close(fig3a, fig_dir / "fig03a_final_attractor_xy")
        
        fig3b, ax3b = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        ax3b.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
        ax3b.plot(x_col, z_col, color="#ef4444", linewidth=0.9, alpha=0.85)
        ax3b.set_title("Final Attractor XZ Projection", fontsize=11, fontweight="bold", pad=12)
        ax3b.set_xlabel("x")
        ax3b.set_ylabel("z")
        save_and_close(fig3b, fig_dir / "fig03b_final_attractor_xz")
        
        fig3c, ax3c = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        ax3c.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
        ax3c.plot(y_col, z_col, color="#ef4444", linewidth=0.9, alpha=0.85)
        ax3c.set_title("Final Attractor YZ Projection", fontsize=11, fontweight="bold", pad=12)
        ax3c.set_xlabel("y")
        ax3c.set_ylabel("z")
        save_and_close(fig3c, fig_dir / "fig03c_final_attractor_yz")
        
        try:
            recon = first_harmonic_reconstruction(traj_data)
            t_final = traj_data[-1, 0]
            mask_50 = traj_data[:, 0] >= (t_final - 50.0)
            traj_data_50 = traj_data[mask_50]
            recon_50 = recon[recon[:, 0] >= (t_final - 50.0)]
            
            orig_small = downsample(traj_data_50, 4000)
            recon_small = downsample(recon_50, 1500)
            
            fig3d, ax3d = plt.subplots(figsize=(8.0, 4.5), dpi=300)
            ax3d.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
            ax3d.plot(orig_small[:, 0], orig_small[:, 1], color="red", linewidth=1.2, alpha=0.85, label="Original")
            ax3d.plot(recon_small[:, 0], recon_small[:, 1], "--", color="purple", linewidth=1.4, alpha=0.9, label="Linearized")
            ax3d.set_title("Linear vs Original: $x(t)$ Component", fontsize=11, fontweight="bold", pad=12)
            ax3d.set_xlabel("Time $t$")
            ax3d.set_ylabel("$x$")
            ax3d.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
            save_and_close(fig3d, fig_dir / "fig03d_linear_vs_original_x")
            
            fig3e, ax3e = plt.subplots(figsize=(8.0, 4.5), dpi=300)
            ax3e.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
            ax3e.plot(orig_small[:, 0], orig_small[:, 2], color="red", linewidth=1.2, alpha=0.85, label="Original")
            ax3e.plot(recon_small[:, 0], recon_small[:, 2], "--", color="purple", linewidth=1.4, alpha=0.9, label="Linearized")
            ax3e.set_title("Linear vs Original: $y(t)$ Component", fontsize=11, fontweight="bold", pad=12)
            ax3e.set_xlabel("Time $t$")
            ax3e.set_ylabel("$y$")
            ax3e.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
            save_and_close(fig3e, fig_dir / "fig03e_linear_vs_original_y")
            
            fig3f, ax3f = plt.subplots(figsize=(8.0, 4.5), dpi=300)
            ax3f.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
            ax3f.plot(orig_small[:, 0], orig_small[:, 3], color="red", linewidth=1.2, alpha=0.85, label="Original")
            ax3f.plot(recon_small[:, 0], recon_small[:, 3], "--", color="purple", linewidth=1.4, alpha=0.9, label="Linearized")
            ax3f.set_title("Linear vs Original: $z(t)$ Component", fontsize=11, fontweight="bold", pad=12)
            ax3f.set_xlabel("Time $t$")
            ax3f.set_ylabel("$z$")
            ax3f.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
            save_and_close(fig3f, fig_dir / "fig03f_linear_vs_original_z")
            
            fig3g = plt.figure(figsize=(8.0, 7.0), dpi=300)
            ax3g = fig3g.add_subplot(111, projection="3d")
            ax3g.plot(orig_small[:, 1], orig_small[:, 2], orig_small[:, 3], color="red", linewidth=1.2, alpha=0.85, label="Original")
            ax3g.plot(recon_small[:, 1], recon_small[:, 2], recon_small[:, 3], "--", color="purple", linewidth=1.5, alpha=0.9, label="Linearized")
            ax3g.set_title("Linear vs Original Attractor in 3D", fontsize=11, fontweight="bold", pad=12)
            ax3g.set_xlabel("x")
            ax3g.set_ylabel("y")
            ax3g.set_zlabel("z")
            ax3g.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
            save_and_close(fig3g, fig_dir / "fig03g_linear_vs_original_3d")
            
        except Exception as e:
            print(f"[Publication Figures] Failed to render first harmonic comparisons: {e}")
            
        try:
            n_burn = int(len(traj_data) * 0.25)
            dt = traj_data[1, 0] - traj_data[0, 0]
            
            tail_x = x_col[n_burn:]
            tail_y = y_col[n_burn:]
            tail_z = z_col[n_burn:]
            
            for component_name, tail_comp, fig_id in [("x", tail_x, "fig11a_fft_x"), ("y", tail_y, "fig11b_fft_y"), ("z", tail_z, "fig11c_fft_z")]:
                centered_comp = tail_comp - tail_comp.mean()
                fft_vals = np.fft.rfft(centered_comp)
                fft_freqs = np.fft.rfftfreq(len(centered_comp), d=dt)
                fft_mag = np.abs(fft_vals) / len(centered_comp)
                
                omega_rad_s = 2.0 * np.pi * fft_freqs
                
                fig11_fft, ax11_fft = plt.subplots(figsize=(7.5, 4.5), dpi=300)
                ax11_fft.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
                ax11_fft.plot(omega_rad_s, fft_mag, color="#111827", linewidth=1.0)
                
                if omega0 and not np.isnan(omega0):
                    ax11_fft.axvline(omega0, color="#ef4444", linestyle=":", label=r"Predicted frequency $\omega_0$")
                    
                ax11_fft.set_title(f"Spectral Analysis FFT: component {component_name}", fontsize=11, fontweight="bold", pad=12)
                ax11_fft.set_xlabel(r"Frequency $\omega$ (rad/s)")
                ax11_fft.set_ylabel("Amplitude")
                ax11_fft.set_xlim(0, 15.0)
                ax11_fft.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
                save_and_close(fig11_fft, fig_dir / fig_id)
                
            for component_name, tail_comp, fig_id in [("x", tail_x, "fig11d_psd_x"), ("y", tail_y, "fig11e_psd_y"), ("z", tail_z, "fig11f_psd_z")]:
                nperseg = min(256, len(tail_comp))
                freqs_psd, psd_vals = welch(tail_comp - tail_comp.mean(), fs=1.0/dt, nperseg=nperseg)
                omega_rad_s = 2.0 * np.pi * freqs_psd
                
                fig11_psd, ax11_psd = plt.subplots(figsize=(7.5, 4.5), dpi=300)
                ax11_psd.grid(True, linestyle="--", linewidth=0.5, color="#cbd5e1")
                ax11_psd.plot(omega_rad_s, psd_vals, color="#0284c7", linewidth=1.2)
                
                if omega0 and not np.isnan(omega0):
                    ax11_psd.axvline(omega0, color="#ef4444", linestyle=":", label=r"Predicted frequency $\omega_0$")
                    
                ax11_psd.set_title(f"Welch PSD Power Density: component {component_name}", fontsize=11, fontweight="bold", pad=12)
                ax11_psd.set_xlabel(r"Frequency $\omega$ (rad/s)")
                ax11_psd.set_ylabel("Power Density")
                ax11_psd.set_xlim(0, 15.0)
                ax11_psd.legend(loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
                save_and_close(fig11_psd, fig_dir / fig_id)
                
        except Exception as e:
            print(f"[Publication Figures] Failed to render spectral figures: {e}")
            
    # 5. FIG04: Matignon stability diagram / Reference section
    try:
        from ..verification.equilibria import solve_equilibria
        eq_pts = solve_equilibria(system)
        
        fig4 = plt.figure(figsize=(7.5, 6.5), dpi=300)
        ax4 = fig4.add_subplot(111)
        
        ax4.set_facecolor("#f0fdf4")
        
        all_eigvals = []
        eq_details = []
        for name, eq_pt in eq_pts.items():
            res = classify_equilibrium_stability(system, eq_pt)
            all_eigvals.extend(res["eigenvalues"])
            eq_details.append((name, res))
            
        all_eigvals = np.array(all_eigvals)
        max_rad = float(np.max(np.abs(all_eigvals))) if len(all_eigvals) > 0 else 1.0
        limit = max_rad * 1.5
        R = limit * 2.0
        
        t_vals = np.linspace(-q * np.pi / 2.0, q * np.pi / 2.0, 400)
        x_fill = [0.0] + list(R * np.cos(t_vals)) + [0.0]
        y_fill = [0.0] + list(R * np.sin(t_vals)) + [0.0]
        ax4.fill(x_fill, y_fill, color="#fee2e2", alpha=0.9, edgecolor="#fca5a5", linewidth=0.8, label="Unstable Region")
        
        ax4.plot([0.0, R * np.cos(q * np.pi / 2.0)], [0.0, R * np.sin(q * np.pi / 2.0)], color="#ef4444", linestyle="--", linewidth=1.1, label=r"Frontier $|\arg(\lambda)| = q\pi/2$")
        ax4.plot([0.0, R * np.cos(-q * np.pi / 2.0)], [0.0, R * np.sin(-q * np.pi / 2.0)], color="#ef4444", linestyle="--", linewidth=1.1)
        
        ax4.axhline(0.0, color="#64748b", linewidth=0.7, linestyle=":")
        ax4.axvline(0.0, color="#64748b", linewidth=0.7, linestyle=":")
        
        colors = {"E0": "#3b82f6", "E+": "#ef4444", "E-": "#f59e0b"}
        markers = {"E0": "^", "E+": "o", "E-": "s"}
        
        for name, res in eq_details:
            color = colors.get(name, "#8b5cf6")
            marker = markers.get(name, "d")
            eigvals = res["eigenvalues"]
            ax4.scatter(np.real(eigvals), np.imag(eigvals), color=color, marker=marker, s=70, edgecolors="black", zorder=10, label=f"{name} eigenvalues")
            
        ax4.set_xlim(-limit, limit)
        ax4.set_ylim(-limit, limit)
        ax4.set_aspect("equal")
        ax4.set_title(f"Matignon Stability Plane (q={q:.4f})", fontsize=11, fontweight="bold", pad=12)
        ax4.set_xlabel(r"$\mathrm{Re}(\lambda)$")
        ax4.set_ylabel(r"$\mathrm{Im}(\lambda)$")
        ax4.legend(loc="upper right", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
        save_and_close(fig4, fig_dir / "fig04_reference_section")
    except Exception as e:
        print(f"[Publication Figures] Failed to render fig04: {e}")
        
    # 7. FIG06 & FIG10: Basin slice overlay plots (if available)
    basin_csv_path = out_dir_path / "basin_results.csv"
    if basin_csv_path.exists():
        try:
            with open(basin_csv_path, "r", newline="", encoding="utf-8") as bf:
                b_rows = list(csv.DictReader(bf))
                
            planes = {}
            for r in b_rows:
                plane = r.get("plane", "global")
                if plane not in planes:
                    planes[plane] = []
                planes[plane].append(r)
                
            for plane_name, p_data in planes.items():
                u_vals = np.array([float(x["u_val"]) for x in p_data])
                v_vals = np.array([float(x["v_val"]) for x in p_data])
                codes = np.array([int(x["classification_code"]) for x in p_data])
                
                u_unique = np.unique(u_vals)
                v_unique = np.unique(v_vals)
                
                if len(u_unique) * len(v_unique) == len(codes):
                    U_mesh, V_mesh = np.meshgrid(u_unique, v_unique)
                    code_mesh = np.zeros((len(v_unique), len(u_unique)))
                    for item in p_data:
                        col_idx = np.where(u_unique == float(item["u_val"]))[0][0]
                        row_idx = np.where(v_unique == float(item["v_val"]))[0][0]
                        code_mesh[row_idx, col_idx] = int(item["classification_code"])
                        
                    fig6, ax6 = plt.subplots(figsize=(7.5, 6.5), dpi=300)
                    
                    from matplotlib.colors import ListedColormap
                    custom_colors = ["#ff66b2", "#8b5cf6", "#94a3b8", "#facc15", "#475569", "#3b82f6"]
                    cmap = ListedColormap(custom_colors[:max(3, len(np.unique(codes)))])
                    
                    mesh = ax6.pcolormesh(U_mesh, V_mesh, code_mesh, cmap=cmap, shading="nearest", alpha=0.92)
                    
                    from matplotlib.patches import Patch
                    labels_mapping = {
                        0: "Target Attractor (Pink)",
                        1: "Stable Equilibrium (Purple)",
                        2: "Divergence (Gray)",
                        3: "Other Attractor (Yellow)",
                        4: "Numerical Failure (Dark)",
                        5: "Unclassified (Blue)"
                    }
                    legend_patches = []
                    for code in np.unique(code_mesh):
                        c_idx = int(code)
                        if c_idx < len(custom_colors):
                            color = custom_colors[c_idx]
                            label = labels_mapping.get(c_idx, f"Code {c_idx}")
                            legend_patches.append(Patch(facecolor=color, label=label, edgecolor="black", linewidth=0.5))
                            
                    eq_selected = eff_config.get("basin", {}).get("equilibrium_selection", "E+")
                    if eq_selected in eq_pts:
                        eq_pt = eq_pts[eq_selected]
                        ax6.scatter([eq_pt[0]], [eq_pt[1]], color="red", marker="*", s=160, edgecolors="black", zorder=12, label=f"Equilibrium {eq_selected}")
                        
                    ax6.legend(handles=legend_patches, loc="best", fontsize=8.5, framealpha=0.9, facecolor="#f8fafc", edgecolor="#e2e8f0")
                    
                    fixed_z = eff_config.get("basin", {}).get("fixed_z", 0.0)
                    ax6.set_title(f"Basin Attraction Slice (Plane xy, z = {fixed_z:.2f})", fontsize=11, fontweight="bold", pad=12)
                    ax6.set_xlabel("u coordinate")
                    ax6.set_ylabel("v coordinate")
                    
                    if plane_name == "xy" or plane_name == "xy_z0":
                        fig_id = "fig06a_basin_overlay_z0"
                    elif "zfinal" in plane_name or "z_final" in plane_name:
                        fig_id = "fig06b_basin_overlay_zfinal"
                    else:
                        fig_id = f"fig10_{plane_name}_basin_slice"
                        
                    save_and_close(fig6, fig_dir / fig_id)
                else:
                    fig6, ax6 = plt.subplots(figsize=(7.5, 6.5), dpi=300)
                    scatter = ax6.scatter(u_vals, v_vals, c=codes, s=4, cmap="plasma", alpha=0.85)
                    fig6.colorbar(scatter, ax=ax6)
                    ax6.set_title(f"Basin Attraction Slice: Plane {plane_name}", fontsize=11, fontweight="bold", pad=12)
                    ax6.set_xlabel("u coordinate")
                    ax6.set_ylabel("v coordinate")
                    save_and_close(fig6, fig_dir / f"fig10_{plane_name}_basin_slice")
                    
        except Exception as e:
            print(f"[Publication Figures] Failed to render basin slices: {e}")
            
    print("[Publication Figures] Success: Completed all publication-grade vectors! [OK]")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate every publication figure for a completed workflow or candidate bundle."
    )
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    generate_all_publication_figures(str(args.output_dir), {})


if __name__ == "__main__":
    main()
