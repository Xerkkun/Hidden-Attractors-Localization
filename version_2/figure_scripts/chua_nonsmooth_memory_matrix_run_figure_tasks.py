#!/usr/bin/env python3
"""Generate per-experiment figures from continuation and hiddenness artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _common import (
    VERSION2_ROOT,
    experiment_spec,
    harmonic_seed_from_payload,
    is_ok_status,
    load_matrix,
    load_trajectory,
    metadata,
    read_csv_rows,
    run_process_pool,
    status_counts,
    write_status,
)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from hidden_attractors.analysis import trajectory_component_spectra
from hidden_attractors.io import read_json
from hidden_attractors.plotting import plot_lure_nyquist_describing_function, plot_phase_space, plot_time_series
from hidden_attractors.systems import get_system


REQUESTED_FIGURES = (
    "seed_nyquist",
    "continuation_path",
    "phase3d_candidate",
    "equilibrium_neighborhood_trajectories",
    "basin_slices_xy_xz",
    "time_series",
    "fft_psd",
    "hiddenness_summary",
)


def _save(fig: plt.Figure, directory: Path, stem: str) -> list[str]:
    """Save a figure in PNG and PDF formats for article reuse."""

    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    fig.tight_layout()
    from version_2.hidden_attractors.plotting.export import intercept_and_export_path
    intercept_and_export_path(fig, png, 'attractor')
    from version_2.hidden_attractors.plotting.export import intercept_and_export_path
    intercept_and_export_path(fig, pdf, 'nyquist')
    plt.close(fig)
    return [str(png), str(pdf)]


def _placeholder(directory: Path, stem: str, message: str) -> list[str]:
    """Write an explicit missing-input panel instead of fabricating dynamics."""

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    return _save(fig, directory, stem)


def _continuations(root: Path, exp_id: str) -> list[tuple[dict[str, str], Path]]:
    """Return continuation rows and directories for one experiment."""

    return [
        (row, root / row["output_dir"])
        for row in read_csv_rows(root / "tasks" / "continuation_tasks.csv")
        if row["exp_id"] == exp_id
    ]


def _hiddenness(root: Path, exp_id: str) -> list[tuple[dict[str, str], Path]]:
    """Return completed hiddenness rows and directories for one experiment."""

    return [
        (row, root / row["output_dir"])
        for row in read_csv_rows(root / "tasks" / "hiddenness_tasks.csv")
        if row["exp_id"] == exp_id
    ]


def _first_valid_trajectory(hidden_dirs: list[tuple[dict[str, str], Path]], cont_dirs: list[tuple[dict[str, str], Path]]) -> np.ndarray | None:
    """Prefer a target Caputo trajectory and otherwise use a continuation tail."""

    for row, path in hidden_dirs:
        candidate = path / "representative_trajectories" / f"target_candidate_{row['sign']}.csv"
        if candidate.exists() and is_ok_status(path / "status.json"):
            return load_trajectory(candidate)
    for _row, path in cont_dirs:
        candidate = path / "trajectory_tail.csv"
        if candidate.exists() and is_ok_status(path / "status.json"):
            return load_trajectory(candidate)
    return None


def _candidate_trajectories(hidden_dirs: list[tuple[dict[str, str], Path]]) -> list[tuple[dict[str, str], np.ndarray]]:
    """Load every target-system candidate integrated after continuation."""

    candidates: list[tuple[dict[str, str], np.ndarray]] = []
    for row, path in hidden_dirs:
        candidate = path / "representative_trajectories" / f"target_candidate_{row['sign']}.csv"
        if candidate.exists() and is_ok_status(path / "status.json"):
            candidates.append((row, load_trajectory(candidate)))
    return candidates


def _candidate_label(row: dict[str, str]) -> str:
    """Build a stable visual identifier for a continued candidate variant."""

    return (
        f"cont_{row['continuation_solver']}"
        f"__target_{row['hiddenness_integrator']}_{row['hiddenness_memory']}"
        f"__{row['sign']}"
    )


def _plot_candidate_overlay(directory: Path, candidates: list[tuple[dict[str, str], np.ndarray]]) -> list[str]:
    """Overlay all candidate attractor trajectories generated after continuation."""

    if not candidates:
        return _placeholder(directory, "phase3d_candidate", "No completed candidate trajectory is available.")
    fig = plt.figure(figsize=(8.4, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    for row, trajectory in candidates:
        sample = trajectory[np.linspace(0, len(trajectory) - 1, min(len(trajectory), 900)).astype(int)]
        ax.plot(sample[:, 1], sample[:, 2], sample[:, 3], lw=0.65, alpha=0.62, label=_candidate_label(row))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Target-system candidates after numerical continuation")
    if len(candidates) <= 8:
        ax.legend(fontsize=6, loc="best")
    return _save(fig, directory, "phase3d_candidate")


def _plot_candidate_time_series(directory: Path, candidates: list[tuple[dict[str, str], np.ndarray]]) -> list[str]:
    """Overlay x(t) for all target-system candidate variants."""

    if not candidates:
        return _placeholder(directory, "time_series", "No completed candidate trajectory is available.")
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for row, trajectory in candidates:
        ax.plot(trajectory[:, 0], trajectory[:, 1], lw=0.75, alpha=0.70, label=_candidate_label(row))
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title("Candidate trajectories after numerical continuation")
    if len(candidates) <= 8:
        ax.legend(fontsize=6, loc="best")
    return _save(fig, directory, "time_series")


def _plot_individual_candidates(directory: Path, candidates: list[tuple[dict[str, str], np.ndarray]]) -> list[str]:
    """Save phase and time-series figures for each candidate configuration."""

    outputs: list[str] = []
    target_dir = directory / "candidate_attractors"
    target_dir.mkdir(parents=True, exist_ok=True)
    for row, trajectory in candidates:
        label = _candidate_label(row)
        for suffix in ("png", "pdf"):
            outputs.append(
                plot_phase_space(
                    trajectory,
                    target_dir / f"phase3d_{label}.{suffix}",
                    title=f"Candidate after continuation: {label}",
                )
            )
            outputs.append(
                plot_time_series(
                    trajectory,
                    target_dir / f"time_series_{label}.{suffix}",
                    title=f"Candidate time series: {label}",
                )
            )
    return outputs


def _plot_continuation(directory: Path, cont_dirs: list[tuple[dict[str, str], Path]]) -> list[str]:
    """Plot endpoint paths for every completed continuation solver cell."""

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    any_data = False
    for row, path in cont_dirs:
        if not is_ok_status(path / "status.json"):
            continue
        rows = read_csv_rows(path / "continuation_path.csv")
        if not rows:
            continue
        eta = np.asarray([float(item["eta"]) for item in rows])
        xout = np.asarray([float(item["x_out"]) for item in rows])
        history = np.asarray([float(item["history_points_out"]) for item in rows])
        label = f"{row['continuation_solver']} / {row['continuation_memory']}"
        axes[0].plot(eta, xout, marker=".", lw=1.0, label=label)
        axes[1].plot(eta, history, marker=".", lw=1.0, label=label)
        any_data = True
    if not any_data:
        plt.close(fig)
        return _placeholder(directory, "continuation_path", "No completed continuation artifact is available.")
    axes[0].set_xlabel("eta (lambda)")
    axes[0].set_ylabel("final x")
    axes[1].set_xlabel("eta (lambda)")
    axes[1].set_ylabel("history points transported")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    return _save(fig, directory, "continuation_path")


def _plot_neighborhood_trajectories(directory: Path, hidden_dirs: list[tuple[dict[str, str], Path]]) -> list[str]:
    """Overlay stored representative equilibrium-neighborhood outcomes."""

    fig = plt.figure(figsize=(7.6, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    plotted = 0
    for _row, path in hidden_dirs:
        rep_dir = path / "representative_trajectories"
        if not rep_dir.exists():
            continue
        for candidate in sorted(rep_dir.glob("*.csv")):
            if candidate.stem.startswith("target_candidate"):
                continue
            traj = load_trajectory(candidate)
            sample = traj[np.linspace(0, len(traj) - 1, min(len(traj), 500)).astype(int)]
            ax.plot(sample[:, 1], sample[:, 2], sample[:, 3], lw=0.55, alpha=0.55, label=candidate.stem if plotted < 7 else None)
            plotted += 1
    if not plotted:
        plt.close(fig)
        return _placeholder(directory, "equilibrium_neighborhood_trajectories", "No representative equilibrium-neighborhood trajectories are available.")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=7)
    return _save(fig, directory, "equilibrium_neighborhood_trajectories")


def _plot_sampled_basin_evidence(directory: Path, hidden_dirs: list[tuple[dict[str, str], Path]]) -> list[str]:
    """Plot tested initial conditions; this is not claimed as a dense basin map."""

    rows: list[dict[str, str]] = []
    for _task, path in hidden_dirs:
        raw = path / "hiddenness_raw.csv"
        if raw.exists() and is_ok_status(path / "status.json"):
            rows.extend(read_csv_rows(raw))
    if not rows:
        return _placeholder(directory, "basin_slices_xy_xz", "No tested neighborhood initial-condition classifications are available.")
    labels = sorted({row["class_label"] for row in rows})
    colors = {label: plt.cm.tab10(index % 10) for index, label in enumerate(labels)}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for label in labels:
        subset = [row for row in rows if row["class_label"] == label]
        x = [float(row["x0"]) for row in subset]
        y = [float(row["y0"]) for row in subset]
        z = [float(row["z0"]) for row in subset]
        axes[0].scatter(x, y, s=6, alpha=0.58, color=colors[label], label=label)
        axes[1].scatter(x, z, s=6, alpha=0.58, color=colors[label])
    axes[0].set_xlabel("x0")
    axes[0].set_ylabel("y0")
    axes[1].set_xlabel("x0")
    axes[1].set_ylabel("z0")
    axes[0].set_title("xy sampled balls")
    axes[1].set_title("xz sampled balls")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Tested initial-condition slices (not a dense global basin map)")
    return _save(fig, directory, "basin_slices_xy_xz")


def _plot_spectra(directory: Path, trajectory: np.ndarray | None) -> list[str]:
    """Plot FFT and Welch PSD of the candidate x-coordinate."""

    if trajectory is None:
        return _placeholder(directory, "fft_psd", "No candidate trajectory is available for spectral diagnostics.")
    fft = trajectory_component_spectra(trajectory, components=(0,), method="fft")[0]
    psd = trajectory_component_spectra(trajectory, components=(0,), method="psd")[0]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].plot(fft.frequency_rad_s, fft.values, lw=0.9)
    axes[1].plot(psd.frequency_rad_s, psd.values, lw=0.9)
    axes[0].set_xlabel("omega [rad/s]")
    axes[1].set_xlabel("omega [rad/s]")
    axes[0].set_ylabel("FFT amplitude")
    axes[1].set_ylabel("PSD")
    axes[0].set_title("FFT, x")
    axes[1].set_title("Welch PSD, x")
    return _save(fig, directory, "fft_psd")


def _plot_hiddenness_summary(directory: Path, hidden_dirs: list[tuple[dict[str, str], Path]]) -> list[str]:
    """Show operational destination counts without promoting hiddenness."""

    all_rows: list[dict[str, str]] = []
    for _row, path in hidden_dirs:
        raw = path / "hiddenness_raw.csv"
        if raw.exists() and is_ok_status(path / "status.json"):
            all_rows.extend(read_csv_rows(raw))
    if not all_rows:
        return _placeholder(directory, "hiddenness_summary", "No completed hiddenness evidence is available.")
    counts = status_counts(all_rows, "class_label")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    names = list(counts)
    ax.bar(np.arange(len(names)), [counts[name] for name in names], color="#2563eb")
    ax.set_xticks(np.arange(len(names)), names, rotation=38, ha="right")
    ax.set_ylabel("tested initial conditions")
    ax.set_title("Finite-radius hiddenness evidence: operational outcomes only")
    return _save(fig, directory, "hiddenness_summary")


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    """Render the requested figure set for one experiment specification."""

    root = Path(job["root"])
    manifest = job["manifest"]
    row = job["row"]
    exp_id = row["exp_id"]
    directory = root / "figures" / exp_id
    status_path = directory / "status.json"
    required = [directory / f"{stem}.png" for stem in REQUESTED_FIGURES]
    if not bool(job["force"]) and is_ok_status(status_path, required):
        return {"task_id": row["task_id"], "status": "skipped_ok"}
    directory.mkdir(parents=True, exist_ok=True)
    spec = experiment_spec(manifest, exp_id)
    figure_row = {**row, "cache_key": row.get("cache_key", row["task_id"])}
    meta = metadata(manifest, figure_row, stage="figures", q=float(manifest["contract"]["q_target"]), integrator="postprocessing", memory_policy="from_source_artifacts", workers=int(job["workers"]))
    outputs: list[str] = []
    try:
        system = get_system("chua-nonsmooth")
        if system.lure is None:
            raise ValueError("Chua Lur'e split is unavailable.")
        seed_name = "fractional" if spec["seed_family"] == "fractional" else "integer_like"
        seed = harmonic_seed_from_payload(read_json(root / "shared" / "seeds" / f"{seed_name}_seed.json"))
        q_plot = float(manifest["contract"]["q_target"]) if seed_name == "fractional" else 1.0
        for suffix in ("png", "pdf"):
            outputs.append(
                plot_lure_nyquist_describing_function(
                    system.lure,
                    seed,
                    directory / f"seed_nyquist.{suffix}",
                    q=q_plot,
                    method="classic",
                    title=f"{exp_id}: classical DF seed",
                )
            )
        continuations = _continuations(root, exp_id)
        hiddenness = _hiddenness(root, exp_id)
        outputs.extend(_plot_continuation(directory, continuations))
        candidates = _candidate_trajectories(hiddenness)
        trajectory = _first_valid_trajectory(hiddenness, continuations)
        outputs.extend(_plot_candidate_overlay(directory, candidates))
        outputs.extend(_plot_candidate_time_series(directory, candidates))
        outputs.extend(_plot_individual_candidates(directory, candidates))
        outputs.extend(_plot_neighborhood_trajectories(directory, hiddenness))
        outputs.extend(_plot_sampled_basin_evidence(directory, hiddenness))
        outputs.extend(_plot_spectra(directory, trajectory))
        outputs.extend(_plot_hiddenness_summary(directory, hiddenness))
        write_status(status_path, status="ok", meta=meta, outputs=outputs)
        return {"task_id": row["task_id"], "status": "ok"}
    except Exception as exc:
        write_status(status_path, status="failed", meta=meta, outputs=outputs, reason=str(exc))
        return {"task_id": row["task_id"], "status": "failed", "reason": str(exc)}


def main() -> None:
    """Dispatch per-experiment postprocessing figures by independent process."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(VERSION2_ROOT / "outputs/chua_nonsmooth_fractional_memory_matrix/tasks/figure_tasks.csv"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root, manifest, rows = load_matrix(args.tasks)
    exp_ids = [row["exp_id"] for row in rows]
    if len(exp_ids) != len(set(exp_ids)):
        raise ValueError("figure task table contains duplicate exp_id output directories.")
    jobs = [{"root": str(root), "manifest": manifest, "row": row, "workers": args.workers, "force": args.force} for row in rows]
    for result in run_process_pool(run_one, jobs, workers=args.workers):
        print(f"{result['task_id']}: {result['status']}")


if __name__ == "__main__":
    main()
