"""Plot fractional reference-attractor trajectories from validation diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_ROOT = ROOT / "validation" / "chaos_validation" / "dynamics_diagnostics" / "trajectories"
OUTDIR = ROOT / "validation" / "chaos_validation" / "dynamics_diagnostics" / "figures" / "fractional_reference_attractors"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_trajectory(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append([float(row["t"]), float(row["x"]), float(row["y"]), float(row["z"])])
    return np.asarray(rows, dtype=float)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("_")


def _sample_for_plot(traj: np.ndarray, max_points: int = 6000) -> np.ndarray:
    if len(traj) <= max_points:
        return traj
    idx = np.linspace(0, len(traj) - 1, max_points, dtype=int)
    return traj[idx]


def _title(meta: dict[str, Any]) -> str:
    return (
        f"{meta.get('case_id')} | {meta.get('trajectory_id')}\n"
        f"{meta.get('source_reference')} | q={meta.get('q')} | h={meta.get('h')} | "
        f"{meta.get('integrator')} | {meta.get('backend')}"
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{stem}.png", dpi=240)
    fig.savefig(OUTDIR / f"{stem}.pdf")
    plt.close(fig)


def plot_case(meta: dict[str, Any], traj: np.ndarray) -> list[str]:
    stem = _safe_name(f"{meta.get('case_id')}_{meta.get('trajectory_id')}")
    sample = _sample_for_plot(traj)
    t, x, y, z = sample[:, 0], sample[:, 1], sample[:, 2], sample[:, 3]
    written: list[str] = []

    fig = plt.figure(figsize=(8.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, color="#2563eb", linewidth=0.45, alpha=0.88)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(_title(meta), fontsize=10)
    _save(fig, f"{stem}_phase3d")
    written.append(f"{stem}_phase3d")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    projections = [
        ("xy", x, y, "x", "y"),
        ("xz", x, z, "x", "z"),
        ("yz", y, z, "y", "z"),
    ]
    for ax, (name, u, v, xlabel, ylabel) in zip(axes, projections):
        ax.plot(u, v, color="#0f766e", linewidth=0.42, alpha=0.9)
        ax.set_title(name.upper())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(color="#cbd5e1", linestyle=":", linewidth=0.7)
    fig.suptitle(_title(meta), fontsize=10)
    _save(fig, f"{stem}_projections")
    written.append(f"{stem}_projections")

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True)
    series = [("x", x, "#2563eb"), ("y", y, "#dc2626"), ("z", z, "#0f766e")]
    for ax, (label, values, color) in zip(axes, series):
        ax.plot(t, values, color=color, linewidth=0.55)
        ax.set_ylabel(label)
        ax.grid(color="#cbd5e1", linestyle=":", linewidth=0.7)
    axes[-1].set_xlabel("t")
    fig.suptitle(_title(meta), fontsize=10)
    _save(fig, f"{stem}_timeseries")
    written.append(f"{stem}_timeseries")

    return written


def plot_fractional_overview(cases: list[tuple[dict[str, Any], np.ndarray]]) -> None:
    if not cases:
        return

    fig = plt.figure(figsize=(11.5, 8.0))
    for idx, (meta, traj) in enumerate(cases, start=1):
        sample = _sample_for_plot(traj, max_points=3500)
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.plot(sample[:, 1], sample[:, 2], sample[:, 3], linewidth=0.4)
        ax.set_title(
            f"{meta.get('source_reference')}\n{meta.get('case_id')} {meta.get('trajectory_id')}",
            fontsize=8,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.suptitle("Fractional validation-diagnostic attractor trajectories", fontsize=12)
    _save(fig, "fractional_attractors_overview_3d")


def write_report(cases: list[tuple[dict[str, Any], np.ndarray]], written: dict[str, list[str]]) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fractional Reference Attractor Figures",
        "",
        "These figures are generated from validation diagnostics, not from the Lure/EFORK experiment outputs.",
        "",
    ]
    for meta, traj in cases:
        key = f"{meta.get('case_id')}:{meta.get('trajectory_id')}"
        lines.extend(
            [
                f"## {key}",
                "",
                f"- Source: {meta.get('source_reference')}",
                f"- q: {meta.get('q')}",
                f"- h: {meta.get('h')}",
                f"- t_final: {meta.get('t_final')}",
                f"- t_burn: {meta.get('t_burn')}",
                f"- integrator: {meta.get('integrator')}",
                f"- backend: {meta.get('backend')}",
                f"- memory_policy: {meta.get('memory_policy')}",
                f"- seed_scope: {meta.get('seed_scope', 'not recorded')}",
                f"- sampled rows plotted: {len(traj)}",
                f"- figures: {', '.join(written.get(key, []))}",
                "",
            ]
        )
        if str(meta.get("case_id", "")).startswith("danca2017"):
            lines.extend(
                [
                    "Danca-specific note: the article does not disclose the exact hidden-attractor initial condition or DF seed parameters in the local reproduction metadata. This plotted trajectory uses the repository diagnostic seed and native full-history ABM contract.",
                    "",
                ]
            )
    (OUTDIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cases: list[tuple[dict[str, Any], np.ndarray]] = []
    written: dict[str, list[str]] = {}

    for meta_path in sorted(TRAJECTORY_ROOT.glob("**/*_metadata.json")):
        meta = _read_json(meta_path)
        q = float(meta.get("q", 1.0))
        if q >= 1.0:
            continue
        csv_rel = meta.get("post_transient_sampled_csv")
        if not csv_rel:
            continue
        csv_path = ROOT / str(csv_rel)
        if not csv_path.exists():
            continue
        traj = _read_trajectory(csv_path)
        cases.append((meta, traj))
        key = f"{meta.get('case_id')}:{meta.get('trajectory_id')}"
        written[key] = plot_case(meta, traj)

    plot_fractional_overview(cases)
    write_report(cases, written)
    print(f"Wrote {sum(len(v) for v in written.values()) + (1 if cases else 0)} figure groups to {OUTDIR}")


if __name__ == "__main__":
    main()
