#!/usr/bin/env python3
"""Visualize top-3 sphere hiddenness controls in 3D.

The sphere-control CSVs contain initial conditions on spheres centered at each
Chua equilibrium and the final class assigned by the EFORK/C classifier.  This
script plots the largest tested sphere, the sampled initial conditions, the
equilibrium center, and a short EFORK trajectory segment from a stratified
subset of initial conditions across all tested radii.  The sphere is only the
tested-neighborhood reference; trajectory curves continue beyond it and are
clipped only when they leave the visible plotting window.

The short segments are visual aids only; the colors come from the completed
long classification run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent
_CACHE_ROOT = ROOT / ".runtime_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg_cache"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from hidden_attractors.io import read_csv_rows, read_json, safe_name, write_json
from hidden_attractors.native import FractionalChuaBackend


CLASS_COLORS = {
    "equilibrium": "#111827",
    "target_positive": "#16a34a",
    "target_negative": "#2563eb",
    "infinity": "#dc2626",
    "unknown": "#9ca3af",
    "numerical_failure": "#f59e0b",
}
CLASS_ORDER = ["equilibrium", "target_positive", "target_negative", "infinity", "unknown", "numerical_failure"]
CLASS_DISPLAY = {
    "equilibrium": "equilibrio",
    "target_positive": "target +",
    "target_negative": "target -",
    "infinity": "diverge",
    "unknown": "unknown",
    "numerical_failure": "fallo numerico",
}


def _row_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _equilibrium_file_label(equilibrium_id: str) -> str:
    labels = {"E+": "Eplus", "E-": "Eminus", "E0": "E0"}
    return labels.get(equilibrium_id, safe_name(equilibrium_id))


def _load_rows(root: Path) -> list[dict[str, str]]:
    merged = root / "sphere_raw.csv"
    if merged.exists():
        return read_csv_rows(merged)
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("sphere_raw_chunk_*.csv")):
        rows.extend(read_csv_rows(path))
    return rows


def _sphere(center: np.ndarray, radius: float, n: int = 28) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, n)
    v = np.linspace(0.0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _sample_rows_by_radius_and_class(
    rows: Sequence[dict[str, str]],
    *,
    max_total: int,
    seed: int,
) -> list[dict[str, str]]:
    """Choose a deterministic, balanced visual subset.

    Mathematical purpose:
        The hiddenness test samples concentric spherical neighborhoods around
        an equilibrium.  A visual diagnostic should therefore retain evidence
        from every tested radius instead of drawing only the outer shell.

    Validity warning:
        This function changes only visualization density.  It does not alter
        the completed EFORK/C classifications stored in the CSV files.
    """

    if len(rows) <= max_total:
        return list(rows)
    rng = np.random.default_rng(int(seed))
    by_radius: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_radius[float(row["radius"])].append(row)
    radii = sorted(by_radius)
    base = max_total // len(radii)
    remainder = max_total % len(radii)
    selected: list[dict[str, str]] = []
    selected_ids: set[int] = set()
    for radius_index, radius in enumerate(radii):
        quota = base + (1 if radius_index < remainder else 0)
        radius_rows = by_radius[radius]
        by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in radius_rows:
            by_class[row.get("class_label", "unknown")].append(row)
        labels = [label for label in CLASS_ORDER if by_class.get(label)]
        class_quota = max(1, quota // max(1, len(labels)))
        radius_selected: list[dict[str, str]] = []
        for label in labels:
            candidates = by_class[label]
            take = min(class_quota, len(candidates))
            if take:
                idx = rng.choice(len(candidates), size=take, replace=False)
                radius_selected.extend(candidates[int(i)] for i in np.atleast_1d(idx))
        if len(radius_selected) < quota:
            used_local = {id(row) for row in radius_selected}
            remaining = [row for row in radius_rows if id(row) not in used_local]
            take = min(quota - len(radius_selected), len(remaining))
            if take:
                idx = rng.choice(len(remaining), size=take, replace=False)
                radius_selected.extend(remaining[int(i)] for i in np.atleast_1d(idx))
        for row in radius_selected[:quota]:
            if id(row) not in selected_ids:
                selected.append(row)
                selected_ids.add(id(row))
    if len(selected) < max_total:
        remaining = [row for row in rows if id(row) not in selected_ids]
        take = min(max_total - len(selected), len(remaining))
        if take:
            idx = rng.choice(len(remaining), size=take, replace=False)
            selected.extend(remaining[int(i)] for i in np.atleast_1d(idx))
    return selected[:max_total]


AxisLimits = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def _axis_limits(points: np.ndarray, center: np.ndarray, radius: float) -> AxisLimits:
    if points.size:
        mins = np.nanmin(points, axis=0)
        maxs = np.nanmax(points, axis=0)
    else:
        mins = center - radius
        maxs = center + radius
    mins = np.minimum(mins, center - radius)
    maxs = np.maximum(maxs, center + radius)
    span = np.maximum(maxs - mins, 8.0 * radius)
    mid = 0.5 * (mins + maxs)
    pad = np.maximum(0.15 * span, 2.0 * radius)
    lo = mid - 0.5 * span - pad
    hi = mid + 0.5 * span + pad
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1])), (float(lo[2]), float(hi[2]))


def _local_window_limits(center: np.ndarray, radius: float, view_radius_factor: float) -> AxisLimits:
    """Build a fixed local plot box around one equilibrium.

    Mathematical purpose:
        The tested sphere has radius ``radius`` around an equilibrium.  The
        local plot shows the beginning of the flow outside that neighborhood,
        so the axis interval is expanded by ``view_radius_factor`` while still
        keeping the equilibrium-centered geometry readable.
    """

    half_width = max(float(radius) * float(view_radius_factor), float(radius))
    lo = center - half_width
    hi = center + half_width
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1])), (float(lo[2]), float(hi[2]))


def _set_equalish(ax: plt.Axes, limits: AxisLimits) -> None:
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$x_3$")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.75)


def _integrate_segments(
    backend: FractionalChuaBackend,
    rows: Sequence[dict[str, str]],
    *,
    q: float,
    h: float,
    Lm: float,
    segment_t: float,
    max_segment_points: int,
) -> dict[int, np.ndarray]:
    segments: dict[int, np.ndarray] = {}
    for idx, row in enumerate(rows):
        x0 = [_row_float(row, "x0"), _row_float(row, "y0"), _row_float(row, "z0")]
        try:
            traj = backend.integrate_efork3(x0, q=q, h=h, Lm=Lm, t_final=segment_t)
            states = traj[:, 1:4]
            if states.shape[0] > max_segment_points:
                take = np.linspace(0, states.shape[0] - 1, max_segment_points).astype(int)
                states = states[take]
            segments[idx] = states
        except Exception:
            segments[idx] = np.asarray([x0], dtype=float)
    return segments


def _point_inside_limits(point: np.ndarray, limits: AxisLimits) -> bool:
    return all(limits[dim][0] <= float(point[dim]) <= limits[dim][1] for dim in range(3))


def _boundary_crossing(prev: np.ndarray, curr: np.ndarray, limits: AxisLimits) -> np.ndarray:
    """Linearly interpolate the first crossing with the plot box boundary."""

    lambdas: list[float] = []
    delta = curr - prev
    for dim in range(3):
        if abs(float(delta[dim])) < 1.0e-15:
            continue
        lo, hi = limits[dim]
        if curr[dim] < lo:
            lambdas.append(float((lo - prev[dim]) / delta[dim]))
        elif curr[dim] > hi:
            lambdas.append(float((hi - prev[dim]) / delta[dim]))
    valid = [lam for lam in lambdas if 0.0 <= lam <= 1.0]
    if not valid:
        return curr
    lam = min(valid)
    return prev + lam * delta


def _clip_segment_to_plot_window(segment: np.ndarray, limits: AxisLimits) -> np.ndarray:
    """Return the initial contiguous trajectory piece inside the visible box."""

    if segment.size == 0:
        return segment
    if not _point_inside_limits(segment[0], limits):
        return segment[:1]
    clipped = [segment[0]]
    for point in segment[1:]:
        if _point_inside_limits(point, limits):
            clipped.append(point)
            continue
        clipped.append(_boundary_crossing(clipped[-1], point, limits))
        break
    return np.asarray(clipped, dtype=float)


def _plot_local(
    outdir: Path,
    candidate_id: str,
    equilibrium_id: str,
    rows: Sequence[dict[str, str]],
    center: np.ndarray,
    radius: float,
    segments: dict[int, np.ndarray],
    *,
    segment_t: float,
    available_rows: Sequence[dict[str, str]],
    view_radius_factor: float,
) -> str:
    fig = plt.figure(figsize=(8.0, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f3f4f6")
    sx, sy, sz = _sphere(center, radius)
    ax.plot_wireframe(sx, sy, sz, color="#6b7280", linewidth=0.55, alpha=0.65)
    ax.scatter([center[0]], [center[1]], [center[2]], s=72, c="white", edgecolors="black", linewidths=1.2, depthshade=False, label=f"{equilibrium_id} equilibrium")
    ax.text(center[0], center[1], center[2], f" {equilibrium_id}", fontsize=9)
    limits = _local_window_limits(center, radius, view_radius_factor)

    all_points: list[np.ndarray] = [center[None, :]]
    for idx, row in enumerate(rows):
        label = row.get("class_label", "unknown")
        color = CLASS_COLORS.get(label, "#9ca3af")
        segment = _clip_segment_to_plot_window(segments.get(idx, np.empty((0, 3))), limits)
        if segment.size:
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, lw=0.55, alpha=0.38)
            all_points.append(segment)
        p = np.array([_row_float(row, "x0"), _row_float(row, "y0"), _row_float(row, "z0")], dtype=float)
        ax.scatter([p[0]], [p[1]], [p[2]], s=22, c=color, edgecolors="black", linewidths=0.35, depthshade=False)
        all_points.append(p[None, :])

    _set_equalish(ax, limits)
    ax.view_init(elev=22, azim=-58)
    counts = Counter(row.get("class_label", "unknown") for row in rows)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[k], markeredgecolor="black", markersize=7, label=f"{CLASS_DISPLAY.get(k, k)}: {counts.get(k, 0)}") for k in CLASS_ORDER if counts.get(k, 0)]
    handles.append(plt.Line2D([0], [0], color="#6b7280", lw=1.0, label=f"largest sphere r={radius:g}"))
    handles.append(plt.Line2D([0], [0], color="#374151", lw=1.0, label=f"segments clipped by plot window"))
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.92)
    radii = sorted({float(row["radius"]) for row in available_rows})
    ax.set_title(
        f"{candidate_id}\n"
        f"{equilibrium_id}: {len(rows)}/{len(available_rows)} sampled ICs across {len(radii)} radii, t<= {segment_t:g}"
    )
    fig.tight_layout()
    path = outdir / f"sphere_geometry_{safe_name(candidate_id)}_{_equilibrium_file_label(equilibrium_id)}.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return str(path)


def _plot_candidate_overview(
    outdir: Path,
    candidate_id: str,
    rows: Sequence[dict[str, str]],
    equilibria: dict[str, list[float]],
    radius: float,
) -> str:
    fig = plt.figure(figsize=(9.2, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f3f4f6")
    all_points: list[np.ndarray] = []
    for eq_id, center_list in equilibria.items():
        center = np.asarray(center_list, dtype=float)
        sx, sy, sz = _sphere(center, radius)
        ax.plot_wireframe(sx, sy, sz, color="#6b7280", linewidth=0.45, alpha=0.55)
        ax.scatter([center[0]], [center[1]], [center[2]], s=68, c="white", edgecolors="black", linewidths=1.1, depthshade=False)
        ax.text(center[0], center[1], center[2], f" {eq_id}", fontsize=9)
        all_points.append(center[None, :])
    for row in rows:
        p = np.array([_row_float(row, "x0"), _row_float(row, "y0"), _row_float(row, "z0")], dtype=float)
        color = CLASS_COLORS.get(row.get("class_label", "unknown"), "#9ca3af")
        ax.scatter([p[0]], [p[1]], [p[2]], s=9, c=color, alpha=0.72, depthshade=False)
        all_points.append(p[None, :])
    points = np.vstack(all_points)
    _set_equalish(ax, _axis_limits(points, np.mean(points, axis=0), radius))
    ax.view_init(elev=20, azim=-56)
    counts = Counter(row.get("class_label", "unknown") for row in rows)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[k], markeredgecolor="black", markersize=7, label=f"{k}: {counts.get(k, 0)}") for k in CLASS_ORDER if counts.get(k, 0)]
    handles.append(plt.Line2D([0], [0], color="#6b7280", lw=1.0, label=f"largest tested spheres r={radius:g}"))
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.92)
    ax.set_title(f"{candidate_id}\nall equilibria, sampled ICs inside largest tested spheres")
    fig.tight_layout()
    path = outdir / f"sphere_geometry_overview_{safe_name(candidate_id)}.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return str(path)


def make_plots(
    root: Path,
    *,
    segment_t: float,
    max_segment_points: int,
    max_trajectories_per_plot: int,
    sample_seed: int,
    view_radius_factor: float,
    output_subdir: str,
) -> dict[str, Any]:
    cfg = read_json(root / "top3_sphere_robustness_config.json")
    rows = _load_rows(root)
    max_radius = max(float(row["radius"]) for row in rows)
    outdir = root / "plots" / output_subdir
    outdir.mkdir(parents=True, exist_ok=True)
    backend = FractionalChuaBackend.build(output_name="chua_frac_top3_sphere_geometry")
    contract = cfg["sphere_contract"]
    q = float(contract["q"])
    h = float(contract["h"])
    Lm = float(contract["memory_length"])
    plots: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_candidate[row["candidate_id"]].append(row)
    for candidate_id, cand_rows in sorted(by_candidate.items()):
        cand_overview_rows: list[dict[str, str]] = []
        for eq_id, center_list in cfg["equilibria"].items():
            local_rows = [row for row in cand_rows if row["equilibrium_id"] == eq_id]
            if not local_rows:
                continue
            local_seed = sample_seed + 1009 * len(summary_rows)
            sampled_rows = _sample_rows_by_radius_and_class(local_rows, max_total=max_trajectories_per_plot, seed=local_seed)
            cand_overview_rows.extend(sampled_rows)
            center = np.asarray(center_list, dtype=float)
            segments = _integrate_segments(
                backend,
                sampled_rows,
                q=q,
                h=h,
                Lm=Lm,
                segment_t=segment_t,
                max_segment_points=max_segment_points,
            )
            plots.append(
                _plot_local(
                    outdir,
                    candidate_id,
                    eq_id,
                    sampled_rows,
                    center,
                    max_radius,
                    segments,
                    segment_t=segment_t,
                    available_rows=local_rows,
                    view_radius_factor=view_radius_factor,
                )
            )
            by_radius = sorted({float(row["radius"]) for row in local_rows})
            for radius in by_radius:
                tol = max(abs(radius) * 1.0e-9, 1.0e-15)
                available_radius_rows = [row for row in local_rows if abs(float(row["radius"]) - radius) <= tol]
                sampled_radius_rows = [row for row in sampled_rows if abs(float(row["radius"]) - radius) <= tol]
                available_counts = Counter(row.get("class_label", "unknown") for row in available_radius_rows)
                sampled_counts = Counter(row.get("class_label", "unknown") for row in sampled_radius_rows)
                summary_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "equilibrium_id": eq_id,
                        "radius": radius,
                        "n_available_initial_conditions": len(available_radius_rows),
                        "n_sampled_initial_conditions": len(sampled_radius_rows),
                        **{f"available_{key}": available_counts.get(key, 0) for key in CLASS_ORDER},
                        **{f"sampled_{key}": sampled_counts.get(key, 0) for key in CLASS_ORDER},
                    }
                )
        plots.append(_plot_candidate_overview(outdir, candidate_id, cand_overview_rows, cfg["equilibria"], max_radius))
    summary = {
        "source_dir": str(root),
        "output_dir": str(outdir),
        "largest_radius": max_radius,
        "segment_t": float(segment_t),
        "max_segment_points": int(max_segment_points),
        "max_trajectories_per_local_plot": int(max_trajectories_per_plot),
        "sample_seed": int(sample_seed),
        "view_radius_factor": float(view_radius_factor),
        "local_plot_half_width": float(max_radius) * float(view_radius_factor),
        "plots": plots,
        "summary_by_candidate_equilibrium": summary_rows,
        "notes": [
            "Colors are final classes from the completed sphere-control classifier.",
            "Trajectory curves are short EFORK visual segments from sampled initial conditions, not a reclassification.",
            "Local plots sample across all tested radii and clip each trajectory only when it leaves the visible plot window.",
            "Overview plots show sampled initial conditions and largest spheres for all equilibria.",
        ],
    }
    write_json(outdir / "sphere_geometry_summary.json", summary)
    with (outdir / "sphere_geometry_counts.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "candidate_id",
            "equilibrium_id",
            "radius",
            "n_available_initial_conditions",
            "n_sampled_initial_conditions",
            *[f"available_{key}" for key in CLASS_ORDER],
            *[f"sampled_{key}" for key in CLASS_ORDER],
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot top3 sphere-control geometry with short EFORK trajectory segments.")
    parser.add_argument("--input-dir", default=str(ROOT / "outputs" / "top3_machado_lure_sphere_C_20260517_v2"))
    parser.add_argument("--segment-t", type=float, default=2.0)
    parser.add_argument("--max-segment-points", type=int, default=90)
    parser.add_argument("--max-trajectories-per-plot", type=int, default=180)
    parser.add_argument("--sample-seed", type=int, default=20260517)
    parser.add_argument("--view-radius-factor", type=float, default=4.0)
    parser.add_argument("--output-subdir", default="sphere_geometry_zoom_r4_plot_window_segments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = make_plots(
        Path(args.input_dir).resolve(),
        segment_t=float(args.segment_t),
        max_segment_points=int(args.max_segment_points),
        max_trajectories_per_plot=int(args.max_trajectories_per_plot),
        sample_seed=int(args.sample_seed),
        view_radius_factor=float(args.view_radius_factor),
        output_subdir=str(args.output_subdir),
    )
    print(json.dumps(summary, indent=2)[:4000])


if __name__ == "__main__":
    main()
