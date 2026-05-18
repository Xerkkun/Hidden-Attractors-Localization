"""Plotting functions for dynamical-system trajectories and scans."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.bifurcation import BifurcationPoint, observable_column
from ..analysis.trajectory import sample_rows


def _output_path(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_phase_space(
    trajectory: np.ndarray,
    output_path: str | Path,
    *,
    dims: Sequence[str | int] = ("x", "y", "z"),
    title: str = "Phase space",
    max_points: int = 5000,
    color_by_time: bool = True,
) -> str:
    """Plot a 2D or 3D phase-space view of a ``t,x,y,z`` trajectory."""

    X = sample_rows(np.asarray(trajectory, dtype=float), max_points)
    cols = [observable_column(dim) for dim in dims]
    if len(cols) not in {2, 3}:
        raise ValueError("dims must contain two or three observables")
    if max(cols) >= X.shape[1]:
        raise ValueError("trajectory does not contain all requested dimensions")

    path = _output_path(output_path)
    fig = plt.figure(figsize=(7.5, 6.5))
    colors = X[:, 0] if color_by_time and X.shape[1] > 0 else "#2563eb"
    if len(cols) == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X[:, cols[0]], X[:, cols[1]], X[:, cols[2]], c=colors, s=2.2, cmap="viridis", alpha=0.78)
        ax.set_zlabel(str(dims[2]))
    else:
        ax = fig.add_subplot(111)
        ax.scatter(X[:, cols[0]], X[:, cols[1]], c=colors, s=2.2, cmap="viridis", alpha=0.78)
    ax.set_xlabel(str(dims[0]))
    ax.set_ylabel(str(dims[1]))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def plot_phase_projections(
    trajectory: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "Phase projections",
    max_points: int = 5000,
) -> str:
    """Plot standard ``xy``, ``xz``, and ``yz`` projections."""

    X = sample_rows(np.asarray(trajectory, dtype=float), max_points)
    path = _output_path(output_path)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    specs = [(1, 2, "x", "y"), (1, 3, "x", "z"), (2, 3, "y", "z")]
    for ax, (a, b, xlabel, ylabel) in zip(axes, specs):
        ax.plot(X[:, a], X[:, b], lw=0.55, color="#2563eb", alpha=0.82)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel}{ylabel}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def plot_time_series(
    trajectory: np.ndarray,
    output_path: str | Path,
    *,
    columns: Sequence[str | int] = ("x", "y", "z"),
    title: str = "Time series",
    max_points: int = 6000,
) -> str:
    """Plot selected trajectory coordinates against time."""

    X = sample_rows(np.asarray(trajectory, dtype=float), max_points)
    path = _output_path(output_path)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for column in columns:
        col = observable_column(column)
        ax.plot(X[:, 0], X[:, col], lw=0.75, label=str(column))
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)


def plot_bifurcation_diagram(
    points: Sequence[BifurcationPoint],
    output_path: str | Path,
    *,
    parameter_label: str = "parameter",
    observable_label: str = "observable",
    title: str = "Bifurcation diagram",
) -> str:
    """Plot extracted bifurcation points from a parameter scan."""

    path = _output_path(output_path)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if points:
        params = np.array([p.parameter for p in points], dtype=float)
        values = np.array([p.observable for p in points], dtype=float)
        ax.scatter(params, values, s=2.5, color="#111827", alpha=0.72)
    ax.set_xlabel(parameter_label)
    ax.set_ylabel(observable_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return str(path)
