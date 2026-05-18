#!/usr/bin/env python3
"""Generate phase-space, time-series, and bifurcation plots.

The default run uses a short synthetic Chua-like signal so the example is fast.
Pass ``--trajectory-csv`` to plot one of this project's real trajectory CSVs
with columns ``t,x,y,z``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hidden_attractors.analysis import bifurcation_points_from_trajectories, bifurcation_summary
from hidden_attractors.io import load_trajectory_csv, write_csv, write_json
from hidden_attractors.paths import OUTPUTS
from hidden_attractors.plotting import (
    plot_bifurcation_diagram,
    plot_phase_projections,
    plot_phase_space,
    plot_time_series,
)


def synthetic_chua_like_trajectory(mu: float, *, rows: int = 2400) -> np.ndarray:
    """Return a small signal using the package trajectory convention."""

    t = np.linspace(0.0, 120.0, rows)
    x = (1.0 + 0.35 * mu) * np.sin((0.65 + 0.08 * mu) * t) + 0.18 * np.sin(2.3 * t)
    y = (0.8 + 0.20 * mu) * np.cos((0.42 + 0.05 * mu) * t + 0.4)
    z = 0.35 * x - 0.55 * y + 0.1 * np.sin(0.13 * t)
    return np.column_stack([t, x, y, z])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        help="Optional CSV with columns t,x,y,z from this project's outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS / "examples" / "dynamical_analysis_gallery",
        help="Directory for generated figures and summary files.",
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=40.0,
        help="Transient cutoff used for bifurcation post-processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.trajectory_csv:
        trajectory = load_trajectory_csv(args.trajectory_csv)
        source = str(args.trajectory_csv)
    else:
        trajectory = synthetic_chua_like_trajectory(1.0)
        source = "synthetic_chua_like_trajectory(mu=1.0)"

    phase_path = plot_phase_space(trajectory, outdir / "phase_space_3d.png", title="Phase space")
    projections_path = plot_phase_projections(trajectory, outdir / "phase_projections.png")
    timeseries_path = plot_time_series(trajectory, outdir / "time_series.png")

    scans = [(mu, synthetic_chua_like_trajectory(mu)) for mu in np.linspace(0.2, 2.5, 40)]
    points = bifurcation_points_from_trajectories(
        scans,
        observable="x",
        t_start=args.t_start,
        mode="maxima",
    )
    bif_path = plot_bifurcation_diagram(
        points,
        outdir / "bifurcation_diagram.png",
        parameter_label="mu",
        observable_label="local maxima of x",
        title="Post-processed bifurcation diagram",
    )

    write_csv(outdir / "bifurcation_points.csv", [point.as_dict() for point in points])
    summary = {
        "trajectory_source": source,
        "phase_space": phase_path,
        "phase_projections": projections_path,
        "time_series": timeseries_path,
        "bifurcation": bif_path,
        "bifurcation_summary": bifurcation_summary(points),
    }
    write_json(outdir / "summary.json", summary)

    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
