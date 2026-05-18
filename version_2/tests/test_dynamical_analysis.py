"""Tests for trajectory-adapted dynamical analysis helpers."""

from __future__ import annotations

import numpy as np

from hidden_attractors.analysis import bifurcation_points_from_trajectories, bifurcation_summary
from hidden_attractors.integrations import external_tool_report
from hidden_attractors.io import load_trajectory_csv


def test_load_trajectory_csv_with_project_columns(tmp_path) -> None:
    path = tmp_path / "trajectory.csv"
    path.write_text("t,x,y,z\n0,1,2,3\n1,4,5,6\n", encoding="utf-8")

    trajectory = load_trajectory_csv(path)

    assert trajectory.shape == (2, 4)
    assert np.allclose(trajectory[:, 0], [0.0, 1.0])
    assert np.allclose(trajectory[:, 3], [3.0, 6.0])


def test_bifurcation_points_from_txyz_trajectories() -> None:
    t = np.linspace(0.0, 10.0, 101)
    scan = []
    for parameter in (0.9, 1.0):
        x = parameter * np.sin(t)
        trajectory = np.column_stack([t, x, np.cos(t), x - np.cos(t)])
        scan.append((parameter, trajectory))

    points = bifurcation_points_from_trajectories(scan, observable="x", t_start=2.0)
    summary = bifurcation_summary(points)

    assert len(points) >= 2
    assert summary["n_points"] == len(points)
    assert summary["parameter_min"] == 0.9
    assert summary["parameter_max"] == 1.0


def test_external_tool_report_documents_companion_tools() -> None:
    report = external_tool_report()
    names = {row["name"] for row in report}

    assert "PyDSTool" in names
    assert "pyComplexity notebook" in names
