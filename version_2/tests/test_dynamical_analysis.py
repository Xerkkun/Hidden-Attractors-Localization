"""Tests for trajectory-adapted dynamical analysis helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hidden_attractors.analysis import bifurcation_points_from_trajectories, bifurcation_summary
from hidden_attractors.integrations import compute_complexity_measures, external_tool_report
from hidden_attractors.integrations import external_tools
from hidden_attractors.io import load_trajectory_csv


ROOT = Path(__file__).resolve().parents[1]


def test_load_trajectory_csv_with_project_columns() -> None:
    path = ROOT / "outputs" / "tests" / "trajectory_fixture.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
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


def test_rosenstein_adapter_uses_sample_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, float] = {}

    def fake_lyap_r(signal: np.ndarray, *, tau: float) -> float:
        observed["tau"] = tau
        observed["samples"] = float(signal.size)
        return 2.5

    fake_nolds = SimpleNamespace(lyap_r=fake_lyap_r)
    monkeypatch.setattr(
        external_tools,
        "require_external",
        lambda import_name: fake_nolds,
    )

    result = compute_complexity_measures(
        np.linspace(0.0, 1.0, 16),
        backend="nolds",
        sample_rate=50.0,
        measures=["lyapunov_rosenstein"],
    )

    assert result == {"lyapunov_rosenstein": 2.5}
    assert observed == {"tau": pytest.approx(0.02), "samples": 16.0}


@pytest.mark.parametrize("sample_rate", [0.0, -1.0, np.inf, np.nan])
def test_complexity_adapter_rejects_invalid_sample_rate(
    sample_rate: float,
) -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        compute_complexity_measures(
            np.linspace(0.0, 1.0, 16),
            backend="nolds",
            sample_rate=sample_rate,
            measures=["lyapunov_rosenstein"],
        )
