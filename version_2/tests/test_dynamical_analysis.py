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
    assert "nolds" in names
    assert "antropy" in names
    assert "pyComplexity notebook" not in names


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


def test_complexity_adapter_rejects_unknown_measure() -> None:
    with pytest.raises(ValueError, match="Unknown complexity measure"):
        compute_complexity_measures(
            np.linspace(0.0, 1.0, 16),
            backend="auto",
            measures=["not_a_measure"],
        )


def test_complexity_adapter_rejects_measure_unsupported_by_explicit_backend() -> None:
    with pytest.raises(ValueError, match="does not support"):
        compute_complexity_measures(
            np.linspace(0.0, 1.0, 16),
            backend="nolds",
            measures=["permutation_entropy"],
        )


def test_complexity_auto_routes_each_measure_to_supporting_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []
    fake_nolds = SimpleNamespace(
        lyap_r=lambda signal, *, tau: 1.25,
    )
    fake_antropy = SimpleNamespace(
        perm_entropy=lambda signal, *, normalize: 0.75,
    )

    monkeypatch.setattr(
        external_tools,
        "available_complexity_backends",
        lambda: ["nolds", "antropy"],
    )

    def fake_require(import_name: str):
        imported.append(import_name)
        return {
            "nolds": fake_nolds,
            "antropy": fake_antropy,
        }[import_name]

    monkeypatch.setattr(external_tools, "require_external", fake_require)

    result = compute_complexity_measures(
        np.linspace(0.0, 1.0, 16),
        backend="auto",
        sample_rate=20.0,
        measures=["permutation_entropy", "lyapunov_rosenstein"],
    )

    assert result == {
        "lyapunov_rosenstein": 1.25,
        "permutation_entropy": 0.75,
    }
    assert imported == ["nolds", "antropy"]


def test_complexity_auto_reports_missing_measure_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        external_tools,
        "available_complexity_backends",
        lambda: ["nolds"],
    )

    with pytest.raises(ImportError, match="permutation_entropy.*antropy"):
        compute_complexity_measures(
            np.linspace(0.0, 1.0, 16),
            backend="auto",
            measures=["permutation_entropy"],
        )
