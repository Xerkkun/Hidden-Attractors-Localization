"""Focused contracts for the high-level Lyapunov workflow."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hidden_attractors.workflows import lyapunov as lyapunov_workflow


def test_integer_workflow_resolves_method_interval_and_memory(
    tmp_path,
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}
    saved_config: dict[str, object] = {}

    result = SimpleNamespace(
        exponents=np.array([0.2, 0.0, -1.0]),
        times=np.array([0.1]),
        convergence=np.array([[0.2, 0.0, -1.0]]),
        status="ok",
        finite_time_local=True,
    )

    def fake_compute(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(result=result, warnings=())

    monkeypatch.setattr(lyapunov_workflow, "compute_lyapunov_spectrum", fake_compute)
    monkeypatch.setattr(
        lyapunov_workflow,
        "plot_lyapunov_convergence_styled",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        lyapunov_workflow,
        "collect_run_metadata",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        lyapunov_workflow,
        "write_run_metadata",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        lyapunov_workflow,
        "save_effective_config",
        lambda config, output_dir: saved_config.update(config),
    )

    summary = lyapunov_workflow.run_lyapunov_workflow(
        {
            "system_id": "chua-nonsmooth",
            # Only the exact endpoint is an integer-order model.  Values
            # arbitrarily close to one still have fractional memory semantics.
            "q": 1.0,
            "integrator": "efork3",
            "memory_mode": "full",
            "memory_window_steps": 400,
            "output_dir": str(tmp_path),
            "lyapunov": {
                "h": 0.01,
                "t_final": 0.1,
                "t_burn": 0.0,
                "reorthonormalize_every": 7,
                "initial_condition": [0.1, 0.0, 0.0],
            },
        }
    )

    assert observed["q"] == 1.0
    assert observed["method"] == "integer_qr_benettin"
    assert observed["reorthonormalize_every"] == 7
    assert observed["memory_mode"] == "not_applicable"
    assert observed["memory_window"] is None
    assert summary["method"] == "integer_qr_benettin"
    assert summary["memory_mode"] == "not_applicable"
    assert summary["memory_window"] is None
    assert saved_config["q"] == 1.0
    assert saved_config["memory_mode"] == "not_applicable"
    assert saved_config["memory_window_steps"] is None
    assert saved_config["lyapunov"]["reorthonormalize_every"] == 7


def test_integer_workflow_accepts_legacy_interval_alias(
    tmp_path,
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}
    result = SimpleNamespace(
        exponents=np.array([0.0]),
        times=np.array([0.1]),
        convergence=np.array([[0.0]]),
        status="ok",
        finite_time_local=True,
    )

    monkeypatch.setattr(
        lyapunov_workflow,
        "compute_lyapunov_spectrum",
        lambda **kwargs: (
            observed.update(kwargs)
            or SimpleNamespace(result=result, warnings=())
        ),
    )
    monkeypatch.setattr(
        lyapunov_workflow,
        "plot_lyapunov_convergence_styled",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        lyapunov_workflow,
        "collect_run_metadata",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        lyapunov_workflow,
        "write_run_metadata",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        lyapunov_workflow,
        "save_effective_config",
        lambda *args, **kwargs: None,
    )

    lyapunov_workflow.run_lyapunov_workflow(
        {
            "system_id": "chua-nonsmooth",
            "q": 1.0,
            "output_dir": str(tmp_path),
            "lyapunov": {
                "h": 0.01,
                "t_final": 0.1,
                "t_burn": 0.0,
                "orthonormalization_interval": 3,
                "initial_condition": [0.1, 0.0, 0.0],
            },
        }
    )

    assert observed["reorthonormalize_every"] == 3
