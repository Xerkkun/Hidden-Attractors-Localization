"""Tests for current-run candidate promotion without historical defaults."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hidden_attractors.candidates import load_final_candidate_records
from hidden_attractors.workflows.fractional_report_run import _dominant_period_return_ratio, _post_continuation_periodicity


def test_current_selection_loader_reads_promoted_json(tmp_path: Path) -> None:
    selected = [
        {
            "candidate_id": f"current_candidate_{rank}",
            "method": "lure_classical_biased",
            "q": 0.9998,
            "A": 1.0 + rank,
            "omega": 2.0,
            "rho_H": 0.1,
            "residual_abs": 0.01,
            "seed": [rank, 0.0, 0.0],
            "robust_start": [rank, 1.0, 0.0],
        }
        for rank in range(1, 4)
    ]
    path = tmp_path / "selected_candidates.json"
    path.write_text(json.dumps({"selected_candidates": selected}), encoding="utf-8")

    records = load_final_candidate_records(path)

    assert [record.candidate_id for record in records] == [
        "current_candidate_1",
        "current_candidate_2",
        "current_candidate_3",
    ]
    assert all("20260515" not in record.source for record in records)


def test_current_selection_loader_rejects_periodic_postcheck_selection(tmp_path: Path) -> None:
    path = tmp_path / "selected_candidates.json"
    path.write_text(
        json.dumps(
            {
                "selection_status": "rejected_near_periodic_postcheck",
                "selected_candidates": [{"candidate_id": f"c{rank}"} for rank in range(3)],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="no está promovida"):
        load_final_candidate_records(path)


def test_dominant_period_return_ratio_rejects_thin_closed_trace() -> None:
    h = 0.01
    times = np.arange(0.0, 20.0 + h, h)
    omega = 2.0 * np.pi
    trajectory = np.column_stack(
        [times, np.cos(omega * times), np.sin(omega * times), 0.2 * np.cos(omega * times)]
    )

    ratio, lag = _dominant_period_return_ratio(
        trajectory, h=h, t_start=10.0, dominant_frequency=1.0
    )

    assert lag == 100
    assert ratio < 1.0e-12


def test_post_continuation_filter_rejects_periodic_multicomponent_trace() -> None:
    h = 0.01
    times = np.arange(0.0, 80.0 + h, h)
    omega = 2.0 * np.pi
    trajectory = np.column_stack(
        [times, np.cos(omega * times), np.sin(omega * times), 0.2 * np.cos(omega * times)]
    )

    result = _post_continuation_periodicity(trajectory, h=h, t_final=80.0)

    assert result["periodicity_status"] == "periodic_post_transient"
    assert result["periodic_post_transient"] is True
    assert set(result["periodic_components"].split(";")) >= {"x", "y"}
