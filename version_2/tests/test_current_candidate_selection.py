"""Tests for loading an explicitly selected candidate set."""

from __future__ import annotations

import json
from pathlib import Path

from hidden_attractors.candidates import load_final_candidate_records


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


def test_explicit_loader_does_not_interpret_validation_status(tmp_path: Path) -> None:
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

    records = load_final_candidate_records(path)

    assert [record.candidate_id for record in records] == ["c0", "c1", "c2"]
