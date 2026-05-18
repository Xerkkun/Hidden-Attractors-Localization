"""Small command-line helpers for installed users."""

from __future__ import annotations

from .candidates import load_final_candidate_records


def list_candidates() -> None:
    """Print final candidate records using the public package API."""

    for record in load_final_candidate_records():
        print(
            f"{record.candidate_id} | route={record.route} | "
            f"q={record.q:.4f} | start={record.robust_start.tolist()} | "
            f"seed={record.seed.tolist()}"
        )
