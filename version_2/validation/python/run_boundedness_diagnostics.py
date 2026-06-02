#!/usr/bin/env python3
"""Generate standardized F5.1 finite-time boundedness outputs."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for item in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from f5_diagnostics_common import CASE_IDS, DIAGNOSTICS_ROOT, relative, write_case_readme, write_csv, write_json  # noqa: E402
from f5_trajectory_provider import load_or_generate_f5_trajectory  # noqa: E402
from hidden_attractors.analysis.boundedness import compute_boundedness_metrics  # noqa: E402


OUTPUT_ROOT = DIAGNOSTICS_ROOT / "boundedness"
GLOBAL_SUMMARY = OUTPUT_ROOT / "boundedness_diagnostics_summary.json"


def _case_status(statuses: list[str]) -> str:
    for status in ("nonfinite_trajectory", "unbounded_candidate", "boundedness_inconclusive", "insufficient_post_transient_data"):
        if status in statuses:
            return status
    return "bounded_candidate"


def run(*, fast: bool = False) -> dict[str, Any]:
    case_summaries = []
    for case_id in CASE_IDS:
        case_dir = OUTPUT_ROOT / case_id
        trajectories = load_or_generate_f5_trajectory(case_id)
        trajectory_summaries = []
        for item in trajectories:
            result = compute_boundedness_metrics(
                item.times,
                item.states,
                burn_time=float(item.metadata["t_burn"]),
                divergence_radius=1.0e6,
            )
            timeseries = result.pop("norm_timeseries")
            summary_path = case_dir / f"{item.trajectory_id}_boundedness_summary.json"
            series_path = case_dir / f"{item.trajectory_id}_boundedness_timeseries.csv"
            payload = {
                "case_id": case_id,
                "trajectory_id": item.trajectory_id,
                "trajectory_metadata": item.metadata,
                **result,
            }
            write_json(summary_path, payload)
            write_csv(series_path, timeseries)
            trajectory_summaries.append(payload)
        status = _case_status([item["boundedness_status"] for item in trajectory_summaries])
        case_summary = {
            "case_id": case_id,
            "status": status,
            "trajectory_count": len(trajectory_summaries),
            "trajectories": trajectory_summaries,
            "boundedness_proves_chaos": False,
            "chaos_certified_by_boundedness": False,
            "hiddenness_certified_by_boundedness": False,
        }
        write_json(case_dir / "boundedness_case_summary.json", case_summary)
        write_case_readme(case_dir / "README.md", case_id, "boundedness")
        case_summaries.append(case_summary)
    payload = {
        "stage": "F5.1_boundedness",
        "status": "completed_structured_outputs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fast": bool(fast),
        "cases_total": len(case_summaries),
        "standardized_outputs": True,
        "boundedness_proves_chaos": False,
        "chaos_certified_by_boundedness": False,
        "hiddenness_certified_by_boundedness": False,
        "case_summaries": case_summaries,
    }
    write_json(GLOBAL_SUMMARY, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    payload = run(fast=args.fast)
    print(f"Recorded {payload['cases_total']} boundedness cases; status={payload['status']}")


if __name__ == "__main__":
    main()
