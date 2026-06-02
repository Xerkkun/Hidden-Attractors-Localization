#!/usr/bin/env python3
"""Generate standardized F5.2 0-1 chaos-test outputs."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for item in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from f5_diagnostics_common import CASE_IDS, DIAGNOSTICS_ROOT, aggregate_state, write_case_readme, write_csv, write_json  # noqa: E402
from f5_trajectory_provider import load_or_generate_f5_trajectory  # noqa: E402
from hidden_attractors.analysis.zero_one import zero_one_multicoordinate, zero_one_test  # noqa: E402


OUTPUT_ROOT = DIAGNOSTICS_ROOT / "zero_one"
GLOBAL_SUMMARY = OUTPUT_ROOT / "zero_one_diagnostics_summary.json"


def synthetic_method_validation(*, fast: bool) -> dict[str, Any]:
    count = 2500 if fast else 5000
    index = np.arange(count, dtype=float)
    sine = np.sin(2.0 * np.pi * 0.037 * index)
    logistic = np.empty(count, dtype=float)
    logistic[0] = 0.123
    for offset in range(count - 1):
        logistic[offset + 1] = 4.0 * logistic[offset] * (1.0 - logistic[offset])
    noise = np.random.default_rng(12345).normal(size=count)
    kwargs = {"n_c": 40 if fast else 100, "max_samples": 2000 if fast else 3000}
    sine_result = zero_one_test(sine, **kwargs)
    logistic_result = zero_one_test(logistic, **kwargs)
    noise_result = zero_one_test(noise, **kwargs)
    if not sine_result["K"] < 0.2 or not logistic_result["K"] > 0.8:
        raise RuntimeError("synthetic 0-1 validation failed.")
    return {
        "regular_sine": "passed",
        "regular_sine_K": sine_result["K"],
        "logistic_map": "passed",
        "logistic_map_K": logistic_result["K"],
        "noise_limitation_documented": True,
        "noise_K": noise_result["K"],
        "noise_interpretation": "stochastic_signal_not_deterministic_chaos",
    }


def run(*, fast: bool = False) -> dict[str, Any]:
    method_validation = synthetic_method_validation(fast=fast)
    case_summaries = []
    for case_id in CASE_IDS:
        case_dir = OUTPUT_ROOT / case_id
        trajectory_summaries = []
        for item in load_or_generate_f5_trajectory(case_id):
            result = zero_one_multicoordinate(
                item.times,
                item.states,
                burn_time=float(item.metadata["t_burn"]),
                n_c=40 if fast else 100,
                max_samples=1500 if fast else 3000,
            )
            rows = [
                {"coordinate": coordinate, **values}
                for coordinate, values in result["coordinate_results"].items()
            ]
            write_csv(case_dir / f"{item.trajectory_id}_zero_one_by_coordinate.csv", rows)
            payload = {
                "case_id": case_id,
                "trajectory_id": item.trajectory_id,
                "trajectory_metadata": item.metadata,
                **result,
            }
            write_json(case_dir / f"{item.trajectory_id}_zero_one_summary.json", payload)
            trajectory_summaries.append(payload)
        status = aggregate_state(
            [item["state_global"] for item in trajectory_summaries],
            regular="zero_one_regular_candidate",
            chaotic="zero_one_chaotic_candidate",
            inconclusive="zero_one_inconclusive",
        )
        case_summary = {
            "case_id": case_id,
            "status": status,
            "trajectory_count": len(trajectory_summaries),
            "trajectories": trajectory_summaries,
            "zero_one_alone_does_not_certify_chaos": True,
            "zero_one_proves_chaos": False,
            "chaos_certified_by_zero_one": False,
            "hiddenness_certified_by_zero_one": False,
        }
        write_json(case_dir / "zero_one_case_summary.json", case_summary)
        write_case_readme(case_dir / "README.md", case_id, "zero-one")
        case_summaries.append(case_summary)
    payload = {
        "stage": "F5.2_zero_one",
        "status": "completed_structured_outputs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fast": bool(fast),
        "cases_total": len(case_summaries),
        "standardized_outputs": True,
        "zero_one_proves_chaos": False,
        "chaos_certified_by_zero_one": False,
        "hiddenness_certified_by_zero_one": False,
        "method_validation": method_validation,
        "case_summaries": case_summaries,
    }
    write_json(GLOBAL_SUMMARY, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    payload = run(fast=args.fast)
    print(f"Recorded {payload['cases_total']} zero-one cases; status={payload['status']}")


if __name__ == "__main__":
    main()
