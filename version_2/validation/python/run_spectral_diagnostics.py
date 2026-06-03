#!/usr/bin/env python3
"""Generate standardized F5.3 FFT/PSD outputs."""

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

from validation.python.f5_diagnostics_common import CASE_IDS, DIAGNOSTICS_ROOT, aggregate_state, write_case_readme, write_csv, write_json  # noqa: E402
from validation.python.f5_trajectory_provider import load_or_generate_f5_trajectory  # noqa: E402
from hidden_attractors.analysis.spectral import compute_fft_psd, spectral_diagnostics_multicoordinate  # noqa: E402


OUTPUT_ROOT = DIAGNOSTICS_ROOT / "psd_fft"
GLOBAL_SUMMARY = OUTPUT_ROOT / "psd_fft_diagnostics_summary.json"


def _without_series(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in {"frequencies", "power"}}


def synthetic_method_validation() -> dict[str, Any]:
    times = np.arange(5000, dtype=float) * 0.01
    sine = np.sin(2.0 * np.pi * 2.0 * times)
    two_frequency = sine + 0.6 * np.sin(2.0 * np.pi * np.sqrt(2.0) * times)
    broadband = np.random.default_rng(12345).normal(size=times.size)
    sine_result = compute_fft_psd(times, sine)
    two_result = compute_fft_psd(times, two_frequency)
    broadband_result = compute_fft_psd(times, broadband)
    if sine_result["state"] != "dominant_periodic_peak":
        raise RuntimeError("pure sine spectral validation failed.")
    if two_result["state"] != "quasiperiodic_candidate":
        raise RuntimeError("two-frequency spectral validation failed.")
    if broadband_result["state"] not in {"broadband_spectrum", "spectral_inconclusive"}:
        raise RuntimeError("broadband spectral validation failed.")
    return {
        "pure_sine": "passed",
        "two_frequency_signal": "passed",
        "broadband_synthetic": "passed_or_documented",
        "noise_interpretation": "broadband_noise_is_not_deterministic_chaos",
    }


def run(*, fast: bool = False) -> dict[str, Any]:
    method_validation = synthetic_method_validation()
    case_summaries = []
    for case_id in CASE_IDS:
        case_dir = OUTPUT_ROOT / case_id
        trajectory_summaries = []
        for item in load_or_generate_f5_trajectory(case_id):
            result = spectral_diagnostics_multicoordinate(
                item.times,
                item.states,
                burn_time=float(item.metadata["t_burn"]),
            )
            rows = [
                {"coordinate": coordinate, **_without_series(values)}
                for coordinate, values in result["coordinate_results"].items()
            ]
            write_csv(case_dir / f"{item.trajectory_id}_psd_by_coordinate.csv", rows)
            compact = {
                **result,
                "coordinate_results": {
                    coordinate: _without_series(values)
                    for coordinate, values in result["coordinate_results"].items()
                },
            }
            payload = {
                "case_id": case_id,
                "trajectory_id": item.trajectory_id,
                "trajectory_metadata": item.metadata,
                **compact,
            }
            write_json(case_dir / f"{item.trajectory_id}_spectral_summary.json", payload)
            trajectory_summaries.append(payload)
        status = aggregate_state(
            [item["state_global"] for item in trajectory_summaries],
            regular="dominant_periodic_peak",
            chaotic="broadband_spectrum",
            inconclusive="spectral_inconclusive",
        )
        if status == "spectral_inconclusive" and all(
            item["state_global"] == "quasiperiodic_candidate" for item in trajectory_summaries
        ):
            status = "quasiperiodic_candidate"
        case_summary = {
            "case_id": case_id,
            "status": status,
            "trajectory_count": len(trajectory_summaries),
            "trajectories": trajectory_summaries,
            "psd_alone_does_not_certify_chaos": True,
            "psd_proves_chaos": False,
            "chaos_certified_by_psd": False,
            "hiddenness_certified_by_psd": False,
        }
        write_json(case_dir / "psd_fft_case_summary.json", case_summary)
        write_case_readme(case_dir / "README.md", case_id, "PSD-FFT")
        case_summaries.append(case_summary)
    payload = {
        "stage": "F5.3_psd_fft",
        "status": "completed_structured_outputs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fast": bool(fast),
        "cases_total": len(case_summaries),
        "standardized_outputs": True,
        "psd_proves_chaos": False,
        "chaos_certified_by_psd": False,
        "hiddenness_certified_by_psd": False,
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
    print(f"Recorded {payload['cases_total']} spectral cases; status={payload['status']}")


if __name__ == "__main__":
    main()
