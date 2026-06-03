#!/usr/bin/env python3
"""Run and combine standardized F5.1-F5.4 dynamics diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for item in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from validation.python.f5_diagnostics_common import CASE_IDS, CERTIFICATIONS, DIAGNOSTICS_ROOT, INVARIANTS, write_json  # noqa: E402
from validation.python.run_boundedness_diagnostics import GLOBAL_SUMMARY as BOUNDEDNESS_SUMMARY, run as run_boundedness  # noqa: E402
from validation.python.run_poincare_diagnostics import GLOBAL_SUMMARY_PATH as POINCARE_SUMMARY, _final_f5_status, run as run_poincare  # noqa: E402
from validation.python.run_spectral_diagnostics import GLOBAL_SUMMARY as SPECTRAL_SUMMARY, run as run_spectral  # noqa: E402
from validation.python.run_zero_one_diagnostics import GLOBAL_SUMMARY as ZERO_ONE_SUMMARY, run as run_zero_one  # noqa: E402


F5_SUMMARY = DIAGNOSTICS_ROOT / "f5_diagnostics_summary.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["case_id"]: item for item in summary["case_summaries"]}


def combine_f5_diagnostics(
    boundedness: dict[str, Any],
    zero_one: dict[str, Any],
    psd_fft: dict[str, Any],
    poincare: dict[str, Any],
) -> dict[str, Any]:
    """Combine indicators into conservative candidate labels, never proofs."""

    bounded_cases = _case_map(boundedness)
    zero_cases = _case_map(zero_one)
    spectral_cases = _case_map(psd_fft)
    poincare_cases = _case_map(poincare)
    per_case = []
    for case_id in CASE_IDS:
        if case_id not in bounded_cases or case_id not in zero_cases or case_id not in spectral_cases or case_id not in poincare_cases:
            label = "insufficient_diagnostics"
        else:
            bounded = bounded_cases[case_id]["status"]
            zero = zero_cases[case_id]["status"]
            spectral = spectral_cases[case_id]["status"]
            section = poincare_cases[case_id]["geometric_interpretation"]
            if bounded in {"nonfinite_trajectory", "unbounded_candidate"}:
                label = "numerical_failure"
            elif (
                bounded == "bounded_candidate"
                and zero == "zero_one_chaotic_candidate"
                and spectral == "broadband_spectrum"
                and section in {"cloud_like", "dispersed_cloud_like"}
            ):
                label = "chaotic_candidate_numerically_supported"
            elif (
                bounded == "bounded_candidate"
                and zero == "zero_one_regular_candidate"
                and spectral in {"dominant_periodic_peak", "quasiperiodic_candidate"}
                and section in {"finite_set_like", "curve_like", "point_like_or_fixed_return"}
            ):
                label = "regular_candidate_numerically_supported"
            else:
                label = "mixed_diagnostics_inconclusive"
        per_case.append(
            {
                "case_id": case_id,
                "status": label,
                "boundedness": bounded_cases.get(case_id, {}).get("status"),
                "zero_one": zero_cases.get(case_id, {}).get("status"),
                "psd_fft": spectral_cases.get(case_id, {}).get("status"),
                "poincare": poincare_cases.get(case_id, {}).get("geometric_interpretation"),
                "chaos_verified": False,
                "hidden_verified": False,
            }
        )
    return {
        "status": "diagnostics_only_not_certification",
        "per_case": per_case,
        "chaos_verified": False,
        "hidden_verified": False,
    }


def _subphase(summary: dict[str, Any], path: str) -> dict[str, Any]:
    return {
        "status": summary["status"],
        "standardized_outputs": bool(summary["standardized_outputs"]),
        "path": path,
    }


def run(
    *,
    boundedness: bool,
    zero_one: bool,
    psd_fft: bool,
    poincare: bool,
    use_existing_poincare: bool,
    fast: bool,
) -> dict[str, Any]:
    if boundedness:
        run_boundedness(fast=fast)
    if zero_one:
        run_zero_one(fast=fast)
    if psd_fft:
        run_spectral(fast=fast)
    if poincare and not use_existing_poincare:
        run_poincare()
    required = [BOUNDEDNESS_SUMMARY, ZERO_ONE_SUMMARY, SPECTRAL_SUMMARY, POINCARE_SUMMARY]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing F5 summaries: {missing}")
    bounded_summary = _read(BOUNDEDNESS_SUMMARY)
    zero_summary = _read(ZERO_ONE_SUMMARY)
    spectral_summary = _read(SPECTRAL_SUMMARY)
    poincare_summary = _read(POINCARE_SUMMARY)
    subphases = {
        "boundedness": _subphase(bounded_summary, "boundedness/boundedness_diagnostics_summary.json"),
        "zero_one": _subphase(zero_summary, "zero_one/zero_one_diagnostics_summary.json"),
        "psd_fft": _subphase(spectral_summary, "psd_fft/psd_fft_diagnostics_summary.json"),
        "poincare": _subphase(poincare_summary, "poincare/poincare_diagnostics_summary.json"),
    }
    payload = {
        "stage": "F5_dynamics_diagnostics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **subphases,
        "poincare_method_validation": poincare_summary["poincare_method_validation"],
        "poincare_application_to_published_cases": poincare_summary["poincare_application_to_published_cases"],
        "final_f5_status": _final_f5_status(subphases),
        "certifications": CERTIFICATIONS,
        "invariants": INVARIANTS,
        "combined_interpretation": combine_f5_diagnostics(
            bounded_summary,
            zero_summary,
            spectral_summary,
            poincare_summary,
        ),
    }
    write_json(F5_SUMMARY, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--boundedness", action="store_true")
    parser.add_argument("--zero-one", action="store_true")
    parser.add_argument("--psd-fft", action="store_true")
    parser.add_argument("--poincare", action="store_true")
    parser.add_argument("--use-existing-poincare", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    selected = args.all or not any((args.boundedness, args.zero_one, args.psd_fft, args.poincare))
    payload = run(
        boundedness=selected or args.boundedness,
        zero_one=selected or args.zero_one,
        psd_fft=selected or args.psd_fft,
        poincare=selected or args.poincare,
        use_existing_poincare=args.use_existing_poincare,
        fast=args.fast,
    )
    print(f"F5 status: {payload['final_f5_status']}")
    print("Chaos verified: false")
    print("Hiddenness verified: false")


if __name__ == "__main__":
    main()
