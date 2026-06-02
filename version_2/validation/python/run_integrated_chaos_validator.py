#!/usr/bin/env python3
"""Assemble conservative F6 integrated chaos-candidate diagnostics."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for item in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from hidden_attractors.analysis.integrated_chaos_validator import (  # noqa: E402
    CASE_Q,
    integrate_case_evidence,
    method_registry_rows,
    normalize_lyapunov_case_evidence,
)
from f5_diagnostics_common import CASE_IDS, write_csv, write_json  # noqa: E402


CHAOS_VALIDATION_ROOT = PROJECT_ROOT / "validation" / "chaos_validation"
DYNAMICS_ROOT = CHAOS_VALIDATION_ROOT / "dynamics_diagnostics"
OUTPUT_ROOT = CHAOS_VALIDATION_ROOT / "integrated_chaos_validator"
F5_SUMMARY = DYNAMICS_ROOT / "f5_diagnostics_summary.json"
BOUNDEDNESS_SUMMARY = DYNAMICS_ROOT / "boundedness" / "boundedness_diagnostics_summary.json"
ZERO_ONE_SUMMARY = DYNAMICS_ROOT / "zero_one" / "zero_one_diagnostics_summary.json"
PSD_FFT_SUMMARY = DYNAMICS_ROOT / "psd_fft" / "psd_fft_diagnostics_summary.json"
POINCARE_SUMMARY = DYNAMICS_ROOT / "poincare" / "poincare_diagnostics_summary.json"
F4_SUMMARY = (
    CHAOS_VALIDATION_ROOT
    / "lyapunov_methods"
    / "F4_internal_validation"
    / "f4_internal_validation_summary.json"
)
F4_INTEGER_RESULTS = (
    CHAOS_VALIDATION_ROOT
    / "lyapunov_methods"
    / "F4_internal_validation"
    / "F4_2_integer_chua_q1"
    / "integer_chua_q1_results.csv"
)
LYAPUNOV_VALIDATION_SUMMARIES = {
    "fractional_variational_abm_qr": (
        CHAOS_VALIDATION_ROOT
        / "lyapunov_methods"
        / "fractional_variational_abm_qr_published"
        / "validation_summary.json"
    ),
    "fractional_variational_dk2018_block_restart_abm_gs": (
        CHAOS_VALIDATION_ROOT
        / "lyapunov_methods"
        / "fractional_variational_dk2018_block_restart_abm_gs_published"
        / "validation_summary.json"
    ),
    "fractional_cloned_dynamics_abm_gs_published": (
        CHAOS_VALIDATION_ROOT
        / "lyapunov_methods"
        / "fractional_cloned_dynamics_abm_gs_published"
        / "validation_summary.json"
    ),
}
SUMMARY_PATH = OUTPUT_ROOT / "integrated_chaos_summary.json"
BY_CASE_PATH = OUTPUT_ROOT / "integrated_chaos_by_case.csv"
MATRIX_PATH = OUTPUT_ROOT / "integrated_chaos_evidence_matrix.csv"
RULES_PATH = OUTPUT_ROOT / "integrated_chaos_rules.json"


RULES = {
    "schema_version": "1.0",
    "stage": "F6_integrated_chaos_validator",
    "allowed_case_statuses": [
        "chaotic_candidate_numerically_supported",
        "regular_candidate_numerically_supported",
        "mixed_diagnostics_inconclusive",
        "insufficient_lyapunov_support",
        "insufficient_f5_support",
        "method_validation_pending",
        "numerical_failure",
        "not_evaluated",
    ],
    "chaotic_candidate_rule": {
        "requires_boundedness": "bounded_candidate",
        "minimum_supporting_indicators": 2,
        "indicators": [
            "validated_applicable_positive_lambda_max",
            "zero_one_chaotic_candidate",
            "broadband_spectrum",
            "cloud_like_or_dispersed_cloud_like_poincare",
        ],
        "blockers": ["numerical_failure", "nonfinite_trajectory", "contradictory_f5_evidence"],
    },
    "regular_candidate_rule": {
        "requires_boundedness": "bounded_candidate",
        "minimum_supporting_indicators": 2,
        "indicators": [
            "validated_applicable_nonpositive_lambda_max",
            "zero_one_regular_candidate",
            "dominant_periodic_peak_or_quasiperiodic_candidate",
            "point_finite_set_or_curve_like_poincare",
        ],
    },
    "conservative_boundaries": {
        "integrated_validator_proves_chaos": False,
        "hiddenness_evaluated": False,
        "hiddenness_certified": False,
        "dk2018_block_restart_does_not_validate_full_history_qr": True,
        "fischer_cloned_dynamics_does_not_validate_full_history_qr": True,
        "f5_conflict_or_spectral_inconclusive_remains_inconclusive": True,
    },
}


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if required:
            raise
        return None


def _case_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["case_id"]: item for item in summary["case_summaries"]}


def _f5_case_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["case_id"]: item for item in summary["combined_interpretation"]["per_case"]}


def load_f4_integer_rows(path: Path = F4_INTEGER_RESULTS) -> list[dict[str, Any]]:
    """Load case-specific integer Chua spectra if the F4 artifact exists."""

    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    **row,
                    "computed_exponents": json.loads(row["computed_exponents"]),
                    "sign_pattern": json.loads(row["sign_pattern"]),
                }
            )
    return rows


def _f4_status(summary: dict[str, Any] | None) -> str:
    return (
        str(summary["status"])
        if summary is not None
        else "f4_internal_validation_missing_or_pending"
    )


def _zero_one_global(case: dict[str, Any]) -> float | None:
    values = [item.get("K_global_median") for item in case.get("trajectories", [])]
    finite = [float(value) for value in values if value is not None]
    return max(finite) if finite else None


def _peak_dominance_max(case: dict[str, Any]) -> float | None:
    values = [
        coordinate.get("peak_dominance")
        for trajectory in case.get("trajectories", [])
        for coordinate in trajectory.get("coordinate_results", {}).values()
    ]
    finite = [float(value) for value in values if value is not None]
    return max(finite) if finite else None


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def run() -> dict[str, Any]:
    """Read F4/F5 evidence, apply F6 rules, and write structured artifacts."""

    f5 = _read_json(F5_SUMMARY)
    boundedness = _case_map(_read_json(BOUNDEDNESS_SUMMARY))
    zero_one = _case_map(_read_json(ZERO_ONE_SUMMARY))
    psd_fft = _case_map(_read_json(PSD_FFT_SUMMARY))
    poincare = _case_map(_read_json(POINCARE_SUMMARY))
    f5_cases = _f5_case_map(f5)
    f4 = _read_json(F4_SUMMARY, required=False)
    f4_status = _f4_status(f4)
    f4_integer_rows = load_f4_integer_rows()
    lyapunov_validation_summaries = {
        method_id: {
            "path": _relative(path),
            "available": (summary := _read_json(path, required=False)) is not None,
            "status": summary.get("status") if summary is not None else "missing_or_pending",
        }
        for method_id, path in LYAPUNOV_VALIDATION_SUMMARIES.items()
    }
    cases = []
    matrix_rows = []
    for case_id in CASE_IDS:
        lyapunov = normalize_lyapunov_case_evidence(
            case_id=case_id,
            q=CASE_Q[case_id],
            f4_integer_rows=f4_integer_rows,
        )
        bounded_case = boundedness.get(case_id, {})
        zero_case = zero_one.get(case_id, {})
        spectral_case = psd_fft.get(case_id, {})
        poincare_case = poincare.get(case_id, {})
        decision = integrate_case_evidence(
            case_id=case_id,
            boundedness_status=bounded_case.get("status"),
            zero_one_status=zero_case.get("status"),
            psd_fft_status=spectral_case.get("status"),
            poincare_status=poincare_case.get("geometric_interpretation"),
            lyapunov_evidence=lyapunov,
            f4_status=f4_status,
        )
        evidence = {
            "f5_combined_status": f5_cases.get(case_id, {}).get("status"),
            "boundedness_status": bounded_case.get("status"),
            "zero_one_status": zero_case.get("status"),
            "zero_one_K_global": _zero_one_global(zero_case),
            "psd_fft_status": spectral_case.get("status"),
            "peak_dominance_max": _peak_dominance_max(spectral_case),
            "poincare_status": poincare_case.get("geometric_interpretation"),
            "poincare_crossing_count": poincare_case.get("crossing_count"),
            "lyapunov_methods": lyapunov,
            "f4_status": f4_status,
        }
        cases.append({**decision, "evidence": evidence})
        available = [item["method_id"] for item in lyapunov if item["applicable"] and item["implemented"]]
        lambda_values = {
            item["method_id"]: item["lambda_max"]
            for item in lyapunov
            if item["lambda_max"] is not None
        }
        validation_status = {
            item["method_id"]: {
                "validated": item["validated"],
                "benchmark_status": item["benchmark_status"],
            }
            for item in lyapunov
            if item["applicable"]
        }
        matrix_rows.append(
            {
                "case_id": case_id,
                "boundedness_status": evidence["boundedness_status"],
                "zero_one_status": evidence["zero_one_status"],
                "zero_one_K_global": evidence["zero_one_K_global"],
                "psd_fft_status": evidence["psd_fft_status"],
                "peak_dominance_max": evidence["peak_dominance_max"],
                "poincare_status": evidence["poincare_status"],
                "poincare_crossing_count": evidence["poincare_crossing_count"],
                "lyapunov_methods_available": available,
                "lyapunov_lambda_max_values": lambda_values,
                "lyapunov_method_validation_status": validation_status,
                "f4_status": f4_status,
                "integrated_status": decision["integrated_status"],
                "decision_reason": decision["decision_reason"],
                "chaos_verified": False,
                "hidden_verified": False,
            }
        )
    counts = Counter(case["integrated_status"] for case in cases)
    payload = {
        "stage": "F6_integrated_chaos_validator",
        "status": "completed_non_certifying_integration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "f5_summary": _relative(F5_SUMMARY),
            "boundedness_summary": _relative(BOUNDEDNESS_SUMMARY),
            "zero_one_summary": _relative(ZERO_ONE_SUMMARY),
            "psd_fft_summary": _relative(PSD_FFT_SUMMARY),
            "poincare_summary": _relative(POINCARE_SUMMARY),
            "lyapunov_registry": "hidden_attractors/analysis/lyapunov_methods.py",
            "lyapunov_validation_summaries": lyapunov_validation_summaries,
            "f4_summary": _relative(F4_SUMMARY) if f4 is not None else None,
        },
        "f4_status": f4_status,
        "cases_total": len(cases),
        "cases": cases,
        "global_counts": {
            status: counts.get(status, 0)
            for status in RULES["allowed_case_statuses"]
        },
        "method_registry": method_registry_rows(),
        "certifications": {"chaos_verified": False, "hidden_verified": False},
        "invariants": {
            "integrated_validator_is_not_a_mathematical_proof": True,
            "integrated_validator_proves_chaos": False,
            "no_single_indicator_certifies_chaos": True,
            "hiddenness_not_evaluated_here": True,
        },
    }
    write_json(SUMMARY_PATH, payload)
    write_csv(BY_CASE_PATH, cases)
    write_csv(MATRIX_PATH, matrix_rows)
    write_json(RULES_PATH, RULES)
    return payload


def main() -> None:
    summary = run()
    print(f"F6 status: {summary['status']}")
    print(f"F4 status: {summary['f4_status']}")
    print("Chaos verified: false")
    print("Hiddenness verified: false")


if __name__ == "__main__":
    main()
