#!/usr/bin/env python3
"""Assemble conservative F7 comparisons across Lyapunov and F5 methods."""

from __future__ import annotations

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
    normalize_lyapunov_case_evidence,
)
from hidden_attractors.analysis.method_comparison import (  # noqa: E402
    classify_method_row,
    compare_f5_diagnostics,
    compare_lyapunov_methods,
)
from f5_diagnostics_common import CASE_IDS, write_csv, write_json  # noqa: E402
from run_integrated_chaos_validator import (  # noqa: E402
    BOUNDEDNESS_SUMMARY,
    CHAOS_VALIDATION_ROOT,
    F4_SUMMARY,
    POINCARE_SUMMARY,
    PSD_FFT_SUMMARY,
    SUMMARY_PATH as F6_SUMMARY,
    ZERO_ONE_SUMMARY,
    load_f4_integer_rows,
)


OUTPUT_ROOT = CHAOS_VALIDATION_ROOT / "method_comparison"
SUMMARY_PATH = OUTPUT_ROOT / "method_comparison_summary.json"
LYAPUNOV_CSV = OUTPUT_ROOT / "lyapunov_method_comparison.csv"
DIAGNOSTIC_CSV = OUTPUT_ROOT / "diagnostic_method_comparison.csv"
CONSENSUS_CSV = OUTPUT_ROOT / "method_consensus_matrix.csv"
REPORT_PATH = OUTPUT_ROOT / "method_discrepancy_report.md"


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if required:
            raise
        return None


def _case_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["case_id"]: item for item in summary["case_summaries"]}


def _write_report(per_case: list[dict[str, Any]], lyapunov_rows: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# F7 Method Discrepancy Report",
        "",
        "## Executive Summary",
        "",
        "F7 compares finite-time numerical evidence. It does not certify chaos or hiddenness.",
        "The current Chua cases retain conservative comparison labels when diagnostics conflict",
        "or case-specific Lyapunov spectra are unavailable.",
        "",
        "## Cases",
        "",
        "| Case | Lyapunov comparison | F5 comparison | F6 status |",
        "|---|---|---|---|",
    ]
    for item in per_case:
        lines.append(
            f"| `{item['case_id']}` | `{item['lyapunov_consensus']}` | "
            f"`{item['f5_consensus']}` | `{item['integrated_f6_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Lyapunov Methods",
            "",
            "| Method | Applicable rows | Validated | Benchmark status |",
            "|---|---:|---|---|",
        ]
    )
    method_ids = list(dict.fromkeys(row["method_id"] for row in lyapunov_rows))
    for method_id in method_ids:
        selected = [row for row in lyapunov_rows if row["method_id"] == method_id]
        applicable = sum(bool(row["applicable"]) for row in selected)
        lines.append(
            f"| `{method_id}` | `{applicable}` | `{selected[0]['validated']}` | "
            f"`{selected[0]['benchmark_status']}` |"
        )
    lines.extend(
        [
            "",
            "## F5 Diagnostics",
            "",
            "| Case | Boundedness | Zero-one | PSD/FFT | Poincare | Comparison |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in diagnostic_rows:
        lines.append(
            f"| `{row['case_id']}` | `{row['boundedness']}` | `{row['zero_one']}` | "
            f"`{row['psd_fft']}` | `{row['poincare']}` | `{row['f5_consensus_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Conflicts Detected",
            "",
        ]
    )
    for item in per_case:
        lines.append(f"- `{item['case_id']}`: {'; '.join(item['method_conflicts']) or 'no explicit conflict'}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- F2 full-history QR is not validated against published benchmarks.",
            "- F3 published GS retains documented Fischer 2020 discrepancies.",
            "- F3 QR is an experimental internal variant.",
            "- Zero-one and PSD/FFT are diagnostics, not proofs.",
            "- Poincare sections for Caputo trajectories are geometric crossings, not exact return maps.",
            "- DK2018 block-restart evidence does not validate full-history Caputo QR.",
            "",
            "## Conservative Conclusion",
            "",
            "F7 reports support, disagreement, and missing evidence. It certifies neither chaos nor hiddenness.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    """Write F7 method comparison artifacts."""

    boundedness = _case_map(_read_json(BOUNDEDNESS_SUMMARY))
    zero_one = _case_map(_read_json(ZERO_ONE_SUMMARY))
    psd_fft = _case_map(_read_json(PSD_FFT_SUMMARY))
    poincare = _case_map(_read_json(POINCARE_SUMMARY))
    f6 = _read_json(F6_SUMMARY, required=False)
    f6_cases = {item["case_id"]: item for item in f6.get("cases", [])} if f6 else {}
    f4 = _read_json(F4_SUMMARY, required=False)
    f4_status = (
        str(f4["status"])
        if f4 is not None
        else "f4_internal_validation_missing_or_pending"
    )
    integer_rows = load_f4_integer_rows()
    lyapunov_rows = []
    diagnostic_rows = []
    consensus_rows = []
    per_case = []
    for case_id in CASE_IDS:
        methods = normalize_lyapunov_case_evidence(
            case_id=case_id,
            q=CASE_Q[case_id],
            f4_integer_rows=integer_rows,
        )
        for method in methods:
            lyapunov_rows.append(
                {
                    "case_id": case_id,
                    "q": CASE_Q[case_id],
                    **method,
                    "status": classify_method_row(method),
                }
            )
        lyapunov_consensus, lyapunov_notes = compare_lyapunov_methods(methods)
        f5_consensus, conflict_note = compare_f5_diagnostics(
            boundedness=boundedness.get(case_id, {}).get("status"),
            zero_one=zero_one.get(case_id, {}).get("status"),
            psd_fft=psd_fft.get(case_id, {}).get("status"),
            poincare=poincare.get(case_id, {}).get("geometric_interpretation"),
        )
        diagnostic_row = {
            "case_id": case_id,
            "boundedness": boundedness.get(case_id, {}).get("status"),
            "zero_one": zero_one.get(case_id, {}).get("status"),
            "psd_fft": psd_fft.get(case_id, {}).get("status"),
            "poincare": poincare.get(case_id, {}).get("geometric_interpretation"),
            "f5_consensus_status": f5_consensus,
            "conflict_notes": conflict_note,
        }
        diagnostic_rows.append(diagnostic_row)
        integrated_status = f6_cases.get(case_id, {}).get("integrated_status", "not_evaluated")
        conflicts = list(lyapunov_notes)
        if f5_consensus in {"f5_diagnostics_conflict", "methods_mixed_inconclusive"}:
            conflicts.append(conflict_note)
        limitations = {
            "finite_time_local_indicators_only",
            "hiddenness_not_evaluated",
        }
        if CASE_Q[case_id] < 1.0:
            limitations.update(
                {
                    "fractional_method_validation_pending",
                    "case_specific_fractional_lyapunov_spectrum_unavailable",
                }
            )
        per_case.append(
            {
                "case_id": case_id,
                "lyapunov_consensus": lyapunov_consensus,
                "f5_consensus": f5_consensus,
                "integrated_f6_status": integrated_status,
                "method_conflicts": conflicts,
                "method_limitations": sorted(limitations),
                "chaos_verified": False,
                "hidden_verified": False,
            }
        )
        consensus_rows.append(
            {
                "case_id": case_id,
                "lyapunov_consensus": lyapunov_consensus,
                "f5_consensus": f5_consensus,
                "integrated_f6_status": integrated_status,
                "f4_status": f4_status,
                "chaos_verified": False,
                "hidden_verified": False,
            }
        )
    payload = {
        "stage": "F7_method_comparison",
        "status": "completed_non_certifying_comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases_total": len(per_case),
        "f4_status": f4_status,
        "lyapunov_comparison": {
            "near_zero_threshold": 0.02,
            "rows_total": len(lyapunov_rows),
            "status_counts": dict(Counter(row["status"] for row in lyapunov_rows)),
            "dk2018_block_restart_does_not_validate_full_history_qr": True,
            "fischer_gs_does_not_validate_full_history_qr": True,
        },
        "diagnostic_comparison": {
            "rows_total": len(diagnostic_rows),
            "status_counts": dict(Counter(row["f5_consensus_status"] for row in diagnostic_rows)),
            "boundedness_is_not_a_chaos_indicator": True,
        },
        "per_case": per_case,
        "certifications": {"chaos_verified": False, "hidden_verified": False},
        "invariants": {
            "method_comparison_is_not_certification": True,
            "method_comparison_proves_chaos": False,
            "hiddenness_not_evaluated_here": True,
        },
    }
    write_json(SUMMARY_PATH, payload)
    write_csv(LYAPUNOV_CSV, lyapunov_rows)
    write_csv(DIAGNOSTIC_CSV, diagnostic_rows)
    write_csv(CONSENSUS_CSV, consensus_rows)
    _write_report(per_case, lyapunov_rows, diagnostic_rows)
    return payload


def main() -> None:
    summary = run()
    print(f"F7 status: {summary['status']}")
    print("Chaos verified: false")
    print("Hiddenness verified: false")


if __name__ == "__main__":
    main()
