"""F7 reports applicability, pending validation, and diagnostic conflicts."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "validation" / "chaos_validation" / "method_comparison"
SUMMARY = OUTPUT / "method_comparison_summary.json"
LYAPUNOV_CSV = OUTPUT / "lyapunov_method_comparison.csv"


def _rows() -> list[dict[str, str]]:
    with LYAPUNOV_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_f7_runner_writes_method_comparison_outputs() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "validation" / "python" / "run_method_comparison.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert SUMMARY.is_file()
    assert LYAPUNOV_CSV.is_file()
    assert (OUTPUT / "diagnostic_method_comparison.csv").is_file()
    assert (OUTPUT / "method_consensus_matrix.csv").is_file()
    assert (OUTPUT / "method_discrepancy_report.md").is_file()


def test_integer_qr_is_not_applied_to_fractional_cases() -> None:
    selected = [row for row in _rows() if row["method_id"] == "integer_qr_benettin"]
    assert len(selected) == 3
    assert next(row for row in selected if row["q"] == "1.0")["applicable"] == "True"
    assert all(row["applicable"] == "False" for row in selected if row["q"] != "1.0")


def test_experimental_qr_remains_unvalidated() -> None:
    selected = [row for row in _rows() if row["method_id"] == "fractional_cloned_dynamics_abm_qr"]
    assert selected
    assert all(row["validated"] == "False" for row in selected)
    assert all(row["benchmark_status"] == "internal_variant_pending" for row in selected)


def test_fischer_published_gs_preserves_discrepancy_status() -> None:
    selected = [
        row for row in _rows() if row["method_id"] == "fractional_cloned_dynamics_abm_gs_published"
    ]
    assert selected
    assert all(row["validated"] == "False" for row in selected)
    assert all(row["benchmark_status"] == "published_benchmarks_pending_discrepancy" for row in selected)


def test_f7_never_certifies_chaos_or_hiddenness() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["status"] == "completed_method_evidence_comparison"
    assert summary["chaos_evidence_level"] == "chaos_evidence_inconclusive"
    assert summary["hiddenness_evidence_level"] == "not_evaluated_by_this_stage"
    assert all(case["chaos_evidence_level"] == "chaos_evidence_inconclusive" for case in summary["per_case"])
    assert all(case["hiddenness_evidence_level"] == "not_evaluated_by_this_stage" for case in summary["per_case"])
    assert summary["lyapunov_comparison"]["dk2018_block_restart_does_not_validate_full_history_qr"] is True
    assert summary["lyapunov_comparison"]["fischer_gs_does_not_validate_full_history_qr"] is True
