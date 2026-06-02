"""Contract tests for the conservative F4 Lyapunov internal-validation layer."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PYTHON = PROJECT_ROOT / "validation" / "python"
if str(VALIDATION_PYTHON) not in sys.path:
    sys.path.insert(0, str(VALIDATION_PYTHON))

from f4_internal_lyapunov_validation import F4_ROOT, GLOBAL_SUMMARY, STAGE_DIRS  # noqa: E402


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_f4_global_summary_closes_without_validation_promotion() -> None:
    summary = _load_json(GLOBAL_SUMMARY)
    assert summary["status"] == "f4_complete_with_documented_discrepancies"
    assert summary["fast_smoke_only"] is True
    assert summary["uses_existing_published_long_evidence"] is True
    assert summary["missing_stages"] == []
    assert summary["closure_checks"]["validation_state_promoted"] is False
    assert len(summary["method_controls"]) == 5


def test_f4_expected_directory_layout_exists() -> None:
    assert F4_ROOT.is_dir()
    assert (F4_ROOT / "summaries").is_dir()
    for stage_dir in STAGE_DIRS.values():
        assert stage_dir.is_dir()
        assert (stage_dir / "summary.json").is_file()
        assert (stage_dir / "README.md").is_file()


def test_f4_1_integer_linear_controls_match_exact_diagonals() -> None:
    rows = _load_csv(STAGE_DIRS["F4_1"] / "integer_linear_results.csv")
    integer_rows = [row for row in rows if row["method_id"] == "integer_qr_benettin"]
    assert len(integer_rows) == 2
    for row in integer_rows:
        computed = np.asarray(json.loads(row["computed_exponents"]), dtype=float)
        expected = np.asarray(json.loads(row["expected_exponents"]), dtype=float)
        assert row["status"] == "ok"
        assert row["controlled_check_passed"] == "True"
        np.testing.assert_allclose(computed, expected, atol=0.03)


def test_f4_1_q1_cloned_controls_include_gs_and_qr() -> None:
    rows = _load_csv(STAGE_DIRS["F4_1"] / "integer_linear_results.csv")
    cloned = {row["orthonormalization"]: row for row in rows if row["orthonormalization"]}
    assert set(cloned) == {"gs_modified", "qr"}
    assert all(row["status"] == "ok" for row in cloned.values())


def test_f4_2_integer_chua_uses_reference_seed_without_invented_spectrum() -> None:
    summary = _load_json(STAGE_DIRS["F4_2"] / "summary.json")
    assert summary["reference_seed"]["omega0"] == 2.039186939959001
    assert summary["reference_seed"]["k"] == 0.209867354515084
    assert summary["reference_seed"]["x0"] == [
        5.856145086257356,
        0.369331578246782,
        -8.36653616833188,
    ]
    assert summary["published_spectrum_claimed"] is False
    assert summary["checks"]["all_spectra_finite"] is True
    assert summary["checks"]["all_trajectories_bounded"] is True


def test_f4_3_keeps_dk2018_rf_discrepancy_and_contract_separation() -> None:
    summary = _load_json(STAGE_DIRS["F4_3"] / "summary.json")
    assert summary["published_lane"]["source_status"] == "published_benchmarks_pending_reproduced_discrepancy"
    assert summary["published_lane"]["rabinovich_fabrikant_status"] == "published_benchmark_failed"
    assert "RF lambda_3" in summary["published_lane"]["documented_discrepancy"]
    assert summary["local_full_history_qr_lane"]["validated_against_dk2018_published_lane"] is False
    contract_rows = _load_csv(STAGE_DIRS["F4_3"] / "dk2018_published_contract.csv")
    assert len(contract_rows) == 2
    assert all(row["sensitivity_interpretation"] == "fixed_published_contract_reused_not_parameter_sweep" for row in contract_rows)


def test_f4_4_reuses_fischer_diagnostics_without_promotion() -> None:
    summary = _load_json(STAGE_DIRS["F4_4"] / "summary.json")
    assert summary["status"] == "controlled_benchmark_with_documented_discrepancies"
    assert summary["validated"] is False
    assert summary["rows_total"] == 24
    assert summary["rows_failed"] == 8
    assert summary["sensitivity_runs_reused"] == 164
    assert set(summary["sensitivity_axes_reused"]) == {"delta", "gs_policy", "h", "k", "q1_mode", "t_clone"}


def test_every_method_control_has_required_references() -> None:
    summary = _load_json(GLOBAL_SUMMARY)
    for row in summary["method_controls"]:
        assert row["controlled_benchmark"]
        assert row["sensitivity_reference"]
        assert (PROJECT_ROOT / row["sensitivity_reference"]).is_file()
        assert row["bibliographic_or_internal_reference"]
        assert row["validated_by_f4"] is False
