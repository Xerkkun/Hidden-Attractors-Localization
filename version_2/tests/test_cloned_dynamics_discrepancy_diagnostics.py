"""Fast contract checks for the F3 Fischer 2020 discrepancy layer."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PYTHON = ROOT / "validation" / "python"
if str(VALIDATION_PYTHON) not in sys.path:
    sys.path.insert(0, str(VALIDATION_PYTHON))

from diagnose_cloned_dynamics_discrepancies import (  # noqa: E402
    DIAGNOSTICS_DIR,
    MATRIX_FIELDS,
    SUMMARY_PATH,
    build_matrix,
    classify_sign_match,
    generate_diagnostics,
)


def _summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def test_summary_parser_generates_all_24_rows_and_required_columns() -> None:
    matrix = build_matrix(_summary())
    assert len(matrix) == 24
    assert set(MATRIX_FIELDS) <= set(matrix[0])


def test_near_zero_sign_policy_distinguishes_boundary_from_strict_mismatch() -> None:
    classified = classify_sign_match([0.004], [-0.003])
    assert classified["strict_sign_match"] is False
    assert classified["tolerant_sign_match"] is True
    assert classified["near_zero_sign_boundary"] is True
    assert classified["sign_component_statuses"] == ["near_zero_compatible"]


def test_diagnostic_generation_does_not_require_ignored_outputs_csv(tmp_path: Path) -> None:
    summary_path = tmp_path / "validation_summary.json"
    shutil.copyfile(SUMMARY_PATH, summary_path)
    generated = generate_diagnostics(
        summary_path=summary_path,
        outputs_csv_path=tmp_path / "absent.csv",
        diagnostics_dir=tmp_path / "diagnostics",
    )
    assert generated["metadata"]["outputs_csv_present"] is False
    assert generated["metadata"]["rows_total"] == 24
    assert (tmp_path / "diagnostics" / "fischer2020_discrepancy_report.md").exists()


def test_diagnostic_regeneration_preserves_partial_sweep_outputs(tmp_path: Path) -> None:
    summary_path = tmp_path / "validation_summary.json"
    diagnostics_dir = tmp_path / "diagnostics"
    shutil.copyfile(SUMMARY_PATH, summary_path)
    generate_diagnostics(
        summary_path=summary_path,
        outputs_csv_path=tmp_path / "absent.csv",
        diagnostics_dir=diagnostics_dir,
    )
    sentinel = "partial sweep evidence\n"
    sensitivity_path = diagnostics_dir / "sensitivity_delta.csv"
    sensitivity_path.write_text(sentinel, encoding="utf-8")
    generate_diagnostics(
        summary_path=summary_path,
        outputs_csv_path=tmp_path / "absent.csv",
        diagnostics_dir=diagnostics_dir,
    )
    assert sensitivity_path.read_text(encoding="utf-8") == sentinel


def test_official_diagnostic_summary_remains_conservative() -> None:
    summary = _summary()
    diagnostics = summary["discrepancy_diagnostics"]
    assert summary["status"] == "published_benchmarks_pending_discrepancy"
    assert summary["validated"] is False
    assert summary["validated_against_published_benchmarks"] is False
    assert diagnostics["status"] == "diagnostics_added"
    assert diagnostics["validated_after_diagnostics"] is False
    assert (
        DIAGNOSTICS_DIR / "fischer2020_discrepancy_report.md"
    ).exists()
    assert "chaos_verified" not in json.dumps(summary)
    assert "hidden_verified" not in json.dumps(summary)


def test_tracked_diagnostic_artifacts_exist() -> None:
    required = {
        "README.md",
        "fischer2020_discrepancy_report.md",
        "fischer2020_discrepancy_matrix.csv",
        "fischer2020_row_classification.csv",
        "sensitivity_plan.yaml",
        "sensitivity_summary.json",
        "sensitivity_delta.csv",
        "sensitivity_t_clone.csv",
        "sensitivity_h.csv",
        "sensitivity_k.csv",
        "sensitivity_gs_policy.csv",
        "near_zero_sign_policy.json",
        "diagnostic_run_metadata.json",
    }
    assert required <= {path.name for path in DIAGNOSTICS_DIR.iterdir()}
