"""Independent Wolfram/Python checks for correlation sums and their fit."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PYTHON = ROOT / "validation" / "python"
if str(VALIDATION_PYTHON) not in sys.path:
    sys.path.insert(0, str(VALIDATION_PYTHON))

from correlation_dimension_compare_wolfram import (  # noqa: E402
    DEFAULT_SUMMARY,
    NUMERIC_TOLERANCE,
    SOURCE_ANCHORS,
    SYSTEM_ID,
    compare_wolfram_summary,
)
from run_wolfram_validations import find_wolframscript  # noqa: E402


CASE_PATH = (
    ROOT
    / "validation"
    / "wolfram"
    / "cases"
    / "correlation_dimension.wl"
)
PERSISTED_SUMMARY = (
    ROOT
    / "validation"
    / "outputs"
    / "wolfram"
    / SYSTEM_ID
    / f"{SYSTEM_ID}_validation_summary.json"
)


@pytest.mark.hygiene
def test_correlation_dimension_wolfram_case_is_independent() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for anchor in SOURCE_ANCHORS.values():
        assert anchor in text
    assert "pointsExact = {" in text
    assert "Norm[" in text
    assert "distance < radius" in text
    assert "theilerWindow = 1;" in text
    assert "LeastSquares[" in text
    assert "localSlopes = Differences[" in text
    assert "hidden_attractors" not in lowered
    assert "main.tex" not in lowered
    assert "local_reports" not in lowered
    assert '"hafo_source_read" -> false' in lowered
    assert '"report_input_used" -> false' in lowered
    assert '"hafo_formula_imported" -> false' in lowered
    assert "finite exact-set" in lowered
    assert "no scaling-region validity" in lowered
    assert "chaos, attraction, or" in lowered


@pytest.mark.hygiene
def test_correlation_dimension_wolfram_case_covers_required_checks() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    required_tests = {
        "positive_theiler_window_is_applied",
        "admissible_pair_denominator_matches_closed_count",
        "strict_norm_less_than_radius_counts_match",
        "pairs_on_radius_are_excluded",
        "correlation_sum_is_count_over_admissible_denominator",
        "fit_range_is_explicit_and_uses_only_open_unit_sums",
        "least_squares_fit_is_finite",
        "local_log_log_slopes_are_exportable",
    }
    for test_name in required_tests:
        assert test_name in text
    assert "radiiExact = {1, 11/10, 3/2, 2, 21/10, 23/10};" in text
    assert "fitRadiusRangeExact = {11/10, 21/10};" in text
    assert "expectedCountsExact = {0, 4, 6, 6, 8, 10};" in text
    assert 'systemID <> "_validation_summary.json"' in text
    assert DEFAULT_SUMMARY == PERSISTED_SUMMARY


@pytest.mark.unit
def test_correlation_dimension_comparator_rejects_missing_artifact(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        compare_wolfram_summary(tmp_path / "missing_validation_summary.json")


@pytest.mark.unit
def test_correlation_dimension_comparator_rejects_wrong_system_id(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "wrong_validation_summary.json"
    artifact.write_text(
        json.dumps({"system_id": "not_correlation_dimension"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected system_id"):
        compare_wolfram_summary(artifact)


def _live_wolfram_result(executable: str) -> dict[str, object]:
    temp_root = Path(r"C:\tmp") if os.name == "nt" else Path(tempfile.gettempdir())
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="hafo_correlation_dimension_",
        dir=temp_root,
    ) as output_name:
        output_dir = Path(output_name)
        environment = os.environ.copy()
        environment["WOLFRAM_OUT"] = str(output_dir)
        completed = subprocess.run(
            [executable, "-file", str(CASE_PATH)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert completed.returncode == 0, (
            "independent correlation-dimension Wolfram case failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
        summary_path = (
            output_dir / f"{SYSTEM_ID}_validation_summary.json"
        )
        assert summary_path.exists()
        return compare_wolfram_summary(summary_path)


@pytest.mark.wolfram
def test_correlation_dimension_wolfram_case_live() -> None:
    executable = find_wolframscript()
    if executable is None:
        pytest.skip("wolframscript is not installed or discoverable")

    probe = subprocess.run(
        [executable, "-code", "$Version"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0:
        diagnostic = (probe.stderr or probe.stdout).strip().replace("\n", " ")
        pytest.skip(
            "wolframscript is installed but its local kernel/license is unavailable: "
            f"{diagnostic[:240]}"
        )

    result = _live_wolfram_result(executable)
    assert result["passed"] is True
    assert result["cross_implementation_max_diff"] <= NUMERIC_TOLERANCE
    assert "no scaling-region validity" in result["evidence_boundary"]


@pytest.mark.wolfram
def test_persisted_correlation_dimension_wolfram_output_matches_python() -> None:
    if not PERSISTED_SUMMARY.exists():
        pytest.skip(
            "persisted correlation-dimension Wolfram artifact is absent; "
            "$env:WOLFRAM_OUT='validation/outputs/wolfram/"
            f"{SYSTEM_ID}'; wolframscript -file '{CASE_PATH}'"
        )
    result = compare_wolfram_summary(PERSISTED_SUMMARY)
    assert result["source_anchors_match"] is True
    assert result["independence_flags_match"] is True
    assert result["wolfram_tests_pass"] is True
    assert result["curve_contract_match"] is True
    assert result["fit_contract_match"] is True
    assert result["cross_implementation_max_diff"] <= result["tolerance"]
    assert result["passed"] is True
    assert "no scaling-region validity" in result["evidence_boundary"]
    assert "fractal-dimension" in result["evidence_boundary"]
