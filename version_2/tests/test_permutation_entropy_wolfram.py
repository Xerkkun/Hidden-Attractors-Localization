"""Independent Wolfram/Python checks for Bandt--Pompe entropy."""

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

from permutation_entropy_compare_wolfram import (  # noqa: E402
    DEFAULT_SUMMARY,
    EXPECTED_FIXTURE_NAMES,
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
    / "permutation_entropy.wl"
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
def test_permutation_entropy_wolfram_case_is_independent() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for anchor in SOURCE_ANCHORS.values():
        assert anchor in text
    assert "CandidateWindows[" in text
    assert "OrdinalPermutationStableIndex[" in text
    assert "LehmerRankLexicographic[" in text
    assert "Log[2, #]" in text
    assert "hidden_attractors" not in lowered
    assert "main.tex" not in lowered
    assert "local_reports" not in lowered
    assert '"hafo_source_read" -> false' in lowered
    assert '"report_input_used" -> false' in lowered
    assert '"hafo_formula_imported" -> false' in lowered
    assert "finite exact-sequence" in lowered
    assert "no entropy-rate, ks-entropy" in lowered
    assert "chaos, attraction, or hiddenness" in lowered


@pytest.mark.hygiene
def test_permutation_entropy_wolfram_case_covers_required_fixtures() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    for fixture_name in EXPECTED_FIXTURE_NAMES:
        assert f'"{fixture_name}"' in text
    required_tests = {
        "chronological_forward_windows_are_exact",
        "lehmer_ranking_is_zero_based_and_lexicographic",
        "m3_tau1_ranks_and_counts_match",
        "m3_tau2_ranks_and_counts_match",
        "stable_index_resolves_equal_values_chronologically",
        "omit_discards_exactly_tied_windows",
        "counts_and_probabilities_are_normalized_by_accepted_windows",
        "entropy_uses_base_two",
        "normalized_entropy_divides_by_log2_factorial_m",
    }
    for test_name in required_tests:
        assert test_name in text
    assert "expectedTau1Ranks = {0, 0, 4, 2, 4, 3};" in text
    assert "expectedTau1Counts = {2, 0, 1, 1, 2, 0};" in text
    assert "expectedTau2Ranks = {1, 0, 5, 4, 3};" in text
    assert "expectedTau2Counts = {1, 1, 0, 1, 1, 1};" in text
    assert "expectedStableRanks = {0, 4, 3, 0, 0, 4, 2, 4};" in text
    assert "expectedStableCounts = {3, 0, 1, 1, 3, 0};" in text
    assert "expectedOmitRanks = {4, 3, 0, 2, 4};" in text
    assert "expectedOmitCounts = {1, 0, 1, 1, 2, 0};" in text
    assert 'systemID <> "_validation_summary.json"' in text
    assert DEFAULT_SUMMARY == PERSISTED_SUMMARY


@pytest.mark.unit
def test_permutation_entropy_comparator_rejects_missing_artifact(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        compare_wolfram_summary(tmp_path / "missing_validation_summary.json")


@pytest.mark.unit
def test_permutation_entropy_comparator_rejects_wrong_system_id(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "wrong_validation_summary.json"
    artifact.write_text(
        json.dumps({"system_id": "not_permutation_entropy"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected system_id"):
        compare_wolfram_summary(artifact)


def _live_wolfram_result(executable: str) -> dict[str, object]:
    temp_root = Path(tempfile.gettempdir())
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="hafo_permutation_entropy_",
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
            "independent permutation-entropy Wolfram case failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
        summary_path = output_dir / f"{SYSTEM_ID}_validation_summary.json"
        assert summary_path.exists()
        return compare_wolfram_summary(summary_path)


@pytest.mark.wolfram
def test_permutation_entropy_wolfram_case_live() -> None:
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
    assert "no entropy-rate, KS-entropy" in result["evidence_boundary"]


@pytest.mark.wolfram
def test_persisted_permutation_entropy_wolfram_output_matches_python() -> None:
    if not PERSISTED_SUMMARY.exists():
        pytest.skip(
            "persisted permutation-entropy Wolfram artifact is absent; "
            "$env:WOLFRAM_OUT='validation/outputs/wolfram/"
            f"{SYSTEM_ID}'; wolframscript -file '{CASE_PATH}'"
        )
    result = compare_wolfram_summary(PERSISTED_SUMMARY)
    assert result["source_anchors_match"] is True
    assert result["independence_flags_match"] is True
    assert result["wolfram_tests_pass"] is True
    assert result["convention_contract_match"] is True
    assert result["fixture_contracts_match"] is True
    assert result["cross_implementation_max_diff"] <= result["tolerance"]
    assert result["passed"] is True
    assert "no entropy-rate, KS-entropy" in result["evidence_boundary"]
    assert "hiddenness claim" in result["evidence_boundary"]
