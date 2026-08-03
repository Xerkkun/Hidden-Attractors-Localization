"""Independent Wolfram/Python checks for the ABC predictor--corrector."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PYTHON = ROOT / "validation" / "python"
if str(VALIDATION_PYTHON) not in sys.path:
    sys.path.insert(0, str(VALIDATION_PYTHON))

from abc_predictor_corrector_compare_wolfram import (  # noqa: E402
    SOURCE_ANCHORS,
    SYSTEM_ID,
    compare_wolfram_summary,
)
from run_wolfram_validations import find_wolframscript  # noqa: E402


CASE_PATH = (
    ROOT / "validation" / "wolfram" / "cases" / "abc_predictor_corrector.wl"
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
def test_abc_pcm_wolfram_case_is_independent_and_source_anchored() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for doi in SOURCE_ANCHORS.values():
        assert doi in text
    assert "Integrate[" in text
    assert "theta0Integral" in text
    assert "theta1Integral" in text
    assert "hidden_attractors" not in lowered
    assert "main.tex" not in lowered
    assert "local_reports" not in lowered
    assert '"hafo_source_read" -> false' in lowered
    assert '"report_input_used" -> false' in lowered
    assert "no convergence theorem" in lowered
    assert "chaos, attraction, or hiddenness" in lowered


@pytest.mark.hygiene
def test_abc_pcm_wolfram_case_covers_weights_and_finite_recurrence() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    required_tests = {
        "symbolic_linear_weights_match_integrals",
        "linear_weight_partition_identity",
        "manufactured_rhs_satisfies_initial_compatibility",
        "implicit_product_trapezoid_startup_converged",
        "lee_kim_jang_recurrence_matches_direct_product_integration",
        "manufactured_solution_finite_grid_error_is_bounded",
    }
    for test_name in required_tests:
        assert test_name in text
    assert "alphaExact = 7/10;" in text
    assert "normalizationExact = 1;" in text
    assert "rhsExact[time_, state_] := (time - lowerTerminalExact)^2;" in text
    assert 'systemID <> "_validation_summary.json"' in text


@pytest.mark.unit
def test_abc_pcm_comparator_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compare_wolfram_summary(tmp_path / "missing_validation_summary.json")


@pytest.mark.unit
def test_abc_pcm_comparator_rejects_wrong_system_id(tmp_path: Path) -> None:
    artifact = tmp_path / "wrong_validation_summary.json"
    artifact.write_text(
        json.dumps({"system_id": "not_abc_predictor_corrector"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected system_id"):
        compare_wolfram_summary(artifact)


@pytest.mark.wolfram
def test_persisted_abc_pcm_wolfram_output_matches_python() -> None:
    if not PERSISTED_SUMMARY.exists():
        pytest.skip(
            "persisted ABC PCM Wolfram artifact is absent; generate it with "
            "$env:WOLFRAM_OUT='validation/outputs/wolfram/"
            f"{SYSTEM_ID}'; wolframscript -file '{CASE_PATH}'"
        )
    result = compare_wolfram_summary(PERSISTED_SUMMARY)
    assert result["source_anchors_match"] is True
    assert result["independence_flags_match"] is True
    assert result["wolfram_tests_pass"] is True
    assert result["symbolic_integrals_pass"] is True
    assert result["solver_contract_match"] is True
    assert (
        result["trajectory_metrics"]["states_python_to_wolfram_max_diff"]
        <= result["tolerance"]
    )
    assert result["cross_implementation_max_diff"] <= result["tolerance"]
    assert result["passed"] is True
    assert "no convergence theorem" in result["evidence_boundary"]
    assert "chaos, attraction, or hiddenness" in result["evidence_boundary"]


@pytest.mark.wolfram
def test_abc_pcm_wolfram_case_live(tmp_path: Path) -> None:
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

    output_dir = tmp_path / SYSTEM_ID
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
        "independent ABC PCM Wolfram case failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    summary_path = output_dir / f"{SYSTEM_ID}_validation_summary.json"
    assert summary_path.exists()
    result = compare_wolfram_summary(summary_path)
    assert result["passed"] is True
    assert result["cross_implementation_max_diff"] <= 5.0e-12
    assert "no convergence theorem" in result["evidence_boundary"]
