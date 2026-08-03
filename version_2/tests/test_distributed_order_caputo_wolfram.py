"""Independent Wolfram/Python checks for distributed-order Caputo L1."""

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

from distributed_order_caputo_l1_compare_wolfram import (  # noqa: E402
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
    / "distributed_order_caputo_l1.wl"
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
def test_distributed_order_wolfram_case_is_independent() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for anchor in SOURCE_ANCHORS.values():
        assert anchor in text
    assert "kernelPrimitives = Table[" in text
    assert "Integrate[" in text
    assert "directLinearL1Recurrence" in text
    assert "hidden_attractors" not in lowered
    assert "main.tex" not in lowered
    assert "local_reports" not in lowered
    assert '"hafo_source_read" -> false' in lowered
    assert '"report_input_used" -> false' in lowered
    assert '"hafo_formula_imported" -> false' in lowered
    assert "finite algebraic/numerical consistency" in lowered
    assert "no global convergence theorem" in lowered
    assert "chaos, attraction, or hiddenness" in lowered


@pytest.mark.hygiene
def test_distributed_order_wolfram_case_covers_required_checks() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    required_tests = {
        "multinode_kernel_matches_integrated_caputo_intervals",
        "combined_kernel_is_positive_and_strictly_decreasing",
        "affine_caputo_integrals_match_closed_forms",
        "manufactured_linear_recurrence_residual_is_small",
        "manufactured_affine_trajectory_matches_exact_samples",
    }
    for test_name in required_tests:
        assert test_name in text
    assert "orderNodesExact = {1/4, 3/5, 4/5};" in text
    assert "orderMassesExact = {1/5, 1/2, 3/10};" in text
    assert "(K0-lambda)x_n" in text
    assert 'systemID <> "_validation_summary.json"' in text
    assert DEFAULT_SUMMARY == PERSISTED_SUMMARY


@pytest.mark.unit
def test_distributed_order_comparator_rejects_missing_artifact(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        compare_wolfram_summary(tmp_path / "missing_validation_summary.json")


@pytest.mark.unit
def test_distributed_order_comparator_rejects_wrong_system_id(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "wrong_validation_summary.json"
    artifact.write_text(
        json.dumps({"system_id": "not_distributed_order_caputo"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected system_id"):
        compare_wolfram_summary(artifact)


@pytest.mark.wolfram
def test_distributed_order_wolfram_case_live() -> None:
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

    output_dir = PERSISTED_SUMMARY.parent
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
        "independent distributed-order Caputo Wolfram case failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert PERSISTED_SUMMARY.exists()
    result = compare_wolfram_summary(PERSISTED_SUMMARY)
    assert result["passed"] is True
    assert result["cross_implementation_max_diff"] <= NUMERIC_TOLERANCE
    assert "no global convergence theorem" in result["evidence_boundary"]


@pytest.mark.wolfram
def test_persisted_distributed_order_wolfram_output_matches_python() -> None:
    if not PERSISTED_SUMMARY.exists():
        pytest.skip(
            "persisted distributed-order Caputo Wolfram artifact is absent; "
            "$env:WOLFRAM_OUT='validation/outputs/wolfram/"
            f"{SYSTEM_ID}'; wolframscript -file '{CASE_PATH}'"
        )
    result = compare_wolfram_summary(PERSISTED_SUMMARY)
    assert result["source_anchors_match"] is True
    assert result["independence_flags_match"] is True
    assert result["wolfram_tests_pass"] is True
    assert result["symbolic_checks_pass"] is True
    assert result["kernel_shape_checks_pass"] is True
    assert result["solver_contract_match"] is True
    assert result["cross_implementation_max_diff"] <= result["tolerance"]
    assert result["passed"] is True
    assert "no global convergence theorem" in result["evidence_boundary"]
    assert "chaos, attraction, or hiddenness" in result["evidence_boundary"]
