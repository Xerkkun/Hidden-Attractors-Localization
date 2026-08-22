"""Independent Wolfram/Python checks for the GL operator and recurrence."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PYTHON = ROOT / "validation" / "python"
if str(VALIDATION_PYTHON) not in sys.path:
    sys.path.insert(0, str(VALIDATION_PYTHON))

from gl_fractional_compare_wolfram import (  # noqa: E402
    SYSTEM_ID,
    compare_wolfram_summary,
)


CASE_PATH = (
    ROOT / "validation" / "wolfram" / "cases" / "gl_fractional_operator_validation.wl"
)
PERSISTED_SUMMARY = (
    ROOT
    / "validation"
    / "reference_cases"
    / SYSTEM_ID
    / f"{SYSTEM_ID}_summary.json"
)


def _find_wolframscript() -> str | None:
    executable = shutil.which("wolframscript")
    if executable is not None:
        return executable
    windows_default = Path(
        r"C:\Program Files\Wolfram Research\WolframScript\wolframscript.exe"
    )
    return str(windows_default) if windows_default.exists() else None


@pytest.mark.hygiene
def test_gl_fractional_case_is_independent_and_source_anchored() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "10.1137/0517050" in text
    assert "978-0-12-558840-9" in text
    assert "978-0-12-525550-9" in text
    assert "hidden_attractors" not in lowered
    assert "main.tex" not in lowered
    assert "local_reports" not in lowered
    assert '"hafo_source_read" -> false' in lowered
    assert '"report_input_used" -> false' in lowered
    assert "no chaos, hiddenness" in lowered


@pytest.mark.hygiene
def test_gl_fractional_case_covers_required_identities() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    required_tests = {
        "weights_match_signed_generalized_binomial",
        "caputo_monomial_beta_gamma_identity",
        "caputo_shifted_t_cubed_converges",
        "riemann_liouville_constant_symbolic_identity",
        "raw_constant_converges_to_riemann_liouville_value",
        "caputo_shifted_constant_is_zero",
        "monomial_continuum_formula_has_q1_limit",
        "q1_weights_are_first_backward_difference",
        "q1_shifted_solver_is_explicit_euler",
        "q1_raw_solver_is_explicit_euler",
        "fractional_constant_forcing_solver_converges",
    }
    for test_name in required_tests:
        assert test_name in text


@pytest.mark.unit
def test_gl_fractional_comparator_rejects_missing_artifact(tmp_path: Path) -> None:
    missing = tmp_path / "missing_gl_fractional_summary.json"
    with pytest.raises(FileNotFoundError):
        compare_wolfram_summary(missing)


@pytest.mark.wolfram
def test_gl_fractional_persisted_wolfram_output_matches_python() -> None:
    if not PERSISTED_SUMMARY.exists():
        pytest.skip(
            "persisted GL Wolfram artifact is absent; generate it explicitly with "
            "$env:WOLFRAM_OUT='validation/outputs/wolfram/"
            f"{SYSTEM_ID}'; wolframscript -file '{CASE_PATH}'"
        )
    result = compare_wolfram_summary(PERSISTED_SUMMARY)
    assert result["wolfram_tests_pass"] is True
    assert result["independence_flags_match"] is True
    assert result["source_anchors_match"] is True
    assert result["worst_numeric_diff"] <= result["tolerance"]
    assert result["passed"] is True


@pytest.mark.wolfram
def test_gl_fractional_wolfram_case_live(tmp_path: Path) -> None:
    executable = _find_wolframscript()
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
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, (
        "independent GL Wolfram case failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    summary_path = output_dir / f"{SYSTEM_ID}_summary.json"
    assert summary_path.exists()
    result = compare_wolfram_summary(summary_path)
    assert result["passed"] is True
    assert result["worst_numeric_diff"] <= 5.0e-12
    assert "no chaos, hiddenness" in result["evidence_boundary"]
