"""Independent Wolfram/Python checks for tempered Fast Method II history."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PYTHON = ROOT / "validation" / "python"
if str(VALIDATION_PYTHON) not in sys.path:
    sys.path.insert(0, str(VALIDATION_PYTHON))

import tempered_fast_multistep_history_compare_wolfram as comparator  # noqa: E402
from run_wolfram_validations import DEFAULT_CASES  # noqa: E402
from tempered_fast_multistep_history_compare_wolfram import (  # noqa: E402
    CORE_TOLERANCE,
    EXPECTED_DEFINITIONS,
    EXPECTED_METHODS,
    INDEPENDENT_TOLERANCE,
    PERSISTED_SUMMARY,
    PUBLIC_API_MODULE,
    SOURCE_ANCHORS,
    SYSTEM_ID,
    compare_wolfram_summary,
    public_api_status,
    validate_wolfram_summary,
)


CASE_RELPATH = "validation/wolfram/cases/tempered_fast_multistep_history.wl"
CASE_PATH = ROOT / CASE_RELPATH
COMPARATOR_PATH = (
    ROOT
    / "validation"
    / "python"
    / "tempered_fast_multistep_history_compare_wolfram.py"
)
PERSISTED_COMPARISON = (
    PERSISTED_SUMMARY.parent / "tempered_fast_multistep_history_comparison.json"
)


@pytest.mark.hygiene
def test_fast_history_wolfram_case_is_independent_and_high_precision() -> None:
    assert CASE_RELPATH in DEFAULT_CASES
    text = CASE_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for anchor in SOURCE_ANCHORS.values():
        assert anchor in text
    assert "workingPrecision = 80;" in text
    assert "FastTemperedOperator[" in text
    assert "DirectTemperedOperator[" in text
    assert "MultistepWeights[" in text
    assert "-Sin[Pi q]/Pi" in text
    assert '"fbdf1"' in text
    assert '"gngf2"' in text
    assert '"fractional_bdf2_claimed" -> False' in text
    assert "hidden_attractors" not in lowered
    assert "main.tex" not in lowered
    assert "local_reports" not in lowered
    assert '"hafo_source_read" -> false' in lowered
    assert '"hafo_formula_imported" -> false' in lowered
    assert "no general trapezoidal-tail certificate" in lowered
    assert "chaos, attraction, or hiddenness" in lowered


@pytest.mark.hygiene
def test_fast_history_case_covers_algebra_history_and_anchor() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    required = (
        "fbdf1_real_axis_sign_matches_beta_reflection",
        "gngf2_coefficients_match_independent_series_coefficients",
        "gngf2_q_one_limit_is_exact_bdf2_polynomial",
        "fbdf1_fast_rl_and_caputo_match_direct_history",
        "gngf2_fast_rl_and_caputo_match_direct_history",
        "conjugated_caputo_exponential_anchor_is_annihilated",
        "all_real_history_recurrences_are_contractively_damped",
    )
    for name in required:
        assert name in text
    assert "fixtureQuadraturePoints = 1297;" in text
    assert "fixtureLocalSteps = 12;" in text
    assert 'systemID <> "_validation_summary.json"' in text


@pytest.mark.hygiene
def test_fast_history_comparator_keeps_public_import_optional() -> None:
    text = COMPARATOR_PATH.read_text(encoding="utf-8")
    assert "from hidden_attractors" not in text
    assert "import hidden_attractors" not in text
    assert "importlib.import_module(PUBLIC_API_MODULE)" in text
    assert 'PUBLIC_API_MODULE = "hidden_attractors.fractional"' in text
    assert "PublicFastHistoryAPIUnavailable" in text
    assert "require_core" in text


@pytest.mark.unit
def test_fast_history_comparator_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_wolfram_summary(tmp_path / "missing.json")


@pytest.mark.unit
def test_fast_history_comparator_rejects_wrong_system_id(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "wrong.json"
    artifact.write_text(
        json.dumps({"system_id": "wrong_fast_history"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected system_id"):
        validate_wolfram_summary(artifact)


@pytest.mark.unit
def test_public_fast_history_dependency_is_explicit() -> None:
    available, diagnostic = public_api_status()
    assert available is True
    assert PUBLIC_API_MODULE in diagnostic


@pytest.mark.unit
def test_persisted_fast_history_evidence_and_core_both_pass() -> None:
    independent = validate_wolfram_summary(PERSISTED_SUMMARY)
    assert independent["passed"] is True
    assert independent["numeric_max_diff"] <= INDEPENDENT_TOLERANCE
    comparison = json.loads(PERSISTED_COMPARISON.read_text(encoding="utf-8"))
    assert comparison["passed"] is True
    assert comparison["require_core"] is True
    assert comparison["cross_implementation_passed"] is True
    live_core = compare_wolfram_summary(PERSISTED_SUMMARY, require_core=True)
    assert live_core["passed"] is True
    assert live_core["cross_implementation_result"]["max_diff"] <= CORE_TOLERANCE


def _find_wolframscript() -> str | None:
    discovered = shutil.which("wolframscript")
    if discovered:
        return discovered
    windows_default = Path(
        r"C:\Program Files\Wolfram Research\WolframScript\wolframscript.exe"
    )
    return str(windows_default) if windows_default.exists() else None


def _wolfram_probe_or_skip(executable: str) -> None:
    probe = subprocess.run(
        [executable, "-code", "Print[$Version]"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    diagnostic = (probe.stdout + "\n" + probe.stderr).strip()
    if (
        probe.returncode != 0
        or "Failed to open config" in diagnostic
        or "Failed to open configuaration" in diagnostic
    ):
        pytest.skip(
            "wolframscript is installed but unavailable: "
            f"{diagnostic.replace(chr(10), ' ')[:240]}"
        )


@pytest.fixture(scope="module")
def live_fast_history_summary_path() -> Path:
    executable = _find_wolframscript()
    if executable is None:
        pytest.skip("wolframscript is not installed or discoverable")
    _wolfram_probe_or_skip(executable)
    temporary_root = (
        Path(r"C:\tmp") if os.name == "nt" else Path(tempfile.gettempdir())
    )
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="hafo_tempered_fast_history_",
        dir=temporary_root,
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
            "independent fast-history Wolfram case failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
        summary = output_dir / f"{SYSTEM_ID}_validation_summary.json"
        assert summary.exists()
        yield summary


@pytest.mark.wolfram
def test_fast_history_wolfram_case_live_independent(
    live_fast_history_summary_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(live_fast_history_summary_path.read_text(encoding="utf-8"))
    assert payload["system_id"] == SYSTEM_ID
    assert payload["passed"] is True
    assert len(payload["tests"]) == 13
    assert all(test["passed"] is True for test in payload["tests"])
    assert payload["conventions"]["working_precision"] == 80
    assert payload["conventions"]["fractional_bdf2_claimed"] is False
    assert {
        (row["method"], row["definition"])
        for row in payload["fixture"]["rows"]
    } == {
        (method, definition)
        for method in EXPECTED_METHODS
        for definition in EXPECTED_DEFINITIONS
    }

    def fail_if_imported(_name: str) -> None:
        raise AssertionError("independent validation attempted a package import")

    monkeypatch.setattr(comparator.importlib, "import_module", fail_if_imported)
    result = validate_wolfram_summary(live_fast_history_summary_path)
    assert result["source_anchors_match"] is True
    assert result["independence_flags_match"] is True
    assert result["convention_contract_match"] is True
    assert result["wolfram_tests_pass"] is True
    assert result["numeric_max_diff"] <= result["tolerance"]
    assert result["passed"] is True


@pytest.mark.wolfram
def test_fast_history_live_artifact_matches_public_core(
    live_fast_history_summary_path: Path,
) -> None:
    comparator._load_public_api.cache_clear()
    result = compare_wolfram_summary(
        live_fast_history_summary_path,
        require_core=True,
    )
    assert result["independent"]["passed"] is True
    assert result["public_core_available"] is True
    assert result["cross_implementation_performed"] is True
    assert result["cross_implementation_passed"] is True
    assert result["cross_implementation_result"]["max_diff"] <= CORE_TOLERANCE
    assert result["passed"] is True
