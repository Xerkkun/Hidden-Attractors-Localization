"""Independent Wolfram/Python checks for integer SALI/GALI indices."""

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

from run_wolfram_validations import find_wolframscript  # noqa: E402
from sali_gali_integer_compare_wolfram import (  # noqa: E402
    CORE_NUMERIC_TOLERANCE,
    DEFAULT_SUMMARY,
    EXPECTED_FIXTURE_NAMES,
    FLOW_NUMERIC_TOLERANCE,
    MAP_NUMERIC_TOLERANCE,
    PUBLIC_API_NAMES,
    SOURCE_ANCHORS,
    SYSTEM_ID,
    PublicAlignmentAPIUnavailable,
    _load_public_api,
    compare_wolfram_summary,
    public_api_status,
)


CASE_PATH = ROOT / "validation" / "wolfram" / "cases" / "sali_gali_integer.wl"
COMPARATOR_PATH = (
    ROOT / "validation" / "python" / "sali_gali_integer_compare_wolfram.py"
)
PERSISTED_SUMMARY = (
    ROOT
    / "validation"
    / "outputs"
    / "wolfram"
    / f"{SYSTEM_ID}_verified"
    / f"{SYSTEM_ID}_validation_summary.json"
)
UNVERIFIED_SUMMARY = (
    ROOT
    / "validation"
    / "outputs"
    / "wolfram"
    / SYSTEM_ID
    / f"{SYSTEM_ID}_validation_summary.json"
)


@pytest.mark.hygiene
def test_sali_gali_wolfram_case_is_independent() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for anchor in SOURCE_ANCHORS.values():
        assert anchor in text
    assert "workingPrecision = 80;" in text
    assert "MatrixPower[" in text
    assert "MatrixExp[" in text
    assert "GALIGramExact[" in text
    assert "GALICauchyBinetExact[" in text
    assert "SingularValueList[" in text
    assert "NormalizeColumnsExact[" in text
    assert "hidden_attractors" not in lowered
    assert "main.tex" not in lowered
    assert "local_reports" not in lowered
    assert '"hafo_source_read" -> false' in lowered
    assert '"report_input_used" -> false' in lowered
    assert '"hafo_formula_imported" -> false' in lowered
    assert "finite exact linear tangent-algebra" in lowered
    assert "no nonlinear chaos classification" in lowered
    assert "hyperbolic fixtures are not" in lowered
    assert "fractional" in lowered
    assert "sali/gali claim" in lowered


@pytest.mark.hygiene
def test_sali_gali_wolfram_case_covers_exact_contracts() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    for fixture_name in EXPECTED_FIXTURE_NAMES:
        assert f'"{fixture_name}"' in text
    required_tests = {
        "sali_matches_minimum_parallel_antiparallel_norm",
        "gali_matches_gram_cauchy_binet_and_svd_ldi",
        "gali2_sali_identity_is_exact",
        "rotation_map_preserves_all_alignment_indices",
        "diagonal_hyperbolic_map_matches_closed_sequences",
        "diagonal_flow_matches_closed_sali_gali2_gali3",
        "lyapunov_gap_limits_match_exact_rates",
        "signed_scale_and_permutation_invariance",
        "individual_normalization_is_not_gram_schmidt",
    }
    for test_name in required_tests:
        assert test_name in text
    assert "rotationExpectedSALI = ConstantArray[2/Sqrt[5]" in text
    assert "rotationExpectedGALI2 = ConstantArray[4/5" in text
    assert "rotationExpectedGALI3 = ConstantArray[4/5" in text
    assert "hyperbolicMapFirstSALIThresholdIteration == 3" in text
    assert "hyperbolicMapFirstGALI2ThresholdIteration == 3" in text
    assert "flowGALI3LogRate == 3" in text
    assert 'systemID <> "_validation_summary.json"' in text
    assert DEFAULT_SUMMARY == PERSISTED_SUMMARY


@pytest.mark.hygiene
def test_sali_gali_comparator_names_every_agreed_public_api() -> None:
    text = COMPARATOR_PATH.read_text(encoding="utf-8")
    for name in PUBLIC_API_NAMES:
        assert f'"{name}"' in text
    assert "importlib.import_module(PUBLIC_API_MODULE)" in text
    assert "PublicAlignmentAPIUnavailable" in text
    assert "backend=\"numpy\"" in text
    assert "q=1.0" in text
    assert "CORE_NUMERIC_TOLERANCE" in text
    assert "MAP_NUMERIC_TOLERANCE" in text
    assert "FLOW_NUMERIC_TOLERANCE" in text


@pytest.mark.hygiene
def test_failed_sali_gali_artifact_is_not_the_promotion_oracle() -> None:
    assert DEFAULT_SUMMARY == PERSISTED_SUMMARY
    assert UNVERIFIED_SUMMARY != PERSISTED_SUMMARY
    assert PERSISTED_SUMMARY.parent.name == f"{SYSTEM_ID}_verified"
    if not UNVERIFIED_SUMMARY.exists():
        pytest.skip("unverified integer SALI/GALI artifact is absent")
    payload = json.loads(UNVERIFIED_SUMMARY.read_text(encoding="utf-8"))
    assert payload.get("passed") is not True


@pytest.mark.unit
def test_sali_gali_comparator_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compare_wolfram_summary(tmp_path / "missing_validation_summary.json")


@pytest.mark.unit
def test_sali_gali_comparator_rejects_wrong_system_id(tmp_path: Path) -> None:
    artifact = tmp_path / "wrong_validation_summary.json"
    artifact.write_text(
        json.dumps({"system_id": "not_sali_gali_integer"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected system_id"):
        compare_wolfram_summary(artifact)


@pytest.mark.unit
def test_sali_gali_public_api_dependency_is_explicit() -> None:
    available, diagnostic = public_api_status()
    if available:
        api = _load_public_api()
        assert tuple(api) == PUBLIC_API_NAMES
        for name in PUBLIC_API_NAMES:
            assert callable(api[name])
            assert name in diagnostic
    else:
        assert "public SALI/GALI" in diagnostic
        with pytest.raises(PublicAlignmentAPIUnavailable):
            _load_public_api()


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
        or not any(character.isdigit() for character in diagnostic)
    ):
        pytest.skip(
            "wolframscript is installed but its local kernel/license is unavailable: "
            f"{diagnostic.replace(chr(10), ' ')[:240]}"
        )


@pytest.fixture(scope="module")
def live_summary_path() -> Path:
    executable = find_wolframscript()
    if executable is None:
        pytest.skip("wolframscript is not installed or discoverable")
    _wolfram_probe_or_skip(executable)

    temp_root = Path(r"C:\tmp") if os.name == "nt" else Path(tempfile.gettempdir())
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="hafo_sali_gali_integer_",
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
            "independent integer SALI/GALI Wolfram case failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
        summary_path = output_dir / f"{SYSTEM_ID}_validation_summary.json"
        assert summary_path.exists()
        yield summary_path


@pytest.mark.wolfram
def test_sali_gali_wolfram_case_live_independent(
    live_summary_path: Path,
) -> None:
    payload = json.loads(live_summary_path.read_text(encoding="utf-8"))
    assert payload["system_id"] == SYSTEM_ID
    assert payload["passed"] is True
    assert tuple(payload["fixtures"]) == EXPECTED_FIXTURE_NAMES
    assert all(test["passed"] is True for test in payload["tests"])
    assert payload["numeric_cross_checks"]["working_precision"] == 80
    assert "no general nonlinear chaos classification" in payload["evidence_boundary"]
    assert "fractional SALI/GALI claim" in payload["evidence_boundary"]


@pytest.mark.wolfram
def test_sali_gali_wolfram_case_live_matches_public_api(
    live_summary_path: Path,
) -> None:
    available, diagnostic = public_api_status()
    if not available:
        pytest.skip(diagnostic)
    result = compare_wolfram_summary(live_summary_path)
    assert result["source_anchors_match"] is True
    assert result["independence_flags_match"] is True
    assert result["wolfram_tests_pass"] is True
    assert result["wolfram_high_precision_match"] is True
    assert result["convention_contract_match"] is True
    assert result["core_max_diff"] <= CORE_NUMERIC_TOLERANCE
    assert result["map_max_diff"] <= MAP_NUMERIC_TOLERANCE
    assert result["flow_max_diff"] <= FLOW_NUMERIC_TOLERANCE
    assert result["passed"] is True
    assert "no general nonlinear chaos classification" in result["evidence_boundary"]


@pytest.mark.wolfram
def test_persisted_sali_gali_wolfram_output_matches_python() -> None:
    if not PERSISTED_SUMMARY.exists():
        pytest.skip(
            "explicitly promoted PASS integer SALI/GALI artifact is absent; "
            "the live Wolfram validation remains the current authority until "
            f"a verified summary is promoted to {PERSISTED_SUMMARY}"
        )
    available, diagnostic = public_api_status()
    if not available:
        pytest.skip(diagnostic)
    result = compare_wolfram_summary(PERSISTED_SUMMARY)
    assert result["passed"] is True
    assert result["core_max_diff"] <= result["tolerances"]["core"]
    assert result["map_max_diff"] <= result["tolerances"]["map"]
    assert result["flow_max_diff"] <= result["tolerances"]["flow"]
    assert "hiddenness" in result["evidence_boundary"]
