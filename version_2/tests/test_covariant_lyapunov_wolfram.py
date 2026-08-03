"""Independent Wolfram/Python checks for integer-order CLV fixtures."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PYTHON = ROOT / "validation" / "python"
if str(VALIDATION_PYTHON) not in sys.path:
    sys.path.insert(0, str(VALIDATION_PYTHON))

from covariant_lyapunov_integer_compare_wolfram import (  # noqa: E402
    CORE_COVARIANCE_TOLERANCE,
    CORE_LINE_TOLERANCE,
    DEFAULT_SUMMARY,
    DIRECT_CHECKPOINT_INDICES,
    EXPECTED_FIXTURE_NAMES,
    EXPECTED_NUMERIC,
    NUMBER_OF_STEPS,
    PUBLIC_API_MODULE,
    RETAINED_INDICES,
    SOURCE_ANCHORS,
    SYSTEM_ID,
    _line_distances,
    _load_public_api,
    _pair_angles,
    compare_wolfram_summary,
    public_api_status,
    validate_wolfram_summary,
)
from run_wolfram_validations import DEFAULT_CASES  # noqa: E402


CASE_PATH = (
    ROOT
    / "validation"
    / "wolfram"
    / "cases"
    / "covariant_lyapunov_integer.wl"
)
COMPARATOR_PATH = (
    ROOT
    / "validation"
    / "python"
    / "covariant_lyapunov_integer_compare_wolfram.py"
)
CASE_RELPATH = "validation/wolfram/cases/covariant_lyapunov_integer.wl"


@pytest.mark.hygiene
def test_clv_wolfram_case_is_independent_and_high_precision() -> None:
    assert CASE_RELPATH in DEFAULT_CASES
    text = CASE_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for anchor in SOURCE_ANCHORS.values():
        assert anchor in text
    assert "workingPrecision = 80;" in text
    assert "numberOfSteps = 120;" in text
    assert "retainedIndices = Range[40, 80];" in text
    assert "directCheckpointIndices = {0, 40, 60, 80, 120};" in text
    assert "PositiveModifiedGramSchmidt[" in text
    assert "BackwardGinelliHistory[" in text
    assert "LinearSolve[" in text
    assert "MatrixExp[" in text
    assert "MatrixPower[" in text
    assert "PairAngles[" in text
    assert "NonNormalCommutatorNorm[" in text
    assert "QRDecomposition[" not in text
    assert "Eigensystem[" not in text
    assert "hidden_attractors" not in lowered
    assert "main.tex" not in lowered
    assert "local_reports" not in lowered
    assert '"hafo_source_read" -> false' in lowered
    assert '"hafo_formula_imported" -> false' in lowered
    assert '"built_in_qr_used" -> false' in lowered
    assert '"built_in_eigensystem_used" -> false' in lowered
    assert "fractional-order clv validity" in lowered


@pytest.mark.hygiene
def test_clv_wolfram_case_contains_exact_agreed_fixtures() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    required_fragments = (
        "mapO2Exact = {{3/5, -4/5}, {4/5, 3/5}};",
        "mapT2Exact = {{4, 1}, {0, 2}};",
        "mapAExact = {{56/25, 33/25}, {8/25, 94/25}};",
        "mapTerminalSeedExact = {{1, 1/3}, {0, 1}};",
        "flowStepExact = Log[2];",
        '"nonnormal_map_2d"',
        '"constant_flow_3d"',
        '"pair_indices_zero_based"',
        '"retained_pair_angles_radians"',
        '"non_normal_commutator_frobenius_norm"',
    )
    for fragment in required_fragments:
        assert fragment in text
    required_tests = (
        "map_exact_similarity_and_eigenlines",
        "flow_exact_similarity_exponential_and_eigenlines",
        "own_modified_gram_schmidt_has_positive_diagonal",
        "ginelli_backward_triangular_recursions_are_projectively_exact",
        "retained_map_clvs_match_declared_exact_eigenlines",
        "retained_flow_clvs_match_declared_exact_eigenlines",
        "direct_matrix_power_checkpoints_match_backward_recursion",
        "unoriented_pair_angles_match_exact_lines_and_are_constant",
        "both_cocycles_are_explicitly_nonnormal",
    )
    for name in required_tests:
        assert name in text


@pytest.mark.hygiene
def test_clv_comparator_is_optional_core_and_projective() -> None:
    text = COMPARATOR_PATH.read_text(encoding="utf-8")
    assert "importlib.import_module(PUBLIC_API_MODULE)" in text
    assert 'PUBLIC_CORE_NAME = "integer_covariant_vectors_from_qr_history"' in text
    assert 'PUBLIC_ANGLE_NAME = "covariant_lyapunov_angles"' in text
    assert "PublicCLVAPIUnavailable" in text
    assert "normalized rank-one projectors" in text
    assert "CORE_LINE_TOLERANCE" in text
    assert "CORE_COVARIANCE_TOLERANCE" in text
    assert "require_core" in text
    if os.name == "nt":
        assert DEFAULT_SUMMARY.drive.lower() == "c:"
        assert str(DEFAULT_SUMMARY).lower().startswith(r"c:\tmp")
    else:
        assert str(DEFAULT_SUMMARY).startswith(tempfile.gettempdir())
    assert "validation\\outputs" not in str(DEFAULT_SUMMARY).lower()


@pytest.mark.unit
def test_projective_line_distance_is_sign_invariant_and_stable() -> None:
    first = np.asarray([[1.0, 1.0], [0.0, 1.0]])
    signed = np.asarray([[-1.0, 1.0], [0.0, 1.0]])
    distances = _line_distances(first, signed)
    np.testing.assert_allclose(distances, 0.0, atol=1.0e-15, rtol=0.0)


@pytest.mark.unit
def test_pair_angles_are_unoriented_acute_angles() -> None:
    vectors = np.asarray([[1.0, -1.0, 1.0], [0.0, 0.0, 1.0]])
    pairs = np.asarray([(0, 1), (0, 2)], dtype=np.int64)
    angles = _pair_angles(vectors, pairs)
    np.testing.assert_allclose(
        angles,
        [0.0, np.pi / 4.0],
        atol=1.0e-15,
        rtol=0.0,
    )


@pytest.mark.unit
def test_clv_comparator_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_wolfram_summary(tmp_path / "missing.json")


@pytest.mark.unit
def test_clv_comparator_rejects_wrong_system_id(tmp_path: Path) -> None:
    artifact = tmp_path / "wrong.json"
    artifact.write_text(
        json.dumps({"system_id": "not_covariant_lyapunov_integer"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected system_id"):
        validate_wolfram_summary(artifact)


@pytest.mark.unit
def test_declared_python_fixture_contract_matches_requested_dimensions() -> None:
    assert tuple(EXPECTED_NUMERIC) == EXPECTED_FIXTURE_NAMES
    assert EXPECTED_NUMERIC["nonnormal_map_2d"]["matrix"].shape == (2, 2)
    assert EXPECTED_NUMERIC["constant_flow_3d"]["cocycle_matrix"].shape == (3, 3)
    assert NUMBER_OF_STEPS == 120
    np.testing.assert_array_equal(RETAINED_INDICES, np.arange(40, 81))
    np.testing.assert_array_equal(
        DIRECT_CHECKPOINT_INDICES,
        [0, 40, 60, 80, 120],
    )


@pytest.mark.unit
def test_public_clv_dependency_is_explicit_but_parser_independent() -> None:
    available, diagnostic = public_api_status()
    if available:
        api = _load_public_api()
        assert callable(api["integer_covariant_vectors_from_qr_history"])
        assert PUBLIC_API_MODULE in diagnostic
    else:
        assert "public CLV" in diagnostic


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
        or not any(character.isdigit() for character in diagnostic)
    ):
        pytest.skip(
            "wolframscript is installed but its local kernel/configuration is "
            f"unavailable: {diagnostic.replace(chr(10), ' ')[:240]}"
        )


@pytest.fixture(scope="module")
def live_clv_summary_path() -> Path:
    executable = _find_wolframscript()
    if executable is None:
        pytest.skip("wolframscript is not installed or discoverable")
    _wolfram_probe_or_skip(executable)

    temp_root = Path(r"C:\tmp") if os.name == "nt" else Path(tempfile.gettempdir())
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="hafo_covariant_lyapunov_integer_",
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
            "independent integer CLV Wolfram case failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
        summary_path = output_dir / f"{SYSTEM_ID}_validation_summary.json"
        assert summary_path.exists()
        yield summary_path


@pytest.mark.wolfram
def test_clv_wolfram_case_live_independent(live_clv_summary_path: Path) -> None:
    payload = json.loads(live_clv_summary_path.read_text(encoding="utf-8"))
    assert payload["system_id"] == SYSTEM_ID
    assert payload["passed"] is True
    assert tuple(payload["fixtures"]) == EXPECTED_FIXTURE_NAMES
    assert all(test["passed"] is True for test in payload["tests"])
    assert payload["numeric_cross_checks"]["working_precision"] == 80
    assert payload["numeric_cross_checks"][
        "map_non_normal_commutator_frobenius_norm"
    ] > 0.0
    assert payload["numeric_cross_checks"][
        "flow_non_normal_commutator_frobenius_norm"
    ] > 0.0
    result = validate_wolfram_summary(live_clv_summary_path)
    assert result["passed"] is True


@pytest.mark.wolfram
def test_clv_wolfram_case_live_optionally_matches_public_api(
    live_clv_summary_path: Path,
) -> None:
    available, diagnostic = public_api_status()
    if not available:
        pytest.skip(diagnostic)
    result = compare_wolfram_summary(live_clv_summary_path, require_core=True)
    assert result["independent"]["passed"] is True
    assert result["cross_implementation_performed"] is True
    assert result["cross_implementation_passed"] is True
    assert result["passed"] is True
    for fixture_result in result["cross_implementation_results"].values():
        assert fixture_result["line_distance_max_diff"] <= CORE_LINE_TOLERANCE
        assert (
            fixture_result["covariance_max_line_distance"]
            <= CORE_COVARIANCE_TOLERANCE
        )
