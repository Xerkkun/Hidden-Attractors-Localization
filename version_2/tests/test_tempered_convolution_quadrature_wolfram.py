"""Independent Wolfram/Python checks for tempered convolution quadrature."""

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

import tempered_convolution_quadrature_compare_wolfram as comparator  # noqa: E402
from tempered_convolution_quadrature_compare_wolfram import (  # noqa: E402
    CORE_TOLERANCE,
    DEFAULT_SUMMARY,
    EXPECTED_BDF_ORDERS,
    EXPECTED_DEFINITIONS,
    INDEPENDENT_TOLERANCE,
    PUBLIC_API_MODULE,
    SOURCE_ANCHORS,
    SYSTEM_ID,
    _factor_expansion_weights,
    _recurrence_weights,
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
    / "tempered_convolution_quadrature.wl"
)
COMPARATOR_PATH = (
    ROOT
    / "validation"
    / "python"
    / "tempered_convolution_quadrature_compare_wolfram.py"
)
CASE_RELPATH = "validation/wolfram/cases/tempered_convolution_quadrature.wl"
PERSISTED_DIR = (
    ROOT
    / "validation"
    / "outputs"
    / "wolfram"
    / "tempered_convolution_quadrature_verified"
)
PERSISTED_SUMMARY = (
    PERSISTED_DIR / "tempered_convolution_quadrature_validation_summary.json"
)
PERSISTED_COMPARISON = (
    PERSISTED_DIR / "tempered_convolution_quadrature_comparison.json"
)


@pytest.mark.hygiene
def test_tempered_cq_wolfram_case_is_independent_and_high_precision() -> None:
    assert CASE_RELPATH in DEFAULT_CASES
    text = CASE_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for anchor in SOURCE_ANCHORS.values():
        assert anchor in text
    assert "workingPrecision = 80;" in text
    assert "BDFWeightRecurrence[" in text
    assert "BDFWeightExpansion[" in text
    assert "TemperedCQDirect[" in text
    assert "TemperedCQConjugated[" in text
    assert "Series[" not in text
    assert "hidden_attractors" not in lowered
    assert "main.tex" not in lowered
    assert "local_reports" not in lowered
    assert '"hafo_source_read" -> false' in lowered
    assert '"hafo_formula_imported" -> false' in lowered
    assert '"built_in_series_used" -> false' in lowered
    assert '"no_minus_lambda_power_x" -> true' in lowered
    assert "delta(exp(-lambda*h)*z)^q/h^q" in text
    assert "[delta(z)/h+lambda]^q is not evaluated or identified" in text
    assert "no stability or convergence theorem" in lowered
    assert "chaos, attraction, or hiddenness" in lowered


@pytest.mark.hygiene
def test_tempered_cq_case_covers_requested_reductions_and_vectors() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    required_tests = (
        "bdf1_recurrence_matches_own_binomial_expansion_at_high_precision",
        "bdf2_recurrence_matches_own_factor_expansion_at_high_precision",
        "q_one_bdf1_and_bdf2_weights_are_exact_finite_polynomials",
        "tempered_rl_direct_weights_equal_exponential_conjugation",
        "tempered_caputo_direct_weights_equal_shifted_exponential_conjugation",
        "lambda_zero_reduces_to_untempered_rl_and_caputo_cq",
        "q_one_reduces_to_conjugated_bdf1_and_bdf2",
        "vector_rl_fixture_is_strictly_componentwise",
        "vector_caputo_fixture_is_strictly_componentwise",
        "vector_fixture_matches_exponential_conjugation",
    )
    for name in required_tests:
        assert name in text
    assert '"tempered_riemann_liouville"' in text
    assert '"tempered_caputo"' in text
    assert "vectorOrdersExact = {2/5, 4/5};" in text
    assert "vectorTemperingsExact = {1/5, 3/5};" in text
    assert 'systemID <> "_validation_summary.json"' in text


@pytest.mark.hygiene
def test_comparator_keeps_core_import_optional_and_temp_scoped() -> None:
    text = COMPARATOR_PATH.read_text(encoding="utf-8")
    assert "from hidden_attractors" not in text
    assert "import hidden_attractors" not in text
    assert "importlib.import_module(PUBLIC_API_MODULE)" in text
    assert 'PUBLIC_API_MODULE = "hidden_attractors.fractional"' in text
    assert "PublicTemperedCQAPIUnavailable" in text
    assert "require_core" in text
    if os.name == "nt":
        assert DEFAULT_SUMMARY.drive.lower() == "c:"
        assert str(DEFAULT_SUMMARY).lower().startswith(r"c:\tmp")
    else:
        assert str(DEFAULT_SUMMARY).startswith(tempfile.gettempdir())
    assert "validation\\outputs" not in str(DEFAULT_SUMMARY).lower()


@pytest.mark.unit
@pytest.mark.parametrize("bdf_order", EXPECTED_BDF_ORDERS)
@pytest.mark.parametrize("order", [0.2, 0.55, 0.9, 1.0])
def test_python_independent_weight_paths_agree(
    bdf_order: int,
    order: float,
) -> None:
    recurrence = _recurrence_weights(order, 40, bdf_order)
    expansion = _factor_expansion_weights(order, 40, bdf_order)
    np.testing.assert_allclose(recurrence, expansion, rtol=2.0e-14, atol=2.0e-16)


@pytest.mark.unit
def test_tempered_comparator_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_wolfram_summary(tmp_path / "missing.json")


@pytest.mark.unit
def test_tempered_comparator_rejects_wrong_system_id(tmp_path: Path) -> None:
    artifact = tmp_path / "wrong.json"
    artifact.write_text(
        json.dumps({"system_id": "not_tempered_convolution_quadrature"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected system_id"):
        validate_wolfram_summary(artifact)


@pytest.mark.unit
def test_public_tempered_dependency_is_explicit_but_optional() -> None:
    available, diagnostic = public_api_status()
    if available:
        assert PUBLIC_API_MODULE in diagnostic
    else:
        assert "public" in diagnostic


@pytest.mark.unit
def test_persisted_tempered_cq_evidence_validates_and_matches_public_core() -> None:
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
        or not any(character.isdigit() for character in diagnostic)
    ):
        pytest.skip(
            "wolframscript is installed but its local kernel/configuration is "
            f"unavailable: {diagnostic.replace(chr(10), ' ')[:240]}"
        )


@pytest.fixture(scope="module")
def live_tempered_cq_summary_path() -> Path:
    executable = _find_wolframscript()
    if executable is None:
        pytest.skip("wolframscript is not installed or discoverable")
    _wolfram_probe_or_skip(executable)

    temporary_root = (
        Path(r"C:\tmp") if os.name == "nt" else Path(tempfile.gettempdir())
    )
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="hafo_tempered_convolution_quadrature_",
        dir=temporary_root,
    ) as output_name:
        output_dir = Path(output_name)
        if os.name == "nt":
            assert str(output_dir.resolve()).lower().startswith("c:\\tmp\\")
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
            "independent tempered-CQ Wolfram case failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
        summary_path = output_dir / f"{SYSTEM_ID}_validation_summary.json"
        assert summary_path.exists()
        yield summary_path


@pytest.mark.wolfram
def test_tempered_cq_wolfram_case_live_independent(
    live_tempered_cq_summary_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(live_tempered_cq_summary_path.read_text(encoding="utf-8"))
    assert payload["system_id"] == SYSTEM_ID
    assert payload["passed"] is True
    assert len(payload["tests"]) == 18
    assert all(test["passed"] is True for test in payload["tests"])
    assert payload["conventions"]["working_precision"] == 80
    assert payload["conventions"]["no_minus_lambda_power_x"] is True
    assert payload["conventions"]["tempered_generating_function"] == (
        "delta(exp(-lambda*h)*z)^q/h^q"
    )
    assert {
        (row["definition"], int(row["bdf_order"]))
        for row in payload["vector_fixture"]["rows"]
    } == {
        (definition, bdf_order)
        for definition in EXPECTED_DEFINITIONS
        for bdf_order in EXPECTED_BDF_ORDERS
    }

    def fail_if_imported(_name: str) -> None:
        raise AssertionError("independent validation attempted a package import")

    monkeypatch.setattr(comparator.importlib, "import_module", fail_if_imported)
    result = validate_wolfram_summary(live_tempered_cq_summary_path)
    assert result["source_anchors_match"] is True
    assert result["independence_flags_match"] is True
    assert result["convention_contract_match"] is True
    assert result["wolfram_tests_pass"] is True
    assert result["numeric_max_diff"] <= result["tolerance"]
    assert result["convergence"]["endpoint_errors_monotonically_decrease"] is True
    assert result["passed"] is True


@pytest.mark.wolfram
def test_tempered_cq_require_core_is_enforced(
    live_tempered_cq_summary_path: Path,
) -> None:
    comparator._load_public_api.cache_clear()
    available, _diagnostic = public_api_status()
    result = compare_wolfram_summary(
        live_tempered_cq_summary_path,
        require_core=True,
    )
    assert result["independent"]["passed"] is True
    assert result["require_core"] is True
    if not available:
        assert result["public_core_available"] is False
        assert result["cross_implementation_performed"] is False
        assert result["passed"] is False
        return
    assert result["public_core_available"] is True
    assert result["cross_implementation_performed"] is True
    assert result["cross_implementation_passed"] is True
    assert result["cross_implementation_result"]["max_diff"] <= CORE_TOLERANCE
    assert result["passed"] is True
