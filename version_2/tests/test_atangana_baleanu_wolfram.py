"""Independent Wolfram/Python checks for the sampled ABC operator."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erfcx


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PYTHON = ROOT / "validation" / "python"
if str(VALIDATION_PYTHON) not in sys.path:
    sys.path.insert(0, str(VALIDATION_PYTHON))

from atangana_baleanu_compare_wolfram import (  # noqa: E402
    SOURCE_ANCHORS,
    SYSTEM_ID,
    compare_wolfram_summary,
)
from run_wolfram_validations import find_wolframscript  # noqa: E402


CASE_PATH = (
    ROOT / "validation" / "wolfram" / "cases" / "atangana_baleanu_operator.wl"
)
PERSISTED_SUMMARY = (
    ROOT
    / "validation"
    / "outputs"
    / "wolfram"
    / SYSTEM_ID
    / f"{SYSTEM_ID}_validation_summary.json"
)


def _half_order_primitive(elapsed: np.ndarray) -> np.ndarray:
    root = np.sqrt(elapsed)
    return erfcx(root) - 1.0 + 2.0 * root / np.sqrt(np.pi)


def _abc_convolution(
    samples: np.ndarray,
    weights: np.ndarray,
    scale: float,
) -> np.ndarray:
    increments = np.diff(np.asarray(samples, dtype=np.float64))
    return np.concatenate(([0.0], scale * np.convolve(increments, weights)[: increments.size]))


def _independent_fixture() -> dict[str, object]:
    alpha = 0.5
    step = 1.0 / 16.0
    n_steps = 12
    lower_terminal = -2.0
    elapsed = step * np.arange(n_steps + 1, dtype=np.float64)
    physical = lower_terminal + elapsed
    normalization = 1.0 - alpha + alpha / math.gamma(alpha)
    scale = normalization / (1.0 - alpha)
    primitive = _half_order_primitive(elapsed)
    weights = np.diff(primitive) / step

    constant_samples = np.full(n_steps + 1, 4.25)
    constant_values = _abc_convolution(constant_samples, weights, scale)
    intercept = -3.0
    slope = 2.75
    ramp_samples = intercept + slope * elapsed
    ramp_values = _abc_convolution(ramp_samples, weights, scale)
    ramp_closed = scale * slope * primitive
    sample_values = (
        3.0 / 5.0
        + (2.0 / 7.0) * elapsed
        + elapsed**2 / 5.0
        + np.sin(3.0 * elapsed) / 11.0
    )
    sample_derivative = _abc_convolution(sample_values, weights, scale)
    identity_elapsed = np.arange(17, dtype=np.float64) / 16.0
    identity_values = erfcx(np.sqrt(identity_elapsed))

    return {
        "system_id": SYSTEM_ID,
        "validation_scope": (
            "independent_ABC_half_order_kernel_weights_and_sampled_operator"
        ),
        "evidence_boundary": (
            "finite-grid operator consistency; no stability, chaos, hiddenness, "
            "FDE-solver convergence, or initial-compatibility claim"
        ),
        "source": {
            **SOURCE_ANCHORS,
            "scope": "ABC kernel identity and finite uniform-grid sampled operator",
            "hafo_source_read": False,
            "report_input_used": False,
        },
        "parameters": {
            "alpha": alpha,
            "kernel_rate": 1.0,
            "step": step,
            "n_steps": n_steps,
            "lower_terminal": lower_terminal,
            "normalization": normalization,
            "normalization_convention": (
                "B(alpha)=1-alpha+alpha/Gamma(alpha)"
            ),
        },
        "kernel_identity": {
            "formula": (
                "MittagLefflerE[1/2,-Sqrt[t]]=Exp[t] Erfc[Sqrt[t]]"
            ),
            "elapsed_times": identity_elapsed.tolist(),
            "mittag_leffler_values": identity_values.tolist(),
            "erfc_values": identity_values.tolist(),
            "symbolic_residual": "0",
            "max_residual": 0.0,
        },
        "weights": {
            "definition": (
                "w_k=h^-1 Integrate[E_alpha(-alpha/(1-alpha) s^alpha),"
                "{s,k h,(k+1) h}]"
            ),
            "backend": "independent erfcx closed-form fixture",
            "values": weights.tolist(),
            "primitive_values": weights.tolist(),
            "primitive_max_residual": 0.0,
        },
        "constant_case": {
            "samples": constant_samples.tolist(),
            "derivative_values": constant_values.tolist(),
            "max_abs": 0.0,
        },
        "ramp_case": {
            "intercept": intercept,
            "slope": slope,
            "elapsed_times": elapsed.tolist(),
            "physical_times": physical.tolist(),
            "samples": ramp_samples.tolist(),
            "derivative_values": ramp_values.tolist(),
            "closed_form_values": ramp_closed.tolist(),
            "closed_form_max_residual": float(
                np.max(np.abs(ramp_values - ramp_closed), initial=0.0)
            ),
        },
        "sample_case": {
            "formula": "3/5+(2/7)t+t^2/5+Sin[3t]/11",
            "elapsed_times": elapsed.tolist(),
            "physical_times": physical.tolist(),
            "samples": sample_values.tolist(),
            "derivative_values": sample_derivative.tolist(),
        },
        "tests": [{"name": "independent_fixture", "passed": True}],
        "passed": True,
    }


@pytest.mark.hygiene
def test_abc_wolfram_case_is_independent_and_source_anchored() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for doi in SOURCE_ANCHORS.values():
        assert doi in text
    assert "MittagLefflerE" in text
    assert "Exp[t] Erfc[Sqrt[t]]" in text
    assert "hidden_attractors" not in lowered
    assert "main.tex" not in lowered
    assert "local_reports" not in lowered
    assert '"hafo_source_read" -> false' in lowered
    assert '"report_input_used" -> false' in lowered
    assert "no stability, chaos, hiddenness" in lowered


@pytest.mark.hygiene
def test_abc_wolfram_case_covers_required_identities_and_schema() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    required_tests = {
        "half_order_mittag_leffler_erfc_identity",
        "half_order_kernel_primitive_closed_form",
        "interval_weights_match_mittag_leffler_primitive",
        "interval_weights_are_positive_and_monotone",
        "constant_abc_derivative_is_zero",
        "ramp_abc_derivative_matches_closed_form",
    }
    for test_name in required_tests:
        assert test_name in text
    assert 'alphaExact = 1/2;' in text
    assert 'systemID <> "_validation_summary.json"' in text


@pytest.mark.unit
def test_abc_comparator_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compare_wolfram_summary(tmp_path / "missing_validation_summary.json")


@pytest.mark.unit
def test_abc_comparator_rejects_wrong_system_id(tmp_path: Path) -> None:
    artifact = tmp_path / "wrong_validation_summary.json"
    artifact.write_text(
        json.dumps({"system_id": "not_atangana_baleanu_operator"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected system_id"):
        compare_wolfram_summary(artifact)


@pytest.mark.unit
def test_abc_comparator_accepts_independent_half_order_fixture(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / f"{SYSTEM_ID}_validation_summary.json"
    artifact.write_text(
        json.dumps(_independent_fixture(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = compare_wolfram_summary(artifact)
    assert result["source_anchors_match"] is True
    assert result["independence_flags_match"] is True
    assert result["wolfram_tests_pass"] is True
    assert result["backends_match"] is True
    assert result["weights"]["python_to_wolfram_max_diff"] <= result["tolerance"]
    assert result["constant_case"]["fft_to_wolfram_max_diff"] <= result["tolerance"]
    assert result["ramp_case"]["numba_to_wolfram_max_diff"] <= result["tolerance"]
    assert result["sample_case"]["python_to_wolfram_max_diff"] <= result["tolerance"]
    assert result["worst_numeric_diff"] <= result["tolerance"]
    assert result["passed"] is True


@pytest.mark.wolfram
def test_persisted_abc_wolfram_output_matches_python() -> None:
    if not PERSISTED_SUMMARY.exists():
        pytest.skip(
            "persisted ABC Wolfram artifact is absent; generate it with "
            "$env:WOLFRAM_OUT='validation/outputs/wolfram/"
            f"{SYSTEM_ID}'; wolframscript -file '{CASE_PATH}'"
        )
    result = compare_wolfram_summary(PERSISTED_SUMMARY)
    assert result["wolfram_tests_pass"] is True
    assert result["independence_flags_match"] is True
    assert result["source_anchors_match"] is True
    assert result["backends_match"] is True
    assert result["worst_numeric_diff"] <= result["tolerance"]
    assert result["passed"] is True


@pytest.mark.wolfram
def test_abc_wolfram_case_live(tmp_path: Path) -> None:
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
        "independent ABC Wolfram case failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    summary_path = output_dir / f"{SYSTEM_ID}_validation_summary.json"
    assert summary_path.exists()
    result = compare_wolfram_summary(summary_path)
    assert result["passed"] is True
    assert result["worst_numeric_diff"] <= 5.0e-12
    assert "no stability, chaos, hiddenness" in result["evidence_boundary"]
