"""Independent Wolfram/Python checks for Hadamard-family operators."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import binom, gamma


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PYTHON = ROOT / "validation" / "python"
if str(VALIDATION_PYTHON) not in sys.path:
    sys.path.insert(0, str(VALIDATION_PYTHON))

from hadamard_compare_wolfram import (  # noqa: E402
    SOURCE_ANCHORS,
    SYSTEM_ID,
    compare_wolfram_summary,
)
from run_wolfram_validations import (  # noqa: E402
    DEFAULT_CASES,
    find_wolframscript,
)


CASE_RELPATH = "validation/wolfram/cases/hadamard_fractional_operator.wl"
CASE_PATH = ROOT / CASE_RELPATH
PERSISTED_SUMMARY = (
    ROOT
    / "validation"
    / "outputs"
    / "wolfram"
    / SYSTEM_ID
    / f"{SYSTEM_ID}_validation_summary.json"
)


@pytest.mark.hygiene
def test_hadamard_wolfram_case_is_independent_and_source_anchored() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    for doi in SOURCE_ANCHORS.values():
        assert doi in text
    assert "hidden_attractors" not in lowered
    assert "main.tex" not in lowered
    assert "local_reports" not in lowered
    assert '"hafo_source_read" -> false' in lowered
    assert '"report_input_used" -> false' in lowered
    assert "no stability, chaos, hiddenness" in lowered


@pytest.mark.hygiene
def test_hadamard_wolfram_case_covers_required_identities_and_runner_schema() -> None:
    text = CASE_PATH.read_text(encoding="utf-8")
    required_tests = {
        "logarithmic_kernel_transform",
        "hadamard_measure_transform",
        "dilation_maps_to_log_derivative",
        "hadamard_integral_log_power_beta_gamma_identity",
        "hadamard_log_power_derivative_identity",
        "caputo_hadamard_log_power_derivative_identity",
        "hadamard_constant_identity",
        "caputo_hadamard_constant_is_zero",
        "log_power_continuum_formula_has_q1_limit",
        "constant_continuum_formula_has_q1_limit",
        "q1_bdf1_weights_match_backward_difference",
        "q1_bdf2_weights_match_bdf2_polynomial",
        "bdf1_caputo_log_power_endpoint_converges",
        "bdf2_caputo_log_power_endpoint_converges",
        "raw_hadamard_constant_endpoint_converges",
        "caputo_hadamard_constant_forcing_closed_form",
    }
    for test_name in required_tests:
        assert test_name in text
    assert 'systemID <> "_validation_summary.json"' in text
    assert CASE_RELPATH in DEFAULT_CASES


@pytest.mark.unit
def test_hadamard_comparator_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compare_wolfram_summary(tmp_path / "missing_validation_summary.json")


@pytest.mark.unit
def test_hadamard_comparator_rejects_wrong_system_id(tmp_path: Path) -> None:
    artifact = tmp_path / "wrong_validation_summary.json"
    artifact.write_text(
        json.dumps({"system_id": "not_hadamard_fractional_operator"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected system_id"):
        compare_wolfram_summary(artifact)


@pytest.mark.unit
def test_hadamard_comparator_accepts_independent_finite_fixture(
    tmp_path: Path,
) -> None:
    order = 0.6
    lower_terminal = 2.0
    degree = 3
    constant_value = 1.4
    n_steps = 8
    log_step = 1.0 / n_steps
    log_times = np.arange(n_steps + 1, dtype=float) * log_step
    physical_times = lower_terminal * np.exp(log_times)
    samples = constant_value + log_times**degree
    weights = np.asarray(
        [(-1.0) ** index * binom(order, index) for index in range(n_steps + 1)]
    )
    scale = log_step ** (-order)
    raw_values = scale * np.convolve(weights, samples)[: n_steps + 1]
    shifted_values = scale * np.convolve(
        weights, samples - samples[0]
    )[: n_steps + 1]
    log_power_analytic = float(gamma(degree + 1.0) / gamma(degree + 1.0 - order))
    raw_constant_analytic = float(constant_value / gamma(1.0 - order))
    raw_constant_values = scale * np.convolve(
        weights, np.full(n_steps + 1, constant_value)
    )[: n_steps + 1]

    abm_log_step = 0.1
    abm_steps = 10
    abm_log_times = np.arange(abm_steps + 1, dtype=float) * abm_log_step
    abm_initial = 1.25
    abm_forcing = 0.8
    abm_states = (
        abm_initial
        + abm_forcing * abm_log_times**order / gamma(order + 1.0)
    )
    payload = {
        "system_id": SYSTEM_ID,
        "evidence_boundary": (
            "symbolic identities and finite-grid numerical consistency; "
            "no stability, chaos, hiddenness, or general convergence claim"
        ),
        "source": {
            **SOURCE_ANCHORS,
            "hafo_source_read": False,
            "report_input_used": False,
        },
        "parameters": {
            "order": order,
            "lower_terminal": lower_terminal,
            "log_power_degree": degree,
            "constant_value": constant_value,
        },
        "cq": {
            "sample_case": {
                "log_times": log_times.tolist(),
                "physical_times": physical_times.tolist(),
                "samples": samples.tolist(),
                "shifted_samples": (samples - samples[0]).tolist(),
                "rows": [
                    {
                        "bdf_order": 1,
                        "weights": weights.tolist(),
                        "raw_values": raw_values.tolist(),
                        "caputo_shifted_values": shifted_values.tolist(),
                        "caputo_constant_max_abs": 0.0,
                    }
                ],
            },
            "convergence_rows": [
                {
                    "bdf_order": 1,
                    "n_steps": n_steps,
                    "log_step": log_step,
                    "caputo_log_power_endpoint": float(shifted_values[-1]),
                    "caputo_log_power_analytic": log_power_analytic,
                    "caputo_log_power_abs_error": abs(
                        float(shifted_values[-1]) - log_power_analytic
                    ),
                    "raw_constant_endpoint": float(raw_constant_values[-1]),
                    "raw_constant_analytic": raw_constant_analytic,
                    "raw_constant_abs_error": abs(
                        float(raw_constant_values[-1]) - raw_constant_analytic
                    ),
                }
            ],
            "q1_bdf1_weights": [1.0, -1.0, 0.0, 0.0],
            "q1_bdf2_weights": [1.5, -2.0, 0.5, 0.0],
        },
        "abm_manufactured": {
            "order": order,
            "lower_terminal": lower_terminal,
            "initial_state": abm_initial,
            "forcing": abm_forcing,
            "log_step": abm_log_step,
            "n_steps": abm_steps,
            "log_times": abm_log_times.tolist(),
            "physical_times": (
                lower_terminal * np.exp(abm_log_times)
            ).tolist(),
            "analytic_states": abm_states.tolist(),
        },
        "tests": [{"name": "independent_fixture", "passed": True}],
        "passed": True,
    }
    artifact = tmp_path / f"{SYSTEM_ID}_validation_summary.json"
    artifact.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = compare_wolfram_summary(artifact)
    assert result["source_anchors_match"] is True
    assert result["independence_flags_match"] is True
    assert result["wolfram_tests_pass"] is True
    assert result["worst_numeric_diff"] <= result["tolerance"]
    assert result["passed"] is True


@pytest.mark.wolfram
def test_persisted_hadamard_wolfram_output_matches_python() -> None:
    if not PERSISTED_SUMMARY.exists():
        pytest.skip(
            "persisted Hadamard Wolfram artifact is absent; generate it with "
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
def test_hadamard_wolfram_case_live(tmp_path: Path) -> None:
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
        "independent Hadamard Wolfram case failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    summary_path = output_dir / f"{SYSTEM_ID}_validation_summary.json"
    assert summary_path.exists()
    result = compare_wolfram_summary(summary_path)
    assert result["passed"] is True
    assert result["worst_numeric_diff"] <= 5.0e-12
    assert "no stability, chaos, hiddenness" in result["evidence_boundary"]
