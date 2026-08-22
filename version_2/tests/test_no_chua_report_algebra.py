from __future__ import annotations

import json
import csv
import hashlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = ROOT / "docs" / "master_report_geometric_topological"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_wolfram_nonchua_report_artifact_is_self_consistent() -> None:
    artifact = _load(
        DOC_ROOT / "wolfram" / "no_chua_complete_algebra_validation.json"
    )
    checks = artifact["global_checks"]
    assert checks["identity_residuals_zero"] is True
    assert checks["maximum_equilibrium_residual"] < 1e-12
    assert checks["maximum_characteristic_residual"] < 1e-10
    assert checks["maximum_trace_sum_residual"] < 1e-10

    case = artifact["cases"]["generalized_lorenz_q0995"]
    origin_index = next(
        index
        for index, point in enumerate(case["equilibria"])
        if max(abs(value) for value in point) < 1e-12
    )
    origin_eigenvalues = sorted(
        pair[0]
        for pair in case["linearizations"][origin_index]["eigenvalues_real_imag"]
        if abs(pair[1]) < 1e-12
    )
    assert origin_eigenvalues == pytest.approx(
        [-7.155804677345547, -1.0, 2.755804677345547], abs=1e-12
    )

    outer = next(
        linearization
        for index, linearization in enumerate(case["linearizations"])
        if index != origin_index
    )
    assert outer["trace"] == pytest.approx(-5.4, abs=1e-12)
    assert outer["characteristic_coefficients_leading_to_constant"] == pytest.approx(
        [1.0, 5.4, 14.84769427061672, 78.88], abs=1e-11
    )


def test_active_reference_record_contains_only_recomputed_lorenz_spectrum() -> None:
    path = ROOT / "validation" / "references" / "published_validation_data_extraction_v1.json"
    text = path.read_text(encoding="utf-8")
    assert "2.5576" not in text
    assert "-7.5576" not in text
    assert "-5.9570" not in text
    assert "3.6026" not in text
    assert "-7.155804677345547" in text
    assert "3.832489172241882" in text


def test_nonchua_lyapunov_figure_manifest_matches_executed_rows() -> None:
    figure_dir = DOC_ROOT / "figures" / "nonchua_benchmarks"
    manifest = _load(figure_dir / "lyapunov_validation_overview_manifest.json")
    assert manifest["row_counts"] == {"dk2018": 2, "fischer2020": 24}
    assert manifest["fischer_classification"] == {
        "jerk": {"quantitative": 0, "sign": 4, "discrepancy": 4},
        "financial": {"quantitative": 5, "sign": 2, "discrepancy": 1},
        "four_wing": {"quantitative": 5, "sign": 0, "discrepancy": 3},
    }
    assert (figure_dir / "lyapunov_validation_overview.pdf").is_file()
    assert (figure_dir / "lyapunov_validation_overview.png").is_file()


def test_kalman_fitts_integrator_robustness_matrix_is_complete() -> None:
    case_dir = ROOT / "validation" / "reference_cases" / "kalman_fitts_integer_q1"
    result = _load(case_dir / "06_integrator_robustness.json")
    with (case_dir / "06_integrator_robustness_matrix.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert result["summary"] == {
        "n_rows": 8,
        "n_consistent": 8,
        "n_discrepant": 0,
        "all_configurations_consistent": True,
        "discrepancies": [],
        "interpretation": (
            "A consistent row supports finite-time numerical persistence of the "
            "maintained cycle under that solver contract only."
        ),
    }
    assert len(rows) == 8
    assert all(row["configuration_consistent"] == "True" for row in rows)
    assert {float(row["horizon"]) for row in rows} == {300.0, 600.0}
    assert {row["config_id"] for row in rows} == {
        "dop853_standard",
        "dop853_tight",
        "rk4_h_0p01",
        "rk4_h_0p005",
    }

    for source_key, hash_key in (
        ("target_seed_source", "target_seed_source_sha256"),
        ("canonical_cloud_source", "canonical_cloud_source_sha256"),
    ):
        source = ROOT / result["inputs"][source_key]
        assert _sha256(source) == result["inputs"][hash_key]
