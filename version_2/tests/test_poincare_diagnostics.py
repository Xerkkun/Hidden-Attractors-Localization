"""Contracts for F5.4 Poincare published-case diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
POINCARE_ROOT = ROOT / "validation" / "chaos_validation" / "dynamics_diagnostics" / "poincare"
CASES = POINCARE_ROOT / "cases"


def _yaml(name: str) -> dict:
    return yaml.safe_load((CASES / name).read_text(encoding="utf-8"))


def test_poincare_case_yamls_record_published_numeric_values() -> None:
    integer = _yaml("chua_integer_q1_reference.yaml")
    danca = _yaml("danca2017_chua_fractional_saturation_q09998.yaml")
    wu = _yaml("wu2023_chua_fractional_arctan_q099.yaml")
    assert integer["seed"]["omega0_python"] == 2.039186939959001
    assert integer["seed"]["k_python"] == 0.209867354515084
    assert integer["seed"]["a0_python"] == 5.856145086257356
    assert danca["q"] == 0.9998
    assert danca["parameters"] == {
        "alpha": 8.4562,
        "beta": 12.0732,
        "gamma": 0.0052,
        "m0": -0.1768,
        "m1": -1.1468,
    }
    assert danca["seed"]["article_initial_condition_reported"] is False
    assert wu["q"] == 0.99
    assert wu["seed"]["x0_plus"] == [13.8, 0.7093, -19.8768]
    assert wu["seed"]["x0_minus"] == [-13.8, -0.7093, 19.8768]
    assert wu["integration"]["memory_length"] == 40.0


def test_global_summary_records_method_validation_and_no_certification() -> None:
    summary = json.loads(
        (POINCARE_ROOT / "poincare_diagnostics_summary.json").read_text(encoding="utf-8")
    )
    assert summary["stage"] == "F5.4_poincare_diagnostic"
    assert summary["status"] == "completed_structured_outputs"
    assert summary["cases_total"] == 3
    assert summary["poincare_method_validation"]["status"] == "passed_synthetic_crossing_tests"
    application = summary["poincare_application_to_published_cases"]
    assert application["status"] == "completed_structured_outputs"
    assert application["chaos_certified_by_poincare"] is False
    assert application["hiddenness_certified_by_poincare"] is False
    assert summary["caputo_periodic_orbit_claim"] is False


def test_each_published_case_has_standardized_outputs() -> None:
    for case_id in (
        "chua_integer_q1_reference",
        "danca2017_chua_fractional_saturation_q09998",
        "wu2023_chua_fractional_arctan_q099",
    ):
        case_dir = POINCARE_ROOT / case_id
        assert {
            "poincare_points.csv",
            "poincare_summary.json",
            "poincare_metadata.json",
            "poincare_section.csv",
            "README.md",
        } <= {path.name for path in case_dir.iterdir()}
        summary = json.loads((case_dir / "poincare_summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == "completed_structured_outputs"
        assert summary["chaos_certified_by_poincare"] is False
        assert summary["hiddenness_certified_by_poincare"] is False
        assert summary["periodic_orbit_exact"] is False
        assert summary["caputo_periodic_orbit_exact"] is False
