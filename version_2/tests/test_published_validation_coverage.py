from __future__ import annotations

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_STATUSES = {
    "executable_regression",
    "implemented_partial_reproduction",
    "reference_data_only",
    "documented_missing_data",
    "not_in_scope_for_current_release",
    "future_extension"
}

def test_published_validation_coverage_is_complete_and_valid() -> None:
    extraction_path = ROOT / "docs" / "published_validation_data_extraction_v1.json"
    coverage_path = ROOT / "validation" / "published_reference_coverage.json"

    assert extraction_path.is_file(), f"Missing extraction data at {extraction_path}"
    assert coverage_path.is_file(), f"Missing coverage data at {coverage_path}"

    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))

    # 1. Extract all expected cases from the truth extraction JSON
    expected_cases = []
    articles = extraction["articles"]

    # Kuznetsov 2017
    for case in articles["kuznetsov2017_chua_integer_df"]["published_cases"]:
        expected_cases.append(("kuznetsov2017_chua_integer_df", case["case_id"]))

    # Danca 2017
    for sys_name in articles["danca2017_fractional_hidden_attractors"]["systems"].keys():
        expected_cases.append(("danca2017_fractional_hidden_attractors", sys_name))

    # Wu 2023
    expected_cases.append(("wu2023_chua_fractional_arctan", "wu2023_chua_fractional_arctan"))

    # DK2018
    for case in articles["danca_kuznetsov2018_lyapunov_fo"]["benchmarks"]:
        expected_cases.append(("danca_kuznetsov2018_lyapunov_fo", case["case_id"]))

    # Fischer 2020
    for key in ["jerk_system", "financial_system", "four_wing_system"]:
        expected_cases.append(("fischer2020_cloned_dynamics", key))

    # 2. Check coverage database
    coverage_map = {(item["article_id"], item["case_id"]): item for item in coverage}

    # Verify that all expected cases are covered
    for article_id, case_id in expected_cases:
        key = (article_id, case_id)
        assert key in coverage_map, f"Missing coverage entry for article={article_id}, case={case_id}"

        item = coverage_map[key]
        assert "coverage_status" in item
        assert "reason" in item
        assert "release_blocking" in item
        assert "missing_data" in item
        assert "validation_artifacts" in item
        assert "test_files" in item

        status = item["coverage_status"]
        assert status in ALLOWED_STATUSES, f"Invalid coverage status '{status}' for {key}"

        # Verify that no case with missing data is marked as executable_regression
        if status == "executable_regression":
            assert len(item["missing_data"]) == 0, f"Executable regression cannot have missing data: {key}"

    # 3. Assert specific expectations from G3 Phase E
    # Kuznetsov Case 18 is executable regression
    kuz_18 = coverage_map[("kuznetsov2017_chua_integer_df", "kuznetsov2017_case_18_hidden_chaotic")]
    assert kuz_18["coverage_status"] == "executable_regression"

    # Kuznetsov Case 21 chaotic and periodic branches are not marked as executable_regression or strong_chaos_evidence
    kuz_21c = coverage_map[("kuznetsov2017_chua_integer_df", "kuznetsov2017_case_21_hidden_chaotic_branch")]
    kuz_21p = coverage_map[("kuznetsov2017_chua_integer_df", "kuznetsov2017_case_21_hidden_periodic_branch")]
    assert kuz_21c["coverage_status"] != "executable_regression"
    assert kuz_21p["coverage_status"] != "executable_regression"

    # Danca chua is implemented_partial_reproduction (since missing seed, LE, etc)
    danca_chua = coverage_map[("danca2017_fractional_hidden_attractors", "chua_fractional_saturation")]
    assert danca_chua["coverage_status"] == "implemented_partial_reproduction"
    assert "exact hidden-attractor initial condition" in danca_chua["missing_data"]

    # Danca Lorenz and RF are reference_data_only and not release_blocking
    danca_lorenz = coverage_map[("danca2017_fractional_hidden_attractors", "generalized_lorenz_fractional")]
    danca_rf = coverage_map[("danca2017_fractional_hidden_attractors", "rabinovich_fabrikant_fractional")]
    assert danca_lorenz["coverage_status"] == "reference_data_only"
    assert danca_rf["coverage_status"] == "reference_data_only"
    assert not danca_lorenz["release_blocking"]
    assert not danca_rf["release_blocking"]

    # Wu 2023 is implemented_partial_reproduction
    wu_arctan = coverage_map[("wu2023_chua_fractional_arctan", "wu2023_chua_fractional_arctan")]
    assert wu_arctan["coverage_status"] == "implemented_partial_reproduction"

    # Fischer Jerk, Financial, and Four Wing systems are implemented_partial_reproduction and have discrepancies reason
    for case_id in ["jerk_system", "financial_system", "four_wing_system"]:
        fischer_case = coverage_map[("fischer2020_cloned_dynamics", case_id)]
        assert fischer_case["coverage_status"] == "implemented_partial_reproduction"
        assert "discrepancies" in fischer_case["reason"]
