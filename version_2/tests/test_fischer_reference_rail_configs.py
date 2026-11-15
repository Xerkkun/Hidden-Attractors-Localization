"""Derived article contracts preserve the immutable targets and their origin."""
import hashlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "validation/lyapunov_benchmarks/fischer_reference_rail_v1"


@pytest.mark.parametrize("path", sorted(CASES.glob("*.yaml")), ids=lambda path: path.stem)
def test_article_contract_provenance_and_physical_grid(path):
    case = yaml.safe_load(path.read_text(encoding="utf-8"))
    provenance = case["provenance"]
    original_path = ROOT / provenance["original_case"]
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == provenance["original_case_sha256"]
    original = yaml.safe_load(original_path.read_text(encoding="utf-8"))
    assert case["system"] == original["system"]
    assert case["expected"]["rows"] == original["expected"]["rows"]
    assert case["expected"]["sign_pattern_required"] is True
    assert case["expected"]["evaluation_criteria_version"] == "published_tolerance_and_required_sign_v2"
    assert provenance["original_case_modified"] is False
    integration = case["integration"]
    assert integration["execution_protocol"] == "fischer_reference_rail_v1"
    assert integration["h"] * integration["N"] == pytest.approx(integration["T_ref"])
    assert integration["h_clone"] * integration["n_clone"] == pytest.approx(integration["t_clone"])
    assert integration["K"] * integration["t_clone"] == pytest.approx(integration["T_ref"])
    assert integration["t_clone"] / integration["h"] == pytest.approx(round(integration["t_clone"] / integration["h"]))
    if case["system"]["kind"] == "jerk":
        assert (integration["t_clone"], integration["n_clone"], integration["K"]) == (10, 1000, 100)
    if case["system"]["kind"] == "financial":
        assert integration["h"] == .05
        assert integration["h_clone"] == .01
        assert integration["t_clone"] / integration["h"] == 200


def test_six_configs_keep_all_twenty_four_rows():
    cases = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in CASES.glob("*.yaml")]
    assert len(cases) == 6
    assert sum(len(case["expected"]["rows"]) for case in cases) == 24
