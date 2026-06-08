from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation" / "00_manifest" / "validation_manifest.json"
SELECTION_MANIFEST = ROOT / "validation" / "00_manifest" / "candidate_selection_manifest.json"
SELECTION_YAML = ROOT / "validation" / "candidate_selection" / "danca2017_nearby_saturation_candidate.yaml"
METHOD_COMPARISON = (
    ROOT
    / "validation"
    / "outputs"
    / "published_continuation_comparison"
    / "danca2017_chua_fractional_saturation_candidate"
    / "candidate_method_comparison.json"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_danca_published_parameters_remain_reference_parameters() -> None:
    manifest = _load_json(MANIFEST)

    assert manifest["main_parameters"]["m0"] == -0.1768
    assert manifest["main_parameters"]["m1"] == -1.1468
    assert manifest["candidate_selection"]["reference_parameters_policy"] == (
        "preserve_published_danca_parameters_as_reference"
    )
    assert manifest["candidate_selection"]["candidate_parameters"]["m0"] == -0.2
    assert manifest["candidate_selection"]["candidate_parameters"]["m1"] == -1.2


def test_candidate_selection_manifest_documents_parameter_change() -> None:
    selection = _load_json(SELECTION_MANIFEST)

    assert selection["reference_case"]["parameters"]["m0"] == -0.1768
    assert selection["reference_case"]["parameters"]["m1"] == -1.1468
    assert selection["selected_candidate"]["parameters"]["m0"] == -0.2
    assert selection["selected_candidate"]["parameters"]["m1"] == -1.2
    assert selection["parameter_change_procedure"]["fixed_parameters"] == {
        "alpha": 8.4562,
        "beta": 12.0732,
        "gamma": 0.0052,
        "q": 0.9998,
    }
    assert selection["selected_candidate"]["fractional_df"]["q_seed"] == 0.9998
    assert "hiddenness_verified" in selection["selected_candidate"]["not_claimed"]


def test_candidate_selection_yaml_is_not_a_published_reproduction_claim() -> None:
    data = yaml.safe_load(SELECTION_YAML.read_text(encoding="utf-8"))

    assert data["published_reference"]["exact_parameters"]["m0"] == -0.1768
    assert data["published_reference"]["exact_parameters"]["m1"] == -1.1468
    assert data["parameter_change_procedure"]["selected_pair"] == {"m0": -0.2, "m1": -1.2}
    assert data["proposed_method"]["q_seed"] == 0.9998
    assert data["article_style_control"]["q_seed"] == 1.0
    assert data["no_claims"]["hiddenness_certified_by_this_pipeline"] is False
    assert data["no_claims"]["chaos_certified_by_this_pipeline"] is False
    assert data["no_claims"]["exact_published_danca_reproduced"] is False


def test_method_comparison_artifact_preserves_proposed_and_control_routes() -> None:
    comparison = _load_json(METHOD_COMPARISON)

    assert comparison["parameters"]["m0"] == -0.2
    assert comparison["parameters"]["m1"] == -1.2
    assert comparison["proposed_fractional_method"]["seed"]["q_seed"] == 0.9998
    assert comparison["proposed_fractional_method"]["continuation"]["integrator"] == "ABM"
    assert comparison["proposed_fractional_method"]["final_simulation"]["hiddenness_tests"] == "not_run"
    assert comparison["proposed_fractional_method"]["final_simulation"]["lyapunov_tests"] == "not_run"
    assert comparison["article_style_integer_comparison"]["seed"]["q_seed"] == 1.0
    assert comparison["article_style_integer_comparison"]["continuation"]["derivative"] == "integer_ode"
