from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent
CPC_ROOT = VERSION_ROOT / "cpc_submission"


@pytest.mark.hygiene
@pytest.mark.cpc_readiness
def test_cpc_sample_input_output_are_populated() -> None:
    sample_input = CPC_ROOT / "sample_input"
    sample_output = CPC_ROOT / "sample_output"

    assert (sample_input / "README.md").exists()
    yaml_files = list(sample_input.glob("*.yaml"))
    assert yaml_files, "sample_input must contain at least one YAML file"
    assert (sample_output / "README.md").exists()
    assert (sample_output / "expected_cli_help_summary.json").exists()
    assert list(sample_output.glob("*.json")), "sample_output must contain at least one JSON file"


@pytest.mark.hygiene
@pytest.mark.cpc_readiness
def test_cpc_sample_inputs_write_only_to_ignored_sample_outputs() -> None:
    for path in (CPC_ROOT / "sample_input").glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        output_dir = data["experiment"]["output_dir"].replace("\\", "/")
        assert output_dir.startswith("outputs/cpc_samples/"), output_dir
        assert "validation/" not in output_dir
        assert "library_figures" not in output_dir
        assert data.get("plots", {}).get("save_figures") is False


@pytest.mark.hygiene
@pytest.mark.cpc_readiness
def test_archive_manifest_references_cpc_samples() -> None:
    manifest = json.loads((CPC_ROOT / "archive_manifest.json").read_text(encoding="utf-8"))
    for key in ("sample_input", "sample_output"):
        assert key in manifest
        assert isinstance(manifest[key], list)
        assert manifest[key], f"{key} must not be empty"
        missing = [rel for rel in manifest[key] if not (REPO_ROOT / rel).exists()]
        assert not missing, f"Missing sample paths in archive manifest: {missing}"

    assert "version_2/cpc_submission/sample_output/expected_cli_help_summary.json" in manifest["sample_output"]
    assert manifest["sample_status"] == "template_only_pending_execution"


@pytest.mark.hygiene
@pytest.mark.cpc_readiness
def test_expected_sample_outputs_are_marked_templates_not_evidence() -> None:
    for path in (CPC_ROOT / "sample_output").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("not_promoted_evidence") is True or path.name == "expected_validation_contract_status.json"
        if path.name != "expected_validation_contract_status.json":
            assert data.get("replace_after_execution") is True
