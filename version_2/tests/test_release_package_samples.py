from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent
RELEASE_ROOT = VERSION_ROOT / "release_package"


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_release_sample_input_output_are_populated() -> None:
    sample_input = RELEASE_ROOT / "sample_input"
    sample_output = RELEASE_ROOT / "sample_output"

    assert (sample_input / "README.md").exists()
    yaml_files = list(sample_input.glob("*.yaml"))
    assert yaml_files == [sample_input / "chua_integer_comprehensive.yaml"]
    assert (sample_output / "README.md").exists()
    assert list(sample_output.glob("*.json")) == [sample_output / "comprehensive_sample_summary.json"]


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_release_sample_inputs_write_only_to_ignored_sample_outputs() -> None:
    for path in (RELEASE_ROOT / "sample_input").glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        output_dir = data["outputs"]["output_dir"].replace("\\", "/")
        figures_dir = data["outputs"]["figures_dir"].replace("\\", "/")
        seed_search = data["seed_search"]
        assert seed_search["route"] == "direct_integer_transfer"
        assert seed_search["fallback_route"] is None
        assert "nscan" not in seed_search
        assert output_dir.startswith("outputs/release_samples/"), output_dir
        assert figures_dir.startswith(output_dir + "/"), figures_dir
        assert "validation/" not in output_dir
        assert "library_figures" not in output_dir


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_archive_manifest_references_release_samples() -> None:
    manifest = json.loads((RELEASE_ROOT / "archive_manifest.json").read_text(encoding="utf-8"))
    for key in ("sample_input", "sample_output"):
        assert key in manifest
        assert isinstance(manifest[key], list)
        assert manifest[key], f"{key} must not be empty"
        missing = [rel for rel in manifest[key] if not (REPO_ROOT / rel).exists()]
        assert not missing, f"Missing sample paths in archive manifest: {missing}"

    assert "version_2/release_package/sample_input/chua_integer_comprehensive.yaml" in manifest["sample_input"]
    assert "version_2/release_package/sample_output/comprehensive_sample_summary.json" in manifest["sample_output"]
    assert manifest["sample_status"] == "executed"


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_expected_sample_outputs_are_executed_but_not_promoted_evidence() -> None:
    for path in (RELEASE_ROOT / "sample_output").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data.get("not_promoted_evidence") is True
        assert data.get("replace_after_execution") is False
        assert data.get("sample_status") == "executed"
        assert data.get("release_version") == "1.2.0"
        assert data["repeatability_check"]["independent_runs"] >= 2
        assert data["repeatability_check"]["deterministic_outputs_identical"] is True
        assert data["deterministic_output_hashes"]
