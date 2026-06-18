from __future__ import annotations

import json
from pathlib import Path

import pytest


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent
CPC_ROOT = VERSION_ROOT / "cpc_submission"


@pytest.mark.hygiene
@pytest.mark.cpc_readiness
def test_cpc_sample_input_output_are_populated() -> None:
    sample_input = CPC_ROOT / "sample_input"
    sample_output = CPC_ROOT / "sample_output"

    assert (sample_input / "README.md").exists()
    assert list(sample_input.glob("*.yaml")), "sample_input must contain at least one YAML file"
    assert (sample_output / "README.md").exists()
    assert list(sample_output.glob("*.json")), "sample_output must contain at least one JSON file"


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

    assert manifest["sample_status"] in {"template_only_pending_execution", "executed"}
