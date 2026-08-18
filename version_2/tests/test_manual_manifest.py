from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT_DIR / "docs" / "manual_manifest.yaml"


def load_manifest() -> dict:
    assert MANIFEST_PATH.is_file(), f"Public manual manifest not found at {MANIFEST_PATH}"
    data = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "docs/manual_manifest.yaml must contain a mapping"
    return data


@pytest.mark.hygiene
def test_manual_manifest_has_minimal_public_schema() -> None:
    data = load_manifest()

    required_keys = {
        "manual_version",
        "package_version",
        "public_cli",
        "entry_point",
        "manual_targets",
        "scientific_scope",
        "publication_boundary",
        "documentation_policy",
    }
    assert required_keys <= data.keys()

    private_or_stale_keys = {
        "freeze_audit",
        "claims_source",
        "canonical_figures",
        "claim_status_summary",
        "forbidden_public_claims",
        "public_evidence_labels",
    }
    assert private_or_stale_keys.isdisjoint(data)


@pytest.mark.hygiene
def test_manual_manifest_public_values_are_consistent() -> None:
    data = load_manifest()

    assert data["manual_version"] == "1.2.0"
    assert data["package_version"] == "1.2.0"
    assert data["public_cli"] == "hidden-attractors"
    assert data["entry_point"] == "hidden_attractors.cli.main:main"

    targets = data["manual_targets"]
    assert (ROOT_DIR / targets["user_manual"]).is_file()
    assert (ROOT_DIR / targets["documentation_root"]).is_dir()

    scope = data["scientific_scope"]
    assert isinstance(scope["public_uses"], list)
    assert len(scope["public_uses"]) >= 2
    assert "finite-data" in scope["evidence_boundary"]
    assert "not a global mathematical proof" in scope["hiddenness_boundary"]

    boundary = data["publication_boundary"]
    assert "excluded from PyPI distributions" in boundary["validation_records"]
    assert "excluded from the public documentation site" in boundary["detailed_reports"]

    policy = data["documentation_policy"]
    assert "supported interfaces" in policy["public_content"]
    assert "forward-looking plans" in policy["excluded_content"]
    assert policy["version_source"] == "pyproject.toml"
