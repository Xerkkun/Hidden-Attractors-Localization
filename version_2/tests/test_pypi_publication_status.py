from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent
PYPI_URL = "https://pypi.org/project/hidden-attractors-fo/"
PUBLICATION_STATE_PAIRS = {
    ("verified_release_candidate", "not_published"),
    ("published", "published"),
}


def _manifest() -> dict:
    return json.loads(
        (VERSION_ROOT / "release_package" / "archive_manifest.json").read_text(encoding="utf-8")
    )


def _project_version() -> str:
    with (VERSION_ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def test_archive_manifest_records_a_closed_coherent_publication_state() -> None:
    manifest = _manifest()
    readiness = manifest["pypi_readiness"]
    version = _project_version()

    assert version == "1.1.0"
    assert readiness["package_name"] == "hidden-attractors-fo"
    assert readiness["version"] == readiness["target_version"] == manifest["version"] == version
    assert manifest["release_tag"] == f"v{version}"
    state_pair = (manifest["release_state"], manifest["publication_status"])
    assert state_pair in PUBLICATION_STATE_PAIRS
    assert readiness["publication_status"] == manifest["publication_status"]
    assert readiness["local_release_candidate_verification"] == "passed"
    if manifest["publication_status"] == "not_published":
        assert readiness["current_public_version"]
        assert readiness["current_public_version"] != readiness["target_version"]
    else:
        assert readiness["current_public_version"] == readiness["target_version"]
    assert readiness["pypi_url"] == PYPI_URL
    assert readiness["authentication"] == "trusted_publishing_oidc"
    assert readiness["publication_gate"] == "pypi_environment"


def test_release_docs_match_the_recorded_publication_state() -> None:
    manifest = _manifest()
    docs = [
        VERSION_ROOT / "release_package" / "README_RELEASE.md",
        VERSION_ROOT / "release_package" / "PUBLISHING_POLICY.md",
        VERSION_ROOT / "release_package" / "PROGRAM_SUMMARY.md",
        VERSION_ROOT / "release_package" / "SAMPLE_RUN.md",
    ]
    false_claims = (
        "1.1.0 is published",
        "version 1.1.0 is published",
        "pypi status: published",
        '"publication_status": "published"',
    )
    for path in docs:
        text = path.read_text(encoding="utf-8").lower()
        if manifest["publication_status"] == "not_published":
            assert not any(claim in text for claim in false_claims), path

    release_readme = re.sub(r"\s+", " ", docs[0].read_text(encoding="utf-8").lower())
    if manifest["publication_status"] == "not_published":
        assert "locally verified release candidate" in release_readme
    else:
        assert "locally verified release candidate" not in release_readme
    assert "python -m pip install hidden-attractors-fo" in release_readme


def test_publish_workflow_uses_guarded_ref_protected_environment_and_scoped_oidc() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml").read_text(encoding="utf-8")

    assert "Require an authorized version-matched release ref" in workflow
    assert 'refs/heads/release/v1.1.0' in workflow
    assert 'version != "1.1.0"' in workflow
    assert 'git rev-parse origin/main' in workflow
    assert "environment:\n      name: pypi" in workflow
    assert workflow.count("id-token: write") == 1

    verify_text, publish_text = workflow.split("\n  publish:", maxsplit=1)
    assert "id-token: write" not in verify_text
    assert "id-token: write" in publish_text
    assert "pypa/gh-action-pypi-publish@release/v1" in publish_text
