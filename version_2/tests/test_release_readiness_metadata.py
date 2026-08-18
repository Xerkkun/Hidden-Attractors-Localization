from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent
PUBLICATION_STATE_PAIRS = {
    ("verified_release_candidate", "not_published"),
    ("published", "published"),
}


@pytest.mark.release_readiness
def test_submission_strict_release_validator_passes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(VERSION_ROOT / "tools" / "release" / "validate_release_readiness.py"),
            "--submission-strict",
        ],
        cwd=VERSION_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.release_readiness
def test_release_readiness_metadata_files_exist() -> None:
    required = [
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / ".zenodo.json",
        REPO_ROOT / "codemeta.json",
        REPO_ROOT / "AUTHORS.md",
        REPO_ROOT / "CHANGELOG.md",
        REPO_ROOT / "RELEASE_NOTES.md",
        REPO_ROOT / "REPRODUCIBILITY.md",
        REPO_ROOT / ".github" / "workflows" / "package.yml",
        REPO_ROOT / ".github" / "workflows" / "publish-pypi.yml",
        VERSION_ROOT / "MANIFEST.in",
        VERSION_ROOT / "MANIFEST.md",
        VERSION_ROOT / "release_package" / "README_RELEASE.md",
        VERSION_ROOT / "release_package" / "PROGRAM_SUMMARY.md",
        VERSION_ROOT / "release_package" / "SAMPLE_RUN.md",
        VERSION_ROOT / "release_package" / "PUBLISHING_POLICY.md",
        VERSION_ROOT / "release_package" / "archive_manifest.json",
        VERSION_ROOT / "release_package" / "sample_input" / "chua_integer_comprehensive.yaml",
        VERSION_ROOT / "release_package" / "sample_output" / "comprehensive_sample_summary.json",
        VERSION_ROOT / "README.md",
        VERSION_ROOT / "USER_MANUAL.md",
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    assert not missing, "Missing release readiness files:\n" + "\n".join(missing)


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_citation_records_archive_doi_without_requiring_article_doi() -> None:
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "10.17605/OSF.IO/ZGK74" in citation
    assert "archived software release" in citation
    assert "CPC" not in citation


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_archive_manifest_records_prepared_release_without_self_reference() -> None:
    manifest = json.loads(
        (VERSION_ROOT / "release_package" / "archive_manifest.json").read_text(encoding="utf-8")
    )
    with (VERSION_ROOT / "pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]

    assert version == manifest["version"] == "1.2.0"
    assert manifest["release_tag"] == f"v{version}"
    assert manifest["source_commit_policy"] == "release_tag_resolves_to_source_commit"
    assert "commit" not in manifest
    assert "commit_status" not in manifest
    assert manifest["sample_status"] == "executed"
    assert manifest["repository_readiness"] == "ready"
    assert manifest["software_package_readiness"] == "ready"
    assert manifest["release_preparation_readiness"] == "ready"
    assert (
        manifest["release_state"],
        manifest["publication_status"],
    ) in PUBLICATION_STATE_PAIRS
    assert manifest["claims_status"] == "finite-time numerical evidence under recorded validation contracts"
    assert not any(
        "remaining" in key.lower() or "blocking" in key.lower()
        for key in manifest
    )


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_archive_manifest_separates_pypi_from_scientific_archive() -> None:
    manifest = json.loads(
        (VERSION_ROOT / "release_package" / "archive_manifest.json").read_text(encoding="utf-8")
    )
    validation = manifest["scientific_validation"]
    source_archive_paths = set(manifest["source_archive_included_paths"])
    wheel_paths = set(manifest["wheel_distribution_included"])
    sdist_paths = set(manifest["sdist_distribution_included"])
    archive_only_paths = {
        "version_2/validation/",
        "version_2/release_package/",
    }

    assert validation["repository_path"] == "version_2/validation/"
    assert validation["pypi_distribution"] == "excluded"
    assert archive_only_paths <= source_archive_paths
    assert archive_only_paths.isdisjoint(wheel_paths)
    assert archive_only_paths.isdisjoint(sdist_paths)
    assert manifest["wheel_distribution_included"] == ["version_2/hidden_attractors/"]
    assert "version_2/examples/chua_integer_lure_reference/" in manifest["sdist_distribution_included"]
