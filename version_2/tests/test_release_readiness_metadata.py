from __future__ import annotations

import json
from pathlib import Path

import pytest


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent


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
        VERSION_ROOT / "release_package" / "README_RELEASE.md",
        VERSION_ROOT / "release_package" / "PROGRAM_SUMMARY.md",
        VERSION_ROOT / "release_package" / "SAMPLE_RUN.md",
        VERSION_ROOT / "release_package" / "REMAINING_WORK.md",
        VERSION_ROOT / "release_package" / "ARCTAN_C590_PROMOTION_BOUNDARY.md",
        VERSION_ROOT / "validation" / "chua_fractional_arctan" / "hiddenness_validation_summary.json",
        VERSION_ROOT / "validation" / "chua_fractional_arctan" / "hiddenness_decisions.csv",
        VERSION_ROOT / "validation" / "chua_fractional_arctan" / "run_metadata.json",
        VERSION_ROOT / "release_package" / "reproducibility_checklist.md",
        VERSION_ROOT / "release_package" / "archive_manifest.json",
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
def test_archive_manifest_records_repository_readiness_and_final_pending_state() -> None:
    manifest = json.loads((VERSION_ROOT / "release_package" / "archive_manifest.json").read_text(encoding="utf-8"))
    assert manifest["commit_status"] == "current"
    assert manifest["freeze_audit_status"] == "passed"
    assert manifest["sample_status"] == "executed"
    assert manifest["ci_status"] == "passed"
    assert "Python 3.11/3.12/3.13" in manifest["ci_status_scope"]
    assert manifest["repository_readiness"] == "passed"
    assert manifest["software_package_readiness"] == "passed"
    assert manifest["final_submission_readiness"] == "passed"
    assert manifest["claims_status"] == "finite-time numerical evidence under recorded validation contracts"
    assert manifest["version"] == "1.0.0"
    assert manifest["release_blocked_for_v1_0_0"] is False
    assert manifest["arctan_promotion_boundary"] == "version_2/release_package/ARCTAN_C590_PROMOTION_BOUNDARY.md"
    assert manifest["blocking_release_items"] == []
    assert "version_2/validation/chua_fractional_arctan/" in manifest["promoted_evidence"]
    assert "version_2/validation/chua_fractional_arctan_c590/" not in manifest["promoted_evidence"]
    assert "radii <= 0.3" in manifest["arctan_status"]
    assert "not exposed as public release CLI commands" in manifest["public_cli_scope_note"]
    assert "passed" in manifest["freeze_audit_note"]
