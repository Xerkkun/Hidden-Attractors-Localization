from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent
RELEASE_ROOT = VERSION_ROOT / "release_package"

EXPECTED_REMAINING_WORK = [
    "complete selected validation runs",
    "regenerate final scientific freeze audit",
    "replace sample-output templates with executed outputs if required",
]


@pytest.mark.release_readiness
def test_remaining_work_file_exists_and_is_limited_to_final_items() -> None:
    path = RELEASE_ROOT / "REMAINING_WORK.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Complete selected validation runs." in text
    assert "Regenerate the final scientific freeze audit after the evidence set is fixed." in text
    assert "absolute path" not in text.lower()
    assert "missing metadata" not in text.lower()


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_archive_manifest_known_remaining_work_is_exact() -> None:
    manifest = json.loads((RELEASE_ROOT / "archive_manifest.json").read_text(encoding="utf-8"))
    assert manifest["known_remaining_work"] == EXPECTED_REMAINING_WORK
    assert manifest["repository_readiness"] == "passed"
    assert manifest["software_package_readiness"] == "passed"
    assert manifest["final_submission_readiness"] == "pending"
    assert manifest["ci_status"] == "passed"
    assert manifest["freeze_audit_status"] == "pending_final_scientific_freeze"
    assert manifest["arctan_status"] == "implemented algebraically, pending full validation; not promoted as a validated hidden attractor"


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_cpc_readiness_strict_allows_declared_final_pending_items() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hidden_attractors.cli.main", "validate", "release-readiness", "--strict", "--json"],
        cwd=VERSION_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed_with_known_pending_items"
    assert payload["repository_readiness"] == "passed"
    assert payload["software_package_readiness"] == "passed"
    assert payload["final_submission_readiness"] == "pending"


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_cpc_readiness_submission_strict_reports_final_pending_items() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hidden_attractors.cli.main", "validate", "release-readiness", "--submission-strict", "--json"],
        cwd=VERSION_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["final_submission_readiness"] == "pending"
    pending_names = {item["name"] for item in payload["final_submission_pending"]}
    assert "final release manuscript/template work" in pending_names
    assert "remaining scientific validation" in pending_names
