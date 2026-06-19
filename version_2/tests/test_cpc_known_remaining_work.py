from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent
CPC_ROOT = VERSION_ROOT / "cpc_submission"

EXPECTED_REMAINING_WORK = [
    "final CPC manuscript writing in official Elsevier/CPC template",
    "fractional arctan Chua validation runs",
    "additional reproducible scientific examples selected for the paper",
    "final scientific freeze audit regeneration after scientific choices are frozen",
    "replace sample-output templates with executed outputs if required by the final CPC package",
]


@pytest.mark.hygiene
@pytest.mark.cpc_readiness
def test_remaining_work_file_exists_and_is_limited_to_final_items() -> None:
    path = CPC_ROOT / "REMAINING_WORK.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Write the final manuscript using the official CPC/Elsevier template." in text
    assert "Run and assess full fractional arctan Chua validation." in text
    assert "Regenerate the full scientific freeze audit on the final commit." in text
    assert "absolute path" not in text.lower()
    assert "missing metadata" not in text.lower()


@pytest.mark.hygiene
@pytest.mark.cpc_readiness
def test_archive_manifest_known_remaining_work_is_exact() -> None:
    manifest = json.loads((CPC_ROOT / "archive_manifest.json").read_text(encoding="utf-8"))
    assert manifest["known_remaining_work"] == EXPECTED_REMAINING_WORK
    assert manifest["repository_readiness"] == "passed"
    assert manifest["software_package_readiness"] == "passed"
    assert manifest["final_submission_readiness"] == "pending"
    assert manifest["ci_status"] == "passed"
    assert manifest["freeze_audit_status"] == "pending_final_scientific_freeze"
    assert manifest["arctan_status"] == "implemented algebraically, pending full validation; not promoted as a validated hidden attractor"


@pytest.mark.hygiene
@pytest.mark.cpc_readiness
def test_cpc_readiness_strict_allows_declared_final_pending_items() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hidden_attractors.cli.main", "validate", "cpc-readiness", "--strict", "--json"],
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
@pytest.mark.cpc_readiness
def test_cpc_readiness_submission_strict_reports_final_pending_items() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hidden_attractors.cli.main", "validate", "cpc-readiness", "--submission-strict", "--json"],
        cwd=VERSION_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["final_submission_readiness"] == "pending"
    pending_names = {item["name"] for item in payload["final_submission_pending"]}
    assert "final CPC manuscript/template work" in pending_names
    assert "remaining scientific validation" in pending_names
