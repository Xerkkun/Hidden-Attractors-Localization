from __future__ import annotations

import json
from pathlib import Path


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent
PYPI_URL = "https://pypi.org/project/hidden-attractors-fo/"


def test_archive_manifest_records_public_pypi_release() -> None:
    manifest = json.loads((VERSION_ROOT / "release_package" / "archive_manifest.json").read_text(encoding="utf-8"))
    readiness = manifest["pypi_readiness"]

    assert readiness["package_name"] == "hidden-attractors-fo"
    assert readiness["version"] == manifest["version"] == "1.0.0"
    assert readiness["pypi_status"] == "published"
    assert readiness["published_version"] == "1.0.0"
    assert readiness["pypi_url"] == PYPI_URL
    assert readiness["testpypi_status"] == "passed"


def test_primary_docs_match_pypi_publication_state() -> None:
    docs = [
        REPO_ROOT / "README.md",
        VERSION_ROOT / "README.md",
        VERSION_ROOT / "USER_MANUAL.md",
        VERSION_ROOT / "docs" / "quick_start.md",
        VERSION_ROOT / "docs" / "getting_started.md",
        VERSION_ROOT / "docs" / "installation.md",
        VERSION_ROOT / "release_package" / "README_RELEASE.md",
        VERSION_ROOT / "release_package" / "PYPI_RELEASE_REPORT.md",
    ]
    pending_terms = ("not_uploaded_by_repository", "manual_pending", "not yet published", "planned pypi")
    for path in docs:
        text = path.read_text(encoding="utf-8").lower()
        assert "python -m pip install hidden-attractors-fo" in text, path
        assert not any(term in text for term in pending_terms), path

    report = (VERSION_ROOT / "release_package" / "PYPI_RELEASE_REPORT.md").read_text(encoding="utf-8")
    assert PYPI_URL in report
    assert "**PyPI status**: published" in report