from __future__ import annotations

from pathlib import Path

import pytest


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent


@pytest.mark.hygiene
@pytest.mark.cpc_readiness
def test_cpc_readiness_metadata_files_exist() -> None:
    required = [
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / ".zenodo.json",
        REPO_ROOT / "codemeta.json",
        REPO_ROOT / "AUTHORS.md",
        REPO_ROOT / "CHANGELOG.md",
        REPO_ROOT / "RELEASE_NOTES.md",
        REPO_ROOT / "REPRODUCIBILITY.md",
        REPO_ROOT / "paper" / "cpc_program_summary.tex",
        REPO_ROOT / "paper" / "cpc_manuscript.tex",
        REPO_ROOT / "paper" / "references.bib",
        VERSION_ROOT / "cpc_submission" / "README_CPC.md",
        VERSION_ROOT / "cpc_submission" / "PROGRAM_SUMMARY.md",
        VERSION_ROOT / "cpc_submission" / "SAMPLE_RUN.md",
        VERSION_ROOT / "cpc_submission" / "reproducibility_checklist.md",
        VERSION_ROOT / "cpc_submission" / "archive_manifest.json",
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    assert not missing, "Missing CPC readiness files:\n" + "\n".join(missing)


@pytest.mark.hygiene
@pytest.mark.cpc_readiness
def test_citation_records_archive_doi_without_requiring_article_doi() -> None:
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "10.17605/OSF.IO/ZGK74" in citation
    assert "associated CPC article when available" in citation
