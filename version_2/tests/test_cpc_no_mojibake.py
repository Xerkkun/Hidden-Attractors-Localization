from __future__ import annotations

from pathlib import Path

import pytest


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent

MOJIBAKE_PATTERNS = [
    "Ãƒ",
    "Ã‚",
    "Ã¢â€\x9dâ‚¬",
    "Ã¢â‚¬â€œ",
    "Ã¢â‚¬â„¢",
    "Ă",
    "â€œ",
    "â€",
    "â”",
]

SCAN_PATTERNS = [
    "README.md",
    "CHANGELOG.md",
    "RELEASE_NOTES.md",
    "REPRODUCIBILITY.md",
    "CITATION.cff",
    ".zenodo.json",
    "codemeta.json",
    "paper/*.tex",
    "paper/*.md",
    "paper/*.bib",
    "version_2/README.md",
    "version_2/USER_MANUAL.md",
    "version_2/MANIFEST.md",
    "version_2/pyproject.toml",
    "version_2/docs/*.md",
    "version_2/cpc_submission/*.md",
    "version_2/cpc_submission/*.json",
    "version_2/cpc_submission/sample_input/*.yaml",
    "version_2/cpc_submission/sample_input/*.md",
    "version_2/cpc_submission/sample_output/*.json",
    "version_2/cpc_submission/sample_output/*.md",
]


def iter_files() -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in SCAN_PATTERNS:
        for path in REPO_ROOT.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    return files


@pytest.mark.hygiene
@pytest.mark.cpc_readiness
def test_main_cpc_text_files_have_no_mojibake() -> None:
    violations: list[str] = []
    for path in iter_files():
        text = path.read_text(encoding="utf-8")
        for pattern in MOJIBAKE_PATTERNS:
            if pattern in text:
                violations.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {pattern!r}")
                break
    assert not violations, "Mojibake found in CPC/main documentation files:\n" + "\n".join(violations)
