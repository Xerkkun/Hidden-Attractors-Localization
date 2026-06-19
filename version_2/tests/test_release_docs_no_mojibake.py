from __future__ import annotations

from pathlib import Path

import pytest


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent

MOJIBAKE_PATTERNS = [
    "\u00c3\u0192",   # Ãƒ
    "\u00c3\u201a",   # Ã‚
    "\u00c3",
    "\u00c2",
    "\u00e2\u20ac",
    "\u00e2\u20ac\u201d",
    "\u00e2\u20ac\u2122",
    "\u00e2\u20ac\u0153",
    "\u00e2\u20ac\u009d",
    "\u00e2\u201d",
]

SCAN_PATTERNS = [
    "README.md",
    "CHANGELOG.md",
    "RELEASE_NOTES.md",
    "REPRODUCIBILITY.md",
    "CITATION.cff",
    ".zenodo.json",
    "codemeta.json",
    "version_2/README.md",
    "version_2/USER_MANUAL.md",
    "version_2/MANIFEST.md",
    "version_2/pyproject.toml",
    "version_2/docs/*.md",
    "version_2/release_package/*.md",
    "version_2/release_package/*.json",
    "version_2/release_package/sample_input/*.yaml",
    "version_2/release_package/sample_input/*.md",
    "version_2/release_package/sample_output/*.json",
    "version_2/release_package/sample_output/*.md",
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
@pytest.mark.release_readiness
def test_main_release_text_files_have_no_mojibake() -> None:
    violations: list[str] = []
    for path in iter_files():
        text = path.read_text(encoding="utf-8")
        for pattern in MOJIBAKE_PATTERNS:
            if pattern in text:
                violations.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {pattern!r}")
                break
    assert not violations, "Mojibake found in release/main documentation files:\n" + "\n".join(violations)
