from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path

import pytest


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent

IGNORED_LOCAL_PREFIXES = [
    "DF y NC Chua entero y fraccionario/",
    "DF y NC Chua entero y fraccionario copy/",
    "version_2/validation_outputs/",
    "version_2/outputs/",
    "version_2/figures/",
    "version_2/runs",
    ".pytest_tmp/",
    ".runtime_cache/",
    "figures/",
    "outputs/",
    "runs",
    "paper/",
]

PROHIBITED_ROOT_PATTERNS = [
    "scratch_*.py",
    "step*.py",
    "generate_*plots*.py",
    "generate_*figures*.py",
    "search_*candidates*.py",
    "compare_*solvers*.py",
    "*_old.py",
    "*_backup.py",
    "*_tmp.py",
]

ALLOWED_TOP_LEVEL = {
    ".github",
    ".gitignore",
    "README.md",
    "version_2",
    "CITATION.cff",
    ".zenodo.json",
    "codemeta.json",
    "AUTHORS.md",
    "CHANGELOG.md",
    "RELEASE_NOTES.md",
    "REPRODUCIBILITY.md",
}


def git_ls_files(*patterns: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", *patterns],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


@pytest.mark.hygiene
def test_no_tracked_files_in_local_ignored_directories() -> None:
    tracked = git_ls_files()
    violations = []
    for path in tracked:
        for prefix in IGNORED_LOCAL_PREFIXES:
            if path.startswith(prefix):
                violations.append(path)
                break
    assert not violations, "Tracked local/regenerable files:\n" + "\n".join(violations)


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_validation_outputs_has_no_tracked_files() -> None:
    assert git_ls_files("version_2/validation_outputs") == []


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_paper_directory_is_local_only() -> None:
    assert git_ls_files("paper") == []
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/paper/" in text


@pytest.mark.hygiene
@pytest.mark.release_readiness
def test_gitignore_keeps_local_outputs_ignored_but_validation_promoted() -> None:
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for line in [
        "/version_2/validation_outputs/",
        "/version_2/outputs/",
        "/version_2/runs*/",
        "/version_2/figures/",
    ]:
        assert line in text
    assert "/version_2/validation/" not in text


@pytest.mark.hygiene
def test_no_active_scratch_scripts_at_repository_root() -> None:
    root_files = [Path(path).name for path in git_ls_files() if "/" not in path]
    violations = [
        name
        for name in root_files
        for pattern in PROHIBITED_ROOT_PATTERNS
        if fnmatch.fnmatch(name, pattern)
    ]
    assert not violations, "Tracked scratch/temporary scripts at root:\n" + "\n".join(sorted(set(violations)))


@pytest.mark.hygiene
def test_root_top_level_paths_are_canonical() -> None:
    top_levels = {path.split("/", 1)[0] for path in git_ls_files()}
    violations = sorted(top_levels - ALLOWED_TOP_LEVEL)
    assert not violations, "Unexpected tracked top-level paths:\n" + "\n".join(violations)
