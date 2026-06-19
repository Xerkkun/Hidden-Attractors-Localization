from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest


VERSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERSION_ROOT.parent

SCAN_PATTERNS = [
    "version_2/validation/**/*.json",
    "version_2/validation/**/*.md",
    "version_2/docs/**/*.md",
    "version_2/release_package/**/*.md",
    "version_2/release_package/**/*.json",
]

JSON_POLICY_KEYS = {
    "legacy_provenance",
    "archived_external_paths",
    "legacy_external_figures_not_promoted",
    "excluded_paths",
    "unpromoted_outputs",
}

LOCAL_PATH_REGEXES = [
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]"),
    re.compile(r"(^|[^A-Za-z0-9_])[\\/]Users[\\/]"),
    re.compile(r"(^|[^A-Za-z0-9_])/home/"),
    re.compile(r"(^|[\\/])Desktop([\\/]|$)"),
    re.compile(r"(^|[\\/])Downloads([\\/]|$)"),
    re.compile(r"OneDrive"),
    re.compile(r"Google Drive"),
]
VALIDATION_OUTPUTS_REGEX = re.compile(r"(^|[\\/])validation_outputs([\\/]|$)|version_2[\\/]validation_outputs")
PROJECT_NAME_PATH_REGEX = re.compile(r"Hidden Attractors Fractional Order[\\/]")


def _is_policy_markdown_line(lines: list[str], index: int) -> bool:
    current_header = ""
    for previous in lines[: index + 1]:
        if previous.startswith("#"):
            current_header = previous.lower()
    line = lines[index].lower()
    return any(
        term in current_header or term in line
        for term in [
            "policy",
            "evidence boundary",
            "local/regenerable",
            "local outputs",
            "unpromoted",
            "non-promoted",
            "legacy",
            "freeze audit",
            "ci and freeze",
        ]
    )


def _bad_string(value: str, *, allow_validation_outputs: bool) -> bool:
    if any(regex.search(value) for regex in LOCAL_PATH_REGEXES):
        return True
    if PROJECT_NAME_PATH_REGEX.search(value):
        return True
    if VALIDATION_OUTPUTS_REGEX.search(value) and not allow_validation_outputs:
        return True
    return False


def _scan_json(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    hits: list[str] = []

    def walk(value: Any, keys: tuple[str, ...] = (), policy_context: bool = False) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, (*keys, str(key)), policy_context or key in JSON_POLICY_KEYS)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                walk(child, (*keys, str(idx)), policy_context)
        elif isinstance(value, str) and _bad_string(value, allow_validation_outputs=policy_context):
            hits.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{'.'.join(keys)}: {value}")

    walk(data)
    return hits


def _scan_text(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    hits = []
    for idx, line in enumerate(lines):
        allow_validation_outputs = path.suffix.lower() == ".md" and _is_policy_markdown_line(lines, idx)
        if _bad_string(line, allow_validation_outputs=allow_validation_outputs):
            hits.append(f"{path.relative_to(REPO_ROOT).as_posix()}:L{idx + 1}: {line.strip()}")
    return hits


@pytest.mark.release_readiness
def test_no_absolute_paths_in_promoted_evidence() -> None:
    violations: list[str] = []
    seen: set[Path] = set()
    for pattern in SCAN_PATTERNS:
        for path in REPO_ROOT.glob(pattern):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            if path.suffix.lower() == ".json":
                violations.extend(_scan_json(path))
            elif path.suffix.lower() in {".md", ".tex", ".bib"}:
                violations.extend(_scan_text(path))
    assert not violations, "Local absolute or active validation_outputs paths found:\n" + "\n".join(violations[:100])
