# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
VERSION2_DIR = ROOT_DIR / "version_2"
VALIDATION_DIR = VERSION2_DIR / "validation"

LEGACY_FIELDS = {
    "legacy_external_figures_not_promoted",
    "legacy_provenance",
    "archived_external_paths",
    "excluded_paths",
    "unpromoted_outputs",
}

BANNED_PATTERNS = [
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]"),
    re.compile(r"(^|[^A-Za-z0-9_])[\\/]Users[\\/]"),
    re.compile(r"(^|[^A-Za-z0-9_])/home/"),
    re.compile(r"(^|[\\/])Desktop([\\/]|$)"),
    re.compile(r"(^|[\\/])Downloads([\\/]|$)"),
    re.compile(r"OneDrive"),
    re.compile(r"Google Drive"),
    re.compile(r"Hidden Attractors Fractional Order[\\/]"),
]
VALIDATION_OUTPUTS_PATTERN = re.compile(r"(^|[\\/])validation_outputs([\\/]|$)|version_2[\\/]validation_outputs")


def contains_banned_pattern(val: str, *, allow_validation_outputs: bool = False) -> bool:
    if not isinstance(val, str):
        return False
    if any(p.search(val) for p in BANNED_PATTERNS):
        return True
    return bool(VALIDATION_OUTPUTS_PATTERN.search(val) and not allow_validation_outputs)


def is_policy_markdown_line(lines: list[str], idx: int) -> bool:
    current_header = ""
    for previous in lines[: idx + 1]:
        if previous.startswith("#"):
            current_header = previous.lower()
    line = lines[idx].lower()
    return any(
        term in current_header or term in line
        for term in [
            "legacy",
            "provenance",
            "non-promoted",
            "unpromoted",
            "policy",
            "local/regenerable",
            "local outputs",
            "evidence boundary",
            "freeze audit",
            "ci and freeze",
        ]
    )


def scan_json(value: Any, rel_file: str, violations: list[str], in_legacy: bool = False) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            scan_json(child, rel_file, violations, in_legacy or key in LEGACY_FIELDS)
    elif isinstance(value, list):
        for item in value:
            scan_json(item, rel_file, violations, in_legacy)
    elif isinstance(value, str) and contains_banned_pattern(value, allow_validation_outputs=in_legacy):
        violations.append(f"JSON {rel_file}: banned path in non-policy value {value!r}")


def scan_markdown(path: Path, rel_file: str, violations: list[str]) -> None:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx, line in enumerate(lines):
        if contains_banned_pattern(line, allow_validation_outputs=is_policy_markdown_line(lines, idx)):
            violations.append(f"Markdown {rel_file}:L{idx + 1}: banned path outside policy/legacy section: {line.strip()!r}")


PROMOTED_FIELDS = {
    "figures",
    "pdf_path",
    "png_path",
    "metadata_path",
    "report_figures",
    "promoted_figures",
}


@pytest.mark.hygiene
def test_no_external_figure_paths_in_validation_json() -> None:
    violations: list[str] = []
    for path in VALIDATION_DIR.glob("**/*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        scan_json(data, path.relative_to(VERSION2_DIR).as_posix(), violations)
    assert not violations, "Banned external paths found in validation JSON:\n" + "\n".join(violations[:100])


@pytest.mark.hygiene
def test_no_external_figure_paths_in_validation_md() -> None:
    violations: list[str] = []
    for path in VALIDATION_DIR.glob("**/*.md"):
        scan_markdown(path, path.relative_to(VERSION2_DIR).as_posix(), violations)
    assert not violations, "Banned external paths found outside legacy/policy markdown sections:\n" + "\n".join(violations[:100])
