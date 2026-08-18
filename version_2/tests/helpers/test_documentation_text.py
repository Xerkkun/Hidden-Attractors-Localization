# -*- coding: utf-8 -*-
from pathlib import Path
from fnmatch import fnmatch
import re

import yaml

ROOT = Path(__file__).resolve().parents[2]  # version_2 directory

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())

def active_doc_paths() -> list[Path]:
    candidates = {
        ROOT / "USER_MANUAL.md",
        ROOT / "README.md",
        ROOT / "INSTALL.md",
        ROOT / "MANIFEST.md",
        ROOT.parent / "README.md",
        ROOT / "REFERENCE_GUIDE.md",
    }

    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    excluded = {
        line.strip()
        for line in str(config.get("exclude_docs", "")).splitlines()
        if line.strip()
    }
    docs_root = ROOT / "docs"
    for path in docs_root.rglob("*.md"):
        relative = path.relative_to(docs_root).as_posix()
        if not any(fnmatch(relative, pattern) for pattern in excluded):
            candidates.add(path)

    for relative_root in ("examples", "benchmarks", "release_package"):
        base = ROOT / relative_root
        if base.is_dir():
            candidates.update(base.rglob("*.md"))

    tools_readme = ROOT / "tools" / "README.md"
    if tools_readme.is_file():
        candidates.add(tools_readme)

    return sorted(path for path in candidates if path.is_file())


def validation_evidence_doc_paths() -> list[Path]:
    """Return maintained validation Markdown, excluding immutable snapshots."""

    validation_root = ROOT / "validation"
    if not validation_root.is_dir():
        return []
    excluded_parts = {
        ("source_snapshots",),
        ("reference_cases", "archive"),
    }
    paths = []
    for path in validation_root.rglob("*.md"):
        relative_parts = path.relative_to(validation_root).parts
        if any(
            relative_parts[: len(prefix)] == prefix
            for prefix in excluded_parts
        ):
            continue
        paths.append(path)
    return sorted(paths)

def find_occurrences(text: str, phrase: str) -> list[int]:
    text_lower = text.lower()
    phrase_lower = phrase.lower()
    indices = []
    start = 0
    while True:
        pos = text_lower.find(phrase_lower, start)
        if pos == -1:
            break
        indices.append(pos)
        start = pos + len(phrase)
    return indices

def get_violations_without_context(text: str, phrase: str, context_terms: list[str], window: int = 160) -> list[str]:
    violations = []
    text_lower = text.lower()
    phrase_lower = phrase.lower()
    start = 0
    while True:
        pos = text_lower.find(phrase_lower, start)
        if pos == -1:
            break
        win_start = max(0, pos - window)
        win_end = min(len(text), pos + len(phrase) + window)
        sub_window = text_lower[win_start:win_end]
        if not any(term.lower() in sub_window for term in context_terms):
            line_num = text[:pos].count('\n') + 1
            line_start = text.rfind('\n', 0, pos) + 1
            line_end = text.find('\n', pos)
            if line_end == -1:
                line_end = len(text)
            line_content = text[line_start:line_end].strip()
            violations.append(f"L{line_num}: '{line_content}' (window lacked context keywords)")
        start = pos + len(phrase)
    return violations
