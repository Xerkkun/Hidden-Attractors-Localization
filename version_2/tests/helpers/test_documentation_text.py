# -*- coding: utf-8 -*-
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]  # version_2 directory

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())

def active_doc_paths() -> list[Path]:
    candidates = [
        ROOT / "USER_MANUAL.md",
        ROOT / "README.md",
        ROOT.parent / "README.md",
        ROOT / "REFERENCE_GUIDE.md",
        ROOT / "THESIS_CLAIMS.md",
        ROOT / "INSTALL.md",
        ROOT / "docs/quick_start.md",
        ROOT / "docs/installation.md",
        ROOT / "docs/testing.md",
        ROOT / "docs/validation_evidence.md",
        ROOT / "docs/unified_report.md",
        ROOT / "docs/figure_export_policy.md",
        ROOT / "docs/manual_manifest.md",
        ROOT / "docs/manual_manifest.yaml",
        ROOT / "docs/reporte_unificado_chua_fraccionario.tex",
        ROOT / "docs/getting_started.md",
        ROOT / "docs/workflows.md",
        ROOT / "docs/examples_index.md",
        ROOT / "docs/index.md",
        ROOT / "docs/code_reference_map.md",
        ROOT / "docs/repository_layout.md",
        ROOT / "docs/library_structure.md",
        ROOT / "docs/contributing.md",
        ROOT / "docs/lure_candidate_route.md",
        ROOT / "tools/cli/README.md",
        ROOT / "tools/legacy/README.md",
        ROOT / "release_package/README_RELEASE.md",
        ROOT / "release_package/SAMPLE_RUN.md",
        ROOT / "release_package/reproducibility_checklist.md",
    ]
    return [p for p in candidates if p.exists()]

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
