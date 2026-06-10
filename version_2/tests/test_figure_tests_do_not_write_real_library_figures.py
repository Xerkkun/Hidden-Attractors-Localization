# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
VERSION2_DIR = ROOT_DIR / "version_2"
TESTS_DIR = VERSION2_DIR / "tests"

# Regex pattern to match actual imports or calls to export_figure, avoiding string/regex match false positives
EXPORT_FIGURE_PATTERN = re.compile(r"(\bimport\s+.*export_figure|\bfrom\s+.*import\s+.*export_figure|\bexport_figure\s*\()")

@pytest.mark.hygiene
def test_tests_do_not_write_real_library_figures():
    """Verify that any test using export_figure redirects LIBRARY_FIGURES_ROOT."""
    violations = []
    
    test_files = list(TESTS_DIR.glob("**/*.py"))
    
    for f in test_files:
        # Ignore this file itself
        if f.name == Path(__file__).name:
            continue
            
        try:
            content = f.read_text(encoding="utf-8")
        except Exception:
            continue
            
        if EXPORT_FIGURE_PATTERN.search(content):
            # Enforce that the test uses tmp_path and monkeypatch to redirect output
            if "tmp_path" not in content or "monkeypatch" not in content:
                violations.append(f"{f.name}: uses export_figure but does not include tmp_path and monkeypatch")
                continue
                
            # Enforce that monkeypatch redirects export and manifest LIBRARY_FIGURES_ROOT
            has_export_patch = "LIBRARY_FIGURES_ROOT" in content and "monkeypatch.setattr" in content
            if not has_export_patch:
                violations.append(f"{f.name}: lacks LIBRARY_FIGURES_ROOT redirection patch")
                
    assert not violations, (
        f"Test files calling export_figure must use monkeypatch and tmp_path to redirect "
        f"LIBRARY_FIGURES_ROOT to a temporary path to avoid contaminating real figures: {violations}"
    )
