# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory

BANNED_PHRASES = [
    r"156\s+tests",
    r"156\s+unit\s+tests",
    r"156\s+pruebas",
]

@pytest.mark.hygiene
def test_no_outdated_test_count_mentions():
    """Fail if any documentation file mentions the outdated 156 test count."""
    markdown_files = list(ROOT_DIR.glob("**/*.md")) + [ROOT_DIR / "README.md", ROOT_DIR / "REFERENCE_GUIDE.md"]
    markdown_files = list(set(f for f in markdown_files if f.exists()))
    
    violations = []
    
    # Compile regexes for banned phrases
    compiled_banned = [re.compile(p, re.IGNORECASE) for p in BANNED_PHRASES]
    
    for f in markdown_files:
        # Skip legacy directories
        if "tools/legacy" in f.as_posix():
            continue
            
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
            
        for line_num, line in enumerate(content.splitlines(), 1):
            for rx in compiled_banned:
                if rx.search(line):
                    rel_path = f.relative_to(ROOT_DIR).as_posix()
                    violations.append(f"{rel_path}:L{line_num} -> '{line.strip()}'")
                    
    assert not violations, (
        "Outdated test counts (156) found in documentation files:\n"
        + "\n".join(violations)
    )

@pytest.mark.hygiene
def test_documentation_references_freeze_audit():
    """Verify that key documents reference the freeze audit directory or the current counts."""
    # We check README.md, REFERENCE_GUIDE.md and INSTALL.md
    files_to_check = [
        ROOT_DIR / "README.md",
        ROOT_DIR / "REFERENCE_GUIDE.md",
        ROOT_DIR / "INSTALL.md",
    ]
    
    missing_references = []
    
    # Expected references (at least one of these should be present in each file)
    expected_patterns = [
        re.compile(r"validation/freeze_audit", re.IGNORECASE),
        re.compile(r"797\s+passed", re.IGNORECASE),
        re.compile(r"34\s+skipped", re.IGNORECASE),
    ]
    
    for f in files_to_check:
        assert f.exists(), f"Required document {f.name} does not exist."
        content = f.read_text(encoding="utf-8")
        
        has_reference = any(p.search(content) for p in expected_patterns)
        if not has_reference:
            rel_path = f.relative_to(ROOT_DIR).as_posix()
            missing_references.append(rel_path)
            
    assert not missing_references, (
        "The following files do not refer to the freeze audit or current test counts:\n"
        + "\n".join(missing_references)
    )
