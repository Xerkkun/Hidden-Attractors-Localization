# -*- coding: utf-8 -*-
import pytest
import re
from pathlib import Path
from tests.helpers.test_documentation_text import read, ROOT, active_doc_paths
from tests.test_manual_manifest import load_manifest

OBSOLETE_COUNT_KEYWORDS = [
    "156 tests",
    "156 unit tests",
    "156 pruebas",
    "suite de 156",
    "156 passed",
]

@pytest.mark.hygiene
def test_manual_freeze_audit_reference_verification():
    # 1. Obsolete counts check
    docs = active_doc_paths()
    violations = []
    
    for p in docs:
        content = read(p)
        content_lower = content.lower()
        for kw in OBSOLETE_COUNT_KEYWORDS:
            if kw in content_lower:
                violations.append(f"{p.name} -> Mentions obsolete test count keyword '{kw}'")
                
    # 2. Require validation/freeze_audit/ in USER_MANUAL.md, README.md, REFERENCE_GUIDE.md
    primary_files = [
        ROOT / "USER_MANUAL.md",
        ROOT / "README.md",
        ROOT.parent / "README.md",
        ROOT / "REFERENCE_GUIDE.md",
    ]
    # Filter only those that exist
    existing_primary = [p for p in primary_files if p.exists()]
    
    # We must have at least USER_MANUAL.md, REFERENCE_GUIDE.md, and at least one README.md
    # Let's check existence and content of each of the primary files
    for p in [ROOT / "USER_MANUAL.md", ROOT / "REFERENCE_GUIDE.md"]:
        assert p.exists(), f"Primary document '{p.name}' is missing"
        content = read(p)
        assert "validation/freeze_audit/" in content, f"Primary document '{p.name}' does not reference 'validation/freeze_audit/'"
        
    readme_found = False
    for p in [ROOT / "README.md", ROOT.parent / "README.md"]:
        if p.exists():
            content = read(p)
            if "validation/freeze_audit/" in content:
                readme_found = True
    assert readme_found, "Neither version_2/README.md nor workspace README.md references 'validation/freeze_audit/'"
    
    # 3. Context around 944 / 28 skipped
    # If a document mentions 944 or 28 skipped, it must mention validation/freeze_audit or freeze audit within ±160 chars
    for p in docs:
        content = read(p)
        content_lower = content.lower()
        
        # Check '944'
        for match in re.finditer(r"\b944\b", content_lower):
            pos = match.start()
            win_start = max(0, pos - 160)
            win_end = min(len(content), pos + len(match.group(0)) + 160)
            sub_window = content_lower[win_start:win_end]
            if "freeze" not in sub_window:
                line_num = content[:pos].count('\n') + 1
                violations.append(
                    f"{p.name}:L{line_num} -> Mention of '{match.group(0)}' is missing 'freeze' or 'freeze_audit' context in nearby window."
                )
                
        # Check '28 skipped' etc.
        for pattern in [r"\b28\s+skipped\b", r"\b28\s+omitidas\b", r"\b28\s+skipped\s+tests\b"]:
            for match in re.finditer(pattern, content_lower):
                pos = match.start()
                win_start = max(0, pos - 160)
                win_end = min(len(content), pos + len(match.group(0)) + 160)
                sub_window = content_lower[win_start:win_end]
                if "freeze" not in sub_window:
                    line_num = content[:pos].count('\n') + 1
                    violations.append(
                        f"{p.name}:L{line_num} -> Mention of '{match.group(0)}' is missing 'freeze' or 'freeze_audit' context in nearby window."
                    )
                    
    # 4. Check audit stdout contents. During run_final_freeze_audit.py the
    # summary is temporarily marked audit_in_progress, so the full stdout may
    # still correspond to the previous run until pytest finishes.
    summary_path = ROOT / "validation/freeze_audit/final_freeze_pytest_summary.json"
    in_progress = False
    if summary_path.exists():
        import json
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        in_progress = summary.get("reason") == "audit_in_progress"
    stdout_path = ROOT / "validation/freeze_audit/final_freeze_pytest_stdout.txt"
    if stdout_path.exists() and not in_progress:
        stdout_content = read(stdout_path)
        assert "944 passed" in stdout_content, "final_freeze_pytest_stdout.txt does not contain '944 passed'"
        assert "28 skipped" in stdout_content, "final_freeze_pytest_stdout.txt does not contain '28 skipped'"
        
    # 5. Check manifest yaml contents
    manifest_data = load_manifest()
    freeze_audit = manifest_data.get("freeze_audit", {})
    assert freeze_audit.get("passed") == 944, f"Manifest passed count is not 944 (got {freeze_audit.get('passed')})"
    assert freeze_audit.get("skipped") == 28, f"Manifest skipped count is not 28 (got {freeze_audit.get('skipped')})"
    assert freeze_audit.get("path") == "validation/freeze_audit/", f"Manifest path is incorrect (got {freeze_audit.get('path')})"
    
    assert not violations, (
        "Freeze audit reference violations found in active documentation:\n"
        + "\n".join(violations)
    )

