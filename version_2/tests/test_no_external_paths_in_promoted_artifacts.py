# -*- coding: utf-8 -*-
import json
import re
import os
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory

BANNED_PATTERNS = [
    re.compile(r"DF y NC", re.IGNORECASE),
    re.compile(r"copy/Figs", re.IGNORECASE),
    re.compile(r"copy\\Figs", re.IGNORECASE),
    re.compile(r"\.\./\.\./\.\./"),  # ../../../
    re.compile(r"\.\.\\\.\.\\\.\.\\"),  # ..\..\..\
]

LEGACY_FIELDS = {
    "legacy_external_figures_not_promoted",
    "legacy_provenance",
    "archived_external_paths",
}

def contains_banned_pattern(val: str) -> bool:
    if not isinstance(val, str):
        return False
    return any(p.search(val) for p in BANNED_PATTERNS)

@pytest.mark.hygiene
def test_no_external_paths_in_promoted_artifacts():
    """Verify that promoted artifacts (JSON, MD, CSV) under validation/, docs/, README,

    and REFERENCE_GUIDE contain no personal/external/non-canonical figure paths,
    unless explicitly nested inside a legacy field or section.
    """
    violations = []
    
    # Define files and folders to scan
    scan_paths = [
        ROOT_DIR / "validation",
        ROOT_DIR / "docs",
        ROOT_DIR / "README.md",
        ROOT_DIR / "REFERENCE_GUIDE.md",
    ]
    
    for path in scan_paths:
        if not path.exists():
            continue
            
        if path.is_file():
            files_to_check = [path]
        else:
            files_to_check = []
            for r, d, fs in os.walk(path):
                # Skip legacy tools directory or similar exclusions if walked
                if "tools/legacy" in r.replace("\\", "/"):
                    continue
                for f in fs:
                    files_to_check.append(Path(r) / f)
                    
        for f_path in files_to_check:
            suffix = f_path.suffix.lower()
            rel_file = f_path.relative_to(ROOT_DIR).as_posix()
            
            if suffix == ".json":
                # Check JSON structure
                try:
                    data = json.loads(f_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                    
                def scan_json(val, in_legacy=False):
                    if isinstance(val, dict):
                        for k, v in val.items():
                            is_leg = in_legacy or (k in LEGACY_FIELDS)
                            scan_json(v, is_leg)
                    elif isinstance(val, list):
                        for item in val:
                            scan_json(item, in_legacy)
                    elif isinstance(val, str):
                        if contains_banned_pattern(val) and not in_legacy:
                            violations.append(
                                f"JSON {rel_file}: Banned pattern in non-legacy field/value '{val}'"
                            )
                            
                scan_json(data)
                
            elif suffix == ".md":
                # Check Markdown structure line-by-line
                try:
                    content = f_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                    
                lines = content.splitlines()
                legacy_header_level = None
                
                for idx, line in enumerate(lines, 1):
                    # Check if line is a markdown header
                    header_match = re.match(r"^(#+)\s+(.*)$", line)
                    if header_match:
                        level = len(header_match.group(1))
                        title = header_match.group(2).lower()
                        # If header contains "legacy", "provenance" or "non-promoted"
                        if any(term in title for term in ["legacy", "provenance", "non-promoted"]):
                            legacy_header_level = level
                        else:
                            # If we see a header of equal or higher level, end legacy section
                            if legacy_header_level is not None and level <= legacy_header_level:
                                legacy_header_level = None
                                
                    if contains_banned_pattern(line):
                        if legacy_header_level is None:
                            violations.append(
                                f"Markdown {rel_file}:L{idx}: Banned pattern in line outside legacy section: '{line.strip()}'"
                            )
                            
            elif suffix == ".csv":
                # Check CSV file
                try:
                    content = f_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                # In CSV, no legacy context is supported, so any match is a violation
                for idx, line in enumerate(content.splitlines(), 1):
                    if contains_banned_pattern(line):
                        violations.append(
                            f"CSV {rel_file}:L{idx}: Banned pattern: '{line.strip()}'"
                        )
                        
    assert not violations, (
        "Banned external paths found in promoted validation, docs, or guides:\n"
        + "\n".join(violations)
    )
