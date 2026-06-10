# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
VERSION2_DIR = ROOT_DIR / "version_2"
VALIDATION_DIR = VERSION2_DIR / "validation"

PROMOTED_FIELDS = {
    "figures",
    "pdf_path",
    "png_path",
    "metadata_path",
    "report_figures",
    "promoted_figures",
}

LEGACY_FIELDS = {
    "legacy_external_figures_not_promoted",
    "legacy_provenance",
    "archived_external_paths",
}

BANNED_PATTERNS = [
    re.compile(r"DF y NC", re.IGNORECASE),
    re.compile(r"copy/Figs", re.IGNORECASE),
    re.compile(r"\.\./\.\./\.\./", re.IGNORECASE),  # ../../../
]

def contains_banned_pattern(value: str) -> bool:
    if not isinstance(value, str):
        return False
    return any(p.search(value) for p in BANNED_PATTERNS)

@pytest.mark.hygiene
def test_no_external_figure_paths_in_validation_json():
    """Verify validation json artifacts do not contain external figure paths in promoted fields."""
    json_files = list(VALIDATION_DIR.glob("**/*.json"))
    violations = []
    
    for f in json_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
            
        # Helper to search dict recursively
        def scan_dict(d):
            for k, v in d.items():
                # Check promoted fields
                if k in PROMOTED_FIELDS:
                    if isinstance(v, list):
                        for item in v:
                            if contains_banned_pattern(item):
                                violations.append(f"{f.name} field '{k}': {item}")
                    elif contains_banned_pattern(v):
                        violations.append(f"{f.name} field '{k}': {v}")
                
                # Make sure banned patterns don't appear in arbitrary fields that aren't marked as legacy
                elif k not in LEGACY_FIELDS:
                    if isinstance(v, list):
                        for item in v:
                            if contains_banned_pattern(item):
                                violations.append(f"{f.name} field '{k}': {item}")
                    elif isinstance(v, dict):
                        scan_dict(v)
                    elif contains_banned_pattern(v):
                        violations.append(f"{f.name} field '{k}': {v}")
                
                elif isinstance(v, dict):
                    scan_dict(v)
                    
        if isinstance(data, dict):
            scan_dict(data)
            
    assert not violations, f"Banned external paths found in validation JSON: {violations}"

@pytest.mark.hygiene
def test_no_external_figure_paths_in_validation_md():
    """Verify validation markdown artifacts only reference legacy paths under a legacy header."""
    md_files = list(VALIDATION_DIR.glob("**/*.md"))
    violations = []
    
    for f in md_files:
        content = f.read_text(encoding="utf-8")
        lines = content.splitlines()
        
        current_header = ""
        for line_num, line in enumerate(lines, 1):
            if line.startswith("#"):
                current_header = line.strip().lower()
                continue
                
            # If line has banned patterns, check if header has 'legacy' or 'non-promoted'
            if any(p.search(line) for p in BANNED_PATTERNS):
                is_legacy_section = "legacy" in current_header or "non-promoted" in current_header
                if not is_legacy_section:
                    violations.append(f"{f.name}:line {line_num} (under header '{current_header}'): '{line.strip()}'")
                    
    assert not violations, f"Banned external paths found outside legacy markdown sections: {violations}"
