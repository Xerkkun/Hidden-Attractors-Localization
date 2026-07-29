# -*- coding: utf-8 -*-
from pathlib import Path
import subprocess
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]

# Prohibited patterns indicating personal absolute Windows paths
PROHIBITED_PATTERNS = [
    "c:/Users",
    "c:\\Users",
    "C:\\Users",
    "C:/Users",
    "Desktop/Codes",
    "Desktop\\Codes",
]

def test_no_absolute_local_paths_in_codebase():
    """Scan the tracked public tree for personal local absolute paths."""
    # Active code extensions to check
    extensions = [".py", ".md", ".tex", ".yaml", ".json"]
    
    violations = []
    
    tracked = subprocess.run(
        ["git", "-C", str(ROOT_DIR), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8").split("\0")

    for relative in tracked:
        if not relative:
            continue
        path = ROOT_DIR / relative
        if not path.is_file():
            continue
            
        # Skip this test script itself to avoid false matching on literal pattern constants
        if path.name == "test_no_absolute_local_paths.py":
            continue
            
        # Check extension
        if path.suffix not in extensions:
            continue
            
        # Read file content and scan for prohibited patterns
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
            
        for pattern in PROHIBITED_PATTERNS:
            if pattern in content:
                # Find matching line for helpful debugging output
                lines = content.splitlines()
                for idx, line in enumerate(lines):
                    if pattern in line:
                        rel_path = path.relative_to(ROOT_DIR).as_posix()
                        violations.append(f"{rel_path}:L{idx+1} -> '{line.strip()}'")
                        
    assert not violations, f"Rutas absolutas personales encontradas:\n" + "\n".join(violations)
