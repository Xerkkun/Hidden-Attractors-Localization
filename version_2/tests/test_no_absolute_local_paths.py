# -*- coding: utf-8 -*-
from pathlib import Path
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
    """Scan the active codebase and ensure no personal local absolute paths are defined."""
    # Active code extensions to check
    extensions = [".py", ".md", ".tex", ".yaml", ".json"]
    
    violations = []
    
    # We walk from the root directory but exclude archived folders and envs
    exclude_dirs = {
        "_archived_figure_scripts",
        "_reference_scripts",
        ".git",
        ".venv",
        "venv",
        ".pytest_cache",
        "__pycache__",
        ".benchmarks",
        ".runtime_cache",
        ".runtime_native",
        "build",
        "hidden_attractors_fo.egg-info",
    }
    
    for path in ROOT_DIR.rglob("*"):
        # Skip directories
        if path.is_dir():
            continue
            
        # Skip this test script itself to avoid false matching on literal pattern constants
        if path.name == "test_no_absolute_local_paths.py":
            continue
            
        # Check if file is in excluded directory
        parts = path.relative_to(ROOT_DIR).parts
        if any(d in parts for d in exclude_dirs):
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
