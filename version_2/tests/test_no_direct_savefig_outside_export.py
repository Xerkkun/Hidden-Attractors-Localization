# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
VERSION2_DIR = ROOT_DIR / "version_2"

# Documented exceptions from figure_scripts_inventory.md
SAVEFIG_EXCEPTIONS = {
    "version_2/hidden_attractors/plotting/export.py",
    "version_2/hidden_attractors/workflows/danca_abm_sphere_controls.py",
    "version_2/hidden_attractors/workflows/fractional_report_run.py",
    "version_2/hidden_attractors/workflows/refined_basin.py",
    "version_2/hidden_attractors/plotting/generate_publication_figures.py",
    "version_2/hidden_attractors/plotting/matignon.py",
    "version_2/hidden_attractors/plotting/dynamics.py",
    "version_2/hidden_attractors/plotting/basin.py",
    "version_2/hidden_attractors/plotting/overlays.py",
}

def is_exception(file_path: Path) -> bool:
    # Resolve path relative to ROOT_DIR
    try:
        rel_parts = file_path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return False
        
    if rel_parts in SAVEFIG_EXCEPTIONS:
        return True
    
    # Generic plotting modules plot_*.py are allowed exceptions
    if "version_2/hidden_attractors/plotting/plot_" in rel_parts:
        return True
        
    # Test directory is exempt
    if "version_2/tests" in rel_parts:
        return True
        
    return False

def test_no_direct_savefig():
    """Scan all active Python production scripts and ensure savefig is not called directly."""
    py_files = list(VERSION2_DIR.glob("**/*.py"))
    
    violations = []
    savefig_pattern = re.compile(r"\bsavefig\s*\(")
    
    for f in py_files:
        if is_exception(f):
            continue
            
        # Ignore pycache/build artifacts
        if "__pycache__" in f.parts or "build" in f.parts or "egg-info" in f.parts:
            continue
            
        try:
            content = f.read_text(encoding="utf-8")
        except Exception:
            continue
            
        if savefig_pattern.search(content):
            # Resolve to relative path for clear error report
            rel_path = f.relative_to(ROOT_DIR).as_posix()
            violations.append(rel_path)
            
    assert not violations, f"Llamadas directas a savefig encontradas en scripts no autorizados: {violations}"
