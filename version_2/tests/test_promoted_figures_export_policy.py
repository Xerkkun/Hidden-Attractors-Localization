# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
VERSION2_DIR = ROOT_DIR / "version_2"

PROMOTED_PATHS = [
    VERSION2_DIR / "hidden_attractors" / "plotting",
    VERSION2_DIR / "hidden_attractors" / "workflows",
    VERSION2_DIR / "examples",
    VERSION2_DIR / "tools" / "cli",
]

ALLOWED_SAVEFIG_FILE = VERSION2_DIR / "hidden_attractors" / "plotting" / "export.py"

@pytest.mark.plotting
@pytest.mark.hygiene
def test_promoted_figures_no_direct_savefig():
    """Verify that only export.py calls savefig directly inside canonical directories."""
    violations = []
    savefig_pattern = re.compile(r"\bsavefig\s*\(")
    
    for path in PROMOTED_PATHS:
        if not path.exists():
            continue
            
        py_files = list(path.glob("**/*.py"))
        for f in py_files:
            # Ignore pycache / egg-info
            if "__pycache__" in f.parts or "egg-info" in f.parts:
                continue
                
            if f.resolve() == ALLOWED_SAVEFIG_FILE.resolve():
                continue
                
            try:
                content = f.read_text(encoding="utf-8")
            except Exception:
                continue
                
            if savefig_pattern.search(content):
                rel_path = f.relative_to(VERSION2_DIR).as_posix()
                violations.append(rel_path)
                
    assert not violations, f"Rutas promovidas/canónicas llaman a savefig directamente fuera de export.py: {violations}"
