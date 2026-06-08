# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]

BDF_PRODUCTION_FILES = [
    "version_2/hidden_attractors/workflows/biased_chua.py",
    "version_2/hidden_attractors/plotting/biased_chua.py",
    "version_2/examples/chua_nonsmooth_biased_hidden_attractor/run_example.py",
]

def test_no_titles_in_bdf_production():
    """Verify that BDF production scripts do not set internal titles (set_title, suptitle, plt.title)."""
    title_pattern = re.compile(r"\b(set_title|suptitle|plt\.title)\s*\(")
    
    violations = []
    for rel_path in BDF_PRODUCTION_FILES:
        full_path = ROOT_DIR / rel_path
        assert full_path.exists(), f"El archivo de producción BDF esperado no existe: {rel_path}"
        
        content = full_path.read_text(encoding="utf-8")
        matches = title_pattern.findall(content)
        if matches:
            violations.append(f"{rel_path} (contiene: {list(set(matches))})")
            
    assert not violations, f"Se encontraron llamadas prohibidas a títulos gráficos en producción: {violations}"
