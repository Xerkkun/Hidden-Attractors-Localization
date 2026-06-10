# -*- coding: utf-8 -*-
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory
WORKSPACE_DIR = ROOT_DIR.parent

@pytest.mark.hygiene
def test_ci_uses_unified_cli():
    """Verify that CI runs hidden-attractors validate contract and not the legacy checker."""
    ci_path = WORKSPACE_DIR / ".github/workflows/ci.yml"
    assert ci_path.exists(), f"CI configuration file not found at {ci_path}"
    content = ci_path.read_text(encoding="utf-8")
    
    assert "hidden-attractors validate contract" in content, (
        "CI workflow should run 'hidden-attractors validate contract'"
    )
    
    assert "hidden-attractors-check-validation" not in content, (
        "CI workflow should not contain legacy command 'hidden-attractors-check-validation'"
    )
