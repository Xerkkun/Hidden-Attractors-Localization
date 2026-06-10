# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory
WORKSPACE_DIR = ROOT_DIR.parent

FILES_TO_CHECK = [
    WORKSPACE_DIR / "README.md",
    ROOT_DIR / "README.md",
    ROOT_DIR / "REFERENCE_GUIDE.md",
    ROOT_DIR / "docs/quick_start.md",
    ROOT_DIR / "docs/installation.md",
    ROOT_DIR / "docs/testing.md",
    ROOT_DIR / "docs/validation_evidence.md",
    ROOT_DIR / "docs/unified_report.md",
    ROOT_DIR / "docs/figure_export_policy.md",
]

PROHIBITED_CLAIMS = [
    "DF proves hiddenness",
    "Nyquist proves hiddenness",
    "continuation proves hiddenness",
    "bounded simulation proves hiddenness",
    "globally verified hidden attractor",
    "Chua arctan hidden attractor verified",
]

@pytest.mark.hygiene
def test_markdown_docs_no_prohibited_claims():
    """Verify that no prohibited claims are made in markdown files."""
    violations = []
    for f in FILES_TO_CHECK:
        if not f.exists():
            continue
        content = f.read_text(encoding="utf-8", errors="ignore")
        for claim in PROHIBITED_CLAIMS:
            if claim.lower() in content.lower():
                violations.append(f"{f.name} contains prohibited claim: '{claim}'")
    assert not violations, "\n".join(violations)

@pytest.mark.hygiene
def test_markdown_docs_machado_fdf_warning():
    """Verify that any file mentioning Machado or FDF contains warnings explaining it's planned/theory only."""
    violations = []
    for f in FILES_TO_CHECK:
        if not f.exists():
            continue
        content = f.read_text(encoding="utf-8", errors="ignore")
        if "machado" in content.lower() or "fdf" in content.lower():
            # Check for warnings indicating it is planned or theory only
            has_warning = (
                "theory" in content.lower() or
                "planned" in content.lower() or
                "not a promoted" in content.lower() or
                "no promovido" in content.lower() or
                "planeado" in content.lower() or
                "teoría" in content.lower()
            )
            if not has_warning:
                violations.append(f"{f.name} mentions Machado/FDF but lacks planned/theory/not-promoted warnings")
    assert not violations, "\n".join(violations)

@pytest.mark.hygiene
def test_markdown_docs_chua_arctan_warning():
    """Verify that any file mentioning Chua arctan contains warnings explaining it's algebraically implemented but pending validation."""
    violations = []
    for f in FILES_TO_CHECK:
        if not f.exists():
            continue
        content = f.read_text(encoding="utf-8", errors="ignore")
        if "arctan" in content.lower():
            has_warning = (
                "pending" in content.lower() or
                "pendiente" in content.lower() or
                "non-certified" in content.lower() or
                "no certificado" in content.lower() or
                "algebraic" in content.lower() or
                "algebraicamente" in content.lower()
            )
            if not has_warning:
                violations.append(f"{f.name} mentions arctan but lacks pending/algebraic validation warnings")
    assert not violations, "\n".join(violations)
