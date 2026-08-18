# -*- coding: utf-8 -*-
import pytest

from tests.helpers.test_documentation_text import active_doc_paths


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
    for f in active_doc_paths():
        content = f.read_text(encoding="utf-8", errors="ignore")
        for claim in PROHIBITED_CLAIMS:
            if claim.lower() in content.lower():
                violations.append(f"{f.name} contains prohibited claim: '{claim}'")
    assert not violations, "\n".join(violations)

@pytest.mark.hygiene
def test_markdown_docs_machado_fdf_warning():
    """Verify that Machado/FDF mentions remain outside the public workflow."""
    violations = []
    for f in active_doc_paths():
        content = f.read_text(encoding="utf-8", errors="ignore")
        if "machado auxiliary" in content.lower() or "fdf" in content.lower():
            has_warning = (
                "theory" in content.lower() or
                "not a promoted" in content.lower() or
                "no promovido" in content.lower() or
                "validation-only" in content.lower() or
                "solo para validación" in content.lower() or
                "teoría" in content.lower() or
                "theory-only" in content.lower()
            )
            if not has_warning:
                violations.append(f"{f.name} mentions Machado/FDF without a theory/validation-only boundary")
    assert not violations, "\n".join(violations)

@pytest.mark.hygiene
def test_markdown_docs_chua_arctan_warning():
    """Verify that arctan mentions include the radius-limited promotion boundary."""
    violations = []
    for f in active_doc_paths():
        content = f.read_text(encoding="utf-8", errors="ignore")
        if "arctan" in content.lower():
            has_warning = (
                "r <= 0.3" in content.lower() or
                "radius-limited" in content.lower() or
                "local radii" in content.lower() or
                "radios locales" in content.lower() or
                "algebraic" in content.lower() or
                "algebraicamente" in content.lower() or
                "validation definition" in content.lower() or
                "not equivalent" in content.lower() or
                "no full-memory" in content.lower() or
                "no hiddenness claim" in content.lower()
            )
            if not has_warning:
                violations.append(f"{f.name} mentions arctan but lacks radius-limited/algebraic boundary context")
    assert not violations, "\n".join(violations)
