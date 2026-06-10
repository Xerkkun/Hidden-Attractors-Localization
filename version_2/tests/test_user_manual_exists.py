# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
from tests.helpers.test_documentation_text import read, normalize, ROOT

@pytest.mark.hygiene
def test_user_manual_exists_and_content_integrity():
    manual_path = ROOT / "USER_MANUAL.md"
    assert manual_path.exists(), "USER_MANUAL.md does not exist at repo root"
    
    content = read(manual_path)
    content_lower = content.lower()
    normalized_content = normalize(content)
    
    # Required references
    required_references = [
        "thesis_claims.md",
        "validation/freeze_audit/",
        "docs/figure_export_policy.md",
        "docs/dependency_policy.md",
        "docs/quick_start.md",
    ]
    for ref in required_references:
        assert ref.lower() in content_lower, f"Required reference '{ref}' is missing in USER_MANUAL.md"
        
    # Public CLI and validation route
    assert "hidden-attractors" in content_lower, "Public command 'hidden-attractors' is missing in USER_MANUAL.md"
    assert "hidden-attractors validate contract" in content_lower, "Unified validation path 'hidden-attractors validate contract' is missing in USER_MANUAL.md"
    
    # Key sections or equivalent text
    required_terms = [
        "hiddenness verification protocol",
        "evidence states",
        "limitations",
        "troubleshooting",
    ]
    for term in required_terms:
        assert term in normalized_content, f"Topic or section '{term}' is missing in USER_MANUAL.md"
        
    # Verify Caputo and memory policy are described
    assert "caputo" in normalized_content, "Caputo derivative reference is missing in USER_MANUAL.md"
    assert "memory" in normalized_content or "history" in normalized_content, "Memory or history policy is missing in USER_MANUAL.md"
        
    # Fails if obsolete test count is present
    forbidden_counts = [
        "156 tests",
        "156 unit tests",
        "156 pruebas",
    ]
    for count in forbidden_counts:
        assert count.lower() not in content_lower, f"Obsolete test count phrase '{count}' found in USER_MANUAL.md"
        
    # Fails on strong claims
    forbidden_claims = [
        "globally verified hidden attractor",
        "chua arctan hidden attractor verified",
        "df proves hiddenness",
        "nyquist proves hiddenness",
        "continuation proves hiddenness",
    ]
    for claim in forbidden_claims:
        assert claim.lower() not in content_lower, f"Forbidden overclaim phrase '{claim}' found in USER_MANUAL.md"
