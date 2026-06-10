# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory
WORKSPACE_DIR = ROOT_DIR.parent

@pytest.mark.hygiene
def test_ci_matrix_versions():
    """Verify that the CI matrix contains 3.11, 3.12, and 3.13."""
    ci_path = WORKSPACE_DIR / ".github/workflows/ci.yml"
    assert ci_path.exists(), f"CI configuration file not found at {ci_path}"
    content = ci_path.read_text(encoding="utf-8")
    
    # Extract python-version array from matrix
    match = re.search(r"python-version:\s*\[([^\]]+)\]", content)
    assert match, "Could not find python-version matrix in ci.yml"
    
    versions = [v.strip().strip("'\"") for v in match.group(1).split(",")]
    
    for v in ["3.11", "3.12", "3.13"]:
        assert v in versions, f"Python version {v} is missing from the main CI matrix"

@pytest.mark.hygiene
def test_dependency_policy_ci_mentions():
    """Verify that dependency_policy.md correctly claims the tested versions and flags missing ones."""
    policy_path = ROOT_DIR / "docs/dependency_policy.md"
    assert policy_path.exists(), f"Dependency policy not found at {policy_path}"
    policy_content = policy_path.read_text(encoding="utf-8")
    
    # Load CI matrix versions
    ci_path = WORKSPACE_DIR / ".github/workflows/ci.yml"
    assert ci_path.exists()
    ci_content = ci_path.read_text(encoding="utf-8")
    match = re.search(r"python-version:\s*\[([^\]]+)\]", ci_content)
    assert match
    ci_versions = [v.strip().strip("'\"") for v in match.group(1).split(",")]
    
    # Check mentions for 3.11, 3.12, 3.13, 3.14
    for major_minor in ["3.11", "3.12", "3.13", "3.14"]:
        pattern_text = rf"{re.escape(major_minor)}\s+(?:is\s+)?tested\s+in\s+CI"
        pattern_table = rf"{re.escape(major_minor)}\s*\|\s*tested\s+in\s+CI"
        
        has_mention = (
            re.search(pattern_text, policy_content, re.IGNORECASE) is not None or
            re.search(pattern_table, policy_content, re.IGNORECASE) is not None
        )
        
        if has_mention:
            assert major_minor in ci_versions, (
                f"docs/dependency_policy.md claims Python {major_minor} is tested in CI, "
                f"but it is not present in the .github/workflows/ci.yml matrix."
            )

@pytest.mark.hygiene
def test_dependency_policy_spirit_phrases():
    """Verify that dependency_policy.md contains core compliance stance phrases."""
    policy_path = ROOT_DIR / "docs/dependency_policy.md"
    assert policy_path.exists()
    content = policy_path.read_text(encoding="utf-8")
    
    assert any(phrase in content for phrase in ["spirit of SPEC-0", "spirit of the [SPEC-0]"]), (
        "docs/dependency_policy.md must reference the spirit of SPEC-0"
    )
    assert any(phrase in content for phrase in ["extended support", "extended-support"]), (
        "docs/dependency_policy.md must reference extended support policy"
    )
