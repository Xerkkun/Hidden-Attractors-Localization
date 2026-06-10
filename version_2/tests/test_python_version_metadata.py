# -*- coding: utf-8 -*-
import tomllib
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory
WORKSPACE_DIR = ROOT_DIR.parent

@pytest.mark.hygiene
def test_pyproject_python_metadata():
    """Verify pyproject.toml Python compatibility and classifier versions."""
    pyproject_path = ROOT_DIR / "pyproject.toml"
    assert pyproject_path.exists(), f"pyproject.toml not found at {pyproject_path}"
    
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
        
    project = data.get("project", {})
    
    # 1. requires-python check
    requires_python = project.get("requires-python", "")
    assert requires_python, "requires-python is missing in pyproject.toml"
    # Must be compatible with >= 3.11
    assert ">=3.11" in requires_python or ">= 3.11" in requires_python, (
        f"requires-python ('{requires_python}') is not compatible with standard >=3.11"
    )
    
    # 2. Extract versions from classifiers
    classifiers = project.get("classifiers", [])
    python_classifiers = [c for c in classifiers if "Programming Language :: Python :: 3." in c]
    
    classifier_versions = []
    for c in python_classifiers:
        m = re.search(r"Programming Language :: Python :: 3\.(\d+)", c)
        if m:
            classifier_versions.append(f"3.{m.group(1)}")
            
    # Load CI matrix versions
    ci_path = WORKSPACE_DIR / ".github/workflows/ci.yml"
    assert ci_path.exists()
    ci_content = ci_path.read_text(encoding="utf-8")
    match = re.search(r"python-version:\s*\[([^\]]+)\]", ci_content)
    assert match
    ci_versions = [v.strip().strip("'\"") for v in match.group(1).split(",")]
    
    # Load dependency policy to check for experimental exemptions
    policy_path = ROOT_DIR / "docs/dependency_policy.md"
    policy_content = policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
    
    for v in classifier_versions:
        if v in ci_versions:
            continue
            
        # If classifier version is not in CI, it must be documented as experimental/untested
        exempted = (
            v in policy_content
            and any(term in policy_content.lower() for term in ["experimental", "untested", "not tested", "not guaranteed"])
        )
        assert exempted, (
            f"Python {v} is listed as a classifier in pyproject.toml but is not present in "
            f"the CI matrix, and is not documented as experimental/untested in docs/dependency_policy.md."
        )
