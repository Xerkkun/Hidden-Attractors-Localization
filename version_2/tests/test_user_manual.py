# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory

REQUIRED_SECTIONS = [
    "Purpose and scientific scope",
    "Installation",
    "Public CLI",
    "Validated end-to-end example",
    "Python API",
    "Independent dynamical characterization",
    "YAML configuration",
    "Runtime outputs and caches",
    "Evidence states and hiddenness verification",
    "Fractional solvers and memory policy",
    "Figure export policy",
    "Troubleshooting",
    "Limitations",
    "Citation and reproducibility",
]

REQUIRED_LINKS = [
    "docs/quick_start.md",
    "docs/scientific_scope.md",
    "docs/api_stability.md",
    "docs/citation.md",
    "validation/freeze_audit/",
]

PROHIBITED_PHRASES = [
    "DF proves hiddenness",
    "Nyquist proves hiddenness",
    "continuation proves hiddenness",
    "globally verified hidden attractor",
    "exact Caputo periodic orbit",
]

@pytest.mark.hygiene
def test_user_manual_exists_and_sections():
    """Verify that USER_MANUAL.md exists and contains all required sections in order."""
    manual_path = ROOT_DIR / "USER_MANUAL.md"
    assert manual_path.exists(), f"USER_MANUAL.md does not exist at {manual_path}"
    
    content = manual_path.read_text(encoding="utf-8")
    
    # Check sections
    for i, section in enumerate(REQUIRED_SECTIONS, 1):
        pattern = rf"##\s+{i}\.\s+{re.escape(section)}"
        assert re.search(pattern, content, re.IGNORECASE) is not None, (
            f"Section '{i}. {section}' is missing or incorrectly formatted in USER_MANUAL.md"
        )

@pytest.mark.hygiene
def test_user_manual_required_links():
    """Verify that USER_MANUAL.md contains all canonical reference links."""
    manual_path = ROOT_DIR / "USER_MANUAL.md"
    content = manual_path.read_text(encoding="utf-8")
    
    for link in REQUIRED_LINKS:
        assert link in content, f"Canonical link reference to '{link}' is missing in USER_MANUAL.md"

@pytest.mark.hygiene
def test_user_manual_cli_commands():
    """Verify that the unified CLI is documented."""
    manual_path = ROOT_DIR / "USER_MANUAL.md"
    content = manual_path.read_text(encoding="utf-8")
    
    # Assert public command is present
    assert "hidden-attractors" in content, "USER_MANUAL.md does not mention the public CLI 'hidden-attractors'"
    assert "hidden-attractors validate contract" in content, "USER_MANUAL.md does not recommend 'hidden-attractors validate contract'"
    
@pytest.mark.hygiene
def test_user_manual_no_overclaims():
    """Verify that USER_MANUAL.md contains no prohibited scientific overclaims."""
    manual_path = ROOT_DIR / "USER_MANUAL.md"
    content = manual_path.read_text(encoding="utf-8")
    
    for phrase in PROHIBITED_PHRASES:
        assert phrase not in content, f"Prohibited overclaim phrase found: '{phrase}'"

@pytest.mark.hygiene
def test_user_manual_fractional_conventions():
    """Verify that correct fractional conventions (W_q(s), s^q I, lambda=(j omega)^q) are documented."""
    manual_path = ROOT_DIR / "USER_MANUAL.md"
    content = manual_path.read_text(encoding="utf-8")
    
    # Check transfer function conventions
    assert "W_q(s)" in content or "W_q(s)" in content.replace(" ", ""), "Missing W_q(s) transfer function reference"
    assert "s^q I" in content or "s^q I" in content.replace(" ", ""), "Missing s^q I complex matrix identity reference"
    
    # Check spectral parameter convention
    spectral_found = (
        "lambda = (j \\omega)^q" in content or 
        "\\lambda = (j \\omega)^q" in content or
        "lambda=(j \\omega)^q" in content or
        "lambda = (j\\omega)^q" in content or
        "lambda=(j\\omega)^q" in content or
        "lambda = (j omega)^q" in content or
        "lambda=(j omega)^q" in content
    )
    assert spectral_found, "Missing spectral parameter (lambda = (j omega)^q) definition"
    
    # Check Caputo memory reference
    assert "Caputo" in content, "Caputo derivative reference is missing"
    assert "memory" in content or "history" in content, "Caputo memory/history references are missing"
