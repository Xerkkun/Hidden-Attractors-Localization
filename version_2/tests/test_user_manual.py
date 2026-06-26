# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory

REQUIRED_SECTIONS = [
    "Purpose and scientific scope",
    "Installation",
    "Repository structure",
    "Public CLI",
    "Minimal examples",
    "Reproducible example 1: Chua integer",
    "Reproducible example 2: Chua nonsmooth BDF",
    "Radius-limited promoted example: Chua arctan c590",
    "YAML configuration format",
    "Output files and expected results",
    "Evidence states and hiddenness labels",
    "Hiddenness verification protocol",
    "Fractional-order solvers and memory policy",
    "Figure export policy",
    "Troubleshooting",
    "Limitations",
    "Citation and reproducibility",
]

REQUIRED_LINKS = [
    "THESIS_CLAIMS.md",
    "docs/quick_start.md",
    "docs/figure_export_policy.md",
    "docs/dependency_policy.md",
    "validation/freeze_audit/",
]

LEGACY_COMMANDS = [
    "hidden-attractors-check-validation",
    "hidden-attractors-protocol",
    "hidden-attractors-sphere-controls",
    "hidden-attractors-refined-basin",
    "hidden-attractors-robustness-overlay",
    "hidden-attractors-danca-abm-sphere-controls",
    "hidden-attractors-fractional-report-run",
]

DEPRECATION_KEYWORDS = [
    "legacy",
    "deprecated",
    "removed",
    "no longer installed",
    "not public",
]

PROHIBITED_PHRASES = [
    "DF proves hiddenness",
    "Nyquist proves hiddenness",
    "continuation proves hiddenness",
    "globally verified hidden attractor",
    "Chua arctan hidden attractor verified",
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
    """Verify that only the unified CLI is recommended and legacy command usage is deprecated."""
    manual_path = ROOT_DIR / "USER_MANUAL.md"
    content = manual_path.read_text(encoding="utf-8")
    
    # Assert public command is present
    assert "hidden-attractors" in content, "USER_MANUAL.md does not mention the public CLI 'hidden-attractors'"
    assert "hidden-attractors validate contract" in content, "USER_MANUAL.md does not recommend 'hidden-attractors validate contract'"
    
    # Check legacy commands deprecation context
    for line in content.splitlines():
        for cmd in LEGACY_COMMANDS:
            if cmd in line:
                # Must contain one of the deprecation keywords
                has_depr = any(kw.lower() in line.lower() for kw in DEPRECATION_KEYWORDS)
                assert has_depr, (
                    f"Legacy command '{cmd}' mentioned without deprecation/legacy context: '{line.strip()}'"
                )

@pytest.mark.hygiene
def test_user_manual_no_overclaims():
    """Verify that USER_MANUAL.md contains no prohibited scientific overclaims."""
    manual_path = ROOT_DIR / "USER_MANUAL.md"
    content = manual_path.read_text(encoding="utf-8")
    
    for phrase in PROHIBITED_PHRASES:
        assert phrase not in content, f"Prohibited overclaim phrase found: '{phrase}'"

@pytest.mark.hygiene
def test_user_manual_chua_arctan_status():
    """Verify that Chua arctan c590 is radius-limited, not global/proved."""
    manual_path = ROOT_DIR / "USER_MANUAL.md"
    content = manual_path.read_text(encoding="utf-8")

    sections = re.split(r"##\s+\d+\.", content)
    arctan_section = None
    for sec in sections:
        if "Chua arctan" in sec:
            arctan_section = sec
            break

    assert arctan_section is not None, "Could not isolate Chua arctan section in USER_MANUAL.md"
    section_lower = arctan_section.lower()
    assert "r <= 0.3" in arctan_section
    assert "8400" in arctan_section
    assert "zero contacts" in section_lower
    assert "global" in section_lower
    assert "proved" not in section_lower

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
