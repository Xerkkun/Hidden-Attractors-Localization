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

LEGACY_COMMANDS = {
    "hidden-attractors-check-validation",
    "hidden-attractors-protocol",
    "hidden-attractors-sphere-controls",
    "hidden-attractors-refined-basin",
    "hidden-attractors-fractional-report-run",
    "hidden-attractors-robustness-overlay",
    "hidden-attractors-danca-abm-sphere-controls",
}

DEPRECATION_KEYWORDS = [
    "legacy",
    "deprecated",
    "migration",
    "no longer installed",
    "not public",
    "no público",
    "no se instala",
    "no ejecutar",
    "ya no",
    "antiguos",
    "deprecación",
    "históricos",
    "antiguas",
]

@pytest.mark.hygiene
def test_markdown_docs_cli_consistency():
    """Verify that only the unified CLI is recommended and legacy command usage is deprecated in markdown docs."""
    violations_legacy = []
    
    for f in FILES_TO_CHECK:
        assert f.exists(), f"Expected file {f} does not exist"
        content = f.read_text(encoding="utf-8", errors="ignore")
        
        # Check active code blocks (bash, sh, shell, powershell)
        code_blocks = re.findall(r"```(?:bash|sh|shell|powershell|cmd)\n(.*?)```", content, re.DOTALL)
        for block in code_blocks:
            for cmd in LEGACY_COMMANDS:
                if cmd in block:
                    violations_legacy.append(f"{f.name} (code block) -> '{cmd}' is actively recommended/written as an executable command")

        # Check line-by-line deprecation context for legacy commands
        for line_num, line in enumerate(content.splitlines(), 1):
            for cmd in LEGACY_COMMANDS:
                if cmd in line:
                    has_context = any(kw.lower() in line.lower() for kw in DEPRECATION_KEYWORDS)
                    if not has_context:
                        violations_legacy.append(f"{f.name}:L{line_num} -> '{line.strip()}' contains '{cmd}' without deprecation context")
                        
    assert not violations_legacy, (
        "Legacy commands found inside active script blocks or without proper deprecation context:\n"
        + "\n".join(violations_legacy)
    )

@pytest.mark.hygiene
def test_markdown_docs_no_outdated_test_counts():
    """Verify that no outdated test count ('156') is present in any markdown file."""
    violations = []
    
    for f in FILES_TO_CHECK:
        content = f.read_text(encoding="utf-8", errors="ignore")
        # Find occurrences of '156'
        for line_num, line in enumerate(content.splitlines(), 1):
            if "156" in line:
                # If it's related to test count, flag it.
                if any(w in line.lower() for w in ["test", "prueba"]):
                    violations.append(f"{f.name}:L{line_num} -> '{line.strip()}' (mentions outdated test count '156')")
                    
    assert not violations, (
        "Outdated test count '156' found in markdown files:\n"
        + "\n".join(violations)
    )

@pytest.mark.hygiene
def test_markdown_docs_reference_manuals():
    """Verify that primary markdown documentation files reference USER_MANUAL.md or THESIS_CLAIMS.md."""
    core_docs = [
        WORKSPACE_DIR / "README.md",
        ROOT_DIR / "README.md",
        ROOT_DIR / "REFERENCE_GUIDE.md",
        ROOT_DIR / "docs/validation_evidence.md",
        ROOT_DIR / "docs/unified_report.md",
        ROOT_DIR / "docs/figure_export_policy.md",
    ]
    
    violations = []
    for f in core_docs:
        content = f.read_text(encoding="utf-8", errors="ignore")
        has_manual = "USER_MANUAL.md" in content
        has_claims = "THESIS_CLAIMS.md" in content
        if not (has_manual or has_claims):
            violations.append(f"{f.name} lacks reference to both USER_MANUAL.md and THESIS_CLAIMS.md")
            
    assert not violations, (
        "Primary documentation files missing links to USER_MANUAL.md or THESIS_CLAIMS.md:\n"
        + "\n".join(violations)
    )
