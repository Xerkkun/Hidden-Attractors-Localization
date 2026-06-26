# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory
LATEX_PATH = ROOT_DIR / "docs/reporte_unificado_chua_fraccionario.tex"

LEGACY_COMMANDS = [
    "hidden-attractors-check-validation",
    "hidden-attractors-protocol",
    "hidden-attractors-sphere-controls",
    "hidden-attractors-refined-basin",
    "hidden-attractors-robustness-overlay",
    "hidden-attractors-danca-abm-sphere-controls",
    "hidden-attractors-fractional-report-run",
    "hidden-attractors-systems",
    "hidden-attractors-workflow-requirements",
]

DEPRECATION_KEYWORDS = [
    "legacy",
    "deprecated",
    "antiguo",
    "obsoleto",
    "no público",
    "no forman parte",
    "reemplazan",
    "deprecado",
    "ya no",
]

@pytest.mark.hygiene
def test_latex_report_exists():
    """Verify that the LaTeX report exists."""
    assert LATEX_PATH.exists(), f"LaTeX report not found at {LATEX_PATH}"

@pytest.mark.hygiene
def test_latex_report_sections_and_docs():
    """Verify sections and document references exist in the LaTeX report."""
    content = LATEX_PATH.read_text(encoding="utf-8", errors="ignore")
    
    assert "\\section{Manual de usuario}" in content, "Missing section 'Manual de usuario'"
    
    required_refs = [
        "USER_MANUAL.md",
        "THESIS_CLAIMS.md",
        "docs/manual_manifest.yaml",
        "validation/freeze_audit",
    ]
    for ref in required_refs:
        assert ref in content, f"Missing reference to '{ref}'"

@pytest.mark.hygiene
def test_latex_report_unified_cli():
    """Verify unified CLI commands are present and legacy ones only appear in deprecation context."""
    content = LATEX_PATH.read_text(encoding="utf-8", errors="ignore")
    
    required_commands = [
        "hidden-attractors inspect systems",
        "hidden-attractors validate contract",
        "hidden-attractors protocol",
    ]
    for cmd in required_commands:
        # Match commands even if they span lines or include cmdbreak/braces/spaces
        pattern = re.escape(cmd).replace(r"\ ", r"\s*(?:\\cmdbreak\{|\s|\}\s*\{\s*|\\codepath\{|\\texttt\{|\\_)?\s*")
        assert re.search(pattern, content) or cmd in content, f"Missing unified CLI command: '{cmd}'"

    # Verify no legacy commands recommended as executable (check each line)
    for line_num, line in enumerate(content.splitlines(), 1):
        for cmd in LEGACY_COMMANDS:
            if cmd in line:
                has_context = any(kw in line.lower() for kw in DEPRECATION_KEYWORDS)
                assert has_context, (
                    f"Legacy command '{cmd}' found in line {line_num} without deprecation context: '{line.strip()}'"
                )

@pytest.mark.hygiene
def test_latex_report_expected_outputs_table():
    """Verify that expected outputs table is present and matches requirements."""
    content = LATEX_PATH.read_text(encoding="utf-8", errors="ignore")
    
    assert re.search(r"hidden-attractors\s*(?:\}\s*\{\s*|\s+)?inspect\s+systems", content)
    assert re.search(r"hidden-attractors\s*(?:\}\s*\{\s*|\s+)?run\s+-p\s+chua\\?_integer", content)
    assert re.search(r"hidden-attractors\s*(?:\}\s*\{\s*|\s+)?validate\s+contract\s+-{1,2}allow-pending", content)

@pytest.mark.hygiene
def test_latex_report_claims_matrix():
    """Verify that the claims matrix contains correct statuses."""
    content = LATEX_PATH.read_text(encoding="utf-8", errors="ignore")
    
    matrix_keywords = [
        "Chua entero",
        "reproducido",
        "Chua no suave Ejemplo 1",
        "candidato",
        "Candidato oficial fraccionario no suave",
        "rechazado",
        "Chua arctan fraccionario",
        "promovido localmente",
        "Metodolog",
        "validado como framework",
    ]
    for kw in matrix_keywords:
        assert kw in content, f"Missing claims keyword/phrase: '{kw}'"

@pytest.mark.hygiene
def test_latex_report_fractional_conventions():
    """Verify that correct fractional conventions are documented and ambiguous ones are defined."""
    content = LATEX_PATH.read_text(encoding="utf-8", errors="ignore")
    
    # Must contain the explicit transfer function conventions
    assert "\\widehat W_q" in content and "\\lambda" in content, "Missing transfer function notation with lambda"
    assert (
        "\\lambda=(\\ii\\omega)^q" in content
        or "\\lambda = (\\ii\\omega)^q" in content
        or "\\lambda={\\left(\\ii\\omega\\right)}^q" in content
        or "\\lambda={\\left(\\ii\\omega\\right)}^q" in content.replace(" ", "")
    ), "Missing definition of lambda"
    
    # Check that any raw W_q(i omega) etc are defined near lambda definition
    for line_num, line in enumerate(content.splitlines(), 1):
        if any(term in line for term in ["W_q(\\ii\\omega)=", "W_q(i\\omega)=", "W_q(j\\omega)="]):
            assert "\\lambda" in line or "lambda" in line, (
                f"Line {line_num} contains ambiguous transfer function assignment: '{line.strip()}'"
            )

@pytest.mark.hygiene
def test_latex_report_no_outdated_test_counts():
    """Verify that no outdated test count ('156') is present in the LaTeX report."""
    content = LATEX_PATH.read_text(encoding="utf-8", errors="ignore")
    
    for line_num, line in enumerate(content.splitlines(), 1):
        if "156" in line:
            if any(w in line.lower() for w in ["test", "prueba"]):
                pytest.fail(f"Line {line_num} mentions outdated test count '156': '{line.strip()}'")

@pytest.mark.hygiene
def test_latex_report_no_arctan_overclaims():
    """Verify that Chua arctan promotion is radius-limited, not global/proved."""
    content = LATEX_PATH.read_text(encoding="utf-8", errors="ignore")
    
    assert "hiddenness_supported_under_tested_local_radii" in content
    assert "r=0.3" in content.replace(" ", "")
    
    prohibited_verified = [
        "arctan es un atractor oculto verificado",
        "arctan oculto verificado",
        "arctan está verificado",
    ]
    for ph in prohibited_verified:
        assert ph not in content.lower(), f"Prohibited overclaim found for arctan: '{ph}'"
