# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest
from tests.helpers.test_documentation_text import active_doc_paths, read

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory
WORKSPACE_DIR = ROOT_DIR.parent

@pytest.mark.hygiene
def test_markdown_docs_cli_consistency():
    """Verify that executable examples use the unified CLI token."""
    violations = []
    docs = active_doc_paths()
    
    for f in docs:
        content = read(f)
        
        code_blocks = re.findall(r"```(?:bash|sh|shell|powershell|cmd|text)?\n(.*?)```", content, re.DOTALL)
        for block in code_blocks:
            for line in block.splitlines():
                token = line.strip().split(maxsplit=1)[0] if line.strip() else ""
                if token.startswith("hidden-attractors-"):
                    violations.append(f"{f.name} -> unsupported executable token '{token}'")
                        
    assert not violations, (
        "Standalone command tokens found instead of the unified CLI:\n"
        + "\n".join(violations)
    )

@pytest.mark.hygiene
def test_markdown_docs_no_outdated_test_counts():
    """Verify that no outdated test count ('156') is present in any markdown file."""
    violations = []
    docs = active_doc_paths()
    
    for f in docs:
        content = read(f)
        # Find occurrences of '156'
        for line_num, line in enumerate(content.splitlines(), 1):
            if "156" in line:
                if any(w in line.lower() for w in ["test", "prueba"]):
                    violations.append(f"{f.name}:L{line_num} -> '{line.strip()}' (mentions outdated test count '156')")
                    
    assert not violations, (
        "Outdated test count '156' found in markdown files:\n"
        + "\n".join(violations)
    )

@pytest.mark.hygiene
def test_markdown_docs_reference_manuals():
    """Verify that public entry documents point readers to USER_MANUAL.md."""
    core_docs = [
        WORKSPACE_DIR / "README.md",
        ROOT_DIR / "README.md",
        ROOT_DIR / "REFERENCE_GUIDE.md",
    ]
    
    violations = []
    for f in core_docs:
        if f.exists():
            content = read(f)
            if "USER_MANUAL.md" not in content:
                violations.append(f"{f.name} lacks a reference to USER_MANUAL.md")
            
    assert not violations, (
        "Public entry documents missing links to USER_MANUAL.md:\n"
        + "\n".join(violations)
    )

@pytest.mark.hygiene
def test_development_installation_is_complete():
    """Verify that development installation recommendations include supported extras."""
    violations = []
    docs = active_doc_paths()
    
    # We look for recommended commands installing version_2 or root folder in editable mode
    for f in docs:
        content = read(f)
        # Find lines recommending pip install -e
        for line_num, line in enumerate(content.splitlines(), 1):
            if ("pip install -e" in line or "pip install" in line) and ("dev" in line or "analysis" in line or "legacy" in line or "docs" in line):
                # Ensure it doesn't suggest incomplete extras
                if "dev" in line and "docs" not in line and "migration" not in line.lower() and "historical" not in line.lower():
                    violations.append(f"{f.name}:L{line_num} -> '{line.strip()}' has incomplete extras (missing 'docs')")
                if "pip install -e ." in line or "pip install -e version_2" in line:
                    if "[" not in line and "migration" not in line.lower() and "historical" not in line.lower():
                        violations.append(f"{f.name}:L{line_num} -> '{line.strip()}' lacks extras (recommends pip install -e . / -e version_2 directly)")

    assert not violations, (
        "Incomplete dev install commands found (e.g. missing 'docs' extra or lacks extras):\n"
        + "\n".join(violations)
    )

@pytest.mark.hygiene
def test_tools_cli_and_legacy_are_not_public():
    """Verify that tools/cli and tools/legacy are not presented as the official public command surface."""
    violations = []
    docs = active_doc_paths()
    
    for f in docs:
        content = read(f)
        content_lower = content.lower()
        if "tools/cli" in content_lower:
            # Should not say "official surface" or similar
            if any(term in content_lower for term in ["official command surface", "superficie oficial", "public command surface"]):
                # Allow it if there is explicit deprecation/migration text in the same file or nearby context
                if not any(dep in content_lower for dep in DEPRECATION_KEYWORDS):
                    violations.append(f"{f.name} -> Describes tools/cli as the official/public surface without deprecation context")
                    
        if "tools/legacy" in content_lower:
            if any(term in content_lower for term in ["public executable", "ejecutable público", "official command"]):
                if not any(dep in content_lower for dep in DEPRECATION_KEYWORDS):
                    violations.append(f"{f.name} -> Describes tools/legacy as public/official without deprecation context")
                    
    assert not violations, (
        "tools/cli or tools/legacy presented as public or official command surface:\n"
        + "\n".join(violations)
    )

@pytest.mark.hygiene
def test_release_docs_have_submission_strict_validation():
    """Verify that release-related documentation contains the submission-strict validation command."""
    release_docs = [
        ROOT_DIR / "release_package/README_RELEASE.md",
        ROOT_DIR / "release_package/SAMPLE_RUN.md",
        ROOT_DIR / "release_package/reproducibility_checklist.md",
    ]
    
    violations = []
    target_cmd = "tools/release/validate_release_readiness.py --submission-strict"
    
    for f in release_docs:
        if f.exists():
            content = read(f)
            if target_cmd not in content:
                violations.append(f"{f.name} does not mention '{target_cmd}'")
                
    assert not violations, (
        "Release documentation files missing required submission-strict validation command:\n"
        + "\n".join(violations)
    )
