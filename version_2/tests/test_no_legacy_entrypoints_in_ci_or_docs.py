# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory
WORKSPACE_DIR = ROOT_DIR.parent

LEGACY_COMMANDS = {
    "hidden-attractors-check-validation",
    "hidden-attractors-protocol",
    "hidden-attractors-sphere-controls",
    "hidden-attractors-refined-basin",
    "hidden-attractors-fractional-report-run",
    "hidden-attractors-robustness-overlay",
    "hidden-attractors-danca-abm-sphere-controls",
}

CONTEXT_WORDS = [
    "legacy",
    "deprecated",
    "migration",
    "no longer installed",
    "no público",
    "no se instala",
    "no ejecutar",
    "ya no",
    "antiguos",
    "deprecación",
    "históricos"
]

@pytest.mark.hygiene
def test_no_legacy_entrypoints_in_ci():
    """Verify that no legacy entrypoint is used in the CI workflow."""
    ci_path = WORKSPACE_DIR / ".github/workflows/ci.yml"
    assert ci_path.exists(), f"CI file not found at {ci_path}"
    content = ci_path.read_text(encoding="utf-8")
    
    violations = []
    for cmd in LEGACY_COMMANDS:
        if cmd in content:
            violations.append(cmd)
            
    assert not violations, f"Legacy commands found in CI: {violations}"

@pytest.mark.hygiene
def test_no_legacy_entrypoints_in_docs():
    """Verify that legacy commands only appear in proper deprecation/migration context."""
    files_to_check = [
        WORKSPACE_DIR / "README.md",
        ROOT_DIR / "README.md",
        ROOT_DIR / "docs/testing.md",
        ROOT_DIR / "docs/quick_start.md",
        ROOT_DIR / "REFERENCE_GUIDE.md",
    ]
    
    violations_context = []
    violations_blocks = []
    
    for f in files_to_check:
        if not f.exists():
            continue
            
        content = f.read_text(encoding="utf-8", errors="ignore")
        
        # Rule C: Check code blocks (bash, sh, shell)
        code_blocks = re.findall(r"```(?:bash|sh|shell)\n(.*?)```", content, re.DOTALL)
        for block in code_blocks:
            for cmd in LEGACY_COMMANDS:
                if cmd in block:
                    violations_blocks.append(f"{f.name} (code block) -> '{cmd}'")
                    
        # Rule B: Context check for every line containing a legacy command
        for line_num, line in enumerate(content.splitlines(), 1):
            for cmd in LEGACY_COMMANDS:
                if cmd in line:
                    # Check if line contains any of the context words (case-insensitive)
                    has_context = any(word.lower() in line.lower() for word in CONTEXT_WORDS)
                    if not has_context:
                        violations_context.append(f"{f.name}:L{line_num} -> '{line.strip()}' (missing context for '{cmd}')")
                        
    assert not violations_blocks, (
        "Legacy commands found inside active bash/sh/shell blocks (not allowed):\n"
        + "\n".join(violations_blocks)
    )
    
    assert not violations_context, (
        "Legacy commands found without explicit deprecation context words:\n"
        + "\n".join(violations_context)
    )
