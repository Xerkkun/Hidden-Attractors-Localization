from __future__ import annotations

import re
import tomllib
from pathlib import Path
import pytest

LEGACY_COMMANDS = {
    "hidden-attractors-protocol",
    "hidden-attractors-robustness-overlay",
    "hidden-attractors-sphere-controls",
    "hidden-attractors-refined-basin",
    "hidden-attractors-strict-target-refinement",
    "hidden-attractors-danca-abm-sphere-controls",
    "hidden-attractors-fractional-report-run",
    "hidden-attractors-list-candidates",
    "hidden-attractors-systems",
    "hidden-attractors-workflow-requirements",
    "hidden-attractors-check-validation",
}

@pytest.mark.hygiene
def test_quick_start_mventions_unified_command_only():
    version_2_root = Path(__file__).resolve().parents[1]
    quick_start_path = version_2_root / "docs" / "quick_start.md"
    assert quick_start_path.exists(), f"Could not find quick_start.md at {quick_start_path}"

    with open(quick_start_path, "r", encoding="utf-8") as f:
        content = f.read()

    # A. Quick Start mentions hidden-attractors
    assert "hidden-attractors" in content, "quick_start.md should mention the unified 'hidden-attractors' command"

    # B. Quick Start does not recommend legacy standalone commands (no active run blocks for them)
    # Check that they do not appear in bash/shell blocks
    code_blocks = re.findall(r"```(?:bash|sh|shell)\n(.*?)```", content, re.DOTALL)
    for block in code_blocks:
        for cmd in LEGACY_COMMANDS:
            assert cmd not in block, f"quick_start.md recommends legacy command '{cmd}' in code block: {block}"

    # C. Old commands can only appear in migration guides, specific tests, or explicit deprecation notes.
    # In quick_start.md, they only appear as explicit deprecation notes (e.g. "ya no se instalan", "No ejecutar comandos legacy")
    for line in content.splitlines():
        for cmd in LEGACY_COMMANDS:
            if cmd in line:
                lower_line = line.lower()
                assert any(depr in lower_line for depr in ["ya no", "legacy", "antiguos", "no ejecutar", "deprecación", "históricos", "no longer", "deprecated", "obsolete"]), \
                    f"Line contains legacy command '{cmd}' without explicit deprecation context: '{line}'"

    # D. Sync check: If pyproject.toml scripts only has hidden-attractors, verify consistency
    pyproject_path = version_2_root / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        project_data = tomllib.load(f)
    scripts = project_data.get("project", {}).get("scripts", {})

    if set(scripts.keys()) == {"hidden-attractors"}:
        # Make sure no other command is recommended
        for cmd in LEGACY_COMMANDS:
            assert cmd not in scripts
