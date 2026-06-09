from __future__ import annotations

import sys
from pathlib import Path

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))


def test_no_redundant_public_scripts_in_pyproject():
    pyproject_path = Path(workspace_root) / "version_2" / "pyproject.toml"
    assert pyproject_path.exists()

    # Parse pyproject.toml manually to find project.scripts
    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Simple line-by-line parser for [project.scripts] block
    in_scripts_section = False
    script_keys = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[project.scripts]":
            in_scripts_section = True
            continue
        if line.startswith("[") and line != "[project.scripts]":
            in_scripts_section = False
            continue

        if in_scripts_section:
            if "=" in line:
                key = line.split("=")[0].strip()
                script_keys.append(key)

    # Allowed keys
    allowed_keys = {
        "hidden-attractors",
        "hidden-attractors-list-candidates",
        "hidden-attractors-systems",
        "hidden-attractors-workflow-requirements",
        "hidden-attractors-check-validation",
        "hidden-attractors-protocol",
        "hidden-attractors-robustness-overlay",
        "hidden-attractors-sphere-controls",
        "hidden-attractors-refined-basin",
        "hidden-attractors-strict-target-refinement",
        "hidden-attractors-danca-abm-sphere-controls",
        "hidden-attractors-fractional-report-run",
    }

    # Verify that every script defined is in the allowed keys list (no new scripts)
    for key in script_keys:
        assert key in allowed_keys, f"Found unregistered redundant script key in pyproject.toml: {key}"

    # Also make sure the new features like bifurcation/lyapunov/chaos-test are not present
    for key in script_keys:
        assert "bifurcation" not in key
        assert "lyapunov" not in key
        assert "chaos" not in key
        assert "zero" not in key
