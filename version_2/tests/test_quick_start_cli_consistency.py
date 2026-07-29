from __future__ import annotations

import re
import tomllib
from pathlib import Path
import pytest

@pytest.mark.hygiene
def test_quick_start_mventions_unified_command_only():
    version_2_root = Path(__file__).resolve().parents[1]
    quick_start_path = version_2_root / "docs" / "quick_start.md"
    assert quick_start_path.exists(), f"Could not find quick_start.md at {quick_start_path}"

    with open(quick_start_path, "r", encoding="utf-8") as f:
        content = f.read()

    # A. Quick Start mentions hidden-attractors
    assert "hidden-attractors" in content, "quick_start.md should mention the unified 'hidden-attractors' command"

    # B. Executable lines use the unified command token.
    code_blocks = re.findall(r"```(?:bash|sh|shell)\n(.*?)```", content, re.DOTALL)
    for block in code_blocks:
        for line in block.splitlines():
            token = line.strip().split(maxsplit=1)[0] if line.strip() else ""
            assert not token.startswith("hidden-attractors-"), (
                f"quick_start.md uses unsupported executable token '{token}'"
            )

    # C. The installed console-script surface is exact.
    pyproject_path = version_2_root / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        project_data = tomllib.load(f)
    scripts = project_data.get("project", {}).get("scripts", {})

    assert scripts == {"hidden-attractors": "hidden_attractors.cli.main:main"}
