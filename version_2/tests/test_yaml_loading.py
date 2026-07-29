from __future__ import annotations

import sys
import pytest
from pathlib import Path

# Add workspace root and version_2 to sys.path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.cli.run import PRESETS, TEMPLATES


def test_public_configs_have_no_runnable_presets():
    from hidden_attractors.paths import list_packaged_example_configs

    assert PRESETS == {}
    assert TEMPLATES == {"workflow_contract": "workflow_contract.yaml"}
    assert list_packaged_example_configs() == ["workflow_contract.yaml"]
