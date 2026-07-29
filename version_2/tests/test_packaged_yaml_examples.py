from __future__ import annotations

import sys
import importlib.resources
import yaml
from pathlib import Path

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

def test_packaged_yamls():
    """Only a non-runnable, case-free structural contract is packaged."""
    from hidden_attractors.paths import list_packaged_example_configs, get_example_config_resource
    
    yaml_filenames = list_packaged_example_configs()
    assert yaml_filenames == ["workflow_contract.yaml"]

    ref = get_example_config_resource(yaml_filenames[0])
    with importlib.resources.as_file(ref) as local_p:
        raw_text = local_p.read_text(encoding="utf-8")
        payload = yaml.safe_load(raw_text)

    assert payload["template"]["runnable"] is False
    assert payload["template"]["scientific_claim"] == "none"
    assert payload["required_inputs"]["system"]["system_id"] is None
    assert payload["required_inputs"]["system"]["parameters"] == {}
    for prohibited in ("chua", "0.9998", "R0_base", "future", "planned"):
        assert prohibited.lower() not in raw_text.lower()
