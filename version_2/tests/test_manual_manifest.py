# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory

def load_manifest():
    manifest_path = ROOT_DIR / "docs" / "manual_manifest.yaml"
    assert manifest_path.exists(), f"docs/manual_manifest.yaml not found at {manifest_path}"
    
    # Try importing yaml, fallback to a simple custom parser if not available
    try:
        import yaml
        with open(manifest_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        # Simple parser for manual_manifest.yaml
        content = manifest_path.read_text(encoding="utf-8")
        data = {}
        current_dict = None
        current_list = None
        
        for line in content.splitlines():
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith("#"):
                continue
                
            # List item parsing
            if line_stripped.startswith("-"):
                val = line_stripped[1:].strip().strip('"\'')
                if current_list is not None:
                    current_list.append(val)
                continue
                
            # Key-value parsing
            parts = line_stripped.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip()
                
                # Check indentation to determine hierarchy
                indent = len(line) - len(line.lstrip())
                
                if indent > 0:
                    # Nested key-value
                    if current_dict is not None:
                        val_clean = val.strip('"\'')
                        if val_clean.isdigit():
                            val_clean = int(val_clean)
                        current_dict[key] = val_clean
                else:
                    # Top-level key-value
                    if not val:
                        # Start of list or dict
                        if key in ["forbidden_public_claims", "public_evidence_labels"]:
                            data[key] = []
                            current_list = data[key]
                            current_dict = None
                        else:
                            data[key] = {}
                            current_dict = data[key]
                            current_list = None
                    else:
                        val_clean = val.strip('"\'')
                        if val_clean.isdigit():
                            val_clean = int(val_clean)
                        data[key] = val_clean
                        current_dict = None
                        current_list = None
        return data

@pytest.mark.hygiene
def test_manual_manifest_keys():
    """Verify that docs/manual_manifest.yaml exists and contains all core keys."""
    data = load_manifest()
    
    required_keys = [
        "manual_version",
        "package_version",
        "public_cli",
        "entry_point",
        "freeze_audit",
        "claims_source",
        "canonical_figures",
        "manual_targets",
        "scientific_scope",
        "claim_status_summary",
        "forbidden_public_claims",
        "public_evidence_labels",
        "documentation_policy"
    ]
    
    for key in required_keys:
        assert key in data, f"Key '{key}' is missing from manual_manifest.yaml"

@pytest.mark.hygiene
def test_manual_manifest_values():
    """Verify specific fields and values inside manual_manifest.yaml."""
    data = load_manifest()
    
    assert data["manual_version"] == "1.0.0"
    assert data["package_version"] == "1.0.0"

    # Assert entry points
    assert data["public_cli"] == "hidden-attractors", "public_cli must be exactly 'hidden-attractors'"
    assert data["entry_point"] == "hidden_attractors.cli.main:main", "entry_point must be 'hidden_attractors.cli.main:main'"
    
    # Assert claims source
    claims_file = ROOT_DIR / data["claims_source"]
    assert claims_file.exists(), f"Claims source file not found at {claims_file}"
    
    # Assert freeze audit
    audit = data["freeze_audit"]
    audit_dir = ROOT_DIR / audit["path"]
    assert audit_dir.exists(), f"Freeze audit directory not found at {audit_dir}"
    
    # Check that freeze counts match stdout
    stdout_file = ROOT_DIR / audit["stdout_txt"]
    assert stdout_file.exists(), f"Freeze audit stdout file not found at {stdout_file}"
    stdout_content = stdout_file.read_text(encoding="utf-8")
    
    expected_passed = f"{audit['passed']} passed"
    expected_skipped = f"{audit['skipped']} skipped"
    
    assert expected_passed in stdout_content, f"Expected '{expected_passed}' in freeze audit stdout but not found"
    assert expected_skipped in stdout_content, f"Expected '{expected_skipped}' in freeze audit stdout but not found"
    
    # Assert canonical figures
    assert data["canonical_figures"] == "library_figures/", "canonical_figures must be exactly 'library_figures/'"
    
    # Assert manual targets
    targets = data["manual_targets"]
    latex_path = ROOT_DIR / targets["latex"]
    assert latex_path.exists(), f"LaTeX target report not found at {latex_path}"
    
    # Assert claim status summary
    status = data["claim_status_summary"]
    assert status["chua_arctan_fractional"] == "promovido_con_limite_de_radio"
    
    # Assert forbidden public claims
    forbidden = data["forbidden_public_claims"]
    required_forbidden = [
        "DF proves hiddenness",
        "globally verified hidden attractor",
        "Chua arctan hidden attractor verified"
    ]
    for claim in required_forbidden:
        assert claim in forbidden, f"Forbidden claim '{claim}' is missing from forbidden_public_claims"
