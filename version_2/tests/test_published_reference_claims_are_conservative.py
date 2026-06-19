from __future__ import annotations

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

def test_published_reference_claims_are_conservative() -> None:
    # 1. Check published_reference_coverage.json
    coverage_path = ROOT / "validation" / "published_reference_coverage.json"
    assert coverage_path.is_file(), f"Missing coverage data at {coverage_path}"
    
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage_map = {(item["article_id"], item["case_id"]): item for item in coverage}
    
    # Kuznetsov 2017 case 18
    kuz_key = ("kuznetsov2017_chua_integer_df", "kuznetsov2017_case_18_hidden_chaotic")
    assert kuz_key in coverage_map
    assert coverage_map[kuz_key]["coverage_status"] == "executable_regression"
    assert coverage_map[kuz_key]["full_published_attractor_reproduced"] is True
    
    # Danca 2017 Chua
    danca_key = ("danca2017_fractional_hidden_attractors", "chua_fractional_saturation")
    assert danca_key in coverage_map
    assert coverage_map[danca_key]["coverage_status"] == "partial_reference_implementation"
    assert coverage_map[danca_key]["full_published_attractor_reproduced"] is False
    
    # Wu 2023 arctan
    wu_key = ("wu2023_chua_fractional_arctan", "wu2023_chua_fractional_arctan")
    assert wu_key in coverage_map
    assert coverage_map[wu_key]["coverage_status"] == "partial_reference_implementation"
    assert coverage_map[wu_key]["full_published_attractor_reproduced"] is False

    # 2. Check scientific_scope.md
    scope_path = ROOT / "docs" / "scientific_scope.md"
    assert scope_path.is_file(), f"Missing {scope_path}"
    scope_text = scope_path.read_text(encoding="utf-8")
    
    assert "## What the library can reproduce" not in scope_text
    assert "Published reference coverage" in scope_text
    assert "partial reference implementation" in scope_text.lower()
    
    # 3. Check USER_MANUAL.md
    manual_path = ROOT / "USER_MANUAL.md"
    assert manual_path.is_file(), f"Missing {manual_path}"
    manual_text = manual_path.read_text(encoding="utf-8")
    
    # Check that Danca 2017 and Wu 2023 are not claimed as fully reproduced in the manual
    assert "Wu 2023 is fully reproduced" not in manual_text
    assert "Danca 2017 is fully reproduced" not in manual_text
    assert "Wu 2023" in manual_text
    assert "Danca 2017" in manual_text
