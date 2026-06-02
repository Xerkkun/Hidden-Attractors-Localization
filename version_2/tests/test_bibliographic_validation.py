"""Unit tests for bibliographic validation and registry matching."""

import sys
from pathlib import Path
import pytest

# Add version_2 to path if needed
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.references.registry import REFERENCE_REGISTRY
from hidden_attractors.references.claims import ClaimType, CLAIM_REFERENCE_MATRIX
from hidden_attractors.references.validator import (
    validate_claim_references,
    write_traceability_matrix_markdown,
    validate_bibliography_manifest,
)


# 1. Verify registry entries (8 mandatory academic references)
def test_registry_entries():
    mandatory_keys = [
        "leonov_kuznetsov_hidden_definition",
        "kuznetsov_2017_chua_df",
        "diethelm_ford_freed_abm_caputo",
        "danca_2017_fractional_hidden",
        "matignon_fractional_stability",
        "machado_2015_fractional_describing_functions",
        "guan_xie_2025_review",
        "wu_2023_fractional_chua_arctan",
    ]
    for key in mandatory_keys:
        assert key in REFERENCE_REGISTRY
        ref = REFERENCE_REGISTRY[key]
        assert "key" in ref
        assert "label" in ref
        assert "authors" in ref
        assert "year" in ref
        assert "title" in ref
        assert "verification_status" in ref


# 2. Verify registry DOI warnings (references missing DOIs)
def test_registry_doi_warnings():
    claims = [
        {
            "claim_id": "test_matignon",
            "claim_type": "FRACTIONAL_MATIGNON_STABILITY",
            "severity": "strong",
            "text": "Using Matignon theorem for fractional stability.",
            "references": ["matignon_fractional_stability"],
        }
    ]
    res = validate_claim_references(claims, strict=True)
    # Warning should exist since matignon_fractional_stability is pending DOI verification
    assert any("matignon_fractional_stability" in w and "pending DOI" in w for w in res["warnings"])


# 3. Verify ClaimType enum contains required types
def test_claim_type_enum():
    required_types = [
        "HIDDEN_ATTRACTOR_DEFINITION",
        "SELF_EXCITED_DEFINITION",
        "DESCRIBING_FUNCTION_CHUA_LOCALIZATION",
        "DESCRIBING_FUNCTION_IS_HEURISTIC",
        "CAPUTO_ABM_INTEGRATION",
        "FRACTIONAL_MATIGNON_STABILITY",
        "MACHADO_FDF",
        "ALTERNATIVE_LOCALIZATION_METHODS",
        "FRACTIONAL_CHUA_ARCTAN_WU2023",
        "WEYL_CAPUTO_BRIDGE",
        "NONSMOOTH_CHUA_LIPSCHITZ_ABM",
        "NUMERICAL_CONTINUATION_LOCALIZATION",
        "HIDDENNESS_OPERATIONAL_VERIFICATION",
    ]
    for val in required_types:
        assert hasattr(ClaimType, val)
        assert ClaimType[val].value == val


# 4. Verify CLAIM_REFERENCE_MATRIX mapping validity
def test_claim_reference_matrix():
    for c_type, ref_list in CLAIM_REFERENCE_MATRIX.items():
        assert isinstance(c_type, ClaimType)
        assert isinstance(ref_list, list)
        for ref in ref_list:
            assert ref in REFERENCE_REGISTRY


# 5. Verify validate_claim_references returns passed for valid claims
def test_validate_claim_references_success():
    claims = [
        {
            "claim_id": "test_hidden_def",
            "claim_type": ClaimType.HIDDEN_ATTRACTOR_DEFINITION,
            "severity": "strong",
            "text": "An attractor is hidden if its basin does not intersect any equilibrium neighborhood.",
            "references": ["leonov_kuznetsov_hidden_definition"],
        }
    ]
    res = validate_claim_references(claims, strict=True)
    assert res["bibliographic_validation_status"] == "passed"
    assert res["claims_valid"] == 1


# 6. Verify validation fails for strong claim without references
def test_validate_claim_references_missing_references():
    claims = [
        {
            "claim_id": "test_missing",
            "claim_type": ClaimType.HIDDEN_ATTRACTOR_DEFINITION,
            "severity": "strong",
            "text": "An attractor is hidden.",
            "references": [],
        }
    ]
    res = validate_claim_references(claims, strict=True)
    assert res["bibliographic_validation_status"] == "failed"
    assert len(res["claims_missing_references"]) == 1
    assert res["claims_missing_references"][0]["claim_id"] == "test_missing"


# 7. Verify validation fails for unregistered reference
def test_validate_claim_references_unregistered():
    claims = [
        {
            "claim_id": "test_unregistered",
            "claim_type": ClaimType.HIDDEN_ATTRACTOR_DEFINITION,
            "severity": "strong",
            "text": "An attractor is hidden.",
            "references": ["nonexistent_reference_key"],
        }
    ]
    res = validate_claim_references(claims, strict=True)
    assert res["bibliographic_validation_status"] == "failed"
    assert len(res["claims_with_unregistered_references"]) == 1
    assert res["claims_with_unregistered_references"][0]["claim_id"] == "test_unregistered"


# 8. Verify validation fails for insufficient references (missing required reference from matrix)
def test_validate_claim_references_insufficient():
    # Provide a reference, but not the one required by the matrix for DESCRIBING_FUNCTION_CHUA_LOCALIZATION
    claims = [
        {
            "claim_id": "test_insufficient",
            "claim_type": ClaimType.DESCRIBING_FUNCTION_CHUA_LOCALIZATION,
            "severity": "strong",
            "text": "Describing functions are used.",
            "references": ["leonov_kuznetsov_hidden_definition"],
        }
    ]
    res = validate_claim_references(claims, strict=True)
    assert res["bibliographic_validation_status"] == "failed"
    assert len(res["claims_with_insufficient_references"]) == 1
    assert res["claims_with_insufficient_references"][0]["claim_id"] == "test_insufficient"


# 9. Verify keyword auto-suggestion logic
def test_auto_suggestion_logic():
    # If claim_type is omitted, auto-suggest based on keywords
    claims = [
        {
            "claim_id": "test_suggest",
            "severity": "strong",
            "text": "This text mentions fractional order describing function of Machado.",
            "references": ["machado_2015_fractional_describing_functions"],
        }
    ]
    res = validate_claim_references(claims, strict=True)
    # The check should suggest MACHADO_FDF and thus validation should succeed (but with warning because claim_type was suggested)
    assert res["bibliographic_validation_status"] == "warning"
    assert res["traceability_matrix"][0]["claim_type"] == ClaimType.MACHADO_FDF


# 10. Verify validate_bibliography_manifest with real YAML manifest
def test_validate_bibliography_manifest():
    manifest_path = workspace_root / "version_2" / "references" / "claims_manifest.yaml"
    res = validate_bibliography_manifest(manifest_path, strict=True)
    # Overall validation status should be warning because of Matignon pending DOI verification,
    # but all claims should be valid
    assert res["bibliographic_validation_status"] in ("warning", "passed")
    assert res["claims_total"] == 12
    assert res["claims_valid"] == 12


# 11. Verify traceability matrix export functionality
def test_traceability_matrix_export(tmp_path):
    manifest_path = workspace_root / "version_2" / "references" / "claims_manifest.yaml"
    res = validate_bibliography_manifest(manifest_path, strict=True)
    out_file = tmp_path / "matrix.md"
    write_traceability_matrix_markdown(res, out_file)
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Bibliographic Traceability Matrix" in content
    assert "Claims Traceability Table" in content
    assert "Registered References" in content
