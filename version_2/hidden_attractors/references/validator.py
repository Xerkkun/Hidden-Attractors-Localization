"""Validator engine for bibliographic claims and traceability matrices."""

import os
from pathlib import Path
from typing import Any, Dict, List, Union
from .registry import REFERENCE_REGISTRY
from .claims import ClaimType, CLAIM_REFERENCE_MATRIX

# Suggestions based on text keywords
KEYWORDS_SUGGESTIONS = [
    ("fractional order describing function", ClaimType.MACHADO_FDF),
    ("hidden attractor is defined", ClaimType.HIDDEN_ATTRACTOR_DEFINITION),
    ("basin does not intersect", ClaimType.HIDDEN_ATTRACTOR_DEFINITION),
    ("describing function method", ClaimType.DESCRIBING_FUNCTION_CHUA_LOCALIZATION),
    ("ABM", ClaimType.CAPUTO_ABM_INTEGRATION),
    ("Caputo", ClaimType.CAPUTO_ABM_INTEGRATION),
    ("Matignon", ClaimType.FRACTIONAL_MATIGNON_STABILITY),
    ("Machado", ClaimType.MACHADO_FDF),
    ("Guan and Xie", ClaimType.ALTERNATIVE_LOCALIZATION_METHODS),
    ("perpetual point", ClaimType.ALTERNATIVE_LOCALIZATION_METHODS),
    ("critical velocity surface", ClaimType.ALTERNATIVE_LOCALIZATION_METHODS),
    ("connecting curve", ClaimType.ALTERNATIVE_LOCALIZATION_METHODS),
    ("Wu et al.", ClaimType.FRACTIONAL_CHUA_ARCTAN_WU2023),
]


def validate_claim_references(
    claims: List[Dict[str, Any]],
    strict: bool = True
) -> Dict[str, Any]:
    """Validate project claims against the REFERENCE_REGISTRY and CLAIM_REFERENCE_MATRIX.

    Parameters
    ----------
    claims : List[Dict[str, Any]]
        List of claim dictionaries.
    strict : bool, default True
        If True, missing or unregistered references cause validation to fail.

    Returns
    -------
    result : Dict[str, Any]
        Structured validation results.
    """
    claims_total = len(claims)
    claims_valid = 0
    claims_missing_references = []
    claims_with_unregistered_references = []
    claims_with_insufficient_references = []
    references_used = set()
    required_references_missing = set()
    warnings = []
    traceability_matrix = []
    has_failed = False

    for claim in claims:
        claim_id = claim.get("claim_id", "unknown_id")
        claim_type = claim.get("claim_type")
        text = claim.get("text", "")
        references = claim.get("references", [])
        severity = claim.get("severity", "weak")
        location = claim.get("location", "unknown_location")

        # 1. Auto-suggest ClaimType if not declared
        if not claim_type:
            suggested_type = None
            for keyword, c_type in KEYWORDS_SUGGESTIONS:
                if keyword.lower() in text.lower():
                    suggested_type = c_type
                    break
            if suggested_type:
                claim_type = suggested_type
                warnings.append(
                    f"Claim '{claim_id}' lacks a claim_type; auto-suggested '{suggested_type.value}' based on keywords."
                )
            else:
                claim_type = "UNKNOWN"

        # Initialize status for this claim
        claim_status = "passed"
        claim_issues = []

        # 2. Check: Strong claims require references
        if severity == "strong" and not references:
            claim_status = "failed"
            claim_issues.append("Strong claim contains no references.")
            claims_missing_references.append(claim)
            has_failed = True

        # 3. Check: All used references must exist in registry
        unregistered = []
        for ref in references:
            references_used.add(ref)
            if ref not in REFERENCE_REGISTRY:
                unregistered.append(ref)
                has_failed = True
            else:
                # Check DOI verification warnings
                ref_info = REFERENCE_REGISTRY[ref]
                if ref_info.get("needs_doi_verification"):
                    warnings.append(f"Reference '{ref}' is pending DOI verification.")

        if unregistered:
            claim_status = "failed"
            claim_issues.append(f"Unregistered references used: {unregistered}")
            claims_with_unregistered_references.append(claim)

        # 4. Check: Matrix verification (sufficient backing references)
        if claim_type in CLAIM_REFERENCE_MATRIX:
            req_list = CLAIM_REFERENCE_MATRIX[claim_type]
            # Must provide at least one of the required references
            has_sufficient = False
            for req_ref in req_list:
                if req_ref in references:
                    has_sufficient = True
                    break
            
            if not has_sufficient and req_list:
                claim_status = "failed"
                claim_issues.append(f"Insufficient references for type '{claim_type}'. Expected at least one of: {req_list}")
                claims_with_insufficient_references.append(claim)
                # Collect missing required references
                for req_ref in req_list:
                    if req_ref not in references:
                        required_references_missing.add(req_ref)
                has_failed = True

        if claim_status == "passed":
            claims_valid += 1

        traceability_matrix.append({
            "claim_id": claim_id,
            "claim_type": claim_type,
            "severity": severity,
            "text": text,
            "references": references,
            "location": location,
            "status": claim_status,
            "issues": claim_issues
        })

    # Determine overall status
    if has_failed:
        status = "failed" if strict else "warning"
    elif warnings:
        status = "warning"
    else:
        status = "passed"

    return {
        "bibliographic_validation_status": status,
        "claims_total": claims_total,
        "claims_valid": claims_valid,
        "claims_missing_references": claims_missing_references,
        "claims_with_unregistered_references": claims_with_unregistered_references,
        "claims_with_insufficient_references": claims_with_insufficient_references,
        "references_used": sorted(list(references_used)),
        "required_references_missing": sorted(list(required_references_missing)),
        "traceability_matrix": traceability_matrix,
        "warnings": warnings
    }


def write_traceability_matrix_markdown(result: Dict[str, Any], path: Union[str, Path]) -> None:
    """Generate and write a clean markdown traceability matrix file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Bibliographic Traceability Matrix",
        "",
        f"**Overall Status**: `{result['bibliographic_validation_status'].upper()}`",
        f"**Claims Validated**: {result['claims_valid']} / {result['claims_total']}",
        "",
        "## Claims Traceability Table",
        "",
        "| Claim ID | Claim Type | Strong Claim | Required References | Provided References | Status |",
        "|---|---|---|---|---|---|",
    ]

    for row in result["traceability_matrix"]:
        strong_str = "Yes" if row["severity"] == "strong" else "No"
        req_list = CLAIM_REFERENCE_MATRIX.get(row["claim_type"], [])
        req_str = ", ".join(f"`{r}`" for r in req_list) if req_list else "None"
        prov_str = ", ".join(f"`{r}`" for r in row["references"]) if row["references"] else "None"
        
        status_emoji = "✅ PASS" if row["status"] == "passed" else "❌ FAIL"
        lines.append(
            f"| {row['claim_id']} | {row['claim_type']} | {strong_str} | {req_str} | {prov_str} | {status_emoji} |"
        )

    lines.extend([
        "",
        "## Registered References",
        "",
    ])

    for key, ref in REFERENCE_REGISTRY.items():
        doi_str = ref.get("doi") or "None"
        url_str = ref.get("url") or "None"
        verification_status = ref.get("verification_status", "pending")
        
        lines.extend([
            f"### `{key}` - {ref.get('label')}",
            f"- **Authors**: {ref.get('authors')}",
            f"- **Year**: {ref.get('year')}",
            f"- **Title**: *{ref.get('title')}*",
            f"- **Journal/Venue**: {ref.get('journal', 'N/A')}",
            f"- **DOI**: `{doi_str}`",
            f"- **URL**: {url_str}",
            f"- **Topics**: {', '.join(ref.get('topics', []))}",
            f"- **Verification Status**: `{verification_status}`",
            "",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")


def validate_bibliography_manifest(
    manifest_path: Union[str, Path],
    strict: bool = True
) -> Dict[str, Any]:
    """Load and validate the bibliography manifest file."""
    import yaml
    from ..paths import PROJECT_ROOT

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        alt_path = PROJECT_ROOT / manifest_path
        if alt_path.exists():
            manifest_path = alt_path
        else:
            alt_path = Path(os.getcwd()) / manifest_path
            if alt_path.exists():
                manifest_path = alt_path

    if not manifest_path.exists():
        return {
            "bibliographic_validation_status": "failed" if strict else "warning",
            "claims_total": 0,
            "claims_valid": 0,
            "claims_missing_references": [],
            "claims_with_unregistered_references": [],
            "claims_with_insufficient_references": [],
            "references_used": [],
            "required_references_missing": [],
            "traceability_matrix": [],
            "warnings": [f"Claims manifest file not found: {manifest_path}"]
        }

    with open(manifest_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    claims = data.get("claims", [])
    return validate_claim_references(claims, strict=strict)

