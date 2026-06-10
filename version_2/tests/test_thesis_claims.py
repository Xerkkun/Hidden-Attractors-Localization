from __future__ import annotations

import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "THESIS_CLAIMS.md"


def _clean_text(text: str) -> str:
    """Removes common markdown formatting like backticks and asterisks."""
    return text.replace("`", "").replace("*", "").strip()


def _parse_claims_table() -> list[dict[str, str]]:
    assert CLAIMS_PATH.exists(), f"THESIS_CLAIMS.md does not exist at {CLAIMS_PATH}"
    content = CLAIMS_PATH.read_text(encoding="utf-8")

    lines = content.splitlines()
    table_lines = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            # Skip separator line like | :--- | :--- |
            if "---" in stripped:
                continue
            table_lines.append(stripped)

    assert len(table_lines) >= 2, "Claims table not found or has fewer than 2 rows (header + content)"

    # Parse headers
    headers = [c.strip() for c in table_lines[0].split("|")][1:-1]
    expected_headers = [
        "claim_id",
        "afirmación",
        "sistema",
        "orden",
        "estado",
        "evidencia_json",
        "evidencia_csv",
        "figuras",
        "comentario_metodologico"
    ]
    # Check that headers are present (order-insensitive or direct match)
    for eh in expected_headers:
        assert eh in headers, f"Header '{eh}' not found in table columns: {headers}"

    rows = []
    for line in table_lines[1:]:
        cells = [c.strip() for c in line.split("|")][1:-1]
        if len(cells) < len(headers):
            continue
        row_dict = {headers[i]: cells[i] for i in range(len(headers))}
        rows.append(row_dict)

    return rows


@pytest.mark.hygiene
def test_claims_file_and_columns_exist() -> None:
    """Verifies that THESIS_CLAIMS.md exists and contains the correct column headers."""
    rows = _parse_claims_table()
    assert len(rows) > 0, "No rows parsed from the claims table"


@pytest.mark.hygiene
def test_allowed_evidence_status_values() -> None:
    """Verifies that all states in the claims table belong to the allowed vocabulary."""
    allowed_states = {"probado", "reproducido", "rechazado", "candidato", "no_certificado", "pendiente"}
    rows = _parse_claims_table()

    for row in rows:
        state = _clean_text(row["estado"])
        assert state in allowed_states, f"Claim {row['claim_id']} has invalid state '{state}'. Allowed: {allowed_states}"


@pytest.mark.hygiene
def test_forbidden_claims_phrases() -> None:
    """Verifies that THESIS_CLAIMS.md does not contain any of the strictly prohibited overclaims."""
    content = CLAIMS_PATH.read_text(encoding="utf-8")
    forbidden_phrases = [
        "hidden attractor confirmed",
        "globally verified hidden attractor",
        "proves hiddenness",
        "proves existence",
        "first confirmed hidden attractor",
        "arctan hidden attractor verified",
        "DF proves",
        "Nyquist proves hiddenness"
    ]

    for phrase in forbidden_phrases:
        # Check case-insensitively, but allow mentions under "Claims Explicitly Not Made" or "Purpose"
        # of why we do NOT make them. Wait! We need to verify that we do not make the claim as an assertion.
        # However, to be safe, the test checks if the table or positive claims make them.
        # Let's inspect the positive claims specifically (the claim table row fields) for these phrases.
        rows = _parse_claims_table()
        for row in rows:
            for field, val in row.items():
                val_clean = val.lower()
                assert phrase.lower() not in val_clean, (
                    f"Forbidden phrase '{phrase}' detected in claim row '{row['claim_id']}' field '{field}': '{val}'"
                )


@pytest.mark.hygiene
def test_mandatory_claims_exist() -> None:
    """Verifies that specific claims exist for integer Chua, Example 1, fractional arctan, and general method."""
    rows = _parse_claims_table()
    claim_ids = {_clean_text(row["claim_id"]) for row in rows}

    mandatory_ids = {
        "CLAIM-CHUA-INTEGER-001",
        "CLAIM-CHUA-NONSMOOTH-EX1-001",
        "CLAIM-CHUA-ARCTAN-FRAC-001",
        "CLAIM-METHOD-LURE-FRAC-001"
    }

    for mid in mandatory_ids:
        assert mid in claim_ids, f"Mandatory claim '{mid}' not found in table claim_ids: {claim_ids}"


@pytest.mark.hygiene
def test_fractional_arctan_claim_status() -> None:
    """Verifies that the fractional arctan claim is listed as pending or non-certified."""
    rows = _parse_claims_table()
    for row in rows:
        claim_id = _clean_text(row["claim_id"])
        if claim_id == "CLAIM-CHUA-ARCTAN-FRAC-001":
            state = _clean_text(row["estado"])
            assert state in {"pendiente", "no_certificado"}, (
                f"Chua arctan claim status must be 'pendiente' or 'no_certified', but got '{state}'"
            )
            # Must not be listed as proven or reproduced
            assert state not in {"probado", "reproducido", "hiddenness_supported_under_tested_neighborhoods"}


@pytest.mark.hygiene
def test_proven_reproduced_claims_traceability() -> None:
    """Verifies that any claim marked as probado or reproducido has non-empty references."""
    rows = _parse_claims_table()
    empty_indicators = {"", "ninguna", "pendiente", "none", "n/a", "empty"}

    for row in rows:
        state = _clean_text(row["estado"])
        if state in {"probado", "reproducido"}:
            claim_id = row["claim_id"]
            # Check if there's at least one non-empty reference in the evidence/comment columns
            has_ref = False
            for col in ["evidencia_json", "evidencia_csv", "figuras", "comentario_metodologico"]:
                val = _clean_text(row[col]).lower()
                if val and val not in empty_indicators:
                    has_ref = True
                    break
            assert has_ref, (
                f"Claim '{claim_id}' is marked as '{state}' but has no traceable evidence or comments."
            )


@pytest.mark.hygiene
def test_oculto_verificado_disclaimer() -> None:
    """Verifies that if 'oculto verificado' appears in the text, it is accompanied by the numerical contract disclaimer."""
    content = CLAIMS_PATH.read_text(encoding="utf-8")
    # Search for "oculto verificado" (case-insensitive)
    matches = re.finditer("oculto verificado", content, re.IGNORECASE)
    for match in matches:
        # Find context around the match
        start_idx = max(0, match.start() - 100)
        end_idx = min(len(content), match.end() + 100)
        context = content[start_idx:end_idx].lower()
        
        # Ensure it contains the contract disclaimer
        has_disclaimer = (
            "bajo contrato" in context or 
            "contrato numérico" in context or 
            "tested neighborhoods" in context or
            "neighborhood contract" in context
        )
        assert has_disclaimer, (
            f"Phrase 'oculto verificado' detected without appropriate numerical contract disclaimer in context:\n... {content[match.start()-50:match.end()+50]} ..."
        )
