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
        "claim",
        "system",
        "order",
        "status",
        "json_evidence",
        "csv_evidence",
        "figures",
        "methodological_comment"
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
    allowed_states = {"validated", "reproduced", "rejected", "candidate", "partial", "pending"}
    rows = _parse_claims_table()

    for row in rows:
        state = _clean_text(row["status"])
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
        # of why we do NOT make them.
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
        "CLAIM-CHUA-FRAC-REJECTED-001",
        "CLAIM-CHUA-ARCTAN-FRAC-001",
        "CLAIM-METHOD-LURE-FRAC-001"
    }

    for mid in mandatory_ids:
        assert mid in claim_ids, f"Mandatory claim '{mid}' not found in table claim_ids: {claim_ids}"


@pytest.mark.hygiene
def test_fractional_arctan_claim_status() -> None:
    """Verifies that the fractional arctan claim is promoted with a radius-limited boundary."""
    rows = _parse_claims_table()
    for row in rows:
        claim_id = _clean_text(row["claim_id"])
        if claim_id == "CLAIM-CHUA-ARCTAN-FRAC-001":
            state = _clean_text(row["status"])
            assert state == "validated"
            assert "validation/chua_fractional_arctan_c590/validation_summary.json" in row["json_evidence"]
            assert "r <= 0.3" in row["methodological_comment"]


@pytest.mark.hygiene
def test_proven_reproduced_claims_traceability() -> None:
    """Verifies that any claim marked as validated or reproduced has non-empty references."""
    rows = _parse_claims_table()
    empty_indicators = {"", "ninguna", "pendiente", "none", "n/a", "empty"}

    for row in rows:
        state = _clean_text(row["status"])
        if state in {"validated", "reproduced"}:
            claim_id = row["claim_id"]
            # Check if there's at least one non-empty reference in the evidence/comment columns
            has_ref = False
            for col in ["json_evidence", "csv_evidence", "figures", "methodological_comment"]:
                val = _clean_text(row[col]).lower()
                if val and val not in empty_indicators:
                    has_ref = True
                    break
            assert has_ref, (
                f"Claim '{claim_id}' is marked as '{state}' but has no traceable evidence or comments."
            )


@pytest.mark.hygiene
def test_no_legacy_labels_and_only_english() -> None:
    content = CLAIMS_PATH.read_text(encoding="utf-8")
    
    # 1. No legacy labels
    legacy_labels = [
        "hidden_verified",
        "chaos_verified",
            "self_excited_contact_detected",
        "hiddenness_inconclusive",
        "candidate_rejected"
    ]
    for label in legacy_labels:
        assert label not in content, f"THESIS_CLAIMS.md contains legacy label: '{label}'"

    # 2. No Spanish keywords in table/headers
    spanish_keywords = ["afirmación", "sistema", "orden", "estado", "evidencia_json", "evidencia_csv", "figuras", "comentario_metodologico"]
    for word in spanish_keywords:
        assert word not in content.lower(), f"THESIS_CLAIMS.md contains Spanish keyword: '{word}'"

    # 3. Does not claim full reproduction of Danca 2017 or Wu 2023 in the claims table
    rows = _parse_claims_table()
    for row in rows:
        claim_text = row["claim"].lower()
        if "danca" in claim_text:
            assert not any(phrase in claim_text for phrase in ["full reproduction", "fully reproduced", "fully validated", "complete reproduction"]), \
                f"Claim {row['claim_id']} overclaims Danca reproduction: '{row['claim']}'"
        if "wu" in claim_text or "arctan" in claim_text:
            assert not any(phrase in claim_text for phrase in ["full reproduction", "fully reproduced", "fully validated", "complete reproduction", "verified arctan hidden"]), \
                f"Claim {row['claim_id']} overclaims Wu/arctan reproduction: '{row['claim']}'"

    # 4. Keeps Chua integer as the only strong reproduced reference case
    for row in rows:
        state = _clean_text(row["status"])
        if state == "reproduced":
            assert _clean_text(row["claim_id"]) == "CLAIM-CHUA-INTEGER-001", \
                f"Only CLAIM-CHUA-INTEGER-001 should be marked as reproduced, but found {row['claim_id']}"

