# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]  # version_2 directory

@pytest.mark.hygiene
def test_user_manual_claims_consistency():
    """Verify that USER_MANUAL.md claims are consistent with THESIS_CLAIMS.md."""
    claims_path = ROOT_DIR / "THESIS_CLAIMS.md"
    manual_path = ROOT_DIR / "USER_MANUAL.md"
    
    assert claims_path.exists(), f"THESIS_CLAIMS.md does not exist at {claims_path}"
    assert manual_path.exists(), f"USER_MANUAL.md does not exist at {manual_path}"
    
    claims_content = claims_path.read_text(encoding="utf-8")
    manual_content = manual_path.read_text(encoding="utf-8")
    
    # 1. Verify Chua arctan radius-limited status
    arctan_claim_line = None
    for line in claims_content.splitlines():
        if "CLAIM-CHUA-ARCTAN-FRAC-001" in line or "Chua fraccionario arctan" in line:
            arctan_claim_line = line
            break
            
    assert arctan_claim_line is not None, "Could not find Chua arctan claim in THESIS_CLAIMS.md"
    
    if "r <= 0.3" in arctan_claim_line.lower() or "radius-limited" in arctan_claim_line.lower():
        sections = re.split(r"##\s+\d+\.", manual_content)
        arctan_section = None
        for sec in sections:
            if "Chua arctan" in sec:
                arctan_section = sec
                break
        assert arctan_section is not None, "Could not isolate Chua arctan section in USER_MANUAL.md"
        assert "r <= 0.3" in arctan_section, "USER_MANUAL.md lacks the arctan local-radius boundary."
        assert "proved" not in arctan_section.lower(), "USER_MANUAL.md claims Chua arctan is proved globally."
        
    # 2. Verify Chua integer status
    integer_claim_line = None
    for line in claims_content.splitlines():
        if "CLAIM-CHUA-INTEGER-001" in line or "Chua entero" in line:
            integer_claim_line = line
            break
            
    assert integer_claim_line is not None, "Could not find Chua integer claim in THESIS_CLAIMS.md"
    
    if "reproducido" in integer_claim_line.lower():
        sections = re.split(r"##\s+\d+\.", manual_content)
        integer_section = None
        for sec in sections:
            if "Chua integer" in sec:
                integer_section = sec
                break
        assert integer_section is not None, "Could not isolate Chua integer section in USER_MANUAL.md"
        assert any(word in integer_section.lower() for word in ["reproduced", "reproducible", "reference", "control"]), (
            "Chua integer section in USER_MANUAL.md does not describe it as reproduced, reference, or control."
        )
        
    # 3. Verify rejected candidate handling
    rejected_claim_line = None
    for line in claims_content.splitlines():
        if "CLAIM-CHUA-FRAC-REJECTED-001" in line or "rechazado" in line:
            if "contacto" in line or "danca2017_nearby_saturation_candidate_q09998" in line:
                rejected_claim_line = line
                break
                
    if rejected_claim_line and "rechazado" in rejected_claim_line.lower():
        assert any(phrase in manual_content.lower() for phrase in [
            "self-excited", "rejected", "not promoted", "contact"
        ]), "USER_MANUAL.md does not describe rejected candidate handling or contact detection."
