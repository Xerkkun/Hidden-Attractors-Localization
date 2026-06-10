# -*- coding: utf-8 -*-
import pytest
import re
from pathlib import Path
from tests.helpers.test_documentation_text import read, ROOT, get_violations_without_context

ARCTAN_CLARIFICATIONS = [
    "implemented algebraically", "pending full hiddenness validation", "pending",
    "non-certified", "pendiente", "no certificado", "implementado algebraicamente",
    "pendiente de validación completa de ocultedad", "pendiente de validación completa"
]

MACHADO_CLARIFICATIONS = [
    "documented as theory", "planned seed family", "not a promoted public workflow",
    "not stable public workflow", "teoría", "familia de semillas planificada",
    "no promovido", "no estable", "no es un flujo público promovido", "planned", "theory",
    "not stable", "teóricos", "teóricas", "planificadas", "planificados", "teórico", "teórica",
    "planificada"
]

FORBIDDEN_ARCTAN_CLAIMS = [
    "chua arctan hidden attractor verified",
    "arctan hiddenness verified",
    "arctan hidden attractor confirmed",
    "arctan hidden attractor proved",
    "arctan globally verified",
    "atractor oculto arctan verificado",
    "ocultedad arctan verificada",
    "atractor oculto arctan probado",
]

FORBIDDEN_MACHADO_CLAIMS = [
    "machado workflow promoted",
    "machado stable public workflow",
    "machado implemented stable workflow",
    "fdf stable public workflow",
    "fdf promoted workflow",
    "flujo público estable machado",
    "flujo promovido machado",
]

FORBIDDEN_PROOF_CLAIMS = [
    "df proves hiddenness",
    "nyquist proves hiddenness",
    "continuation proves hiddenness",
    "bounded simulation proves hiddenness",
    "figure proves hiddenness",
]

def clean_text_for_claims(text: str) -> str:
    # 1. Remove bibliography section in LaTeX
    text = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", text, flags=re.DOTALL)
    # 2. Remove citations like \cite{...}
    text = re.sub(r"\\cite\{[^\}]*\}", "", text)
    # 3. Remove file paths/names with extensions (.pdf, .png, etc.)
    text = re.sub(r"[\w/\.-]+\.(?:pdf|png|jpg|md|yaml|tex|wl)\b", "", text)
    # 4. Remove codepath markup like \codepath{...} or \texttt{...}
    text = re.sub(r"\\(?:codepath|texttt|maybeincludegraphics|ref|label)\{[^\}]*\}", "", text)
    # 5. Remove markdown links [text](url) -> keep text only
    text = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", text)
    return text

@pytest.mark.hygiene
def test_manual_claims_consistency_check():
    claims_path = ROOT / "THESIS_CLAIMS.md"
    assert claims_path.exists(), f"THESIS_CLAIMS.md not found at {claims_path}"
    claims_content = read(claims_path)
    
    # 1. Determine if Chua arctan is pending/non-certified in THESIS_CLAIMS.md
    arctan_claim_line = None
    for line in claims_content.splitlines():
        if "CLAIM-CHUA-ARCTAN-FRAC-001" in line or "Chua fraccionario arctan" in line:
            arctan_claim_line = line
            break
            
    assert arctan_claim_line is not None, "Could not locate Chua arctan claim in THESIS_CLAIMS.md"
    
    is_arctan_pending = any(w in arctan_claim_line.lower() for w in ["pendiente", "no_certificado", "pending", "non-certified"])
    
    manuals = [
        ROOT / "USER_MANUAL.md",
        ROOT / "README.md",
        ROOT.parent / "README.md",
        ROOT / "REFERENCE_GUIDE.md",
        ROOT / "docs/quick_start.md",
        ROOT / "docs/validation_evidence.md",
        ROOT / "docs/unified_report.md",
        ROOT / "docs/reporte_unificado_chua_fraccionario.tex",
    ]
    manuals = [p for p in manuals if p.exists()]
    
    violations = []
    
    for p in manuals:
        raw_content = read(p)
        content = clean_text_for_claims(raw_content)
        content_lower = content.lower()
        
        # Rule 1: No false verified claims for arctan if pending
        if is_arctan_pending:
            for claim in FORBIDDEN_ARCTAN_CLAIMS:
                if claim in content_lower:
                    violations.append(f"{p.name} -> Contains forbidden verified claim: '{claim}'")
                    
        # Rule 2: Chua arctan mentions must be near clarification terms (within ±300 chars)
        for match in re.finditer(r"chua\s+arctan", content_lower):
            pos = match.start()
            win_start = max(0, pos - 300)
            win_end = min(len(content), pos + len(match.group(0)) + 300)
            sub_window = content_lower[win_start:win_end]
            if not any(term.lower() in sub_window for term in ARCTAN_CLARIFICATIONS):
                line_num = raw_content[:pos].count('\n') + 1
                violations.append(
                    f"{p.name}:L{line_num} -> Mention of '{match.group(0)}' lacks pending/non-certified clarification context."
                )
                
        # Rule 3: Machado / FDF mentions must be near clarification terms (within ±300 chars)
        for match in re.finditer(r"machado|fdf", content_lower):
            pos = match.start()
            win_start = max(0, pos - 300)
            win_end = min(len(content), pos + len(match.group(0)) + 300)
            sub_window = content_lower[win_start:win_end]
            if not any(term.lower() in sub_window for term in MACHADO_CLARIFICATIONS):
                line_num = raw_content[:pos].count('\n') + 1
                violations.append(
                    f"{p.name}:L{line_num} -> Mention of '{match.group(0)}' lacks theoretical/planned clarification context."
                )
                
        # Rule 4: Fails on Machado/FDF as promoted stable workflow
        for claim in FORBIDDEN_MACHADO_CLAIMS:
            if claim in content_lower:
                violations.append(f"{p.name} -> Claimed Machado/FDF as promoted public/stable workflow: '{claim}'")
                
        # Rule 5: Fails on false proof claims
        for claim in FORBIDDEN_PROOF_CLAIMS:
            if claim in content_lower:
                violations.append(f"{p.name} -> Claimed approximate method proves hiddenness: '{claim}'")
                
    assert not violations, (
        "Scientific claims inconsistency found in active documentation:\n"
        + "\n".join(violations)
    )
