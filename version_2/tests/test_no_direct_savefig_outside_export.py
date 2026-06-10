# -*- coding: utf-8 -*-
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
VERSION2_DIR = ROOT_DIR / "version_2"
POLICY_FILE = VERSION2_DIR / "docs" / "figure_export_policy.md"

PROMOTED_PREFIXES = [
    "hidden_attractors/plotting",
    "hidden_attractors/workflows",
    "examples",
    "tools/cli",
]

ALLOWED_SAVEFIG_FILE = "hidden_attractors/plotting/export.py"

ALLOWED_STATES = {
    "legacy",
    "exploratorio",
    "no promovido",
    "pendiente de migración",
    "pendiente",
}

def parse_policy_table() -> list[dict[str, str]]:
    """Parse the exceptions table from docs/figure_export_policy.md."""
    rows = []
    if not POLICY_FILE.exists():
        return rows
        
    content = POLICY_FILE.read_text(encoding="utf-8")
    in_table = False
    
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("|") and "Archivo" in line and "Motivo" in line:
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                in_table = False
                continue
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if not parts or all(x == "" or x.startswith("---") for x in parts):
                continue
                
            # Columns: Archivo | Motivo | Estado | Permitido en evidencia promovida
            if len(parts) >= 4:
                file_path = parts[0].replace("`", "").strip()
                motivo = parts[1]
                estado = parts[2].lower()
                permitido = parts[3].lower()
                
                rows.append({
                    "path": file_path,
                    "motivo": motivo,
                    "estado": estado,
                    "permitido": permitido
                })
    return rows

@pytest.mark.plotting
@pytest.mark.hygiene
def test_no_direct_savefig():
    """Verify that savefig is not called directly outside export.py and exceptions are strictly validated."""
    # 1. Parse and validate policy exceptions table
    policy_rows = parse_policy_table()
    exceptions = set()
    
    policy_violations = []
    for r in policy_rows:
        path = r["path"]
        estado = r["estado"]
        permitido = r["permitido"]
        
        # Rule 4: Fail if a documented exception is allowed in promoted evidence
        if "sí" in permitido or "si" in permitido or permitido == "yes":
            policy_violations.append(f"{path}: permitido en evidencia promovida = sí")
            
        # Rule 5: Fail if the state is not legacy/exploratorio/no promovido/pendiente de migración
        has_valid_state = any(s in estado for s in ALLOWED_STATES)
        if not has_valid_state:
            policy_violations.append(f"{path}: estado '{estado}' no es válido (debe ser legacy/exploratorio/no promovido/pendiente de migración)")
            
        exceptions.add(path)
        
    assert not policy_violations, f"Errores de formato/seguridad en docs/figure_export_policy.md:\n" + "\n".join(policy_violations)
    
    # 2. Scan python scripts for savefig calls
    py_files = list(VERSION2_DIR.glob("**/*.py"))
    savefig_pattern = re.compile(r"\bsavefig\s*\(")
    
    violations = []
    unregistered_exceptions = []
    
    for f in py_files:
        try:
            rel_path = f.relative_to(VERSION2_DIR).as_posix()
        except ValueError:
            continue
            
        # Skip tests, pycache, build, egg-info
        if "tests" in f.parts or "__pycache__" in f.parts or "build" in f.parts or "egg-info" in f.parts:
            continue
            
        # Skip export.py
        if rel_path == ALLOWED_SAVEFIG_FILE:
            continue
            
        try:
            content = f.read_text(encoding="utf-8")
        except Exception:
            continue
            
        if savefig_pattern.search(content):
            # Check if it is a promoted route
            is_promoted = any(rel_path.startswith(prefix) for prefix in PROMOTED_PREFIXES)
            
            if is_promoted:
                violations.append(f"{rel_path} (promoted route)")
            elif rel_path not in exceptions:
                # Rule 3: Fail if direct savefig is called in a file not documented in policy
                unregistered_exceptions.append(rel_path)
                
    assert not violations, f"Llamadas directas a savefig encontradas en rutas promovidas/canónicas: {violations}"
    assert not unregistered_exceptions, (
        f"Llamadas directas a savefig encontradas en scripts legados/exploratorios no registrados "
        f"en docs/figure_export_policy.md: {unregistered_exceptions}"
    )
