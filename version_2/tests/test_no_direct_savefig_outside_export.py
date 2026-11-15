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
    "no promovido",
}

IGNORED_GENERATED_DIRS = {
    ".pytest_tmp",
    ".runtime_cache",
    ".venv",
    "build",
    "dist",
    "egg-info",
    "site",
    "source_snapshots",
    "tmp",
    "__pycache__",
}

# These are local experiment/report copies, not installed Python packages or
# canonical export entry points. Preserve their recorded source bytes; audit
# their live implementations under hidden_attractors instead. Do not extend
# this boundary to a package, workflow, CLI or maintained validation module.
NON_RUNTIME_REPORT_PREFIXES = (
    "local_reports/",
    "docs/reportes_revision_tesis/",
)


def scan_direct_exports(root: Path, exceptions: set[str]):
    """Audit every maintained source; exceptions are exact local-only files."""
    violations, unregistered = [], []
    savefig_pattern = re.compile(r"\bsavefig\s*\(")
    for path in root.glob("**/*.py"):
        relative = path.relative_to(root).as_posix()
        if ("tests" in path.parts
                or any(part in IGNORED_GENERATED_DIRS for part in path.parts)
                or relative.startswith(NON_RUNTIME_REPORT_PREFIXES)):
            continue
        if relative == ALLOWED_SAVEFIG_FILE:
            continue
        content = path.read_text(encoding="utf-8")
        if not savefig_pattern.search(content):
            continue
        promoted = any(relative == prefix or relative.startswith(prefix + "/")
                       for prefix in PROMOTED_PREFIXES)
        if promoted:
            violations.append(f"{relative} (promoted route)")
        elif relative not in exceptions:
            unregistered.append(relative)
    return violations, unregistered

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
    assert POLICY_FILE.is_file(), "The maintained figure export policy is missing"
    policy_rows = parse_policy_table()
    exceptions = set()
    
    policy_violations = []
    for r in policy_rows:
        path = r["path"]
        estado = r["estado"]
        permitido = r["permitido"]
        if path in exceptions or not path.endswith(".py") or "*" in path:
            policy_violations.append(f"{path}: exceptions must name one unique Python file")
        if not (VERSION2_DIR / path).is_file():
            policy_violations.append(f"{path}: documented exception does not exist")
        
        # Rule 4: Fail if a documented exception is allowed in promoted evidence
        if "sí" in permitido or "si" in permitido or permitido == "yes":
            policy_violations.append(f"{path}: permitido en evidencia promovida = sí")
            
        # Rule 5: only completed historical/non-promoted classifications are
        # accepted. Exploratory or pending-work inventories are not public
        # policy content.
        has_valid_state = any(s in estado for s in ALLOWED_STATES)
        if not has_valid_state:
            policy_violations.append(
                f"{path}: estado '{estado}' no es válido "
                "(debe ser legacy o no promovido)"
            )
            
        exceptions.add(path)
        
    assert not policy_violations, f"Errores de formato/seguridad en docs/figure_export_policy.md:\n" + "\n".join(policy_violations)
    
    # 2. Enforce the same boundary on all current maintained source files.
    violations, unregistered_exceptions = scan_direct_exports(VERSION2_DIR, exceptions)
                
    assert not violations, f"Llamadas directas a savefig encontradas en rutas promovidas/canónicas: {violations}"
    assert not unregistered_exceptions, (
        f"Llamadas directas a savefig encontradas en scripts legados/exploratorios no registrados "
        f"en docs/figure_export_policy.md: {unregistered_exceptions}"
    )


@pytest.mark.hygiene
def test_report_copies_do_not_exempt_a_new_maintained_validation_route(tmp_path):
    paths = ["local_reports/sealed/fuentes_congeladas/historical.py",
             "docs/reportes_revision_tesis/report_figure_copy.py",
             "hidden_attractors/validation/new_local_method.py"]
    for relative in paths:
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("figure.savefig('figure.png')\n", encoding="utf-8")
    violations, unregistered = scan_direct_exports(tmp_path, set())
    assert violations == []
    assert unregistered == ["hidden_attractors/validation/new_local_method.py"]


@pytest.mark.hygiene
def test_exact_local_exception_cannot_waive_a_promoted_workflow(tmp_path):
    relative = "hidden_attractors/workflows/new_workflow.py"
    path = tmp_path/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("figure.savefig('figure.png')\n", encoding="utf-8")
    violations, unregistered = scan_direct_exports(tmp_path, {relative})
    assert violations == [relative+" (promoted route)"]
    assert unregistered == []
