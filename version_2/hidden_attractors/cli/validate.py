"""CLI commands for validating validation evidence contracts and claims bibliography.

Stability: internal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ..validation_contract import main as contract_main
from ..references.validator import validate_bibliography_manifest, write_traceability_matrix_markdown
import json


def validate_contract(argv: Sequence[str] | None = None) -> None:
    """Validate numerical validation evidence contract."""
    # We can delegate directly to the validation_contract module's main
    sys.exit(contract_main(argv))


def validate_bibliography(argv: Sequence[str] | None = None) -> None:
    """Validate claims bibliography manifest against bibliographic registry."""
    parser = argparse.ArgumentParser(description="Validate bibliography manifest")
    parser.add_argument("-m", "--manifest", type=str, default="version_2/references/claims_manifest.yaml", help="Path to the claims_manifest.yaml file")
    parser.add_argument("--strict", action="store_true", help="Fail with exit code 1 if bibliographic verification fails")
    parser.add_argument("--json", action="store_true", help="Output validation results in JSON format")
    parser.add_argument("-o", "--markdown-output", type=str, help="Path to write the generated markdown traceability matrix")
    
    args = parser.parse_args(argv)
    
    strict = bool(args.strict)
    manifest_path = args.manifest
    
    print(f"Validating bibliography manifest from: {manifest_path} (strict={strict})")
    
    try:
        res = validate_bibliography_manifest(manifest_path, strict=strict)
        
        # Output JSON if requested
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"Overall Validation Status: {res['bibliographic_validation_status'].upper()}")
            print(f"Total Claims: {res['claims_total']}")
            print(f"Valid Claims: {res['claims_valid']}")
            if res["warnings"]:
                print("\nWarnings:")
                for w in res["warnings"]:
                    print(f"  - {w}")
            if res["claims_missing_references"]:
                print("\nClaims missing references (failed):")
                for c in res["claims_missing_references"]:
                    print(f"  - {c.get('claim_id')}: {c.get('text')}")
            if res["claims_with_unregistered_references"]:
                print("\nClaims with unregistered references (failed):")
                for c in res["claims_with_unregistered_references"]:
                    print(f"  - {c.get('claim_id')}: {c.get('references')}")
            if res["claims_with_insufficient_references"]:
                print("\nClaims with insufficient references (failed):")
                for c in res["claims_with_insufficient_references"]:
                    print(f"  - {c.get('claim_id')}: {c.get('references')}")
                    
        # Write markdown output if requested
        if args.markdown_output:
            write_traceability_matrix_markdown(res, args.markdown_output)
            print(f"\nTraceability matrix written to: {args.markdown_output}")
            
        if res["bibliographic_validation_status"] == "failed" and strict:
            sys.exit(1)
            
    except Exception as e:
        print(f"Bibliography validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_ls_files(root: Path, *patterns: str) -> list[str]:
    import subprocess

    cmd = ["git", "ls-files", *patterns]
    result = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def validate_cpc_readiness(argv: Sequence[str] | None = None) -> None:
    """Validate CPC metadata and repository hygiene without changing science artifacts."""
    parser = argparse.ArgumentParser(description="Validate CPC readiness metadata and hygiene")
    parser.add_argument("--json", action="store_true", help="Output machine-readable results")
    args = parser.parse_args(argv)

    root = _repo_root()
    version_root = root / "version_2"

    required = [
        "CITATION.cff",
        ".zenodo.json",
        "codemeta.json",
        "AUTHORS.md",
        "CHANGELOG.md",
        "RELEASE_NOTES.md",
        "REPRODUCIBILITY.md",
        "paper/cpc_program_summary.tex",
        "paper/cpc_manuscript.tex",
        "paper/references.bib",
        "version_2/cpc_submission/README_CPC.md",
        "version_2/cpc_submission/PROGRAM_SUMMARY.md",
        "version_2/cpc_submission/SAMPLE_RUN.md",
        "version_2/cpc_submission/reproducibility_checklist.md",
        "version_2/cpc_submission/archive_manifest.json",
        "version_2/validation/freeze_audit/final_freeze_pytest_summary.json",
        "version_2/README.md",
        "version_2/USER_MANUAL.md",
    ]

    checks: list[dict[str, object]] = []
    missing = [rel for rel in required if not (root / rel).exists()]
    checks.append({"name": "required_files", "ok": not missing, "details": missing})

    validation_outputs_tracked = _git_ls_files(root, "version_2/validation_outputs")
    checks.append({
        "name": "validation_outputs_untracked",
        "ok": not validation_outputs_tracked,
        "details": validation_outputs_tracked,
    })

    promoted = ["version_2/validation/chua_integer_saturation"]
    missing_promoted = [rel for rel in promoted if not (root / rel).exists()]
    checks.append({"name": "promoted_validation_evidence", "ok": not missing_promoted, "details": missing_promoted})

    arctan_promoted = _git_ls_files(root, "version_2/validation/chua_fractional_arctan")
    checks.append({"name": "arctan_not_promoted_validation_root", "ok": not arctan_promoted, "details": arctan_promoted})

    pyproject_text = (version_root / "pyproject.toml").read_text(encoding="utf-8")
    entry_ok = 'hidden-attractors = "hidden_attractors.cli.main:main"' in pyproject_text
    legacy_entries = [
        "hidden-attractors-check-validation",
        "hidden-attractors-protocol",
        "hidden-attractors-fractional-report-run",
    ]
    legacy_public = [name for name in legacy_entries if f"{name} =" in pyproject_text]
    checks.append({"name": "single_public_entry_point", "ok": entry_ok and not legacy_public, "details": legacy_public})

    citation_path = root / "CITATION.cff"
    citation_text = citation_path.read_text(encoding="utf-8") if citation_path.exists() else ""
    doi_ok = "10.17605/OSF.IO/ZGK74" in citation_text
    checks.append({"name": "doi_recorded", "ok": doi_ok, "details": [] if doi_ok else ["10.17605/OSF.IO/ZGK74 missing"]})

    ok = all(bool(check["ok"]) for check in checks)
    payload = {"status": "passed" if ok else "failed", "checks": checks}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"CPC readiness: {payload['status']}")
        for check in checks:
            status = "ok" if check["ok"] else "FAIL"
            print(f"- {status}: {check['name']}")
            if check["details"]:
                for item in check["details"]:
                    print(f"  - {item}")

    sys.exit(0 if ok else 1)
