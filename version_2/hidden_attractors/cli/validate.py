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
