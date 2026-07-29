"""Public CLI commands for validation contracts and bibliography checks.

Stability: internal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ..references.validator import (
    validate_bibliography_manifest,
    write_traceability_matrix_markdown,
)
from ..validation_contract import main as contract_main


def validate_contract(argv: Sequence[str] | None = None) -> None:
    """Validate a numerical evidence contract."""
    sys.exit(contract_main(argv, deprecation_warning=False))  # type: ignore[call-arg]


def validate_bibliography(argv: Sequence[str] | None = None) -> None:
    """Validate a claims manifest against the bibliographic registry."""
    parser = argparse.ArgumentParser(description="Validate bibliography manifest")
    parser.add_argument(
        "-m",
        "--manifest",
        type=str,
        default="validation/references/claims_manifest.yaml",
        help="Path to the validation claims bibliography manifest",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with exit code 1 if bibliographic verification fails",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output validation results in JSON format",
    )
    parser.add_argument(
        "-o",
        "--markdown-output",
        type=str,
        help="Path to write the generated markdown traceability matrix",
    )

    args = parser.parse_args(argv)
    strict = bool(args.strict)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        root = _repo_root()
        candidates = [
            root / args.manifest,
            root / "version_2" / args.manifest,
            root / "version_2" / "validation" / "references" / "claims_manifest.yaml",
        ]
        manifest_path = next(
            (candidate for candidate in candidates if candidate.exists()),
            manifest_path,
        )

    if not manifest_path.is_file():
        parser.error(
            "claims manifest not found; pass --manifest with an explicit "
            "repository validation manifest"
        )

    print(f"Validating bibliography manifest from: {manifest_path} (strict={strict})")

    try:
        result = validate_bibliography_manifest(str(manifest_path), strict=strict)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(
                "Overall Validation Status: "
                f"{result['bibliographic_validation_status'].upper()}"
            )
            print(f"Total Claims: {result['claims_total']}")
            print(f"Valid Claims: {result['claims_valid']}")
            if result["warnings"]:
                print("\nWarnings:")
                for warning in result["warnings"]:
                    print(f"  - {warning}")
            if result["claims_missing_references"]:
                print("\nClaims missing references (failed):")
                for claim in result["claims_missing_references"]:
                    print(f"  - {claim.get('claim_id')}: {claim.get('text')}")
            if result["claims_with_unregistered_references"]:
                print("\nClaims with unregistered references (failed):")
                for claim in result["claims_with_unregistered_references"]:
                    print(
                        f"  - {claim.get('claim_id')}: "
                        f"{claim.get('references')}"
                    )
            if result["claims_with_insufficient_references"]:
                print("\nClaims with insufficient references (failed):")
                for claim in result["claims_with_insufficient_references"]:
                    print(
                        f"  - {claim.get('claim_id')}: "
                        f"{claim.get('references')}"
                    )

        if args.markdown_output:
            write_traceability_matrix_markdown(result, args.markdown_output)
            print(f"\nTraceability matrix written to: {args.markdown_output}")

        if result["bibliographic_validation_status"] == "FAILED" and strict:
            sys.exit(1)
    except Exception as exc:
        print(f"Bibliography validation failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]
