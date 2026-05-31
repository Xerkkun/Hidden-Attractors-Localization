#!/usr/bin/env python3
"""Regenerate validation/00_manifest from official stage summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hidden_attractors.validation.manifest import DEFAULT_CONTRACT, DEFAULT_VALIDATION_ROOT, regenerate_validation_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--validation-id", default="chua_fractional_validation_evidence")
    args = parser.parse_args()
    manifest = regenerate_validation_manifest(
        args.validation_root,
        contract_path=args.contract,
        validation_id=args.validation_id,
    )
    print(json.dumps({"pending_stages": manifest["pending_stages"], "final_report": manifest["final_report"]}, indent=2))


if __name__ == "__main__":
    main()
