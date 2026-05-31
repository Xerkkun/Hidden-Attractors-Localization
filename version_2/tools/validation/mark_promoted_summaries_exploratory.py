#!/usr/bin/env python3
"""Mark existing promoted stage summaries as evidence from an earlier exploratory run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "configs" / "validation_contract.json"
DEFAULT_VALIDATION_ROOT = ROOT / "validation"


def mark_summaries(validation_root: Path, contract_path: Path) -> list[str]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    updated: list[str] = []
    for stage in contract["stages"]:
        summary_path = validation_root / stage["id"] / stage["summary"]
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            content = handle.read()
        if '"evidence_scope"' in content:
            continue
        scope = {
            "classification": "prior_exploratory_run",
            "current_contract": "configs/unified_caputo_protocol.json",
            "current_contract_applied": False,
            "note": (
                "This promoted summary predates the current unified Caputo protocol. "
                "It remains historical exploratory evidence and must not be treated "
                "as a completed run of configs/unified_caputo_protocol.json."
            ),
        }
        scope_json = json.dumps({"evidence_scope": scope}, indent=2)
        block = scope_json[1:-1].strip("\n")
        closing_index = content.rfind("}")
        if closing_index < 0:
            raise ValueError(f"Summary is not a JSON object: {summary_path}")
        content = content[:closing_index].rstrip("\r\n") + ",\n" + block + "\n}\n"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        updated.append(summary_path.relative_to(validation_root).as_posix())
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    print(json.dumps({"updated_summaries": mark_summaries(args.validation_root, args.contract)}, indent=2))


if __name__ == "__main__":
    main()
