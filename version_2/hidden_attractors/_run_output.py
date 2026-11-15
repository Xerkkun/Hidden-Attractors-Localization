"""Claim a fresh run directory before computation or evidence writes.

Run ownership is explicit so a multi-stage workflow can share a directory
without providing an overwrite switch for already recorded experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid


RUN_MARKER = ".hafo_run.json"


@dataclass(frozen=True)
class RunOutputClaim:
    path: Path
    run_id: str
    run_kind: str


def claim_output_directory(path: str | Path, *, run_kind: str) -> RunOutputClaim:
    """Reserve a new or empty directory; reject every preexisting artifact.

    The exclusive marker creation also rejects competing cooperating writers
    that both observed an empty directory. A failed run keeps its marker and
    partial evidence; callers must choose another output directory.
    """
    if not isinstance(run_kind, str) or not run_kind.strip():
        raise ValueError("run_kind must be a nonempty string")
    output = Path(path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(
            f"Run output directory is not empty: {output}. "
            "Use a new directory; recorded evidence is never overwritten."
        )
    claim = RunOutputClaim(output, str(uuid.uuid4()), run_kind)
    record = {
        "schema_version": "1.0", "run_id": claim.run_id,
        "run_kind": claim.run_kind, "status": "in_progress",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(output),
    }
    with (output / RUN_MARKER).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return claim


def require_output_claim(path: str | Path, claim: RunOutputClaim) -> Path:
    """Validate a shared active run claim before a child stage writes."""
    if not isinstance(claim, RunOutputClaim):
        raise TypeError("output_claim must be a RunOutputClaim, not an overwrite flag")
    output = Path(path).expanduser().resolve()
    if output != claim.path:
        raise ValueError("output_claim belongs to a different directory")
    try:
        record = json.loads((output / RUN_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Output run marker is missing or invalid") from exc
    if (record.get("run_id") != claim.run_id or record.get("run_kind") != claim.run_kind
            or record.get("status") != "in_progress"):
        raise ValueError("Output claim does not identify this active run")
    return output


def complete_output_claim(claim: RunOutputClaim) -> None:
    """Mark the owned run complete after all outputs have been persisted."""
    output = require_output_claim(claim.path, claim)
    marker = output / RUN_MARKER
    record = json.loads(marker.read_text(encoding="utf-8"))
    record.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    temporary = output / (RUN_MARKER + "." + claim.run_id + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, marker)


__all__ = ["RunOutputClaim", "claim_output_directory", "require_output_claim", "complete_output_claim"]
