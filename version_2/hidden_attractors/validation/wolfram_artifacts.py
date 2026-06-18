"""Resolve generated Wolfram evidence without copying ignored runtime outputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_SYSTEM_ID = "chua_fractional_saturation"
REQUIRED_WOLFRAM_ARTIFACTS: Mapping[str, str] = {
    "wolfram_equilibria_residuals.csv": "equilibria_residuals.csv",
    "wolfram_jacobians.csv": "jacobians.csv",
    "wolfram_eigenvalues_matignon.csv": "eigenvalues_matignon.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_passed(path: Path) -> bool | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return bool(data.get("passed"))


@dataclass(frozen=True)
class WolframArtifactSet:
    """Resolved Wolfram CSV inputs and their generated-output provenance."""

    system_id: str
    source_kind: str
    files: Mapping[str, Path]
    generated_output_dir: Path
    validation_summary: Path | None
    python_consistency_summary: Path | None
    summaries_pass: bool

    @property
    def complete(self) -> bool:
        return all(name in self.files for name in REQUIRED_WOLFRAM_ARTIFACTS)

    def provenance(self, *, relative_to: Path) -> dict[str, object]:
        def describe(path: Path) -> dict[str, str]:
            try:
                source = str(path.resolve().relative_to(relative_to.resolve()))
            except ValueError:
                source = str(path.resolve())
            return {"source": source.replace("\\", "/"), "sha256": _sha256(path)}

        return {
            "system_id": self.system_id,
            "source_kind": self.source_kind,
            "summaries_pass": self.summaries_pass,
            "artifacts": {name: describe(path) for name, path in self.files.items()},
            "validation_summary": (
                describe(self.validation_summary) if self.validation_summary is not None else None
            ),
            "python_consistency_summary": (
                describe(self.python_consistency_summary)
                if self.python_consistency_summary is not None
                else None
            ),
        }


def resolve_wolfram_artifacts(
    validation_root: Path,
    *,
    system_id: str = DEFAULT_SYSTEM_ID,
    generated_output_root: Path | None = None,
) -> WolframArtifactSet:
    """Resolve official Wolfram inputs, preferring generated prefixed outputs.

    The ignored ``validation/outputs`` tree remains runtime evidence.  The
    official algebraic stage consumes those files directly and records hashes.
    Legacy unprefixed CSV files inside ``02_algebraic_validation`` remain a
    fallback for deliberately promoted evidence.
    """

    validation_root = validation_root.resolve()
    algebra = validation_root / "02_algebraic_validation"
    
    if generated_output_root is not None:
        output_dirs = [generated_output_root.resolve()]
    else:
        output_dirs = [
            validation_root / system_id,
            validation_root.parent / "outputs" / system_id,
            validation_root / "outputs" / "wolfram" / system_id,
        ]

    chosen_dir = None
    chosen_prefixed = None
    for d in output_dirs:
        prefixed = {
            official: d / f"{system_id}_{suffix}"
            for official, suffix in REQUIRED_WOLFRAM_ARTIFACTS.items()
        }
        if all(path.exists() for path in prefixed.values()):
            chosen_dir = d
            chosen_prefixed = prefixed
            break

    if chosen_dir is not None and chosen_prefixed is not None:
        output_dir = chosen_dir
        prefixed = chosen_prefixed
        validation_summary = output_dir / f"{system_id}_validation_summary.json"
        consistency_summary = output_dir / f"{system_id}_python_consistency_summary.json"
        summary_states = (
            _json_passed(validation_summary),
            _json_passed(consistency_summary),
        )
        return WolframArtifactSet(
            system_id=system_id,
            source_kind="generated_prefixed_outputs",
            files=prefixed,
            generated_output_dir=output_dir,
            validation_summary=validation_summary if validation_summary.exists() else None,
            python_consistency_summary=consistency_summary if consistency_summary.exists() else None,
            summaries_pass=summary_states == (True, True),
        )

    # Fallback/default if not found
    output_dir = output_dirs[0]
    promoted = {
        official: algebra / official
        for official in REQUIRED_WOLFRAM_ARTIFACTS
        if (algebra / official).exists()
    }
    return WolframArtifactSet(
        system_id=system_id,
        source_kind="promoted_unprefixed_outputs" if promoted else "missing",
        files=promoted,
        generated_output_dir=output_dir,
        validation_summary=None,
        python_consistency_summary=None,
        summaries_pass=bool(promoted) and len(promoted) == len(REQUIRED_WOLFRAM_ARTIFACTS),
    )
