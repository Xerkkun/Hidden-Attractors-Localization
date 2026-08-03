"""Read and write the central figure manifest without hiding I/O failures."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from hidden_attractors.paths import OUTPUTS


LIBRARY_FIGURES_ROOT = OUTPUTS / "library_figures"

MANIFEST_FIELDS = [
    "figure_id",
    "caption_key",
    "kind",
    "source_script",
    "source_function",
    "data_sources",
    "run_id",
    "system_id",
    "q",
    "parameters",
    "integrator",
    "memory_mode",
    "t_final",
    "t_burn",
    "pdf_path",
    "png_path",
    "metadata_path",
    "created_at",
    "git_commit",
    "export_targets",
]


def load_manifest():
    """Load the JSON manifest, raising if an existing file is malformed."""

    json_path = LIBRARY_FIGURES_ROOT / "manifests" / "figure_manifest.json"
    if not json_path.exists():
        return []
    with json_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        raise ValueError(f"figure manifest must contain a JSON list: {json_path}")
    return entries


def merge_manifest_entries(current_entries, new_entries):
    """Replace figure ids present in ``new_entries`` as one logical batch."""

    additions = list(new_entries)
    figure_ids = [entry.get("figure_id") for entry in additions]
    if any(not isinstance(figure_id, str) or not figure_id for figure_id in figure_ids):
        raise ValueError("every manifest entry requires a non-empty figure_id")
    if len(set(figure_ids)) != len(figure_ids):
        raise ValueError("duplicate figure_id in manifest batch")
    replaced = set(figure_ids)
    return [entry for entry in current_entries if entry.get("figure_id") not in replaced] + additions


def serialize_manifest(entries):
    """Return the complete JSON and CSV representations as UTF-8 bytes."""

    normalized = list(entries)
    json_bytes = (json.dumps(normalized, indent=2) + "\n").encode("utf-8")

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    for entry in normalized:
        row = {}
        for field in MANIFEST_FIELDS:
            value = entry.get(field, "")
            row[field] = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        writer.writerow(row)
    return json_bytes, csv_buffer.getvalue().encode("utf-8")


def write_manifest_files(entries, json_path, csv_path):
    """Write a JSON/CSV pair to explicit paths and propagate every failure."""

    json_path = Path(json_path)
    csv_path = Path(csv_path)
    if json_path.parent != csv_path.parent:
        raise ValueError("JSON and CSV manifests must share a directory")
    json_bytes, csv_bytes = serialize_manifest(entries)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(json_bytes)
    csv_path.write_bytes(csv_bytes)


def _replace_bytes_atomically(path: Path, payload: bytes):
    """Replace one file atomically after fsyncing its temporary sibling."""

    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_manifest(entries):
    """Commit the JSON/CSV manifest pair, restoring both if either write fails."""

    manifest_dir = LIBRARY_FIGURES_ROOT / "manifests"
    json_path = manifest_dir / "figure_manifest.json"
    csv_path = manifest_dir / "figure_manifest.csv"
    json_bytes, csv_bytes = serialize_manifest(entries)

    directory_existed = manifest_dir.exists()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    snapshots = {
        json_path: json_path.read_bytes() if json_path.exists() else None,
        csv_path: csv_path.read_bytes() if csv_path.exists() else None,
    }
    try:
        _replace_bytes_atomically(json_path, json_bytes)
        _replace_bytes_atomically(csv_path, csv_bytes)
    except Exception:
        rollback_errors = []
        for path, previous in snapshots.items():
            try:
                if previous is None:
                    if path.exists():
                        path.unlink()
                else:
                    _replace_bytes_atomically(path, previous)
            except Exception as rollback_error:  # pragma: no cover - catastrophic filesystem failure
                rollback_errors.append(f"{path}: {rollback_error}")
        if not directory_existed:
            try:
                manifest_dir.rmdir()
            except OSError:
                pass
        if rollback_errors:
            raise RuntimeError("manifest rollback failed: " + "; ".join(rollback_errors))
        raise


def update_manifest_batch(new_entries):
    """Replace multiple entries and persist the pair in one manifest commit."""

    save_manifest(merge_manifest_entries(load_manifest(), new_entries))


def update_manifest(entry):
    """Backward-compatible one-entry manifest update."""

    update_manifest_batch([entry])
