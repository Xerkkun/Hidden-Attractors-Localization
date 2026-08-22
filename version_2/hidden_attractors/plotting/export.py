import json
import shutil
import datetime
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# Base directory for generated library figures.  It must remain outside the
# installed package directory when running from a wheel.
from hidden_attractors.paths import OUTPUTS
LIBRARY_FIGURES_ROOT = OUTPUTS / "library_figures"

def get_git_commit():
    """
    Returns the current git commit hash, or 'unknown' if not available.
    """
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], 
                             cwd=str(LIBRARY_FIGURES_ROOT.parent), 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE, 
                             text=True, 
                             check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"

from .manifest import (
    load_manifest,
    merge_manifest_entries,
    update_manifest,
    write_manifest_files,
)


def _prepare_figure_for_export(fig):
    """Apply the repository-wide title and background policy in place."""

    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.set_title("")
    fig.suptitle("")


def save_figure_pair_local(fig, output_path, dpi=300):
    """Save a title-free local PNG/PDF pair without global promotion."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    _prepare_figure_for_export(fig)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(png_path, format="png", dpi=int(dpi), bbox_inches="tight", facecolor="white", transparent=False)
    return pdf_path, png_path


def save_report_figure_pair(
    fig,
    output_path,
    *,
    dpi=300,
    pad_inches=0.08,
    pdf_metadata=None,
):
    """Save a report-local PDF/PNG pair while preserving panel titles.

    Report composites may use semantic panel labels such as ``(a)`` through
    ``(d)``.  This centralized route enforces the white-background and paired
    export contract without applying the title-free promotion policy used by
    :func:`save_figure_pair_local`.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.patch.set_facecolor("white")
    for axis in fig.axes:
        axis.set_facecolor("white")
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=float(pad_inches),
        facecolor="white",
        transparent=False,
        metadata=dict(pdf_metadata or {}),
    )
    fig.savefig(
        png_path,
        format="png",
        dpi=int(dpi),
        bbox_inches="tight",
        pad_inches=float(pad_inches),
        facecolor="white",
        transparent=False,
    )
    return pdf_path, png_path


@dataclass(frozen=True)
class _FileSnapshot:
    data: bytes
    mode: int
    atime_ns: int
    mtime_ns: int


def _safe_component(value, *, label):
    component = str(value)
    if (
        not component
        or component in {".", ".."}
        or Path(component).name != component
        or "/" in component
        or "\\" in component
    ):
        raise ValueError(f"{label} must be one safe path component: {value!r}")
    return component


def _snapshot(path):
    path = Path(path)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"promotion destination must be a regular file: {path}")
    stat = path.stat()
    return _FileSnapshot(
        data=path.read_bytes(),
        mode=stat.st_mode,
        atime_ns=stat.st_atime_ns,
        mtime_ns=stat.st_mtime_ns,
    )


def _restore_snapshot(path, snapshot):
    path = Path(path)
    if snapshot is None:
        if path.exists() or path.is_symlink():
            if path.is_dir() and not path.is_symlink():
                raise IsADirectoryError(path)
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(snapshot.data)
    path.chmod(snapshot.mode)
    path.touch(exist_ok=True)
    import os

    os.utime(path, ns=(snapshot.atime_ns, snapshot.mtime_ns))


def _mkdir_with_tracking(directory, created_directories):
    directory = Path(directory)
    missing = []
    cursor = directory
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    if not cursor.is_dir():
        raise NotADirectoryError(cursor)
    # Create one level at a time so a failure cannot leave an untracked parent.
    for missing_directory in reversed(missing):
        missing_directory.mkdir()
        created_directories.append(missing_directory)


def _manifest_entry(*, figure_id, kind, metadata_dict, run_id, targets, pdf_path, png_path, metadata_path):
    return {
        "figure_id": figure_id,
        "caption_key": metadata_dict.get("caption_key", f"fig_{figure_id}"),
        "kind": kind,
        "source_script": metadata_dict.get("source_script", "unknown"),
        "source_function": metadata_dict.get("source_function", "unknown"),
        "data_sources": metadata_dict.get("data_sources", []),
        "run_id": run_id,
        "system_id": metadata_dict.get("system_id", "chua_nonsmooth"),
        "q": metadata_dict.get("q", "1.0"),
        "parameters": metadata_dict.get("parameters", {}),
        "integrator": metadata_dict.get("integrator", "unknown"),
        "memory_mode": metadata_dict.get("memory_mode", "unknown"),
        "t_final": metadata_dict.get("t_final", 0.0),
        "t_burn": metadata_dict.get("t_burn", 0.0),
        "pdf_path": str(pdf_path.relative_to(LIBRARY_FIGURES_ROOT.parent)).replace("\\", "/"),
        "png_path": str(png_path.relative_to(LIBRARY_FIGURES_ROOT.parent)).replace("\\", "/"),
        "metadata_path": str(metadata_path.relative_to(LIBRARY_FIGURES_ROOT.parent)).replace("\\", "/"),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "export_targets": targets,
    }


def promote_local_figure_pairs_batch(promotions, *, run_id, validator=None):
    """Promote local PNG/PDF pairs as one fail-closed transaction.

    ``promotions`` is an iterable of mappings with ``png_path``, ``kind`` and
    ``metadata_dict`` keys and an optional ``export_targets`` iterable.  Every
    source and all metadata are validated before any global path is created or
    changed.  On a later failure, every affected global file is restored
    byte-for-byte and directories created by this transaction are removed.

    If provided, ``validator`` runs after all promoted files and both manifests
    have been written, while rollback snapshots are still live.  It receives
    keyword arguments ``run_id``, ``promoted_pairs``, ``manifest_entries`` and
    ``manifest_paths``.  Raising, or returning exactly ``False``, rejects the
    commit and triggers the same complete rollback.
    """

    run_id = _safe_component(run_id, label="run_id")
    requested = list(promotions)
    if not requested:
        raise ValueError("at least one figure pair is required")
    if validator is not None and not callable(validator):
        raise TypeError("validator must be callable")

    # Complete preflight: no global path is touched before this loop finishes.
    prepared = []
    seen_ids = set()
    for index, specification in enumerate(requested):
        if not isinstance(specification, Mapping):
            raise TypeError(f"promotion {index} must be a mapping")
        try:
            source_png = Path(specification["png_path"])
            kind = str(specification["kind"])
            metadata_dict = dict(specification["metadata_dict"])
        except KeyError as error:
            raise ValueError(f"promotion {index} is missing {error.args[0]}") from error
        source_pdf = source_png.with_suffix(".pdf")
        if source_png.suffix.lower() != ".png":
            raise ValueError(f"promotion source must be a PNG path: {source_png}")
        if not kind:
            raise ValueError(f"promotion {index} requires a non-empty kind")
        if not source_png.is_file() or not source_pdf.is_file():
            raise FileNotFoundError(f"local figure pair is incomplete: {source_png}, {source_pdf}")
        if source_png.is_symlink() or source_pdf.is_symlink():
            raise ValueError(f"local figure pair must use regular files: {source_png}, {source_pdf}")
        figure_id = _safe_component(source_png.stem, label="figure_id")
        if figure_id in seen_ids:
            raise ValueError(f"duplicate figure_id in promotion batch: {figure_id}")
        seen_ids.add(figure_id)
        targets = [
            _safe_component(target, label="export target")
            for target in (specification.get("export_targets") or [])
        ]
        if len(set(targets)) != len(targets):
            raise ValueError(f"duplicate export target for {figure_id}")
        # Readability and JSON serializability are part of validation.
        png_payload = source_png.read_bytes()
        pdf_payload = source_pdf.read_bytes()
        if not png_payload or not pdf_payload:
            raise ValueError(f"local figure pair contains an empty file: {source_png}, {source_pdf}")
        json.dumps(metadata_dict)
        prepared.append((figure_id, kind, metadata_dict, targets, png_payload, pdf_payload))

    run_dir = LIBRARY_FIGURES_ROOT / "by_run" / run_id
    planned = []
    entries = []
    results = []
    for figure_id, kind, metadata_dict, targets, png_payload, pdf_payload in prepared:
        pdf_path = run_dir / "pdf" / f"{figure_id}.pdf"
        png_path = run_dir / "png" / f"{figure_id}.png"
        metadata_path = run_dir / "metadata" / f"{figure_id}.json"
        planned.extend(
            [
                (pdf_payload, pdf_path),
                (png_payload, png_path),
                ((json.dumps(metadata_dict, indent=2) + "\n").encode("utf-8"), metadata_path),
                (pdf_payload, LIBRARY_FIGURES_ROOT / "current" / "pdf" / pdf_path.name),
                (png_payload, LIBRARY_FIGURES_ROOT / "current" / "png" / png_path.name),
            ]
        )
        for target in targets:
            target_root = LIBRARY_FIGURES_ROOT / "by_export" / target
            planned.extend(
                [
                    (pdf_payload, target_root / "pdf" / pdf_path.name),
                    (png_payload, target_root / "png" / png_path.name),
                ]
            )
        entries.append(
            _manifest_entry(
                figure_id=figure_id,
                kind=kind,
                metadata_dict=metadata_dict,
                run_id=run_id,
                targets=targets,
                pdf_path=pdf_path,
                png_path=png_path,
                metadata_path=metadata_path,
            )
        )
        results.append((pdf_path, png_path))

    manifest_json = LIBRARY_FIGURES_ROOT / "manifests" / "figure_manifest.json"
    manifest_csv = LIBRARY_FIGURES_ROOT / "manifests" / "figure_manifest.csv"
    updated_manifest = merge_manifest_entries(load_manifest(), entries)

    # Stage all generated content, including both complete manifest formats.
    with tempfile.TemporaryDirectory(prefix="hidden-attractors-figure-promotion-") as temporary:
        staging = Path(temporary)
        staged_files = []
        for order, (payload, destination) in enumerate(planned):
            suffix = destination.suffix or ".bin"
            staged = staging / f"payload-{order:04d}{suffix}"
            staged.write_bytes(payload)
            staged_files.append((staged, destination))
        staged_manifest_json = staging / "figure_manifest.json"
        staged_manifest_csv = staging / "figure_manifest.csv"
        write_manifest_files(updated_manifest, staged_manifest_json, staged_manifest_csv)
        staged_files.extend(
            [
                (staged_manifest_json, manifest_json),
                (staged_manifest_csv, manifest_csv),
            ]
        )

        destinations = [destination for _, destination in staged_files]
        if len(set(destinations)) != len(destinations):
            raise ValueError("promotion batch contains colliding global destinations")
        snapshots = {destination: _snapshot(destination) for destination in destinations}
        created_directories = []
        try:
            for staged, destination in staged_files:
                _mkdir_with_tracking(destination.parent, created_directories)
                shutil.copy2(staged, destination)
            if validator is not None:
                validation_result = validator(
                    run_id=run_id,
                    promoted_pairs=tuple(results),
                    manifest_entries=tuple(entries),
                    manifest_paths=(manifest_json, manifest_csv),
                )
                if validation_result is False:
                    raise RuntimeError("figure promotion validator rejected the transaction")
        except Exception as promotion_error:
            rollback_errors = []
            for destination in reversed(destinations):
                try:
                    _restore_snapshot(destination, snapshots[destination])
                except Exception as rollback_error:  # pragma: no cover - catastrophic filesystem failure
                    rollback_errors.append(f"{destination}: {rollback_error}")
            for directory in reversed(created_directories):
                try:
                    directory.rmdir()
                except OSError:
                    pass
            if rollback_errors:
                raise RuntimeError(
                    f"figure promotion failed ({promotion_error}); rollback failed: "
                    + "; ".join(rollback_errors)
                ) from promotion_error
            raise

    return results


def promote_local_figure_pair(
    png_path,
    *,
    kind,
    metadata_dict,
    run_id,
    export_targets=None,
):
    """Backward-compatible one-pair wrapper around the batch transaction."""

    return promote_local_figure_pairs_batch(
        [
            {
                "png_path": png_path,
                "kind": kind,
                "metadata_dict": metadata_dict,
                "export_targets": list(export_targets or []),
            }
        ],
        run_id=run_id,
    )[0]

def export_figure(fig, figure_id, kind, metadata_dict, run_id="default_run", export_targets=None):
    """
    Exports a figure to the canonical folder structure.
    Saves:
      - PDF and PNG in run-specific directory
      - JSON metadata in run-specific directory
      - Copies PDF/PNG to active/current directory
      - Copies PDF/PNG to caller-selected export directories if requested
      - Appends entry to figure_manifest.json and figure_manifest.csv
    """
    if export_targets is None:
        export_targets = []
        
    # Standardize paths
    run_dir = LIBRARY_FIGURES_ROOT / "by_run" / run_id
    pdf_dir = run_dir / "pdf"
    png_dir = run_dir / "png"
    meta_dir = run_dir / "metadata"
    
    for d in [pdf_dir, png_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    pdf_path = pdf_dir / f"{figure_id}.pdf"
    png_path = png_dir / f"{figure_id}.png"
    metadata_path = meta_dir / f"{figure_id}.json"
    
    # Save files
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white", transparent=False)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    
    # Write metadata JSON
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2)
        
    # Copy to current/ folders
    current_pdf = LIBRARY_FIGURES_ROOT / "current" / "pdf"
    current_png = LIBRARY_FIGURES_ROOT / "current" / "png"
    current_pdf.mkdir(parents=True, exist_ok=True)
    current_png.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(pdf_path, current_pdf / f"{figure_id}.pdf")
    shutil.copy2(png_path, current_png / f"{figure_id}.png")
    
    # Copy to explicitly requested export targets.
    for target in export_targets:
        export_dir = LIBRARY_FIGURES_ROOT / "by_export" / target
        target_pdf = export_dir / "pdf"
        target_png = export_dir / "png"
        target_pdf.mkdir(parents=True, exist_ok=True)
        target_png.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(pdf_path, target_pdf / f"{figure_id}.pdf")
        shutil.copy2(png_path, target_png / f"{figure_id}.png")
        
    # Build manifest entry
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git_commit = get_git_commit()
    
    entry = {
        "figure_id": figure_id,
        "caption_key": metadata_dict.get("caption_key", f"fig_{figure_id}"),
        "kind": kind,
        "source_script": metadata_dict.get("source_script", "unknown"),
        "source_function": metadata_dict.get("source_function", "unknown"),
        "data_sources": metadata_dict.get("data_sources", []),
        "run_id": run_id,
        "system_id": metadata_dict.get("system_id", "chua_nonsmooth"),
        "q": metadata_dict.get("q", "1.0"),
        "parameters": metadata_dict.get("parameters", {}),
        "integrator": metadata_dict.get("integrator", "unknown"),
        "memory_mode": metadata_dict.get("memory_mode", "unknown"),
        "t_final": metadata_dict.get("t_final", 0.0),
        "t_burn": metadata_dict.get("t_burn", 0.0),
        "pdf_path": str(pdf_path.relative_to(LIBRARY_FIGURES_ROOT.parent)).replace('\\', '/'),
        "png_path": str(png_path.relative_to(LIBRARY_FIGURES_ROOT.parent)).replace('\\', '/'),
        "metadata_path": str(metadata_path.relative_to(LIBRARY_FIGURES_ROOT.parent)).replace('\\', '/'),
        "created_at": created_at,
        "git_commit": git_commit,
        "export_targets": export_targets
    }
    
    update_manifest(entry)
    
    return pdf_path, png_path

def intercept_and_export_path(
    fig,
    output_path,
    kind,
    metadata_dict=None,
    export_targets=None,
):
    """
    Intercept a save operation, export it to the central figure store, update
    the manifest, and write it to the caller's requested location.
    """
    output_path = Path(output_path)
    figure_id = output_path.stem
    
    # Try to parse run_id from path
    parts = output_path.parts
    run_id = "default_run"
    for i, p in enumerate(parts):
        if p == "outputs" and i + 2 < len(parts):
            # Format: outputs/system_id/run_id/
            run_id = parts[i + 2]
            break
        elif p == "outputs" and i + 1 < len(parts):
            # Format: outputs/run_id/
            run_id = parts[i + 1]
            break
            
    if metadata_dict is None:
        metadata_dict = {}
        
    metadata_dict.setdefault("source_script", "plotting_interception")
    metadata_dict.setdefault("caption_key", f"fig_{figure_id}")
    
    # Enforce pure white background and title-free promoted figures.
    _prepare_figure_for_export(fig)
    
    if export_targets is None:
        export_targets = []
        
    pdf_p, png_p = export_figure(
        fig,
        figure_id,
        kind,
        metadata_dict,
        run_id=run_id,
        export_targets=export_targets,
    )
    
    # Copy to original destination
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".png":
        shutil.copy2(png_p, output_path)
    elif output_path.suffix == ".pdf":
        shutil.copy2(pdf_p, output_path)
    else:
        fig.savefig(output_path, dpi=300, facecolor="white", transparent=False)
        
    # Write dual files in original dir
    try:
        fig.savefig(output_path.with_suffix(".pdf"), format="pdf", facecolor="white", transparent=False)
        fig.savefig(output_path.with_suffix(".png"), format="png", dpi=300, facecolor="white", transparent=False)
    except Exception:
        pass
        
    return str(output_path)
