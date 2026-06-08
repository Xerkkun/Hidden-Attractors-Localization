import os
import json
import csv
import shutil
import datetime
import subprocess
from pathlib import Path

# Base directory for library figures
LIBRARY_FIGURES_ROOT = Path("c:/Users/moren/Desktop/Codes/Hidden Attractors Fractional Order/version_2/library_figures")

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

def update_manifest(entry):
    """
    Updates both figure_manifest.json and figure_manifest.csv with the new entry.
    """
    manifest_dir = LIBRARY_FIGURES_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = manifest_dir / "figure_manifest.json"
    csv_path = manifest_dir / "figure_manifest.csv"
    
    # 1. Update JSON manifest
    entries = []
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            entries = []
            
    # Remove existing entry with same figure_id if present to avoid duplicates
    entries = [e for e in entries if e.get("figure_id") != entry["figure_id"]]
    entries.append(entry)
    
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON manifest: {e}")
        
    # 2. Update CSV manifest
    fields = [
        "figure_id", "caption_key", "kind", "source_script", "source_function",
        "data_sources", "run_id", "system_id", "q", "parameters", "integrator",
        "memory_mode", "t_final", "t_burn", "pdf_path", "png_path", "metadata_path",
        "created_at", "git_commit", "report_targets"
    ]
    
    csv_rows = []
    for e in entries:
        row = {}
        for fld in fields:
            val = e.get(fld, "")
            if isinstance(val, (list, dict)):
                row[fld] = json.dumps(val)
            else:
                row[fld] = str(val)
        csv_rows.append(row)
        
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
    except Exception as e:
        print(f"Error saving CSV manifest: {e}")

def export_figure(fig, figure_id, kind, metadata_dict, run_id="default_run", report_targets=None):
    """
    Exports a figure to the canonical folder structure.
    Saves:
      - PDF and PNG in run-specific directory
      - JSON metadata in run-specific directory
      - Copies PDF/PNG to active/current directory
      - Copies PDF/PNG to report-specific directories if requested
      - Appends entry to figure_manifest.json and figure_manifest.csv
    """
    if report_targets is None:
        report_targets = []
        
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
    
    # Copy to report_targets
    for target in report_targets:
        report_dir = LIBRARY_FIGURES_ROOT / "by_report" / target
        r_pdf = report_dir / "pdf"
        r_png = report_dir / "png"
        r_pdf.mkdir(parents=True, exist_ok=True)
        r_png.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(pdf_path, r_pdf / f"{figure_id}.pdf")
        shutil.copy2(png_path, r_png / f"{figure_id}.png")
        
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
        "report_targets": report_targets
    }
    
    update_manifest(entry)
    
    return pdf_path, png_path

def intercept_and_export_path(fig, output_path, kind, metadata_dict=None):
    """
    Helper to intercept savefig calls in older plotting scripts, formatting them,
    exporting them to the central library figures repository, updating manifests,
    and writing them back to the originally requested locations.
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
        
    metadata_dict.setdefault("source_script", "legacy_or_workflow_interception")
    metadata_dict.setdefault("caption_key", f"fig_{figure_id}")
    
    # Enforce pure white background
    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
        # Enforce no titles to satisfy "Sin títulos internos" rule
        ax.set_title("")
    fig.suptitle("")
    
    # If this is a report asset, register report target
    report_targets = []
    if "report" in str(output_path).lower() or "figs" in str(output_path).lower() or "validation" in str(output_path).lower():
        report_targets.append("df_nc_chua")
        report_targets.append("unified_chua_fractional")
        
    pdf_p, png_p = export_figure(fig, figure_id, kind, metadata_dict, run_id=run_id, report_targets=report_targets)
    
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
