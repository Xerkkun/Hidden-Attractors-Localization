import os
import json
import csv
from pathlib import Path

# Base directory for library figures
from hidden_attractors.paths import PROJECT_ROOT
LIBRARY_FIGURES_ROOT = PROJECT_ROOT / "library_figures"

def load_manifest():
    """
    Loads the JSON manifest from figure_manifest.json.
    """
    json_path = LIBRARY_FIGURES_ROOT / "manifests" / "figure_manifest.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_manifest(entries):
    """
    Saves the list of entries to figure_manifest.json and figure_manifest.csv.
    """
    manifest_dir = LIBRARY_FIGURES_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = manifest_dir / "figure_manifest.json"
    csv_path = manifest_dir / "figure_manifest.csv"
    
    # 1. Save JSON manifest
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON manifest: {e}")
        
    # 2. Save CSV manifest
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

def update_manifest(entry):
    """
    Updates figure_manifest.json and figure_manifest.csv with the new entry.
    """
    entries = load_manifest()
    
    # Remove existing entry with same figure_id if present to avoid duplicates
    entries = [e for e in entries if e.get("figure_id") != entry["figure_id"]]
    entries.append(entry)
    
    save_manifest(entries)
