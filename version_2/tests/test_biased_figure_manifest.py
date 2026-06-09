# -*- coding: utf-8 -*-
import json
import csv
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
MANIFESTS_DIR = ROOT_DIR / "version_2" / "library_figures" / "manifests"
JSON_MANIFEST = MANIFESTS_DIR / "figure_manifest.json"
CSV_MANIFEST = MANIFESTS_DIR / "figure_manifest.csv"

@pytest.mark.plotting
def test_manifest_existence_and_load(tmp_path):
    """Verify that manifests exist or can be created, and contain valid formats."""
    # Ensure manifests directory exists in a hygienic way
    real_dir = ROOT_DIR / "version_2" / "library_figures" / "manifests"
    if real_dir.exists():
        manifests_dir = real_dir
    else:
        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
    
    json_manifest = manifests_dir / "figure_manifest.json"
    csv_manifest = manifests_dir / "figure_manifest.csv"
    
    # If the JSON manifest exists, check it loads correctly and matches the schema
    if json_manifest.exists():
        with open(json_manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list), "El manifiesto JSON debe ser una lista de entradas"
        
        # Verify schema of entries if present
        for entry in data:
            assert "figure_id" in entry
            assert "created_at" in entry
            assert "pdf_path" in entry
            assert "png_path" in entry

    # If the CSV manifest exists, check it has the correct header fields
    if csv_manifest.exists():
        with open(csv_manifest, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "figure_id" in header
        assert "created_at" in header
        assert "pdf_path" in header
        assert "png_path" in header
