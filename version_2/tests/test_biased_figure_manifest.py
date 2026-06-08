# -*- coding: utf-8 -*-
import json
import csv
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
MANIFESTS_DIR = ROOT_DIR / "version_2" / "library_figures" / "manifests"
JSON_MANIFEST = MANIFESTS_DIR / "figure_manifest.json"
CSV_MANIFEST = MANIFESTS_DIR / "figure_manifest.csv"

def test_manifest_existence_and_load():
    """Verify that manifests exist or can be created, and contain valid formats."""
    # Ensure manifests directory exists
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # If the JSON manifest exists, check it loads correctly and matches the schema
    if JSON_MANIFEST.exists():
        with open(JSON_MANIFEST, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list), "El manifiesto JSON debe ser una lista de entradas"
        
        # Verify schema of entries if present
        for entry in data:
            assert "figure_id" in entry
            assert "created_at" in entry
            assert "pdf_path" in entry
            assert "png_path" in entry

    # If the CSV manifest exists, check it has the correct header fields
    if CSV_MANIFEST.exists():
        with open(CSV_MANIFEST, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "figure_id" in header
        assert "created_at" in header
        assert "pdf_path" in header
        assert "png_path" in header
