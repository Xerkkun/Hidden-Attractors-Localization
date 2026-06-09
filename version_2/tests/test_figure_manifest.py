import os
import json
import csv
import shutil
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
from hidden_attractors.plotting.export import export_figure, LIBRARY_FIGURES_ROOT

@pytest.mark.plotting
def test_manifest_entries(tmp_path, monkeypatch):
    import hidden_attractors.plotting.export as export_mod
    import hidden_attractors.plotting.manifest as manifest_mod
    monkeypatch.setattr(export_mod, "LIBRARY_FIGURES_ROOT", tmp_path / "library_figures")
    monkeypatch.setattr(manifest_mod, "LIBRARY_FIGURES_ROOT", tmp_path / "library_figures")
    LIBRARY_FIGURES_ROOT = tmp_path / "library_figures"

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    
    test_run_id = "test_run_manifest_contract"
    test_fig_id = "test_manifest_fig"
    metadata = {
        "caption_key": "test_manifest_fig_key",
        "source_script": "test_figure_manifest.py",
        "source_function": "test_manifest_entries",
        "system_id": "test_chua",
        "q": "0.99",
        "parameters": {"alpha": 10.0, "beta": 15.0},
        "integrator": "efork3",
        "memory_mode": "full",
        "t_final": 50.0,
        "t_burn": 10.0,
        "data_sources": ["simulated_test_data"]
    }
    
    # Export to trigger manifest update
    export_figure(
        fig=fig,
        figure_id=test_fig_id,
        kind="attractor",
        metadata_dict=metadata,
        run_id=test_run_id
    )
    plt.close(fig)
    
    # Verify manifests exist
    manifest_dir = LIBRARY_FIGURES_ROOT / "manifests"
    json_path = manifest_dir / "figure_manifest.json"
    csv_path = manifest_dir / "figure_manifest.csv"
    
    assert json_path.exists()
    assert csv_path.exists()
    
    # Read and verify JSON manifest
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
        
    found_entry = None
    for entry in entries:
        if entry.get("figure_id") == test_fig_id:
            found_entry = entry
            break
            
    assert found_entry is not None
    assert found_entry["caption_key"] == "test_manifest_fig_key"
    assert found_entry["kind"] == "attractor"
    assert found_entry["system_id"] == "test_chua"
    assert found_entry["q"] == "0.99"
    assert found_entry["integrator"] == "efork3"
    assert found_entry["t_final"] == 50.0
    
    # Read and verify CSV manifest
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    found_row = None
    for row in rows:
        if row.get("figure_id") == test_fig_id:
            found_row = row
            break
            
    assert found_row is not None
    assert found_row["caption_key"] == "test_manifest_fig_key"
    assert found_row["kind"] == "attractor"
    assert found_row["system_id"] == "test_chua"
    assert float(found_row["t_final"]) == 50.0
    
    # Clean up test output
    test_run_dir = LIBRARY_FIGURES_ROOT / "by_run" / test_run_id
    if test_run_dir.exists():
        shutil.rmtree(test_run_dir)
        
    # Clean up manifest entry in manifest files if possible
    entries = [e for e in entries if e.get("figure_id") != test_fig_id]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
        
    rows = [r for r in rows if r.get("figure_id") != test_fig_id]
    fields = list(rows[0].keys()) if rows else []
    if fields:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
