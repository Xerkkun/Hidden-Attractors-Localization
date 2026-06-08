import os
import shutil
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
from hidden_attractors.plotting.export import export_figure, LIBRARY_FIGURES_ROOT

def test_export_figure_saves_both_formats():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    
    test_run_id = "test_run_export_contract"
    test_fig_id = "test_contract_fig"
    metadata = {
        "caption_key": "test_fig_key",
        "source_script": "test_figure_export_contract.py",
        "source_function": "test_export_figure_saves_both_formats",
        "q": "1.0",
        "parameters": {"alpha": 10.0},
        "t_final": 10.0
    }
    
    # Export
    pdf_path, png_path = export_figure(
        fig=fig,
        figure_id=test_fig_id,
        kind="attractor",
        metadata_dict=metadata,
        run_id=test_run_id
    )
    
    plt.close(fig)
    
    # Check existence
    assert pdf_path.exists()
    assert png_path.exists()
    assert pdf_path.suffix == ".pdf"
    assert png_path.suffix == ".png"
    
    # Check metadata existence
    meta_path = LIBRARY_FIGURES_ROOT / "by_run" / test_run_id / "metadata" / f"{test_fig_id}.json"
    assert meta_path.exists()
    
    # Clean up test output
    test_run_dir = LIBRARY_FIGURES_ROOT / "by_run" / test_run_id
    if test_run_dir.exists():
        shutil.rmtree(test_run_dir)
