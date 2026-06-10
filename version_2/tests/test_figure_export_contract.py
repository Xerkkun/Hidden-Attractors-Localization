import pytest
import matplotlib.pyplot as plt
from pathlib import Path
from hidden_attractors.plotting.export import export_figure

@pytest.mark.plotting
def test_export_figure_saves_both_formats(tmp_path, monkeypatch):
    import hidden_attractors.plotting.export as export_mod
    import hidden_attractors.plotting.manifest as manifest_mod
    
    mock_root = tmp_path / "library_figures"
    monkeypatch.setattr(export_mod, "LIBRARY_FIGURES_ROOT", mock_root)
    monkeypatch.setattr(manifest_mod, "LIBRARY_FIGURES_ROOT", mock_root)
    
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
    meta_path = mock_root / "by_run" / test_run_id / "metadata" / f"{test_fig_id}.json"
    assert meta_path.exists()
    
    # Assert paths reside within tmp_path (subpath verification)
    assert tmp_path in pdf_path.parents
    assert tmp_path in png_path.parents
