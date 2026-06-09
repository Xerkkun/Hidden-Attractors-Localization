from __future__ import annotations

import sys
import json
import pytest
from pathlib import Path
from hidden_attractors.cli.run import main

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))


def test_bifurcation_cli_lifecycle(tmp_path):
    # Determine config path
    from hidden_attractors.paths import get_packaged_examples_path
    config_path = get_packaged_examples_path() / "chua_fractional_bifurcation.yaml"
    
    # Run bifurcation workflow via CLI
    main([
        "bifurcation", "run",
        "-c", str(config_path),
        "--output_dir", str(tmp_path),
        "--bifurcation.values.n", "2",
        "--bifurcation.discard_time", "0.1",
        "--bifurcation.sample_time", "0.1",
        "--bifurcation.h", "0.01",
        "--bifurcation.save_csv", "true",
        "--bifurcation.save_plot", "false",
        "--use_c_backend", "false",
        "--plot_enabled", "false",
    ])
    
    # Verify outputs
    csv_path = tmp_path / "bifurcation_data.csv"
    summary_path = tmp_path / "bifurcation_summary.json"
    
    assert csv_path.exists()
    assert summary_path.exists()
    
    # Run bifurcation plot CLI command
    plot_out = tmp_path / "custom_plot.png"
    main([
        "bifurcation", "plot",
        "-i", str(csv_path),
        "-o", str(plot_out),
    ])
    assert plot_out.exists()
    
    # Run bifurcation inspect CLI command
    main([
        "bifurcation", "inspect",
        "-i", str(summary_path),
    ])
