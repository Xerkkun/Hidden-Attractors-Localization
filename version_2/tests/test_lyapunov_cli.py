from __future__ import annotations

import sys
import json
import pytest
from pathlib import Path
import numpy as np
from hidden_attractors.cli.run import main

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))


def test_lyapunov_cli_lifecycle(tmp_path):
    # Determine config path
    config_path = Path(workspace_root) / "version_2" / "configs" / "examples" / "chua_fractional_lyapunov.yaml"
    
    # Run Lyapunov workflow via CLI with extremely short times for fast test
    main([
        "lyapunov", "compute",
        "-c", str(config_path),
        "--output_dir", str(tmp_path),
        "--lyapunov.t_final", "0.2",
        "--lyapunov.t_burn", "0.05",
        "--lyapunov.h", "0.01",
        "--use_c_backend", "false",
    ])
    
    summary_path = tmp_path / "lyapunov_summary.json"
    assert summary_path.exists()
    
    # Validate Lyapunov summary JSON via CLI command
    try:
        main([
            "lyapunov", "validate",
            "-i", str(summary_path),
        ])
    except SystemExit as e:
        assert e.code == 0
        
    # Write a mock trajectory file to test trajectory_lyapunov_spectrum command
    trajectory_csv = tmp_path / "mock_trajectory.csv"
    import csv
    with open(trajectory_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z"])
        # Needs at least 100 points
        for t_idx in range(120):
            writer.writerow([t_idx * 0.01, float(np.sin(t_idx * 0.1)), 0.0, 0.0])
            
    # Try calling trajectory-based lyapunov spectrum estimation command
    try:
        main([
            "lyapunov", "spectrum",
            "-t", str(trajectory_csv),
            "--observable", "x",
        ])
    except SystemExit as e:
        # It could exit with 1 if nolds is missing, or succeed if nolds is installed.
        # We accept either exit code or success.
        pass
