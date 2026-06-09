from __future__ import annotations

import sys
import json
import csv
import pytest
from pathlib import Path
import numpy as np
from hidden_attractors.cli.run import main

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))


def test_zero_one_cli_lifecycle(tmp_path):
    # Determine config path
    config_path = Path(workspace_root) / "version_2" / "configs" / "examples" / "chua_fractional_zero_one.yaml"
    
    # Run 0-1 test workflow via CLI
    main([
        "chaos-test", "zero-one",
        "-c", str(config_path),
        "-o", str(tmp_path),
        "--zero_one.t_final", "1.0",
        "--zero_one.t_burn", "0.2",
        "--use_c_backend", "false",
    ])
    
    summary_path = tmp_path / "zero_one_summary.json"
    assert summary_path.exists()
    
    # Run inspect command
    main([
        "chaos-test", "inspect",
        "-i", str(summary_path),
    ])
    
    # Run 0-1 test on a trajectory CSV directly
    trajectory_csv = tmp_path / "trajectory.csv"
    with open(trajectory_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z"])
        for idx in range(150):
            writer.writerow([idx * 0.01, float(np.sin(idx * 0.1)), 0.0, 0.0])
            
    main([
        "chaos-test", "zero-one",
        "-t", str(trajectory_csv),
        "--observable", "x",
        "-o", str(tmp_path / "traj_out"),
    ])
    
    assert (tmp_path / "traj_out" / "zero_one_summary.json").exists()
