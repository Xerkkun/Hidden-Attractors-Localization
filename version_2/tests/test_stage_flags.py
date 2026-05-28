from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# Add workspace root and version_2 to sys.path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.workflows.simple_runner import run_simple_workflow


def test_simple_runner_attractor_only():
    config = {
        "run_attractor_only": True,
        "run_bifurcation": False,
        "run_basin_slices": False,
    }
    with patch("hidden_attractors.workflows.simple_runner.run_attractor_only_workflow") as mock_attractor:
        mock_attractor.return_value = {"status": "ok"}
        run_simple_workflow(config)
        mock_attractor.assert_called_once_with(config)


def test_simple_runner_bifurcation():
    config = {
        "run_attractor_only": False,
        "run_bifurcation": True,
        "run_basin_slices": False,
    }
    with patch("hidden_attractors.workflows.simple_runner.run_bifurcation_workflow") as mock_bifurcation:
        mock_bifurcation.return_value = {"status": "ok"}
        run_simple_workflow(config)
        mock_bifurcation.assert_called_once_with(config)


def test_simple_runner_basin():
    config = {
        "run_attractor_only": False,
        "run_bifurcation": False,
        "run_basin_slices": True,
    }
    with patch("hidden_attractors.workflows.simple_runner.run_basin_workflow") as mock_basin:
        mock_basin.return_value = {"status": "ok"}
        run_simple_workflow(config)
        mock_basin.assert_called_once_with(config)
