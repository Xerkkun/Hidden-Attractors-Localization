from __future__ import annotations

import sys
import pytest
from pathlib import Path

# Add version_2 to sys.path if not present
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.workflows.config_loader import _normalize_memory_config, load_config
from hidden_attractors.paths import get_packaged_examples_path


def test_only_memory_policy_finite_window():
    """If the user specifies only memory_policy: finite_window, infer memory_mode = window."""
    flat = {
        "memory_policy": "finite_window",
        "memory_window_steps": 100,
        "h": 0.01,
    }
    _normalize_memory_config(flat)
    assert flat["memory_mode"] == "window"
    assert flat["memory_policy"] == "finite_window"


def test_only_memory_mode_window():
    """If the user specifies only memory_mode: window, infer memory_policy = finite_window."""
    flat = {
        "memory_mode": "window",
        "memory_window_steps": 100,
        "h": 0.01,
    }
    _normalize_memory_config(flat)
    assert flat["memory_policy"] == "finite_window"
    assert flat["memory_mode"] == "window"


def test_both_compatible():
    """If both are compatible, accept."""
    flat = {
        "memory_mode": "window",
        "memory_policy": "finite_window",
        "memory_window_steps": 100,
        "h": 0.01,
    }
    _normalize_memory_config(flat)
    assert flat["memory_mode"] == "window"
    assert flat["memory_policy"] == "finite_window"


def test_both_incompatible():
    """If both are incompatible, raise ValueError."""
    flat = {
        "memory_mode": "window",
        "memory_policy": "full_caputo",
        "memory_window_steps": 100,
        "h": 0.01,
    }
    with pytest.raises(ValueError, match="Incompatible memory settings"):
        _normalize_memory_config(flat)


def test_memory_window_time():
    """If memory_window_time exists, calculate steps."""
    flat = {
        "memory_mode": "window",
        "memory_window_time": 2.0,
        "h": 0.01,
    }
    _normalize_memory_config(flat)
    assert flat["memory_window_steps"] == 200
    assert flat["memory_window_length"] == 200


def test_memory_mode_window_missing_window_params():
    """If memory_mode = window, require window parameters."""
    flat = {
        "memory_mode": "window",
        "h": 0.01,
    }
    with pytest.raises(ValueError, match="memory_window_length, memory_window_steps or memory_window_time must be specified"):
        _normalize_memory_config(flat)
