import os
import re
import csv
import pytest
from pathlib import Path
from hidden_attractors.workflows.config_loader import load_config
from hidden_attractors.paths import get_packaged_examples_path


def test_anti_regression_no_src_imports():
    """Verify that no file in version_2/hidden_attractors or version_2/tests imports from src."""
    import_re = re.compile(r'^\s*(?:import\s+src|from\s+src)')
    base_dir = Path(__file__).resolve().parents[1]  # version_2
    
    for root, dirs, files in os.walk(base_dir / "hidden_attractors"):
        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                with open(path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if import_re.match(line):
                            pytest.fail(f"Active code imports from src in {path}:{line_num}: {line.strip()}")

    for root, dirs, files in os.walk(base_dir / "tests"):
        for file in files:
            if file.endswith(".py"):
                if file == "test_src_integrator_contracts.py":
                    # This is specifically a legacy/facade test, we can skip or inspect it.
                    # But active scientific tests should not import src.
                    continue
                path = Path(root) / file
                with open(path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if import_re.match(line):
                            pytest.fail(f"Test code imports from src in {path}:{line_num}: {line.strip()}")


def test_incompatible_memory_mode():
    config_path = get_packaged_examples_path() / "chua_fractional_centered_lure_df.yaml"
    cfg = load_config(config_path)
    
    cfg["memory_policy"] = "full_caputo"
    cfg["memory_mode"] = "window"
    with pytest.raises(ValueError, match="Incompatible settings"):
        from hidden_attractors.workflows.config_loader import _normalize
        _normalize(cfg)


def test_invalid_memory_window():
    config_path = get_packaged_examples_path() / "chua_fractional_centered_lure_df.yaml"
    cfg = load_config(config_path)
    
    cfg["memory_mode"] = "window"
    cfg["memory_window_length"] = 0
    cfg["memory_window_steps"] = 0
    with pytest.raises(ValueError, match="memory_window_length must be a positive integer"):
        from hidden_attractors.workflows.config_loader import _validate
        _validate(cfg)


def test_legacy_parameters_rejected():
    config_path = get_packaged_examples_path() / "chua_fractional_centered_lure_df.yaml"
    cfg = load_config(config_path)
    
    cfg["m"] = 0.4
    with pytest.raises(ValueError, match="Legacy parameter keys"):
        from hidden_attractors.workflows.config_loader import _validate
        _validate(cfg)


def test_bifurcation_sweep_workflow(tmp_path):
    from hidden_attractors.workflows.bifurcation import run_bifurcation_workflow
    
    config = {
        "system_id": "chua_fractional_saturation",
        "q": 0.998,
        "integrator": "efork3",
        "memory_mode": "window",
        "memory_window_length": 100,
        "use_c_backend": False,  # Python fallback for test
        "bifurcation": {
            "parameter": "beta",
            "values": {"min": 11.0, "max": 13.0, "n": 3},
            "continuation_between_values": True,
            "initial_condition": [0.1, 0.0, 0.0],
            "discard_time": 2.0,
            "sample_time": 2.0,
            "h": 0.02,
            "coordinate": "x",
            "sampling": {
                "method": "local_maxima",
                "max_points_per_parameter": 10,
            },
            "save_csv": True,
            "save_plot": False,
        },
        "plot_enabled": False,
        "output_dir": str(tmp_path),
    }
    
    res = run_bifurcation_workflow(config)
    assert res["workflow_mode"] == "bifurcation"
    
    csv_path = Path(tmp_path) / "bifurcation_data.csv"
    assert csv_path.exists()
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == [
            "parameter_name", "parameter_value", "t", "x", "y", "z",
            "coordinate", "coordinate_value", "sample_type", "status"
        ]
