from __future__ import annotations

import csv
import json
import pytest
from pathlib import Path
from hidden_attractors.cli.run import main

def test_continuation_memory_policy_carry(tmp_path):
    output_dir = tmp_path / "cont_memory_outputs"
    output_dir.mkdir()
    
    seeds_csv = output_dir / "seeds.csv"
    with open(seeds_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "family", "centered_or_biased", "A", "sigma0", "omega", "q", "harmonic_residual", "rho_H", "x0", "reconstruction_metadata", "source_config"])
        w.writerow(["centered_classical_b0", "lure_classical_centered", "centered", "1.5", "0.0", "1.2", "0.9998", "0.01", "0.05", json.dumps([0.1, 0.0, 0.0]), json.dumps({"gain": 1.2}), "dummy_config.yaml"])
        
    # Run continuation with default memory policy (carry)
    main(["continuation", "run", "-s", str(seeds_csv), "-o", str(output_dir), "--lambda-values", "0.0,0.5,1.0"])
    
    summary_path = output_dir / "continuation_summary.json"
    assert summary_path.exists()
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        assert summary["history_carried"] is True

def test_continuation_multiparameter_policy(tmp_path):
    output_dir = tmp_path / "cont_multi_outputs"
    output_dir.mkdir()
    
    seeds_csv = output_dir / "seeds.csv"
    with open(seeds_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "family", "centered_or_biased", "A", "sigma0", "omega", "q", "harmonic_residual", "rho_H", "x0", "reconstruction_metadata", "source_config"])
        w.writerow(["centered_classical_b0", "lure_classical_centered", "centered", "1.5", "0.0", "1.2", "0.9998", "0.01", "0.05", json.dumps([0.1, 0.0, 0.0]), json.dumps({"gain": 1.2}), "dummy_config.yaml"])
        
    # Create multiparameter path config
    config_data = {
        "system": {
            "system_id": "chua_fractional_saturation",
        },
        "integrator": {
            "h": 0.01,
        },
        "continuation": {
            "mode": "multiparameter",
            "steps": 5,
            "parameters": {
                "eta": {"start": 0.0, "end": 1.0},
                "q": {"start": 0.9998, "end": 0.9998}
            },
            "memory_policy": "carry_window",
            "memory_window_time": 10.0
        }
    }
    config_path = output_dir / "multiparam_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(config_data, f)
        
    main(["continuation", "multiparameter", "-c", str(config_path), "-s", str(seeds_csv), "-o", str(output_dir)])
    
    summary_path = output_dir / "continuation_summary.json"
    assert summary_path.exists()
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        assert summary["memory_policy"] == "carry_window"
        assert summary["history_carried"] is True
