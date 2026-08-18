from __future__ import annotations

import sys
import pytest
import json
import csv
import yaml
from pathlib import Path
from hidden_attractors.cli.run import main


def _write_integer_validation_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "system": {
                    "system_id": "chua_integer_saturation",
                    "q": 1.0,
                },
                "modes": {
                    "transfer_mode": "integer",
                    "seed_mode": "integer",
                    "continuation_mode": "integer",
                    "dynamics_mode": "integer",
                },
                "seed": {
                    "df_order": "integer",
                    "transfer_mode": "integer",
                    "q_seed": 1.0,
                },
                "dynamics": {
                    "dynamics_order": "integer",
                    "q_dynamics": 1.0,
                },
                "integrator": {
                    "name": "rk4",
                    "h": 0.01,
                    "memory_mode": "none",
                    "memory_policy": "none",
                    "use_c_backend": False,
                    "allow_python_fallback": False,
                },
                "continuation": {
                    "continuation_order": "integer",
                    "q_continuation": 1.0,
                    "use_period_based_times": False,
                    "t_transient": 0.02,
                    "t_keep": 0.02,
                },
                "simulation": {"divergence_norm": 120.0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

def test_continuation_cli_help(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["continuation", "-h"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "run" in captured.out
    assert "multiparameter" in captured.out

def test_continuation_run_help(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["continuation", "run", "-h"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--config" in captured.out
    assert "--seed-file" in captured.out

def test_continuation_multiparameter_help(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["continuation", "multiparameter", "-h"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--path" in captured.out

def test_continuation_run_execution(tmp_path):
    output_dir = tmp_path / "cont_outputs"
    output_dir.mkdir()
    
    # 1. Create a dummy seeds.csv
    seeds_csv = output_dir / "seeds.csv"
    with open(seeds_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "family", "centered_or_biased", "A", "sigma0", "omega", "q", "harmonic_residual", "rho_H", "x0", "reconstruction_metadata", "source_config"])
        w.writerow(["centered_classical_b0", "lure_classical_centered", "centered", "1.5", "0.0", "1.2", "1.0", "0.01", "0.05", json.dumps([0.1, 0.0, 0.0]), json.dumps({"gain": 1.2}), "software_validation.yaml"])

    config_path = output_dir / "software_validation.yaml"
    _write_integer_validation_config(config_path)
        
    # 2. Run scalar continuation
    main([
        "continuation",
        "run",
        "-c",
        str(config_path),
        "-s",
        str(seeds_csv),
        "-o",
        str(output_dir),
        "--lambda-values",
        "0.0,0.5,1.0",
    ])
    
    # Check that outputs are created
    trace_path = output_dir / "continuation_trace.csv"
    summary_path = output_dir / "continuation_summary.json"
    final_path = output_dir / "final_candidates.csv"
    metadata_path = output_dir / "run_metadata.json"
    
    assert trace_path.exists()
    assert summary_path.exists()
    assert final_path.exists()
    assert metadata_path.exists()
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        assert "memory_policy" in summary
        assert "history_carried" in summary
        assert summary["history_carried"] is False


def test_continuation_rejects_missing_scientific_contract(tmp_path):
    output_dir = tmp_path / "missing_contract"
    output_dir.mkdir()
    seeds_csv = output_dir / "seeds.csv"
    with seeds_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "candidate_id",
            "family",
            "centered_or_biased",
            "A",
            "sigma0",
            "omega",
            "q",
            "harmonic_residual",
            "rho_H",
            "x0",
            "reconstruction_metadata",
            "source_config",
        ])
        writer.writerow([
            "fixture",
            "software_validation",
            "centered",
            "1.0",
            "0.0",
            "1.0",
            "1.0",
            "0.0",
            "0.0",
            json.dumps([0.1, 0.0, 0.0]),
            json.dumps({"gain": 1.0}),
            "none",
        ])

    config_path = output_dir / "empty_contract.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="explicit system_id"):
        main([
            "continuation",
            "run",
            "-c",
            str(config_path),
            "-s",
            str(seeds_csv),
            "--lambda-values",
            "0.0,1.0",
        ])
