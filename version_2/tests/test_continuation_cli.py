from __future__ import annotations

import sys
import pytest
import json
import csv
from pathlib import Path
from hidden_attractors.cli.run import main

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
        w.writerow(["centered_classical_b0", "lure_classical_centered", "centered", "1.5", "0.0", "1.2", "0.9998", "0.01", "0.05", json.dumps([0.1, 0.0, 0.0]), json.dumps({"gain": 1.2}), "dummy_config.yaml"])
        
    # 2. Run scalar continuation
    main(["continuation", "run", "-s", str(seeds_csv), "-o", str(output_dir), "--lambda-values", "0.0,0.5,1.0"])
    
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
        assert summary["history_carried"] is True
