from __future__ import annotations

import sys
import pytest
import json
from pathlib import Path
from hidden_attractors.cli.run import main

def test_seed_cli_help(capsys):
    # Test hidden-attractors seed --help
    with pytest.raises(SystemExit) as excinfo:
        main(["seed", "-h"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "lure-centered" in captured.out
    assert "lure-biased" in captured.out

def test_seed_lure_centered_help(capsys):
    # Test hidden-attractors seed lure-centered --help
    with pytest.raises(SystemExit) as excinfo:
        main(["seed", "lure-centered", "-h"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--config" in captured.out
    assert "--preset" in captured.out

def test_seed_lure_biased_help(capsys):
    # Test hidden-attractors seed lure-biased --help
    with pytest.raises(SystemExit) as excinfo:
        main(["seed", "lure-biased", "-h"])
    # If it fails due to missing arguments, that is correct, or if help works:
    # Let's test with -h explicitly
    with pytest.raises(SystemExit) as excinfo:
        main(["seed", "lure-biased", "-h"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--config" in captured.out

def test_seed_machado_unsupported(capsys):
    # Test hidden-attractors seed machado-centered throws planned/unsupported error
    with pytest.raises(SystemExit) as excinfo:
        main(["seed", "machado-centered"])
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "planned but not implemented" in captured.out

def test_seed_lure_centered_execution(tmp_path):
    # Test executing lure-centered seed generation
    output_dir = tmp_path / "seed_outputs"
    main(["seed", "lure-centered", "--preset", "chua_fractional", "-o", str(output_dir), "--grid_size_omega", "100"])
    
    # Check that minimum outputs exist
    summary_path = output_dir / "seed_generation_summary.json"
    residuals_path = output_dir / "harmonic_residuals.csv"
    seeds_path = output_dir / "seeds.csv"
    metadata_path = output_dir / "run_metadata.json"
    
    assert summary_path.exists()
    assert residuals_path.exists()
    assert seeds_path.exists()
    assert metadata_path.exists()
    
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["family"] == "lure_classical_centered"
        assert "candidates" in data
