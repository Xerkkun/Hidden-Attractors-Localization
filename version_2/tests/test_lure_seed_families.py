from __future__ import annotations

import pytest
import json
from pathlib import Path
from hidden_attractors.cli.run import main

def test_lure_seed_families_centered(tmp_path):
    output_dir = tmp_path / "seed_centered_outputs"
    main(["seed", "lure-centered", "--preset", "chua_fractional", "-o", str(output_dir), "--grid_size_omega", "50"])
    
    summary_path = output_dir / "seed_generation_summary.json"
    assert summary_path.exists()
    
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["family"] == "lure_classical_centered"
        for cand in data["candidates"]:
            assert cand["family"] == "lure_classical_centered"
            assert cand["centered_or_biased"] == "centered"
            assert cand["sigma0"] == 0.0

def test_lure_seed_families_biased(tmp_path):
    output_dir = tmp_path / "seed_biased_outputs"
    main(["seed", "lure-biased", "--preset", "chua_fractional", "-o", str(output_dir), "--grid_size_omega", "50", "--amplitude_max", "10.0"])
    
    summary_path = output_dir / "seed_generation_summary.json"
    assert summary_path.exists()
    
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["family"] == "lure_classical_biased"
        assert "scientific_warning" in data
        for cand in data["candidates"]:
            assert cand["family"] == "lure_classical_biased"
            assert cand["centered_or_biased"] == "biased"
            assert abs(cand["sigma0"]) > 1.0e-3
