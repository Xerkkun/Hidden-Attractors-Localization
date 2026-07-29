from __future__ import annotations

import sys
import inspect
import pytest
import json
import yaml
from pathlib import Path
from hidden_attractors.cli.run import main


def test_seed_helpers_have_no_case_profile_defaults():
    from hidden_attractors.cli import seed as seed_module

    for function in (
        seed_module.compute_rho_H_for_lure,
        seed_module.search_biased_seeds,
    ):
        assert all(
            parameter.default is inspect.Parameter.empty
            for parameter in inspect.signature(function).parameters.values()
        )
    source = Path(seed_module.__file__).read_text(encoding="utf-8").lower()
    assert "chua" not in source
    for leaked_default in ("20000", "500.0", "120.0", "fixed_reproducible"):
        assert leaked_default not in source

def test_seed_cli_help(capsys):
    # Test hidden-attractors seed --help
    with pytest.raises(SystemExit) as excinfo:
        main(["seed", "-h"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "lure-centered" in captured.out
    assert "lure-biased" in captured.out
    assert "machado-centered" not in captured.out
    assert "machado-biased" not in captured.out

def test_seed_lure_centered_help(capsys):
    # Test hidden-attractors seed lure-centered --help
    with pytest.raises(SystemExit) as excinfo:
        main(["seed", "lure-centered", "-h"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--config" in captured.out
    assert "--preset" not in captured.out

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

def test_seed_machado_not_public(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["seed", "machado-centered"])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err
    assert "planned but not implemented" not in captured.out

def test_seed_lure_centered_execution(tmp_path):
    # Test executing lure-centered seed generation
    output_dir = tmp_path / "seed_outputs"
    config_path = Path(__file__).parent / "fixtures" / "software_validation_fractional.yaml"
    main(["seed", "lure-centered", "--config", str(config_path), "-o", str(output_dir), "--grid_size_omega", "100"])
    
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


@pytest.mark.parametrize(
    ("section_path", "missing_name"),
    [
        (("system", "system_id"), "system_id"),
        (("seed_search", "omega_min"), "omega_min"),
        (("seed_search", "grid_size_omega"), "grid_size_omega"),
        (("seed_search", "seed_theta"), "seed_theta"),
        (("integrator", "name"), "integrator"),
        (("integrator", "h"), "h"),
        (("integrator", "memory_mode"), "memory_mode"),
        (("integrator", "memory_policy"), "memory_policy"),
        (("simulation", "t_final"), "final_simulation.t_final"),
        (("simulation", "t_burn"), "final_simulation.t_burn"),
        (("seed", "calculation", "harmonics"), "seed.calculation.harmonics"),
        (
            ("seed", "metadata", "random_seed_policy"),
            "seed.metadata.random_seed_policy",
        ),
        (("experiment", "random_seed"), "experiment.random_seed"),
        (("integrator", "use_c_backend"), "integrator.use_c_backend"),
        (
            ("integrator", "allow_python_fallback"),
            "integrator.allow_python_fallback",
        ),
    ],
)
def test_seed_execution_rejects_missing_numerical_contract(
    tmp_path,
    section_path,
    missing_name,
):
    fixture = Path(__file__).parent / "fixtures" / "software_validation_fractional.yaml"
    payload = yaml.safe_load(fixture.read_text(encoding="utf-8"))
    payload["stages"]["bifurcation"] = False
    payload["bifurcation"]["enabled"] = False
    parent = payload
    for part in section_path[:-1]:
        parent = parent[part]
    del parent[section_path[-1]]
    config_path = tmp_path / "missing_seed_contract.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=missing_name.replace(".", r"\.")):
        main(
            [
                "seed",
                "lure-centered",
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "outputs"),
            ]
        )


def test_biased_seed_execution_rejects_missing_search_contract(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "software_validation_fractional.yaml"
    payload = yaml.safe_load(fixture.read_text(encoding="utf-8"))
    del payload["seed"]["biased_search"]["sigma0_grid_size"]
    config_path = tmp_path / "missing_biased_search.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"seed\.biased_search\.sigma0_grid_size"):
        main(
            [
                "seed",
                "lure-biased",
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "outputs"),
            ]
        )
