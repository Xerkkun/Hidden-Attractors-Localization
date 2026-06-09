import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
import pytest
import yaml

from hidden_attractors.models.chua import chua_nonsmooth_parameters, chua_parameters
from hidden_attractors.systems.builtins import chua_system
from validation.python.published_reproduction import (
    W_published_integer,
    W_fractional_spectral,
    compute_seed_for_reproduction,
    run_all_published_cases,
    run_case_reproduction,
)

CASES_DIR = REPO_ROOT / "validation" / "published_cases"
OUTPUTS_DIR = REPO_ROOT / "validation" / "outputs" / "published_cases"


@pytest.mark.literature_traceability
def test_published_cases_exist() -> None:
    """Verificar que existen los tres YAML."""
    expected_files = [
        "kuznetsov2017_chua_integer.yaml",
        "danca2017_chua_fractional_saturation.yaml",
        "wu2023_chua_fractional_arctan.yaml",
    ]
    for filename in expected_files:
        path = CASES_DIR / filename
        assert path.exists(), f"Missing expected case config file: {filename}"


@pytest.mark.literature_traceability
def test_bibliographic_metadata_correct() -> None:
    """Verificar que la metadata bibliográfica en los YAML es correcta y precisa."""
    # 1. Danca 2017
    with open(CASES_DIR / "danca2017_chua_fractional_saturation.yaml", "r", encoding="utf-8") as f:
        danca = yaml.safe_load(f)
    assert danca["paper"]["doi_or_url"] == "10.1007/s11071-017-3472-7"
    assert danca["paper"]["journal"] == "Nonlinear Dynamics"
    assert "Marius-F. Danca" in danca["paper"]["authors"]

    # 2. Wu 2023
    with open(CASES_DIR / "wu2023_chua_fractional_arctan.yaml", "r", encoding="utf-8") as f:
        wu = yaml.safe_load(f)
    assert wu["paper"]["doi_or_url"] == "10.1016/j.rinp.2023.106866"
    assert wu["paper"]["journal"] == "Results in Physics"
    assert wu["paper"]["title"] == "Hidden attractors in a new fractional-order Chua system with arctan nonlinearity and its DSP implementation"
    
    expected_authors = ["Xianming Wu", "Longxiang Fu", "Shaobo He", "Zhao Yao", "Huihai Wang", "Jiayu Han"]
    for author in expected_authors:
        assert author in wu["paper"]["authors"]


@pytest.mark.literature_traceability
@pytest.mark.scientific_contract
def test_published_fractional_cases_use_integer_seed_transfer() -> None:
    """Para Danca 2017 y Wu 2023:
    assert seed_transfer_mode == "published_integer_laplace"
    assert q_dependent_seed is False
    """
    for filename in ["danca2017_chua_fractional_saturation.yaml", "wu2023_chua_fractional_arctan.yaml"]:
        path = CASES_DIR / filename
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        seed_rep = cfg.get("seed_reproduction", {})
        assert seed_rep.get("seed_transfer_mode") == "published_integer_laplace"
        assert seed_rep.get("q_seed") == 1.0
        assert seed_rep.get("q_dependent_seed") is False
        assert seed_rep.get("transfer_exponent_applied") is False


@pytest.mark.literature_traceability
@pytest.mark.scientific_contract
def test_fractional_published_dynamics_configs_record_integration_contract() -> None:
    with open(CASES_DIR / "danca2017_chua_fractional_saturation.yaml", "r", encoding="utf-8") as f:
        danca = yaml.safe_load(f)
    assert danca["dynamics"]["integrator"] == "ABM"
    assert danca["dynamics"]["memory_mode"] == "full"
    assert danca["dynamics"]["memory_policy"] == "full_history"
    assert danca["dynamics"]["caputo_history_accumulated"] is True

    with open(CASES_DIR / "wu2023_chua_fractional_arctan.yaml", "r", encoding="utf-8") as f:
        wu = yaml.safe_load(f)
    assert wu["dynamics"]["integrator"] == "ADM_WU2023"
    assert wu["dynamics"]["memory_policy"] == "none_local_adm"
    assert wu["dynamics"]["caputo_history_accumulated"] is False


@pytest.mark.literature_traceability
@pytest.mark.scientific_contract
def test_published_integer_seed_uses_closed_form_not_modal(monkeypatch) -> None:
    """Parchear np.linalg.eig y np.linalg.eigh para que fallen, y verificar que
    compute_seed_for_reproduction con published_integer_laplace funciona (usa fórmula cerrada).
    """
    def broken_eig(*args, **kwargs):
        raise RuntimeError("np.linalg.eig was called but should be avoided in closed-form mode!")

    # Patch both eig and eigh
    monkeypatch.setattr(np.linalg, "eig", broken_eig)
    monkeypatch.setattr(np.linalg, "eigh", broken_eig)

    path = CASES_DIR / "kuznetsov2017_chua_integer.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Must run successfully without calling eigenvalue solvers!
    seed_res = compute_seed_for_reproduction(cfg)
    assert seed_res["status"] == "ok"
    assert "seed_plus" in seed_res
    assert len(seed_res["seed_plus"]) == 3


@pytest.mark.literature_traceability
@pytest.mark.scientific_contract
def test_published_seed_independent_of_q() -> None:
    """Para un sistema fraccionario publicado:
    - calcular semilla con q_dynamics = 1;
    - calcular semilla con q_dynamics = 0.9998;
    - usando seed_transfer_mode = "published_integer_laplace";
    - verificar que omega0, k y a0 coinciden.
    """
    path = CASES_DIR / "danca2017_chua_fractional_saturation.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Compute with q_dynamics = 1.0
    cfg["dynamics"]["q"] = 1.0
    seed_q1 = compute_seed_for_reproduction(cfg)
    assert seed_q1["status"] == "ok"

    # Compute with q_dynamics = 0.9998
    cfg["dynamics"]["q"] = 0.9998
    seed_q_frac = compute_seed_for_reproduction(cfg)
    assert seed_q_frac["status"] == "ok"

    # In published_integer_laplace, the seed parameters must not depend on dynamics q
    assert np.isclose(seed_q1["omega0"], seed_q_frac["omega0"], atol=1e-12)
    assert np.isclose(seed_q1["k"], seed_q_frac["k"], atol=1e-12)
    assert np.isclose(seed_q1["a0"], seed_q_frac["a0"], atol=1e-12)
    assert np.allclose(seed_q1["seed_plus"], seed_q_frac["seed_plus"], atol=1e-12)


@pytest.mark.literature_traceability
@pytest.mark.scientific_contract
def test_fractional_extension_seed_is_q_dependent_or_transfer_differs() -> None:
    """Usando seed_transfer_mode = "fractional_spectral":
    - calcular con q = 1;
    - calcular con q = 0.9998;
    - verificar que W, omega0 o al menos la condición de frecuencia cambian.
    También verificar que W_published_integer(omega) != W_fractional_spectral(omega, q) para q < 1.
    """
    path = CASES_DIR / "danca2017_chua_fractional_saturation.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["seed_reproduction"]["seed_transfer_mode"] = "fractional_spectral"

    # Compute with q = 1.0
    cfg["dynamics"]["q"] = 1.0
    seed_q1 = compute_seed_for_reproduction(cfg)
    assert seed_q1["status"] == "ok"

    # Compute with q = 0.9998
    cfg["dynamics"]["q"] = 0.9998
    seed_q_frac = compute_seed_for_reproduction(cfg)
    assert seed_q_frac["status"] == "ok"

    # Frequency omega0 or gain k must change
    diff_omega = abs(seed_q1["omega0"] - seed_q_frac["omega0"])
    diff_k = abs(seed_q1["k"] - seed_q_frac["k"])
    assert diff_omega > 1e-6 or diff_k > 1e-6, "Seed should be dependent on q in fractional_spectral mode"

    # Verify W formulas differ for q < 1
    sys_obj = chua_system("nonsmooth")
    P = sys_obj.lure.matrix
    b = sys_obj.lure.input_vector
    r = sys_obj.lure.output_vector
    omega = 2.039

    W_int = W_published_integer(omega, P, b, r)
    W_frac = W_fractional_spectral(omega, 0.9998, P, b, r)
    assert abs(W_int - W_frac) > 1e-5


@pytest.mark.scientific_contract
def test_W_published_integer_formula() -> None:
    """Verificar directamente W_published_integer(omega, P, b, r) = r @ solve(1j*omega*I - P, b)"""
    sys_obj = chua_system("nonsmooth")
    P = sys_obj.lure.matrix
    b = sys_obj.lure.input_vector
    r = sys_obj.lure.output_vector
    omega = 2.039

    W_calc = W_published_integer(omega, P, b, r)
    W_expected = r @ np.linalg.solve(1j * omega * np.eye(len(P)) - P, b)
    assert np.isclose(W_calc, W_expected, atol=1e-12)


@pytest.mark.scientific_contract
def test_W_fractional_spectral_formula() -> None:
    """Verificar directamente W_fractional_spectral(omega, q, P, b, r) = r @ solve((1j*omega)**q*I - P, b)"""
    sys_obj = chua_system("nonsmooth")
    P = sys_obj.lure.matrix
    b = sys_obj.lure.input_vector
    r = sys_obj.lure.output_vector
    omega = 2.039
    q = 0.9998

    W_calc = W_fractional_spectral(omega, q, P, b, r)
    # Check branch formulation
    W_expected = r @ np.linalg.solve(((omega ** q) * np.exp(1j * q * np.pi / 2.0)) * np.eye(len(P)) - P, b)
    assert np.isclose(W_calc, W_expected, atol=1e-12)


@pytest.mark.validation_contract
def test_outputs_do_not_claim_hidden_verified(tmp_path) -> None:
    """Los outputs de esta fase no deben contener la clave hidden_verified.
    Se permite la clave/texto 'hiddenness_claim'.
    """
    # Ensure they are generated without dynamics by default
    run_all_published_cases(tmp_path, run_dynamics=False)

    # Scan the directory and verify no generated JSON contains "hidden_verified" as a key
    for p in tmp_path.glob("**/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        def check_keys(d):
            if isinstance(d, dict):
                assert "hidden_verified" not in d, f"File {p.name} contains forbidden key 'hidden_verified'"
                for v in d.values():
                    check_keys(v)
            elif isinstance(d, list):
                for item in d:
                    check_keys(item)
        
        check_keys(data)


@pytest.mark.validation_contract
def test_reproduction_outputs_schema(tmp_path) -> None:
    """Verificar que reproduction_summary.json contiene las claves requeridas y correctas."""
    run_all_published_cases(tmp_path, run_dynamics=False)

    expected_keys = {
        "case_id",
        "reference",
        "seed_transfer_mode",
        "q_dependent_seed",
        "dynamics_q",
        "dynamics_integrator",
        "dynamics_backend",
        "dynamics_memory_mode",
        "dynamics_memory_policy",
        "caputo_history_accumulated",
        "statuses",
        "missing_data",
        "no_hidden_verified_claim",
        "hiddenness_certified_by_this_pipeline"
    }

    for p in tmp_path.glob("**/reproduction_summary.json"):
        with open(p, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        assert expected_keys.issubset(summary.keys()), f"Missing keys in {p}: {expected_keys - summary.keys()}"
        assert summary["no_hidden_verified_claim"] is True
        assert summary["hiddenness_certified_by_this_pipeline"] is False
        assert summary["q_dependent_seed"] is False  # For all initial cases in Laplace transfer mode


@pytest.mark.validation_contract
def test_dynamics_skipped_by_default(tmp_path) -> None:
    """Verificar que sin run_dynamics, dynamics_reproduction.json contiene status: 'skipped'
    y no aparece la etiqueta 'paper_trajectory_reproduced' ni 'paper_fully_reproduced' en los statuses del summary.
    """
    # Clear outputs first
    for case_id in ["kuznetsov2017_chua_integer", "danca2017_chua_fractional_saturation", "wu2023_chua_fractional_arctan"]:
        dyn_file = tmp_path / case_id / "dynamics_reproduction.json"
        if dyn_file.exists():
            dyn_file.unlink()

    # Run with run_dynamics=False
    results = run_all_published_cases(tmp_path, run_dynamics=False)

    for case_id, res in results.items():
        assert "paper_trajectory_reproduced" not in res["statuses"]
        assert "paper_fully_reproduced" not in res["statuses"]
        
        # Read dynamics file
        dyn_file = tmp_path / case_id / "dynamics_reproduction.json"
        assert dyn_file.exists()
        with open(dyn_file, "r", encoding="utf-8") as f:
            dyn_data = json.load(f)
        assert dyn_data["status"] == "skipped"
        assert dyn_data["reason"] == "run_dynamics=false"


@pytest.mark.literature_traceability
@pytest.mark.scientific_contract
def test_arctan_closed_form_seed_uses_a1_not_m1(monkeypatch) -> None:
    """Verificar que la semilla cerrada del caso arctan usa a1 y no m1, y no usa eigenvectors."""
    def broken_eig(*args, **kwargs):
        raise RuntimeError("np.linalg.eig was called but should be avoided in closed-form mode!")

    monkeypatch.setattr(np.linalg, "eig", broken_eig)
    monkeypatch.setattr(np.linalg, "eigh", broken_eig)

    path = CASES_DIR / "wu2023_chua_fractional_arctan.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Wu 2023 configuration uses a1 = 0.4 and has a default/reconstructed m1 = -1.1468
    seed_res = compute_seed_for_reproduction(cfg)
    assert seed_res["status"] == "ok"

    a0 = seed_res["a0"]
    k = seed_res["k"]
    seed_plus = seed_res["seed_plus"]

    # y0 expected formula with a1
    a1 = cfg["expected"]["parameters"]["a1"]
    y_expected = a0 * (a1 + 1.0 + k)
    assert abs(abs(seed_plus[1]) - abs(y_expected)) < 1e-10

    # m1 is -1.1468, check that it does not match
    m1 = -1.1468
    y_wrong = a0 * (m1 + 1.0 + k)
    assert abs(abs(seed_plus[1]) - abs(y_wrong)) > 1e-6


@pytest.mark.validation_contract
def test_run_dynamics_does_not_claim_full_reproduction(tmp_path) -> None:
    """Verificar que la ejecución con run_dynamics=True no añade 'paper_trajectory_reproduced'
    ni 'paper_fully_reproduced' y usa en su lugar 'paper_trajectory_integrated'.
    """
    path = CASES_DIR / "kuznetsov2017_chua_integer.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Use a small t_final for fast integration
    cfg["dynamics"]["t_final"] = 1.0

    # Write to a temporary YAML file
    temp_yaml = tmp_path / "temp_kuznetsov.yaml"
    with open(temp_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Run case reproduction with run_dynamics=True
    summary = run_case_reproduction(temp_yaml, tmp_path, run_dynamics=True)

    # Verify status lists
    assert "paper_trajectory_integrated" in summary["statuses"]
    assert "paper_trajectory_reproduced" not in summary["statuses"]
    assert "paper_fully_reproduced" not in summary["statuses"]
    assert summary["hiddenness_certified_by_this_pipeline"] is False

    # Read dynamics reproduction output to check status is not skipped
    dyn_file = tmp_path / "kuznetsov2017_chua_integer" / "dynamics_reproduction.json"
    assert dyn_file.exists()
    with open(dyn_file, "r", encoding="utf-8") as f:
        dyn_data = json.load(f)
    assert dyn_data.get("status") != "skipped"
