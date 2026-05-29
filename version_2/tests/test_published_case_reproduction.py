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
)

CASES_DIR = REPO_ROOT / "validation" / "published_cases"
OUTPUTS_DIR = REPO_ROOT / "validation" / "outputs" / "published_cases"


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
        assert seed_rep.get("q_dependent_seed") is False


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


def test_fractional_extension_seed_is_q_dependent() -> None:
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


def test_W_fractional_spectral_formula() -> None:
    """Verificar directamente W_fractional_spectral(omega, q, P, b, r) = r @ solve((1j*omega)**q*I - P, b)"""
    sys_obj = chua_system("nonsmooth")
    P = sys_obj.lure.matrix
    b = sys_obj.lure.input_vector
    r = sys_obj.lure.output_vector
    omega = 2.039
    q = 0.9998

    W_calc = W_fractional_spectral(omega, q, P, b, r)
    W_expected = r @ np.linalg.solve(((1j * omega) ** q) * np.eye(len(P)) - P, b)
    assert np.isclose(W_calc, W_expected, atol=1e-12)


def test_no_hidden_verified_claim_in_published_reproduction() -> None:
    """Los outputs de esta fase no deben contener hidden_verified."""
    # Ensure they are generated
    run_all_published_cases(OUTPUTS_DIR)

    # Scan the directory and verify no generated JSON contains "hidden_verified" as a key
    for p in OUTPUTS_DIR.glob("**/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        def check_keys(d):
            if isinstance(d, dict):
                assert "hidden_verified" not in d, f"File {p.name} contains key 'hidden_verified'"
                for v in d.values():
                    check_keys(v)
            elif isinstance(d, list):
                for item in d:
                    check_keys(item)
        
        check_keys(data)


def test_reproduction_outputs_schema() -> None:
    """Verificar que reproduction_summary.json contiene:
    case_id, reference, seed_transfer_mode, q_dependent_seed, dynamics_q, statuses, missing_data, no_hidden_verified_claim.
    """
    # Ensure they are generated
    run_all_published_cases(OUTPUTS_DIR)

    expected_keys = {
        "case_id",
        "reference",
        "seed_transfer_mode",
        "q_dependent_seed",
        "dynamics_q",
        "statuses",
        "missing_data",
        "no_hidden_verified_claim",
    }

    for p in OUTPUTS_DIR.glob("**/reproduction_summary.json"):
        with open(p, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        assert expected_keys.issubset(summary.keys()), f"Missing keys in {p}: {expected_keys - summary.keys()}"
        assert summary["no_hidden_verified_claim"] is True
