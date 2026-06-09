from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Add version_2 to path
version_2_dir = Path(__file__).resolve().parents[2]
if str(version_2_dir) not in sys.path:
    sys.path.insert(0, str(version_2_dir))

from hidden_attractors.workflows.config_loader import resolve_seed_transfer_contract
from hidden_attractors.systems import get_system

def test_resolve_seed_transfer_contract_integer():
    system = get_system("chua-nonsmooth")
    config = {
        "q": 1.0,
        "seed": {
            "df_order": "integer",
            "transfer_mode": "published_integer_laplace",
            "q_seed": 1.0
        }
    }
    contract = resolve_seed_transfer_contract(config, system)
    assert contract["df_order"] == "integer"
    assert contract["transfer_mode"] == "published_integer_laplace"
    assert contract["q_seed"] == 1.0
    assert contract["lambda_frequency_rule"] == "lambda=jomega"

def test_resolve_seed_transfer_contract_fractional():
    system = get_system("chua-nonsmooth")
    config = {
        "q": 0.98,
        "seed": {
            "df_order": "fractional",
            "transfer_mode": "fractional_spectral",
            "q_seed": 0.98
        }
    }
    contract = resolve_seed_transfer_contract(config, system)
    assert contract["df_order"] == "fractional"
    assert contract["transfer_mode"] == "fractional_spectral"
    assert contract["q_seed"] == 0.98
    assert contract["lambda_frequency_rule"] == "lambda=(jomega)^q"
