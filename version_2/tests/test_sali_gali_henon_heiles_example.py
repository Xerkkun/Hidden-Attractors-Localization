from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "sali_gali_henon_heiles.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("hafo_sali_gali_hh_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_henon_heiles_example_compares_two_integer_propagation_routes() -> None:
    module = _load_example_module()
    result = module.run_example(duration=0.5)

    assert result["system"] == "Henon-Heiles Hamiltonian"
    assert result["order"] == 1.0
    assert result["gali_orders"] == [2, 3, 4]
    assert result["variational"]["evolution_method"] == "variational"
    assert result["multi_particle"]["evolution_method"] == "multi_particle"
    assert result["variational"]["jacobian_source"] == "analytic"
    assert result["multi_particle"]["jacobian_source"] == "not_used_multi_particle"
    assert result["variational"]["sample_count"] == 3
    assert result["multi_particle"]["sample_count"] == 3
    assert result["variational"]["censored_cells"] == 0
    assert result["multi_particle"]["censored_cells"] == 0
    assert np.isfinite(result["finite_window_comparison"]["maximum_absolute_gali_difference"])
    assert result["variational"]["relative_energy_drift"] < 1.0e-8
    assert result["multi_particle"]["relative_energy_drift"] < 1.0e-8
    assert "no automatic orbit classification" in result["evidence_scope"]

