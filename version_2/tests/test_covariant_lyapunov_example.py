from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "covariant_lyapunov_henon_map.py"


def _load_example():
    specification = importlib.util.spec_from_file_location(
        "covariant_lyapunov_henon_map_example", EXAMPLE
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_henon_map_clv_example_has_finite_bounded_contract() -> None:
    example = _load_example()
    record = example.run_example(
        iterations=80,
        transient_iterations=120,
        forward_transient_iterations=80,
        backward_transient_iterations=80,
        backend="numpy",
    )

    assert record["system"] == "Henon map"
    assert record["order"] == 1.0
    assert record["sample_count"] == 81
    assert record["method_id"] == "integer_map_ginelli_clv"
    assert len(record["finite_time_exponents_per_iteration"]) == 2
    assert np.all(np.isfinite(record["finite_time_exponents_per_iteration"]))
    angle = record["clv_pair_angle_radians"]
    assert 0.0 <= angle["minimum"] <= angle["median"] <= angle["maximum"] <= np.pi / 2.0
    assert record["maximum_projective_covariance_residual"] < 5.0e-8
    assert record["auto_transient_stopping"] is False
    assert "no automatic" in record["evidence_scope"]

