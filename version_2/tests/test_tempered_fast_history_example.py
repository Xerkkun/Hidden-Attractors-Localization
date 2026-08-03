from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tempered_fast_history_chua.py"


def _load_example():
    specification = importlib.util.spec_from_file_location(
        "hafo_tempered_fast_history_chua_example", EXAMPLE
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_tempered_fast_history_chua_example_is_finite_and_parity_gated() -> None:
    result = _load_example().run_example()
    assert result["source_system"] == "chua-nonsmooth"
    assert result["source_status"] == "ok"
    assert result["sample_count"] == 801
    assert result["dimension"] == 3
    assert result["multistep_method"] == "gngf2"
    assert result["backend"] == "numba"
    assert result["compression_tolerance_satisfied"] is True
    assert max(result["l1_relative_weight_error"]) <= 1.0e-9
    assert result["maximum_fast_direct_difference"] <= 5.0e-10
    assert result["finite_values"] is True
    assert "not a fractional Chua solve" in result["evidence_scope"]
