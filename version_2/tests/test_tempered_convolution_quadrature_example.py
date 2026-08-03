from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "tempered_convolution_quadrature.py"
    )
    spec = importlib.util.spec_from_file_location("tempered_cq_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tempered_cq_manufactured_nonlinear_example() -> None:
    summary = _load_example().run_example(intervals=256, backend="fft")
    assert summary["definition"] == "tempered_caputo"
    assert summary["tempering_convention"] == (
        "unnormalized_exponential_conjugation"
    )
    assert summary["positive_exponential_materialized"] is False
    assert summary["starting_corrections"] == "none_implemented"
    assert max(summary["endpoint_abs_error"]) < 2.0e-4
    assert "not a nonlinear FDE solver validation" in summary["evidence_boundary"]
