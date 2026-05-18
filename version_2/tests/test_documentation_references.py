"""Documentation reference-policy checks."""

from __future__ import annotations

from pathlib import Path

import hidden_attractors.analysis.bifurcation as bifurcation
import hidden_attractors.analysis.trajectory as trajectory
import hidden_attractors.integrations.external_tools as external_tools
import hidden_attractors.models.chua as chua


ROOT = Path(__file__).resolve().parents[1]


def test_reference_documents_exist() -> None:
    required = [
        ROOT / "docs" / "references.bib",
        ROOT / "docs" / "code_reference_map.md",
        ROOT / "docs" / "reporte_unificado_chua_fraccionario.tex",
    ]

    assert all(path.exists() for path in required)


def test_core_calculation_modules_have_reference_notes() -> None:
    modules = [chua, trajectory, bifurcation, external_tools]

    for module in modules:
        assert module.__doc__ is not None
        assert "Reference notes:" in module.__doc__


def test_code_reference_map_mentions_required_titles() -> None:
    text = (ROOT / "docs" / "code_reference_map.md").read_text(encoding="utf-8")
    required_titles = [
        "Hidden Chaotic Attractors in Fractional-Order Systems",
        "Chaos in Chua's Circuit",
        "Stability Results for Fractional Differential Equations with Applications to Control Processing",
        "Lyapunov Characteristic Exponents for Smooth Dynamical Systems and for Hamiltonian Systems",
    ]

    for title in required_titles:
        assert title in text
