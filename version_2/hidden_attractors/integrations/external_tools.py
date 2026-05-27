"""Adapters and references for external dynamical-systems tools.

The project should not copy algorithms that are already maintained elsewhere.
This module records recommended external tools and exposes small optional
adapters when the dependency is installed.

Reference notes:
    External complexity and continuation methods must be cited at the package
    or paper level. The local functions in this module are adapters, not copied
    algorithm implementations. See ``docs/external_tools.md``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ExternalTool:
    name: str
    import_name: str | None
    url: str
    install_hint: str
    capabilities: tuple[str, ...]
    recommended_use: str


EXTERNAL_TOOLS: tuple[ExternalTool, ...] = (
    ExternalTool(
        name="PyDSTool",
        import_name="PyDSTool",
        url="https://pydstool.github.io/PyDSTool/FrontPage.html",
        install_hint="Install PyDSTool only in a compatible Python environment; it is optional here.",
        capabilities=("simulation", "phase-plane analysis", "continuation", "bifurcation analysis"),
        recommended_use="Use for continuation/branch-tracking when environment compatibility is confirmed.",
    ),
    ExternalTool(
        name="pyComplexity notebook",
        import_name=None,
        url="https://github.com/relopezbriega/relopezbriega.github.io/blob/master/downloads/pyComplexity.ipynb",
        install_hint="Reference notebook, not treated as an installable dependency.",
        capabilities=("complexity measures", "notebook-style exposition"),
        recommended_use="Use as a reference style for documented complexity analysis; do not copy code without license review.",
    ),
    ExternalTool(
        name="nolds",
        import_name="nolds",
        url="https://pypi.org/project/nolds/",
        install_hint="python -m pip install nolds",
        capabilities=("sample entropy", "correlation dimension", "Lyapunov exponents", "Hurst exponent", "DFA"),
        recommended_use="Use for scalar nonlinear time-series measures from simulated coordinates.",
    ),
    ExternalTool(
        name="antropy",
        import_name="antropy",
        url="https://pypi.org/project/antropy/",
        install_hint="python -m pip install antropy",
        capabilities=("permutation entropy", "spectral entropy", "sample entropy", "fractal dimensions", "DFA"),
        recommended_use="Use for entropy/fractal diagnostics on scalar observables.",
    ),
)


def require_external(import_name: str, package_name: str | None = None) -> Any:
    """Import an optional dependency or raise a clear installation error."""

    try:
        return importlib.import_module(import_name)
    except Exception as exc:
        pkg = package_name or import_name
        raise ImportError(f"Optional dependency {pkg!r} is required. Install it with `python -m pip install {pkg}`.") from exc


def external_tool_report() -> list[dict[str, Any]]:
    """Return documentation-ready metadata for the registered tools."""

    rows: list[dict[str, Any]] = []
    for tool in EXTERNAL_TOOLS:
        available = False
        if tool.import_name:
            try:
                importlib.import_module(tool.import_name)
                available = True
            except Exception:
                available = False
        rows.append(
            {
                "name": tool.name,
                "available": available,
                "url": tool.url,
                "capabilities": list(tool.capabilities),
                "recommended_use": tool.recommended_use,
                "install_hint": tool.install_hint,
            }
        )
    return rows


def available_complexity_backends() -> list[str]:
    """Return installed optional complexity backends."""

    names: list[str] = []
    for import_name in ("nolds", "antropy"):
        try:
            importlib.import_module(import_name)
            names.append(import_name)
        except Exception:
            continue
    return names


def _first_available_backend(preferred: str | None = None) -> str:
    if preferred and preferred != "auto":
        return preferred
    available = available_complexity_backends()
    if not available:
        raise ImportError(
            "No optional complexity backend is installed. Install one of: "
            "`python -m pip install nolds` or `python -m pip install antropy`."
        )
    return available[0]


def compute_complexity_measures(
    signal: Sequence[float],
    *,
    backend: str = "auto",
    sample_rate: float = 1.0,
    measures: Iterable[str] | None = None,
) -> dict[str, float]:
    """Compute scalar complexity measures through optional external libraries.

    This function is an adapter: it delegates calculations to external
    libraries instead of reimplementing their algorithms.
    """

    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        raise ValueError("signal must contain at least 8 finite values")

    selected = set(measures or ())
    use_all = not selected
    chosen = _first_available_backend(backend)
    out: dict[str, float] = {}

    if chosen == "nolds":
        nolds = require_external("nolds")
        if use_all or "sample_entropy" in selected:
            out["sample_entropy"] = float(nolds.sampen(x))
        if use_all or "correlation_dimension" in selected:
            out["correlation_dimension"] = float(nolds.corr_dim(x, emb_dim=2))
        if use_all or "lyapunov_rosenstein" in selected:
            out["lyapunov_rosenstein"] = float(nolds.lyap_r(x))
        if use_all or "hurst_rs" in selected:
            out["hurst_rs"] = float(nolds.hurst_rs(x))
        if use_all or "dfa" in selected:
            out["dfa"] = float(nolds.dfa(x))
        return out

    if chosen == "antropy":
        ant = require_external("antropy")
        if use_all or "permutation_entropy" in selected:
            out["permutation_entropy"] = float(ant.perm_entropy(x, normalize=True))
        if use_all or "spectral_entropy" in selected:
            out["spectral_entropy"] = float(ant.spectral_entropy(x, sf=sample_rate, normalize=True))
        if use_all or "sample_entropy" in selected:
            out["sample_entropy"] = float(ant.sample_entropy(x))
        if use_all or "higuchi_fd" in selected:
            out["higuchi_fd"] = float(ant.higuchi_fd(x))
        if use_all or "dfa" in selected:
            out["dfa"] = float(ant.detrended_fluctuation(x))
        return out

    raise ValueError("backend must be 'auto', 'nolds', or 'antropy'")
