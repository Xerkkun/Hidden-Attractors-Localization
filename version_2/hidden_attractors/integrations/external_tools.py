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

# Supported scalar measures, kept explicit so callers can discover which
# optional backend can satisfy a request.  Tuple order is the preference order
# used by backend="auto" when more than one implementation is installed.
COMPLEXITY_BACKEND_MEASURES: dict[str, tuple[str, ...]] = {
    "nolds": (
        "sample_entropy",
        "correlation_dimension",
        "lyapunov_rosenstein",
        "hurst_rs",
        "dfa",
    ),
    "antropy": (
        "permutation_entropy",
        "spectral_entropy",
        "sample_entropy",
        "higuchi_fd",
        "dfa",
    ),
}

COMPLEXITY_MEASURE_BACKENDS: dict[str, tuple[str, ...]] = {
    measure: tuple(
        backend
        for backend, supported in COMPLEXITY_BACKEND_MEASURES.items()
        if measure in supported
    )
    for measure in dict.fromkeys(
        measure
        for supported in COMPLEXITY_BACKEND_MEASURES.values()
        for measure in supported
    )
}


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


def _available_registered_backends() -> tuple[str, ...]:
    available = available_complexity_backends()
    if not available:
        raise ImportError(
            "No optional complexity backend is installed. Install one of: "
            "`python -m pip install nolds` or `python -m pip install antropy`."
        )
    return tuple(
        name for name in COMPLEXITY_BACKEND_MEASURES if name in available
    )


def _requested_measures(
    measures: Iterable[str] | None,
) -> tuple[str, ...] | None:
    if measures is None:
        return None
    requested = tuple(
        dict.fromkeys(str(measure).strip() for measure in measures)
    )
    if not requested:
        return None
    unknown = tuple(
        measure
        for measure in requested
        if measure not in COMPLEXITY_MEASURE_BACKENDS
    )
    if unknown:
        supported = ", ".join(COMPLEXITY_MEASURE_BACKENDS)
        raise ValueError(
            "Unknown complexity measure(s): "
            f"{', '.join(repr(name) for name in unknown)}. "
            f"Supported measures: {supported}."
        )
    return requested


def _measure_backend_plan(
    backend: str,
    requested: tuple[str, ...] | None,
) -> dict[str, tuple[str, ...]]:
    chosen = str(backend).strip().lower()
    if chosen != "auto" and chosen not in COMPLEXITY_BACKEND_MEASURES:
        raise ValueError("backend must be 'auto', 'nolds', or 'antropy'")

    if chosen != "auto":
        selected = requested or COMPLEXITY_BACKEND_MEASURES[chosen]
        unsupported = tuple(
            measure
            for measure in selected
            if measure not in COMPLEXITY_BACKEND_MEASURES[chosen]
        )
        if unsupported:
            raise ValueError(
                f"Backend {chosen!r} does not support measure(s): "
                f"{', '.join(repr(name) for name in unsupported)}."
            )
        return {chosen: tuple(selected)}

    available = _available_registered_backends()
    if requested is None:
        first = available[0]
        return {first: COMPLEXITY_BACKEND_MEASURES[first]}

    plan: dict[str, list[str]] = {}
    unavailable: list[str] = []
    for measure in requested:
        selected_backend = next(
            (
                candidate
                for candidate in COMPLEXITY_MEASURE_BACKENDS[measure]
                if candidate in available
            ),
            None,
        )
        if selected_backend is None:
            unavailable.append(measure)
            continue
        plan.setdefault(selected_backend, []).append(measure)

    if unavailable:
        required = tuple(
            dict.fromkeys(
                candidate
                for measure in unavailable
                for candidate in COMPLEXITY_MEASURE_BACKENDS[measure]
            )
        )
        install = " or ".join(
            f"`python -m pip install {name}`" for name in required
        )
        raise ImportError(
            "No installed backend supports requested measure(s): "
            f"{', '.join(repr(name) for name in unavailable)}. "
            f"Install {install}."
        )
    return {
        name: tuple(plan[name])
        for name in COMPLEXITY_BACKEND_MEASURES
        if name in plan
    }


def _compute_backend_measures(
    backend: str,
    signal: np.ndarray,
    sample_rate: float,
    measures: tuple[str, ...],
) -> dict[str, float]:
    selected = set(measures)
    out: dict[str, float] = {}

    if backend == "nolds":
        nolds = require_external("nolds")
        if "sample_entropy" in selected:
            out["sample_entropy"] = float(nolds.sampen(signal))
        if "correlation_dimension" in selected:
            out["correlation_dimension"] = float(
                nolds.corr_dim(signal, emb_dim=2)
            )
        if "lyapunov_rosenstein" in selected:
            out["lyapunov_rosenstein"] = float(
                nolds.lyap_r(signal, tau=1.0 / sample_rate)
            )
        if "hurst_rs" in selected:
            out["hurst_rs"] = float(nolds.hurst_rs(signal))
        if "dfa" in selected:
            out["dfa"] = float(nolds.dfa(signal))
        return out

    if backend == "antropy":
        ant = require_external("antropy")
        if "permutation_entropy" in selected:
            out["permutation_entropy"] = float(
                ant.perm_entropy(signal, normalize=True)
            )
        if "spectral_entropy" in selected:
            out["spectral_entropy"] = float(
                ant.spectral_entropy(signal, sf=sample_rate, normalize=True)
            )
        if "sample_entropy" in selected:
            out["sample_entropy"] = float(ant.sample_entropy(signal))
        if "higuchi_fd" in selected:
            out["higuchi_fd"] = float(ant.higuchi_fd(signal))
        if "dfa" in selected:
            out["dfa"] = float(ant.detrended_fluctuation(signal))
        return out

    raise AssertionError(f"Unhandled registered complexity backend: {backend}")


def compute_complexity_measures(
    signal: Sequence[float],
    *,
    backend: str = "auto",
    sample_rate: float = 1.0,
    measures: Iterable[str] | None = None,
) -> dict[str, float]:
    """Compute scalar complexity measures through optional external libraries.

    This function is an adapter: it delegates calculations to external
    libraries instead of reimplementing their algorithms.  ``sample_rate``
    is expressed in samples per unit time.  The Rosenstein estimate returned
    as ``lyapunov_rosenstein`` is normalized with
    ``tau = 1 / sample_rate`` and therefore has inverse-time units.

    With ``backend="auto"``, each requested measure is routed according to
    :data:`COMPLEXITY_MEASURE_BACKENDS`; one call may therefore use both
    optional libraries. Unknown measures and measures unsupported by an
    explicitly selected backend raise :class:`ValueError`.
    """

    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        raise ValueError("signal must contain at least 8 finite values")
    sample_rate = float(sample_rate)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate must be a positive finite value")

    requested = _requested_measures(measures)
    plan = _measure_backend_plan(backend, requested)
    out: dict[str, float] = {}
    for chosen_backend, selected in plan.items():
        out.update(
            _compute_backend_measures(
                chosen_backend,
                x,
                sample_rate,
                selected,
            )
        )
    return out
