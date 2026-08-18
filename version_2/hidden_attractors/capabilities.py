"""Machine-readable HAFO capability catalog and expansion backlog.

Stability: experimental

The catalog maps useful dynamical-systems functionality to integer and
fractional applicability.  It records design inspiration from pynamicalsys and
DynamicalSystems.jl without importing or copying either implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Capability:
    """One analysis/simulation capability and its evidence boundary."""

    name: str
    category: str
    integer_status: str
    fractional_status: str
    trajectory_based: bool
    backend: str
    notes: str
    inspiration: tuple[str, ...] = ()


_CAPABILITIES = {
    "continuous_simulation": Capability(
        "continuous_simulation", "simulation", "implemented", "implemented", False,
        "hafo", "RK4 and the validated EFORK q=1 limit for integer dynamics; validated fractional method contracts for q<1.",
        ("pynamicalsys", "DynamicalSystems.jl"),
    ),
    "discrete_maps": Capability(
        "discrete_maps", "simulation", "implemented", "not_applicable", False,
        "hafo", "Integer/discrete iteration is distinct from fractional-difference equations.",
        ("pynamicalsys", "DynamicalSystems.jl"),
    ),
    "fractional_difference_equations": Capability(
        "fractional_difference_equations", "simulation", "not_applicable", "planned", False,
        "hafo", "Requires a discrete fractional-difference problem type, not the map API.",
        ("FractionalDiffEq.jl",),
    ),
    "multi_term_caputo_l1": Capability(
        "multi_term_caputo_l1", "simulation", "implemented_limit", "implemented", False,
        "numba/python", "Finite positive-coefficient Caputo sums reuse the combined distributed-order L1 kernel; coefficients are never normalized, duplicate orders are coalesced exactly, and alpha=1 is the backward-Euler limit.",
        ("DynamicalSystems.jl", "FractionalDiffEq.jl"),
    ),
    "tempered_convolution_quadrature": Capability(
        "tempered_convolution_quadrature", "fractional_operator",
        "not_applicable", "implemented", False, "numba/fft",
        (
            "BDF1/BDF2 sampled tempered RL and conjugated-Caputo operators; "
            "direct Numba and offline FFT are executable. Symbol-shift CQ "
            "remains planned; recurrent FBDF1/GNGF2 fast history is a "
            "separate experimental capability."
        ),
        ("FractionalDiffEq.jl",),
    ),
    "tempered_fast_multistep_history": Capability(
        "tempered_fast_multistep_history", "fractional_operator",
        "not_applicable", "implemented", False, "numba/python",
        (
            "Real-axis recurrent Fast Method II for FBDF1 and GNGF2 with an "
            "exact local window, exact conjugated-Caputo anchor, O(Q+n0) "
            "active history and complete finite-grid weight calibration. "
            "Its tolerance is a compression check, not an FDE error claim."
        ),
        ("FractionalDiffEq.jl",),
    ),
    "numba_compiled_flow_map": Capability(
        "numba_compiled_flow_map", "performance", "implemented", "partial", False,
        "numba", "Generic q=1 kernels are implemented; fractional GL sampled operators are compiled.",
        ("pynamicalsys",),
    ),
    "equilibria_and_local_stability": Capability(
        "equilibria_and_local_stability", "local_analysis", "implemented", "implemented", False,
        "hafo", "Fractional stability must use the selected derivative/order stability criterion.",
        ("DynamicalSystems.jl",),
    ),
    "lyapunov_spectrum": Capability(
        "lyapunov_spectrum", "chaos", "implemented", "experimental", True,
        "hafo", "Fractional variational and cloned-dynamics results retain method labels.",
        ("pynamicalsys", "DynamicalSystems.jl"),
    ),
    "covariant_lyapunov_vectors": Capability(
        "covariant_lyapunov_vectors", "chaos", "implemented", "research_required", False,
        "numpy/numba/scipy", "Integer q=1 Ginelli CLVs and stable pair/subspace-angle postprocessing are implemented for flows and maps; nonlocal fractional CLVs still require operator-specific history-space tangent theory.",
        ("pynamicalsys", "DynamicalSystems.jl"),
    ),
    "sali_gali_alignment_indices": Capability(
        "sali_gali_alignment_indices", "chaos", "implemented", "research_required", False,
        "numpy/numba/scipy", "Integer q=1 flows and maps support variational and multi-particle SALI/GALI/LDI with independent vector normalization and finite-time evidence only; every nonlocal fractional extension still requires an operator-specific history-space tangent theory.",
        ("pynamicalsys", "DynamicalSystems.jl"),
    ),
    "zero_one_test": Capability(
        "zero_one_test", "chaos", "implemented", "implemented", True,
        "hafo", "A time-series diagnostic for either order; not proof of hiddenness.",
        ("DynamicalSystems.jl",),
    ),
    "recurrence_quantification": Capability(
        "recurrence_quantification", "complexity", "implemented", "implemented", True,
        "numba", "Auto/cross/joint RQA, explicit thresholds, Theiler exclusion, metrics, blocks and memory guards.",
        ("pynamicalsys", "DynamicalSystems.jl"),
    ),
    "correlation_dimension": Capability(
        "correlation_dimension", "complexity", "implemented", "implemented", True,
        "c/numba", "Exact q=2 correlation-sum curves with Theiler exclusion and an explicit caller-selected D2 fit range; a fractional result describes only the supplied projection.",
        ("pynamicalsys", "DynamicalSystems.jl", "FractalDimensions.jl"),
    ),
    "trajectory_analysis_contract": Capability(
        "trajectory_analysis_contract", "data", "implemented", "implemented", True,
        "hafo", "Immutable sampled trajectory, explicit prehistory, history policy, solver provenance, SHA-256, and backend-neutral result envelope.",
        ("pynamicalsys", "DynamicalSystems.jl"),
    ),
    "permutation_entropy": Capability(
        "permutation_entropy", "complexity", "implemented", "implemented", True,
        "c/numba", "Dense Bandt--Pompe ordinal histogram and finite plug-in Shannon entropy with explicit delay, factorial normalization, tie policy, and dimension-aware C/Numba dispatch; not a chaos or hiddenness certificate.",
        ("pynamicalsys", "DynamicalSystems.jl", "ComplexityMeasures.jl"),
    ),
    "complexity_measure_adapters": Capability(
        "complexity_measure_adapters", "complexity", "implemented", "implemented", True,
        "nolds/antropy", "Validated adapters with explicit optional-backend and unit contracts; results describe supplied sampled trajectories only.",
        ("pynamicalsys", "DynamicalSystems.jl", "ComplexityMeasures.jl"),
    ),
    "poincare_sections": Capability(
        "poincare_sections", "geometry", "implemented", "implemented", True,
        "hafo", "Crossings are trajectory diagnostics; interpretation depends on the model.",
        ("pynamicalsys", "DynamicalSystems.jl"),
    ),
    "bifurcation_diagrams": Capability(
        "bifurcation_diagrams", "parameter_analysis", "implemented", "implemented", True,
        "hafo", "Fractional sweeps must preserve derivative and memory metadata.",
        ("pynamicalsys", "DynamicalSystems.jl"),
    ),
    "basin_classification": Capability(
        "basin_classification", "attractors", "implemented", "implemented", True,
        "hafo", "Finite basin sampling supports but does not prove hiddenness globally.",
        ("DynamicalSystems.jl",),
    ),
    "basin_entropy": Capability(
        "basin_entropy", "attractors", "implemented", "implemented", True,
        "numba", "Finite-grid basin and boundary-basin entropy with explicit box scale and ignored labels.",
        ("pynamicalsys", "DynamicalSystems.jl"),
    ),
    "uncertainty_fraction": Capability(
        "uncertainty_fraction", "attractors", "implemented", "implemented", True,
        "hafo", "Finite paired-label diagnostic with Wilson sampling interval and an explicit scale-fit API.",
        ("pynamicalsys",),
    ),
    "continuation": Capability(
        "continuation", "parameter_analysis", "implemented", "implemented", False,
        "hafo", "Fractional continuation records full, finite-window, or restarted memory.",
        ("DynamicalSystems.jl",),
    ),
    "periodic_orbits": Capability(
        "periodic_orbits", "invariant_sets", "planned", "research_required", False,
        "hafo", "Fractional memory prevents blindly reusing a finite-dimensional ODE shooting map.",
        ("pynamicalsys", "DynamicalSystems.jl"),
    ),
    "delay_embedding": Capability(
        "delay_embedding", "data", "implemented", "implemented", True,
        "numpy/scipy", "General multivariate embedding plus ACF/MI delay and FNN dimension diagnostics.",
        ("DynamicalSystems.jl",),
    ),
    "surrogate_testing": Capability(
        "surrogate_testing", "statistics", "planned", "planned", True,
        "optional", "Applicable to sampled outputs of either order with declared null model.",
        ("DynamicalSystems.jl",),
    ),
}

CAPABILITIES: Mapping[str, Capability] = MappingProxyType(_CAPABILITIES)


def list_capabilities(*, category: str | None = None) -> tuple[Capability, ...]:
    """Return catalog entries, optionally filtered by category."""

    return tuple(
        capability for capability in CAPABILITIES.values()
        if category is None or capability.category == category
    )


def get_capability(name: str) -> Capability:
    """Return one catalog entry by canonical name."""

    try:
        return CAPABILITIES[str(name).strip().lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown HAFO capability: {name!r}") from exc


__all__ = ["CAPABILITIES", "Capability", "get_capability", "list_capabilities"]
