"""Built-in chaotic-system registrations."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..models.chua import chua_parameters, equilibria_nonsmooth, rhs_chua
from .base import ChaoticSystem, register_system
from .lure import LureSystem


def _bisect_root(func, left: float, right: float, *, maxiter: int = 100, xtol: float = 1.0e-12) -> float:
    lo = float(left)
    hi = float(right)
    flo = float(func(lo))
    fhi = float(func(hi))
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError("root is not bracketed.")
    for _ in range(int(maxiter)):
        mid = 0.5 * (lo + hi)
        fmid = float(func(mid))
        if abs(fmid) <= xtol or abs(hi - lo) <= xtol:
            return mid
        if flo * fmid <= 0.0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)


def _chua_rhs(state: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    return rhs_chua(state, chua_parameters(**dict(parameters)))


def _chua_equilibria(parameters: Mapping[str, Any]) -> dict[str, np.ndarray]:
    params = chua_parameters(**dict(parameters))
    if params.model != "nonsmooth":
        return {"E0": np.zeros(3, dtype=float)}
    return equilibria_nonsmooth(params)


def _chua_jacobian(state: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    params = chua_parameters(**dict(parameters))
    x = float(np.asarray(state, dtype=float)[0])
    if params.model == "arctan":
        dphi = params.a1 + params.a2 * params.rho / (1.0 + (params.rho * x) ** 2)
    else:
        dphi = params.m0 if abs(x) < 1.0 else params.m1
    return np.array(
        [
            [-params.alpha * (1.0 + dphi), params.alpha, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, -params.beta, -params.gamma],
        ],
        dtype=float,
    )


def _chua_lure_system(parameters: Mapping[str, Any]) -> LureSystem:
    params = chua_parameters(**dict(parameters))
    model = params.model
    base_slope = params.a1 if model == "arctan" else params.m1
    matrix = np.array(
        [
            [-params.alpha * (1.0 + base_slope), params.alpha, 0.0],
            [1.0, -1.0, 1.0],
            [0.0, -params.beta, -params.gamma],
        ],
        dtype=float,
    )
    input_vector = np.array([-params.alpha, 0.0, 0.0], dtype=float)
    output_vector = np.array([1.0, 0.0, 0.0], dtype=float)
    sat_gain = float(params.m0 - params.m1)

    def psi(sigma: float) -> float:
        value = float(sigma)
        if model == "arctan":
            return float(params.a2 * np.arctan(params.rho * value))
        return float(sat_gain * np.clip(value, -1.0, 1.0))

    def describing_function(amplitude: float) -> float:
        amp = float(amplitude)
        if amp <= 0.0 or not np.isfinite(amp):
            raise ValueError("amplitude must be positive and finite.")
        if model == "arctan":
            return float(
                params.a2
                * 2.0
                * (np.sqrt(1.0 + (params.rho * amp) ** 2) - 1.0)
                / (params.rho * amp * amp)
            )
        if amp <= 1.0:
            return sat_gain
        return float((2.0 * sat_gain / np.pi) * (np.arcsin(1.0 / amp) + np.sqrt(amp * amp - 1.0) / (amp * amp)))

    def machado_describing_function(amplitude: float, mu: float) -> float:
        if model != "nonsmooth":
            raise ValueError("The real Machado branch is defined here only for non-smooth Chua with N(A) > 0.")
        exponent = float(mu)
        if exponent <= 0.0 or not np.isfinite(exponent):
            raise ValueError("mu must be positive and finite.")
        base = describing_function(amplitude)
        if base <= 0.0:
            raise ValueError("Machado real branch requires N(A) > 0.")
        return float(base**exponent)

    def gain_compatible(gain: float) -> bool:
        k = float(gain)
        if model == "arctan":
            return np.sign(k) == np.sign(params.a2) and 0.0 < abs(k) < abs(params.a2) * params.rho
        return np.sign(k) == np.sign(sat_gain) and 0.0 < abs(k) <= abs(sat_gain) + 1.0e-10

    def amplitude_from_gain(gain: float) -> float:
        k = float(gain)
        if not gain_compatible(k):
            raise RuntimeError("gain is not compatible with the selected Chua describing function.")
        if model == "arctan":
            amplitude_sq = 4.0 * params.a2 * (params.a2 * params.rho - k) / (k * k * params.rho)
            if amplitude_sq <= 0.0:
                raise RuntimeError("computed arctan amplitude is not real positive.")
            return float(np.sqrt(amplitude_sq))
        if abs(k - sat_gain) < 1.0e-10:
            return 1.0
        grid = np.linspace(1.0 + 1.0e-9, 100.0, 20_000)
        values = np.array([describing_function(a) - k for a in grid], dtype=float)
        for i in range(len(grid) - 1):
            if values[i] == 0.0:
                return float(grid[i])
            if values[i] * values[i + 1] < 0.0:
                return float(_bisect_root(lambda a: describing_function(a) - k, grid[i], grid[i + 1], maxiter=500))
        raise RuntimeError("No amplitude solved the requested describing-function gain.")

    return LureSystem(
        name=f"chua-{model}-lure",
        matrix=matrix,
        input_vector=input_vector,
        output_vector=output_vector,
        nonlinearity=psi,
        describing_function=describing_function,
        machado_describing_function=machado_describing_function,
        gain_compatible=gain_compatible,
        amplitude_from_gain=amplitude_from_gain,
        description="Manual Lur'e split used by the Chua harmonic-balance workflows.",
    )


def chua_system(model: str = "nonsmooth") -> ChaoticSystem:
    params = chua_parameters(model=model)
    return ChaoticSystem(
        name=f"chua-{params.model}",
        dimension=3,
        rhs=_chua_rhs,
        equilibria=_chua_equilibria,
        jacobian=_chua_jacobian,
        parameters={
            "model": params.model,
            "alpha": params.alpha,
            "beta": params.beta,
            "gamma": params.gamma,
            "m0": params.m0,
            "m1": params.m1,
            "a1": params.a1,
            "a2": params.a2,
            "rho": params.rho,
        },
        description="Fractional-order Chua vector field used by the hidden-attractor workflows.",
        tags=("chua", "fractional", "hidden-attractors"),
        workflows={
            "full": "hidden-attractors-unified-chua",
            "robustness": "hidden-attractors-robustness-overlay",
            "sphere-controls": "hidden-attractors-sphere-controls",
            "refined-basin": "hidden-attractors-refined-basin",
        },
        lure=_chua_lure_system(
            {
                "model": params.model,
                "alpha": params.alpha,
                "beta": params.beta,
                "gamma": params.gamma,
                "m0": params.m0,
                "m1": params.m1,
                "a1": params.a1,
                "a2": params.a2,
                "rho": params.rho,
            }
        ),
    )


def register_builtin_systems() -> None:
    """Register built-in systems, replacing stale registrations if reloaded."""

    register_system(chua_system("nonsmooth"), replace=True)
    register_system(chua_system("arctan"), replace=True)
