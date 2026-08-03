import numpy as np
from typing import Any, Dict
from .jacobian import compute_jacobian

_MATIGNON_DERIVATIVES = frozenset({"caputo"})


def classify_equilibrium_stability(
    system: Any,
    eq_point: np.ndarray,
    tol: float = 1e-8,
    *,
    q: float | None = None,
    derivative_definition: str | None = None,
    order_mode: str = "commensurate",
) -> Dict[str, Any]:
    """Classify the local stability of an equilibrium point.

    Integer order uses the spectral abscissa.  Fractional order uses
    Matignon's sector criterion only for the commensurate Caputo contract for
    which this implementation is sourced and tested.  Other derivative
    definitions and component-wise orders are rejected rather than silently
    inheriting a criterion with different hypotheses.
    """
    parameters = getattr(system, "parameters", {}) or {}
    q_value = float(parameters.get("q", 1.0) if q is None else q)
    if not np.isfinite(q_value) or q_value <= 0.0 or q_value > 1.0:
        raise ValueError("q must be finite and lie in (0, 1].")
    derivative = str(
        derivative_definition
        if derivative_definition is not None
        else parameters.get(
            "fractional_derivative",
            parameters.get("derivative_definition", "caputo"),
        )
    ).strip().lower()
    normalized_order_mode = str(order_mode).strip().lower()
    tolerance = float(tol)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tol must be finite and non-negative.")
    J = compute_jacobian(system, eq_point)
    eigvals = np.linalg.eigvals(J)

    if q_value == 1.0:
        # Integer stability: Re(lambda) < 0
        spectral_abscissa = float(np.max(np.real(eigvals)))
        stable = spectral_abscissa < -tolerance
        alpha_min = float("nan")
        instability_measure = float("nan")
        stability_class = (
            "stable"
            if stable
            else "unstable"
            if spectral_abscissa > tolerance
            else "marginal_or_inconclusive"
        )
        matignon_margin = float("nan")
        matignon_threshold = float("nan")
        criterion = "integer_spectral_abscissa"
        criterion_applicable = True
    else:
        if derivative not in _MATIGNON_DERIVATIVES:
            raise NotImplementedError(
                "Matignon classification in HAFO is currently validated only "
                "for commensurate Caputo systems; received "
                f"derivative_definition={derivative!r}."
            )
        if normalized_order_mode != "commensurate":
            raise NotImplementedError(
                "Matignon classification in HAFO currently requires "
                "order_mode='commensurate'."
            )
        # Fractional stability (Matignon's criterion): |arg(lambda)| > q * pi / 2
        # np.angle returns angle in [-pi, pi], so we take absolute value
        angles = np.abs(np.angle(eigvals))
        threshold = q_value * np.pi / 2.0
        
        # margin_i = |arg(lambda_i)| - q*pi/2
        margins = angles - threshold
        margin_min = float(np.min(margins))
        
        if margin_min > tolerance:
            stable = True
            stability_class = "stable"
        elif margin_min < -tolerance:
            stable = False
            stability_class = "unstable"
        else:
            stable = False
            stability_class = "marginal_or_inconclusive"
            
        alpha_min = float(np.min(angles))
        instability_measure = float(q_value - 2.0 * alpha_min / np.pi)
        matignon_margin = margin_min
        matignon_threshold = threshold
        spectral_abscissa = float(np.max(np.real(eigvals)))
        criterion = "matignon_commensurate_caputo"
        criterion_applicable = True
        
    return {
        "eigenvalues": eigvals,
        "stable": stable,
        "stability_class": stability_class,
        "matignon_margin": matignon_margin,
        "matignon_threshold": matignon_threshold,
        "alpha_min": alpha_min,
        "instability_measure": instability_measure,
        "spectral_abscissa": spectral_abscissa,
        "q": q_value,
        "derivative_definition": derivative if q_value < 1.0 else "integer_order",
        "order_mode": normalized_order_mode,
        "criterion": criterion,
        "criterion_applicable": criterion_applicable,
    }
