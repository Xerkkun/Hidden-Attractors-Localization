"""Lur'e compatibility validator.

Verifies whether a system matches the Lur'e feedback representation:
    D^q X = P X + b psi(r^T X)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..systems.base import ChaoticSystem


class LureCompatibilityValidator:
    """Validator to classify Lur'e equivalence and calculate splitting details."""

    @staticmethod
    def validate(system: ChaoticSystem, config: Any = None) -> dict[str, Any]:
        """Validate system compatibility with the Lur'e feedback model.

        Parameters
        ----------
        system : ChaoticSystem
            The system under validation.
        config : Any, optional
            Configuration namespace or dictionary.

        Returns
        -------
        report : dict
            Compatibility metrics and classification.
        """
        lure = system.lure
        warnings_list: list[str] = []
        assumptions_list: list[str] = [
            "Feedback loop has a single scalar nonlinearity psi(sigma)",
            "System is autonomous",
        ]

        # Check force heuristic configuration
        force_heuristic = False
        if config is not None:
            if hasattr(config, "force_heuristic_describing_function"):
                force_heuristic = bool(config.force_heuristic_describing_function)
            elif isinstance(config, dict):
                force_heuristic = bool(config.get("force_heuristic_describing_function", False))

        if lure is None:
            return {
                "class": "NOT_COMPATIBLE",
                "P": None,
                "b": None,
                "r": None,
                "psi_name": None,
                "transform_S": None,
                "det_S": None,
                "cond_S": None,
                "residual_norm": float("inf"),
                "nonlinear_channels": 0,
                "scalar_output_sigma": False,
                "warnings": ["System does not define a Lur'e decomposition."],
                "assumptions": assumptions_list,
                "allowed_methods": ["classic"] if force_heuristic else [],
            }

        # Generate a point cloud to compute residual reconstruction error
        rng = np.random.default_rng(20260602)
        dim = system.dimension
        points = rng.uniform(-5.0, 5.0, size=(100, dim))

        residuals: list[float] = []
        for X in points:
            try:
                # F(X) from system
                f_sys = system.evaluate(X)
                # F_lure(X) = P * X + b * psi(r^T * X)
                sigma = float(lure.output_vector @ X)
                psi_val = float(lure.nonlinearity(sigma))
                f_lure = lure.matrix @ X + lure.input_vector * psi_val
                residuals.append(float(np.linalg.norm(f_sys - f_lure)))
            except Exception as exc:
                residuals.append(float("inf"))
                warnings_list.append(f"Reconstruction failed: {exc}")

        residual_norm = max(residuals) if residuals else float("inf")

        # Classify based on reconstruction error
        lure_class = "NOT_COMPATIBLE"
        transform_S = np.eye(dim)
        det_S = 1.0
        cond_S = 1.0

        # Check for linear transformation if configured
        # e.g., if there is a known S, check it. But default is direct.
        if residual_norm < 1e-10:
            lure_class = "LURE_DIRECT"
            allowed_methods = ["classic", "machado"]
        elif residual_norm < 1e-3:
            lure_class = "LURE_APPROXIMATE"
            warnings_list.append("Lur'e reconstruction is approximate; seed is not methodologically exact.")
            allowed_methods = ["classic", "machado"]
        else:
            # Let's check if the residual error is large, classify as NOT_COMPATIBLE
            lure_class = "NOT_COMPATIBLE"
            warnings_list.append(f"Lur'e reconstruction error is too large (max residual={residual_norm:.3e}).")



        if lure_class == "NOT_COMPATIBLE":
            if force_heuristic:
                warnings_list.append(
                    "Describing function scan forced by configuration on incompatible system. "
                    "Cap validation state at 'seed_found'."
                )
                allowed_methods = ["classic"]
            else:
                allowed_methods = []

        # Nonlinearity name
        psi_name = lure.nonlinearity.__name__ if hasattr(lure.nonlinearity, "__name__") else "psi"

        return {
            "class": lure_class,
            "P": lure.matrix.tolist(),
            "b": lure.input_vector.tolist(),
            "r": lure.output_vector.tolist(),
            "psi_name": psi_name,
            "transform_S": transform_S.tolist(),
            "det_S": float(det_S),
            "cond_S": float(cond_S),
            "residual_norm": residual_norm,
            "nonlinear_channels": 1,
            "scalar_output_sigma": True,
            "warnings": warnings_list,
            "assumptions": assumptions_list,
            "allowed_methods": allowed_methods,
        }
