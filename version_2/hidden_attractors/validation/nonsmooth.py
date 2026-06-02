"""Validator for piecewise continuous and non-smooth nonlinearities."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..systems.base import ChaoticSystem


class NonSmoothNonlinearityValidator:
    """Analyze continuity, Lipschitz property, switching crossings, and regional stability."""

    @staticmethod
    def analyze_nonlinearity(system: ChaoticSystem) -> dict[str, Any]:
        """Analyze system nonlinearity type and continuity properties."""
        name = system.name.lower()
        if "nonsmooth" in name or "saturation" in name:
            nl_type = "sat"
            continuous = True
            lipschitz = True
            switching_surfaces = [-1.0, 1.0]
            warnings = []
        elif "abs" in name:
            nl_type = "abs"
            continuous = True
            lipschitz = True
            switching_surfaces = [0.0]
            warnings = []
        elif "sign" in name or "step" in name:
            nl_type = "sign"
            continuous = False
            lipschitz = False
            switching_surfaces = [0.0]
            warnings = [
                "Discontinuous non-Lipschitz vector field detected. Blocks standard ODE integrators "
                "unless Filippov or regularized solvers are enabled."
            ]
        else:
            nl_type = "smooth"
            continuous = True
            lipschitz = True
            switching_surfaces = []
            warnings = []

        return {
            "type": nl_type,
            "continuous": continuous,
            "lipschitz": lipschitz,
            "switching_surfaces": switching_surfaces,
            "warnings": warnings,
        }

    @staticmethod
    def jacobian_region(system: ChaoticSystem, X: np.ndarray) -> np.ndarray:
        """Evaluate the Jacobian matrix matching the region of state X."""
        name = system.name.lower()
        if "nonsmooth" not in name:
            return system.jacobian_matrix(X)

        # For non-smooth Chua saturation
        x = float(X[0])
        p = system.parameters
        alpha = float(p.get("alpha", 8.4562))
        beta = float(p.get("beta", 12.0732))
        gamma = float(p.get("gamma", 0.0052))
        m0 = float(p.get("m0", -0.1768))
        m1 = float(p.get("m1", -1.1468))

        # Check regional slope based on the switching boundaries at x = +-1
        if abs(x) < 1.0:
            dphi = m0
        else:
            dphi = m1

        return np.array(
            [
                [-alpha * (1.0 + dphi), alpha, 0.0],
                [1.0, -1.0, 1.0],
                [0.0, -beta, -gamma],
            ],
            dtype=float,
        )

    @staticmethod
    def detect_switching_crossings(trajectory: np.ndarray, system: ChaoticSystem) -> dict[str, Any]:
        """Detect when the trajectory crosses switching surfaces."""
        props = NonSmoothNonlinearityValidator.analyze_nonlinearity(system)
        surfaces = props["switching_surfaces"]
        
        crossings = []
        if not surfaces or trajectory.ndim != 2 or trajectory.shape[0] < 2:
            return {
                "crossings_detected": 0,
                "crossings": [],
                "warnings": [],
            }

        times = trajectory[:, 0]
        x_coords = trajectory[:, 1]  # The feedback state variable is x[0]
        
        for val in surfaces:
            # Detect crossings by identifying where (x(t) - val) changes sign
            diffs = x_coords - val
            idx = np.where(np.diff(np.sign(diffs)))[0]
            for i in idx:
                crossings.append(
                    {
                        "time": float(times[i]),
                        "surface": float(val),
                        "state_from": trajectory[i, 1:4].tolist(),
                        "state_to": trajectory[i + 1, 1:4].tolist(),
                    }
                )

        warnings = []
        if crossings:
            warnings.append(
                "Trayectoria cruza superficie de conmutación; los Jacobianos simbólicos globales no son válidos en esos puntos."
            )

        return {
            "crossings_detected": len(crossings),
            "crossings": crossings,
            "warnings": warnings,
        }

    @staticmethod
    def validate_equilibrium_stability(
        system: ChaoticSystem, equilibrium: np.ndarray, q: float
    ) -> str:
        """Validate stability of equilibrium using regional eigenvalues and Matignon criteria.
        
        If the equilibrium lies exactly on a switching surface, returns 'nonsmooth_indeterminate'.
        """
        props = NonSmoothNonlinearityValidator.analyze_nonlinearity(system)
        surfaces = props["switching_surfaces"]
        x = float(equilibrium[0])
        
        # Check if exactly on a switching surface (with tolerance 1e-10)
        if any(abs(x - s) < 1e-10 for s in surfaces):
            return "nonsmooth_indeterminate"
            
        # Otherwise, compute regional eigenvalues
        J = NonSmoothNonlinearityValidator.jacobian_region(system, equilibrium)
        eigvals = np.linalg.eigvals(J)
        
        # Matignon stability criterion: |arg(eigval)| > q * pi / 2
        stable = True
        for val in eigvals:
            angle = np.abs(np.angle(val))
            if angle <= q * np.pi / 2.0:
                stable = False
                break
                
        return "stable" if stable else "unstable"
