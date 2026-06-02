"""Validator for system symmetries and symmetric seed generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..systems.base import ChaoticSystem


class SymmetryValidator:
    """Validator to detect symmetries and generate symmetric initial states."""

    STANDARD_TRANSFORMS = {
        "inversion": lambda X: -np.asarray(X, dtype=float),
        "rotation_z": lambda X: np.array([-X[0], -X[1], X[2]], dtype=float) if len(X) >= 3 else np.asarray(X, dtype=float),
    }

    @staticmethod
    def detect_symmetries(system: ChaoticSystem, tolerance: float = 1e-5) -> list[str]:
        """Detect which standard symmetries are valid for the system.

        Checks F(T(X)) = T(F(X)) numerically on a point cloud.
        """
        rng = np.random.default_rng(20260602)
        dim = system.dimension
        points = rng.uniform(-5.0, 5.0, size=(100, dim))
        
        valid_symmetries = []
        for name, transform in SymmetryValidator.STANDARD_TRANSFORMS.items():
            is_valid = True
            for X in points:
                try:
                    # T(X)
                    tx = transform(X)
                    # F(T(X))
                    f_tx = system.evaluate(tx)
                    # T(F(X))
                    t_fx = transform(system.evaluate(X))
                    
                    if np.linalg.norm(f_tx - t_fx) >= tolerance:
                        is_valid = False
                        break
                except Exception:
                    is_valid = False
                    break
            if is_valid:
                valid_symmetries.append(name)
                
        return valid_symmetries

    @staticmethod
    def generate_symmetric_seeds(
        system: ChaoticSystem,
        seeds: list[dict[str, Any]],
        tolerance: float = 1e-5,
    ) -> list[dict[str, Any]]:
        """Generate symmetric seeds for each seed in list using detected symmetries.

        Deduplicates against parent seeds and already generated seeds.
        """
        valid_transforms = SymmetryValidator.detect_symmetries(system, tolerance)
        if not valid_transforms:
            return list(seeds)

        all_seeds = list(seeds)
        # Ensure we have parent IDs for existing seeds
        for idx, s in enumerate(all_seeds):
            if "seed_id" not in s:
                s["seed_id"] = s.get("candidate_id", f"seed_{idx}")
            if "symmetry_group_id" not in s:
                s["symmetry_group_id"] = s["seed_id"]
            if "parent_seed_id" not in s:
                s["parent_seed_id"] = None
            if "transform_name" not in s:
                s["transform_name"] = "identity"
            if "is_symmetric_generated" not in s:
                s["is_symmetric_generated"] = False

        new_seeds: list[dict[str, Any]] = []
        for transform_name in valid_transforms:
            transform = SymmetryValidator.STANDARD_TRANSFORMS[transform_name]
            for parent_seed in seeds:
                parent_x0 = np.asarray(parent_seed.get("x0", parent_seed.get("seed", [0.0, 0.0, 0.0])), dtype=float)
                tx0 = transform(parent_x0)
                
                # Check deduplication against parent_x0
                if np.linalg.norm(tx0 - parent_x0) < tolerance:
                    continue  # Invariant seed under this transform, skip

                # Check deduplication against all seeds currently in all_seeds and new_seeds
                is_duplicate = False
                for existing in all_seeds + new_seeds:
                    exist_x0 = np.asarray(existing.get("x0", existing.get("seed", [0.0, 0.0, 0.0])), dtype=float)
                    if np.linalg.norm(tx0 - exist_x0) < tolerance:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue

                parent_id = parent_seed["seed_id"]
                new_id = f"{parent_id}_sym_{transform_name}"
                
                # Create a copy with the transformed coordinates and symmetric metadata
                sym_seed = dict(parent_seed)
                sym_seed.update({
                    "candidate_id": new_id,
                    "seed_id": new_id,
                    "symmetry_group_id": parent_seed.get("symmetry_group_id", parent_id),
                    "parent_seed_id": parent_id,
                    "transform_name": transform_name,
                    "is_symmetric_generated": True,
                })
                # Update both coordinate formats
                sym_seed["x0"] = tx0.tolist()
                if "seed" in sym_seed:
                    sym_seed["seed"] = tx0.tolist()
                if "seed_x" in sym_seed:
                    sym_seed["seed_x"] = float(tx0[0])
                    sym_seed["seed_y"] = float(tx0[1])
                    sym_seed["seed_z"] = float(tx0[2])
                
                new_seeds.append(sym_seed)

        return all_seeds + new_seeds
