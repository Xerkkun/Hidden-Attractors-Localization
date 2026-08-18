"""Experimental geometric localization primitives for autonomous flows.

Stability: experimental
    Public names are tested, but their signatures may evolve with the
    geometric/topological campaign before promotion to the stable API.

The package supplies local algebraic objects used to prioritize initial
conditions: critical surfaces, connecting-set residuals, perpetual-point and
Picard--Caputo reset-startup filters, explicit PWL regions, and declared
symmetries.  None of these objects is by itself evidence of an attractor,
chaos, basin membership, or hiddenness.
"""

from .core import (
    DifferentialGeometryEvaluation,
    JacobianMode,
    evaluate_differential_geometry,
    evaluate_field,
    evaluate_jacobian,
    finite_difference_jacobian,
    jacobian_field_product,
)
from .piecewise import (
    AffineHalfSpace,
    AffineRegion,
    DiscontinuousFieldError,
    IncompatiblePartitionError,
    NondifferentiablePointError,
    PWLPartition,
    PWLSystemBinding,
    PiecewiseGeometryError,
    RegionResolution,
    RegionStatus,
    SwitchingSurface,
    chua_nonsmooth_partition,
    registered_pwl_partition,
)
from .residuals import (
    ConnectingCurveEvaluation,
    CriticalSurfaceValues,
    PerpetualPointEvaluation,
    PicardCaputoStartupEvaluation,
    connecting_curve_from_evaluation,
    connecting_curve_residual,
    connecting_minor_pairs,
    connecting_minors,
    critical_surface_values,
    fractional_perpetual_startup_residual,
    normalized_perpetual_residual,
    perpetual_point_residual,
    picard_caputo_startup_from_evaluation,
)
from .symmetry import (
    AffineSymmetry,
    SymmetryImage,
    SymmetryValidation,
    generate_symmetry_images,
    identity_symmetry,
    sign_flip_symmetry,
    translation_symmetry,
    validate_affine_symmetry,
)


__all__ = [
    "AffineHalfSpace",
    "AffineRegion",
    "AffineSymmetry",
    "ConnectingCurveEvaluation",
    "CriticalSurfaceValues",
    "DifferentialGeometryEvaluation",
    "DiscontinuousFieldError",
    "IncompatiblePartitionError",
    "JacobianMode",
    "NondifferentiablePointError",
    "PWLPartition",
    "PWLSystemBinding",
    "PerpetualPointEvaluation",
    "PicardCaputoStartupEvaluation",
    "PiecewiseGeometryError",
    "RegionResolution",
    "RegionStatus",
    "SwitchingSurface",
    "SymmetryImage",
    "SymmetryValidation",
    "chua_nonsmooth_partition",
    "connecting_curve_from_evaluation",
    "connecting_curve_residual",
    "connecting_minor_pairs",
    "connecting_minors",
    "critical_surface_values",
    "evaluate_differential_geometry",
    "evaluate_field",
    "evaluate_jacobian",
    "finite_difference_jacobian",
    "fractional_perpetual_startup_residual",
    "generate_symmetry_images",
    "identity_symmetry",
    "jacobian_field_product",
    "normalized_perpetual_residual",
    "perpetual_point_residual",
    "picard_caputo_startup_from_evaluation",
    "registered_pwl_partition",
    "sign_flip_symmetry",
    "translation_symmetry",
    "validate_affine_symmetry",
]
