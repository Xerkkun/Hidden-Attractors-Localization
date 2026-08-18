from .equilibria import solve_equilibria
from .jacobian import compute_jacobian
from .stability import classify_equilibrium_stability
from .hiddenness import (
    run_neighborhood_probe,
    generate_neighborhood_points,
    evaluate_target_match
)
from .sphere_tests import run_sphere_probe_sweep
from .classifiers import classify_hiddenness_verdict
from .basins import generate_basin_slice
from .hiddenness_contract import HiddennessVerificationStatus, verify_hiddenness_contract
from .status_labels import CANONICAL_ATTRACTOR_STATUS, normalize_attractor_status
from .attractor_reference import (
    AttractorReferenceCalibration,
    calibrate_attractor_reference,
    classify_cloud_against_reference,
)
from .destination_classifier import (
    DESTINATION_CLASSIFIER_SCHEMA_VERSION,
    DESTINATION_LABELS,
    DestinationClassification,
    DestinationClassifierContract,
    classify_destination,
)
from .edge_tracking import (
    EdgeDestination,
    EdgeEvaluationContext,
    EdgeEvaluationRecord,
    EdgeIteration,
    EdgeTrackingConfig,
    EdgeTrackingResult,
    ScaledCylindricalGeometry,
    ScaledEuclideanGeometry,
    edge_destination_from_classification,
    track_edge_bracket,
)

__all__ = [
    "solve_equilibria",
    "compute_jacobian",
    "classify_equilibrium_stability",
    "run_neighborhood_probe",
    "generate_neighborhood_points",
    "evaluate_target_match",
    "run_sphere_probe_sweep",
    "classify_hiddenness_verdict",
    "generate_basin_slice",
    "HiddennessVerificationStatus",
    "verify_hiddenness_contract",
    "CANONICAL_ATTRACTOR_STATUS",
    "normalize_attractor_status",
    "AttractorReferenceCalibration",
    "calibrate_attractor_reference",
    "classify_cloud_against_reference",
    "DESTINATION_CLASSIFIER_SCHEMA_VERSION",
    "DESTINATION_LABELS",
    "DestinationClassification",
    "DestinationClassifierContract",
    "classify_destination",
    "EdgeDestination",
    "EdgeEvaluationContext",
    "EdgeEvaluationRecord",
    "EdgeIteration",
    "EdgeTrackingConfig",
    "EdgeTrackingResult",
    "ScaledCylindricalGeometry",
    "ScaledEuclideanGeometry",
    "edge_destination_from_classification",
    "track_edge_bracket",
]
