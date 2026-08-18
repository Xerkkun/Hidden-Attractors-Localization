"""Validate independent Wolfram CLV fixtures and optionally compare HAFO.

The Wolfram artifact is produced from exact rational map/flow similarities,
an explicitly written positive-diagonal modified Gram--Schmidt factorization,
and the Ginelli backward triangular recursion at 80-digit precision.  This
module always validates that artifact without importing HAFO.  If the public
``covariant_lyapunov`` module is available, it additionally compares its
directions by orientation-free line distance and by the covariance identity.

A passing result is finite constant-cocycle integer-order evidence only.  It
is not a nonlinear convergence theorem, a hyperbolicity or chaos label, or
evidence for fractional-order covariant Lyapunov vectors.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ID = "covariant_lyapunov_integer"
TEMP_ROOT = Path(tempfile.gettempdir())
DEFAULT_SUMMARY = (
    TEMP_ROOT
    / "hafo_covariant_lyapunov_integer"
    / f"{SYSTEM_ID}_validation_summary.json"
)

EXPECTED_FIXTURE_NAMES = ("nonnormal_map_2d", "constant_flow_3d")
RETAINED_INDICES = np.arange(40, 81, dtype=np.int64)
DIRECT_CHECKPOINT_INDICES = np.asarray([0, 40, 60, 80, 120], dtype=np.int64)
NUMBER_OF_STEPS = 120

SOURCE_ANCHORS = {
    "ginelli_doi": "10.1103/PhysRevLett.99.130601",
    "kuptsov_parlitz_doi": "10.1007/s00332-012-9126-5",
    "froyland_comparison_doi": "10.1016/j.physd.2012.12.005",
}

EXPECTED_NUMERIC = {
    "nonnormal_map_2d": {
        "matrix": np.asarray([[56 / 25, 33 / 25], [8 / 25, 94 / 25]]),
        "orthogonal_similarity": np.asarray([[3 / 5, -4 / 5], [4 / 5, 3 / 5]]),
        "schur_factor": np.asarray([[4.0, 1.0], [0.0, 2.0]]),
        "terminal_coefficients": np.asarray([[1.0, 1 / 3], [0.0, 1.0]]),
        "exact_directions_columns": np.asarray(
            [[3 / 5, -11 / (5 * np.sqrt(5))],
             [4 / 5, 2 / (5 * np.sqrt(5))]]
        ),
        "exact_exponents": np.log(np.asarray([4.0, 2.0])),
    },
    "constant_flow_3d": {
        "generator": np.asarray(
            [[1 / 3, -1 / 9, -7 / 18],
             [4 / 3, 4 / 9, -7 / 9],
             [0.0, -8 / 9, -7 / 9]]
        ),
        "cocycle_matrix": np.asarray(
            [[67 / 54, -1 / 54, -49 / 216],
             [34 / 27, 41 / 27, -35 / 54],
             [-8 / 27, -16 / 27, 20 / 27]]
        ),
        "orthogonal_similarity": np.asarray(
            [[1 / 9, 8 / 9, 4 / 9],
             [8 / 9, 1 / 9, -4 / 9],
             [-4 / 9, 4 / 9, -7 / 9]]
        ),
        "schur_generator": np.asarray(
            [[1.0, 1.0, 1 / 2], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        ),
        "terminal_coefficients": np.asarray(
            [[1.0, 1 / 3, -1 / 5], [0.0, 1.0, 2 / 7], [0.0, 0.0, 1.0]]
        ),
        "exact_directions_columns": np.asarray(
            [[1 / 9, 7 / (9 * np.sqrt(2)), -5 / (3 * np.sqrt(33))],
             [8 / 9, -7 / (9 * np.sqrt(2)), -4 / (3 * np.sqrt(33))],
             [-4 / 9, 8 / (9 * np.sqrt(2)), -16 / (3 * np.sqrt(33))]]
        ),
        "exact_exponents": np.asarray([1.0, 0.0, -1.0]),
    },
}

PYTHON_SCHEMA_TOLERANCE = 5.0e-14
PYTHON_QR_TOLERANCE = 5.0e-14
PYTHON_LINE_TOLERANCE = 5.0e-11
PYTHON_ANGLE_TOLERANCE = 5.0e-11
CORE_LINE_TOLERANCE = 5.0e-10
CORE_COVARIANCE_TOLERANCE = 5.0e-10

PUBLIC_API_MODULE = "hidden_attractors.analysis.covariant_lyapunov"
PUBLIC_CORE_NAME = "integer_covariant_vectors_from_qr_history"
PUBLIC_ANGLE_NAME = "covariant_lyapunov_angles"


class PublicCLVAPIUnavailable(RuntimeError):
    """The optional public CLV core is not importable in this checkout."""


def _load(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _array(value: Any, *, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions, received {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("Wolfram fixture contains a non-finite number")
    return result


def _max_abs(left: Any, right: Any) -> float:
    left_array = _array(left)
    right_array = _array(right)
    if left_array.shape != right_array.shape:
        raise ValueError(f"shape mismatch: {left_array.shape} != {right_array.shape}")
    return float(np.max(np.abs(left_array - right_array), initial=0.0))


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = _array(matrix, ndim=2)
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms <= 0.0):
        raise ValueError("cannot normalize a zero CLV column")
    return matrix / norms


def _line_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Stable unoriented line distance via normalized rank-one projectors."""

    left = _normalize_columns(first)
    right = _normalize_columns(second)
    if left.shape != right.shape:
        raise ValueError(f"CLV shape mismatch: {left.shape} != {right.shape}")
    values = []
    for column in range(left.shape[1]):
        projector_left = np.outer(left[:, column], left[:, column])
        projector_right = np.outer(right[:, column], right[:, column])
        values.append(
            np.linalg.norm(projector_left - projector_right, ord="fro")
            / np.sqrt(2.0)
        )
    return np.asarray(values, dtype=np.float64)


def _pair_angles(matrix: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    normalized = _normalize_columns(matrix)
    angles = []
    for first, second in np.asarray(pairs, dtype=np.int64):
        cosine = abs(float(normalized[:, first] @ normalized[:, second]))
        angles.append(np.arccos(np.clip(cosine, 0.0, 1.0)))
    return np.asarray(angles, dtype=np.float64)


def _cocycle_matrix(fixture: dict[str, Any]) -> np.ndarray:
    key = "matrix" if fixture["kind"] == "map" else "cocycle_matrix"
    return _array(fixture[key], ndim=2)


def _validate_fixture(name: str, fixture: dict[str, Any]) -> dict[str, Any]:
    expected = EXPECTED_NUMERIC[name]
    dimension = int(fixture["dimension"])
    q_history = _array(fixture["q_history"], ndim=3)
    r_history = _array(fixture["r_history"], ndim=3)
    retained = np.asarray(fixture["retained_indices"], dtype=np.int64)
    retained_clvs = _array(fixture["retained_clvs"], ndim=3)
    exact_directions = _array(fixture["exact_directions_columns"], ndim=2)
    exact_exponents = _array(fixture["exact_exponents"], ndim=1)
    matrix = _cocycle_matrix(fixture)
    coordinate_step = float(fixture["coordinate_step"])
    pairs = np.asarray(fixture["pair_indices_zero_based"], dtype=np.int64)
    exact_angles = _array(fixture["exact_pair_angles_radians"], ndim=1)
    retained_angles = _array(fixture["retained_pair_angles_radians"], ndim=2)

    expected_pairs = np.asarray(
        [(i, j) for i in range(dimension) for j in range(i + 1, dimension)],
        dtype=np.int64,
    )
    shape_contract = bool(
        q_history.shape == (NUMBER_OF_STEPS + 1, dimension, dimension)
        and r_history.shape == (NUMBER_OF_STEPS, dimension, dimension)
        and retained_clvs.shape == (RETAINED_INDICES.size, dimension, dimension)
        and retained_angles.shape == (RETAINED_INDICES.size, len(expected_pairs))
        and exact_directions.shape == (dimension, dimension)
        and exact_exponents.shape == (dimension,)
        and np.array_equal(retained, RETAINED_INDICES)
        and np.array_equal(pairs, expected_pairs)
    )
    if not shape_contract:
        raise ValueError(f"{name}: invalid history, retained, or pair shape contract")

    declared_numeric_diffs = {
        key: _max_abs(fixture[key], expected_value)
        for key, expected_value in expected.items()
    }
    declared_numeric_max_diff = max(declared_numeric_diffs.values())

    identity = np.eye(dimension)
    orthogonality_max = float(
        max(
            np.linalg.norm(q.T @ q - identity, ord="fro")
            for q in q_history
        )
    )
    qr_relative_max = 0.0
    for index in range(NUMBER_OF_STEPS):
        left = matrix @ q_history[index]
        right = q_history[index + 1] @ r_history[index]
        denominator = max(1.0, np.linalg.norm(left, ord="fro"))
        qr_relative_max = max(
            qr_relative_max,
            float(np.linalg.norm(left - right, ord="fro") / denominator),
        )
    positive_r_diagonal = bool(
        np.all(np.diagonal(r_history, axis1=1, axis2=2) > 0.0)
    )

    exact_line_max = max(
        float(np.max(_line_distances(sample, exact_directions), initial=0.0))
        for sample in retained_clvs
    )
    covariance_max = 0.0
    for index in range(retained_clvs.shape[0] - 1):
        covariance_max = max(
            covariance_max,
            float(
                np.max(
                    _line_distances(matrix @ retained_clvs[index], retained_clvs[index + 1]),
                    initial=0.0,
                )
            ),
        )

    recomputed_exact_angles = _pair_angles(exact_directions, pairs)
    recomputed_retained_angles = np.vstack(
        [_pair_angles(sample, pairs) for sample in retained_clvs]
    )
    angle_artifact_max_diff = max(
        _max_abs(exact_angles, recomputed_exact_angles),
        _max_abs(retained_angles, recomputed_retained_angles),
    )
    angle_exact_parity_max = _max_abs(
        retained_angles,
        np.repeat(exact_angles[None, :], len(retained_angles), axis=0),
    )
    angle_constancy_max = float(
        np.max(np.ptp(retained_angles, axis=0), initial=0.0)
    )

    finite_time_exponents = np.sum(
        np.log(np.diagonal(r_history, axis1=1, axis2=2)), axis=0
    ) / (NUMBER_OF_STEPS * coordinate_step)
    exponent_artifact_max_diff = _max_abs(
        fixture["finite_time_exponents"], finite_time_exponents
    )
    exponent_exact_max_diff = _max_abs(finite_time_exponents, exact_exponents)

    commutator = matrix.T @ matrix - matrix @ matrix.T
    nonnormal_norm = float(np.linalg.norm(commutator, ord="fro"))
    nonnormal_artifact_diff = abs(
        nonnormal_norm - float(fixture["non_normal_commutator_frobenius_norm"])
    )

    checkpoint_indices = np.asarray(
        fixture["direct_checkpoints"]["indices"], dtype=np.int64
    )
    checkpoint_contract = bool(np.array_equal(checkpoint_indices, DIRECT_CHECKPOINT_INDICES))
    passed = bool(
        declared_numeric_max_diff <= PYTHON_SCHEMA_TOLERANCE
        and orthogonality_max <= PYTHON_QR_TOLERANCE
        and qr_relative_max <= PYTHON_QR_TOLERANCE
        and positive_r_diagonal
        and exact_line_max <= PYTHON_LINE_TOLERANCE
        and covariance_max <= PYTHON_LINE_TOLERANCE
        and angle_artifact_max_diff <= PYTHON_ANGLE_TOLERANCE
        and angle_exact_parity_max <= PYTHON_ANGLE_TOLERANCE
        and angle_constancy_max <= PYTHON_ANGLE_TOLERANCE
        and exponent_artifact_max_diff <= PYTHON_SCHEMA_TOLERANCE
        and exponent_exact_max_diff <= PYTHON_SCHEMA_TOLERANCE
        and nonnormal_norm > 0.0
        and nonnormal_artifact_diff <= PYTHON_SCHEMA_TOLERANCE
        and checkpoint_contract
    )
    return {
        "fixture": name,
        "shape_contract": shape_contract,
        "declared_numeric_diffs": declared_numeric_diffs,
        "declared_numeric_max_diff": declared_numeric_max_diff,
        "q_orthogonality_max_residual": orthogonality_max,
        "qr_factorization_max_relative_residual": qr_relative_max,
        "positive_r_diagonal": positive_r_diagonal,
        "retained_exact_line_max_distance": exact_line_max,
        "retained_covariance_max_line_distance": covariance_max,
        "angle_artifact_max_diff": angle_artifact_max_diff,
        "angle_exact_parity_max_residual": angle_exact_parity_max,
        "angle_constancy_max_residual": angle_constancy_max,
        "finite_time_exponent_artifact_max_diff": exponent_artifact_max_diff,
        "finite_time_exponent_exact_max_diff": exponent_exact_max_diff,
        "non_normal_commutator_frobenius_norm": nonnormal_norm,
        "non_normal_commutator_artifact_diff": nonnormal_artifact_diff,
        "direct_checkpoint_contract": checkpoint_contract,
        "passed": passed,
    }


def validate_wolfram_summary(summary_path: Path = DEFAULT_SUMMARY) -> dict[str, Any]:
    """Validate the independent artifact without importing HAFO."""

    summary_path = Path(summary_path).resolve()
    payload = _load(summary_path)
    if payload.get("system_id") != SYSTEM_ID:
        raise ValueError(
            f"unexpected system_id {payload.get('system_id')!r}; expected {SYSTEM_ID!r}"
        )
    fixtures = payload.get("fixtures")
    if not isinstance(fixtures, dict) or tuple(fixtures) != EXPECTED_FIXTURE_NAMES:
        raise ValueError(f"unexpected Wolfram fixtures: {tuple(fixtures or ())!r}")

    source = payload.get("source", {})
    source_anchors_match = all(
        source.get(key) == value for key, value in SOURCE_ANCHORS.items()
    )
    independence_flags_match = bool(
        source.get("hafo_source_read") is False
        and source.get("report_input_used") is False
        and source.get("hafo_formula_imported") is False
        and source.get("built_in_qr_used") is False
        and source.get("built_in_eigensystem_used") is False
    )
    conventions = payload.get("conventions", {})
    convention_contract_match = bool(
        conventions.get("working_precision") == 80
        and conventions.get("number_of_steps") == NUMBER_OF_STEPS
        and conventions.get("retained_indices_inclusive") == RETAINED_INDICES.tolist()
        and conventions.get("direct_checkpoint_indices")
        == DIRECT_CHECKPOINT_INDICES.tolist()
        and conventions.get("vectors_are_columns") is True
        and conventions.get("fractional_order_supported_by_this_oracle") is False
        and conventions.get("sign_orientation") == "unoriented_projective_lines"
        and "modified_gram_schmidt" in conventions.get("qr_algorithm", "")
    )
    wolfram_tests_pass = bool(
        payload.get("passed") is True
        and len(payload.get("tests", ())) >= 17
        and all(test.get("passed") is True for test in payload["tests"])
    )
    evidence_boundary = str(payload.get("evidence_boundary", ""))
    evidence_boundary_match = bool(
        "integer-order" in evidence_boundary
        and "no general nonlinear CLV convergence theorem" in evidence_boundary
        and "fractional-order CLV validity" in evidence_boundary
        and "nonunique" in evidence_boundary
    )
    fixture_results = {
        name: _validate_fixture(name, fixtures[name])
        for name in EXPECTED_FIXTURE_NAMES
    }
    passed = bool(
        source_anchors_match
        and independence_flags_match
        and convention_contract_match
        and wolfram_tests_pass
        and evidence_boundary_match
        and all(result["passed"] for result in fixture_results.values())
    )
    return {
        "validation_scope": payload.get("validation_scope"),
        "evidence_boundary": evidence_boundary,
        "summary_path": str(summary_path),
        "source_anchors_match": source_anchors_match,
        "independence_flags_match": independence_flags_match,
        "convention_contract_match": convention_contract_match,
        "wolfram_tests_pass": wolfram_tests_pass,
        "evidence_boundary_match": evidence_boundary_match,
        "fixture_results": fixture_results,
        "passed": passed,
    }


@lru_cache(maxsize=1)
def _load_public_api() -> dict[str, Callable[..., Any] | None]:
    try:
        module = importlib.import_module(PUBLIC_API_MODULE)
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        raise PublicCLVAPIUnavailable(
            f"public CLV module {PUBLIC_API_MODULE!r} is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not hasattr(module, PUBLIC_CORE_NAME):
        raise PublicCLVAPIUnavailable(
            f"public CLV module {PUBLIC_API_MODULE!r} lacks {PUBLIC_CORE_NAME!r}"
        )
    return {
        PUBLIC_CORE_NAME: getattr(module, PUBLIC_CORE_NAME),
        PUBLIC_ANGLE_NAME: getattr(module, PUBLIC_ANGLE_NAME, None),
    }


def public_api_status() -> tuple[bool, str]:
    """Report the optional core status without making parser use depend on it."""

    try:
        api = _load_public_api()
    except PublicCLVAPIUnavailable as exc:
        return False, str(exc)
    angle = "available" if api[PUBLIC_ANGLE_NAME] is not None else "absent"
    return True, f"{PUBLIC_API_MODULE}.{PUBLIC_CORE_NAME}; angle API {angle}"


def _supported_kwargs(function: Callable[..., Any], values: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(function).parameters
    return {key: value for key, value in values.items() if key in parameters}


def _extract_vectors(result: Any) -> np.ndarray:
    if isinstance(result, np.ndarray):
        return _array(result, ndim=3)
    if isinstance(result, (tuple, list)) and result:
        candidate = np.asarray(result[0])
        if candidate.ndim == 3:
            return _array(candidate, ndim=3)
    if isinstance(result, dict):
        for name in ("clvs", "vectors", "covariant_vectors"):
            if name in result:
                return _array(result[name], ndim=3)
    for name in ("clvs", "vectors", "covariant_vectors"):
        if hasattr(result, name):
            return _array(getattr(result, name), ndim=3)
    raise TypeError("public CLV result exposes no three-dimensional vector history")


def _retained_from_history(history: np.ndarray) -> np.ndarray:
    if history.shape[0] == NUMBER_OF_STEPS + 1:
        return history[RETAINED_INDICES]
    if history.shape[0] == NUMBER_OF_STEPS:
        return history[RETAINED_INDICES - 1]
    if history.shape[0] == RETAINED_INDICES.size:
        return history
    raise ValueError(f"unsupported public CLV history length {history.shape[0]}")


def _best_vector_layout(history: np.ndarray, expected: np.ndarray) -> np.ndarray:
    candidates = [history]
    if history.shape[1] == history.shape[2]:
        candidates.append(np.swapaxes(history, 1, 2))
    scored = []
    for candidate in candidates:
        retained = _retained_from_history(candidate)
        if retained.shape != expected.shape:
            continue
        score = max(
            float(np.max(_line_distances(left, right), initial=0.0))
            for left, right in zip(retained, expected, strict=True)
        )
        scored.append((score, candidate))
    if not scored:
        raise ValueError("public CLV history cannot be aligned to Wolfram layout")
    return min(scored, key=lambda item: item[0])[1]


def _invoke_core(function: Callable[..., Any], fixture: dict[str, Any]) -> Any:
    q_history = _array(fixture["q_history"], ndim=3)
    r_history = _array(fixture["r_history"], ndim=3)
    terminal = _array(fixture["terminal_coefficients"], ndim=2)
    keyword_values = {
        "terminal_coefficients": terminal,
        "terminal_seed": terminal,
        "backend": "numpy",
        "q": 1.0,
        "coordinates": np.arange(NUMBER_OF_STEPS + 1, dtype=np.float64),
        "system_kind": str(fixture["kind"]),
        "coordinate_kind": "iteration" if fixture["kind"] == "map" else "time_step",
        "metadata": {"oracle": "independent_wolfram_constant_cocycle"},
    }
    return function(
        q_history,
        r_history,
        **_supported_kwargs(function, keyword_values),
    )


def _extract_angles(result: Any) -> np.ndarray:
    if isinstance(result, np.ndarray):
        return _array(result)
    if isinstance(result, dict):
        for name in ("pair_angles", "angles"):
            if name in result:
                return _array(result[name])
    for name in ("pair_angles", "angles"):
        if hasattr(result, name):
            return _array(getattr(result, name))
    raise TypeError("public CLV angle result exposes no angle array")


def _compare_public_fixture(
    name: str,
    fixture: dict[str, Any],
    api: dict[str, Callable[..., Any] | None],
) -> dict[str, Any]:
    expected = _array(fixture["retained_clvs"], ndim=3)
    core_result = _invoke_core(api[PUBLIC_CORE_NAME], fixture)  # type: ignore[arg-type]
    history = _best_vector_layout(_extract_vectors(core_result), expected)
    retained = _retained_from_history(history)
    matrix = _cocycle_matrix(fixture)
    line_max = max(
        float(np.max(_line_distances(left, right), initial=0.0))
        for left, right in zip(retained, expected, strict=True)
    )
    covariance_max = max(
        float(
            np.max(
                _line_distances(matrix @ retained[index], retained[index + 1]),
                initial=0.0,
            )
        )
        for index in range(retained.shape[0] - 1)
    )

    angle_function = api[PUBLIC_ANGLE_NAME]
    angle_performed = angle_function is not None
    angle_max_diff: float | None = None
    if angle_function is not None:
        pairs = np.asarray(fixture["pair_indices_zero_based"], dtype=np.int64)
        keyword_values = {
            "pairs": [tuple(map(int, pair)) for pair in pairs],
            "pair_indices": [tuple(map(int, pair)) for pair in pairs],
            "window": None,
            "q": 1.0,
        }
        # The HAFO public contract is (samples, n_vectors, dimension), while
        # the Wolfram oracle stores mathematical matrices with vectors in
        # columns, (samples, dimension, n_vectors).
        angle_result = angle_function(
            np.swapaxes(retained, 1, 2),
            **_supported_kwargs(angle_function, keyword_values),
        )
        observed_angles = _extract_angles(angle_result)
        expected_angles = _array(fixture["retained_pair_angles_radians"], ndim=2)
        if observed_angles.ndim == 1 and expected_angles.shape[1] == 1:
            observed_angles = observed_angles[:, None]
        if observed_angles.shape != expected_angles.shape:
            raise ValueError(
                f"public angle shape {observed_angles.shape} != {expected_angles.shape}"
            )
        angle_max_diff = _max_abs(observed_angles, expected_angles)

    passed = bool(
        line_max <= CORE_LINE_TOLERANCE
        and covariance_max <= CORE_COVARIANCE_TOLERANCE
        and (angle_max_diff is None or angle_max_diff <= PYTHON_ANGLE_TOLERANCE)
    )
    return {
        "fixture": name,
        "line_distance_max_diff": line_max,
        "covariance_max_line_distance": covariance_max,
        "angle_comparison_performed": angle_performed,
        "angle_max_diff": angle_max_diff,
        "passed": passed,
    }


def compare_wolfram_summary(
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    require_core: bool = False,
) -> dict[str, Any]:
    """Validate Wolfram independently, then compare HAFO when importable."""

    independent = validate_wolfram_summary(summary_path)
    available, diagnostic = public_api_status()
    public_results: dict[str, Any] = {}
    public_passed: bool | None = None
    public_error: str | None = None
    if available:
        try:
            api = _load_public_api()
            payload = _load(Path(summary_path))
            public_results = {
                name: _compare_public_fixture(name, payload["fixtures"][name], api)
                for name in EXPECTED_FIXTURE_NAMES
            }
            public_passed = all(result["passed"] for result in public_results.values())
        except (TypeError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            public_passed = False
            public_error = f"{type(exc).__name__}: {exc}"
    passed = bool(
        independent["passed"]
        and (not available or public_passed is True)
        and (not require_core or (available and public_passed is True))
    )
    return {
        "validation_scope": (
            "independent_Wolfram_integer_CLV_constant_cocycle_consistency"
        ),
        "evidence_boundary": independent["evidence_boundary"],
        "summary_path": str(Path(summary_path).resolve()),
        "independent": independent,
        "public_api_module": PUBLIC_API_MODULE,
        "public_core_name": PUBLIC_CORE_NAME,
        "public_angle_name": PUBLIC_ANGLE_NAME,
        "public_api_available": available,
        "public_api_diagnostic": diagnostic,
        "cross_implementation_performed": available,
        "cross_implementation_results": public_results,
        "cross_implementation_passed": public_passed,
        "cross_implementation_error": public_error,
        "require_core": require_core,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--require-core", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = compare_wolfram_summary(args.summary, require_core=args.require_core)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
