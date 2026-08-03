"""Compare public integer SALI/GALI APIs with independent Wolfram fixtures.

The Wolfram case constructs exact tangent histories for an orthogonal map, an
area-preserving hyperbolic diagonal map, and a constant-Jacobian diagonal
flow.  It derives SALI from the parallel/antiparallel norms and GALI from Gram
and Cauchy--Binet formulas before checking the SVD/LDI volume independently.
It never reads HAFO source or generated reports.

This comparator imports the agreed public API lazily.  A checkout in which
``hidden_attractors.analysis.alignment_indices`` is not yet present therefore
raises :class:`PublicAlignmentAPIUnavailable` only when a valid comparison is
requested; artifact schema and hygiene tests remain collectable.

Passing is finite exact-linear tangent-algebra and numerical consistency
evidence only.  It is not a nonlinear chaos classification, a convergence
theorem, a Lyapunov-spectrum validation, or evidence of attraction,
hiddenness, or fractional SALI/GALI validity.
"""

from __future__ import annotations

import argparse
import importlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ID = "sali_gali_integer"
DEFAULT_SUMMARY = (
    ROOT
    / "validation"
    / "outputs"
    / "wolfram"
    / f"{SYSTEM_ID}_verified"
    / f"{SYSTEM_ID}_validation_summary.json"
)

CORE_NUMERIC_TOLERANCE = 5.0e-13
MAP_NUMERIC_TOLERANCE = 2.0e-12
FLOW_NUMERIC_TOLERANCE = 2.0e-10

SOURCE_ANCHORS = {
    "sali_doi": "10.1088/0305-4470/37/24/006",
    "gali_doi": "10.1016/j.physd.2007.04.004",
    "rolim_sales_leonel_antonopoulos_doi":
        "10.1016/j.chaos.2026.117884",
}
EXPECTED_FIXTURE_NAMES = (
    "orthogonal_rotation_map",
    "hyperbolic_diagonal_map",
    "hyperbolic_diagonal_flow",
)
PUBLIC_API_MODULE = "hidden_attractors.analysis.alignment_indices"
PUBLIC_API_NAMES = (
    "smaller_alignment_index",
    "generalized_alignment_index",
    "linear_dependence_index",
    "alignment_indices_from_tangent_history",
    "integer_map_alignment_indices",
    "integer_flow_alignment_indices",
)


class PublicAlignmentAPIUnavailable(RuntimeError):
    """The agreed public SALI/GALI module or one of its symbols is absent."""


@lru_cache(maxsize=1)
def _load_public_api() -> dict[str, Callable[..., Any]]:
    try:
        module = importlib.import_module(PUBLIC_API_MODULE)
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        raise PublicAlignmentAPIUnavailable(
            f"public SALI/GALI API module {PUBLIC_API_MODULE!r} is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    missing = tuple(name for name in PUBLIC_API_NAMES if not hasattr(module, name))
    if missing:
        raise PublicAlignmentAPIUnavailable(
            f"public SALI/GALI module {PUBLIC_API_MODULE!r} is missing symbols: "
            + ", ".join(missing)
        )
    return {name: getattr(module, name) for name in PUBLIC_API_NAMES}


def public_api_status() -> tuple[bool, str]:
    """Return availability plus a stable diagnostic without failing import."""

    try:
        _load_public_api()
    except PublicAlignmentAPIUnavailable as exc:
        return False, str(exc)
    return True, f"{PUBLIC_API_MODULE}: " + ", ".join(PUBLIC_API_NAMES)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _max_abs(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError(
            "shape mismatch in Wolfram/Python comparison: "
            f"{left_array.shape} != {right_array.shape}"
        )
    return float(np.max(np.abs(left_array - right_array), initial=0.0))


def _orders_for_fixture(fixture: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        order for order in (2, 3) if f"gali_{order}" in fixture
    )


def _expected_gali_matrix(
    fixture: dict[str, Any],
    orders: tuple[int, ...],
) -> np.ndarray:
    return np.column_stack(
        [np.asarray(fixture[f"gali_{order}"], dtype=np.float64) for order in orders]
    )


def _normalized_columns(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    return matrix / norms


def _compare_public_core_fixture(
    name: str,
    fixture: dict[str, Any],
    api: dict[str, Callable[..., Any]],
) -> dict[str, Any]:
    matrices = np.asarray(
        fixture["tangent_matrices_columns"],
        dtype=np.float64,
    )
    history = np.asarray(fixture["tangent_history"], dtype=np.float64)
    expected_sali = np.asarray(fixture["sali"], dtype=np.float64)
    orders = _orders_for_fixture(fixture)
    expected_gali = _expected_gali_matrix(fixture, orders)
    coordinates_key = "times" if fixture["kind"] == "flow" else "iterations"
    coordinates = np.asarray(fixture[coordinates_key], dtype=np.float64)

    smaller_alignment_index = api["smaller_alignment_index"]
    generalized_alignment_index = api["generalized_alignment_index"]
    linear_dependence_index = api["linear_dependence_index"]
    alignment_indices_from_tangent_history = api[
        "alignment_indices_from_tangent_history"
    ]

    direct_sali = np.asarray(
        [smaller_alignment_index(matrix) for matrix in matrices],
        dtype=np.float64,
    )
    direct_gali = np.column_stack(
        [
            np.asarray(
                [
                    generalized_alignment_index(
                        matrix,
                        order=order,
                        backend="numpy",
                    )
                    for matrix in matrices
                ],
                dtype=np.float64,
            )
            for order in orders
        ]
    )
    direct_ldi = np.column_stack(
        [
            np.asarray(
                [
                    linear_dependence_index(
                        matrix,
                        order=order,
                        backend="numpy",
                    )
                    for matrix in matrices
                ],
                dtype=np.float64,
            )
            for order in orders
        ]
    )
    result = alignment_indices_from_tangent_history(
        history,
        coordinates=coordinates,
        gali_orders=orders,
        backend="numpy",
        system_kind=str(fixture["kind"]),
        coordinate_kind=("time" if fixture["kind"] == "flow" else "iteration"),
        method="independent_wolfram_exact_tangent_history",
        q=1.0,
        metadata={"fixture": name, "oracle": "independent_wolfram"},
    )

    metrics = {
        "direct_sali_max_diff": _max_abs(direct_sali, expected_sali),
        "direct_gali_max_diff": _max_abs(direct_gali, expected_gali),
        "direct_ldi_max_diff": _max_abs(direct_ldi, expected_gali),
        "direct_ldi_to_gali_max_diff": _max_abs(direct_ldi, direct_gali),
        "history_coordinates_max_diff": _max_abs(result.coordinates, coordinates),
        "history_sali_max_diff": _max_abs(result.sali, expected_sali),
        "history_gali_max_diff": _max_abs(result.gali, expected_gali),
        "history_log_sali_max_diff": _max_abs(
            result.log_sali,
            np.log(expected_sali),
        ),
        "history_log_gali_max_diff": _max_abs(
            result.log_gali,
            np.log(expected_gali),
        ),
        "initial_deviations_max_diff": _max_abs(
            result.initial_deviations,
            _normalized_columns(matrices[0]),
        ),
        "final_deviations_max_diff": _max_abs(
            result.final_deviations,
            _normalized_columns(matrices[-1]),
        ),
    }
    metrics["max_diff"] = max(metrics.values())
    contract_match = bool(
        result.status == "ok"
        and result.method_id == "alignment_indices_from_tangent_history"
        and result.system_kind == fixture["kind"]
        and result.backend == "numpy"
        and result.volume_method == "svd_product"
        and result.derivative_model == "integer"
        and result.q == 1.0
        and result.normalization == "independent_l2"
        and result.orthonormalization == "none_during_evolution"
        and tuple(int(value) for value in result.gali_orders) == orders
        and result.gali.shape == expected_gali.shape
        and result.censored.shape == expected_gali.shape
        and not np.any(result.censored)
        and result.initial_deviations.shape == matrices[0].shape
        and result.final_deviations.shape == matrices[-1].shape
        and result.metadata["history_layout"] == "samples_vectors_dimension"
        and result.metadata["instantaneous_layout"]
        == "dimension_vectors_columns"
        and result.metadata["fixture"] == name
    )
    return {
        "fixture": name,
        "orders": list(orders),
        "contract_match": contract_match,
        "metrics": metrics,
        "passed": bool(
            contract_match
            and metrics["max_diff"] <= CORE_NUMERIC_TOLERANCE
        ),
    }


def _linear_callbacks(
    matrix: np.ndarray,
) -> tuple[Callable[[float, np.ndarray], np.ndarray], Callable[[float, np.ndarray], np.ndarray]]:
    def evolution(_coordinate: float, state: np.ndarray) -> np.ndarray:
        return matrix @ state

    def jacobian(_coordinate: float, _state: np.ndarray) -> np.ndarray:
        return matrix

    return evolution, jacobian


def _compare_map_facade(
    name: str,
    fixture: dict[str, Any],
    api: dict[str, Callable[..., Any]],
) -> dict[str, Any]:
    matrix = np.asarray(fixture["matrix"], dtype=np.float64)
    initial = np.asarray(
        fixture["initial_deviations_columns"],
        dtype=np.float64,
    )
    coordinates = np.asarray(fixture["iterations"], dtype=np.float64)
    expected_sali = np.asarray(fixture["sali"], dtype=np.float64)
    orders = _orders_for_fixture(fixture)
    expected_gali = _expected_gali_matrix(fixture, orders)
    evolution, jacobian = _linear_callbacks(matrix)

    result = api["integer_map_alignment_indices"](
        evolution,
        jacobian,
        np.zeros(matrix.shape[0], dtype=np.float64),
        iterations=int(coordinates[-1]),
        sample_every=1,
        gali_orders=orders,
        initial_deviations=initial,
        method="variational",
        backend="numpy",
        q=1.0,
    )
    metrics = {
        "coordinates_max_diff": _max_abs(result.coordinates, coordinates),
        "sali_max_diff": _max_abs(result.sali, expected_sali),
        "gali_max_diff": _max_abs(result.gali, expected_gali),
        "log_sali_max_diff": _max_abs(result.log_sali, np.log(expected_sali)),
        "log_gali_max_diff": _max_abs(result.log_gali, np.log(expected_gali)),
        "final_deviations_max_diff": _max_abs(
            result.final_deviations,
            np.asarray(
                fixture["normalized_tangent_matrices_columns"][-1],
                dtype=np.float64,
            ),
        ),
    }
    metrics["max_diff"] = max(metrics.values())
    contract_match = bool(
        result.status == "ok"
        and result.method_id == "integer_map_sali_gali_variational"
        and result.system_kind == "map"
        and result.coordinate_kind == "iteration_after_transient"
        and result.evolution_method == "variational"
        and result.backend == "numpy"
        and result.volume_method == "svd_product"
        and result.q == 1.0
        and result.metadata["jacobian_source"] == "analytic"
        and result.metadata["normalization"]
        == "independent_l2_no_evolution_qr"
        and result.metadata["iterations_completed"] == int(coordinates[-1])
        and not np.any(result.censored)
    )
    return {
        "fixture": name,
        "contract_match": contract_match,
        "metrics": metrics,
        "passed": bool(
            contract_match
            and metrics["max_diff"] <= MAP_NUMERIC_TOLERANCE
        ),
    }


def _select_coordinates(
    available: np.ndarray,
    requested: np.ndarray,
) -> np.ndarray:
    indices: list[int] = []
    for value in requested:
        matches = np.flatnonzero(np.isclose(available, value, atol=2.0e-14, rtol=0.0))
        if matches.size != 1:
            raise ValueError(
                f"expected one flow output at {value}, found {matches.size}"
            )
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=int)


def _compare_flow_facade(
    fixture: dict[str, Any],
    api: dict[str, Callable[..., Any]],
) -> dict[str, Any]:
    generator = np.asarray(fixture["generator"], dtype=np.float64)
    initial = np.asarray(
        fixture["initial_deviations_columns"],
        dtype=np.float64,
    )
    requested_times = np.asarray(fixture["times"], dtype=np.float64)
    expected_sali = np.asarray(fixture["sali"], dtype=np.float64)
    orders = _orders_for_fixture(fixture)
    expected_gali = _expected_gali_matrix(fixture, orders)
    evolution, jacobian = _linear_callbacks(generator)

    result = api["integer_flow_alignment_indices"](
        evolution,
        jacobian,
        np.zeros(generator.shape[0], dtype=np.float64),
        t_final=float(requested_times[-1]),
        renormalization_time=0.25,
        gali_orders=orders,
        initial_deviations=initial,
        method="variational",
        rtol=1.0e-12,
        atol=1.0e-14,
        max_step=0.05,
        backend="numpy",
        q=1.0,
    )
    indices = _select_coordinates(result.coordinates, requested_times)
    selected_sali = result.sali[indices]
    selected_gali = result.gali[indices]
    metrics = {
        "coordinates_max_diff": _max_abs(
            result.coordinates[indices],
            requested_times,
        ),
        "sali_max_diff": _max_abs(selected_sali, expected_sali),
        "gali_max_diff": _max_abs(selected_gali, expected_gali),
        "log_sali_max_diff": _max_abs(
            result.log_sali[indices],
            np.log(expected_sali),
        ),
        "log_gali_max_diff": _max_abs(
            result.log_gali[indices],
            np.log(expected_gali),
        ),
        "final_deviations_max_diff": _max_abs(
            result.final_deviations,
            np.asarray(
                fixture["normalized_tangent_matrices_columns"][-1],
                dtype=np.float64,
            ),
        ),
        "sampled_state_zero_max": float(
            np.max(np.abs(result.sampled_states), initial=0.0)
        ),
    }
    metrics["max_diff"] = max(metrics.values())
    contract_match = bool(
        result.status == "ok"
        and result.method_id == "integer_flow_sali_gali_variational"
        and result.system_kind == "flow"
        and result.coordinate_kind == "time_after_burn"
        and result.evolution_method == "variational"
        and result.backend == "numpy"
        and result.volume_method == "svd_product"
        and result.q == 1.0
        and result.metadata["solver_method"] == "DOP853"
        and result.metadata["jacobian_source"] == "analytic"
        and result.metadata["normalization"]
        == "independent_l2_no_evolution_qr"
        and result.metadata["segments"] == 12
        and not np.any(result.censored)
    )
    return {
        "fixture": "hyperbolic_diagonal_flow",
        "contract_match": contract_match,
        "metrics": metrics,
        "passed": bool(
            contract_match
            and metrics["max_diff"] <= FLOW_NUMERIC_TOLERANCE
        ),
    }


def compare_wolfram_summary(
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    core_tolerance: float = CORE_NUMERIC_TOLERANCE,
    map_tolerance: float = MAP_NUMERIC_TOLERANCE,
    flow_tolerance: float = FLOW_NUMERIC_TOLERANCE,
) -> dict[str, Any]:
    """Recompute all exact fixtures through the agreed public HAFO APIs."""

    summary_path = Path(summary_path).resolve()
    payload = _load(summary_path)
    if payload.get("system_id") != SYSTEM_ID:
        raise ValueError(
            f"unexpected system_id {payload.get('system_id')!r}; "
            f"expected {SYSTEM_ID!r}"
        )

    fixtures = payload.get("fixtures")
    if not isinstance(fixtures, dict):
        raise TypeError("Wolfram summary must contain a fixtures object")
    if tuple(fixtures) != EXPECTED_FIXTURE_NAMES:
        raise ValueError(
            "unexpected Wolfram fixture ordering/names: "
            f"{tuple(fixtures)!r}"
        )

    api = _load_public_api()
    core_results = {
        name: _compare_public_core_fixture(name, fixtures[name], api)
        for name in EXPECTED_FIXTURE_NAMES
    }
    map_results = {
        name: _compare_map_facade(name, fixtures[name], api)
        for name in EXPECTED_FIXTURE_NAMES[:2]
    }
    flow_result = _compare_flow_facade(
        fixtures["hyperbolic_diagonal_flow"],
        api,
    )

    source = payload["source"]
    source_anchors_match = all(
        source.get(key) == expected
        for key, expected in SOURCE_ANCHORS.items()
    )
    independence_flags_match = bool(
        source.get("hafo_source_read") is False
        and source.get("report_input_used") is False
        and source.get("hafo_formula_imported") is False
    )
    wolfram_tests_pass = bool(
        payload.get("passed") is True
        and payload.get("tests")
        and all(test.get("passed") is True for test in payload["tests"])
    )
    conventions = payload.get("conventions", {})
    convention_contract_match = bool(
        conventions.get("instantaneous_matrix_shape")
        == "dimension_by_n_vectors_columns"
        and conventions.get("history_shape")
        == "n_samples_by_n_vectors_by_dimension"
        and conventions.get("normalization")
        == "independent_l2_normalization_per_vector"
        and conventions.get("orthogonalization_between_vectors") == "none"
        and conventions.get("gali_orders") == [2, 3]
        and conventions.get("threshold_rule") == "strictly_less_than"
    )
    wolfram_numeric = payload.get("numeric_cross_checks", {})
    wolfram_high_precision_match = bool(
        int(wolfram_numeric.get("working_precision", 0)) == 80
        and abs(
            float(
                wolfram_numeric[
                    "rotation_gali3_gram_to_svd_ldi_max_residual"
                ]
            )
        )
        <= 1.0e-40
        and abs(
            float(
                wolfram_numeric[
                    "map_gali2_gram_to_svd_ldi_max_residual"
                ]
            )
        )
        <= 1.0e-40
        and abs(
            float(
                wolfram_numeric[
                    "flow_gali3_gram_to_svd_ldi_max_residual"
                ]
            )
        )
        <= 1.0e-40
    )

    core_max_diff = max(
        result["metrics"]["max_diff"] for result in core_results.values()
    )
    map_max_diff = max(
        result["metrics"]["max_diff"] for result in map_results.values()
    )
    flow_max_diff = float(flow_result["metrics"]["max_diff"])
    core_pass = bool(
        all(result["contract_match"] for result in core_results.values())
        and core_max_diff <= float(core_tolerance)
    )
    map_pass = bool(
        all(result["contract_match"] for result in map_results.values())
        and map_max_diff <= float(map_tolerance)
    )
    flow_pass = bool(
        flow_result["contract_match"]
        and flow_max_diff <= float(flow_tolerance)
    )
    passed = bool(
        source_anchors_match
        and independence_flags_match
        and wolfram_tests_pass
        and convention_contract_match
        and wolfram_high_precision_match
        and core_pass
        and map_pass
        and flow_pass
    )
    return {
        "validation_scope": (
            "independent_Wolfram_to_HAFO_integer_SALI_GALI_"
            "finite_exact_linear_consistency"
        ),
        "evidence_boundary": payload["evidence_boundary"],
        "summary_path": str(summary_path),
        "public_api_module": PUBLIC_API_MODULE,
        "public_api_names": list(PUBLIC_API_NAMES),
        "tolerances": {
            "core": float(core_tolerance),
            "map": float(map_tolerance),
            "flow": float(flow_tolerance),
        },
        "source_anchors_match": source_anchors_match,
        "independence_flags_match": independence_flags_match,
        "wolfram_tests_pass": wolfram_tests_pass,
        "wolfram_high_precision_match": wolfram_high_precision_match,
        "convention_contract_match": convention_contract_match,
        "core_results": core_results,
        "map_results": map_results,
        "flow_result": flow_result,
        "core_max_diff": core_max_diff,
        "map_max_diff": map_max_diff,
        "flow_max_diff": flow_max_diff,
        "cross_implementation_max_diff": max(
            core_max_diff,
            map_max_diff,
            flow_max_diff,
        ),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--core-tolerance",
        type=float,
        default=CORE_NUMERIC_TOLERANCE,
    )
    parser.add_argument(
        "--map-tolerance",
        type=float,
        default=MAP_NUMERIC_TOLERANCE,
    )
    parser.add_argument(
        "--flow-tolerance",
        type=float,
        default=FLOW_NUMERIC_TOLERANCE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination for the Python comparison JSON.",
    )
    args = parser.parse_args()
    result = compare_wolfram_summary(
        args.summary,
        core_tolerance=args.core_tolerance,
        map_tolerance=args.map_tolerance,
        flow_tolerance=args.flow_tolerance,
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
