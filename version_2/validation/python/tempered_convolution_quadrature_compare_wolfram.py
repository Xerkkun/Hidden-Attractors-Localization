"""Validate an independent Wolfram tempered-CQ artifact and compare HAFO.

The Wolfram case builds BDF1/BDF2 fractional weights independently from an
explicit coefficient recurrence and from generalized-binomial factor
expansions at 80-digit precision.  It evaluates exponentially tempered
Riemann--Liouville and Caputo sampled operators through two algebraically
equivalent paths.  This module validates the exported artifact without
importing HAFO.  Only the optional public-core comparison imports the package.

Passing is finite sampled-grid implementation evidence.  It is not a general
stability or convergence theorem and is not evidence of chaos, attraction, or
hiddenness.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import tempfile
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ID = "tempered_convolution_quadrature"
TEMP_ROOT = Path(tempfile.gettempdir())
DEFAULT_SUMMARY = (
    TEMP_ROOT
    / "hafo_tempered_convolution_quadrature"
    / f"{SYSTEM_ID}_validation_summary.json"
)

SOURCE_ANCHORS = {
    "lubich_doi": "10.1137/0517050",
    "sabzikar_meerschaert_chen_doi": "10.1016/j.jcp.2014.04.024",
    "guo_tempered_cq_doi": "10.1137/18M1230153",
    "chen_deng_tempered_algorithms_doi": "10.1051/m2an/2014037",
}

EXPECTED_DEFINITIONS = (
    "tempered_riemann_liouville",
    "tempered_caputo",
)
EXPECTED_BDF_ORDERS = (1, 2)
INDEPENDENT_TOLERANCE = 2.0e-12
CORE_TOLERANCE = 2.0e-11

PUBLIC_API_MODULE = "hidden_attractors.fractional"
PUBLIC_CORE_CANDIDATES = (
    "tempered_convolution_quadrature",
    "tempered_lubich_convolution_quadrature",
)
PUBLIC_CORE_NAME = PUBLIC_CORE_CANDIDATES[0]


class PublicTemperedCQAPIUnavailable(RuntimeError):
    """The optional public tempered-CQ implementation is unavailable."""


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
        raise ValueError("Wolfram fixture contains a non-finite value")
    return result


def _max_abs(left: Any, right: Any) -> float:
    left_array = _array(left)
    right_array = _array(right)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"shape mismatch in comparison: {left_array.shape} != "
            f"{right_array.shape}"
        )
    return float(np.max(np.abs(left_array - right_array), initial=0.0))


def _bdf_coefficients(bdf_order: int) -> np.ndarray:
    if bdf_order == 1:
        return np.asarray([1.0, -1.0])
    if bdf_order == 2:
        return np.asarray([1.5, -2.0, 0.5])
    raise ValueError(f"unsupported BDF order {bdf_order}")


def _recurrence_weights(order: float, count: int, bdf_order: int) -> np.ndarray:
    """Rebuild coefficients from ``delta W'=q delta' W`` in float64."""

    delta = _bdf_coefficients(bdf_order)
    weights = np.zeros(int(count), dtype=np.float64)
    if count == 0:
        return weights
    weights[0] = delta[0] ** float(order)
    degree = delta.size - 1
    for n in range(1, count):
        numerator = 0.0
        for j in range(1, min(degree, n) + 1):
            numerator += (
                j * (float(order) + 1.0) - n
            ) * delta[j] * weights[n - j]
        weights[n] = numerator / (n * delta[0])
    return weights


def _unit_factor_weights(order: float, count: int) -> np.ndarray:
    """Coefficients of ``(1-z)**order`` from generalized binomials."""

    result = np.zeros(count, dtype=np.float64)
    if count == 0:
        return result
    result[0] = 1.0
    for index in range(1, count):
        result[index] = (
            result[index - 1] * (index - 1.0 - float(order)) / index
        )
    return result


def _factor_expansion_weights(
    order: float,
    count: int,
    bdf_order: int,
) -> np.ndarray:
    first = _unit_factor_weights(order, count)
    if bdf_order == 1:
        return first
    if bdf_order != 2:
        raise ValueError(f"unsupported BDF order {bdf_order}")
    second = first / (3.0 ** np.arange(count, dtype=np.float64))
    return (1.5**float(order)) * np.convolve(first, second)[:count]


def _cq_apply(
    samples: Any,
    order: float,
    step: float,
    bdf_order: int,
) -> np.ndarray:
    values = _array(samples, ndim=1)
    weights = _recurrence_weights(order, values.size, bdf_order)
    return np.convolve(values, weights)[: values.size] / (float(step) ** order)


def _is_caputo(definition: str) -> bool:
    if definition not in EXPECTED_DEFINITIONS:
        raise ValueError(f"unexpected tempered definition {definition!r}")
    return definition == "tempered_caputo"


def _tempered_direct(
    samples: Any,
    order: float,
    tempering: float,
    step: float,
    bdf_order: int,
    definition: str,
) -> np.ndarray:
    values = _array(samples, ndim=1)
    elapsed = float(step) * np.arange(values.size, dtype=np.float64)
    weights = _recurrence_weights(order, values.size, bdf_order)
    damped_weights = weights * np.exp(-float(tempering) * elapsed)
    result = np.convolve(values, damped_weights)[: values.size]
    if _is_caputo(definition):
        partial_weight_sums = np.cumsum(weights)
        result -= values[0] * np.exp(-float(tempering) * elapsed) * partial_weight_sums
    return result / (float(step) ** order)


def _tempered_conjugated(
    samples: Any,
    order: float,
    tempering: float,
    step: float,
    bdf_order: int,
    definition: str,
) -> np.ndarray:
    values = _array(samples, ndim=1)
    elapsed = float(step) * np.arange(values.size, dtype=np.float64)
    transformed = np.exp(float(tempering) * elapsed) * values
    if _is_caputo(definition):
        transformed = transformed - transformed[0]
    return np.exp(-float(tempering) * elapsed) * _cq_apply(
        transformed,
        order,
        step,
        bdf_order,
    )


def _rows_by_contract(rows: Any) -> dict[tuple[str, int], dict[str, Any]]:
    if not isinstance(rows, list):
        raise TypeError("expected a list of Wolfram fixture rows")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("expected every Wolfram fixture row to be an object")
        key = (str(row["definition"]), int(row["bdf_order"]))
        if key in result:
            raise ValueError(f"duplicate Wolfram fixture row {key!r}")
        result[key] = row
    expected = {
        (definition, bdf_order)
        for definition in EXPECTED_DEFINITIONS
        for bdf_order in EXPECTED_BDF_ORDERS
    }
    if set(result) != expected:
        raise ValueError(
            f"unexpected fixture rows {tuple(result)!r}; expected {tuple(expected)!r}"
        )
    return result


def _validate_weights(payload: dict[str, Any]) -> dict[str, Any]:
    fixture = payload["weights"]
    order = float(fixture["order"])
    count = int(fixture["count"])
    rows = fixture["rows"]
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("weight fixture must contain exactly BDF1 and BDF2 rows")
    metrics: dict[str, float] = {}
    for row in rows:
        bdf_order = int(row["bdf_order"])
        recurrence = _recurrence_weights(order, count, bdf_order)
        expansion = _factor_expansion_weights(order, count, bdf_order)
        metrics[f"bdf{bdf_order}_recurrence_max_diff"] = _max_abs(
            recurrence,
            row["recurrence_weights"],
        )
        metrics[f"bdf{bdf_order}_factor_expansion_max_diff"] = _max_abs(
            expansion,
            row["factor_expansion_weights"],
        )
        metrics[f"bdf{bdf_order}_python_paths_max_diff"] = _max_abs(
            recurrence,
            expansion,
        )
        metrics[f"bdf{bdf_order}_wolfram_internal_residual"] = abs(
            float(row["max_recurrence_expansion_residual"])
        )
    metrics["max_diff"] = max(metrics.values())
    return metrics


def _validate_scalar_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    fixture = payload["scalar_fixture"]
    order = float(fixture["order"])
    tempering = float(fixture["tempering"])
    step = float(fixture["step"])
    samples = _array(fixture["samples"], ndim=1)
    elapsed = _array(fixture["elapsed"], ndim=1)
    if samples.shape != elapsed.shape:
        raise ValueError("scalar samples and elapsed grid have different shapes")
    metrics: dict[str, float] = {
        "elapsed_grid_max_diff": _max_abs(
            elapsed,
            step * np.arange(samples.size),
        )
    }
    rows = _rows_by_contract(fixture["rows"])
    for key, row in rows.items():
        definition, bdf_order = key
        direct = _tempered_direct(
            samples,
            order,
            tempering,
            step,
            bdf_order,
            definition,
        )
        conjugated = _tempered_conjugated(
            samples,
            order,
            tempering,
            step,
            bdf_order,
            definition,
        )
        prefix = f"{definition}_bdf{bdf_order}"
        metrics[f"{prefix}_direct_max_diff"] = _max_abs(
            direct,
            row["direct_values"],
        )
        metrics[f"{prefix}_conjugated_max_diff"] = _max_abs(
            conjugated,
            row["conjugated_values"],
        )
        metrics[f"{prefix}_python_identity_max_diff"] = _max_abs(
            direct,
            conjugated,
        )
        metrics[f"{prefix}_wolfram_identity_residual"] = abs(
            float(row["max_conjugation_residual"])
        )
    metrics["max_diff"] = max(metrics.values())
    return metrics


def _validate_lambda_zero_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    fixture = payload["lambda_zero_fixture"]
    order = float(fixture["order"])
    step = float(fixture["step"])
    samples = _array(fixture["samples"], ndim=1)
    rows = _rows_by_contract(fixture["rows"])
    metrics: dict[str, float] = {}
    for key, row in rows.items():
        definition, bdf_order = key
        expected = _tempered_direct(
            samples,
            order,
            0.0,
            step,
            bdf_order,
            definition,
        )
        prefix = f"{definition}_bdf{bdf_order}"
        metrics[f"{prefix}_tempered_max_diff"] = _max_abs(
            expected,
            row["tempered_values"],
        )
        metrics[f"{prefix}_untempered_max_diff"] = _max_abs(
            expected,
            row["untempered_values"],
        )
        metrics[f"{prefix}_wolfram_residual"] = abs(
            float(row["max_lambda_zero_residual"])
        )
    metrics["max_diff"] = max(metrics.values())
    return metrics


def _validate_q_one_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    fixture = payload["q_one_fixture"]
    order = float(fixture["order"])
    tempering = float(fixture["tempering"])
    step = float(fixture["step"])
    samples = _array(fixture["samples"], ndim=1)
    rows = _rows_by_contract(fixture["rows"])
    if order != 1.0:
        raise ValueError("q=1 fixture does not declare order 1")
    metrics: dict[str, float] = {}
    for key, row in rows.items():
        definition, bdf_order = key
        expected_weights = np.zeros(samples.size, dtype=np.float64)
        delta = _bdf_coefficients(bdf_order)
        expected_weights[: delta.size] = delta
        expected_values = _tempered_direct(
            samples,
            1.0,
            tempering,
            step,
            bdf_order,
            definition,
        )
        prefix = f"{definition}_bdf{bdf_order}"
        metrics[f"{prefix}_weights_max_diff"] = _max_abs(
            expected_weights,
            row["weights"],
        )
        metrics[f"{prefix}_declared_expected_weights_max_diff"] = _max_abs(
            expected_weights,
            row["expected_weights"],
        )
        metrics[f"{prefix}_values_max_diff"] = _max_abs(
            expected_values,
            row["direct_values"],
        )
        metrics[f"{prefix}_conjugated_bdf_max_diff"] = _max_abs(
            expected_values,
            row["expected_conjugated_bdf_values"],
        )
        metrics[f"{prefix}_wolfram_weight_residual"] = abs(
            float(row["max_weight_residual"])
        )
        metrics[f"{prefix}_wolfram_value_residual"] = abs(
            float(row["max_value_residual"])
        )
    metrics["max_diff"] = max(metrics.values())
    return metrics


def _validate_vector_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    fixture = payload["vector_fixture"]
    orders = _array(fixture["orders"], ndim=1)
    temperings = _array(fixture["temperings"], ndim=1)
    step = float(fixture["step"])
    samples = _array(fixture["samples"], ndim=2)
    if samples.shape[1] != orders.size or orders.shape != temperings.shape:
        raise ValueError("invalid componentwise vector fixture shape contract")
    rows = _rows_by_contract(fixture["rows"])
    metrics: dict[str, float] = {}
    for key, row in rows.items():
        definition, bdf_order = key
        expected = np.column_stack(
            [
                _tempered_direct(
                    samples[:, component],
                    float(orders[component]),
                    float(temperings[component]),
                    step,
                    bdf_order,
                    definition,
                )
                for component in range(samples.shape[1])
            ]
        )
        conjugated = np.column_stack(
            [
                _tempered_conjugated(
                    samples[:, component],
                    float(orders[component]),
                    float(temperings[component]),
                    step,
                    bdf_order,
                    definition,
                )
                for component in range(samples.shape[1])
            ]
        )
        prefix = f"{definition}_bdf{bdf_order}"
        metrics[f"{prefix}_values_max_diff"] = _max_abs(
            expected,
            row["values"],
        )
        metrics[f"{prefix}_scalar_components_max_diff"] = _max_abs(
            expected,
            row["scalar_component_values"],
        )
        metrics[f"{prefix}_conjugated_components_max_diff"] = _max_abs(
            conjugated,
            row["conjugated_component_values"],
        )
        metrics[f"{prefix}_python_identity_max_diff"] = _max_abs(
            expected,
            conjugated,
        )
        metrics[f"{prefix}_wolfram_component_residual"] = abs(
            float(row["max_componentwise_residual"])
        )
        metrics[f"{prefix}_wolfram_conjugation_residual"] = abs(
            float(row["max_conjugation_residual"])
        )
    metrics["max_diff"] = max(metrics.values())
    return metrics


def _validate_convergence_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    scalar = payload["scalar_fixture"]
    order = float(scalar["order"])
    tempering = float(scalar["tempering"])
    constant = float(scalar["constant"])
    power = int(scalar["power"])
    fixture = payload["convergence"]
    expected_grid = tuple(int(value) for value in fixture["resolution_grid"])
    if expected_grid != (64, 128, 256):
        raise ValueError(f"unexpected convergence grid {expected_grid!r}")
    rows = fixture["rows"]
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("convergence fixture must contain six rows")

    rl_exact = math.exp(-tempering) * (
        constant / math.gamma(1.0 - order)
        + math.gamma(power + 1.0) / math.gamma(power + 1.0 - order)
    )
    caputo_exact = (
        math.exp(-tempering)
        * math.gamma(power + 1.0)
        / math.gamma(power + 1.0 - order)
    )
    metrics: dict[str, float] = {}
    error_sequences: dict[tuple[str, int], list[float]] = {
        (definition, bdf_order): []
        for definition in EXPECTED_DEFINITIONS
        for bdf_order in EXPECTED_BDF_ORDERS
    }
    for row in rows:
        bdf_order = int(row["bdf_order"])
        resolution = int(row["n_steps"])
        step = 1.0 / resolution
        elapsed = step * np.arange(resolution + 1, dtype=np.float64)
        samples = np.exp(-tempering * elapsed) * (constant + elapsed**power)
        rl_endpoint = _tempered_direct(
            samples,
            order,
            tempering,
            step,
            bdf_order,
            "tempered_riemann_liouville",
        )[-1]
        caputo_endpoint = _tempered_direct(
            samples,
            order,
            tempering,
            step,
            bdf_order,
            "tempered_caputo",
        )[-1]
        prefix = f"bdf{bdf_order}_n{resolution}"
        metrics[f"{prefix}_step_max_diff"] = abs(float(row["step"]) - step)
        metrics[f"{prefix}_rl_endpoint_max_diff"] = abs(
            float(row["rl_endpoint"]) - rl_endpoint
        )
        metrics[f"{prefix}_caputo_endpoint_max_diff"] = abs(
            float(row["caputo_endpoint"]) - caputo_endpoint
        )
        metrics[f"{prefix}_rl_analytic_max_diff"] = abs(
            float(row["rl_endpoint_analytic"]) - rl_exact
        )
        metrics[f"{prefix}_caputo_analytic_max_diff"] = abs(
            float(row["caputo_endpoint_analytic"]) - caputo_exact
        )
        rl_error = abs(rl_endpoint - rl_exact)
        caputo_error = abs(caputo_endpoint - caputo_exact)
        metrics[f"{prefix}_rl_error_field_max_diff"] = abs(
            float(row["rl_endpoint_abs_error"]) - rl_error
        )
        metrics[f"{prefix}_caputo_error_field_max_diff"] = abs(
            float(row["caputo_endpoint_abs_error"]) - caputo_error
        )
        error_sequences[("tempered_riemann_liouville", bdf_order)].append(
            rl_error
        )
        error_sequences[("tempered_caputo", bdf_order)].append(caputo_error)
    monotone = all(
        sequence[0] > sequence[1] > sequence[2]
        for sequence in error_sequences.values()
    )
    metrics["max_diff"] = max(metrics.values(), default=0.0)
    return {
        "metrics": metrics,
        "endpoint_errors_monotonically_decrease": monotone,
        "endpoint_error_sequences": {
            f"{definition}_bdf{bdf_order}": sequence
            for (definition, bdf_order), sequence in error_sequences.items()
        },
    }


def validate_wolfram_summary(
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    tolerance: float = INDEPENDENT_TOLERANCE,
) -> dict[str, Any]:
    """Validate the Wolfram artifact without importing the HAFO package."""

    summary_path = Path(summary_path).resolve()
    payload = _load(summary_path)
    if payload.get("system_id") != SYSTEM_ID:
        raise ValueError(
            f"unexpected system_id {payload.get('system_id')!r}; "
            f"expected {SYSTEM_ID!r}"
        )

    source = payload.get("source", {})
    source_anchors_match = all(
        source.get(key) == expected for key, expected in SOURCE_ANCHORS.items()
    )
    independence_flags_match = bool(
        source.get("hafo_source_read") is False
        and source.get("hafo_formula_imported") is False
        and source.get("report_input_used") is False
        and source.get("built_in_series_used") is False
    )
    conventions = payload.get("conventions", {})
    convention_contract_match = bool(
        conventions.get("working_precision") == 80
        and conventions.get("elapsed_coordinate") == "tau=t-a"
        and conventions.get("bdf1_delta") == "1-z"
        and conventions.get("bdf2_delta") == "3/2-2*z+z^2/2"
        and conventions.get("starting_corrections") == "none_implemented"
        and conventions.get("no_minus_lambda_power_x") is True
        and conventions.get("tempered_generating_function")
        == "delta(exp(-lambda*h)*z)^q/h^q"
        and "not evaluated or identified"
        in str(conventions.get("shifted_symbol_family_not_evaluated", ""))
        and "component" in str(conventions.get("vector_semantics", ""))
    )
    wolfram_tests = payload.get("tests", ())
    wolfram_tests_pass = bool(
        payload.get("passed") is True
        and len(wolfram_tests) >= 18
        and all(test.get("passed") is True for test in wolfram_tests)
    )
    evidence_boundary = str(payload.get("evidence_boundary", ""))
    evidence_boundary_match = bool(
        "no stability or convergence theorem" in evidence_boundary
        and "no arbitrary nonlinear FDE certification" in evidence_boundary
        and "no evidence of chaos, attraction, or hiddenness" in evidence_boundary
    )

    weight_metrics = _validate_weights(payload)
    scalar_metrics = _validate_scalar_fixture(payload)
    lambda_zero_metrics = _validate_lambda_zero_fixture(payload)
    q_one_metrics = _validate_q_one_fixture(payload)
    vector_metrics = _validate_vector_fixture(payload)
    convergence = _validate_convergence_fixture(payload)
    numeric_max_diff = max(
        weight_metrics["max_diff"],
        scalar_metrics["max_diff"],
        lambda_zero_metrics["max_diff"],
        q_one_metrics["max_diff"],
        vector_metrics["max_diff"],
        convergence["metrics"]["max_diff"],
    )
    passed = bool(
        source_anchors_match
        and independence_flags_match
        and convention_contract_match
        and wolfram_tests_pass
        and evidence_boundary_match
        and convergence["endpoint_errors_monotonically_decrease"]
        and numeric_max_diff <= float(tolerance)
    )
    return {
        "validation_scope": payload.get("validation_scope"),
        "evidence_boundary": evidence_boundary,
        "summary_path": str(summary_path),
        "tolerance": float(tolerance),
        "source_anchors_match": source_anchors_match,
        "independence_flags_match": independence_flags_match,
        "convention_contract_match": convention_contract_match,
        "wolfram_tests_pass": wolfram_tests_pass,
        "evidence_boundary_match": evidence_boundary_match,
        "weight_metrics": weight_metrics,
        "scalar_metrics": scalar_metrics,
        "lambda_zero_metrics": lambda_zero_metrics,
        "q_one_metrics": q_one_metrics,
        "vector_metrics": vector_metrics,
        "convergence": convergence,
        "numeric_max_diff": numeric_max_diff,
        "passed": passed,
    }


@lru_cache(maxsize=1)
def _load_public_api() -> tuple[ModuleType, str, Callable[..., Any]]:
    try:
        module = importlib.import_module(PUBLIC_API_MODULE)
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        raise PublicTemperedCQAPIUnavailable(
            f"public tempered-CQ module {PUBLIC_API_MODULE!r} is unavailable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    for name in PUBLIC_CORE_CANDIDATES:
        function = getattr(module, name, None)
        if callable(function):
            return module, name, function
    raise PublicTemperedCQAPIUnavailable(
        f"public module {PUBLIC_API_MODULE!r} exposes none of "
        f"{PUBLIC_CORE_CANDIDATES!r}"
    )


def public_api_status() -> tuple[bool, str]:
    """Report optional public-core availability; independent parsing skips it."""

    try:
        _module, name, _function = _load_public_api()
    except PublicTemperedCQAPIUnavailable as exc:
        return False, str(exc)
    return True, f"{PUBLIC_API_MODULE}.{name}"


def _initial_semantics(module: ModuleType, definition: str) -> str:
    if definition == "tempered_riemann_liouville":
        names = (
            "TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION",
            "RL_OPERATOR_ONLY_INITIAL_CONDITION",
        )
        fallback = "operator_only_no_ivp"
    else:
        names = (
            "TEMPERED_CAPUTO_INITIAL_CONDITION",
            "TEMPERED_CAPUTO_SHIFTED_INITIAL_CONDITION",
            "CAPUTO_SHIFTED_INITIAL_CONDITION",
        )
        fallback = "point_value_shift_x_minus_x0"
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, str):
            return value
    return fallback


def _invoke_public_core(
    module: ModuleType,
    function: Callable[..., Any],
    *,
    samples: np.ndarray,
    orders: float | np.ndarray,
    temperings: float | np.ndarray,
    step: float,
    bdf_order: int,
    definition: str,
) -> Any:
    signature = inspect.signature(function)
    parameters = signature.parameters
    keyword_values: dict[str, Any] = {}

    aliases = (
        (("samples", "values", "data"), samples),
        (("orders", "order", "q"), orders),
        (
            (
                "temperings",
                "tempering",
                "lambda_values",
                "lambdas",
                "lambda_",
            ),
            temperings,
        ),
    )
    for candidates, value in aliases:
        name = next((candidate for candidate in candidates if candidate in parameters), None)
        if name is None:
            raise TypeError(
                f"public tempered-CQ signature {signature} has no parameter in "
                f"{candidates!r}"
            )
        keyword_values[name] = value

    optional_values = {
        "bdf_order": bdf_order,
        "definition": definition,
        "step": step,
        "lower_terminal": 0.0,
        "initial_condition_semantics": _initial_semantics(module, definition),
        "backend": "python",
    }
    for name, value in optional_values.items():
        if name in parameters:
            keyword_values[name] = value
    return function(**keyword_values)


def _extract_public_values(result: Any) -> np.ndarray:
    if isinstance(result, np.ndarray):
        return _array(result)
    if isinstance(result, dict):
        for name in ("values", "derivative", "result"):
            if name in result:
                return _array(result[name])
    for name in ("values", "derivative", "result"):
        if hasattr(result, name):
            return _array(getattr(result, name))
    raise TypeError("public tempered-CQ result exposes no sampled values")


def _compare_public_core(payload: dict[str, Any]) -> dict[str, Any]:
    module, public_name, function = _load_public_api()
    metrics: dict[str, float] = {}

    scalar = payload["scalar_fixture"]
    scalar_samples = _array(scalar["samples"], ndim=1)
    scalar_rows = _rows_by_contract(scalar["rows"])
    for (definition, bdf_order), row in scalar_rows.items():
        result = _invoke_public_core(
            module,
            function,
            samples=scalar_samples,
            orders=float(scalar["order"]),
            temperings=float(scalar["tempering"]),
            step=float(scalar["step"]),
            bdf_order=bdf_order,
            definition=definition,
        )
        metrics[f"scalar_{definition}_bdf{bdf_order}_max_diff"] = _max_abs(
            _extract_public_values(result),
            row["direct_values"],
        )

    lambda_zero = payload["lambda_zero_fixture"]
    lambda_zero_samples = _array(lambda_zero["samples"], ndim=1)
    lambda_zero_rows = _rows_by_contract(lambda_zero["rows"])
    for (definition, bdf_order), row in lambda_zero_rows.items():
        result = _invoke_public_core(
            module,
            function,
            samples=lambda_zero_samples,
            orders=float(lambda_zero["order"]),
            temperings=0.0,
            step=float(lambda_zero["step"]),
            bdf_order=bdf_order,
            definition=definition,
        )
        metrics[f"lambda_zero_{definition}_bdf{bdf_order}_max_diff"] = _max_abs(
            _extract_public_values(result),
            row["tempered_values"],
        )

    q_one = payload["q_one_fixture"]
    q_one_samples = _array(q_one["samples"], ndim=1)
    q_one_rows = _rows_by_contract(q_one["rows"])
    for (definition, bdf_order), row in q_one_rows.items():
        result = _invoke_public_core(
            module,
            function,
            samples=q_one_samples,
            orders=1.0,
            temperings=float(q_one["tempering"]),
            step=float(q_one["step"]),
            bdf_order=bdf_order,
            definition=definition,
        )
        metrics[f"q_one_{definition}_bdf{bdf_order}_max_diff"] = _max_abs(
            _extract_public_values(result),
            row["direct_values"],
        )

    vector = payload["vector_fixture"]
    vector_samples = _array(vector["samples"], ndim=2)
    vector_rows = _rows_by_contract(vector["rows"])
    for (definition, bdf_order), row in vector_rows.items():
        result = _invoke_public_core(
            module,
            function,
            samples=vector_samples,
            orders=_array(vector["orders"], ndim=1),
            temperings=_array(vector["temperings"], ndim=1),
            step=float(vector["step"]),
            bdf_order=bdf_order,
            definition=definition,
        )
        metrics[f"vector_{definition}_bdf{bdf_order}_max_diff"] = _max_abs(
            _extract_public_values(result),
            row["values"],
        )

    maximum = max(metrics.values(), default=math.inf)
    return {
        "public_api": f"{PUBLIC_API_MODULE}.{public_name}",
        "backend_requested": "python",
        "tolerance": CORE_TOLERANCE,
        "metrics": metrics,
        "max_diff": maximum,
        "passed": maximum <= CORE_TOLERANCE,
    }


def compare_wolfram_summary(
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    tolerance: float = INDEPENDENT_TOLERANCE,
    require_core: bool = False,
) -> dict[str, Any]:
    """Validate independently and optionally compare the public HAFO core."""

    independent = validate_wolfram_summary(summary_path, tolerance=tolerance)
    available, diagnostic = public_api_status()
    cross_result: dict[str, Any] | None = None
    cross_error: str | None = None
    if available:
        try:
            cross_result = _compare_public_core(_load(Path(summary_path)))
        except Exception as exc:  # explicit diagnostic for an evolving public API
            cross_error = f"{type(exc).__name__}: {exc}"

    cross_performed = cross_result is not None
    cross_passed = bool(cross_result and cross_result["passed"])
    passed = bool(
        independent["passed"]
        and (not cross_performed or cross_passed)
        and (not require_core or (cross_performed and cross_passed))
    )
    return {
        "validation_scope": (
            "independent_Wolfram_tempered_CQ_with_optional_public_HAFO_cross_check"
        ),
        "evidence_boundary": independent["evidence_boundary"],
        "independent": independent,
        "require_core": bool(require_core),
        "public_core_available": available,
        "public_core_diagnostic": diagnostic,
        "cross_implementation_performed": cross_performed,
        "cross_implementation_passed": cross_passed,
        "cross_implementation_result": cross_result,
        "cross_implementation_error": cross_error,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--tolerance", type=float, default=INDEPENDENT_TOLERANCE)
    parser.add_argument("--require-core", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the comparison JSON; use a temporary path live.",
    )
    args = parser.parse_args()
    result = compare_wolfram_summary(
        args.summary,
        tolerance=args.tolerance,
        require_core=args.require_core,
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
