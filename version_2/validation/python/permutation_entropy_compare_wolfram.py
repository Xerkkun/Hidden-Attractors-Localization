"""Compare HAFO's public Bandt--Pompe API with independent Wolfram data.

The Wolfram case builds four exact scalar fixtures, constructs chronological
forward delay windows, ranks ordinal patterns with a zero-based lexicographic
Lehmer code, applies the declared ``stable_index`` and ``omit`` tie policies,
and computes plug-in Shannon entropy in base two.  It does not read HAFO
source or generated reports.

This comparator reaches HAFO only through the public
``ordinal_pattern_distribution``, ``permutation_entropy_from_distribution``,
and ``permutation_entropy`` APIs, forcing their transparent Python backend.
Passing is finite-sequence combinatorial and numerical consistency evidence
only.  It is not an entropy-rate or Kolmogorov--Sinai entropy validation, nor
proof of asymptotic behavior, chaos, attraction, or hiddenness.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from hidden_attractors.analysis.permutation_entropy import (
    ordinal_pattern_distribution,
    permutation_entropy,
    permutation_entropy_from_distribution,
)


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_ID = "permutation_entropy"
DEFAULT_SUMMARY = (
    ROOT
    / "validation"
    / "outputs"
    / "wolfram"
    / SYSTEM_ID
    / f"{SYSTEM_ID}_validation_summary.json"
)
NUMERIC_TOLERANCE = 5.0e-14

SOURCE_ANCHORS = {
    "bandt_pompe_doi": "10.1103/PhysRevLett.88.174102",
}
EXPECTED_FIXTURE_NAMES = (
    "no_ties_tau1",
    "no_ties_tau2",
    "ties_stable_index",
    "ties_omit",
)


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


def _compare_fixture(
    name: str,
    fixture: dict[str, Any],
) -> dict[str, Any]:
    series = np.asarray(fixture["series"], dtype=np.float64)
    embedding_dimension = int(fixture["embedding_dimension"])
    delay = int(fixture["delay"])
    tie_policy = str(fixture["tie_policy"])

    distribution = ordinal_pattern_distribution(
        series,
        embedding_dimension=embedding_dimension,
        delay=delay,
        tie_policy=tie_policy,
        backend="python",
        fallback=False,
        sampling=f"declared exact Wolfram fixture: {name}",
        projection="identity scalar observation",
    )
    result = permutation_entropy_from_distribution(distribution, log_base=2.0)
    direct = permutation_entropy(
        series,
        embedding_dimension=embedding_dimension,
        delay=delay,
        tie_policy=tie_policy,
        log_base=2.0,
        backend="python",
        fallback=False,
        sampling=f"declared exact Wolfram fixture: {name}",
        projection="identity scalar observation",
    )

    wolfram_counts = np.asarray(fixture["counts"], dtype=np.int64)
    wolfram_probabilities = np.asarray(
        fixture["probabilities"],
        dtype=np.float64,
    )
    numeric_metrics = {
        "counts_max_diff": _max_abs(distribution.counts, wolfram_counts),
        "probabilities_max_diff": _max_abs(
            distribution.probabilities,
            wolfram_probabilities,
        ),
        "entropy_base2_abs_diff": abs(
            float(result.entropy) - float(fixture["entropy_base2"])
        ),
        "normalized_entropy_abs_diff": abs(
            float(result.normalized_entropy)
            - float(fixture["normalized_entropy"])
        ),
        "maximum_entropy_abs_diff": abs(
            float(result.maximum_entropy)
            - float(np.log2(math.factorial(embedding_dimension)))
        ),
        "direct_counts_max_diff": _max_abs(
            direct.distribution.counts,
            wolfram_counts,
        ),
        "direct_entropy_abs_diff": abs(
            float(direct.entropy) - float(fixture["entropy_base2"])
        ),
        "direct_normalized_entropy_abs_diff": abs(
            float(direct.normalized_entropy)
            - float(fixture["normalized_entropy"])
        ),
    }
    numeric_metrics["max_diff"] = max(numeric_metrics.values())

    expected_total = int(fixture["candidate_window_count"])
    expected_valid = int(fixture["accepted_window_count"])
    expected_omitted = int(fixture["omitted_tie_count"])
    expected_tied = int(sum(bool(value) for value in fixture["tie_mask"]))
    contract_match = bool(
        distribution.embedding_dimension == embedding_dimension
        and distribution.delay == delay
        and distribution.tie_policy == tie_policy
        and distribution.total_windows == expected_total
        and distribution.valid_windows == expected_valid
        and distribution.tied_windows == expected_tied
        and distribution.possible_patterns
        == math.factorial(embedding_dimension)
        and distribution.observed_patterns
        == int(np.count_nonzero(wolfram_counts))
        and int(np.sum(distribution.counts)) == expected_valid
        and expected_total - expected_valid == expected_omitted
        and distribution.backend == "python"
        and result.distribution is distribution
        and result.log_base == 2.0
        and result.estimator == "plugin"
        and result.normalization == "log_factorial_outcome_space"
        and direct.distribution.backend == "python"
    )
    return {
        "fixture": name,
        "tie_policy": tie_policy,
        "embedding_dimension": embedding_dimension,
        "delay": delay,
        "python_backend": distribution.backend,
        "total_windows": distribution.total_windows,
        "valid_windows": distribution.valid_windows,
        "tied_windows": distribution.tied_windows,
        "contract_match": contract_match,
        "numeric_metrics": numeric_metrics,
        "passed": bool(contract_match),
    }


def compare_wolfram_summary(
    summary_path: Path = DEFAULT_SUMMARY,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> dict[str, Any]:
    """Recompute all exact fixtures through HAFO's public Python API."""

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

    fixture_results = {
        name: _compare_fixture(name, fixtures[name])
        for name in EXPECTED_FIXTURE_NAMES
    }
    cross_implementation_max_diff = max(
        result["numeric_metrics"]["max_diff"]
        for result in fixture_results.values()
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
        conventions.get("window_order")
        == "chronological_forward_x[t+j*tau]"
        and conventions.get("rank_encoding")
        == "zero_based_lexicographic_Lehmer"
        and conventions.get("logarithm_base") == 2
        and conventions.get("normalization") == "H/log2(m!)"
        and conventions.get("tie_policies") == ["stable_index", "omit"]
    )
    fixture_contracts_match = all(
        result["contract_match"] for result in fixture_results.values()
    )
    passed = bool(
        source_anchors_match
        and independence_flags_match
        and wolfram_tests_pass
        and convention_contract_match
        and fixture_contracts_match
        and cross_implementation_max_diff <= float(tolerance)
    )
    return {
        "validation_scope": (
            "independent_Wolfram_to_HAFO_Bandt_Pompe_finite_consistency"
        ),
        "evidence_boundary": payload["evidence_boundary"],
        "summary_path": str(summary_path),
        "tolerance": float(tolerance),
        "source_anchors_match": source_anchors_match,
        "independence_flags_match": independence_flags_match,
        "wolfram_tests_pass": wolfram_tests_pass,
        "convention_contract_match": convention_contract_match,
        "fixture_contracts_match": fixture_contracts_match,
        "fixture_results": fixture_results,
        "cross_implementation_max_diff": cross_implementation_max_diff,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--tolerance", type=float, default=NUMERIC_TOLERANCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON destination for the Python comparison summary.",
    )
    args = parser.parse_args()
    result = compare_wolfram_summary(args.summary, tolerance=args.tolerance)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
