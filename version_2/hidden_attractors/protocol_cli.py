"""Command-line interface for the official Caputo hidden-attractor protocol.

The CLI records and validates stage summaries. Numerical engines and legacy
adapters may calculate stage payloads, but promotion into official evidence
must pass through this stage vocabulary and JSON envelope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .io import write_json
from .reproducibility import (
    CONSERVATIVE_HIDDENNESS_LABEL,
    collect_run_metadata,
    extract_run_metadata,
    metadata_to_jsonable,
    validate_run_metadata,
    write_run_metadata,
)
from .verification.candidate_gate import normalize_hiddenness_label
from .workflows.protocol import (
    OFFICIAL_STAGE_ORDER,
    ContinuationPlan,
    HiddennessTestResult,
    NumericalContract,
    SoftPrecheckResult,
    StageEnvelope,
)


COMMAND_TO_STAGE = {
    "generate-seeds": "seed_generation",
    "soft-precheck": "soft_precheck",
    "continue": "continuation",
    "filter-survivors": "post_continuation_filter",
    "build-reference": "dynamic_reference",
    "robustness": "robustness",
    "hiddenness": "hiddenness_tests",
    "diagnostics": "diagnostics",
}


def _read_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object.")
    return value


def _numerical_contract(payload: dict[str, Any]) -> NumericalContract:
    source = payload.get("numerical_contract", payload)
    if not isinstance(source, dict):
        raise ValueError("numerical contract must be an object.")
    accepted = NumericalContract.__dataclass_fields__
    values = {key: value for key, value in source.items() if key in accepted}
    for key in ("q", "h", "t_final"):
        if key not in values:
            raise ValueError(f"numerical contract requires '{key}'.")
    if isinstance(values.get("hiddenness_radii"), list):
        values["hiddenness_radii"] = tuple(values["hiddenness_radii"])
    contract = NumericalContract(**values)
    errors = contract.validate()
    if errors:
        raise ValueError("; ".join(errors))
    return contract


def _backend_name(backend: str) -> str:
    value = backend.lower()
    if "python" in value:
        return "python"
    if "native" in value or value.endswith("_c") or value == "c":
        return "native"
    return "unknown"


def _run_metadata(
    args: argparse.Namespace,
    contract: NumericalContract,
    data: dict[str, Any],
) -> dict[str, Any]:
    supplied = extract_run_metadata(data)
    if supplied is not None:
        return write_run_metadata(args.output.parent / "run_metadata.json", supplied)
    continuation = data.get("continuation", {})
    if args.command == "continue" and not continuation:
        eta_path = [float(value) for value in args.lambda_values.split(",") if value.strip()]
        continuation = {
            "used": True,
            "eta_path": eta_path,
            "continuation_mode": "fractional" if contract.q < 1.0 else "integer",
            "memory_window_propagated": contract.memory_policy == "full_history",
            "final_eta": eta_path[-1] if eta_path else None,
        }
    metadata = collect_run_metadata(
        run_id=str(data.get("run_id", args.output.stem)),
        workflow=f"protocol_cli:{args.command}",
        system=args.system,
        q=contract.q,
        h=contract.h,
        t_final=contract.t_final,
        t_burn=contract.effective_transient,
        memory_mode=contract.memory_policy,
        memory_window_time=contract.memory_length,
        is_full_caputo=contract.memory_policy == "full_history",
        integrator_name=contract.backend,
        integrator_backend=_backend_name(contract.backend),
        caputo=True,
        parameters=data.get("parameters", {}),
        lure=data.get("lure"),
        seed=data.get("seed"),
        random_seed=contract.random_seed,
        random_seed_policy=contract.random_seed_policy if contract.random_seed is not None else "not_applicable",
        provenance=data.get("provenance", {}),
        continuation=continuation or None,
        tolerances=contract.tolerances or None,
    )
    return write_run_metadata(args.output.parent / "run_metadata.json", metadata)


def _stage_specific_payload(
    args: argparse.Namespace,
    data: dict[str, Any],
    run_metadata: dict[str, Any],
    contract: NumericalContract,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    inputs = dict(data.get("inputs", {})) if isinstance(data.get("inputs", {}), dict) else {}
    metrics = dict(data.get("metrics", {})) if isinstance(data.get("metrics", {}), dict) else {}
    verdict = data.get("verdict")
    if args.command == "generate-seeds":
        verdict = verdict or "seed_only"
    elif args.command == "soft-precheck":
        label = str(data.get("label", args.precheck_label or "pre_continuation_admissible"))
        if label == "pre_continuation_periodic":
            result = SoftPrecheckResult.periodic(args.candidate_id or "")
        else:
            result = SoftPrecheckResult(
                candidate_id=args.candidate_id or "",
                label=label,
                admissible_for_continuation=bool(data.get("admissible_for_continuation", True)),
                finite_trajectory=bool(data.get("finite_trajectory", True)),
                immediate_numerical_failure=bool(data.get("immediate_numerical_failure", False)),
                catastrophic_divergence=bool(data.get("catastrophic_divergence", False)),
                exact_duplicate=bool(data.get("exact_duplicate", False)),
            )
        errors = result.validate()
        if errors:
            raise ValueError("; ".join(errors))
        inputs["soft_precheck"] = result.__dict__
        verdict = result.label
    elif args.command == "continue":
        values = tuple(float(value) for value in args.lambda_values.split(",") if value.strip())
        plan = ContinuationPlan(values, {"public_parameter": "lambda", "internal_parameter": args.internal_parameter})
        errors = plan.validate()
        if errors:
            raise ValueError("; ".join(errors))
        inputs["continuation_plan"] = {
            "lambda_values": list(plan.lambda_values),
            "mapping": dict(plan.mapping),
        }
    elif args.command == "hiddenness" and normalize_hiddenness_label(str(verdict)) == "hidden_under_tested_neighborhoods":
        evidence = data.get("hiddenness_test_result")
        if not isinstance(evidence, dict):
            evidence = {}
        result = HiddennessTestResult(
            candidate_id=args.candidate_id or "",
            tested_equilibria=tuple(evidence.get("tested_equilibria", ())),
            tested_radii=tuple(float(value) for value in evidence.get("tested_radii", ())),
            neighborhood_sampling_mode=str(evidence.get("neighborhood_sampling_mode", "")),
            target_contacts=int(evidence.get("target_contacts", 0)),
            numerical_failures=int(evidence.get("numerical_failures", 0)),
            basin_planes=tuple(evidence.get("basin_planes", ())),
            reference_was_robust=bool(evidence.get("reference_was_robust", False)),
            final_label="hidden_under_tested_neighborhoods",
            run_metadata=run_metadata,
            required_equilibria=tuple(evidence.get("required_equilibria", ())),
            required_radii=tuple(float(value) for value in evidence.get("required_radii", contract.hiddenness_radii)),
        )
        errors = result.validate()
        metrics["hiddenness_promotion_errors"] = errors
        inputs["hiddenness_test_result"] = {
            **evidence,
            "promotion_verdict": result.promotion_verdict,
        }
        verdict = result.promotion_verdict
    return inputs, metrics, str(verdict) if verdict is not None else None


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official Caputo hidden-attractor protocol stage interface.")
    parser.add_argument("command", choices=list(COMMAND_TO_STAGE))
    parser.add_argument("--contract", type=Path, required=True, help="JSON numerical contract or envelope containing numerical_contract.")
    parser.add_argument("--payload", type=Path, help="Optional JSON payload calculated by the numerical stage implementation.")
    parser.add_argument("--output", type=Path, required=True, help="Machine-readable official stage summary JSON.")
    parser.add_argument("--system", default="fractional_nonsmooth_chua")
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--status", default="recorded")
    parser.add_argument("--precheck-label", default="")
    parser.add_argument("--lambda-values", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--internal-parameter", default="lambda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import warnings
    warnings.warn(
        "Deprecated: use 'hidden-attractors protocol <command> ...'",
        DeprecationWarning,
        stacklevel=2
    )
    print("Deprecated: use 'hidden-attractors protocol <command> ...'")
    args = make_parser().parse_args(argv)
    contract = _numerical_contract(_read_object(args.contract))
    data = _read_object(args.payload)
    run_metadata = _run_metadata(args, contract, data)
    inputs, metrics, verdict = _stage_specific_payload(args, data, run_metadata, contract)
    state = data.get("state")
    if state == "hidden_verified" and verdict != "hiddenness_supported_under_tested_neighborhoods":
        state = "hidden_compatible"
    elif state == "hidden_verified":
        state = "hiddenness_supported_under_tested_neighborhoods"
    envelope = StageEnvelope(
        stage=COMMAND_TO_STAGE[args.command],
        status=args.status,
        candidate_id=args.candidate_id,
        system=args.system,
        numerical_contract=contract.to_dict(),
        inputs=inputs,
        outputs=data.get("outputs", {}) if isinstance(data.get("outputs", {}), dict) else {},
        metrics=metrics,
        verdict=verdict,
        files=data.get("files", {}) if isinstance(data.get("files", {}), dict) else {},
        provenance=data.get("provenance", {}) if isinstance(data.get("provenance", {}), dict) else {},
        run_metadata=run_metadata,
        metadata_validation_errors=validate_run_metadata(metadata_to_jsonable(run_metadata)),
        state=state,
        state_history=data.get("state_history", []),
        evidence=data.get("evidence", {}) if isinstance(data.get("evidence", {}), dict) else {},
        failed_requirements=data.get("failed_requirements", []),
        method_scope=data.get("method_scope", ""),
        warnings=data.get("warnings", []),
        literature_note=data.get("literature_note", ""),
    )
    errors = envelope.validate()
    if errors:
        raise ValueError("; ".join(errors))
    write_json(args.output, envelope.to_dict())
    print(f"stage={envelope.stage}")
    print(f"summary={args.output}")
    return 0


__all__ = ["COMMAND_TO_STAGE", "main", "make_parser"]


if __name__ == "__main__":
    raise SystemExit(main())
