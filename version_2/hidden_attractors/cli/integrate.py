"""Reproducible full-history L1/PECE comparisons on registered flow systems."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
from typing import Sequence

import numpy as np

from .._run_output import claim_output_directory, complete_output_claim
from .._time_grid import exact_fixed_step_count
from ..integrations.selector import integrate, validate_integrator_compatibility
from ..systems import get_system


def compare_integrators(argv: Sequence[str] | None = None) -> None:
    """Write all trajectories and a uniform diagnostic/provenance summary."""
    parser = argparse.ArgumentParser(prog="hidden-attractors integrate compare", allow_abbrev=False)
    parser.add_argument("--system", required=True, help="Registered continuous-time system")
    parser.add_argument("--parameters", default="{}", help="JSON object overriding system parameters")
    parser.add_argument("--x0", nargs="+", type=float, required=True)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--h", nargs="+", type=float, required=True, help="One or more fixed step sizes")
    parser.add_argument("--t-final", type=float, required=True)
    parser.add_argument("--methods", nargs="+", choices=("l1_implicit", "abm_stable"), default=["l1_implicit", "abm_stable"])
    parser.add_argument("--divergence-norm", type=float, default=120.0)
    parser.add_argument("--newton-tol", type=float, default=2e-14)
    parser.add_argument("--newton-maxiter", type=int, default=20)
    parser.add_argument("--fd-relative-step", type=float, default=float(np.sqrt(np.finfo(float).eps)))
    parser.add_argument("--finite-difference-jacobian", action="store_true", help="Use central differences even if an analytic Jacobian is registered")
    parser.add_argument("--no-numba", action="store_true", help="Use the pure Python compensated history reduction")
    parser.add_argument("--max-history-terms", type=int, default=50_000_000,
                        help="Per-run work budget for the O(N^2) history sum; raise explicitly for a longer experiment")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty output directory; results cannot be overwritten")
    args = parser.parse_args(argv)
    try:
        system = get_system(args.system)
    except KeyError as exc:
        parser.error(str(exc))
    if not system.is_continuous:
        parser.error("Caputo integration requires a continuous-time flow")
    try:
        overrides = json.loads(args.parameters)
        if not isinstance(overrides, dict):
            raise ValueError("--parameters must be a JSON object")
        unknown_parameters = set(overrides) - set(system.parameters)
        if unknown_parameters:
            raise ValueError(f"Unknown parameters for {system.name}: {sorted(unknown_parameters)}")
        parameters = {**system.parameters, **overrides}
        json.dumps(parameters, allow_nan=False)
        anchor = np.asarray(args.x0, dtype=float)
        if anchor.shape != (system.dimension,) or not np.all(np.isfinite(anchor)):
            raise ValueError(f"--x0 must contain {system.dimension} finite values")
        if len(set(args.h)) != len(args.h) or len(set(args.methods)) != len(args.methods):
            raise ValueError("duplicate step sizes or methods are not allowed")
        for method in args.methods:
            validate_integrator_compatibility(method, args.q)
        for step in args.h:
            steps = exact_fixed_step_count(step, args.t_final, caller="integrate compare")
            if args.max_history_terms < 1:
                raise ValueError("--max-history-terms must be positive")
            estimated = steps * steps if "abm_stable" in args.methods else steps * (steps - 1) // 2
            if estimated > args.max_history_terms:
                raise ValueError(f"step {step} requires {estimated} history terms per run, exceeding --max-history-terms={args.max_history_terms}; raise the budget explicitly")
        for name in ("divergence_norm", "newton_tol", "fd_relative_step"):
            value = getattr(args, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive and finite")
        if args.newton_maxiter < 1:
            raise ValueError("--newton-maxiter must be positive")
        # Preserve the solver's first-event contract during CLI preflight:
        # an initial threshold crossing does not require evaluating f(x0).
        if np.linalg.norm(anchor) <= args.divergence_norm:
            initial = system.evaluate(anchor, parameters)
            if not np.all(np.isfinite(initial)):
                raise ValueError("initial system RHS must be finite")
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    # The claim precedes every integration, including JIT compilation.
    claim = claim_output_directory(args.output_dir, run_kind="caputo_integrator_comparison_v1")
    output = claim.path
    rhs = lambda _t, x: system.evaluate(x, parameters)
    analytic = system.jacobian is not None and not args.finite_difference_jacobian
    jacobian = (lambda _t, x: system.jacobian_matrix(x, parameters)) if analytic else None
    options = dict(newton_tol=args.newton_tol, newton_maxiter=args.newton_maxiter,
                   fd_relative_step=args.fd_relative_step, use_numba=not args.no_numba)
    source_paths = [Path(__file__), Path(__file__).parents[1] / "integrations" / "caputo_diagnostics.py",
                    Path(__file__).parents[1] / "integrations" / "selector.py",
                    Path(__file__).parents[1] / "integrations" / "general.py"]
    module_file = inspect.getsourcefile(system.rhs)
    if module_file:
        source_paths.append(Path(module_file))
    sources = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for path in dict.fromkeys(source_paths)]
    contract = dict(schema_version="caputo_comparison_v1", run_id=claim.run_id,
                    system=system.name, parameters=parameters, initial_state=anchor.tolist(),
                    order=args.q, steps=args.h, requested_horizon=args.t_final,
                    methods=args.methods, solver_options=options, jacobian="analytic" if analytic else "central_finite_difference",
                    maximum_history_terms_per_run=args.max_history_terms,
                    caputo_lower_limit=0.0, memory_mode="full", history_restarted=False,
                    divergence_norm=args.divergence_norm, python_version=platform.python_version(),
                    numpy_version=np.__version__, source_files=sources,
                    interpretation="Finite-horizon numerical sensitivity; no automatic attractor, chaos, or asymptotic basin claim.")
    (output / "contract.json").write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    runs = []
    trajectories = {}
    for step_index, step in enumerate(args.h):
        for method in args.methods:
            metadata = {}
            times, states, status = integrate(
                rhs, anchor, args.q, step, args.t_final, integrator=method,
                jacobian=jacobian, solver_options=options, diagnostics=metadata,
                divergence_norm=args.divergence_norm, memory_mode="full")
            name = f"h{step_index:02d}_{method}.npz"
            np.savez_compressed(output / name, t=times, x=states)
            metadata.update(trajectory=name, trajectory_sha256=hashlib.sha256((output / name).read_bytes()).hexdigest())
            runs.append(metadata)
            trajectories[step_index, method] = (times, states)
    comparisons = []
    if len(args.methods) == 2:
        for i, step in enumerate(args.h):
            t1, x1 = trajectories[i, args.methods[0]]
            t2, x2 = trajectories[i, args.methods[1]]
            shared = min(len(t1), len(t2))
            differences = np.linalg.norm(x1[:shared] - x2[:shared], axis=1)
            comparisons.append(dict(step=step, methods=args.methods, shared_rows=shared,
                                    shared_final_time=float(t1[shared - 1]),
                                    max_state_l2_difference=float(np.max(differences)),
                                    last_shared_state_l2_difference=float(differences[-1]),
                                    both_reached_requested_horizon=all(r["full_horizon_reached"] for r in runs if r["step"] == step)))
    summary = dict(**contract, runs=runs, method_comparisons=comparisons,
                   all_runs_completed=all(r["full_horizon_reached"] for r in runs))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    complete_output_claim(claim)
    print(json.dumps({"output_directory": str(output), "runs": len(runs), "all_runs_completed": summary["all_runs_completed"]}))


__all__ = ["compare_integrators"]
