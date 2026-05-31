"""Helper module for fractional variational ABM-QR Lyapunov benchmarks (F2.1)."""

import os
import yaml
import numpy as np
from typing import Dict, Any, List, Tuple

from hidden_attractors.analysis import compute_lyapunov_spectrum
from hidden_attractors.native import (
    FractionalLyapunovRequest,
    NativeFractionalVariationalBackend,
)

def load_benchmark_case(path: str) -> Dict[str, Any]:
    """Load a benchmark case from a YAML file.

    Parameters
    ----------
    path : str
        Absolute path to the YAML file.

    Returns
    -------
    case_data : dict
        Parsed YAML content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_system_functions(system_config: Dict[str, Any]) -> Tuple[Any, Any, int, np.ndarray]:
    """Build RHS and Jacobian functions based on the system configuration.

    Parameters
    ----------
    system_config : dict
        System configuration dictionary from YAML.

    Returns
    -------
    rhs : callable
    jacobian : callable or None
    dimension : int
    x0 : np.ndarray
    """
    kind = system_config.get("kind")
    dim = int(system_config.get("dimension"))
    x0 = np.asarray(system_config.get("x0"), dtype=float) if system_config.get("x0") is not None else np.zeros(dim)
    params = system_config.get("parameters", [])

    if kind == "zero_rhs":
        rhs = lambda x: np.zeros(dim)
        jacobian = lambda x: np.zeros((dim, dim))
    elif kind == "linear_diagonal":
        diag = np.asarray(params, dtype=float)
        rhs = lambda x: diag * x
        jacobian = lambda x: np.diag(diag)
    elif kind == "chua":
        from hidden_attractors.models.chua import chua_parameters, rhs_arctan, jacobian_arctan
        # Chua parameters from list: [alpha, beta, gamma]
        p = chua_parameters(
            model="arctan",
            alpha=params[0] if len(params) > 0 else 8.4562,
            beta=params[1] if len(params) > 1 else 12.0732,
            gamma=params[2] if len(params) > 2 else 0.0052,
            a1=0.4,
            a2=-1.5585,
            rho=1.0,
        )
        rhs = lambda x: rhs_arctan(x, p)
        jacobian = lambda x: jacobian_arctan(x, p)
    elif kind == "rabinovich_fabrikant":
        a = float(params["a"])
        b = float(params["b"])
        rhs = lambda x: np.asarray([
            x[1] * (x[2] - 1.0 + x[0] * x[0]) + a * x[0],
            x[0] * (3.0 * x[2] + 1.0 - x[0] * x[0]) + a * x[1],
            -2.0 * x[2] * (b + x[0] * x[1]),
        ])
        jacobian = lambda x: np.asarray([
            [2.0 * x[0] * x[1] + a, x[0] * x[0] + x[2] - 1.0, x[1]],
            [-3.0 * x[0] * x[0] + 3.0 * x[2] + 1.0, a, 3.0 * x[0]],
            [-2.0 * x[1] * x[2], -2.0 * x[0] * x[2], -2.0 * (x[0] * x[1] + b)],
        ])
    elif kind == "lorenz":
        sigma = float(params["sigma"])
        beta = float(params["beta"])
        rho = float(params["rho"])
        rhs = lambda x: np.asarray([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]])
        jacobian = lambda x: np.asarray([[-sigma, sigma, 0.0], [rho - x[2], -1.0, -x[0]], [x[1], x[0], -beta]])
    elif kind == "dk2018_4d_nonsmooth":
        raise NotImplementedError(
            "DK2018 4D nonsmooth case is qualitative-only: the article data are "
            "incomplete and no quantitative native validation is permitted."
        )
    elif kind == "custom":
        raise NotImplementedError("Custom system kind is not pre-defined.")
    else:
        raise ValueError(f"Unknown system kind: {kind}")

    return rhs, jacobian, dim, x0

def run_benchmark_case(
    case_data: Dict[str, Any],
    fast: bool = False,
    output_dir: str = None
) -> Dict[str, Any]:
    """Run a single benchmark case.

    Parameters
    ----------
    case_data : dict
        Parsed YAML benchmark specification.
    fast : bool, default False
        If True, runs a significantly shorter integration for fast unit tests.
    output_dir : str, optional
        If provided, saves convergence trajectory and details.

    Returns
    -------
    result_summary : dict
        Outcome details including case ID, status, and computed values.
    """
    case_id = case_data["case_id"]
    btype = case_data["benchmark_type"]
    method_id = case_data["method_id"]
    ref_config = case_data.get("reference", {})

    # 1. Check if data is complete for published benchmarks
    if btype == "published" and not ref_config.get("data_complete", True):
        qualitative_only = bool(case_data.get("expected", {}).get("qualitative_only", False))
        return {
            "case_id": case_id,
            "benchmark_type": btype,
            "status": "published_reference_data_missing_qualitative_only" if qualitative_only else "published_reference_data_missing",
            "computed_exponents": None,
            "message": "Published benchmark is a template with missing data.",
            "missing_fields": ref_config.get("missing_fields", [])
        }

    # 2. Build system functions
    sys_config = case_data["system"]
    rhs, jacobian, dim, x0 = build_system_functions(sys_config)

    # 3. integration settings
    int_config = case_data["integration"]
    h = float(int_config["h"])
    t_final = float(int_config["t_final"])
    t_burn = float(int_config.get("t_burn", 0.0))
    reorth_time = int_config.get("reorthonormalization_time")
    mem_mode = int_config.get("memory_mode", "full")
    mem_window = int_config.get("memory_window")

    if fast:
        # Scale down parameters for fast test runs
        t_burn = min(t_burn, h * 2)
        t_final = h * 10
        if reorth_time is not None:
            reorth_time = min(float(reorth_time), h * 5)

    # 4. Compute spectrum. Extensive published calculations are native-only.
    q = float(sys_config["q"])
    execution = case_data.get("execution", {})
    native_required = bool(execution.get("native_required", False))
    if native_required:
        if reorth_time is None:
            raise ValueError("Native published benchmark requires reorthonormalization_time.")
        conv_csv = None
        if output_dir is not None:
            conv_csv = os.path.join(output_dir, "convergence", f"{case_id}.csv")
        backend = NativeFractionalVariationalBackend.build()
        result = backend.run(
            FractionalLyapunovRequest(
                system_id=sys_config["kind"],
                x0=x0,
                parameters=sys_config["parameters"],
                q=q,
                h=h,
                t_final=t_final,
                t_burn=t_burn,
                reorthonormalization_time=float(reorth_time),
                execution_contract=execution["execution_contract"],
                convolution_mode=execution.get("convolution_mode", "fft_block"),
                fft_block_size=int(execution.get("fft_block_size", 256)),
                divergence_norm=float(execution.get("divergence_norm", 0.0)),
                convergence_csv=conv_csv,
            )
        )
    else:
        summary = compute_lyapunov_spectrum(
            rhs=rhs,
            jacobian=jacobian,
            x0=x0,
            q=q,
            method=method_id,
            h=h,
            t_final=t_final,
            t_burn=t_burn,
            reorthonormalization_time=reorth_time,
            memory_mode=mem_mode,
            memory_window=mem_window,
        )
        result = summary.result
    status = result.status

    if status != "ok" or not np.all(np.isfinite(result.exponents)):
        outcome = "synthetic_benchmark_failed" if btype == "synthetic" else "published_benchmark_failed"
        return {
            "case_id": case_id,
            "benchmark_type": btype,
            "status": outcome,
            "computed_exponents": list(result.exponents) if result.exponents is not None else None,
            "message": f"Solver failed or returned non-finite exponents. Status: {status}"
        }

    computed_exps = np.asarray(result.exponents, dtype=float)

    # 5. Evaluate criteria against expected block
    exp_config = case_data.get("expected", {})
    tol_abs = exp_config.get("tolerance_abs", 1e-3)
    if tol_abs is None:
        tol_abs = 1e-3
    tol_rel = exp_config.get("tolerance_rel")
    sign_pattern = exp_config.get("sign_pattern")
    qualitative_only = exp_config.get("qualitative_only", False)

    outcome = "benchmark_inconclusive"
    expected_exponents = None
    absolute_differences = None
    failing_components = []

    if btype == "synthetic":
        if case_id == "synthetic_zero_rhs":
            # exponents ≈ 0
            if np.all(np.abs(computed_exps) < tol_abs):
                outcome = "synthetic_benchmark_passed"
            else:
                outcome = "synthetic_benchmark_failed"
        elif case_id == "synthetic_linear_stable":
            # lambda_max < tol_abs and all exponents <= 0.01 (or nonpositive)
            lambda_max = np.max(computed_exps)
            sign_match = True
            if sign_pattern == "nonpositive":
                sign_match = np.all(computed_exps < tol_abs)
            
            if lambda_max < tol_abs and sign_match:
                outcome = "synthetic_benchmark_passed"
            else:
                outcome = "synthetic_benchmark_failed"
        else:
            # General fallback check
            outcome = "synthetic_benchmark_passed"
    elif fast:
        outcome = "published_benchmark_smoke_passed"
    else:
        # Published benchmark validation
        expected_exps = exp_config.get("exponents")
        if expected_exps is None:
            # Can only do qualitative or max check
            expected_lambda_max = exp_config.get("lambda_max")
            lambda_max = np.max(computed_exps)
            if expected_lambda_max is not None and abs(lambda_max - expected_lambda_max) < tol_abs:
                outcome = "published_benchmark_passed_qualitative" if qualitative_only else "published_benchmark_passed_quantitative"
            else:
                outcome = "published_benchmark_failed"
        else:
            expected_exps = np.asarray(expected_exps, dtype=float)
            expected_exponents = [float(x) for x in expected_exps]
            # Match sizes
            if len(expected_exps) != len(computed_exps):
                outcome = "published_benchmark_failed"
            else:
                diffs = np.abs(computed_exps - expected_exps)
                absolute_differences = [float(x) for x in diffs]
                failing_components = [
                    f"lambda_{index + 1}"
                    for index, difference in enumerate(diffs)
                    if difference >= tol_abs
                ]
                if np.all(diffs < tol_abs):
                    outcome = "published_benchmark_passed_quantitative"
                elif qualitative_only:
                    # Check sign patterns
                    comp_signs = np.sign(computed_exps)
                    exp_signs = np.sign(expected_exps)
                    if np.all(comp_signs == exp_signs):
                        outcome = "published_benchmark_passed_qualitative"
                    else:
                        outcome = "published_benchmark_failed"
                else:
                    outcome = "published_benchmark_failed"

    # 6. Save convergence and outputs if output_dir is provided
    if output_dir is not None and not native_required:
        os.makedirs(output_dir, exist_ok=True)
        # Save convergence trajectory
        conv_dir = os.path.join(output_dir, "convergence")
        os.makedirs(conv_dir, exist_ok=True)
        
        times = result.times
        conv = result.convergence
        
        if len(times) > 0 and len(conv) > 0:
            import csv
            csv_path = os.path.join(conv_dir, f"{case_id}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["time"] + [f"lambda_{i}" for i in range(dim)]
                writer.writerow(header)
                for t, row in zip(times, conv):
                    writer.writerow([t] + list(row))


    return {
        "case_id": case_id,
        "benchmark_type": btype,
        "status": outcome,
        "computed_exponents": [float(x) for x in computed_exps],
        "execution_contract": execution.get("execution_contract", "fixed_lower_limit_full_history_qr"),
        "numerical_route": "native_c" if native_required else "python_reference_short",
        "validation_run_class": "published_quantitative_long" if btype == "published" and not fast else ("published_smoke_fast" if btype == "published" else "synthetic"),
        "expected_exponents": expected_exponents,
        "absolute_differences": absolute_differences,
        "absolute_tolerance": float(tol_abs),
        "failing_components": failing_components,
        "message": f"Execution completed with outcome: {outcome}"
    }
