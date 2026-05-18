#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the default fractional Chua piecewise seed case."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict

import numpy as np

import chua_initial_cond as chua_ic
import run_hidden_verify_frac_hybrid as hidden_verify
import unified_nyquist_hidden_pipeline as pipeline


EXPECTED_PARAMS = {
    "model": "piecewise",
    "alpha": 8.4562,
    "beta": 12.0732,
    "gamma": 0.0052,
    "m0": -0.1768,
    "m1": -1.1468,
}
EXPECTED_Q = 0.9998
EXPECTED_BRANCHES = [
    {
        "omega0": 2.04028605107949,
        "k": 0.210022792962112,
        "a0": 5.85176778548633,
        "seed": np.array([5.85176778548633, 0.370408600306164, -8.36097293442065], dtype=float),
    },
    {
        "omega0": 3.24492673097452,
        "k": 0.956945404927651,
        "a0": 1.05301661025676,
        "seed": np.array([1.05301661025676, 0.853223482977955, -1.50950868065190], dtype=float),
    },
]
ABS_TOL = 1e-5
SEED_TOL = 1e-4
RES_TOL = 1e-8


def _check_close(label: str, actual: float, expected: float, tol: float, failures: list[str]) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tol:
        failures.append(f"{label}: actual={actual:.17g}, expected={expected:.17g}, tol={tol:g}")


def _default_case_params() -> Dict[str, Any]:
    return {
        "model": str(chua_ic.PARAMS.get("model", "piecewise")).strip().lower(),
        "alpha": float(chua_ic.PARAMS["alpha"]),
        "beta": float(chua_ic.PARAMS["beta"]),
        "gamma": float(chua_ic.PARAMS["gamma"]),
        "m0": float(chua_ic.PARAMS["m0"]),
        "m1": float(chua_ic.PARAMS["m1"]),
        "a1": float(chua_ic.PARAMS.get("a1", 0.4)),
        "a2": float(chua_ic.PARAMS.get("a2", -1.5585)),
        "rho": float(chua_ic.PARAMS.get("rho", 1.0)),
    }


def _backend_arg(cmd: list[str], flag: str) -> str:
    try:
        idx = cmd.index(flag)
    except ValueError as exc:
        raise AssertionError(f"backend command is missing {flag}") from exc
    if idx + 1 >= len(cmd):
        raise AssertionError(f"backend command has no value after {flag}")
    return cmd[idx + 1]


def _validate_backend_command(failures: list[str]) -> None:
    cfg = copy.deepcopy(hidden_verify.DEFAULT_CONFIG)
    cmd = hidden_verify.build_backend_command(cfg)
    checks = {
        "--alpha_chua": 8.4562,
        "--beta": 12.0732,
        "--gamma_chua": 0.0052,
        "--m0": -0.1768,
        "--m1": -1.1468,
        "--frac_order": EXPECTED_Q,
    }
    for flag, expected in checks.items():
        _check_close(f"backend {flag}", float(_backend_arg(cmd, flag)), expected, 1e-12, failures)
    model = _backend_arg(cmd, "--model").strip().lower()
    if model != "piecewise":
        failures.append(f"backend --model: actual={model!r}, expected='piecewise'")


def _validate_pipeline_config(failures: list[str]) -> None:
    cfg = pipeline.CONFIG
    if pipeline.normalize_chua_model(cfg["model"]["kind"]) != "piecewise":
        failures.append(f"pipeline model: actual={cfg['model']['kind']!r}, expected='piecewise'")
    p = cfg["params"]
    checks = {
        "alpha_chua": 8.4562,
        "beta": 12.0732,
        "gamma_chua": 0.0052,
        "m0": -0.1768,
        "m1": -1.1468,
    }
    for key, expected in checks.items():
        _check_close(f"pipeline params.{key}", float(p[key]), expected, 1e-12, failures)
    _check_close("pipeline frac_order", float(cfg["frac_order"]), EXPECTED_Q, 1e-12, failures)
    _check_close("pipeline basin.q", float(cfg["basin"]["q"]), EXPECTED_Q, 1e-12, failures)


def _validate_native_efork(params: Dict[str, Any], qord: float, failures: list[str]) -> None:
    pipeline.ensure_current_chua_params(pipeline.CONFIG)
    x0 = EXPECTED_BRANCHES[0]["seed"]
    t_total = 0.2
    h = 0.01
    lm = 0.2
    traj_c = pipeline.integrate_original(x0, params, qord=qord, h=h, Lm=lm, t_total=t_total)
    traj_py = chua_ic.efork3_integrate(lambda x: chua_ic.rhs_original(x, params), x0, qord=qord, h=h, Lm=lm, t_f=t_total)
    err = float(np.max(np.abs(traj_c - traj_py)))
    if err > 1e-10:
        failures.append(f"native EFORK C mismatch: max_abs_error={err:.17g}, tol=1e-10")
    print(f"Native EFORK C smoke: max_abs_error={err:.3e}")


def main() -> int:
    failures: list[str] = []
    params = _default_case_params()
    qord = float(chua_ic.QORD)

    for key, expected in EXPECTED_PARAMS.items():
        actual = params[key]
        if isinstance(expected, str):
            if actual != expected:
                failures.append(f"param {key}: actual={actual!r}, expected={expected!r}")
        else:
            _check_close(f"param {key}", float(actual), float(expected), 1e-12, failures)
    _check_close("q", qord, EXPECTED_Q, 1e-12, failures)

    pairs = chua_ic.find_omega_k_pairs(qord, params, wmin=chua_ic.WMIN, wmax=chua_ic.WMAX, nscan=chua_ic.NSCAN)
    if len(pairs) < len(EXPECTED_BRANCHES):
        failures.append(f"branches: actual={len(pairs)}, expected_at_least={len(EXPECTED_BRANCHES)}")

    branch_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, float]] = []
    for i, expected in enumerate(EXPECTED_BRANCHES):
        if i >= len(pairs):
            break
        omega0, k = pairs[i][:2]
        a0 = chua_ic.solve_amplitude_from_k(k, params)
        seed, eigvec, _eig_match = chua_ic.build_fractional_seed(qord, params, omega0, k, a0)
        W0 = chua_ic.W_frac(omega0, qord, params)
        P0 = chua_ic.build_P0(params, k).astype(np.complex128)
        zeta0 = chua_ic.cpower_iw_q(omega0, qord)
        eig_resid = float(np.linalg.norm((P0 - zeta0 * np.eye(3, dtype=np.complex128)) @ eigvec))
        n_resid = float(chua_ic.N_sat(a0, params) - k)
        im_resid = float(np.imag(W0))
        seed_error = float(np.linalg.norm(seed - expected["seed"]))

        _check_close(f"branch {i} omega0", float(omega0), float(expected["omega0"]), ABS_TOL, failures)
        _check_close(f"branch {i} k", float(k), float(expected["k"]), ABS_TOL, failures)
        _check_close(f"branch {i} a0", float(a0), float(expected["a0"]), ABS_TOL, failures)
        if seed_error > SEED_TOL:
            failures.append(f"branch {i} seed_error={seed_error:.17g}, tol={SEED_TOL:g}")
        if abs(im_resid) > RES_TOL:
            failures.append(f"branch {i} Im(W_code)={im_resid:.17g}, tol={RES_TOL:g}")
        if abs(n_resid) > RES_TOL:
            failures.append(f"branch {i} N(a0)-k={n_resid:.17g}, tol={RES_TOL:g}")
        if eig_resid > RES_TOL:
            failures.append(f"branch {i} eigen_residual={eig_resid:.17g}, tol={RES_TOL:g}")

        branch_rows.append({
            "index": i,
            "omega0": float(omega0),
            "k": float(k),
            "a0": float(a0),
            "seed": np.asarray(seed, dtype=float).tolist(),
        })
        residual_rows.append({
            "index": i,
            "Im(W_code(i omega0))": im_resid,
            "N(a0)-k": n_resid,
            "norm((P0-zeta0 I)v)": eig_resid,
        })

    if branch_rows:
        _check_close("selected branch k", float(branch_rows[0]["k"]), EXPECTED_BRANCHES[0]["k"], ABS_TOL, failures)

    _validate_pipeline_config(failures)
    _validate_native_efork(params, qord, failures)
    _validate_backend_command(failures)

    print("Effective parameters:")
    print(f"  model={params['model']}")
    print(
        "  "
        f"alpha={params['alpha']:.17g}, beta={params['beta']:.17g}, "
        f"gamma={params['gamma']:.17g}, m0={params['m0']:.17g}, m1={params['m1']:.17g}"
    )
    print(f"  q={qord:.17g}")
    print("Branches found:")
    for row in branch_rows:
        print(
            f"  [{row['index']}] omega0={row['omega0']:.14f}, "
            f"k={row['k']:.15f}, a0={row['a0']:.14f}, seed={row['seed']}"
        )
    if branch_rows:
        print("Selected branch:")
        print(f"  index=0, seed={branch_rows[0]['seed']}")
    print("Residuals:")
    for row in residual_rows:
        print(
            f"  [{row['index']}] Im(W_code)={row['Im(W_code(i omega0))']:.3e}, "
            f"N(a0)-k={row['N(a0)-k']:.3e}, "
            f"norm((P0-zeta0 I)v)={row['norm((P0-zeta0 I)v)']:.3e}"
        )

    if failures:
        print("Validation failed:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
