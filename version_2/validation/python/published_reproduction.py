"""Core reproduction logic for the published_case_reproduction validation layer.

This module implements functions to compute seeds and trajectories using the published
methods, compare them with paper data, and produce standard JSON outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.special import gamma as scipy_gamma

from hidden_attractors.models.chua import chua_parameters
from hidden_attractors.systems.builtins import chua_system, chua_arctan_wu2023_system
from hidden_attractors.lure.seeds import build_lure_seed
from hidden_attractors.seed_generation.chua import find_harmonic_seed
from hidden_attractors.solvers import efork3_caputo_integrate, efork_q1_integrate


def W_published_integer(omega: float, P: np.ndarray, b: np.ndarray, r: np.ndarray) -> complex:
    """W_pub(j omega) = r^T (j omega I - P)^(-1) b.

    This is the classical transfer function mode used in the publications.
    """
    z = 1j * omega
    I = np.eye(len(P))
    return complex(r @ np.linalg.solve(z * I - P, b))


def W_fractional_spectral(omega: float, q: float, P: np.ndarray, b: np.ndarray, r: np.ndarray) -> complex:
    """W_q(j omega) = r^T ((j omega)^q I - P)^(-1) b.

    This is the fractional spectral extension of the transfer function.
    """
    if omega <= 0.0:
        raise ValueError("omega must be positive and finite for spectral transfer.")
    if not (0.0 < q <= 1.0):
        raise ValueError("q must satisfy 0 < q <= 1.")
    
    # Use the explicit branch formula
    z = (omega ** q) * np.exp(1j * q * np.pi / 2.0)
    I = np.eye(len(P))
    return complex(r @ np.linalg.solve(z * I - P, b))


def build_published_integer_seed_closed_form(
    system: any,
    a0: float,
    omega0: float,
    k: float,
    seed_sign_convention: str = "kuznetsov"
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the seed using the algebraic closed-form formula for q = 1 without eigenvectors."""
    alpha = float(system.parameters.get("alpha", 8.4562))

    if "m1" in system.parameters:
        m_linear = float(system.parameters["m1"])
    elif "a1" in system.parameters:
        m_linear = float(system.parameters["a1"])
    else:
        raise ValueError(
            "System parameters must contain either 'm1' (saturation) or 'a1' (arctan)."
        )

    x0 = a0
    y0 = a0 * (m_linear + 1.0 + k)
    z0 = a0 * (alpha * (m_linear + k) - omega0 ** 2) / alpha

    if seed_sign_convention == "wu":
        y0 = -y0

    X_seed = np.array([x0, y0, z0], dtype=float)
    return X_seed, -X_seed


def get_reproduction_system(system_id: str, params_dict: dict):
    """Reconstruct the ChaoticSystem with custom parameters, rebuilding its Lur'e representation."""
    model = params_dict.get("model", "nonsmooth")
    if "a1" in params_dict or model == "arctan":
        model = "arctan"
    else:
        model = "nonsmooth"

    p = chua_parameters(
        model=model,
        alpha=params_dict.get("alpha", 8.4562),
        beta=params_dict.get("beta", 12.0732),
        gamma=params_dict.get("gamma", 0.0052),
        m0=params_dict.get("m0", -0.1768),
        m1=params_dict.get("m1", -1.1468),
        a1=params_dict.get("a1", 0.4),
        a2=params_dict.get("a2", -1.5585),
        rho=params_dict.get("rho", 1.0)
    )

    from hidden_attractors.systems.builtins import _chua_rhs, _chua_equilibria, _chua_jacobian, _chua_lure_system
    from hidden_attractors.systems import ChaoticSystem

    parameters = {
        "model": p.model,
        "alpha": p.alpha,
        "beta": p.beta,
        "gamma": p.gamma,
        "m0": p.m0,
        "m1": p.m1,
        "a1": p.a1,
        "a2": p.a2,
        "rho": p.rho,
    }

    return ChaoticSystem(
        name=f"reproduction-{system_id}",
        dimension=3,
        rhs=_chua_rhs,
        equilibria=_chua_equilibria,
        jacobian=_chua_jacobian,
        parameters=parameters,
        description="Reproduction system",
        lure=_chua_lure_system(parameters),
    )


def compute_seed_for_reproduction(case_config: dict) -> dict:
    """Compute the Lur'e describing function seed according to the case's seed_transfer_mode."""
    system_id = case_config.get("system_id")
    seed_rep = case_config.get("seed_reproduction", {})
    seed_transfer_mode = seed_rep.get("seed_transfer_mode", "published_integer_laplace")

    dynamics_cfg = case_config.get("dynamics", {})
    q_dynamics = float(dynamics_cfg.get("q", 1.0))

    expected_cfg = case_config.get("expected", {})
    params_dict = expected_cfg.get("parameters", {})

    sys = get_reproduction_system(system_id, params_dict)
    P = sys.lure.matrix
    b = sys.lure.input_vector
    r = sys.lure.output_vector
    
    # Extract the ChuaParameters object
    p = chua_parameters(
        model=sys.parameters["model"],
        alpha=sys.parameters["alpha"],
        beta=sys.parameters["beta"],
        gamma=sys.parameters["gamma"],
        m0=sys.parameters["m0"],
        m1=sys.parameters["m1"],
        a1=sys.parameters["a1"],
        a2=sys.parameters["a2"],
        rho=sys.parameters["rho"]
    )

    if seed_transfer_mode == "published_integer_laplace":
        q_seed = 1.0
        transfer_mode = "integer"
        z_rule = "z = j*omega"
        q_dependent_seed = False
        W_mode_used = "W_pub(j omega) = r^T(j omega I - P)^(-1)b"
    else:
        q_seed = q_dynamics
        transfer_mode = "fractional"
        z_rule = "z = (j*omega)^q"
        q_dependent_seed = True
        W_mode_used = "W_q(j omega) = r^T((j omega)^q I - P)^(-1)b"

    try:
        if seed_transfer_mode == "published_integer_laplace":
            from hidden_attractors.seed_generation.chua import find_omega_gain_candidates, solve_amplitude_from_gain
            pairs = find_omega_gain_candidates(
                q=q_seed,
                params=p,
                wmin=1.0e-4,
                wmax=10.0,
                nscan=20_000,
                compatible_only=True
            )
            if not pairs:
                raise RuntimeError("no omega/gain candidate was found.")
            omega0, k = pairs[0]
            a0 = solve_amplitude_from_gain(k, p)

            seed_plus, seed_minus = build_published_integer_seed_closed_form(
                sys,
                a0=a0,
                omega0=omega0,
                k=k,
                seed_sign_convention=seed_rep.get("seed_sign_convention", "kuznetsov")
            )
        else:
            seed_obj = find_harmonic_seed(
                q=q_seed,
                params=p,
                branch_index=0,
                method="classic",
                wmin=1.0e-4,
                wmax=10.0,
                nscan=20_000
            )
            omega0 = float(seed_obj.omega)
            k = float(seed_obj.gain)
            a0 = float(seed_obj.amplitude)

            seed_plus, seed_minus = build_lure_seed(
                sys,
                A0=a0,
                omega0=omega0,
                k=k,
                seed_sign_convention=seed_rep.get("seed_sign_convention", "kuznetsov"),
                q=q_seed,
                transfer_mode=transfer_mode,
                theta=0.0,
                seed_construction="modal",
            )

        # Calculate W_at_omega0 using the corresponding formula
        if seed_transfer_mode == "published_integer_laplace":
            W_val = W_published_integer(omega0, P, b, r)
        else:
            W_val = W_fractional_spectral(omega0, q_seed, P, b, r)

        return {
            "status": "ok",
            "q_seed": q_seed,
            "omega0": omega0,
            "k": k,
            "a0": a0,
            "seed_plus": seed_plus.tolist(),
            "seed_minus": seed_minus.tolist(),
            "seed_transfer_mode": seed_transfer_mode,
            "z_rule": z_rule,
            "q_dependent_seed": q_dependent_seed,
            "dynamics_q": q_dynamics,
            "W_mode_used": W_mode_used,
            "W_at_omega0": {
                "re": float(W_val.real),
                "im": float(W_val.imag)
            }
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "seed_transfer_mode": seed_transfer_mode,
            "z_rule": z_rule,
            "q_dependent_seed": q_dependent_seed,
            "dynamics_q": q_dynamics,
            "W_mode_used": W_mode_used
        }


def compare_paper_seed(case_config: dict, computed_seed: dict) -> dict:
    """Compare computed seed values against expected values in the case configuration."""
    expected = case_config.get("expected", {})

    if (expected.get("omega0") is None or
        expected.get("k") is None or
        expected.get("a0") is None or
        expected.get("seed_plus") is None or
        expected.get("seed_minus") is None):
        return {
            "status": "paper_data_missing",
            "notes": "One or more expected seed metrics (omega0, k, a0, seed_plus/minus) are null in case config."
        }

    if computed_seed.get("status") == "error":
        return {
            "status": "not_reproduced",
            "error": computed_seed.get("error")
        }

    omega0_diff = abs(computed_seed["omega0"] - expected["omega0"])
    k_diff = abs(computed_seed["k"] - expected["k"])
    a0_diff = abs(computed_seed["a0"] - expected["a0"])
    seed_plus_diff = max(abs(c - e) for c, e in zip(computed_seed["seed_plus"], expected["seed_plus"]))
    seed_minus_diff = max(abs(c - e) for c, e in zip(computed_seed["seed_minus"], expected["seed_minus"]))

    passed = (
        omega0_diff <= 1e-8 and
        k_diff <= 1e-8 and
        a0_diff <= 1e-8 and
        seed_plus_diff <= 1e-7 and
        seed_minus_diff <= 1e-7
    )

    if passed:
        return {
            "status": "paper_seed_reproduced",
            "omega0_diff": omega0_diff,
            "k_diff": k_diff,
            "a0_diff": a0_diff,
            "seed_plus_diff": seed_plus_diff,
            "seed_minus_diff": seed_minus_diff
        }
    else:
        return {
            "status": "not_reproduced",
            "omega0_diff": omega0_diff,
            "k_diff": k_diff,
            "a0_diff": a0_diff,
            "seed_plus_diff": seed_plus_diff,
            "seed_minus_diff": seed_minus_diff
        }


def caputo_abm_integrate(
    rhs: callable,
    x0: list[float],
    q: float,
    h: float,
    t_final: float,
    divergence_norm: float = 120.0
) -> tuple[np.ndarray, str]:
    """Integrate commensurate Caputo system using Diethelm ABM solver."""
    q = float(q)
    h = float(h)
    n_steps = int(np.ceil(t_final / h))
    x0_arr = np.asarray(x0, dtype=float).copy()

    x_hist = np.zeros((n_steps + 1, len(x0_arr)), dtype=float)
    f_hist = np.zeros((n_steps + 1, len(x0_arr)), dtype=float)
    x_hist[0] = x0_arr
    f_hist[0] = rhs(x0_arr)

    powers = np.arange(n_steps + 2, dtype=float)
    pow_q = powers ** q
    pow_q1 = powers ** (q + 1.0)
    pred_scale = (h ** q) / scipy_gamma(q + 1.0)
    corr_scale = (h ** q) / scipy_gamma(q + 2.0)

    diverged = False
    for i in range(n_steps):
        b = pow_q[1 : i + 2][::-1] - pow_q[0 : i + 1][::-1]
        predictor = x0_arr + pred_scale * (b @ f_hist[: i + 1])
        fp = rhs(predictor)

        if i == 0:
            a_weights = np.array([q], dtype=float)
        else:
            r_indices = np.arange(i, 0, -1, dtype=int)
            mid = pow_q1[r_indices + 1] + pow_q1[r_indices - 1] - 2.0 * pow_q1[r_indices]
            a0 = pow_q1[i] - (float(i) - q) * pow_q[i + 1]
            a_weights = np.concatenate(([a0], mid))

        corrected = x0_arr + corr_scale * ((a_weights @ f_hist[: i + 1]) + fp)
        x_hist[i + 1] = corrected

        if not np.all(np.isfinite(corrected)) or np.linalg.norm(corrected) > divergence_norm:
            x_hist = x_hist[: i + 2]
            diverged = True
            break
        f_hist[i + 1] = rhs(corrected)

    t = np.arange(len(x_hist)) * h
    traj = np.column_stack((t, x_hist))
    return traj, "ok" if not diverged else "diverged"


def run_case_reproduction(
    case_path: str | Path,
    output_dir: str | Path,
    run_dynamics: bool | None = None
) -> dict:
    """Run reproduction steps for a single case YAML file."""
    import yaml
    case_path = Path(case_path)
    output_dir = Path(output_dir)

    # Get repo root dynamically to resolve reference JSON
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]

    with open(case_path, "r", encoding="utf-8") as f:
        case_config = yaml.safe_load(f)

    case_id = case_config.get("case_id")
    reference = case_config.get("reference")
    system_id = case_config.get("system_id")
    seed_rep = case_config.get("seed_reproduction", {})
    seed_transfer_mode = seed_rep.get("seed_transfer_mode", "published_integer_laplace")
    q_dependent_seed = seed_rep.get("q_dependent_seed", False)

    dynamics_cfg = case_config.get("dynamics", {})
    derivative = dynamics_cfg.get("derivative", "Caputo")
    dynamics_q = float(dynamics_cfg.get("q", 1.0))
    integrator = dynamics_cfg.get("integrator", "EFORK")
    h = float(dynamics_cfg.get("h", 0.01))
    t_final = float(dynamics_cfg.get("t_final", 100.0))

    # Read run_dynamics from YAML if not specified explicitly as argument
    if run_dynamics is None:
        run_dynamics = case_config.get("run_dynamics", False)

    # Load reference JSON file if specified and merge it with expected
    expected_cfg = case_config.get("expected", {})
    ref_file = case_config.get("reference_data_file")
    if ref_file:
        ref_path = repo_root / ref_file
        if ref_path.exists():
            with open(ref_path, "r", encoding="utf-8") as rf:
                ref_data = json.load(rf)
            
            # YAML expected overrides reference JSON
            base_expected = {}
            for k, v in ref_data.items():
                if k not in ["case_id", "paper"]:
                    base_expected[k] = v
            
            for k, v in expected_cfg.items():
                if v is not None:
                    if isinstance(v, dict) and isinstance(base_expected.get(k), dict):
                        base_expected[k] = {**base_expected[k], **v}
                    else:
                        base_expected[k] = v
            expected_cfg = base_expected

    case_config["expected"] = expected_cfg
    expected = expected_cfg
    params_dict = expected.get("parameters", {})

    # Create target directories
    case_output_dir = output_dir / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Formula validation
    sys = get_reproduction_system(system_id, params_dict)
    P = sys.lure.matrix
    b = sys.lure.input_vector
    r = sys.lure.output_vector
    
    formula_ok = True
    for test_omega in [1.0, 2.0, 5.0]:
        W_int_pub = W_published_integer(test_omega, P, b, r)
        W_int_exact = r @ np.linalg.solve(1j * test_omega * np.eye(len(P)) - P, b)
        if abs(W_int_pub - W_int_exact) > 1e-12:
            formula_ok = False

        W_frac_pub = W_fractional_spectral(test_omega, dynamics_q, P, b, r)
        W_frac_exact = r @ np.linalg.solve(
            ((test_omega ** dynamics_q) * np.exp(1j * dynamics_q * np.pi / 2.0)) * np.eye(len(P)) - P, b
        )
        if abs(W_frac_pub - W_frac_exact) > 1e-12:
            formula_ok = False

    # 2. Compute seed
    computed_seed = compute_seed_for_reproduction(case_config)
    seed_comparison = compare_paper_seed(case_config, computed_seed)

    # Save seed_reproduction.json
    seed_rep_out = {
        "computed_seed": computed_seed,
        "comparison": seed_comparison
    }
    with open(case_output_dir / "seed_reproduction.json", "w", encoding="utf-8") as f:
        json.dump(seed_rep_out, f, indent=2)

    # 3. Simulate Dynamics (or skip)
    trajectories_info = {}
    trajectory_reproduced = True
    ic_reproduced = False

    if run_dynamics:
        # Determine initial conditions to test
        ic_from_paper = expected.get("initial_conditions_from_paper")
        ic_list = {}
        if ic_from_paper is not None:
            ic_reproduced = True
            if isinstance(ic_from_paper, dict):
                for name, ic in ic_from_paper.items():
                    ic_list[name] = ic
            else:
                ic_list["paper_ic"] = ic_from_paper
        else:
            # Fall back to computed seed
            if computed_seed.get("status") == "ok":
                ic_list["seed_plus"] = computed_seed["seed_plus"]
                ic_list["seed_minus"] = computed_seed["seed_minus"]

        # Define RHS
        def rhs_func(x):
            return sys.rhs(x, sys.parameters)

        def rhs_t_y(t, y):
            return sys.rhs(y, sys.parameters)

        # Run integration for each initial condition
        for ic_name, x0 in ic_list.items():
            if derivative == "integer":
                traj, status = efork_q1_integrate(rhs_func, np.asarray(x0), t_final=t_final, h=h)
            else:
                # Caputo
                if integrator == "ABM":
                    traj, status = caputo_abm_integrate(rhs_func, x0, q=dynamics_q, h=h, t_final=t_final)
                else:
                    # EFORK
                    times, states = efork3_caputo_integrate(rhs_t_y, np.asarray(x0), alpha=dynamics_q, h=h, t_final=t_final)
                    traj = np.column_stack((times, states))
                    last_norm = np.linalg.norm(states[-1])
                    status = "ok" if (last_norm < 120.0 and np.all(np.isfinite(states))) else "diverged"

            trajectories_info[ic_name] = {
                "initial_condition": x0,
                "status": status,
                "final_state": traj[-1, 1:].tolist() if len(traj) > 0 else None,
                "max_norm": float(np.max(np.linalg.norm(traj[:, 1:], axis=1))) if len(traj) > 0 else 0.0
            }
            if status != "ok":
                trajectory_reproduced = False

        dynamics_rep_out = {
            "derivative": derivative,
            "q": dynamics_q,
            "integrator": integrator,
            "trajectories": trajectories_info
        }
    else:
        trajectory_reproduced = False
        dynamics_rep_out = {
            "status": "skipped",
            "reason": "run_dynamics=false"
        }

    with open(case_output_dir / "dynamics_reproduction.json", "w", encoding="utf-8") as f:
        json.dump(dynamics_rep_out, f, indent=2)

    # 4. Missing data report
    missing_data_list = case_config.get("data_status", {}).get("missing_values", [])
    notes = case_config.get("data_status", {}).get("notes", "")
    missing_rep_out = {
        "missing_values": missing_data_list,
        "notes": notes
    }
    with open(case_output_dir / "missing_data_report.json", "w", encoding="utf-8") as f:
        json.dump(missing_rep_out, f, indent=2)

    # 5. Populate statuses
    statuses = []
    if formula_ok:
        statuses.append("paper_formula_reproduced")

    # Evaluate seed reproduction
    seed_status = seed_comparison.get("status")
    if seed_status == "paper_seed_reproduced":
        statuses.append("paper_seed_reproduced")
    elif seed_status == "paper_data_missing":
        statuses.append("paper_data_missing")
        statuses.append("paper_partially_reproduced_missing_seed_data")

    # Evaluate initial condition and trajectory
    ic_from_paper = expected.get("initial_conditions_from_paper")
    if ic_from_paper is not None:
        statuses.append("paper_initial_condition_reproduced")
    else:
        statuses.append("paper_partially_reproduced_missing_ic_data")

    if run_dynamics and len(trajectories_info) > 0 and trajectory_reproduced:
        statuses.append("paper_trajectory_reproduced")

    # General reproduction categories
    if "paper_data_missing" in statuses:
        statuses.append("paper_partially_reproduced")
    elif not run_dynamics:
        # If dynamics skipped and not fully validated
        statuses.append("paper_partially_reproduced")
        if all(s in statuses for s in ["paper_formula_reproduced", "paper_seed_reproduced"]):
            # No dynamic verification, but formula + seed match
            pass
    elif all(s in statuses for s in ["paper_formula_reproduced", "paper_seed_reproduced", "paper_initial_condition_reproduced", "paper_trajectory_reproduced"]):
        statuses.append("paper_fully_reproduced")
    else:
        statuses.append("paper_partially_reproduced")

    if len(statuses) == 1 and "paper_formula_reproduced" in statuses:
        statuses.append("paper_formula_reproduced_only")

    # Save reproduction_summary.json (strictly schema compliant)
    summary_out = {
        "case_id": case_id,
        "reference": reference,
        "seed_transfer_mode": seed_transfer_mode,
        "q_dependent_seed": q_dependent_seed,
        "dynamics_q": dynamics_q,
        "statuses": statuses,
        "missing_data": missing_data_list,
        "no_hidden_verified_claim": True,
        "hiddenness_certified_by_this_pipeline": False
    }
    with open(case_output_dir / "reproduction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    return summary_out


def run_all_published_cases(
    output_dir: str | Path = "validation/outputs/published_cases",
    run_dynamics: bool = False
) -> dict:
    """Run reproduction for all registered cases."""
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    
    cases_dir = repo_root / "validation" / "published_cases"
    out_dir = repo_root / Path(output_dir)
    
    results = {}
    for case_file in sorted(cases_dir.glob("*.yaml")):
        summary = run_case_reproduction(case_file, out_dir, run_dynamics=run_dynamics)
        results[summary["case_id"]] = summary
    return results
