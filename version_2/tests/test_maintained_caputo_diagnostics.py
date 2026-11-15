"""Independent analytic controls and contracts for maintained Caputo methods."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import erfcx, gamma

from hidden_attractors.integrations.abm_fractional import integrate_fractional_abm
from hidden_attractors.integrations.abm_reference import integrate_reference_abm
from hidden_attractors.integrations.caputo_diagnostics import (
    integrate_abm_stable, integrate_l1_implicit, stable_caputo_weights,
)
from hidden_attractors.integrations.selector import integrate, validate_integrator_compatibility


@pytest.mark.parametrize("initial", [0., [0.], [[0., 0.]], [0., 0., 0.]])
def test_reference_rejects_initial_rhs_broadcast(initial):
    with pytest.raises(ValueError, match="initial RHS"):
        integrate_reference_abm(lambda t, x: initial, np.zeros(2), .9, .01, .03)


@pytest.mark.parametrize("bad_call", [2, 3])
@pytest.mark.parametrize("bad_value", [0., [0.], [[0., 0.]], [0., 0., 0.]])
def test_reference_rejects_later_rhs_before_assignment(bad_call, bad_value):
    calls = 0
    def rhs(t, x):
        nonlocal calls
        calls += 1
        return bad_value if calls == bad_call else np.ones(2)
    times, states, status = integrate_reference_abm(rhs, np.zeros(2), .9, .01, .03)
    assert status == "solver_exception"
    assert len(times) == len(states) == (1 if bad_call == 2 else 2)


@pytest.mark.parametrize("x0", [0., [[1.]], [], [np.nan]])
def test_reference_rejects_nonvector_initial_state(x0):
    with pytest.raises(ValueError, match="x0"):
        integrate_reference_abm(lambda t, x: x, x0, .5, .01, .03)


@pytest.mark.parametrize("bad_call", [2, 3])
def test_reference_distinguishes_nonfinite_rhs_from_threshold(bad_call):
    calls = 0
    def rhs(t, x):
        nonlocal calls
        calls += 1
        return np.full(2, np.nan) if calls == bad_call else np.ones(2)
    times, states, status = integrate_reference_abm(rhs, np.zeros(2), .9, .01, .03)
    assert status == "nonfinite_solution"
    assert np.all(np.isfinite(states))


@pytest.mark.parametrize("method", ["l1_implicit", "abm_stable"])
def test_diagnostic_selector_contracts(method):
    assert validate_integrator_compatibility(method, .9) == method
    with pytest.raises(ValueError, match="requires q < 1"):
        validate_integrator_compatibility(method, 1.)
    for options in [{"memory_mode": "window"}, {"memory_window_length": 8},
                    {"early_stop_config": {"enabled": True}},
                    {"solver_options": {"misspelled_tolerance": .1}}]:
        with pytest.raises(ValueError):
            integrate(lambda t, x: -x, np.ones(1), .9, .01, .03, integrator=method, **options)


@pytest.mark.parametrize("method", ["l1_implicit", "abm_stable"])
@pytest.mark.parametrize("shape", [(), (1,), (1, 2), (3,)])
def test_diagnostic_rejects_initial_rhs_broadcast(method, shape):
    with pytest.raises(ValueError, match="invalid initial RHS"):
        integrate(lambda t, x: np.zeros(shape), np.zeros(2), .9, .01, .03, integrator=method)


@pytest.mark.parametrize("method", ["l1_implicit", "abm_stable"])
def test_later_bad_rhs_returns_only_accepted_states_and_metadata(method):
    metadata = {}
    def rhs(t, x):
        return np.ones(2) if t == 0 else 1.
    times, states, status = integrate(rhs, np.zeros(2), .9, .01, .03,
                                    integrator=method, diagnostics=metadata)
    assert status == "solver_exception"
    assert len(times) == len(states) == 1
    assert "RHS shape" in metadata["failure_detail"]
    assert not metadata["full_horizon_reached"]


def test_stable_pece_matches_independent_existing_direct_scheme():
    def rhs(t, x):
        return np.array([-.4 * x[0] + .1 * x[1] + .01 * t, -.2 * x[1] + .05 * x[0]])
    anchor = np.array([.1, -.2])
    _, expected, old_status = integrate_fractional_abm(rhs, anchor, [.9, .9], .01, 83)
    _, actual, status = integrate_abm_stable(rhs, anchor, .9, .01, .83, use_numba=False)
    assert old_status == status == "ok"
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=3e-14)


@pytest.mark.parametrize("q", [.5, .9, .9999])
def test_stable_pece_constant_forcing_has_caputo_closed_form(q):
    forcing = np.array([.7, -.2])
    t, x, status = integrate_abm_stable(lambda t, x: forcing, np.array([.1, -.2]), q, .01, 1., use_numba=False)
    exact = np.array([.1, -.2]) + t[:, None]**q * forcing / gamma(q + 1)
    assert status == "ok"
    np.testing.assert_allclose(x, exact, rtol=2e-13, atol=3e-14)


@pytest.mark.parametrize("solver", [integrate_l1_implicit, integrate_abm_stable])
def test_half_order_linear_closed_form_refinement(solver):
    errors = []
    for h in [.02, .01, .005]:
        t, x, status = solver(lambda t, x: -x, np.ones(1), .5, h, 1., use_numba=False)
        assert status == "ok"
        errors.append(float(np.max(abs(x[:, 0] - erfcx(np.sqrt(t))))))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < .025


@pytest.mark.parametrize("solver", [integrate_l1_implicit, integrate_abm_stable])
def test_manufactured_smooth_solution_refinement(solver):
    q = .7
    errors = []
    for h in [.04, .02, .01]:
        t, x, status = solver(lambda t, x: np.array([gamma(q + 2) * t]), np.zeros(1), q, h, 1., use_numba=False)
        assert status == "ok"
        errors.append(float(np.max(abs(x[:, 0] - t**(q + 1)))))
    if solver is integrate_abm_stable:
        # PECE integrates a linear forcing exactly; only roundoff remains.
        assert max(errors) < 5e-14
    else:
        assert errors[2] < errors[1] < errors[0]


def test_l1_integer_limit_is_implicit_euler_without_changing_contract():
    q = np.nextafter(1., 0.)
    h = .02
    t, x, status = integrate_l1_implicit(lambda t, x: -x, np.ones(1), q, h, .4,
                                       jacobian=lambda t, x: -np.eye(1), use_numba=False)
    assert status == "ok"
    np.testing.assert_allclose(x[:, 0], (1 + h)**(-np.arange(len(t))), atol=3e-14, rtol=3e-14)
    # An exactly integer order is deliberately handled by the integer solvers.
    with pytest.raises(ValueError, match="0 < q < 1"):
        integrate_l1_implicit(lambda t, x: -x, np.ones(1), 1., h, .4)


def test_l1_analytic_jacobian_and_declared_finite_difference_agree():
    rhs = lambda t, x: -x*x
    exact_jacobian = lambda t, x: np.diag(-2*x)
    analytic, finite = {}, {}
    _, a, sa = integrate_l1_implicit(rhs, np.array([.5, 1.]), .8, .01, .3, jacobian=exact_jacobian, diagnostics=analytic, use_numba=False)
    _, f, sf = integrate_l1_implicit(rhs, np.array([.5, 1.]), .8, .01, .3, diagnostics=finite, use_numba=False)
    assert sa == sf == "ok"
    np.testing.assert_allclose(a, f, atol=2e-13, rtol=2e-13)
    assert analytic["jacobian"] == "analytic"
    assert finite["jacobian"] == "central_finite_difference"
    assert finite["nfev"] > analytic["nfev"]
    assert analytic["newton_residual_max"] <= analytic["newton_tolerance"]


def test_l1_jacobian_shape_error_and_failed_newton_are_explicit():
    metadata = {}
    _, x, status = integrate_l1_implicit(lambda t, x: -x, np.ones(2), .7, .01, .1,
                                       jacobian=lambda t, x: 1., diagnostics=metadata, use_numba=False)
    assert status == "solver_exception" and len(x) == 1
    assert "Jacobian" in metadata["failure_detail"]
    _, x, status = integrate_l1_implicit(lambda t, x: x*x, np.ones(1), .7, .1, .2,
                                       newton_maxiter=1, diagnostics=metadata, use_numba=False)
    assert status == "newton_failure" and len(x) == 1
    assert metadata["newton_failures"] == 1


@pytest.mark.parametrize("method", ["l1_implicit", "abm_stable"])
def test_full_memory_metadata_threshold_and_no_restart(method):
    metadata = {}
    t, x, status = integrate(lambda t, x: np.ones(1), np.zeros(1), .5, .01, .5,
                             integrator=method, divergence_norm=.2, diagnostics=metadata,
                             solver_options={"use_numba": False})
    assert status == "diverged" and x[-1, 0] > .2
    assert metadata["caputo_lower_limit"] == 0 and metadata["memory_mode"] == "full"
    assert not metadata["history_restarted"] and not metadata["history_truncated"]
    assert metadata["completed_steps"] == len(t)-1
    assert not metadata["full_horizon_reached"]


@pytest.mark.parametrize("method", ["l1_implicit", "abm_stable"])
def test_compiled_and_python_compensated_reductions_match(method):
    kwargs = dict(rhs=lambda t, x: -.3*x, x0=np.array([1., -.2]), q=.9, h=.01,
                  t_final=.15, integrator=method)
    _, python, pstatus = integrate(**kwargs, solver_options={"use_numba": False})
    metadata = {}
    _, compiled, cstatus = integrate(**kwargs, solver_options={"use_numba": True}, diagnostics=metadata)
    assert cstatus == pstatus == "ok"
    np.testing.assert_array_equal(python, compiled)


def test_stable_weights_do_not_erase_large_lag_near_integer_memory():
    q = np.nextafter(1., 0.)
    (weights,) = stable_caputo_weights(q, 2001, "l1_implicit")
    assert np.all(weights[1:] > 0)
    p, a, a0 = stable_caputo_weights(.9999, 2001, "abm_stable")
    assert np.all(p > 0) and np.all(a > 0) and np.all(a0 > 0)


def test_cli_compare_complete_provenance_and_refuses_overwrite(tmp_path):
    output = tmp_path / "comparison"
    command = [sys.executable, "-m", "hidden_attractors.cli.main", "integrate", "compare",
               "--system", "chua-nonsmooth", "--x0", ".01", ".02", ".03", "--q", ".9",
               "--h", ".02", ".01", "--t-final", ".1", "--no-numba", "--output-dir", str(output)]
    repository = Path(__file__).resolve().parents[1]
    process = subprocess.run(command, cwd=repository, text=True, capture_output=True)
    assert process.returncode == 0, process.stderr + process.stdout
    summary = json.loads((output / "summary.json").read_text())
    assert summary["all_runs_completed"] and len(summary["runs"]) == 4
    assert len(summary["method_comparisons"]) == 2
    assert summary["source_files"] and all(len(s["sha256"]) == 64 for s in summary["source_files"])
    for run in summary["runs"]:
        data = np.load(output / run["trajectory"])
        assert len(data["t"]) == len(data["x"]) == run["rows"]
        assert data["t"][0] == 0 and data["t"][-1] == pytest.approx(.1)
    before = (output / "summary.json").read_bytes()
    repeated = subprocess.run(command, cwd=repository, text=True, capture_output=True)
    assert repeated.returncode != 0 and "not empty" in repeated.stderr
    assert (output / "summary.json").read_bytes() == before


def test_cli_work_budget_rejects_large_experiment_before_claim(tmp_path):
    output = tmp_path / "over_budget"
    command = [sys.executable, "-m", "hidden_attractors.cli.main", "integrate", "compare",
               "--system", "chua-nonsmooth", "--x0", ".01", ".02", ".03", "--q", ".9",
               "--h", ".001", "--t-final", "1000", "--output-dir", str(output)]
    process = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert process.returncode != 0 and "exceeding --max-history-terms" in process.stderr
    assert not output.exists()


def test_finite_pece_threshold_precedes_rhs_at_crossed_state():
    calls = 0
    def rhs(t, x):
        nonlocal calls
        calls += 1
        # The third callback would evaluate f at the crossed corrected state.
        if calls == 3:
            raise OverflowError("unneeded RHS after the threshold")
        return np.ones(1)
    _, x, status = integrate_abm_stable(rhs, np.zeros(1), .5, .001, .01,
                                      divergence_norm=.02, use_numba=False)
    assert status == "diverged" and len(x) == 2
    assert calls == 2


def test_cli_rejects_parameter_typo_before_claim(tmp_path):
    output = tmp_path / "bad_parameter"
    command = [sys.executable, "-m", "hidden_attractors.cli.main", "integrate", "compare",
               "--system", "chua-nonsmooth", "--parameters", '{"alhpa": 8.0}',
               "--x0", ".01", ".02", ".03", "--q", ".9", "--h", ".01",
               "--t-final", ".1", "--output-dir", str(output)]
    process = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert process.returncode != 0 and "Unknown parameters" in process.stderr
    assert not output.exists()


@pytest.mark.parametrize("method", ["l1_implicit", "abm_stable"])
def test_yaml_accepts_diagnostics_and_rejects_history_transport(tmp_path, method):
    from hidden_attractors.workflows.config_loader import load_config
    from hidden_attractors.contracts import validate_contracts
    yaml = tmp_path / "diagnostic.yaml"
    base = ("system:\n  system_id: chua-nonsmooth\n  q: 0.9\n"
            f"integrator:\n  name: {method}\n  h: 0.01\n  memory_policy: full_caputo\n")
    yaml.write_text(base, encoding="utf-8")
    cfg = load_config(yaml)
    validate_contracts(cfg, resolved=True)
    assert cfg["integrator"] == method and cfg["memory_mode"] == "full"
    yaml.write_text(base + "stages:\n  continuation: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="do not implement continuation"):
        load_config(yaml)
    yaml.write_text(base + "early_stop:\n  enabled: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="do not support early_stop"):
        load_config(yaml)
    yaml.write_text(base.replace("full_caputo", "finite_window") + "memory_window_length: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="full memory"):
        load_config(yaml)


@pytest.mark.parametrize("method", ["l1_implicit", "abm_stable"])
def test_initial_norm_crossing_precedes_rhs_and_preserves_only_t0(method):
    metadata = {}
    def forbidden_rhs(t, x):
        raise AssertionError("RHS must not be evaluated after the initial threshold event")
    t, x, status = integrate(forbidden_rhs, np.array([1000.]), .5, 1., 1.,
                             integrator=method, divergence_norm=120., diagnostics=metadata,
                             solver_options={"use_numba": False})
    assert status == "diverged"
    np.testing.assert_array_equal(t, [0.])
    np.testing.assert_array_equal(x, [[1000.]])
    assert metadata["nfev"] == metadata["njev"] == metadata["history_terms"] == 0
    assert metadata["completed_steps"] == metadata["final_time"] == 0
    assert not metadata["full_horizon_reached"]


@pytest.mark.parametrize("method", ["l1_implicit", "abm_stable"])
def test_initial_stiff_decay_cannot_hide_already_crossed_norm(method):
    metadata = {}
    t, x, status = integrate(lambda t, x: -100*x, np.array([1000.]), .5, 1., 1.,
                             integrator=method, divergence_norm=120., diagnostics=metadata,
                             solver_options={"use_numba": False})
    assert status == "diverged" and len(t) == len(x) == 1
    assert metadata["nfev"] == 0 and not metadata["full_horizon_reached"]


@pytest.mark.parametrize("method", ["l1_implicit", "abm_stable"])
def test_initial_norm_equal_to_threshold_is_not_crossing(method):
    metadata = {}
    t, x, status = integrate(lambda t, x: np.zeros_like(x), np.array([120.]), .5, .1, .2,
                             integrator=method, divergence_norm=120., diagnostics=metadata,
                             solver_options={"use_numba": False})
    assert status == "ok" and len(t) == 3
    assert metadata["nfev"] > 0 and metadata["full_horizon_reached"]


def test_cli_initial_crossing_skips_overflowing_rhs_including_preflight(tmp_path, monkeypatch):
    from hidden_attractors.cli import integrate as cli_module
    from hidden_attractors.systems import ChaoticSystem
    calls = []
    def forbidden_rhs(x, parameters):
        calls.append(x.copy())
        raise OverflowError("no field evaluation is needed after the t=0 crossing")
    system = ChaoticSystem(name="initial-crossing-test", dimension=1, rhs=forbidden_rhs)
    monkeypatch.setattr(cli_module, "get_system", lambda name: system)
    output = tmp_path / "outside"
    cli_module.compare_integrators(["--system", system.name, "--x0", "1000", "--q", ".5",
                                    "--h", "1", "--t-final", "1", "--no-numba", "--output-dir", str(output)])
    assert calls == []
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert not summary["all_runs_completed"] and len(summary["runs"]) == 2
    for run in summary["runs"]:
        assert run["status"] == "diverged" and run["nfev"] == 0
        assert run["rows"] == 1 and run["final_time"] == 0
        assert not run["full_horizon_reached"]
        with np.load(output / run["trajectory"]) as data:
            np.testing.assert_array_equal(data["t"], [0.])
            np.testing.assert_array_equal(data["x"], [[1000.]])


def test_cli_inside_threshold_still_validates_initial_rhs_before_claim(tmp_path, monkeypatch):
    from hidden_attractors.cli import integrate as cli_module
    from hidden_attractors.systems import ChaoticSystem
    calls = []
    def nonfinite_rhs(x, parameters):
        calls.append(x.copy())
        return np.array([np.nan])
    system = ChaoticSystem(name="initial-inside-test", dimension=1, rhs=nonfinite_rhs)
    monkeypatch.setattr(cli_module, "get_system", lambda name: system)
    output = tmp_path / "inside"
    with pytest.raises(SystemExit) as exc:
        cli_module.compare_integrators(["--system", system.name, "--x0", "1", "--q", ".5",
                                        "--h", "1", "--t-final", "1", "--no-numba", "--output-dir", str(output)])
    assert exc.value.code == 2 and len(calls) == 1
    assert not output.exists()
