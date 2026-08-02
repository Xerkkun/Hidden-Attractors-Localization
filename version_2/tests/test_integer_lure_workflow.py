"""Tests for reusable integer-order Lur'e workflow pieces."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from hidden_attractors.analysis import integer_system_lyapunov_exponents
from hidden_attractors.plotting import (
    plot_integer_hiddenness_controls,
    plot_integer_lure_continuation,
    plot_lure_nyquist_describing_function,
    plot_lure_transfer_components,
)
from hidden_attractors.seed_generation import (
    find_integer_lure_omega_gain_candidates_direct,
    find_lure_omega_gain_candidates,
)
from hidden_attractors.systems import ChaoticSystem, get_system
import hidden_attractors.workflows.integer_lure as integer_lure_module
from hidden_attractors.workflows.contracts import (
    FullWorkflowContract,
    validate_full_workflow_system,
)
from hidden_attractors.workflows.protocol import ContinuationPlan
from hidden_attractors.workflows.integer_lure import (
    continue_integer_lure_seed,
    final_integer_lure_attractor,
    integer_lure_seed,
    run_integer_lure_hiddenness_controls,
    summarize_integer_hiddenness_controls,
)


import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_integer_reference_example_uses_only_the_direct_primary_route() -> None:
    config_path = ROOT / "examples" / "chua_integer_lure_reference" / "reproducibility.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed_search = config["seed_search"]
    assert seed_search["route"] == "direct_integer_transfer"
    assert seed_search["fallback_route"] is None
    assert "nscan" not in seed_search
    assert "alternative_frequency_scan" not in seed_search

@pytest.mark.unit
def test_builtin_chua_has_required_lure_form() -> None:
    system = get_system("chua-nonsmooth")

    assert system.lure is not None
    pairs = find_integer_lure_omega_gain_candidates_direct(system.lure, wmax=50.0)

    assert pairs
    assert pairs[0][0] == pytest.approx(2.039186939959001, abs=1.0e-11)
    assert pairs[0][1] == pytest.approx(0.20986735451508398, abs=1.0e-11)


@pytest.mark.unit
def test_frequency_scan_remains_an_explicit_alternative() -> None:
    system = get_system("chua-nonsmooth")
    assert system.lure is not None
    pairs = find_lure_omega_gain_candidates(1.0, system.lure, nscan=1500, wmax=50.0)
    assert pairs


@pytest.mark.unit
def test_default_integer_seed_route_does_not_call_frequency_scan(monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("frequency scan must not run in the primary integer route")

    monkeypatch.setattr(integer_lure_module, "find_lure_harmonic_seed", fail_if_called)
    seed = integer_lure_module.integer_lure_seed(get_system("chua-nonsmooth"))
    assert seed.search_route == "direct_integer_transfer"
    assert seed.omega == pytest.approx(2.039186939959001, abs=1.0e-11)


@pytest.mark.integration
def test_integer_lure_seed_and_short_continuation_are_reusable(tmp_path) -> None:
    outdir = tmp_path / "integer_lure_seed"
    outdir.mkdir(parents=True, exist_ok=True)
    system = get_system("chua-nonsmooth")
    seed = integer_lure_seed(system, wmax=50.0)
    steps = continue_integer_lure_seed(
        system,
        seed,
        plan=ContinuationPlan((0.0, 0.5, 1.0), {"internal_parameter": "epsilon"}),
        t_transient=0.05,
        t_keep=0.05,
        h=0.01,
        div_threshold=120.0,
    )

    assert seed.seed.shape == (3,)
    assert steps
    assert [step.lambda_value for step in steps] == [0.0, 0.5, 1.0]
    assert steps[-1].provenance["mapping"]["internal_parameter"] == "epsilon"
    assert steps[-1].x_out.shape == (3,)

    plot_lure_nyquist_describing_function(system.lure, seed, outdir / "nyquist.png", q=1.0)
    plot_lure_transfer_components(system.lure, seed, outdir / "transfer_components.png", q=1.0)
    plot_integer_lure_continuation(steps, outdir / "continuation.png")

    assert (outdir / "nyquist.png").exists()
    assert (outdir / "transfer_components.png").exists()
    assert (outdir / "continuation.png").exists()


@pytest.mark.integration
def test_integer_hiddenness_controls_and_lyapunov_smoke(tmp_path) -> None:
    outdir = tmp_path / "integer_lure_hiddenness"
    outdir.mkdir(parents=True, exist_ok=True)
    system = get_system("chua-nonsmooth")
    seed = integer_lure_seed(system, wmax=50.0)
    _target_seed, trajectory, status = final_integer_lure_attractor(
        system,
        seed.seed,
        t_burn=0.05,
        t_keep=0.10,
        h=0.01,
        div_threshold=120.0,
    )
    probes = run_integer_lure_hiddenness_controls(
        system,
        trajectory,
        radii=(1.0e-4,),
        samples_per_radius=1,
        t_final=0.05,
        t_burn=0.0,
        h=0.01,
        target_cloud_tol=1.0,
        random_seed=3,
    )
    summary = summarize_integer_hiddenness_controls(probes)
    lyap = integer_system_lyapunov_exponents(
        system,
        seed.seed,
        h=0.01,
        t_final=0.05,
        t_burn=0.0,
        reorthonormalize_every=1,
        div_threshold=120.0,
    )

    assert status == "ok"
    assert summary["n_probes"] == 3
    assert summary["sampling_modes"] == ["ball"]
    assert all(probe.distance_from_equilibrium <= probe.radius for probe in probes)
    assert lyap.exponents.shape == (3,)

    plot_integer_hiddenness_controls(trajectory, probes, outdir / "hiddenness.png")
    assert (outdir / "hiddenness.png").exists()


@pytest.mark.unit
def test_full_workflow_rejects_system_without_lure() -> None:
    system = ChaoticSystem(
        name="rhs-only",
        dimension=1,
        rhs=lambda state, _p: np.array([-state[0]], dtype=float),
        equilibria=lambda _p: {"E0": np.array([0.0])},
    )
    workflow = FullWorkflowContract(
        seed_generator=lambda *_args, **_kwargs: None,  # type: ignore[arg-type]
        continuation=lambda *_args, **_kwargs: None,  # type: ignore[arg-type]
        hiddenness_verifier=lambda *_args, **_kwargs: None,  # type: ignore[arg-type]
        basin_classifier=lambda *_args, **_kwargs: {},  # type: ignore[arg-type]
        report_writer=lambda *_args, **_kwargs: {},  # type: ignore[arg-type]
    )

    try:
        validate_full_workflow_system(system, workflow)
    except ValueError as exc:
        assert "Lur'e" in str(exc)
    else:
        raise AssertionError("system without Lur'e form should not validate")
