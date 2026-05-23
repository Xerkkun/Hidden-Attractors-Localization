"""Tests for reusable integer-order Lur'e workflow pieces."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hidden_attractors.analysis import integer_system_lyapunov_exponents
from hidden_attractors.plotting import (
    plot_integer_hiddenness_controls,
    plot_integer_lure_continuation,
    plot_lure_nyquist_describing_function,
    plot_lure_transfer_components,
)
from hidden_attractors.seed_generation import find_lure_omega_gain_candidates
from hidden_attractors.systems import ChaoticSystem, get_system
from hidden_attractors.workflows.contracts import (
    FullWorkflowContract,
    validate_full_workflow_system,
)
from hidden_attractors.workflows.integer_lure import (
    continue_integer_lure_seed,
    final_integer_lure_attractor,
    integer_lure_seed,
    run_integer_lure_hiddenness_controls,
    summarize_integer_hiddenness_controls,
)


def test_builtin_chua_has_required_lure_form() -> None:
    system = get_system("chua-nonsmooth")

    assert system.lure is not None
    pairs = find_lure_omega_gain_candidates(1.0, system.lure, nscan=1500, wmax=50.0)

    assert pairs


def test_integer_lure_seed_and_short_continuation_are_reusable() -> None:
    outdir = Path("outputs/test_artifacts/integer_lure_seed")
    outdir.mkdir(parents=True, exist_ok=True)
    system = get_system("chua-nonsmooth")
    seed = integer_lure_seed(system, nscan=1500, wmax=50.0)
    steps = continue_integer_lure_seed(
        system,
        seed,
        eps_values=(0.5, 1.0),
        t_transient=0.05,
        t_keep=0.05,
        h=0.01,
        div_threshold=120.0,
    )

    assert seed.seed.shape == (3,)
    assert steps
    assert steps[-1].x_out.shape == (3,)

    plot_lure_nyquist_describing_function(system.lure, seed, outdir / "nyquist.png", q=1.0)
    plot_lure_transfer_components(system.lure, seed, outdir / "transfer_components.png", q=1.0)
    plot_integer_lure_continuation(steps, outdir / "continuation.png")

    assert (outdir / "nyquist.png").exists()
    assert (outdir / "transfer_components.png").exists()
    assert (outdir / "continuation.png").exists()


def test_integer_hiddenness_controls_and_lyapunov_smoke() -> None:
    outdir = Path("outputs/test_artifacts/integer_lure_hiddenness")
    outdir.mkdir(parents=True, exist_ok=True)
    system = get_system("chua-nonsmooth")
    seed = integer_lure_seed(system, nscan=1500, wmax=50.0)
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
    assert lyap.exponents.shape == (3,)

    plot_integer_hiddenness_controls(trajectory, probes, outdir / "hiddenness.png")
    assert (outdir / "hiddenness.png").exists()


def test_full_workflow_rejects_system_without_lure() -> None:
    system = ChaoticSystem(
        name="rhs-only",
        dimension=1,
        rhs=lambda state, _p: np.array([-state[0]], dtype=float),
        equilibria=lambda _p: {"E0": np.array([0.0])},
    )
    workflow = FullWorkflowContract(
        seed_generator=lambda *_args, **_kwargs: None,  # type: ignore[arg-type]
        machado_seed_generator=lambda *_args, **_kwargs: None,  # type: ignore[arg-type]
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
