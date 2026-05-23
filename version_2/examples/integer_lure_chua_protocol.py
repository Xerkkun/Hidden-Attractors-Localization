"""Small integer-order Lur'e workflow using the built-in Chua system.

The same calls are intended for any user system that provides:

- a vector field and equilibria through ``ChaoticSystem``;
- a manual Lur'e form ``A, b, c, psi(c^T x)``;
- classical and Machado describing functions;
- continuation and hiddenness settings appropriate for the study.
"""

from __future__ import annotations

from pathlib import Path

from hidden_attractors import get_system
from hidden_attractors.analysis import integer_system_lyapunov_exponents
from hidden_attractors.plotting import (
    plot_integer_hiddenness_controls,
    plot_integer_lure_continuation,
    plot_lyapunov_convergence,
    plot_lure_nyquist_describing_function,
    plot_lure_transfer_components,
    plot_phase_projections,
    plot_phase_space,
    plot_trajectory_spectra,
)
from hidden_attractors.workflows.integer_lure import (
    continue_integer_lure_seed,
    final_integer_lure_attractor,
    integer_lure_seed,
    run_integer_lure_hiddenness_controls,
    summarize_integer_hiddenness_controls,
)


def main() -> None:
    outdir = Path("outputs/examples/integer_lure_chua")
    outdir.mkdir(parents=True, exist_ok=True)

    system = get_system("chua-nonsmooth")
    seed = integer_lure_seed(system, nscan=3000, wmax=50.0)
    steps = continue_integer_lure_seed(
        system,
        seed,
        eps_values=(0.25, 0.5, 0.75, 1.0),
        t_transient=2.0,
        t_keep=2.0,
        h=0.01,
        div_threshold=120.0,
    )
    final_state = steps[-1].x_out
    target_seed, trajectory, status = final_integer_lure_attractor(
        system,
        final_state,
        t_burn=2.0,
        t_keep=4.0,
        h=0.01,
        div_threshold=120.0,
    )
    probes = run_integer_lure_hiddenness_controls(
        system,
        trajectory,
        radii=(1.0e-4,),
        samples_per_radius=2,
        t_final=2.0,
        t_burn=0.5,
        h=0.01,
        target_cloud_tol=0.20,
        random_seed=7,
    )
    lyap = integer_system_lyapunov_exponents(
        system,
        target_seed,
        h=0.01,
        t_final=1.0,
        t_burn=0.2,
        reorthonormalize_every=5,
        div_threshold=120.0,
    )

    plot_lure_nyquist_describing_function(system.lure, seed, outdir / "integer_lure_nyquist.png", q=1.0)
    plot_lure_transfer_components(system.lure, seed, outdir / "integer_lure_transfer_components.png", q=1.0)
    plot_integer_lure_continuation(steps, outdir / "integer_lure_continuation.png")
    plot_phase_space(trajectory, outdir / "integer_lure_attractor.png", title="Integer Lur'e attractor")
    plot_phase_projections(trajectory, outdir / "integer_lure_projections.png", title="Integer Lur'e projections")
    plot_integer_hiddenness_controls(trajectory, probes, outdir / "integer_lure_hiddenness_controls.png")
    plot_trajectory_spectra(trajectory, outdir, method="fft", prefix="integer_lure")
    plot_trajectory_spectra(trajectory, outdir, method="psd", prefix="integer_lure")
    plot_lyapunov_convergence(lyap, outdir / "integer_lure_lyapunov_convergence.png")

    print(f"seed={seed.seed.tolist()}")
    print(f"final_status={status}")
    print(f"hiddenness={summarize_integer_hiddenness_controls(probes)}")
    print(f"lyapunov_status={lyap.status}")
    print(f"lyapunov={lyap.exponents.tolist()}")
    print(f"figures={outdir}")


if __name__ == "__main__":
    main()
