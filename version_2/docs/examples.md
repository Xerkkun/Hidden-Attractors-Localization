# Examples

The public example surface consists of small, runnable API entry points that
import from `hidden_attractors`.

## Completed reference validation

The source distribution includes the completed integer Chua `q=1` software
reference:

```bash
cd version_2
python examples/chua_integer_lure_reference/run_example.py --quick
```

It exercises the integer seed, continuation, integration, diagnostics, and
sampled equilibrium-neighborhood controls. Its result is finite numerical
software validation, not a global hiddenness proof.

## Included API examples

```bash
python examples/quickstart_equilibria.py
python examples/minimal_chua_protocol.py
```

`minimal_chua_protocol.py` writes a claim-free schema example and command by
default. Add `--run` only when you intend to launch the numerical workflow.

Independent trajectory and time-series characterization is documented with
complete API examples in [Dynamical Analysis](dynamical_analysis.md), including
trajectory metrics, spectra, Poincare sections, bifurcation post-processing,
complexity measures, and time-series Lyapunov estimation.
