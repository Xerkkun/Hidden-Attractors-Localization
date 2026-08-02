# Chua Integer Lur'e Validation Record

This source-repository validation record demonstrates the complete
integer-order route before fractional Caputo memory is introduced. It is not a
PyPI user example.

## Status

| Item | Value |
| --- | --- |
| Case id | `chua_integer_lure_reference` |
| System | non-smooth Chua saturation |
| Order | `q = 1.0` |
| Role | reproduced software reference/control for the Lur'e route |
| Claim boundary | validates the integer workflow only; it does not validate fractional hiddenness |

Promoted evidence for the corrected integer reference lives under
`validation/reference_cases/chua_integer_q1/`. The hiddenness summary reports no
target-basin hits from sampled equilibrium neighborhoods under that integer
contract.

## Run

From `version_2`:

```bash
python examples/chua_integer_lure_reference/run_example.py --quick
python examples/chua_integer_lure_reference/run_example.py
```

Run selected stages:

```bash
python examples/chua_integer_lure_reference/run_example.py --steps search
python examples/chua_integer_lure_reference/run_example.py --steps search continuation verification figures
```

The configuration is `reproducibility.yaml` in this directory.

## Stages

| Stage | Package functions used | Output role |
| --- | --- | --- |
| `search` | `integer_lure_seed` | Direct integer transfer/DF derivation of `omega0`, `k`, `a0`, and the seed |
| `continuation` | `continue_integer_lure_seed` | Transport from auxiliary Lur'e seed to target system |
| `verification` | `final_integer_lure_attractor`, `run_integer_lure_hiddenness_controls` | Final trajectory and sampled equilibrium-neighborhood controls |
| `figures` | `hidden_attractors.plotting.*` | Phase, transfer, continuation, hiddenness, spectra, and Lyapunov diagnostics |

## System contract

The example requires:

- a registered `ChaoticSystem`;
- equilibria and Jacobian;
- an explicit `LureSystem` split `(P, b, r, psi)`;
- a describing-function convention;
- a numerical contract for integration and neighborhood sampling.

All reusable functions and methods are listed in `docs/api_reference.md`.

## Search hierarchy

The reproducible base case uses `route: direct_integer_transfer`.  It
recomputes the integer Nyquist crossings from the rational transfer function
declared by `(P, b, r)`, solves the registered describing-function relation,
and reconstructs the harmonic seed.  Stored Mathematica/MATLAB values are
regression references, not inputs.

No frequency-grid, biased-transfer, or multiparameter-search configuration is
present in this base example.  Those mechanisms remain library-level,
explicit alternatives for separate workflows when the direct route does not
produce a candidate.
