# Chua Integer Lur'e Reference Example

This is the first official report example. It demonstrates the complete
integer-order route before fractional Caputo memory is introduced.

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
| `search` | `integer_lure_seed` | DF/Nyquist seed only |
| `continuation` | `continue_integer_lure_seed` | Transport from auxiliary Lur'e seed to target system |
| `verification` | `final_integer_lure_attractor`, `run_integer_lure_hiddenness_controls` | Final trajectory and sampled equilibrium-neighborhood controls |
| `figures` | `hidden_attractors.plotting.*` | Phase, transfer, continuation, hiddenness, spectra, and Lyapunov diagnostics |

## Reuse for new systems

A new integer Lur'e system can follow this example when it provides:

- a registered `ChaoticSystem`;
- equilibria and Jacobian;
- an explicit `LureSystem` split `(P, b, r, psi)`;
- a describing-function convention;
- a numerical contract for integration and neighborhood sampling.

All reusable functions and methods are listed in `docs/api_reference.md`.
