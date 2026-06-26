# Non-Smooth Fractional Chua BDF Example

This is the official fractional non-smooth Chua methodology example. It uses a
biased describing function (BDF) to generate candidate seeds, transports them by
continuation, and runs finite neighborhood checks.

## Status

| Item | Value |
| --- | --- |
| Case id | `chua_nonsmooth_biased_hidden_attractor` |
| System | non-smooth Chua saturation |
| Typical order | `q = 0.9998` |
| Role | proposed biased-DF methodology lane |
| Claim boundary | candidate/compatible only under declared local radii; not a full Danca 2017 trajectory reproduction |

The Danca 2017 article does not report all values needed for a complete
independent trajectory reproduction: exact DF frequency/gain/amplitude, seed
coordinates, hidden-attractor initial condition, and continuation details are
missing.

The official nearby candidate stored in the current validation package,
`danca2017_nearby_saturation_candidate_q09998`, is rejected under the current
contract because `validation/09_hiddenness_tests/hiddenness_tests_validation_summary.json`
records 1305 target contacts from neighborhoods of `E+` and `E-`.

## Run

From this directory:

```bash
python run_example.py --quick
python run_example.py
python run_example.py --all
python run_example.py --steps 1 2 3 5
```

From `version_2`:

```bash
python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick
```

`--quick` is a smoke check. `--all` includes the extended multiprocessing
hiddenness stage and can take hours.

## Stages

| Stage | Meaning |
| --- | --- |
| 1 | Centered describing-function baseline |
| 2 | Biased describing-function root search, seed reconstruction, continuation, final simulation |
| 3 | Standard sampled neighborhood hiddenness check |
| 4 | Extended hiddenness sampling |
| 5 | Summary and figure gallery |

Step wrappers (`01_search.py` through `04_plot.py`) are thin convenience entry
points. `run_example.py` is the official orchestrator.

## Methodological boundary

BDF, Nyquist, and continuation are seed/candidate tools. They do not prove
hiddenness. A promoted hiddenness label requires all-equilibrium neighborhood or
basin evidence under the declared Caputo memory, step size, horizon, radii,
samples, and classifier thresholds.
