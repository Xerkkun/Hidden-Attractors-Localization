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
| Dynamical classification | regular attracting set |
| Basin classification | locally consistent with hiddenness for the declared balls through `r = 0.01` |

The Danca 2017 model, parameters, equilibria, stability classification, and
full-history ABM contract are retained as an independent bibliographic control.
The target set in this example originates from the proposed biased describing
function search and memory-preserving Caputo continuation.

## Run

From this directory:

```bash
python run_example.py --quick
python run_example.py
python run_example.py --all
python run_example.py --steps 1 2
```

From `version_2`:

```bash
python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick
python tools/rerun_paper07_nonsmooth_hiddenness.py
```

`--quick` is a smoke check. The default command runs the two localization
stages that produce the reported seed and causal continuation. `--all`
explicitly enables the validation stages and can take hours. The Paper 07
runner is the canonical end-to-end reconstruction used for the reported
candidate and writes to `../outputs/paper07_nonsmooth_corrected`. Its
volumetric survey processes radii in ascending order, completes all three
equilibrium neighborhoods at the first radius with contacts, and omits every
larger radius.

## Stages

| Stage | Meaning |
| --- | --- |
| 1 | Centered describing-function baseline |
| 2 | Biased describing-function root search, seed reconstruction, continuation, final simulation |
| 3 | Standard sampled neighborhood hiddenness check |
| 4 | Radius-major extended sampling with first-contact stopping |
| 5 | Summary and figure gallery |

Step wrappers (`01_search.py` through `04_plot.py`) are thin convenience entry
points. `run_example.py` is the official orchestrator.

## Methodological boundary

BDF, Nyquist, and continuation are seed/candidate tools. They do not prove
hiddenness. A promoted hiddenness label requires all-equilibrium local
neighborhood or basin evidence under the declared Caputo memory, step size,
horizon, radii, samples, and classifier thresholds. Extended-radius contacts
must be reported as basin-geometry audits unless the stored local-radius
contract itself records disqualifying contact.

## Paper 07 recorded result

The canonical volumetric output is
`../outputs/paper07_nonsmooth_corrected/extended_first_contact_clean`.
It contains 17,400 completed probes and no numerical failures. The first
13,800 probes, through `r = 0.1`, produced no target contacts; this includes
7,200 probes in the declared local region through `r = 0.01`. The first
sampled radius intersecting the target basin was `r = 0.3`, where all 3,600
planned probes were completed and 37 contacts were recorded. No probe at
`r = 1` or `r = 2` belongs to this stopped experiment.
