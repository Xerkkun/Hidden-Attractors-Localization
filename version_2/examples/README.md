# Examples

This directory contains small public API examples and explicitly labelled
validation/search references. The API examples demonstrate reusable library
features; each validation reference retains its own numerical and evidence
boundary.

## Validation and search references

Every maintained reference directory is listed here. These executions
reproduce bounded software-validation or search records; they do not broaden
the scientific claims stored with those records.

Run from `version_2`:

```bash
python examples/chua_arctan_wu2023/run_example.py
python examples/chua_integer_lure_reference/run_example.py --quick
python examples/kalman_fitts_integer_lure_reference/run_example.py --quick
python examples/modified_van_der_pol_duffing_integer_lure_audit/run_example.py
python examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/run_example.py
python examples/pll_lead_lag_integer_lure_reference/run_example.py --quick
```

| Validation record | Role | Evidence boundary |
| --- | --- | --- |
| `chua_arctan_wu2023/` | Fractional arctan-Chua bibliographic algebra and local-ADM audit | The reported initial conditions are periodic/nonchaotic under the stored local contract; no full-memory Caputo, chaos, or hiddenness claim |
| `chua_integer_lure_reference/` | Integer non-smooth Chua `q=1` Lur'e reference: seed, continuation, trajectory, neighborhood controls, figures | Reproduced integer Chua control only |
| `kalman_fitts_integer_lure_reference/` | Direct-route incompatibility followed by the explicit switching-map and `sign`-to-`tanh` continuation | Finite hidden-periodic candidate reproduced in Python and Julia; no global basin proof |
| `modified_van_der_pol_duffing_integer_lure_audit/` | Direct MAVPD branches at `xi=3.1` plus the `xi=3.5` negative audit | Finite periodic-hiddenness audit reproduced in Python and Julia; it is not evidence for the separate chaotic point |
| `modified_van_der_pol_duffing_integer_hidden_chaos_search/` | Direct base route, finite chaos-screen trigger, Hopf-relative parameter continuation, strict diagnostics, and equilibrium-neighborhood controls | Label `chaotic_hidden_under_tested_neighborhoods` reproduced independently in Python and Julia under 112 finite probes; no global basin proof |
| `pll_lead_lag_integer_lure_reference/` | Analytic direct-route rejection and Andronov continuation on a cylindrical state space | Python--Julia `quick` and `full` comparisons approved under the same 96 finite probes; no global basin proof |

For the hidden-chaos search, neither direct base branch exceeds the declared
finite-time Lyapunov screening threshold. That observation only enables the
separately declared parameter-continuation alternative; it is not a proof that
either base trajectory has a particular asymptotic classification. The
selected `xi=2.85` parameter point is a local numerical result, not a tuple
attributed to the cited MAVPD article.

## Small API examples

```bash
python examples/quickstart_equilibria.py
python examples/minimal_chua_protocol.py
python examples/custom_system_definition.py
python examples/new_system_workflow_spec.py
python examples/integer_lure_chua_protocol.py
python examples/dynamical_analysis_gallery.py
python examples/fractional_core_catalog.py
python examples/caputo_hadamard_chua.py
python examples/chua_advanced_analysis.py
python examples/correlation_dimension_integer_fractional.py
python examples/permutation_entropy_integer_fractional.py
python examples/multi_term_caputo_relaxation.py
python examples/tempered_convolution_quadrature.py
python examples/tempered_fast_history_chua.py
```

`minimal_chua_protocol.py` writes the explicit command and JSON contract by
default. Add `--run` only when launching the numerical protocol intentionally.

`dynamical_analysis_gallery.py` accepts `--trajectory-csv path/to/trajectory.csv`
for plotting an existing trajectory.

`fractional_core_catalog.py` exercises the distinct sampled-operator contracts
(GL, RL, tempered, variable/distributed order, conformable, Caputo--Fabrizio,
Lubich CQ and Hadamard CQ) and finite manufactured GL, conformable, ABC,
tempered-Caputo, variable-order Caputo Type III and distributed-order Caputo L1
solvers. `caputo_hadamard_chua.py` runs a
short experimental Chua IVP on a logarithmic grid and compares it with the
integer `q=1` log-coordinate limit. `chua_advanced_analysis.py` applies delay
selection and advanced RQA to an integer Chua trajectory, and basin metrics to
an exactly classified bistable flow. All three report finite numerical
diagnostics only; none certifies chaos, attraction or hiddenness.

`correlation_dimension_integer_fractional.py` applies one explicit q=2
correlation-sum contract to a small RK4 trajectory with `q=1` and a Caputo
ABM--PECE trajectory with `q=0.85`. Its fitted slopes describe only the finite
standardized projections; the analysis API itself also accepts trajectories
from the other fractional definitions implemented by HAFO.

`multi_term_caputo_relaxation.py` applies the semantic finite-sum facade to a
forced multiscale relaxation with an affine manufactured trajectory. Its
coefficients sum to `1.85` and are not normalized; repeated orders are grouped
without rebuilding the distributed-order L1 kernel. The finite error is a
consistency check, not a general convergence, stability, chaos, attraction or
hiddenness result.

`tempered_convolution_quadrature.py` constructs a two-component nonlinear
manufactured equation with distinct (q_i) and (lambda_i), evaluates its
conjugated tempered-Caputo left side with FFT-BDF2, and reports finite-grid
residuals. It is a sampled-operator example, not an implicit CQ solver or a
chaos/attraction/hiddenness test.

`tempered_fast_history_chua.py` obtains a uniformly sampled integer-order Chua
trajectory and applies componentwise conjugated tempered-Caputo GNGF2 as a
sampled-history postprocessor. It compares the recurrent output with an
independent direct convolution and reports the finite-grid compression
calibration. The source trajectory does not become a fractional Chua solution;
the example supplies no chaos, attraction, basin, or hiddenness evidence.

The maintained interactive entry point is
`notebooks/hidden_attractors_quickstart.ipynb`. It demonstrates finite-time
trajectory characterization on synthetic data and does not certify chaos or
hiddenness.

## Rules

When adding an example:

1. import from `hidden_attractors` whenever the name belongs to
   `PUBLIC_API_STABLE` or `PUBLIC_API_EXPERIMENTAL`;
2. register new systems through `hidden_attractors.systems`;
3. record a `WorkflowInputSpec` before presenting reusable workflows;
4. write outputs under `outputs/` or require `--output-dir`;
5. document whether the run is a smoke check, a long job, a diagnostic, or a validation helper;
6. keep module-qualified experimental helpers visibly separate from the
   top-level public API;
7. update `docs/api_reference.md` when new functions, classes, or methods become part of the release surface.
