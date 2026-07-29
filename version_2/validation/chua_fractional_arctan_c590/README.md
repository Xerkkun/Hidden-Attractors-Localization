# Chua fractional arctan c590 validation

This directory reports the c590 arctan candidate as one radius-limited finite
hiddenness-evidence lane, matching the same conservative convention used for
other methodology examples.

## Status

`hiddenness_supported_under_tested_local_radii_with_macro_radius_review`

The local claim is limited to tested radii through `0.3`.
Those local radii contain `8400` probes around all equilibria and `0` target
contacts. Of these, `8396` reached the recorded full horizon (`status=ok`) and
`4` met the recorded equilibrium-convergence criterion early
(`status=converged_equilibrium_early`). Every stored local row has
`finite=True`.

## Extended-radius audit

The extended macroscopic radii are retained as review evidence, not as a reason
to erase the local-radius claim:

| Radius | Tests | Contacts | `ok` | Early equilibrium | Divergence threshold | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `1e-05` | 300 | 0 | 296 | 4 | 0 | `no_contact_detected` |
| `0.0001` | 600 | 0 | 600 | 0 | 0 | `no_contact_detected` |
| `0.001` | 900 | 0 | 900 | 0 | 0 | `no_contact_detected` |
| `0.01` | 1200 | 0 | 1200 | 0 | 0 | `no_contact_detected` |
| `0.03` | 1500 | 0 | 1500 | 0 | 0 | `no_contact_detected` |
| `0.1` | 1800 | 0 | 1800 | 0 | 0 | `no_contact_detected` |
| `0.3` | 2100 | 0 | 2100 | 0 | 0 | `no_contact_detected` |
| `1` | 2400 | 22 | 2400 | 0 | 0 | `macro_radius_contact_detected` |
| `2` | 2700 | 588 | 2569 | 0 | 131 | `macro_radius_contact_detected` |


Contacts occur only at radii `1, 2`.
The total extended-radius contact count is `610` out of
`5100` macro-radius probes. At `r=2`, `2569` trajectories reached the full
horizon and `131` reached the declared divergence threshold. The `finite`
column remains `True` for those threshold events because the last stored
states are finite; the `status` column records why integration stopped.

## Algebraic validation of c590

The Wolfram Language case
[`chua_fractional_arctan_c590.wl`](../wolfram/cases/chua_fractional_arctan_c590.wl)
evaluates the exact c590 nonlinearity

\[
h(x)=a_1x+a_2\arctan(\rho x),\qquad
\rho=1.7984259332820332,
\]

instead of reusing the bibliographic arctangent case, for which \(\rho=1\).
The canonical decimals enter Wolfram as exact rational numbers before any
high-precision numerical evaluation.
It verifies the Lur'e representation, transfer-function identity, Jacobian
identity, three equilibria, equilibrium residuals, the closed-form describing
function, and the local Matignon classification at \(q=0.9999\). The recorded
fractional seed is checked against the canonical c590 contract and the Python
right-hand side, while its provenance remains
`bounded_integer_search_and_independent_caputo_refinement`; it is not labeled
as a seed produced by the describing function.

The portable snapshot in [`algebraic_validation/`](algebraic_validation/)
contains all ten passing Wolfram tests and the passing Wolfram--Python
comparison. Regenerate it from the repository root with:

```powershell
python validation/python/run_wolfram_validations.py `
  --case validation/wolfram/cases/chua_fractional_arctan_c590.wl `
  --out validation/outputs/wolfram
```

The algebraic result validates the c590 equations and the recorded
parameter/seed contract. Dynamic integration, post-transient behavior, and
radius-limited hiddenness evidence are validated by their separate numerical
lanes.

## Rebuild

Regenerate the JSON and all derived CSV summaries without rerunning any
simulation:

```powershell
python tools/summarize_c590_hiddenness.py
```

The six complete `hiddenness_*_rows.csv` tables and their
`*_run_config.json` contracts are retained in this tracked directory, so the
default rebuild does not depend on ignored `outputs/`. The script groups the
evidence by radius, equilibrium, status, contact, and finite-state flag, and
updates the directly derived public package in
`validation/chua_fractional_arctan/`.

## Boundary

This is finite deterministic neighborhood evidence under the recorded Caputo
ABM full-memory contract. It is not a filled-ball proof and not a global basin
proof. The Wu2023 bibliographic ADM lane remains separate and non-promoted as a
Caputo hiddenness validation.
