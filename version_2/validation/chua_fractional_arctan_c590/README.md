# Chua fractional arctan c590 validation

This directory contains the fixed row-level evidence for one radius-limited
arctan-Chua validation record.

## Numerical result

| Radius | Tests | Contacts | Full horizon | Early equilibrium | Divergence threshold | Role |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `1e-05` | 300 | 0 | 296 | 4 | 0 | local |
| `0.0001` | 600 | 0 | 600 | 0 | 0 | local |
| `0.001` | 900 | 0 | 900 | 0 | 0 | local |
| `0.01` | 1200 | 0 | 1200 | 0 | 0 | local |
| `0.03` | 1500 | 0 | 1500 | 0 | 0 | local |
| `0.1` | 1800 | 0 | 1800 | 0 | 0 | local |
| `0.3` | 2100 | 0 | 2100 | 0 | 0 | local |
| `1` | 2400 | 22 | 2400 | 0 | 0 | macro audit |
| `2` | 2700 | 588 | 2569 | 0 | 131 | macro audit |

The local contract therefore contains `8,400` finite probes around all
equilibria and `0` target contacts through `r = 0.3`. The two macro radii
contain `610` contacts in `5,100` probes.

## Algebraic validation

The Wolfram case
[`chua_fractional_arctan_c590.wl`](../wolfram/cases/chua_fractional_arctan_c590.wl)
checks the exact c590 nonlinearity, Lur'e representation, transfer-function
identity, Jacobian identity, three equilibria, residuals, closed-form
describing function, and local Matignon classification at `q = 0.9999`.
The portable snapshot in `algebraic_validation/` contains ten passing Wolfram
checks and the Wolfram--Python comparison.

## Fixed artifacts

- `validation_summary.json`
- `summary_by_radius.csv`
- `summary_by_radius_equilibrium.csv`
- `summary_by_radius_equilibrium_status_contact.csv`
- six row-level `hiddenness_*_rows.csv` tables
- six matching `*_run_config.json` numerical contracts

The aggregate JSON and CSV projections can be checked from the fixed row-level
tables with:

```powershell
python validation/paper07_chua/scripts/summarize_c590_hiddenness.py
```

## Boundary

This is finite deterministic neighborhood evidence under the recorded Caputo
ABM full-memory contract. It is not a global basin proof, and it does not
certify chaos. The macro-radius observations remain separate from the local
claim.
