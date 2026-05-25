# Chua Integer `q=1` Reference Case

This directory promotes the established integer-order Chua baseline into the
version 2 validation surface. It does not overwrite the fractional validation
contract in `validation/`.

Sources are separated by role:

- numeric artifacts promoted into this directory from the superseded source
  tree before that duplicate tree was removed;
- the supplied theoretical report, registered under
  `08_literature_comparison/sources/`;
- the existing MATLAB reproduction source;
- a published-value comparison against Guan and Xie (2025), Example 6;
- the supplied Wolfram Language symbolic derivation and its recorded
  `wolframscript` output.

MATLAB supplies a numerical recomputation for the harmonic and canonical
checks. The Wolfram source is explicitly symbolic only; it validates the
algebraic derivation without claiming a second trajectory-level reproduction.
The Guan--Xie comparison uses the values displayed to four decimals in the
paper, so its relative differences include publication rounding.

The previously stored integration-dependent outputs were invalidated after the
EFORK-3 third-stage order was compared with the published method and with the
script supplied by Dr. Luis Gerardo de la Fraga. The current continuation,
dynamic, basin, and hiddenness artifacts were regenerated with
`K3 = a31*K1 + a32*K2`. The independent method reproduction is stored in
`validation/reference_cases/efork3_ghoreishi_ghaffari/`.

Run:

```bash
hidden-attractors-check-validation \
  --contract configs/validation_chua_integer_q1.json \
  --validation-root validation/reference_cases/chua_integer_q1
```
