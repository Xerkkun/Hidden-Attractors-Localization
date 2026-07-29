# Wu2023 arctan-Chua bibliographic validation

This directory reproduces a fixed bibliographic validation record for the
arctan-Chua equations reported by Wu et al. It checks the algebraic model,
equilibria, Lur'e split, the reported initial conditions, and the local ADM
recurrence under the recorded contract.

The completed result is deliberately narrow: the reported initial conditions
are periodic/nonchaotic under that local ADM contract. This record does not
claim a full-memory Caputo reproduction, chaos, or hiddenness.

From `version_2`, rebuild the algebra and seed checks with:

```bash
python examples/chua_arctan_wu2023/run_example.py
```

To also execute the fixed reported-initial-condition checks:

```bash
python examples/chua_arctan_wu2023/run_example.py \
  --run-published-trajectories
```

The fixed contract and generated record are stored in:

- `reproducibility.yaml`;
- `validation/reference_cases/fractional_chua_arctan_wu2023/config.json`;
- `validation/reference_cases/fractional_chua_arctan_wu2023/validation_summary.json`.

The separate c590 assessment is retained only as a closed, radius-limited
validation record under
`validation/chua_fractional_arctan_c590/validation_summary.json`. Its
finite-neighborhood result is not a global basin proof.
