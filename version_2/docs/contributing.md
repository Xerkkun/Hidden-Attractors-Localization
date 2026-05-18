# Contributing

## Development Setup

```bash
python -m pip install -e ".[dev]"
python -m compileall hidden_attractors examples tests tools/cli
python -m pytest -q
```

## Adding Library Code

- Put reusable model code in `hidden_attractors/models/`.
- Put reusable diagnostics in `hidden_attractors/analysis/`.
- Put reusable workflow orchestration in `hidden_attractors/workflows/`.
- Put C backend sources in `hidden_attractors/native/csrc/`.
- Keep command wrappers thin and place them in `tools/cli/`.

## Adding Examples

Examples belong in `examples/`. They should be short and runnable from the
repository root.

## Migrating Legacy Scripts

When a legacy script becomes important:

1. extract reusable functions into `hidden_attractors/`;
2. add tests or a smoke example;
3. leave a thin wrapper in `tools/cli/` if the command name matters;
4. document the numerical contract and outputs.

## Scientific Standards

Do not report hiddenness without the equilibrium-neighborhood and basin
evidence required by the workflow. Always report the numerical contract.
