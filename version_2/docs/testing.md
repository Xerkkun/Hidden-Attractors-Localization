# Testing

This document details the validation and testing suite for `hidden-attractors-fo`.

## Running the Test Suite

The test suite requires the package installed with developer extra dependencies. Run the tests from the `version_2/` directory:

```bash
python -m pytest -q
```

### 1. Repository Cleanliness and Hygiene Checks

To run fast repository cleanliness checks (which verify formatting, docstring policies, file naming constraints, and entry point rules):

```bash
python -m pytest -q -m "hygiene"
```

### 2. Packaging and Release Readiness Checks

To check if the package meets release metadata requirements and passes smoke checks on sample configurations:

```bash
python -m pytest -q -m "release_readiness"
```

### 3. Syntax Verification / Compilation Check

To check that the entire code compiles cleanly without syntax errors:

```bash
python -m compileall hidden_attractors examples tests tools/cli
```

---

## Folder Roles in Testing

- **`tests/`**: Contains the automated tests files.
- **`tools/cli/`**: Houses internal helper script templates, not public entry points. Their behavior is verified as internal utilities by the tests but they are not public command boundaries.
- **`tools/legacy/`**: Preserves historical C solver compatibility material. It is strictly internal and excluded from regular active public testing execution.

## Boundary of validation tests

The test suite asserts that the Chua arctan c590 candidate is promoted under the local-radius validation contract (`r <= 0.3`, 8400 probes, zero contacts). The testing boundaries check this local neighborhood and do not assume a global basin proof. Any change to the testing parameters must ensure the local claims contract remains auditable in `THESIS_CLAIMS.md`.
