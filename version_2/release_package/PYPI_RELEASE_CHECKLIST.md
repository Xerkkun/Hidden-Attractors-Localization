# PyPI release checklist

This checklist prepares `hidden-attractors-fo` for TestPyPI and PyPI without storing tokens or publishing automatically from local automation.

## 1. Verify the scientific release

Run from `version_2`:

```bash
python -m pytest -q
hidden-attractors validate contract
hidden-attractors validate bibliography --strict
hidden-attractors validate release-readiness --submission-strict --json
```

Do not modify parameters, seeds, tolerances, classifiers, evidence JSON/CSV, figures, or scientific conclusions to force these checks to pass.

## 2. Build the distribution

```bash
rm -rf dist build *.egg-info
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
```

On Windows PowerShell, use:

```powershell
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
```

## 3. Test local wheel installation

```bash
python -m venv .venv_wheel_test
# Windows
.venv_wheel_test\Scripts\activate
# Linux/macOS
# source .venv_wheel_test/bin/activate

python -m pip install --upgrade pip
python -m pip install dist/*.whl
hidden-attractors --help
hidden-attractors seed --help
python -c "import hidden_attractors; print('import ok')"
```

The seed help must not expose Machado/FDF routes.

## 4. Upload to TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

## 5. Install from TestPyPI

```bash
python -m venv .venv_testpypi
.venv_testpypi\Scripts\activate
python -m pip install --upgrade pip
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps hidden-attractors-fo==1.0.0
python -m pip install numpy matplotlib scipy numba PyYAML
hidden-attractors --help
python -c "import hidden_attractors; print('testpypi import ok')"
```

For Linux/macOS activation, use:

```bash
source .venv_testpypi/bin/activate
```

## 6. Publish to real PyPI

Only after TestPyPI succeeds:

```bash
python -m twine upload dist/*
```

Alternatively, use the manual GitHub Actions workflow `publish-pypi.yml` after configuring Trusted Publishing in PyPI.

Trusted Publishing project settings:

- owner: `Xerkkun`
- repository: `Hidden-Attractors-Localization`
- workflow filename: `publish-pypi.yml`
- environment: `pypi`

## 7. Create the release tag

```bash
git tag -a v1.0.0 -m "Release hidden-attractors-fo v1.0.0"
git push origin v1.0.0
```

## 8. Stop conditions

Do not upload if any of these are true:

- `twine check` fails.
- The wheel does not install in a clean environment.
- `hidden-attractors --help` fails from the installed wheel.
- `hidden-attractors seed --help` exposes Machado/FDF.
- `import hidden_attractors` fails from the installed wheel.
- `hidden-attractors validate release-readiness --submission-strict` fails.
- PyPI rejects the project name.
- There are uncommitted changes between the tag and the build.
- Code changes were made after the tag and before the package upload.

## 9. Scientific boundary

PyPI distributes the software package. It does not strengthen the scientific evidence. DF, BDF, Nyquist, continuation, Lyapunov, FFT/PSD, Poincare, and plots remain seed-generation or diagnostic tools. The Chua arctan c590 status remains finite-time, local/radius-limited numerical evidence under the recorded contract, not a global mathematical proof of hiddenness.
