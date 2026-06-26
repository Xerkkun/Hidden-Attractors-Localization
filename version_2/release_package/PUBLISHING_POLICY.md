# Publishing policy

This repository does not store PyPI tokens, API keys, passwords, or upload credentials.

Publication policy for `hidden-attractors-fo`:

- Real publication is manual through `twine` or manual GitHub Actions Trusted Publishing.
- No workflow publishes automatically on `push` or `pull_request`.
- Do not publish if the local commit differs from the release tag.
- Do not publish if `python -m twine check dist/*` fails.
- Do not publish if local wheel installation fails in a clean environment.
- TestPyPI must be uploaded and installed successfully before real PyPI.
- The PyPI release does not change scientific claims or validation status.
- PyPI distributes software; scientific evidence remains in GitHub and the archived DOI record.
- Chua arctan remains local/radius-limited finite-time evidence under the recorded contract, not a global proof.
- Machado/FDF remains theory/internal planned support and is not a public seed workflow in this release.
