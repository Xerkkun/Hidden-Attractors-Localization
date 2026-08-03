# Reproducibility Git bundles

This directory stores one-file Git bundles for scientific source snapshots.
The bundles are canonical transport artifacts; expanded bare repositories with
the suffix `.git/` are local working material and are intentionally ignored.

## MAVPD integer hidden-chaos snapshot

- Bundle: `mavpd_integer_hidden_chaos_f1d10f792a3a.bundle`
- Frozen source bundle SHA-256:
  `f1d10f792a3a2c5c30f766167bbb0bdbcc2d4186a7b96b35675eb1acebb008ce`
- Git commit: `a23db64a43bbe22096aa2a71645fd084774dbc7f`
- Bundle file SHA-256:
  `9ea0227e4bbb356891604e88aefe7bead821fb7deccc11cb9175f405aebea4c1`

Verify the bundle from `version_2`:

```powershell
git bundle verify `
  validation\source_snapshots\git\mavpd_integer_hidden_chaos_f1d10f792a3a.bundle
git bundle list-heads `
  validation\source_snapshots\git\mavpd_integer_hidden_chaos_f1d10f792a3a.bundle
```

Restore the documented bare repository only when a rerun needs it:

```powershell
git clone --bare `
  validation\source_snapshots\git\mavpd_integer_hidden_chaos_f1d10f792a3a.bundle `
  validation\source_snapshots\git\mavpd_integer_hidden_chaos_f1d10f792a3a.git
```

With `GIT_WORK_TREE` pointing at the matching immutable source snapshot,
`git status --porcelain=v1 --untracked-files=all` must be empty and
`git rev-parse HEAD` must return the commit recorded above.
