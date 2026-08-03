# Invalidated staging run

This staging directory is not scientific or timing evidence and must not be
resumed, promoted, or copied into the canonical validation case.

- Run ID: `mavpd_integer_hidden_chaos-full-20260803T105928613981Z-f1d10f792a3a-c9404c753964`
- Scientific source bundle: `f1d10f792a3a2c5c30f766167bbb0bdbcc2d4186a7b96b35675eb1acebb008ce`
- Snapshot Git commit: `a23db64a43bbe22096aa2a71645fd084774dbc7f`
- Operator action: stopped during `search` after detecting a separate
  CPU-intensive Python process that predated this run and would contaminate
  the requested timing comparison.
- Promotion: not executed.

The attempt is retained only as an audit trace. A fresh direct full run must
start after all unrelated Python calculations have ended.
