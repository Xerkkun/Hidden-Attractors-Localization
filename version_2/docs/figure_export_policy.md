# Figure export policy

The maintained plotting, workflow, example and CLI routes use
`hidden_attractors/plotting/export.py`. An active claimed run exports fresh
local PNG/PDF pairs without implicit central promotion. Existing pair members,
completed runs and invalid ownership markers reject writes. Explicit promotion
uses `export_figure` or the transaction-based batch API.

The local-only exception below is a maintained numerical validation renderer,
not a legacy renderer. Its completed input validation and output-directory
claim precede every integration and write; per-point NPZ data, matrices,
effective contract, source hashes and artifact hashes accompany its PNG files.
It does not use or update the global figure store. Its functional tests enforce
the guard, actual method dispatch, saved trajectories and manifest integrity.
Its bounded mean-sign classes carry **no hiddenness claim** and do not identify
an attractor or certify chaos. Local ADM is not equivalent to full-history
Caputo integration.

| Archivo | Motivo | Estado | Permitido en evidencia promovida |
| --- | --- | --- | --- |
| `hidden_attractors/validation/wu2023_basin_maps.py` | Guarded local diagnostic renderer with full trajectory and hash-linked map manifest; explicit promotion is separate | no promovido | no |

Exceptions name exact files. They cannot exempt a canonical plotting,
workflow, example or CLI route. A new maintained validation renderer must use
the central export API or document and test its own local-only contract.

## Archived and local report sources

`local_reports/` contains local experiment dossiers, sealed source copies,
historical executions and their reproduction/analysis scripts. It is excluded
from version control as a local artifact tree. `docs/reportes_revision_tesis/`
is a duplicated thesis-reading/report dossier, including sources and audit
records, rather than an installed package or a canonical figure-export entry
point. Neither tree is selected by the package finder, which installs only
`hidden_attractors*`.

The direct-export hygiene scan does not reinterpret those copied report
sources as additional maintained renderers and does not edit their recorded
bytes to satisfy a current code policy. Their numerical provenance and reuse
remain subject to each dossier's own manifests and scientific admission
rules. This boundary does not exempt `hidden_attractors/validation/`, any
workflow, public example or CLI: those current sources continue to be audited.
