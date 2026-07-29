# Source-distribution manifest

The `hidden-attractors-fo` source distribution contains only the importable
library, supported resources, user-facing documentation, and a validated
end-to-end integer reference example. The wheel contains the importable
package and supported package data; project documents and examples remain in
the source distribution.

## Included

- `hidden_attractors/`: Python package and native C sources.
- `hidden_attractors/configs/examples/`: packaged configuration resources.
- `examples/chua_integer_lure_reference/`: validated comprehensive example.
- `examples/quickstart_equilibria.py` and `examples/minimal_chua_protocol.py`.
- `README.md`, `USER_MANUAL.md`, `LICENSE`, and packaging metadata.
- A small whitelist of installation, quick-start, API-stability, scientific
  scope, and citation pages.

## Excluded

- exploratory configurations and ordinary run outputs;
- internal study notes and project plans;
- repository test infrastructure and maintainer release checklists;
- full validation datasets, large figures, and literature PDFs;
- local caches, compiled artifacts, and editorial files.

The complete validation tree remains available in the matching release tag and
archived DOI snapshot. Excluding it from PyPI avoids presenting an installed
wheel as a scientific evidence archive.

Generated outputs default to `./outputs`; runtime caches use the operating
system user-cache directory. Neither location is inside the installed package.
