# Repository Validation Wrappers

The scripts in this directory run case-specific validation records from a
source checkout. They are intentionally excluded from the PyPI distribution
and are not subcommands of the installed `hidden-attractors` CLI.

Public library workflows use `hidden-attractors --help`. Validation records
that cite one of these wrappers must be run from the matching tagged
repository so that their inputs and provenance tree are available.
