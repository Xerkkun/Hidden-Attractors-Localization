# Archive Policy

`_archived_figure_scripts/` is a local historical archive used during thesis development.
It is not part of the active library, package distribution, tests, examples, validation
evidence or public workflow.

The repository must remain functional if `_archived_figure_scripts/` is absent.

Active code must not import from this directory.
Documentation must not point to it as a required execution path.
Tests must not require files from it.
Figures and reports must be regenerated through the active plotting API.

