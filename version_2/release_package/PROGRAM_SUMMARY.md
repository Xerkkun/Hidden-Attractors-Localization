# Program Summary

Program title: `hidden-attractors-fo`

Archived title: A Python Library for the Theoretical-Numerical Search and Localization of Hidden Attractors

Repository: <https://github.com/Xerkkun/Hidden-Attractors-Localization>

Licensing provisions: MIT for the software package.

Programming language: Python, with optional C backends.

External routines/libraries: NumPy, SciPy, Matplotlib, Numba, pytest for
validation, optional antropy/nolds, MkDocs for documentation, and an optional C
compiler for native backends.

Supplementary material: <https://doi.org/10.17605/OSF.IO/ZGK74>

Nature of problem: reproducible search, localization, audit, and conservative
classification of candidate hidden attractors in integer- and commensurate
Caputo fractional-order Lur'e-compatible systems.

Solution method: scalar Lur'e formulation, describing-function/Nyquist seed
generation, integer or Caputo continuation, ABM/EFORK integration, finite-time
diagnostics, all-equilibrium neighborhood tests, figures, manifests, and release
metadata.

Official examples: integer Chua Lur'e reference, non-smooth fractional Chua BDF
methodology, and smooth arctan Wu2023/c590 lane. The c590 arctan lane is one
radius-limited validation example for local radii `r <= 0.3`.

Restrictions: scalar Lur'e systems, commensurate order, Caputo fractional
contracts for `0 < q <= 1`, and finite numerical evidence. The package does not
prove global hiddenness. The manifest declares `v1.0.0`; all hiddenness claims are bounded by the tested local-radius contract, while macro-radius contacts are extended basin-geometry audit evidence.

Running time: seconds for metadata/smoke checks; minutes or hours for long
fractional-memory sweeps, hiddenness sampling, or published quantitative
comparison lanes.
