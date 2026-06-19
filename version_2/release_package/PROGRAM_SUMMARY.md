# Program Summary

Program title: `hidden-attractors-fo`

Archived title: A Python Library for the Theoretical-Numerical Search and Localization of Hidden Attractors

Repository: <https://github.com/Xerkkun/Hidden-Attractors-Localization>

Licensing provisions: MIT for the software package.

Programming language: Python, with optional C backends.

External routines/libraries: NumPy, SciPy, Matplotlib, Numba, pytest for validation, optional antropy/nolds, and an optional C compiler for native backends.

Supplementary material: <https://doi.org/10.17605/OSF.IO/ZGK74>

Nature of problem: theoretical-numerical search, localization, reproduction, audit, and conservative classification of candidate hidden attractors in integer- and commensurate fractional-order Chua/Lur'e systems with Caputo derivatives for the fractional case.

Solution method: describing-function seeding, BDF-style algebraic continuation, Caputo ABM/EFORK integration for fractional order, integer-order integration paths, finite-time chaos diagnostics, and equilibrium-neighborhood tests.

Restrictions: scalar Lur'e systems and commensurate fractional order in the fractional workflow. Numerical evidence does not prove global mathematical hiddenness. The arctan route is implemented algebraically, pending full validation, and not promoted as a validated hidden attractor.

Running time: seconds for metadata and smoke checks; minutes or hours for long validation or fractional-memory sweeps, depending on horizon, step size, memory policy, backend, and sampling plan.
