# Neighborhood probes: finite terminal policy

New calls to `verification.hiddenness.run_neighborhood_probe` use
`neighborhood_terminal_v2`. `legacy_v1` is explicit replay of the former early
capture rule, including its original contact-cloud subsampling. Historical
campaigns and their stored labels are not rewritten.

The new policy disables heuristic equilibrium and growth early exits. The
declared state-norm threshold remains a safety event. A finite trajectory whose
norm actually crosses it is `threshold_crossed`; nonfinite states or solver
exceptions are `numerical_failure`. A solver status alone never proves a norm
crossing or equilibrium capture.

Successful terminal classification requires an increasing actual time array,
initial time zero, the requested final time, and at least three samples after
the declared burn-in. An optional positive `terminal_window` can further restrict
the tail. Burn-in is never replaced with half a short trajectory. Short,
heuristically stopped or otherwise incomplete observations are `inconclusive`.

An equilibrium is `equilibrium_terminal` only when the last distance and 95th
percentile tail distance are at most `equilibrium_tol`, and the Euclidean norm
of the coordinate-wise tail ranges is at most `equilibrium_span_tol` (default
twice the distance tolerance). These are finite criteria, not a proof of
asymptotic convergence. The returned diagnostics record actual tail times,
sample count, distances, span, solver status and policy.

Other completed observations are `reference_contact` or `other_finite`. A
contact is only the declared metric passing against a finite reference cloud;
it does not establish identity of attractors, chaos, or self-excitation. The
hiddenness contract reports new contacts as `reference_contact_detected`.
Legacy records keep the former interpretation through their explicit policy.
Unknown destinations and incomplete observations block promotion.

`verification.basins.generate_basin_slice(..., return_diagnostics=True)` adds
per-point diagnostics to its traditional `(u, v, matrix)` return. Its new
operational codes are 0 terminal equilibrium, 1 finite reference contact,
2 other finite, 3 observed norm threshold, 4 numerical failure and 5 inconclusive.
The centered Lur'e workflow persists that table and all point diagnostics.

Neighborhood sampling is explicit: `sphere_random` is uniform on the spherical
surface; `ball_random` is uniform in volume using radius `U**(1/d)`;
`coordinate_axes` and `hybrid` remain surface protocols. The sphere workflow
records domain, measure, PCG64 stream, requested and actual counts, initial
points, unit directions and radial fractions. A surface sweep is never reported
to the hiddenness contract as a volume-ball sweep. Axis requests exceeding
`2*d` record their actual available sample count.

Each sweep saves a compressed NPZ for every full returned trajectory, including
actual times, initial point, equilibrium, direction and radial fraction; JSONL
and CSV records link them by SHA-256. The reference cloud and its hash, effective
contact metric and numerical policy are also persisted. Nonfinite raw data
remain in NPZ; JSON diagnostics use `null`, never nonstandard `NaN` constants.

Sweep outputs require a new or empty directory. A parent multi-stage workflow
can pass its active `RunOutputClaim`; a boolean overwrite flag is not accepted.
Existing or partially completed experiments remain untouched.
