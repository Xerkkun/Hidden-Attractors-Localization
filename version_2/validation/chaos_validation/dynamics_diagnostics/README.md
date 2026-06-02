# Dynamics diagnostics contract

These diagnostics support finite-time numerical characterization. They do not
certify chaos or hiddenness. A single positive indicator is insufficient.

Allowed statuses are defined in `diagnostics_contract.json`.

## F5.1-F5.3 structured diagnostics

F5.1 boundedness, F5.2 zero-one, and F5.3 FFT/PSD outputs are stored under
`boundedness/`, `zero_one/`, and `psd_fft/`. They reuse compressed
post-transient trajectories under `trajectories/`.

The four F5 diagnostics can close as
`f5_diagnostics_structured_outputs_ready` when every subphase has
`standardized_outputs=true`. This is an output-readiness state, not a chaos
or hiddenness certification.

## F5.4 Poincare diagnostic

F5.4 adds standardized numerical crossing sections under `poincare/`.
Integer ODE cases may use `x=0, xdot>0`. Caputo cases use geometric sampled
crossings and explicitly set `exact_poincare_map=false`. These outputs do not
certify chaos, hiddenness, or exact periodic orbits in Caputo systems.
