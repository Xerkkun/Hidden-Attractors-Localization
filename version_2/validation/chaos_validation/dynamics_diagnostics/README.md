# Dynamics diagnostics contract

These diagnostics support finite-time numerical characterization. They do not
certify chaos or hiddenness. A single positive indicator is insufficient.

Allowed statuses are defined in `diagnostics_contract.json`.

## F5.4 Poincare diagnostic

F5.4 adds standardized numerical crossing sections under `poincare/`.
Integer ODE cases may use `x=0, xdot>0`. Caputo cases use geometric sampled
crossings and explicitly set `exact_poincare_map=false`. These outputs do not
certify chaos, hiddenness, or exact periodic orbits in Caputo systems.
