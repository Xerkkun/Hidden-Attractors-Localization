# Integer MAVPD Lur'e audit

This example audits the integer modified autonomous Van der Pol--Duffing
system without using the values printed in Table 1 as search inputs.

The primary regime is `xi=3.1`.  The script derives every positive transfer
branch without a frequency grid, processes branch 0 first, transports each
seed through lambda continuation, clusters the resulting trajectories, and
selects the largest recurrent cluster without consulting the published seed.
The finite hiddenness screen consumes the same explicit 108-condition CSV as
the Julia implementation.

The fixed-step library trajectory is treated as localization evidence.  A
strict independent DOP853 refinement, interpolated Poincare sections, a 0-1
diagnostic, increasing observation windows, and variational QR exponents are
used for dynamic qualification.  For the declared equations this local
qualification supports a stable periodic cycle, although the source describes
the corresponding `xi=3.1` orbit as quasiperiodic; both statements are retained
as an explicit reproduction discrepancy.

The secondary regime `xi=3.5` is a negative reproduction.  Both direct cycles
are tested against all 108 rows of that same shared CSV.  The script records
the first contact, completes the 108-condition census, and only then executes
the small deterministic branch/phase/initial-condition fallback.  The
published seed is integrated last as a post-derivation audit.

Run the short contract from `version_2`:

```powershell
& '..\.venv\Scripts\python.exe' examples\modified_van_der_pol_duffing_integer_lure_audit\run_example.py --quick
```

Use another explicit probe contract with:

```powershell
& '..\.venv\Scripts\python.exe' examples\modified_van_der_pol_duffing_integer_lure_audit\run_example.py --quick --probe-input path\to\probes.csv
```

Outputs are written under
`validation/reference_cases/mavpd_integer_q1_audit/`.  A zero-contact result
is a finite sampled-neighborhood statement, never a global proof of
hiddenness.
