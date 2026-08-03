# MAVPD integer hidden-chaos search

This example searches for a chaotic hidden-attractor candidate without using
a published seed, a stored numerical state, or a frequency grid as an input.
It starts from the registered MAVPD equations and their exact scalar Lur'e
decomposition.

The reproducible route is:

1. derive both positive roots of the integer transfer condition and the
   describing-function quantities `omega0`, `k`, `a0`, and `seed`;
2. run lambda continuation from the theoretical harmonic seed to the original
   nonlinear system at `(xi,gamma)=(3.1,0.1)`;
3. establish only that neither direct base branch passes the declared
   finite-time Lyapunov chaos screen;
4. only then select the successful base branch with the lowest direct harmonic
   frequency and enable the alternative state continuation to the declared
   `xi` endpoint and then in `gamma`;
5. derive the high-gamma Routh--Hurwitz imaginary-pair crossing candidate of
   `E+` and `E-`, and inspect declared offsets around it; transversality and
   Hopf nondegeneracy are not asserted;
6. reject chaotic nodes whose basin is reached from sampled neighborhoods of
   `E0`, reject nodes that fail the declared screen, and refine the surviving
   candidate;
7. evaluate boundedness, DOP853 variational QR exponents, a combined control
   that perturbs the transported candidate initial state and changes
   tolerances, maximum step, burn-in, and accumulation horizon, a fixed-step
   EFORK cross-check, 0--1 return-map diagnostics,
   Poincare geometry, FFT power, equilibrium stability, and sampled
   hiddenness as separate evidence blocks.

The maintained full configuration declares the local `xi` endpoint and, at
that endpoint, selects `gamma`:

```text
xi    = 2.85
gamma = 0.1538037983994911
delta = 100
rho   = 200
```

Here `xi=2.85` is not selected by the candidate screen. `gamma` is the derived
high crossing candidate plus the offset `0.010` chosen by the declared finite
screen; that offset is not stored as an expected result. The checked-in
validation run must be consulted for measured exponents, basin contacts, and
timings.

Run the complete reproduction from `version_2`:

```powershell
& '..\.venv\Scripts\python.exe' examples\modified_van_der_pol_duffing_integer_hidden_chaos_search\run_example.py --output-dir tmp\mavpd-integer-hidden-chaos-full-staging
```

The archived reference run is made from an immutable scientific-source
snapshot rather than from a checkout that may change during the long
calculation.  Its maintained-source bundle is
`f1d10f792a3a2c5c30f766167bbb0bdbcc2d4186a7b96b35675eb1acebb008ce`,
stored at
`validation/source_snapshots/mavpd_integer_hidden_chaos_20260803_f1d10f792a3a`.
Verify the snapshot from the mutable checkout, then change into the snapshot
and invoke the runner as a module so Python imports only its frozen package:

```powershell
& '..\.venv\Scripts\python.exe' `
  validation\python\freeze_mavpd_scientific_sources.py `
  --snapshot-root `
  validation\source_snapshots\mavpd_integer_hidden_chaos_20260803_f1d10f792a3a `
  --verify-only

$snapshot = Resolve-Path `
  'validation\source_snapshots\mavpd_integer_hidden_chaos_20260803_f1d10f792a3a'
$env:HIDDEN_ATTRACTORS_OUTPUT_DIR = (Resolve-Path 'outputs').Path
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:GIT_DIR = (Resolve-Path `
  'validation\source_snapshots\git\mavpd_integer_hidden_chaos_f1d10f792a3a.git').Path
$env:GIT_WORK_TREE = $snapshot.Path
Push-Location $snapshot
& '..\..\..\..\.venv\Scripts\python.exe' -B -m `
  examples.modified_van_der_pol_duffing_integer_hidden_chaos_search.run_example `
  --output-dir '<absolute-fresh-staging-directory>'
Pop-Location
```

The local snapshot commit
`a23db64a43bbe22096aa2a71645fd084774dbc7f` records the exact frozen files and
allows the run metadata to verify a clean work tree.  It is a reproducibility
commit for this evidence bundle, not an upstream release or publication
commit.

Run the shorter integration smoke test:

```powershell
& '..\.venv\Scripts\python.exe' examples\modified_van_der_pol_duffing_integer_hidden_chaos_search\run_example.py --quick --output-dir tmp\mavpd-hidden-chaos-quick
```

Quick mode is always non-promotable. Without `--output-dir` it writes to the
isolated `tmp/mavpd_integer_hidden_chaos_quick_smoke` directory, never to the
canonical validation bundle. Its figure pairs remain local and do not update
the active global `outputs/library_figures/current` pointers or figure
manifest. The repository-level `library_figures/` tree is legacy evidence and
is not the destination used by this runner.

Derive only the algebraic/numerical contract and direct seeds:

```powershell
& '..\.venv\Scripts\python.exe' examples\modified_van_der_pol_duffing_integer_hidden_chaos_search\run_example.py --quick --contract-only --output-dir tmp\mavpd-hidden-chaos-contract
```

The direct polynomial roots are filtered only by the declared admissible
interval `1e-5 <= omega <= 50`; this is not a frequency grid. The Nyquist
figure samples the transfer curve only for display. Those samples are not
searched to produce the seed. Maintained figures are emitted as title-free PNG
and PDF pairs. The normalized FFT-power figure is not labeled as a Welch PSD.

The runner hashes its maintained scientific sources at startup and aborts
before promotion if that bundle changes during a phase. Resume mode also
requires a full-run status ledger, the same resolved-configuration digest, and
matching hashes for every contract, search, trajectory, diagnostic, and timing
artifact. Global figure promotion is delayed until the full joint gate passes.

The complete result remains in staging until its ledger is checked. Validate
the completion state, joint gate, and every recorded artifact hash before any
canonical copy:

```powershell
$stage = Resolve-Path 'tmp\mavpd-integer-hidden-chaos-full-staging'
$status = Get-Content -Raw (Join-Path $stage 'run_status.json') | ConvertFrom-Json
if ($status.status -ne 'complete' -or $status.quick_mode -ne $false) {
    throw 'staging is incomplete or quick'
}
$gate = Get-Content -Raw (Join-Path $stage '09_candidate_gate.json') | ConvertFrom-Json
if ($gate.gate.chaotic_hidden_promotion_allowed -ne $true) {
    throw 'joint scientific gate did not pass'
}
foreach ($item in $status.artifacts.PSObject.Properties) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $stage $item.Name)).Hash.ToLowerInvariant()
    if ($actual -ne ([string]$item.Value).ToLowerInvariant()) {
        throw "artifact hash mismatch: $($item.Name)"
    }
}
$canonical = Join-Path (Resolve-Path '.').Path 'validation\reference_cases\mavpd_integer_hidden_chaos'
if (Test-Path -LiteralPath $canonical) {
    throw 'archive the existing canonical bundle first'
}
Copy-Item -LiteralPath $stage -Destination $canonical -Recurse
```

The final lines copy the complete staging bundle only after validation to
`validation/reference_cases/mavpd_integer_hidden_chaos/`. If that directory
already exists, archive it to an explicit recoverable path first; do not merge
old and new evidence.

The Wolfram case derives the algebra from the source vector field. Its
`Step 2b` block declares `xi=57/20` (the YAML's local continuation endpoint)
and the offset `1/100` (the Python screen's selected result) only to verify the
resulting point a posteriori. It does not select either number. It derives the
Routh--Hurwitz boundary and imaginary-pair frequency, but does not certify
chaos or hiddenness.

The reported divergence residual compares the strict variational exponent sum
with a mean divergence sampled on a distinct candidate-trajectory window. It
is a finite-window consistency check, not an identity evaluated on the same
integration and horizon. The `candidate_gate` phase time includes gate
evaluation plus figure-manifest finalization/promotion.

The final status is necessarily finite: zero contacts in the declared balls
or spheres supports “hidden under tested neighborhoods”; it is not a global
proof that no equilibrium neighborhood intersects the basin.
