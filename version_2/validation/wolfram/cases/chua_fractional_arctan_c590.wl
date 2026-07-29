(* ::Package:: *)

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];
Get[FileNameJoin[{root, "common", "chua_arctan_validation.wl"}]];

(*
  Case-specific algebraic validation for the c590 parameterization used in
  Paper 07.  The recorded fractional seed was selected by bounded integer-order
  search and independent Caputo refinement; it is not a describing-function
  seed.  This script validates the system algebra and recorded parameterization,
  not attractor existence, chaos, basin membership, or hiddenness.
*)
RunChuaArctanValidation[<|
  "SystemID" -> "chua_fractional_arctan_c590",
  "ValidationScope" -> "algebra_and_recorded_candidate",
  "SeedOrigin" -> "bounded_integer_search_and_independent_caputo_refinement",
  "RunCanonicalSeed" -> False,
  "Parameters" -> {
    alpha -> 21849356906616716/10^15,
    beta -> 19081840840860202/10^15,
    gamma -> 7378011979156531/10^18,
    a1c -> 4228979343578827/10^17,
    a2c -> -33367815123026694/10^16,
    rhoNL -> 17984259332820332/10^16
  },
  "QCases" -> {9999/10000},
  "RecordedSeed" -> {
    5864244979081692/10^15,
    15847111486491057/10^16,
    32155806477633915/10^16
  },
  "EquilibriumSeeds" -> {
    -20, -10, -6, -5, -4.5, -2, -1, -0.1, 0,
    0.1, 1, 2, 4.5, 5, 6, 10, 20
  },
  "ExpectedEquilibriumCount" -> 3,
  (* Roots are sorted by x: E-, E0, E+. *)
  "ExpectedMatignonStableFlags" -> {True, False, True},
  "DescribingFunctionTestAmplitudes" -> {0.05, 0.2, 1.0, 4.0, 12.0},
  "ExitOnFailure" -> True
|>]
