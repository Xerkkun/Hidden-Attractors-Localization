(* ::Package:: *)

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];
Get[FileNameJoin[{root, "common", "chua_arctan_validation.wl"}]];

(* Bibliographic Wu et al. parameterization:
   f(x)=a1 x + a2 ArcTan[rho x], with rho=1. *)
RunChuaArctanValidation[<|
  "SystemID" -> "chua_fractional_arctan",
  "Parameters" -> {
    alpha -> 42281/5000,
    beta -> 30183/2500,
    gamma -> 13/2500,
    a1c -> 2/5,
    a2c -> -3117/2000,
    rhoNL -> 1
  },
  "QCases" -> {99/100},
  "OmegaSeeds" -> {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0},
  "AmplitudeSeeds" -> {0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 12, 20, 40},
  "EquilibriumSeeds" -> {-20, -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 20},
  "ExitOnFailure" -> True
|>]
