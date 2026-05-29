(* ::Package:: *)

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];
Get[FileNameJoin[{root, "common", "chua_arctan_validation.wl"}]];

(* Parameters are editable. The default set follows the arctan Chua form:
   f(x)=m x + (n-m) ArcTan[x]. *)
RunChuaArctanValidation[<|
  "SystemID" -> "chua_fractional_arctan",
  "Parameters" -> {
    alpha -> 8.4562,
    beta -> 12.0732,
    gamma -> 0.0052,
    m -> 0.4,
    n -> -1.1585
  },
  "QCases" -> {0.9998},
  "OmegaSeeds" -> {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0},
  "AmplitudeSeeds" -> {0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 12, 20, 40},
  "EquilibriumSeeds" -> {-20, -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 20},
  "ExitOnFailure" -> True
|>]
