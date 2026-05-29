(* ::Package:: *)

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];
Get[FileNameJoin[{root, "common", "chua_saturation_validation.wl"}]];

RunChuaSaturationValidation[<|
  "SystemID" -> "chua_integer_saturation",
  "Parameters" -> {
    alpha -> 8.4562,
    beta -> 12.0732,
    gamma -> 0.0052,
    m0 -> -0.1768,
    m1 -> -1.1468
  },
  "QCases" -> {1},
  "OmegaSeeds" -> {1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.2, 3.4, 3.8},
  "ExitOnFailure" -> True
|>]
