(* ::Package:: *)

(* ============================================================= *)
(* Independent integer covariant-Lyapunov-vector validation      *)
(*                                                               *)
(* The fixtures are exact constant tangent cocycles.  This file  *)
(* implements modified Gram--Schmidt at 80-digit precision and   *)
(* the Ginelli backward triangular recursion directly.  It does  *)
(* not call QRDecomposition, Eigensystem, HAFO, generated reports,*)
(* or a Python implementation.                                   *)
(*                                                               *)
(* Evidence boundary: finite, constant, diagonalizable integer-  *)
(* order linear cocycles only.  Passing does not establish CLV    *)
(* convergence for nonlinear dynamics, hyperbolicity, attraction,*)
(* hiddenness, chaos, or any fractional-order CLV formulation.   *)
(* ============================================================= *)

ClearAll["Global`*"];

validationRoot = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{validationRoot, "common", "ha_validation_common.wl"}]];

systemID = "covariant_lyapunov_integer";
defaultOutDir = If[
  $OperatingSystem === "Windows",
  FileNameJoin[{"C:\\tmp", "hafo_covariant_lyapunov_integer"}],
  FileNameJoin[{$TemporaryDirectory, "hafo_covariant_lyapunov_integer"}]
];
outDir = EnsureDirectory[GetCommandOption["--out", defaultOutDir]];

workingPrecision = 80;
numberOfSteps = 120;
retainedIndices = Range[40, 80];
directCheckpointIndices = {0, 40, 60, 80, 120};
lineTolerance = 5 10^-11;
algebraTolerance = 10^-30;
directTolerance = 10^-18;

source = <|
  "ginelli_doi" -> "10.1103/PhysRevLett.99.130601",
  "kuptsov_parlitz_doi" -> "10.1007/s00332-012-9126-5",
  "froyland_comparison_doi" -> "10.1016/j.physd.2012.12.005",
  "dataset_origin" ->
    "declared exact rational similarities and deterministic rational terminal coefficients",
  "hafo_source_read" -> False,
  "report_input_used" -> False,
  "hafo_formula_imported" -> False,
  "built_in_qr_used" -> False,
  "built_in_eigensystem_used" -> False
|>;

(* All bases and CLV matrices use columns: dimension by n_vectors. *)
ClearAll[
  NormalizeColumns, PositiveModifiedGramSchmidt, ForwardQRHistory,
  BackwardGinelliHistory, LineDistance, ColumnLineDistances,
  MaxAbsNumeric, RelativeMatrixResidual, QRResidual,
  OrthogonalityResidual, BackwardRecursionResidual,
  CovarianceResidual, ExactLineResidual, DirectCheckpointData,
  FiniteTimeExponents, PairAngles, PairAngleHistory,
  PairAngleParityResidual, PairAngleConstancyResidual,
  NonNormalCommutatorNorm
];

NormalizeColumns[matrix_] := Module[{columns, norms},
  columns = Transpose[matrix];
  norms = Sqrt[# . #] & /@ columns;
  If[AnyTrue[norms, TrueQ[# <= 0] &],
    Print[
      "Cannot normalize a zero column; norms=",
      InputForm[norms],
      "; matrix=",
      InputForm[matrix]
    ];
    Exit[2]
  ];
  SetPrecision[
    Transpose[MapThread[#1/#2 &, {columns, norms}]],
    workingPrecision
  ]
];

(* Modified Gram--Schmidt written here, with norm-positive R diagonal. *)
PositiveModifiedGramSchmidt[matrix_] := Module[
  {dimensions, rows, columns, q, r, vector, projection, diagonal, i, j},
  dimensions = Dimensions[matrix];
  rows = dimensions[[1]];
  columns = dimensions[[2]];
  q = ConstantArray[0, {rows, columns}];
  r = ConstantArray[0, {columns, columns}];
  For[j = 1, j <= columns, j++,
    vector = matrix[[All, j]];
    For[i = 1, i < j, i++,
      projection = q[[All, i]] . vector;
      r[[i, j]] = projection;
      vector = vector - projection q[[All, i]];
    ];
    diagonal = Sqrt[vector . vector];
    If[! TrueQ[diagonal > 0],
      Print[
        "Rank loss in independent modified Gram--Schmidt at column ",
        j,
        "; diagonal=",
        InputForm[diagonal],
        "; matrix=",
        InputForm[matrix]
      ];
      Exit[2]
    ];
    r[[j, j]] = diagonal;
    q[[All, j]] = vector/diagonal;
  ];
  (* Exact rational fixtures make Mathematica's cancellation-based precision
     accounting overly pessimistic after many identical QR steps.  Round each
     completed factor back to the declared 80-digit working precision; no
     values are converted to machine precision. *)
  {SetPrecision[q, workingPrecision], SetPrecision[r, workingPrecision]}
];

ForwardQRHistory[cocycle_, initialQ_, steps_Integer?Positive] := Module[
  {qHistory, rHistory, pair, index},
  qHistory = ConstantArray[0, steps + 1];
  rHistory = ConstantArray[0, steps];
  qHistory[[1]] = initialQ;
  For[index = 1, index <= steps, index++,
    pair = PositiveModifiedGramSchmidt[cocycle . qHistory[[index]]];
    qHistory[[index + 1]] = pair[[1]];
    rHistory[[index]] = pair[[2]];
  ];
  <|"q_history" -> qHistory, "r_history" -> rHistory|>
];

BackwardGinelliHistory[qHistory_, rHistory_, terminalSeed_] := Module[
  {steps, coefficientHistory, clvHistory, index},
  steps = Length[rHistory];
  coefficientHistory = ConstantArray[0, steps + 1];
  coefficientHistory[[steps + 1]] = NormalizeColumns[terminalSeed];
  For[index = steps, index >= 1, index--,
    coefficientHistory[[index]] = NormalizeColumns[
      LinearSolve[rHistory[[index]], coefficientHistory[[index + 1]]]
    ];
  ];
  clvHistory = MapThread[
    NormalizeColumns[#1 . #2] &,
    {qHistory, coefficientHistory}
  ];
  <|
    "coefficient_history" -> coefficientHistory,
    "clv_history" -> clvHistory
  |>
];

LineDistance[first_, second_] := Module[{u, v, cosineSquared},
  u = first/Sqrt[first . first];
  v = second/Sqrt[second . second];
  cosineSquared = Abs[u . v]^2;
  Sqrt[Max[0, 1 - cosineSquared]]
];

ColumnLineDistances[first_, second_] := MapThread[
  LineDistance[#1, #2] &,
  {Transpose[first], Transpose[second]}
];

MaxAbsNumeric[value_] := Max[Abs[Flatten[N[value, workingPrecision]]]];

RelativeMatrixResidual[first_, second_] := Module[{denominator},
  denominator = Max[1, Norm[first, "Frobenius"], Norm[second, "Frobenius"]];
  N[Norm[first - second, "Frobenius"]/denominator, workingPrecision]
];

QRResidual[cocycle_, qHistory_, rHistory_] := Max[Table[
  RelativeMatrixResidual[
    cocycle . qHistory[[index]],
    qHistory[[index + 1]] . rHistory[[index]]
  ],
  {index, 1, Length[rHistory]}
]];

OrthogonalityResidual[qHistory_] := Max[
  RelativeMatrixResidual[
    Transpose[#] . #,
    IdentityMatrix[Dimensions[#][[2]]]
  ] & /@ qHistory
];

BackwardRecursionResidual[rHistory_, coefficientHistory_] := Max[Flatten[
  Table[
    ColumnLineDistances[
      rHistory[[index]] . coefficientHistory[[index]],
      coefficientHistory[[index + 1]]
    ],
    {index, 1, Length[rHistory]}
  ]
]];

CovarianceResidual[cocycle_, clvHistory_] := Max[Flatten[
  Table[
    ColumnLineDistances[
      cocycle . clvHistory[[index]],
      clvHistory[[index + 1]]
    ],
    {index, 1, Length[clvHistory] - 1}
  ]
]];

ExactLineResidual[clvHistory_, exactDirections_, indices_] := Max[Flatten[
  ColumnLineDistances[clvHistory[[# + 1]], exactDirections] & /@ indices
]];

DirectCheckpointData[cocycle_, clvHistory_, indices_] := Module[
  {initial, direct, residuals},
  initial = First[clvHistory];
  direct = (
    NormalizeColumns[MatrixPower[cocycle, #] . initial] & /@ indices
  );
  residuals = MapThread[
    ColumnLineDistances[#1, clvHistory[[#2 + 1]]] &,
    {direct, indices}
  ];
  <|
    "indices" -> indices,
    "direct_clv_lines" -> direct,
    "line_distances_to_recursion" -> residuals,
    "max_line_distance" -> Max[Flatten[residuals]]
  |>
];

FiniteTimeExponents[rHistory_, coordinateStep_] := N[
  Total[(Log[Diagonal[#]] &) /@ rHistory]/
    (Length[rHistory] coordinateStep),
  workingPrecision
];

PairAngles[matrix_] := Module[{normalized, pairs},
  normalized = NormalizeColumns[N[matrix, workingPrecision]];
  pairs = Subsets[Range[Dimensions[normalized][[2]]], {2}];
  N[
    ArcCos[
      Min[1, Max[0, Abs[
        normalized[[All, #[[1]]]] . normalized[[All, #[[2]]]]
      ]]]
    ] & /@ pairs,
    workingPrecision
  ]
];

PairAngleHistory[history_, indices_] := PairAngles[history[[# + 1]]] & /@ indices;

PairAngleParityResidual[observed_, exact_] := MaxAbsNumeric[
  observed - ConstantArray[exact, Length[observed]]
];

PairAngleConstancyResidual[history_] := Max[
  (Max[#] - Min[#]) & /@ Transpose[history]
];

NonNormalCommutatorNorm[matrix_] := N[
  Norm[
    Transpose[matrix] . matrix - matrix . Transpose[matrix],
    "Frobenius"
  ],
  workingPrecision
];

(* ------------------------------------------------------------- *)
(* Fixture 1: nonnormal two-dimensional linear map.              *)
(* A = O2.T2.O2^T, with exact eigenlines supplied independently. *)
(* ------------------------------------------------------------- *)

mapO2Exact = {{3/5, -4/5}, {4/5, 3/5}};
mapT2Exact = {{4, 1}, {0, 2}};
mapAExact = {{56/25, 33/25}, {8/25, 94/25}};
mapTerminalSeedExact = {{1, 1/3}, {0, 1}};
mapExactDirections = Transpose[{
  {3/5, 4/5},
  {-11/(5 Sqrt[5]), 2/(5 Sqrt[5])}
}];
mapExactMultipliers = {4, 2};
mapExactExponents = Log[mapExactMultipliers];

mapO2 = N[mapO2Exact, workingPrecision];
mapA = N[mapAExact, workingPrecision];
mapTerminalSeed = N[mapTerminalSeedExact, workingPrecision];
mapForward = ForwardQRHistory[mapA, mapO2, numberOfSteps];
mapBackward = BackwardGinelliHistory[
  mapForward["q_history"],
  mapForward["r_history"],
  mapTerminalSeed
];
mapQHistory = mapForward["q_history"];
mapRHistory = mapForward["r_history"];
mapCoefficientHistory = mapBackward["coefficient_history"];
mapCLVHistory = mapBackward["clv_history"];

mapSimilarityExact = TrueQ[
  mapAExact == mapO2Exact . mapT2Exact . Transpose[mapO2Exact]
];
mapOrthogonalExact = TrueQ[
  Transpose[mapO2Exact] . mapO2Exact == IdentityMatrix[2]
];
mapEigenlineExact = And @@ MapThread[
  TrueQ[FullSimplify[mapAExact . #1 - #2 #1] == {0, 0}] &,
  {Transpose[mapExactDirections], mapExactMultipliers}
];
mapQRResidual = QRResidual[mapA, mapQHistory, mapRHistory];
mapOrthogonalityResidual = OrthogonalityResidual[mapQHistory];
mapPositiveDiagonal = And @@ Flatten[
  (TrueQ[# > 0] & /@ Diagonal[#]) & /@ mapRHistory
];
mapBackwardResidual = BackwardRecursionResidual[
  mapRHistory,
  mapCoefficientHistory
];
mapCovarianceResidual = CovarianceResidual[mapA, mapCLVHistory];
mapExactLineResidual = ExactLineResidual[
  mapCLVHistory,
  N[mapExactDirections, workingPrecision],
  retainedIndices
];
mapDirectCheckpoints = DirectCheckpointData[
  mapA,
  mapCLVHistory,
  directCheckpointIndices
];
mapFiniteTimeExponents = FiniteTimeExponents[mapRHistory, 1];
mapExponentResidual = MaxAbsNumeric[
  mapFiniteTimeExponents - N[mapExactExponents, workingPrecision]
];
mapPairIndices = Subsets[Range[2], {2}];
mapExactPairAngles = PairAngles[mapExactDirections];
mapRetainedPairAngles = PairAngleHistory[mapCLVHistory, retainedIndices];
mapPairAngleParityResidual = PairAngleParityResidual[
  mapRetainedPairAngles,
  mapExactPairAngles
];
mapPairAngleConstancyResidual = PairAngleConstancyResidual[
  mapRetainedPairAngles
];
mapNonNormalCommutatorNorm = NonNormalCommutatorNorm[mapA];

(* ------------------------------------------------------------- *)
(* Fixture 2: nonnormal three-dimensional constant flow.         *)
(* Phi = Exp[B h], h=Log[2], is an exact rational cocycle step.  *)
(* ------------------------------------------------------------- *)

flowO3Exact = {{1, 8, 4}, {8, 1, -4}, {-4, 4, -7}}/9;
flowGExact = {{1, 1, 1/2}, {0, 0, 1}, {0, 0, -1}};
flowBExact = {
  {1/3, -1/9, -7/18},
  {4/3, 4/9, -7/9},
  {0, -8/9, -7/9}
};
flowStepExact = Log[2];
flowPhiExact = {
  {67/54, -1/54, -49/216},
  {34/27, 41/27, -35/54},
  {-8/27, -16/27, 20/27}
};
flowTerminalSeedExact = {
  {1, 1/3, -1/5},
  {0, 1, 2/7},
  {0, 0, 1}
};
flowExactDirections = Transpose[{
  {1/9, 8/9, -4/9},
  {7/(9 Sqrt[2]), -7/(9 Sqrt[2]), 8/(9 Sqrt[2])},
  {-5/(3 Sqrt[33]), -4/(3 Sqrt[33]), -16/(3 Sqrt[33])}
}];
flowExactExponents = {1, 0, -1};
flowExactMultipliers = 2^flowExactExponents;

flowO3 = N[flowO3Exact, workingPrecision];
flowB = N[flowBExact, workingPrecision];
flowStep = N[flowStepExact, workingPrecision];
flowPhi = N[flowPhiExact, workingPrecision];
flowTerminalSeed = N[flowTerminalSeedExact, workingPrecision];
flowForward = ForwardQRHistory[flowPhi, flowO3, numberOfSteps];
flowBackward = BackwardGinelliHistory[
  flowForward["q_history"],
  flowForward["r_history"],
  flowTerminalSeed
];
flowQHistory = flowForward["q_history"];
flowRHistory = flowForward["r_history"];
flowCoefficientHistory = flowBackward["coefficient_history"];
flowCLVHistory = flowBackward["clv_history"];

flowSimilarityExact = TrueQ[
  flowBExact == flowO3Exact . flowGExact . Transpose[flowO3Exact]
];
flowOrthogonalExact = TrueQ[
  Transpose[flowO3Exact] . flowO3Exact == IdentityMatrix[3]
];
flowMatrixExponentialExact = TrueQ[FullSimplify[
  MatrixExp[flowBExact flowStepExact] - flowPhiExact
] == ConstantArray[0, {3, 3}]];
flowGeneratorEigenlineExact = And @@ MapThread[
  TrueQ[FullSimplify[flowBExact . #1 - #2 #1] == {0, 0, 0}] &,
  {Transpose[flowExactDirections], flowExactExponents}
];
flowCocycleEigenlineExact = And @@ MapThread[
  TrueQ[FullSimplify[flowPhiExact . #1 - #2 #1] == {0, 0, 0}] &,
  {Transpose[flowExactDirections], flowExactMultipliers}
];
flowQRResidual = QRResidual[flowPhi, flowQHistory, flowRHistory];
flowOrthogonalityResidual = OrthogonalityResidual[flowQHistory];
flowPositiveDiagonal = And @@ Flatten[
  (TrueQ[# > 0] & /@ Diagonal[#]) & /@ flowRHistory
];
flowBackwardResidual = BackwardRecursionResidual[
  flowRHistory,
  flowCoefficientHistory
];
flowCovarianceResidual = CovarianceResidual[flowPhi, flowCLVHistory];
flowExactLineResidual = ExactLineResidual[
  flowCLVHistory,
  N[flowExactDirections, workingPrecision],
  retainedIndices
];
flowDirectCheckpoints = DirectCheckpointData[
  flowPhi,
  flowCLVHistory,
  directCheckpointIndices
];
flowFiniteTimeExponents = FiniteTimeExponents[
  flowRHistory,
  flowStep
];
flowExponentResidual = MaxAbsNumeric[
  flowFiniteTimeExponents - N[flowExactExponents, workingPrecision]
];
flowPairIndices = Subsets[Range[3], {2}];
flowExactPairAngles = PairAngles[flowExactDirections];
flowRetainedPairAngles = PairAngleHistory[flowCLVHistory, retainedIndices];
flowPairAngleParityResidual = PairAngleParityResidual[
  flowRetainedPairAngles,
  flowExactPairAngles
];
flowPairAngleConstancyResidual = PairAngleConstancyResidual[
  flowRetainedPairAngles
];
flowNonNormalCommutatorNorm = NonNormalCommutatorNorm[flowPhi];

tests = {
  MakeTest[
    "map_exact_similarity_and_eigenlines",
    mapSimilarityExact && mapOrthogonalExact && mapEigenlineExact
  ],
  MakeTest[
    "flow_exact_similarity_exponential_and_eigenlines",
    flowSimilarityExact && flowOrthogonalExact &&
      flowMatrixExponentialExact && flowGeneratorEigenlineExact &&
      flowCocycleEigenlineExact
  ],
  MakeTest[
    "own_modified_gram_schmidt_has_positive_diagonal",
    mapPositiveDiagonal && flowPositiveDiagonal &&
      mapOrthogonalityResidual < algebraTolerance &&
      flowOrthogonalityResidual < algebraTolerance
  ],
  MakeTest[
    "forward_qr_factorizations_match_cocycles",
    mapQRResidual < algebraTolerance && flowQRResidual < algebraTolerance
  ],
  MakeTest[
    "ginelli_backward_triangular_recursions_are_projectively_exact",
    mapBackwardResidual < algebraTolerance &&
      flowBackwardResidual < algebraTolerance
  ],
  MakeTest[
    "map_clvs_are_covariant_lines",
    mapCovarianceResidual < algebraTolerance
  ],
  MakeTest[
    "flow_clvs_are_covariant_lines",
    flowCovarianceResidual < algebraTolerance
  ],
  MakeTest[
    "retained_map_clvs_match_declared_exact_eigenlines",
    mapExactLineResidual < lineTolerance
  ],
  MakeTest[
    "retained_flow_clvs_match_declared_exact_eigenlines",
    flowExactLineResidual < lineTolerance
  ],
  MakeTest[
    "direct_matrix_power_checkpoints_match_backward_recursion",
    mapDirectCheckpoints["max_line_distance"] < directTolerance &&
      flowDirectCheckpoints["max_line_distance"] < directTolerance
  ],
  MakeTest[
    "finite_time_qr_exponents_match_exact_spectrum",
    mapExponentResidual < algebraTolerance &&
      flowExponentResidual < algebraTolerance
  ],
  MakeTest[
    "unoriented_pair_angles_match_exact_lines_and_are_constant",
    mapPairAngleParityResidual < lineTolerance &&
      flowPairAngleParityResidual < lineTolerance &&
      mapPairAngleConstancyResidual < lineTolerance &&
      flowPairAngleConstancyResidual < lineTolerance
  ],
  MakeTest[
    "both_cocycles_are_explicitly_nonnormal",
    mapNonNormalCommutatorNorm > 0 && flowNonNormalCommutatorNorm > 0
  ],
  MakeTest["hafo_source_not_read", source["hafo_source_read"] === False],
  MakeTest[
    "hafo_formula_not_imported",
    source["hafo_formula_imported"] === False
  ],
  MakeTest["report_not_used", source["report_input_used"] === False],
  MakeTest[
    "built_in_qr_and_eigensystem_not_used",
    source["built_in_qr_used"] === False &&
      source["built_in_eigensystem_used"] === False
  ]
};

evidenceBoundary =
  "finite 80-digit constant diagonalizable integer-order linear-cocycle " <>
  "algebra and projective covariance only; no general nonlinear CLV " <>
  "convergence theorem, hyperbolicity, chaos classification, attraction, " <>
  "hiddenness, or fractional-order CLV validity; individual CLVs may be " <>
  "nonunique for repeated or nearly degenerate Lyapunov exponents";

summary = <|
  "system_id" -> systemID,
  "validation_scope" ->
    "independent_integer_CLV_Ginelli_constant_cocycle_consistency",
  "evidence_boundary" -> evidenceBoundary,
  "source" -> source,
  "conventions" -> <|
    "working_precision" -> workingPrecision,
    "number_of_steps" -> numberOfSteps,
    "retained_indices_inclusive" -> retainedIndices,
    "direct_checkpoint_indices" -> directCheckpointIndices,
    "q_history_shape" -> "steps_plus_one_dimension_vectors",
    "r_history_shape" -> "steps_vectors_vectors",
    "clv_history_shape" -> "steps_plus_one_dimension_vectors",
    "vectors_are_columns" -> True,
    "forward_identity" -> "M.Q[n-1] = Q[n].R[n]",
    "qr_algorithm" ->
      "independent_modified_gram_schmidt_with_positive_R_diagonal",
    "backward_identity" ->
      "C[n-1] = column_normalize(linear_solve(R[n], C[n]))",
    "clv_reconstruction" -> "V[n] = column_normalize(Q[n].C[n])",
    "line_distance" -> "sqrt(max(0,1-abs(dot(unit(u),unit(v)))^2))",
    "sign_orientation" -> "unoriented_projective_lines",
    "fractional_order_supported_by_this_oracle" -> False
  |>,
  "fixtures" -> <|
    "nonnormal_map_2d" -> <|
      "kind" -> "map",
      "dimension" -> 2,
      "matrix" -> N[mapAExact, 40],
      "orthogonal_similarity" -> N[mapO2Exact, 40],
      "schur_factor" -> N[mapT2Exact, 40],
      "coordinate_step" -> 1.0,
      "terminal_coefficients" -> N[mapTerminalSeedExact, 40],
      "exact_directions_columns" -> N[mapExactDirections, 40],
      "exact_multipliers" -> N[mapExactMultipliers, 40],
      "exact_exponents" -> N[mapExactExponents, 40],
      "q_history" -> N[mapQHistory, 40],
      "r_history" -> N[mapRHistory, 40],
      "retained_indices" -> retainedIndices,
      "retained_clvs" -> N[mapCLVHistory[[retainedIndices + 1]], 40],
      "direct_checkpoints" -> mapDirectCheckpoints,
      "finite_time_exponents" -> N[mapFiniteTimeExponents, 40],
      "pair_indices_zero_based" -> (mapPairIndices - 1),
      "exact_pair_angles_radians" -> N[mapExactPairAngles, 40],
      "retained_pair_angles_radians" -> N[mapRetainedPairAngles, 40],
      "non_normal_commutator_frobenius_norm" ->
        N[mapNonNormalCommutatorNorm, 40]
    |>,
    "constant_flow_3d" -> <|
      "kind" -> "flow",
      "dimension" -> 3,
      "generator" -> N[flowBExact, 40],
      "coordinate_step" -> N[flowStepExact, 40],
      "cocycle_matrix" -> N[flowPhiExact, 40],
      "orthogonal_similarity" -> N[flowO3Exact, 40],
      "schur_generator" -> N[flowGExact, 40],
      "terminal_coefficients" -> N[flowTerminalSeedExact, 40],
      "exact_directions_columns" -> N[flowExactDirections, 40],
      "exact_multipliers" -> N[flowExactMultipliers, 40],
      "exact_exponents" -> N[flowExactExponents, 40],
      "q_history" -> N[flowQHistory, 40],
      "r_history" -> N[flowRHistory, 40],
      "retained_indices" -> retainedIndices,
      "retained_times" -> N[retainedIndices flowStepExact, 40],
      "retained_clvs" -> N[flowCLVHistory[[retainedIndices + 1]], 40],
      "direct_checkpoints" -> flowDirectCheckpoints,
      "finite_time_exponents" -> N[flowFiniteTimeExponents, 40],
      "pair_indices_zero_based" -> (flowPairIndices - 1),
      "exact_pair_angles_radians" -> N[flowExactPairAngles, 40],
      "retained_pair_angles_radians" -> N[flowRetainedPairAngles, 40],
      "non_normal_commutator_frobenius_norm" ->
        N[flowNonNormalCommutatorNorm, 40]
    |>
  |>,
  "numeric_cross_checks" -> <|
    "working_precision" -> workingPrecision,
    "map_qr_factorization_max_relative_residual" -> mapQRResidual,
    "flow_qr_factorization_max_relative_residual" -> flowQRResidual,
    "map_q_orthogonality_max_relative_residual" ->
      mapOrthogonalityResidual,
    "flow_q_orthogonality_max_relative_residual" ->
      flowOrthogonalityResidual,
    "map_backward_projective_max_residual" -> mapBackwardResidual,
    "flow_backward_projective_max_residual" -> flowBackwardResidual,
    "map_covariance_max_line_distance" -> mapCovarianceResidual,
    "flow_covariance_max_line_distance" -> flowCovarianceResidual,
    "map_retained_exact_line_max_distance" -> mapExactLineResidual,
    "flow_retained_exact_line_max_distance" -> flowExactLineResidual,
    "map_direct_checkpoint_max_line_distance" ->
      mapDirectCheckpoints["max_line_distance"],
    "flow_direct_checkpoint_max_line_distance" ->
      flowDirectCheckpoints["max_line_distance"],
    "map_finite_time_exponent_max_residual" -> mapExponentResidual,
    "flow_finite_time_exponent_max_residual" -> flowExponentResidual,
    "map_pair_angle_exact_parity_max_residual" ->
      mapPairAngleParityResidual,
    "flow_pair_angle_exact_parity_max_residual" ->
      flowPairAngleParityResidual,
    "map_pair_angle_constancy_max_residual" ->
      mapPairAngleConstancyResidual,
    "flow_pair_angle_constancy_max_residual" ->
      flowPairAngleConstancyResidual,
    "map_non_normal_commutator_frobenius_norm" ->
      mapNonNormalCommutatorNorm,
    "flow_non_normal_commutator_frobenius_norm" ->
      flowNonNormalCommutatorNorm,
    "line_tolerance" -> N[lineTolerance, 24],
    "algebra_tolerance" -> N[algebraTolerance, 24],
    "direct_tolerance" -> N[directTolerance, 24]
  |>,
  "tests" -> tests,
  "passed" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests)
|>;

ExportJSON[
  FileNameJoin[{outDir, systemID <> "_validation_summary.json"}],
  summary
];

ExitFromTests[tests];
