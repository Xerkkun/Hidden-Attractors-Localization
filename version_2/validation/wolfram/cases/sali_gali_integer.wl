(* ::Package:: *)

(* ============================================================= *)
(* Independent integer SALI/GALI alignment-index validation      *)
(*                                                               *)
(* Exact tangent matrices are generated with MatrixPower for two *)
(* maps and MatrixExp for one constant-Jacobian flow. SALI is     *)
(* built from the parallel/antiparallel norms. GALI is built      *)
(* independently from both a Gram determinant and Cauchy--Binet   *)
(* minors, then checked numerically against a singular-value      *)
(* product (the corresponding LDI formulation).                  *)
(*                                                               *)
(* No HAFO source, generated report, or HAFO formula is read.     *)
(* Evidence boundary: finite exact linear tangent-algebra and     *)
(* numerical consistency only; no nonlinear chaos classification,*)
(* convergence theorem, attraction, hiddenness, or fractional     *)
(* SALI/GALI claim. In particular, the hyperbolic fixtures are not*)
(* labelled as chaotic attractors.                               *)
(* ============================================================= *)

ClearAll["Global`*"];

validationRoot = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{validationRoot, "common", "ha_validation_common.wl"}]];

systemID = "sali_gali_integer";
repositoryRoot = ParentDirectory[ParentDirectory[validationRoot]];
outDir = EnsureDirectory[
  GetCommandOption[
    "--out",
    FileNameJoin[{
      repositoryRoot, "validation", "outputs", "wolfram", systemID
    }]
  ]
];

source = <|
  "sali_doi" -> "10.1088/0305-4470/37/24/006",
  "gali_doi" -> "10.1016/j.physd.2007.04.004",
  "rolim_sales_leonel_antonopoulos_doi" ->
    "10.1016/j.chaos.2026.117884",
  "dataset_origin" ->
    "declared exact linear maps, constant-Jacobian flow, and deterministic tangent bases",
  "hafo_source_read" -> False,
  "report_input_used" -> False,
  "hafo_formula_imported" -> False
|>;

workingPrecision = 80;

(* All instantaneous tangent matrices use columns (dimension,k). *)
ClearAll[
  NormalizeColumnsExact, SALIFromColumnsExact, GALIGramExact,
  GALICauchyBinetExact, GALISVDNumeric
];

NormalizeColumnsExact[matrix_] := Module[{columns},
  columns = Transpose[matrix];
  Transpose[(#/Sqrt[# . #]) & /@ columns]
];

SALIFromColumnsExact[matrix_] := Module[{normalized, first, second},
  normalized = NormalizeColumnsExact[matrix[[All, 1 ;; 2]]];
  first = normalized[[All, 1]];
  second = normalized[[All, 2]];
  Min[Norm[first + second], Norm[first - second]]
];

GALIGramExact[matrix_, order_Integer?Positive] := Module[
  {normalized, gram},
  normalized = NormalizeColumnsExact[matrix[[All, 1 ;; order]]];
  gram = Transpose[normalized] . normalized;
  FullSimplify[Sqrt[Det[gram]]]
];

GALICauchyBinetExact[matrix_, order_Integer?Positive] := Module[
  {normalized, rowSubsets, squaredMinors},
  normalized = NormalizeColumnsExact[matrix[[All, 1 ;; order]]];
  rowSubsets = Subsets[Range[Length[normalized]], {order}];
  squaredMinors = (Det[normalized[[#, All]]]^2 &) /@ rowSubsets;
  FullSimplify[Sqrt[Total[squaredMinors]]]
];

GALISVDNumeric[matrix_, order_Integer?Positive] := Module[{normalized},
  normalized = N[
    NormalizeColumnsExact[matrix[[All, 1 ;; order]]],
    workingPrecision
  ];
  Times @@ SingularValueList[normalized]
];

(* ------------------------------------------------------------- *)
(* Fixture 1: exact orthogonal rotation map.                     *)
(* The nonorthogonal initial columns deliberately detect any      *)
(* accidental Gram--Schmidt/QR replacement of individual norms. *)
(* ------------------------------------------------------------- *)

rotationMapExact = {
  {0, -1, 0},
  {1,  0, 0},
  {0,  0, 1}
};
rotationInitialDeviationsExact = {
  {1, 3/5, 0},
  {0, 4/5, 0},
  {0,   0, 1}
};
rotationIterations = Range[0, 8];
rotationTangentMatricesExact = (
  MatrixPower[rotationMapExact, #] . rotationInitialDeviationsExact & /@
    rotationIterations
);
rotationNormalizedMatricesExact =
  NormalizeColumnsExact /@ rotationTangentMatricesExact;
rotationSALIExact = SALIFromColumnsExact /@ rotationTangentMatricesExact;
rotationGALI2GramExact =
  (GALIGramExact[#, 2] &) /@ rotationTangentMatricesExact;
rotationGALI2CauchyExact =
  (GALICauchyBinetExact[#, 2] &) /@ rotationTangentMatricesExact;
rotationGALI3GramExact =
  (GALIGramExact[#, 3] &) /@ rotationTangentMatricesExact;
rotationGALI3CauchyExact =
  (GALICauchyBinetExact[#, 3] &) /@ rotationTangentMatricesExact;
rotationGALI3SVD =
  (GALISVDNumeric[#, 3] &) /@ rotationTangentMatricesExact;

rotationExpectedSALI = ConstantArray[2/Sqrt[5], Length[rotationIterations]];
rotationExpectedGALI2 = ConstantArray[4/5, Length[rotationIterations]];
rotationExpectedGALI3 = ConstantArray[4/5, Length[rotationIterations]];
rotationSVDMaxResidual = N[
  Max[Abs[rotationGALI3SVD - N[rotationExpectedGALI3, workingPrecision]]],
  40
];

rotationSignedScaledInitialExact =
  rotationInitialDeviationsExact . DiagonalMatrix[{7, -3, 5}];
rotationSignedScaledTangentExact = (
  MatrixPower[rotationMapExact, #] . rotationSignedScaledInitialExact & /@
    rotationIterations
);
rotationSignedScaledSALIExact =
  SALIFromColumnsExact /@ rotationSignedScaledTangentExact;
rotationSignedScaledGALI3Exact =
  (GALIGramExact[#, 3] &) /@ rotationSignedScaledTangentExact;
rotationPermutedGALI3Exact = (
  GALIGramExact[#[[All, {3, 1, 2}]], 3] & /@
    rotationTangentMatricesExact
);

(* ------------------------------------------------------------- *)
(* Fixture 2: exact area-preserving hyperbolic diagonal map.     *)
(* ------------------------------------------------------------- *)

hyperbolicMapExact = DiagonalMatrix[{2, 1/2}];
hyperbolicMapInitialDeviationsExact = {
  {1/Sqrt[2],  1/Sqrt[2]},
  {1/Sqrt[2], -1/Sqrt[2]}
};
hyperbolicMapIterations = Range[0, 8];
hyperbolicMapTangentMatricesExact = (
  MatrixPower[hyperbolicMapExact, #] .
      hyperbolicMapInitialDeviationsExact & /@
    hyperbolicMapIterations
);
hyperbolicMapNormalizedMatricesExact =
  NormalizeColumnsExact /@ hyperbolicMapTangentMatricesExact;
hyperbolicMapSALIExact =
  SALIFromColumnsExact /@ hyperbolicMapTangentMatricesExact;
hyperbolicMapGALI2GramExact =
  (GALIGramExact[#, 2] &) /@ hyperbolicMapTangentMatricesExact;
hyperbolicMapGALI2CauchyExact =
  (GALICauchyBinetExact[#, 2] &) /@ hyperbolicMapTangentMatricesExact;
hyperbolicMapGALI2SVD =
  (GALISVDNumeric[#, 2] &) /@ hyperbolicMapTangentMatricesExact;

hyperbolicMapExpectedSALI =
  (2/Sqrt[16^# + 1] &) /@ hyperbolicMapIterations;
hyperbolicMapExpectedGALI2 =
  (2 4^#/(16^# + 1) &) /@ hyperbolicMapIterations;
hyperbolicMapSVDMaxResidual = N[
  Max[Abs[
    hyperbolicMapGALI2SVD -
      N[hyperbolicMapExpectedGALI2, workingPrecision]
  ]],
  40
];
hyperbolicMapFirstSALIThresholdIteration =
  First[Pick[
    hyperbolicMapIterations,
    (TrueQ[# < 1/10] &) /@ hyperbolicMapSALIExact
  ]];
hyperbolicMapFirstGALI2ThresholdIteration =
  First[Pick[
    hyperbolicMapIterations,
    (TrueQ[# < 1/10] &) /@ hyperbolicMapGALI2GramExact
  ]];

ClearAll[iterationSymbol];
hyperbolicMapSALIRatioLimit = FullSimplify[Limit[
  (2/Sqrt[16^(iterationSymbol + 1) + 1])/
    (2/Sqrt[16^iterationSymbol + 1]),
  iterationSymbol -> Infinity
]];
hyperbolicMapGALI2RatioLimit = FullSimplify[Limit[
  (2 4^(iterationSymbol + 1)/(16^(iterationSymbol + 1) + 1))/
    (2 4^iterationSymbol/(16^iterationSymbol + 1)),
  iterationSymbol -> Infinity
]];

(* ------------------------------------------------------------- *)
(* Fixture 3: constant-Jacobian diagonal integer-order flow.     *)
(* ------------------------------------------------------------- *)

flowGeneratorExact = DiagonalMatrix[{1, 0, -1}];
flowInitialDeviationsExact = (1/Sqrt[3]) {
  {1,  1,  1},
  {1, -1,  1},
  {1,  1, -1}
};
ClearAll[timeSymbol];
flowFundamentalSymbolic = MatrixExp[flowGeneratorExact timeSymbol];
flowTangentSymbolic =
  flowFundamentalSymbolic . flowInitialDeviationsExact;
flowNormalizedSymbolic = NormalizeColumnsExact[flowTangentSymbolic];
flowScaleSymbolic = Exp[2 timeSymbol] + 1 + Exp[-2 timeSymbol];

flowFirstSymbolic = flowNormalizedSymbolic[[All, 1]];
flowSecondSymbolic = flowNormalizedSymbolic[[All, 2]];
flowDifferenceNormSquaredSymbolic = FullSimplify[
  (flowFirstSymbolic - flowSecondSymbolic) .
    (flowFirstSymbolic - flowSecondSymbolic),
  Assumptions -> Element[timeSymbol, Reals] && timeSymbol >= 0
];
flowSumNormSquaredSymbolic = FullSimplify[
  (flowFirstSymbolic + flowSecondSymbolic) .
    (flowFirstSymbolic + flowSecondSymbolic),
  Assumptions -> Element[timeSymbol, Reals] && timeSymbol >= 0
];
flowSALIDefinitionSymbolic = Sqrt[flowDifferenceNormSquaredSymbolic];
flowGALI2GramSymbolic = GALIGramExact[flowTangentSymbolic, 2];
flowGALI2CauchySymbolic =
  GALICauchyBinetExact[flowTangentSymbolic, 2];
flowGALI3GramSymbolic = GALIGramExact[flowTangentSymbolic, 3];
flowGALI3CauchySymbolic =
  GALICauchyBinetExact[flowTangentSymbolic, 3];

flowExpectedSALISymbolic = 2/Sqrt[flowScaleSymbolic];
flowExpectedGALI2Symbolic =
  2 Sqrt[flowScaleSymbolic - 1]/flowScaleSymbolic;
flowExpectedGALI3Symbolic = 4/flowScaleSymbolic^(3/2);

flowSALIClosedMatch = TrueQ[FullSimplify[
  flowSALIDefinitionSymbolic - flowExpectedSALISymbolic,
  Assumptions -> Element[timeSymbol, Reals] && timeSymbol >= 0
] == 0] && TrueQ[FullSimplify[
  flowSumNormSquaredSymbolic - flowDifferenceNormSquaredSymbolic >= 0,
  Assumptions -> Element[timeSymbol, Reals] && timeSymbol >= 0
]];
flowGALI2ClosedMatch = TrueQ[FullSimplify[
  flowGALI2GramSymbolic^2 - flowExpectedGALI2Symbolic^2,
  Assumptions -> Element[timeSymbol, Reals] && timeSymbol >= 0
] == 0] && TrueQ[FullSimplify[
  flowGALI2CauchySymbolic^2 - flowExpectedGALI2Symbolic^2,
  Assumptions -> Element[timeSymbol, Reals] && timeSymbol >= 0
] == 0] && TrueQ[FullSimplify[
  flowExpectedGALI2Symbolic > 0,
  Assumptions -> Element[timeSymbol, Reals] && timeSymbol >= 0
]];
flowGALI3ClosedMatch = TrueQ[FullSimplify[
  flowGALI3GramSymbolic^2 - flowExpectedGALI3Symbolic^2,
  Assumptions -> Element[timeSymbol, Reals] && timeSymbol >= 0
] == 0] && TrueQ[FullSimplify[
  flowGALI3CauchySymbolic^2 - flowExpectedGALI3Symbolic^2,
  Assumptions -> Element[timeSymbol, Reals] && timeSymbol >= 0
] == 0] && TrueQ[FullSimplify[
  flowExpectedGALI3Symbolic > 0,
  Assumptions -> Element[timeSymbol, Reals] && timeSymbol >= 0
]];

flowTimesExact = {0, 1/4, 1/2, 1, 3/2, 2, 5/2, 3};
flowTangentMatricesExact = (
  (flowFundamentalSymbolic /. timeSymbol -> #) .
      flowInitialDeviationsExact & /@
    flowTimesExact
);
flowNormalizedMatricesExact =
  NormalizeColumnsExact /@ flowTangentMatricesExact;
flowSALIExact = SALIFromColumnsExact /@ flowTangentMatricesExact;
flowGALI2GramExact =
  (GALIGramExact[#, 2] &) /@ flowTangentMatricesExact;
flowGALI2CauchyExact =
  (GALICauchyBinetExact[#, 2] &) /@ flowTangentMatricesExact;
flowGALI3GramExact =
  (GALIGramExact[#, 3] &) /@ flowTangentMatricesExact;
flowGALI3CauchyExact =
  (GALICauchyBinetExact[#, 3] &) /@ flowTangentMatricesExact;
flowGALI3SVD = (GALISVDNumeric[#, 3] &) /@ flowTangentMatricesExact;
flowExpectedSALI =
  (flowExpectedSALISymbolic /. timeSymbol -> # &) /@ flowTimesExact;
flowExpectedGALI2 =
  (flowExpectedGALI2Symbolic /. timeSymbol -> # &) /@ flowTimesExact;
flowExpectedGALI3 =
  (flowExpectedGALI3Symbolic /. timeSymbol -> # &) /@ flowTimesExact;
flowSVDMaxResidual = N[
  Max[Abs[flowGALI3SVD - N[flowExpectedGALI3, workingPrecision]]],
  40
];

flowSALILogRate = FullSimplify[Limit[
  -Log[flowExpectedSALISymbolic]/timeSymbol,
  timeSymbol -> Infinity
]];
flowGALI2LogRate = FullSimplify[Limit[
  -Log[flowExpectedGALI2Symbolic]/timeSymbol,
  timeSymbol -> Infinity
]];
flowGALI3LogRate = FullSimplify[Limit[
  -Log[flowExpectedGALI3Symbolic]/timeSymbol,
  timeSymbol -> Infinity
]];

flowFundamentalClosedMatch = And @@ Flatten[MapThread[
  TrueQ[FullSimplify[
    #1 - #2,
    Assumptions -> Element[timeSymbol, Reals]
  ] == 0] &,
  {
    flowFundamentalSymbolic,
    DiagonalMatrix[{Exp[timeSymbol], 1, Exp[-timeSymbol]}]
  },
  2
]];
flowSALISampleMatch = And @@ MapThread[
  TrueQ[FullSimplify[#1 - #2] == 0] &,
  {flowSALIExact, flowExpectedSALI}
];
flowGALI2SampleMatch = And @@ MapThread[
  (
    TrueQ[FullSimplify[#1^2 - #2^2] == 0] &&
      TrueQ[N[#1, workingPrecision] >= 0] &&
      TrueQ[N[#2, workingPrecision] >= 0]
  ) &,
  {flowGALI2GramExact, flowExpectedGALI2}
];
flowGALI3SampleMatch = And @@ MapThread[
  TrueQ[FullSimplify[#1 - #2] == 0] &,
  {flowGALI3GramExact, flowExpectedGALI3}
];

(* GALI_2/SALI identity for all finite samples. *)
allSALIValues = Join[
  rotationSALIExact,
  hyperbolicMapSALIExact,
  flowSALIExact
];
allGALI2Values = Join[
  rotationGALI2GramExact,
  hyperbolicMapGALI2GramExact,
  flowGALI2GramExact
];
gali2SALIIdentityMatch = And @@ MapThread[
  TrueQ[FullSimplify[
    4 #2^2 - #1^2 (4 - #1^2)
  ] == 0] &,
  {allSALIValues, allGALI2Values}
];

tests = {
  MakeTest[
    "sali_matches_minimum_parallel_antiparallel_norm",
    TrueQ[FullSimplify[rotationSALIExact == rotationExpectedSALI]] &&
      TrueQ[FullSimplify[
        hyperbolicMapSALIExact == hyperbolicMapExpectedSALI
      ]] && flowSALIClosedMatch
  ],
  MakeTest[
    "gali_matches_gram_cauchy_binet_and_svd_ldi",
    TrueQ[FullSimplify[
      rotationGALI2GramExact == rotationGALI2CauchyExact
    ]] && TrueQ[FullSimplify[
      rotationGALI3GramExact == rotationGALI3CauchyExact
    ]] && TrueQ[FullSimplify[
      hyperbolicMapGALI2GramExact == hyperbolicMapGALI2CauchyExact
    ]] && TrueQ[FullSimplify[
      flowGALI2GramExact == flowGALI2CauchyExact
    ]] && TrueQ[FullSimplify[
      flowGALI3GramExact == flowGALI3CauchyExact
    ]] && rotationSVDMaxResidual < 10^-40 &&
      hyperbolicMapSVDMaxResidual < 10^-40 &&
      flowSVDMaxResidual < 10^-40
  ],
  MakeTest[
    "gali2_sali_identity_is_exact",
    gali2SALIIdentityMatch
  ],
  MakeTest[
    "rotation_map_preserves_all_alignment_indices",
    TrueQ[Transpose[rotationMapExact] . rotationMapExact == IdentityMatrix[3]] &&
      TrueQ[FullSimplify[rotationSALIExact == rotationExpectedSALI]] &&
      TrueQ[FullSimplify[
        rotationGALI2GramExact == rotationExpectedGALI2
      ]] && TrueQ[FullSimplify[
        rotationGALI3GramExact == rotationExpectedGALI3
      ]]
  ],
  MakeTest[
    "diagonal_hyperbolic_map_matches_closed_sequences",
    TrueQ[Det[hyperbolicMapExact] == 1] &&
      TrueQ[FullSimplify[
        hyperbolicMapSALIExact == hyperbolicMapExpectedSALI
      ]] && TrueQ[FullSimplify[
        hyperbolicMapGALI2GramExact == hyperbolicMapExpectedGALI2
      ]] && hyperbolicMapFirstSALIThresholdIteration == 3 &&
      hyperbolicMapFirstGALI2ThresholdIteration == 3
  ],
  MakeTest[
    "diagonal_flow_matches_closed_sali_gali2_gali3",
    flowFundamentalClosedMatch && flowSALIClosedMatch &&
      flowGALI2ClosedMatch && flowGALI3ClosedMatch &&
      flowSALISampleMatch && flowGALI2SampleMatch &&
      flowGALI3SampleMatch,
    <|
      "fundamental_matrix_match" -> flowFundamentalClosedMatch,
      "sali_symbolic_match" -> flowSALIClosedMatch,
      "gali2_symbolic_match" -> flowGALI2ClosedMatch,
      "gali3_symbolic_match" -> flowGALI3ClosedMatch,
      "sali_sample_match" -> flowSALISampleMatch,
      "gali2_sample_match" -> flowGALI2SampleMatch,
      "gali3_sample_match" -> flowGALI3SampleMatch
    |>
  ],
  MakeTest[
    "lyapunov_gap_limits_match_exact_rates",
    hyperbolicMapSALIRatioLimit == 1/4 &&
      hyperbolicMapGALI2RatioLimit == 1/4 &&
      flowSALILogRate == 1 && flowGALI2LogRate == 1 &&
      flowGALI3LogRate == 3
  ],
  MakeTest[
    "signed_scale_and_permutation_invariance",
    TrueQ[FullSimplify[
      rotationSignedScaledSALIExact == rotationSALIExact
    ]] && TrueQ[FullSimplify[
      rotationSignedScaledGALI3Exact == rotationGALI3GramExact
    ]] && TrueQ[FullSimplify[
      rotationPermutedGALI3Exact == rotationGALI3GramExact
    ]]
  ],
  MakeTest[
    "individual_normalization_is_not_gram_schmidt",
    First[rotationSALIExact] == 2/Sqrt[5] &&
      First[rotationGALI2GramExact] == 4/5 &&
      First[rotationGALI3GramExact] == 4/5 &&
      First[rotationGALI3GramExact] =!= 1
  ],
  MakeTest["hafo_source_not_read", source["hafo_source_read"] === False],
  MakeTest[
    "hafo_formula_not_imported",
    source["hafo_formula_imported"] === False
  ],
  MakeTest["report_not_used", source["report_input_used"] === False]
};

evidenceBoundary =
  "finite exact linear tangent-algebra and numerical consistency only; " <>
  "no general nonlinear chaos classification, convergence theorem, " <>
  "Lyapunov-spectrum validation, attraction, hiddenness, or fractional " <>
  "SALI/GALI claim; hyperbolic fixtures are not chaotic-attractor evidence";

summary = <|
  "system_id" -> systemID,
  "validation_scope" ->
    "independent_integer_SALI_GALI_exact_linear_tangent_consistency",
  "evidence_boundary" -> evidenceBoundary,
  "source" -> source,
  "conventions" -> <|
    "instantaneous_matrix_shape" -> "dimension_by_n_vectors_columns",
    "history_shape" -> "n_samples_by_n_vectors_by_dimension",
    "normalization" -> "independent_l2_normalization_per_vector",
    "orthogonalization_between_vectors" -> "none",
    "sali_definition" ->
      "min(norm(q1+q2),norm(q1-q2)) for independently normalized columns",
    "gali_definition" -> "sqrt(det(transpose(Q).Q))",
    "cauchy_binet_definition" ->
      "sqrt(sum of squared maximal minors of normalized Q)",
    "ldi_svd_definition" ->
      "product of singular values of independently normalized Q",
    "gali_orders" -> {2, 3},
    "threshold_rule" -> "strictly_less_than",
    "threshold" -> N[1/10, 24]
  |>,
  "fixtures" -> <|
    "orthogonal_rotation_map" -> <|
      "kind" -> "map",
      "matrix" -> N[rotationMapExact, 24],
      "iterations" -> rotationIterations,
      "initial_deviations_columns" ->
        N[rotationInitialDeviationsExact, 24],
      "tangent_matrices_columns" ->
        N[rotationTangentMatricesExact, 30],
      "tangent_history" ->
        N[Transpose /@ rotationTangentMatricesExact, 30],
      "normalized_tangent_matrices_columns" ->
        N[rotationNormalizedMatricesExact, 30],
      "sali" -> N[rotationSALIExact, 30],
      "gali_2" -> N[rotationGALI2GramExact, 30],
      "gali_3" -> N[rotationGALI3GramExact, 30],
      "ldi_2" -> N[rotationGALI2GramExact, 30],
      "ldi_3" -> N[rotationGALI3SVD, 30]
    |>,
    "hyperbolic_diagonal_map" -> <|
      "kind" -> "map",
      "matrix" -> N[hyperbolicMapExact, 24],
      "iterations" -> hyperbolicMapIterations,
      "initial_deviations_columns" ->
        N[hyperbolicMapInitialDeviationsExact, 30],
      "tangent_matrices_columns" ->
        N[hyperbolicMapTangentMatricesExact, 30],
      "tangent_history" ->
        N[Transpose /@ hyperbolicMapTangentMatricesExact, 30],
      "normalized_tangent_matrices_columns" ->
        N[hyperbolicMapNormalizedMatricesExact, 30],
      "sali" -> N[hyperbolicMapSALIExact, 30],
      "gali_2" -> N[hyperbolicMapGALI2GramExact, 30],
      "ldi_2" -> N[hyperbolicMapGALI2SVD, 30],
      "first_sali_below_one_tenth" ->
        hyperbolicMapFirstSALIThresholdIteration,
      "first_gali_2_below_one_tenth" ->
        hyperbolicMapFirstGALI2ThresholdIteration,
      "asymptotic_step_ratio" -> N[1/4, 24],
      "lyapunov_gap_per_iteration" -> N[Log[4], 30]
    |>,
    "hyperbolic_diagonal_flow" -> <|
      "kind" -> "flow",
      "generator" -> N[flowGeneratorExact, 24],
      "times" -> N[flowTimesExact, 30],
      "initial_deviations_columns" ->
        N[flowInitialDeviationsExact, 30],
      "tangent_matrices_columns" -> N[flowTangentMatricesExact, 30],
      "tangent_history" ->
        N[Transpose /@ flowTangentMatricesExact, 30],
      "normalized_tangent_matrices_columns" ->
        N[flowNormalizedMatricesExact, 30],
      "sali" -> N[flowSALIExact, 30],
      "gali_2" -> N[flowGALI2GramExact, 30],
      "gali_3" -> N[flowGALI3GramExact, 30],
      "ldi_2" -> N[flowGALI2CauchyExact, 30],
      "ldi_3" -> N[flowGALI3SVD, 30],
      "closed_log_decay_rates" -> <|
        "sali" -> flowSALILogRate,
        "gali_2" -> flowGALI2LogRate,
        "gali_3" -> flowGALI3LogRate
      |>
    |>
  |>,
  "numeric_cross_checks" -> <|
    "working_precision" -> workingPrecision,
    "rotation_gali3_gram_to_svd_ldi_max_residual" ->
      rotationSVDMaxResidual,
    "map_gali2_gram_to_svd_ldi_max_residual" ->
      hyperbolicMapSVDMaxResidual,
    "flow_gali3_gram_to_svd_ldi_max_residual" ->
      flowSVDMaxResidual
  |>,
  "tests" -> tests,
  "passed" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests)
|>;

ExportJSON[
  FileNameJoin[{outDir, systemID <> "_validation_summary.json"}],
  summary
];

ExitFromTests[tests];
