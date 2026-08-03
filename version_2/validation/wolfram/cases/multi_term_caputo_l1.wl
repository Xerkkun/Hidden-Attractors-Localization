(* ::Package:: *)

(* ============================================================= *)
(* Independent finite multi-term Caputo L1 validation            *)
(*                                                               *)
(* This script starts from the Caputo integral, derives every     *)
(* fractional L1 interval coefficient with Integrate, adds the    *)
(* exact alpha=1 backward-Euler term, and advances an independently*)
(* written scalar linear recurrence.                              *)
(*                                                               *)
(* The original term list is deliberately permuted, contains a    *)
(* duplicated order and one zero coefficient. Its canonical       *)
(* positive coefficients sum to 37/20, so silent unit-mass        *)
(* normalization cannot pass this case.                           *)
(*                                                               *)
(* No HAFO source, generated report, or HAFO formula is read.     *)
(* Evidence boundary: finite algebraic/numerical consistency only;*)
(* no convergence theorem, nonlinear stability, chaos, attraction,*)
(* or hiddenness claim.                                           *)
(* ============================================================= *)

ClearAll["Global`*"];

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];

systemID = "multi_term_caputo_l1";
outDir = EnsureDirectory[
  GetCommandOption[
    "--out",
    FileNameJoin[{
      Directory[], "validation", "outputs", "wolfram", systemID
    }]
  ]
];

source = <|
  "diethelm_ford_doi" -> "10.1016/S0096-3003(03)00739-2",
  "ren_sun_doi" -> "10.4208/EAJAM.181113.280514A",
  "she_li_sun_doi" -> "10.1016/j.matcom.2021.11.005",
  "zaky_machado_doi" -> "10.1016/j.camwa.2019.07.008",
  "scope" ->
    "finite multi-term Caputo identity, canonical sum, mixed alpha=1 kernel, and direct affine recurrence",
  "hafo_source_read" -> False,
  "report_input_used" -> False,
  "hafo_formula_imported" -> False
|>;

workingPrecision = 80;
lowerTerminalExact = 1/10;
stepExact = 1/16;
nSteps = 12;

(* Original zero-based source indices after Python serialization:
   alpha=1     <- {0}
   alpha=2/3   <- {1,3}
   alpha=1/3   <- {2}
   zero term   <- {4}. *)
originalOrdersExact = {1, 2/3, 1/3, 2/3, 1/2};
originalCoefficientsExact = {3/4, 1/5, 2/5, 1/2, 0};
canonicalOrdersExact = {1/3, 2/3, 1};
canonicalCoefficientsExact = {2/5, 7/10, 3/4};
expectedSourceIndicesZeroBased = {{2}, {1, 3}, {0}};
expectedZeroIndicesZeroBased = {4};
coefficientSumExact = Total[canonicalCoefficientsExact];

initialValueExact = 11/10;
slopeExact = 5/7;
lambdaExact = -2/5;
timesExact = Table[
  lowerTerminalExact + index stepExact,
  {index, 0, nSteps}
];

(* Independent interval integration for 0<alpha<1 and an explicit
   classical derivative branch at alpha=1. Derive one generic primitive
   with Integrate, then evaluate that primitive at each rational endpoint;
   this avoids repeating the same symbolic integration for every lag. *)
ClearAll[
  distance, orderSymbol, integratedTermCoefficient,
  formulaTermCoefficient
];
kernelPrimitiveGeneric = Integrate[
  distance^(-orderSymbol),
  distance,
  Assumptions -> distance > 0 && 0 < orderSymbol < 1,
  GenerateConditions -> False
];
integratedTermCoefficient[
  order_, coefficient_, lag_Integer?NonNegative
] := If[
  TrueQ[order == 1],
  If[lag == 0, coefficient/stepExact, 0],
  coefficient/(stepExact Gamma[1 - order]) *
    (
      (kernelPrimitiveGeneric /.
        orderSymbol -> order /.
        distance -> (lag + 1) stepExact) -
      (kernelPrimitiveGeneric /.
        orderSymbol -> order /.
        distance -> lag stepExact)
    )
];

formulaTermCoefficient[
  order_, coefficient_, lag_Integer?NonNegative
] := If[
  TrueQ[order == 1],
  If[lag == 0, coefficient/stepExact, 0],
  coefficient stepExact^(-order)/Gamma[2 - order] *
    ((lag + 1)^(1 - order) - lag^(1 - order))
];

canonicalIntegratedPerTermExact = Table[
  integratedTermCoefficient[
    canonicalOrdersExact[[termIndex]],
    canonicalCoefficientsExact[[termIndex]],
    lag
  ],
  {termIndex, 1, Length[canonicalOrdersExact]},
  {lag, 0, nSteps - 1}
];
canonicalFormulaPerTermExact = Table[
  formulaTermCoefficient[
    canonicalOrdersExact[[termIndex]],
    canonicalCoefficientsExact[[termIndex]],
    lag
  ],
  {termIndex, 1, Length[canonicalOrdersExact]},
  {lag, 0, nSteps - 1}
];
canonicalCombinedKernelExact = Total[canonicalIntegratedPerTermExact];

kernelSymbolicResiduals = FullSimplify[
  canonicalIntegratedPerTermExact - canonicalFormulaPerTermExact
];
kernelSymbolicMatch = And @@ (
  TrueQ[# == 0] & /@ Flatten[kernelSymbolicResiduals]
);

(* Independently sum the original permuted/split/zero list. *)
originalCombinedKernelExact = Total[Table[
  integratedTermCoefficient[
    originalOrdersExact[[termIndex]],
    originalCoefficientsExact[[termIndex]],
    lag
  ],
  {termIndex, 1, Length[originalOrdersExact]},
  {lag, 0, nSteps - 1}
]];
permutedCombinedKernelExact = Total[Reverse[canonicalIntegratedPerTermExact]];
canonicalizationResiduals = FullSimplify[
  originalCombinedKernelExact - canonicalCombinedKernelExact
];
permutationResiduals = FullSimplify[
  permutedCombinedKernelExact - canonicalCombinedKernelExact
];
canonicalizationSymbolicMatch = And @@ (
  TrueQ[# == 0] & /@ Join[
    canonicalizationResiduals,
    permutationResiduals
  ]
);

singleTermKernelExact = Table[
  formulaTermCoefficient[2/3, 7/10, lag],
  {lag, 0, nSteps - 1}
];
singleTermReductionMatch = And @@ (
  TrueQ[# == 0] & /@ FullSimplify[
    singleTermKernelExact - canonicalFormulaPerTermExact[[2]]
  ]
);
alphaOneKernelExact = canonicalFormulaPerTermExact[[3]];
alphaOneBackwardEulerMatch = TrueQ[
  alphaOneKernelExact[[1]] == (3/4)/stepExact
] && And @@ (TrueQ[# == 0] & /@ Rest[alphaOneKernelExact]);

coefficientSumPreserved = TrueQ[
  Total[originalCoefficientsExact] == coefficientSumExact == 37/20
];

(* Symbolic Caputo identity for a quadratic monomial, derived from the
   integral definition. The alpha=1 branch is checked by differentiation. *)
ClearAll[tSymbol, sSymbol, alphaSymbol];
quadraticCaputoIntegratedGeneric = FullSimplify[
  1/Gamma[1 - alphaSymbol] Integrate[
    (tSymbol - sSymbol)^(-alphaSymbol) *
      D[(sSymbol - lowerTerminalExact)^2, sSymbol],
    {sSymbol, lowerTerminalExact, tSymbol},
    Assumptions ->
      tSymbol > lowerTerminalExact && 0 < alphaSymbol < 1,
    GenerateConditions -> False
  ],
  Assumptions ->
    tSymbol > lowerTerminalExact && 0 < alphaSymbol < 1
];
quadraticCaputoClosedGeneric =
  2 (tSymbol - lowerTerminalExact)^(2 - alphaSymbol)/
    Gamma[3 - alphaSymbol];
quadraticFractionalIdentityMatch = TrueQ[FullSimplify[
  quadraticCaputoIntegratedGeneric - quadraticCaputoClosedGeneric,
  Assumptions ->
    tSymbol > lowerTerminalExact && 0 < alphaSymbol < 1
] == 0];
quadraticAlphaOneIdentityMatch = TrueQ[FullSimplify[
  D[(tSymbol - lowerTerminalExact)^2, tSymbol] -
    (quadraticCaputoClosedGeneric /. alphaSymbol -> 1)
] == 0];
quadraticMultiTermIntegrated =
  canonicalCoefficientsExact[[1]] *
      (quadraticCaputoIntegratedGeneric /. alphaSymbol -> 1/3) +
    canonicalCoefficientsExact[[2]] *
      (quadraticCaputoIntegratedGeneric /. alphaSymbol -> 2/3) +
    canonicalCoefficientsExact[[3]] *
      D[(tSymbol - lowerTerminalExact)^2, tSymbol];
quadraticMultiTermClosed = Sum[
  canonicalCoefficientsExact[[termIndex]] *
    2 (tSymbol - lowerTerminalExact)^
      (2 - canonicalOrdersExact[[termIndex]])/
    Gamma[3 - canonicalOrdersExact[[termIndex]]],
  {termIndex, 1, Length[canonicalOrdersExact]}
];
quadraticMultiTermIdentityMatch = TrueQ[FullSimplify[
  quadraticMultiTermIntegrated - quadraticMultiTermClosed,
  Assumptions -> tSymbol > lowerTerminalExact
] == 0];
quadraticProbeTimeExact = lowerTerminalExact + 3/7;
quadraticProbeValue = N[
  quadraticMultiTermClosed /. tSymbol -> quadraticProbeTimeExact,
  workingPrecision
];

(* Affine solution: L1 is exact for every fractional term and backward Euler
   is exact for its alpha=1 derivative. *)
ClearAll[exactState, exactTermDerivative, exactMultiTermDerivative, forcing];
exactState[time_] :=
  initialValueExact + slopeExact (time - lowerTerminalExact);
exactTermDerivative[time_, order_, coefficient_] := If[
  TrueQ[order == 1],
  coefficient slopeExact,
  If[
    TrueQ[time == lowerTerminalExact],
    0,
    coefficient slopeExact *
      (time - lowerTerminalExact)^(1 - order)/Gamma[2 - order]
  ]
];
exactMultiTermDerivative[time_] := Sum[
  exactTermDerivative[
    time,
    canonicalOrdersExact[[termIndex]],
    canonicalCoefficientsExact[[termIndex]]
  ],
  {termIndex, 1, Length[canonicalOrdersExact]}
];
forcing[time_] :=
  exactMultiTermDerivative[time] - lambdaExact exactState[time];

combinedKernel = N[canonicalCombinedKernelExact, workingPrecision];
currentCoefficient = combinedKernel[[1]];

(* Direct scalar solve, deliberately not a Picard iteration. *)
ClearAll[directLinearMultiTermL1Recurrence];
directLinearMultiTermL1Recurrence[] := Module[
  {trajectory, outputIndex, history, denominator},
  trajectory = ConstantArray[SetPrecision[0, workingPrecision], nSteps + 1];
  trajectory[[1]] = N[initialValueExact, workingPrecision];
  denominator = N[currentCoefficient - lambdaExact, workingPrecision];
  Do[
    history = If[
      outputIndex == 1,
      SetPrecision[0, workingPrecision],
      Sum[
        combinedKernel[[outputIndex - historyIndex]] *
          (trajectory[[historyIndex + 2]] -
            trajectory[[historyIndex + 1]]),
        {historyIndex, 0, outputIndex - 2}
      ]
    ];
    trajectory[[outputIndex + 1]] = N[
      (
        currentCoefficient trajectory[[outputIndex]] - history +
        forcing[timesExact[[outputIndex + 1]]]
      )/denominator,
      workingPrecision
    ],
    {outputIndex, 1, nSteps}
  ];
  trajectory
];

manufacturedTrajectory = directLinearMultiTermL1Recurrence[];
manufacturedExactStates = N[exactState /@ timesExact, workingPrecision];
manufacturedMaxExactError = N[
  Max[Abs[manufacturedTrajectory - manufacturedExactStates]],
  40
];
manufacturedResiduals = N[Table[
  Module[{history},
    history = If[
      outputIndex == 1,
      0,
      Sum[
        combinedKernel[[outputIndex - historyIndex]] *
          (manufacturedTrajectory[[historyIndex + 2]] -
            manufacturedTrajectory[[historyIndex + 1]]),
        {historyIndex, 0, outputIndex - 2}
      ]
    ];
    currentCoefficient *
      (manufacturedTrajectory[[outputIndex + 1]] -
        manufacturedTrajectory[[outputIndex]]) + history -
      lambdaExact manufacturedTrajectory[[outputIndex + 1]] -
      forcing[timesExact[[outputIndex + 1]]]
  ],
  {outputIndex, 1, nSteps}
], workingPrecision];
manufacturedMaxRecurrenceResidual = N[
  Max[Abs[manufacturedResiduals]],
  40
];

tests = {
  MakeTest[
    "coefficient_sum_is_37_over_20_without_normalization",
    coefficientSumPreserved,
    <|"coefficient_sum" -> coefficientSumExact|>
  ],
  MakeTest[
    "integrated_multiterm_l1_kernel_matches_closed_formula_including_alpha_one",
    kernelSymbolicMatch && alphaOneBackwardEulerMatch,
    <|
      "kernel_symbolic_match" -> kernelSymbolicMatch,
      "alpha_one_backward_euler_match" -> alphaOneBackwardEulerMatch
    |>
  ],
  MakeTest[
    "permutation_zero_and_duplicate_coalescence_are_invariant",
    canonicalizationSymbolicMatch,
    <|"canonicalization_symbolic_match" -> canonicalizationSymbolicMatch|>
  ],
  MakeTest[
    "single_term_fractional_kernel_reduction_matches",
    singleTermReductionMatch,
    <|"single_term_reduction_match" -> singleTermReductionMatch|>
  ],
  MakeTest[
    "quadratic_caputo_monomial_identity_matches",
    quadraticFractionalIdentityMatch && quadraticAlphaOneIdentityMatch &&
      quadraticMultiTermIdentityMatch,
    <|
      "fractional_match" -> quadraticFractionalIdentityMatch,
      "alpha_one_match" -> quadraticAlphaOneIdentityMatch,
      "multi_term_match" -> quadraticMultiTermIdentityMatch
    |>
  ],
  MakeTest[
    "manufactured_linear_recurrence_residual_is_small",
    manufacturedMaxRecurrenceResidual < 10^-35,
    <|"max_residual" -> manufacturedMaxRecurrenceResidual|>
  ],
  MakeTest[
    "manufactured_affine_trajectory_matches_exact_samples",
    manufacturedMaxExactError < 10^-35,
    <|"max_error" -> manufacturedMaxExactError|>
  ],
  MakeTest["hafo_source_not_read", source["hafo_source_read"] === False],
  MakeTest["hafo_formula_not_imported", source["hafo_formula_imported"] === False],
  MakeTest["report_not_used", source["report_input_used"] === False]
};

summary = <|
  "system_id" -> systemID,
  "validation_scope" ->
    "independent_finite_multi_term_Caputo_uniform_L1_consistency",
  "evidence_boundary" ->
    "finite algebraic-numerical consistency only; no convergence theorem, nonlinear stability, chaos, attraction, or hiddenness claim",
  "source" -> source,
  "parameters" -> <|
    "lower_terminal" -> N[lowerTerminalExact, 40],
    "step" -> N[stepExact, 40],
    "n_steps" -> nSteps,
    "original_orders" -> N[originalOrdersExact, 40],
    "original_coefficients" -> N[originalCoefficientsExact, 40],
    "canonical_orders" -> N[canonicalOrdersExact, 40],
    "canonical_coefficients" -> N[canonicalCoefficientsExact, 40],
    "expected_source_indices_zero_based" -> expectedSourceIndicesZeroBased,
    "expected_zero_indices_zero_based" -> expectedZeroIndicesZeroBased,
    "coefficient_sum" -> N[coefficientSumExact, 40],
    "initial_value" -> N[initialValueExact, 40],
    "slope" -> N[slopeExact, 40],
    "lambda" -> N[lambdaExact, 40]
  |>,
  "canonicalization" -> <|
    "definition" ->
      "drop exact zero coefficients, sort ascending, and coalesce exactly equal orders",
    "coefficient_normalization" -> "none",
    "coefficient_sum_exact" -> ExprString[coefficientSumExact],
    "coefficient_sum_preserved" -> coefficientSumPreserved,
    "symbolic_match" -> canonicalizationSymbolicMatch,
    "single_term_reduction_match" -> singleTermReductionMatch
  |>,
  "kernel" -> <|
    "definition" ->
      "sum_j c_j h^(-alpha_j)/Gamma(2-alpha_j) b_k(alpha_j), with alpha=1 backward-Euler delta",
    "per_term_integrated_values" -> N[canonicalIntegratedPerTermExact, 40],
    "per_term_formula_values" -> N[canonicalFormulaPerTermExact, 40],
    "combined_integrated_values" -> N[canonicalCombinedKernelExact, 40],
    "original_combined_values" -> N[originalCombinedKernelExact, 40],
    "permuted_combined_values" -> N[permutedCombinedKernelExact, 40],
    "single_term_values" -> N[singleTermKernelExact, 40],
    "symbolic_residuals" -> (ExprString /@ Flatten[kernelSymbolicResiduals]),
    "symbolic_match" -> kernelSymbolicMatch,
    "alpha_one_backward_euler_match" -> alphaOneBackwardEulerMatch,
    "current_step_coefficient" -> N[currentCoefficient, 40]
  |>,
  "quadratic_caputo_identity" -> <|
    "fractional_integrated_form" -> ExprString[quadraticCaputoIntegratedGeneric],
    "closed_form" -> ExprString[quadraticCaputoClosedGeneric],
    "fractional_symbolic_match" -> quadraticFractionalIdentityMatch,
    "alpha_one_symbolic_match" -> quadraticAlphaOneIdentityMatch,
    "multi_term_symbolic_match" -> quadraticMultiTermIdentityMatch,
    "probe_time" -> N[quadraticProbeTimeExact, 40],
    "probe_value" -> N[quadraticProbeValue, 40]
  |>,
  "manufactured_case" -> <|
    "exact_solution" -> "x(t)=11/10+(5/7)(t-1/10)",
    "rhs" ->
      "f(t,x)=lambda*x+sum_j c_j*CaputoD(alpha_j,x_exact)-lambda*x_exact(t)",
    "linear_recurrence" ->
      "(K0-lambda)x_n=K0*x_(n-1)-history_n+forcing(t_n)",
    "times" -> N[timesExact, 40],
    "forcing_values" -> N[forcing /@ timesExact, 40],
    "states" -> N[manufacturedTrajectory, 40],
    "exact_states" -> N[manufacturedExactStates, 40],
    "recurrence_residuals" -> N[manufacturedResiduals, 40],
    "max_recurrence_residual" -> manufacturedMaxRecurrenceResidual,
    "max_exact_error" -> manufacturedMaxExactError
  |>,
  "tests" -> tests,
  "passed" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests)
|>;

ExportJSON[
  FileNameJoin[{outDir, systemID <> "_validation_summary.json"}],
  summary
];

ExitFromTests[tests];
