(* ::Package:: *)

(* ============================================================= *)
(* Independent Grunwald--Letnikov operator and solver validation  *)
(*                                                               *)
(* Mathematical source anchors (the script does not read HAFO):  *)
(*                                                               *)
(* 1. I. Podlubny, Fractional Differential Equations, Academic   *)
(*    Press, 1999, ISBN 978-0-12-558840-9, Sections 2.2--2.4.    *)
(* 2. K. B. Oldham and J. Spanier, The Fractional Calculus,      *)
(*    Academic Press, 1974, ISBN 978-0-12-525550-9.               *)
(* 3. C. Lubich, "Discretized Fractional Calculus", SIAM J.      *)
(*    Math. Anal. 17 (1986), 704--719.                            *)
(*    DOI: 10.1137/0517050                                       *)
(* 4. A. V. Letnikov, "Theory of differentiation with an         *)
(*    arbitrary index", Mat. Sb. 3 (1868), 1--68.                *)
(*                                                               *)
(* The validation covers the unshifted finite-history GL sum,     *)
(* GL applied to x(t)-x(0) (the Caputo-compatible shift for       *)
(* 0<q<1), and the explicit discrete recurrence implemented by    *)
(* HAFO. It validates formulas and finite-grid convergence only;  *)
(* it does not certify chaos, hiddenness, or solver stability for *)
(* arbitrary nonlinear systems.                                  *)
(* ============================================================= *)

ClearAll["Global`*"];

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];

systemID = "gl_fractional_operator_validation";
outDir = EnsureDirectory[
  GetCommandOption["--out", FileNameJoin[{Directory[], "validation_outputs", systemID}]]
];

source = <|
  "podlubny_isbn" -> "978-0-12-558840-9",
  "oldham_spanier_isbn" -> "978-0-12-525550-9",
  "lubich_doi" -> "10.1137/0517050",
  "scope" -> "GL weights, sampled derivatives, and explicit scalar recurrences",
  "hafo_source_read" -> False,
  "report_input_used" -> False
|>;

$Assumptions = 0 < q < 1 && Element[{k, n, m}, Integers] &&
  k >= 0 && n >= 0 && m >= 1;

(* Recursive weights used independently of the Python/Numba code. *)
ClearAll[glWeights];
glWeights[order_, count_Integer?NonNegative] := Module[{weights, index},
  If[count == 0, Return[{}]];
  weights = ConstantArray[0, count];
  weights[[1]] = 1;
  Do[
    weights[[index + 1]] = weights[[index]] (1 - (order + 1)/index),
    {index, 1, count - 1}
  ];
  weights
];

(* Left-sided GL history on a uniform sampled series. *)
ClearAll[glDerivativeSeries];
glDerivativeSeries[samples_List, order_, step_, shiftInitial_] := Module[
  {weights, anchor, count},
  count = Length[samples];
  weights = glWeights[order, count];
  anchor = If[TrueQ[shiftInitial], First[samples], 0];
  Table[
    step^-order Sum[
      weights[[lag + 1]] (samples[[sampleIndex - lag]] - anchor),
      {lag, 0, sampleIndex - 1}
    ],
    {sampleIndex, 1, count}
  ]
];

(* Scalar version of HAFO's explicit recurrence, derived directly from
   h^-q Sum_{j=0}^n w_j (x_(n-j)-anchor) = f_(n-1). *)
ClearAll[glExplicitScalar];
glExplicitScalar[forcing_, order_, step_, nSteps_Integer?Positive,
  xInitial_, shiftInitial_] := Module[
  {weights, states, anchor, history, sampleIndex},
  weights = glWeights[order, nSteps + 1];
  states = ConstantArray[0, nSteps + 1];
  states[[1]] = xInitial;
  anchor = If[TrueQ[shiftInitial], xInitial, 0];
  Do[
    history = Sum[
      weights[[lag + 1]] (states[[sampleIndex - lag + 1]] - anchor),
      {lag, 1, sampleIndex}
    ];
    states[[sampleIndex + 1]] = anchor + step^order
      forcing[(sampleIndex - 1) step, states[[sampleIndex]]] - history,
    {sampleIndex, 1, nSteps}
  ];
  states
];

(* ----------------------------------------------------------------- *)
(* 1. Exact weight identities.                                       *)
(* ----------------------------------------------------------------- *)

symbolicWeightCount = 10;
recursiveSymbolicWeights = glWeights[q, symbolicWeightCount];
closedSymbolicWeights = Table[(-1)^k Binomial[q, k],
  {k, 0, symbolicWeightCount - 1}];
weightSymbolicResidual = FullSimplify[
  recursiveSymbolicWeights - closedSymbolicWeights,
  Assumptions -> 0 < q < 1
];

(* The finite sum identity is established by its base case and induction
   step, avoiding any reliance on a numerical summation routine:
   Sum_{k=0}^n (-1)^k Binomial[q,k] = (-1)^n Binomial[q-1,n]. *)
constantSumBaseResidual = FullSimplify[
  1 - (-1)^0 Binomial[q - 1, 0],
  Assumptions -> 0 < q < 1
];
constantSumStepResidual = FullSimplify[
  (-1)^n Binomial[q - 1, n] + (-1)^(n + 1) Binomial[q, n + 1] -
    (-1)^(n + 1) Binomial[q - 1, n + 1],
  Assumptions -> 0 < q < 1 && Element[n, Integers] && n >= 0
];

(* For 0<q<1 and m>=1, the Caputo integral of t^m reduces through
   the Euler beta integral to
   m t^(m-q) Beta[m,1-q]/Gamma[1-q]. *)
caputoMonomialCoefficientResidual = FullSimplify[
  FunctionExpand[m Beta[m, 1 - q]/Gamma[1 - q]] -
    Gamma[m + 1]/Gamma[m + 1 - q],
  Assumptions -> 0 < q < 1 && Element[m, Integers] && m >= 1
];

(* The RL derivative of a constant is obtained by differentiating its
   fractional integral c t^(1-q)/Gamma[2-q]. *)
rlConstantSymbolicResidual = FullSimplify[
  D[c t^(1 - q)/Gamma[2 - q], t] - c t^-q/Gamma[1 - q],
  Assumptions -> 0 < q < 1 && t > 0 && Element[c, Reals]
];

monomialQ1LimitResidual = Block[{$Assumptions = True},
  FullSimplify[
    Limit[
      Gamma[m + 1] t^(m - q)/Gamma[m + 1 - q],
      q -> 1,
      Direction -> "FromBelow",
      Assumptions -> t > 0 && Element[m, Integers] && m >= 1
    ] - m t^(m - 1),
    Assumptions -> t > 0 && Element[m, Integers] && m >= 1
  ]
];

sampleOrderExact = 3/5;
sampleOrder = N[sampleOrderExact, 50];
weightSample = N[glWeights[sampleOrder, 16], 30];

(* ----------------------------------------------------------------- *)
(* 2. Monomial and constant derivatives at t=1.                     *)
(* ----------------------------------------------------------------- *)

resolutionGrid = {32, 64, 128, 256, 512};
monomialDegree = 3;
monomialExact = N[
  Gamma[monomialDegree + 1]/Gamma[monomialDegree + 1 - sampleOrderExact],
  40
];
monomialRows = Table[
  Module[{step = N[1/resolution, 50], value, error},
    value = N[
      step^-sampleOrder Sum[
        (-1)^lag Binomial[sampleOrder, lag]
          (1 - lag step)^monomialDegree,
        {lag, 0, resolution}
      ],
      40
    ];
    error = N[Abs[value - monomialExact], 30];
    <|
      "n_steps" -> resolution,
      "step" -> N[step, 20],
      "value" -> value,
      "analytic" -> monomialExact,
      "abs_error" -> error
    |>
  ],
  {resolution, resolutionGrid}
];

constantValueExact = 7/5;
constantValue = N[constantValueExact, 50];
constantRLExact = N[constantValueExact/Gamma[1 - sampleOrderExact], 40];
constantRows = Table[
  Module[{step = N[1/resolution, 50], rawValue, shiftedValues, error},
    rawValue = N[
      step^-sampleOrder constantValue Total[
        glWeights[sampleOrder, resolution + 1]
      ],
      40
    ];
    shiftedValues = glDerivativeSeries[
      ConstantArray[constantValue, resolution + 1],
      sampleOrder,
      step,
      True
    ];
    error = N[Abs[rawValue - constantRLExact], 30];
    <|
      "n_steps" -> resolution,
      "step" -> N[step, 20],
      "raw_value" -> rawValue,
      "rl_analytic" -> constantRLExact,
      "raw_abs_error" -> error,
      "caputo_shifted_max_abs" -> N[Max[Abs[shiftedValues]], 30]
    |>
  ],
  {resolution, resolutionGrid}
];

(* ----------------------------------------------------------------- *)
(* 3. Integer limit q=1 for the sampled operator.                    *)
(* ----------------------------------------------------------------- *)

q1Step = 1/10;
q1Times = Table[index q1Step, {index, 0, 10}];
q1Samples = (1 + # + #^3) & /@ q1Times;
q1RawDerivative = glDerivativeSeries[q1Samples, 1, q1Step, False];
q1ShiftedDerivative = glDerivativeSeries[q1Samples, 1, q1Step, True];
q1BackwardDifferences = Table[
  (q1Samples[[index]] - q1Samples[[index - 1]])/q1Step,
  {index, 2, Length[q1Samples]}
];
q1RawExpected = Join[{First[q1Samples]/q1Step}, q1BackwardDifferences];
q1ShiftedExpected = Join[{0}, q1BackwardDifferences];
q1WeightResidual = glWeights[1, 12] - Join[{1, -1}, ConstantArray[0, 10]];
q1RawResidual = FullSimplify[q1RawDerivative - q1RawExpected];
q1ShiftedResidual = FullSimplify[q1ShiftedDerivative - q1ShiftedExpected];

(* ----------------------------------------------------------------- *)
(* 4. Explicit solver recurrence.                                    *)
(* ----------------------------------------------------------------- *)

solverInitial = N[2, 50];
solverExactAtOne = N[
  2 + 1/Gamma[1 + sampleOrderExact],
  40
];
solverRows = Table[
  Module[{step = N[1/resolution, 50], states, value, error},
    states = glExplicitScalar[(1 + 0 #1 + 0 #2 &), sampleOrder, step, resolution,
      solverInitial, True];
    value = N[Last[states], 40];
    error = N[Abs[value - solverExactAtOne], 30];
    <|
      "n_steps" -> resolution,
      "step" -> N[step, 20],
      "value" -> value,
      "analytic" -> solverExactAtOne,
      "abs_error" -> error
    |>
  ],
  {resolution, resolutionGrid}
];

q1SolverStep = 1/20;
q1SolverSteps = 20;
q1Lambda = -2/5;
q1SolverInitial = 7/5;
q1EulerExpected = Table[
  q1SolverInitial (1 + q1Lambda q1SolverStep)^index,
  {index, 0, q1SolverSteps}
];
q1SolverShifted = glExplicitScalar[
  (q1Lambda #2 &), 1, q1SolverStep, q1SolverSteps,
  q1SolverInitial, True
];
q1SolverRaw = glExplicitScalar[
  (q1Lambda #2 &), 1, q1SolverStep, q1SolverSteps,
  q1SolverInitial, False
];
q1SolverShiftedResidual = FullSimplify[q1SolverShifted - q1EulerExpected];
q1SolverRawResidual = FullSimplify[q1SolverRaw - q1EulerExpected];

monomialErrors = Lookup[monomialRows, "abs_error"];
constantErrors = Lookup[constantRows, "raw_abs_error"];
solverErrors = Lookup[solverRows, "abs_error"];

tests = {
  MakeTest[
    "weights_match_signed_generalized_binomial",
    TrueQ[weightSymbolicResidual == ConstantArray[0, symbolicWeightCount]]
  ],
  MakeTest[
    "constant_partial_sum_identity_base",
    TrueQ[constantSumBaseResidual == 0]
  ],
  MakeTest[
    "constant_partial_sum_identity_induction_step",
    TrueQ[constantSumStepResidual == 0]
  ],
  MakeTest[
    "caputo_monomial_beta_gamma_identity",
    TrueQ[caputoMonomialCoefficientResidual == 0]
  ],
  MakeTest[
    "riemann_liouville_constant_symbolic_identity",
    TrueQ[rlConstantSymbolicResidual == 0]
  ],
  MakeTest[
    "monomial_continuum_formula_has_q1_limit",
    TrueQ[monomialQ1LimitResidual == 0]
  ],
  MakeTest[
    "caputo_shifted_t_cubed_converges",
    And @@ Thread[Differences[monomialErrors] < 0] && Last[monomialErrors] < 35/10000,
    <|"finest_abs_error" -> Last[monomialErrors]|>
  ],
  MakeTest[
    "raw_constant_converges_to_riemann_liouville_value",
    And @@ Thread[Differences[constantErrors] < 0] && Last[constantErrors] < 2/10000,
    <|"finest_abs_error" -> Last[constantErrors]|>
  ],
  MakeTest[
    "caputo_shifted_constant_is_zero",
    Max[Lookup[constantRows, "caputo_shifted_max_abs"]] == 0
  ],
  MakeTest[
    "q1_weights_are_first_backward_difference",
    TrueQ[q1WeightResidual == ConstantArray[0, 12]]
  ],
  MakeTest[
    "q1_raw_operator_is_backward_difference_after_initial_sample",
    TrueQ[q1RawResidual == ConstantArray[0, Length[q1Samples]]]
  ],
  MakeTest[
    "q1_shifted_operator_is_backward_difference_after_initial_sample",
    TrueQ[q1ShiftedResidual == ConstantArray[0, Length[q1Samples]]]
  ],
  MakeTest[
    "fractional_constant_forcing_solver_converges",
    And @@ Thread[Differences[solverErrors] < 0] && Last[solverErrors] < 5/10000,
    <|"finest_abs_error" -> Last[solverErrors]|>
  ],
  MakeTest[
    "q1_shifted_solver_is_explicit_euler",
    TrueQ[q1SolverShiftedResidual == ConstantArray[0, q1SolverSteps + 1]]
  ],
  MakeTest[
    "q1_raw_solver_is_explicit_euler",
    TrueQ[q1SolverRawResidual == ConstantArray[0, q1SolverSteps + 1]]
  ],
  MakeTest["hafo_source_not_read", source["hafo_source_read"] === False],
  MakeTest["report_not_used", source["report_input_used"] === False]
};

summary = <|
  "system_id" -> systemID,
  "validation_scope" -> "independent_GL_operator_and_scalar_recurrence",
  "evidence_boundary" ->
    "finite-grid algebra and convergence; no chaos, hiddenness, or general nonlinear stability claim",
  "source" -> source,
  "parameters" -> <|
    "order" -> N[sampleOrder, 20],
    "monomial_degree" -> monomialDegree,
    "constant_value" -> N[constantValue, 20],
    "resolution_grid" -> resolutionGrid
  |>,
  "weight_identity" -> <|
    "closed_form" -> "w_k=(-1)^k Binomial[q,k]",
    "recurrence" -> "w_0=1; w_k=w_(k-1)*(1-(q+1)/k)",
    "sample" -> weightSample,
    "symbolic_residual" -> ExprString[weightSymbolicResidual],
    "constant_partial_sum" ->
      "Sum[k=0..n] w_k=(-1)^n Binomial[q-1,n]"
  |>,
  "monomial_caputo_shifted" -> <|
    "function" -> "t^3",
    "symbolic_formula" ->
      "D_C^q t^m=Gamma[m+1]/Gamma[m+1-q] t^(m-q), m>=1",
    "symbolic_coefficient_residual" ->
      ExprString[caputoMonomialCoefficientResidual],
    "q1_limit_residual" -> ExprString[monomialQ1LimitResidual],
    "analytic_at_t1" -> monomialExact,
    "rows" -> monomialRows
  |>,
  "constant_raw_gl" -> <|
    "function" -> "7/5",
    "symbolic_formula" -> "D_RL^q c=c t^(-q)/Gamma[1-q]",
    "symbolic_residual" -> ExprString[rlConstantSymbolicResidual],
    "rl_analytic_at_t1" -> constantRLExact,
    "rows" -> constantRows
  |>,
  "q1_operator" -> <|
    "step" -> N[q1Step, 20],
    "times" -> N[q1Times, 20],
    "samples" -> N[q1Samples, 20],
    "raw_values" -> N[q1RawDerivative, 20],
    "shifted_values" -> N[q1ShiftedDerivative, 20],
    "backward_differences" -> N[q1BackwardDifferences, 20]
  |>,
  "fractional_solver" -> <|
    "equation" -> "D_C^q x=1, x(0)=2",
    "analytic_at_t1" -> solverExactAtOne,
    "rows" -> solverRows
  |>,
  "q1_solver" -> <|
    "equation" -> "x'=lambda*x",
    "lambda" -> N[q1Lambda, 20],
    "initial" -> N[q1SolverInitial, 20],
    "step" -> N[q1SolverStep, 20],
    "n_steps" -> q1SolverSteps,
    "euler_values" -> N[q1EulerExpected, 20],
    "shifted_values" -> N[q1SolverShifted, 20],
    "raw_values" -> N[q1SolverRaw, 20]
  |>,
  "tests" -> tests,
  "passed" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests)
|>;

ExportJSON[
  FileNameJoin[{outDir, systemID <> "_summary.json"}],
  summary
];

ExitFromTests[tests];
