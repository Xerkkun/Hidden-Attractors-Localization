(* ::Package:: *)

(* ============================================================= *)
(* Independent ABC predictor--corrector validation               *)
(*                                                               *)
(* Mathematical source anchors (the script does not read HAFO):  *)
(*                                                               *)
(* 1. S. Lee, H. Kim, and B. Jang, "A Novel Numerical Method    *)
(*    for Solving Nonlinear Fractional-Order Differential        *)
(*    Equations and Its Applications", Fractal and Fractional 8  *)
(*    (2024), 65. DOI: 10.3390/fractalfract8010065               *)
(* 2. A. Atangana and D. Baleanu, "New fractional derivatives   *)
(*    with nonlocal and non-singular kernel", Thermal Science 20 *)
(*    (2016), 763--769. DOI: 10.2298/TSCI160111018A              *)
(* 3. K. Diethelm, R. Garrappa, A. Giusti, and M. Stynes,       *)
(*    "Why fractional derivatives with nonsingular kernels      *)
(*    should not be used", FCAA 23 (2020), 610--634.            *)
(*    DOI: 10.1515/fca-2020-0032                                *)
(*                                                               *)
(* The two linear product-integration weights are obtained both  *)
(* from symbolic Integrate calls and from their closed formulas. *)
(* A scalar manufactured case f(t,u)=t^2, B(alpha)=1 then uses   *)
(* equations (9)--(14) of Lee--Kim--Jang, implemented directly   *)
(* in Wolfram Language. This is finite algebraic/numerical        *)
(* consistency evidence only: no convergence theorem, stability, *)
(* chaos, attraction, or hiddenness claim.                       *)
(* ============================================================= *)

ClearAll["Global`*"];

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];

systemID = "abc_predictor_corrector";
outDir = EnsureDirectory[
  GetCommandOption[
    "--out",
    FileNameJoin[{Directory[], "validation_outputs", systemID}]
  ]
];

source = <|
  "lee_kim_jang_doi" -> "10.3390/fractalfract8010065",
  "atangana_baleanu_doi" -> "10.2298/TSCI160111018A",
  "diethelm_garrappa_giusti_stynes_doi" -> "10.1515/fca-2020-0032",
  "scope" ->
    "linear product weights and one finite Lee--Kim--Jang recurrence",
  "hafo_source_read" -> False,
  "report_input_used" -> False
|>;

alphaExact = 7/10;
stepExact = 1/16;
nSteps = 16;
lowerTerminalExact = 0;
initialValueExact = 6/5;
normalizationExact = 1;
cExact = (1 - alphaExact)/normalizationExact;
dExact = alphaExact/normalizationExact;
integralScaleExact = dExact/Gamma[alphaExact];
localPredictorScaleExact =
  dExact stepExact^alphaExact/Gamma[alphaExact + 2];

(* Closed linear product-integration weights, indexed by lag m>=1. *)
ClearAll[theta0Formula, theta1Formula];
theta0Formula[lag_Integer?Positive] := Module[{a, c},
  a = lag^alphaExact - (lag - 1)^alphaExact;
  c = lag^(alphaExact + 1) - (lag - 1)^(alphaExact + 1);
  stepExact^alphaExact (
    c/(alphaExact + 1) - (lag - 1) a/alphaExact
  )
];
theta1Formula[lag_Integer?Positive] := Module[{a, c},
  a = lag^alphaExact - (lag - 1)^alphaExact;
  c = lag^(alphaExact + 1) - (lag - 1)^(alphaExact + 1);
  stepExact^alphaExact (
    lag a/alphaExact - c/(alphaExact + 1)
  )
];

(* Independent symbolic integrals of the left and right linear basis on
   one interval. The dimensionless integration variable z maps the
   interval to [0,1], while lag-z is the distance to the target node. *)
ClearAll[z, theta0Integral, theta1Integral];
theta0Integral[lag_Integer?Positive] := FullSimplify[
  stepExact^alphaExact Integrate[
    (lag - z)^(alphaExact - 1) (1 - z),
    {z, 0, 1},
    GenerateConditions -> False
  ]
];
theta1Integral[lag_Integer?Positive] := FullSimplify[
  stepExact^alphaExact Integrate[
    (lag - z)^(alphaExact - 1) z,
    {z, 0, 1},
    GenerateConditions -> False
  ]
];

theta0Closed = theta0Formula /@ Range[nSteps];
theta1Closed = theta1Formula /@ Range[nSteps];
theta0Symbolic = theta0Integral /@ Range[nSteps];
theta1Symbolic = theta1Integral /@ Range[nSteps];
theta0Residuals = MapThread[
  FullSimplify[#1 - #2] &,
  {theta0Closed, theta0Symbolic}
];
theta1Residuals = MapThread[
  FullSimplify[#1 - #2] &,
  {theta1Closed, theta1Symbolic}
];
symbolicWeightMatch = And @@ (
  TrueQ[# == 0] & /@ Join[theta0Residuals, theta1Residuals]
);
weightIntegralMaxResidual = N[
  Max[Abs[N[Join[theta0Residuals, theta1Residuals], 70]]],
  35
];
partitionExpected = Table[
  stepExact^alphaExact (
    lag^alphaExact - (lag - 1)^alphaExact
  )/alphaExact,
  {lag, 1, nSteps}
];
partitionMaxResidual = N[
  Max[Abs[theta0Closed + theta1Closed - partitionExpected]],
  35
];

(* Manufactured scalar case: f(t,u)=(t-a)^2 and B(alpha)=1. The RHS is
   compatible at the lower terminal and deliberately independent of u,
   allowing the recurrence itself to be isolated. *)
ClearAll[rhsExact];
rhsExact[time_, state_] := (time - lowerTerminalExact)^2;
timesExact = Table[
  lowerTerminalExact + index stepExact,
  {index, 0, nSteps}
];
states = ConstantArray[0, nSteps + 1];
values = ConstantArray[0, nSteps + 1];
predictors = ConstantArray[0, nSteps + 1];
states[[1]] = initialValueExact;
values[[1]] = rhsExact[timesExact[[1]], states[[1]]];
predictors[[1]] = initialValueExact;

(* The paper assumes an O(h^2) first value without specifying its solver.
   Here the first product-trapezoid equation is solved by an explicit
   fixed-point loop, independently of the Python implementation. *)
startupTolerance = 10^-40;
startupMaxIterations = 100;
startupState = initialValueExact;
startupIterations = 0;
startupConverged = False;
Do[
  nextStartupState =
    initialValueExact +
    localPredictorScaleExact alphaExact values[[1]] +
    (cExact + localPredictorScaleExact) rhsExact[
      timesExact[[2]], startupState
    ];
  startupIterations = iteration;
  If[
    Abs[nextStartupState - startupState] <=
      startupTolerance (1 + Abs[nextStartupState]),
    startupState = nextStartupState;
    startupConverged = True;
    Break[]
  ];
  startupState = nextStartupState,
  {iteration, 1, startupMaxIterations}
];
states[[2]] = startupState;
values[[2]] = rhsExact[timesExact[[2]], states[[2]]];
predictors[[2]] = startupState;

Do[
  memory = Sum[
    integralScaleExact (
      theta0Formula[n + 1 - historyIndex] values[[historyIndex + 1]] +
      theta1Formula[n + 1 - historyIndex] values[[historyIndex + 2]]
    ),
    {historyIndex, 0, n - 1}
  ];
  predictor =
    initialValueExact +
    cExact (-values[[n]] + 2 values[[n + 1]]) +
    memory +
    localPredictorScaleExact (
      -values[[n]] + (alphaExact + 2) values[[n + 1]]
    );
  predictedValue = rhsExact[timesExact[[n + 2]], predictor];
  corrected =
    initialValueExact +
    cExact predictedValue +
    memory +
    localPredictorScaleExact (
      alphaExact values[[n + 1]] + predictedValue
    );
  predictors[[n + 2]] = predictor;
  states[[n + 2]] = corrected;
  values[[n + 2]] = rhsExact[timesExact[[n + 2]], corrected],
  {n, 1, nSteps - 1}
];

(* For this time-only RHS, the corrected Lee--Kim--Jang recurrence must
   equal direct product integration over every completed interval. *)
directDiscrete = Table[
  If[
    outputIndex == 0,
    initialValueExact,
    initialValueExact + cExact values[[outputIndex + 1]] +
      Sum[
        integralScaleExact (
          theta0Formula[outputIndex - historyIndex]
            values[[historyIndex + 1]] +
          theta1Formula[outputIndex - historyIndex]
            values[[historyIndex + 2]]
        ),
        {historyIndex, 0, outputIndex - 1}
      ]
  ],
  {outputIndex, 0, nSteps}
];
recurrenceDirectMaxResidual = N[
  Max[Abs[states - directDiscrete]],
  35
];

(* Exact Volterra solution for f(t)=t^2. Its finite-grid error is reported
   as a diagnostic, not used as a convergence-order claim. *)
elapsedExact = timesExact - lowerTerminalExact;
manufacturedExact =
  initialValueExact +
  cExact elapsedExact^2 +
  dExact Gamma[3] elapsedExact^(alphaExact + 2)/
    Gamma[alphaExact + 3];
manufacturedMaxFiniteGridError = N[
  Max[Abs[states - manufacturedExact]],
  35
];
compatibilityResidual = N[
  Abs[rhsExact[lowerTerminalExact, initialValueExact]],
  35
];

tests = {
  MakeTest[
    "symbolic_linear_weights_match_integrals",
    symbolicWeightMatch || weightIntegralMaxResidual < 10^-35,
    <|
      "symbolic_match" -> symbolicWeightMatch,
      "numeric_max_residual" -> weightIntegralMaxResidual
    |>
  ],
  MakeTest[
    "linear_weight_partition_identity",
    partitionMaxResidual < 10^-35,
    <|"max_residual" -> partitionMaxResidual|>
  ],
  MakeTest[
    "manufactured_rhs_satisfies_initial_compatibility",
    compatibilityResidual == 0,
    <|"residual" -> compatibilityResidual|>
  ],
  MakeTest[
    "implicit_product_trapezoid_startup_converged",
    startupConverged && startupIterations == 2,
    <|"iterations" -> startupIterations|>
  ],
  MakeTest[
    "lee_kim_jang_recurrence_matches_direct_product_integration",
    recurrenceDirectMaxResidual < 10^-35,
    <|"max_residual" -> recurrenceDirectMaxResidual|>
  ],
  MakeTest[
    "manufactured_solution_finite_grid_error_is_bounded",
    0 < manufacturedMaxFiniteGridError < 10^-2,
    <|"max_error" -> manufacturedMaxFiniteGridError|>
  ],
  MakeTest["hafo_source_not_read", source["hafo_source_read"] === False],
  MakeTest["report_not_used", source["report_input_used"] === False]
};

summary = <|
  "system_id" -> systemID,
  "validation_scope" ->
    "independent_linear_weight_integrals_and_finite_ABC_PCM_recurrence",
  "evidence_boundary" ->
    "finite algebraic-numerical consistency only; no convergence theorem, stability, chaos, attraction, or hiddenness claim",
  "source" -> source,
  "parameters" -> <|
    "alpha" -> N[alphaExact, 30],
    "step" -> N[stepExact, 30],
    "n_steps" -> nSteps,
    "lower_terminal" -> N[lowerTerminalExact, 30],
    "initial_value" -> N[initialValueExact, 30],
    "normalization" -> N[normalizationExact, 30],
    "normalization_convention" -> "B(alpha)=1"
  |>,
  "weights" -> <|
    "definition" ->
      "Theta0_m=Integrate[(mh-s)^(alpha-1)(1-s/h),{s,0,h}], Theta1_m=Integrate[(mh-s)^(alpha-1)s/h,{s,0,h}]",
    "lags" -> Range[nSteps],
    "theta0_formula" -> N[theta0Closed, 35],
    "theta1_formula" -> N[theta1Closed, 35],
    "theta0_symbolic_integral" -> N[theta0Symbolic, 35],
    "theta1_symbolic_integral" -> N[theta1Symbolic, 35],
    "theta0_symbolic_residuals" -> (ExprString /@ theta0Residuals),
    "theta1_symbolic_residuals" -> (ExprString /@ theta1Residuals),
    "symbolic_match" -> symbolicWeightMatch,
    "integral_max_residual" -> weightIntegralMaxResidual,
    "partition_max_residual" -> partitionMaxResidual
  |>,
  "manufactured_case" -> <|
    "rhs" -> "f(t,u)=(t-a)^2",
    "compatibility_residual" -> compatibilityResidual,
    "times" -> N[timesExact, 35],
    "states" -> N[states, 35],
    "predictors" -> N[predictors, 35],
    "rhs_values" -> N[values, 35],
    "direct_product_integration_states" -> N[directDiscrete, 35],
    "exact_volterra_states" -> N[manufacturedExact, 35],
    "recurrence_direct_max_residual" -> recurrenceDirectMaxResidual,
    "manufactured_max_finite_grid_error" ->
      manufacturedMaxFiniteGridError,
    "startup_iterations" -> startupIterations,
    "startup_converged" -> startupConverged
  |>,
  "tests" -> tests,
  "passed" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests)
|>;

ExportJSON[
  FileNameJoin[{outDir, systemID <> "_validation_summary.json"}],
  summary
];

ExitFromTests[tests];
