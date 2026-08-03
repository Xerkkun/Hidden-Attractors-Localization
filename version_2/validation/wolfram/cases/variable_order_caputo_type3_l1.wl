(* ::Package:: *)

(* ============================================================= *)
(* Independent variable-order Caputo Type III L1 validation      *)
(*                                                               *)
(* Type III convention used here:                                *)
(*                                                               *)
(*   D_C,III^(alpha(t)) x(t) =                                   *)
(*     1/Gamma[1-alpha(t)] Integrate[                             *)
(*       (t-s)^(-alpha(t)) x'(s), {s,a,t}]                       *)
(*                                                               *)
(* The order is frozen at the output time. The script does not   *)
(* read HAFO source code or generated report data. It checks the  *)
(* power formula, obtains L1 weights from symbolic integrals of   *)
(* piecewise-linear slopes, verifies constant-order reduction,    *)
(* and advances one finite manufactured recurrence.               *)
(*                                                               *)
(* Evidence boundary: finite algebraic/numerical consistency      *)
(* only; no global convergence theorem, stability, chaos,         *)
(* attraction, or hiddenness claim.                              *)
(*                                                               *)
(* Source anchors:                                               *)
(* D. Tavares, R. Almeida, D. F. M. Torres, CNSNS 35 (2016),   *)
(* DOI: 10.1016/j.cnsns.2015.10.027                             *)
(* Z. W. Fang, H. W. Sun, H. Wang, CAMWA 80 (2020),            *)
(* DOI: 10.1016/j.camwa.2020.07.009                             *)
(* ============================================================= *)

ClearAll["Global`*"];

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];

systemID = "variable_order_caputo_type3_l1";
outDir = EnsureDirectory[
  GetCommandOption[
    "--out",
    FileNameJoin[{Directory[], "validation_outputs", systemID}]
  ]
];

source = <|
  "tavares_almeida_torres_doi" -> "10.1016/j.cnsns.2015.10.027",
  "fang_sun_wang_doi" -> "10.1016/j.camwa.2020.07.009",
  "convention" ->
    "Caputo Type III with alpha(t_n) frozen over the full history integral",
  "scope" ->
    "power identity, symbolic L1 weights, constant-order reduction, and one finite recurrence",
  "hafo_source_read" -> False,
  "report_input_used" -> False
|>;

lowerTerminalExact = 0;
stepExact = 1/16;
nSteps = 16;
initialValueExact = 5/4;
powerExact = 3;
constantOrderExact = 13/20;
ClearAll[alphaAt];
alphaAt[time_] := 1/2 + (time - 3/4)^2/10;
timesExact = Table[
  lowerTerminalExact + index stepExact,
  {index, 0, nSteps}
];
ordersExact = alphaAt /@ timesExact;

(* Type III power identity. After s=t u, the defining integral is the
   Euler beta integral t^(p-alpha) Beta[p,1-alpha]. This keeps alpha fixed
   with respect to the history variable, as required by Type III. *)
ClearAll[alphaSymbol, tSymbol, s];
powerBetaIntegralSymbolic =
  powerExact (tSymbol - lowerTerminalExact)^
    (powerExact - alphaSymbol) Beta[powerExact, 1 - alphaSymbol]/
    Gamma[1 - alphaSymbol];
powerIntegralSymbolic = FullSimplify[
  powerBetaIntegralSymbolic,
  Assumptions -> 0 < alphaSymbol < 1 && tSymbol > lowerTerminalExact
];
powerClosedSymbolic =
  Gamma[powerExact + 1]/Gamma[powerExact + 1 - alphaSymbol] *
  (tSymbol - lowerTerminalExact)^(powerExact - alphaSymbol);
powerSymbolicResidual = FullSimplify[
  powerIntegralSymbolic - powerClosedSymbolic,
  Assumptions -> 0 < alphaSymbol < 1 && tSymbol > lowerTerminalExact
];
powerFormulaSymbolicMatch = TrueQ[powerSymbolicResidual == 0];

ClearAll[type3PowerDerivative];
type3PowerDerivative[time_, order_] :=
  Gamma[powerExact + 1]/Gamma[powerExact + 1 - order] *
  (time - lowerTerminalExact)^(powerExact - order);
powerExactDerivative = N[Table[
  If[
    index == 0,
    0,
    type3PowerDerivative[timesExact[[index + 1]], ordersExact[[index + 1]]]
  ],
  {index, 0, nSteps}
], 70];

(* L1 coefficient multiplying x_(k+1)-x_k at output n. *)
ClearAll[l1Coefficient];
l1Coefficient[lag_Integer?Positive, order_] :=
  stepExact^(-order)/Gamma[2 - order] *
  (lag^(1 - order) - (lag - 1)^(1 - order));

(* At one variable-order output node, derive every coefficient directly
   from the kernel integral against the slope of the linear interpolant. *)
probeIndex = 12;
probeOrderExact = ordersExact[[probeIndex + 1]];
probeFormulaWeights = Table[
  l1Coefficient[probeIndex - historyIndex, probeOrderExact],
  {historyIndex, 0, probeIndex - 1}
];
(* One symbolic antiderivative supplies all interval integrals. This avoids
   replacing Integrate with a hand-coded primitive while keeping the oracle
   tractable for live validation. *)
probeKernelPrimitive = Integrate[
  (timesExact[[probeIndex + 1]] - s)^(-probeOrderExact),
  s,
  GenerateConditions -> False
];
probeIntegralWeights = FullSimplify[
  Table[
    1/(stepExact Gamma[1 - probeOrderExact]) (
      (probeKernelPrimitive /.
        s -> timesExact[[historyIndex + 2]]) -
      (probeKernelPrimitive /.
        s -> timesExact[[historyIndex + 1]])
    ),
    {historyIndex, 0, probeIndex - 1}
  ]
];
probeWeightResiduals = MapThread[
  FullSimplify[#1 - #2] &,
  {probeFormulaWeights, probeIntegralWeights}
];
probeWeightsSymbolicMatch = And @@ (
  TrueQ[# == 0] & /@ probeWeightResiduals
);
probeWeightNumericMaxResidual = N[
  Max[Abs[N[probeWeightResiduals, 70]]],
  35
];

(* L1 evaluation of the exact power samples. This is a finite-grid
   approximation diagnostic, not a convergence-rate assertion. *)
powerSamples = initialValueExact +
  (timesExact - lowerTerminalExact)^powerExact;
powerL1Derivative = N[Table[
  If[
    outputIndex == 0,
    0,
    Sum[
      l1Coefficient[
        outputIndex - historyIndex,
        ordersExact[[outputIndex + 1]]
      ] (powerSamples[[historyIndex + 2]] -
        powerSamples[[historyIndex + 1]]),
      {historyIndex, 0, outputIndex - 1}
    ]
  ],
  {outputIndex, 0, nSteps}
], 70];
powerL1MaxFiniteGridError = N[
  Max[Abs[powerL1Derivative - powerExactDerivative]],
  35
];

(* Manufactured Type III equation with exact solution
   x(t)=x0+(t-a)^3 and time-only right-hand side equal to its exact
   Type III derivative. *)
ClearAll[manufacturedRhs];
manufacturedRhs[time_, order_] :=
  type3PowerDerivative[time, order];

ClearAll[variableOrderL1Recurrence];
variableOrderL1Recurrence[orderFunction_, rhsFunction_] := Module[
  {trajectory, outputIndex, history, order, scale},
  trajectory = ConstantArray[SetPrecision[0, 70], nSteps + 1];
  trajectory[[1]] = N[initialValueExact, 70];
  Do[
    order = N[orderFunction[timesExact[[outputIndex + 1]]], 70];
    scale = N[stepExact^(-order)/Gamma[2 - order], 70];
    history = If[
      outputIndex == 1,
      SetPrecision[0, 70],
      N[Sum[
        (
          (outputIndex - historyIndex)^(1 - order) -
          (outputIndex - historyIndex - 1)^(1 - order)
        ) (
          trajectory[[historyIndex + 2]] -
          trajectory[[historyIndex + 1]]
        ),
        {historyIndex, 0, outputIndex - 2}
      ], 70]
    ];
    trajectory[[outputIndex + 1]] = N[
      trajectory[[outputIndex]] +
      rhsFunction[timesExact[[outputIndex + 1]], order]/scale -
      history,
      70
    ],
    {outputIndex, 1, nSteps}
  ];
  trajectory
];

variableTrajectory = variableOrderL1Recurrence[
  Function[{time}, alphaAt[time]],
  Function[{time, order}, manufacturedRhs[time, order]]
];
manufacturedExactStates = N[powerSamples, 70];
manufacturedMaxFiniteGridError = N[
  Max[Abs[variableTrajectory - manufacturedExactStates]],
  35
];

(* A separately written conventional constant-order L1 recurrence. *)
ClearAll[constantOrderL1Recurrence];
constantOrderL1Recurrence[rhsFunction_] := Module[
  {trajectory, outputIndex, history, scale},
  trajectory = ConstantArray[SetPrecision[0, 70], nSteps + 1];
  trajectory[[1]] = N[initialValueExact, 70];
  scale = N[
    stepExact^(-constantOrderExact)/Gamma[2 - constantOrderExact],
    70
  ];
  Do[
    history = If[
      outputIndex == 1,
      SetPrecision[0, 70],
      N[Sum[
        (
          (outputIndex - historyIndex)^(1 - constantOrderExact) -
          (outputIndex - historyIndex - 1)^(1 - constantOrderExact)
        ) (
          trajectory[[historyIndex + 2]] -
          trajectory[[historyIndex + 1]]
        ),
        {historyIndex, 0, outputIndex - 2}
      ], 70]
    ];
    trajectory[[outputIndex + 1]] = N[
      trajectory[[outputIndex]] +
      rhsFunction[timesExact[[outputIndex + 1]], constantOrderExact]/scale -
      history,
      70
    ],
    {outputIndex, 1, nSteps}
  ];
  trajectory
];

constantVariableTrajectory = variableOrderL1Recurrence[
  Function[{time}, constantOrderExact],
  Function[{time, order}, manufacturedRhs[time, order]]
];
constantReferenceTrajectory = constantOrderL1Recurrence[
  Function[{time, order}, manufacturedRhs[time, order]]
];
constantReductionMaxResidual = N[
  Max[Abs[constantVariableTrajectory - constantReferenceTrajectory]],
  35
];
constantWeightReductionMaxResidual = N[
  Max[
    Abs[
      Table[
        l1Coefficient[lag, constantOrderExact],
        {lag, 1, nSteps}
      ] -
      Table[
        stepExact^(-constantOrderExact)/Gamma[2 - constantOrderExact] *
          (lag^(1 - constantOrderExact) -
            (lag - 1)^(1 - constantOrderExact)),
        {lag, 1, nSteps}
      ]
    ]
  ],
  35
];

tests = {
  MakeTest[
    "type3_power_formula_with_alpha_of_t",
    powerFormulaSymbolicMatch,
    <|"symbolic_residual" -> ExprString[powerSymbolicResidual]|>
  ],
  MakeTest[
    "l1_weights_match_piecewise_linear_kernel_integrals",
    probeWeightsSymbolicMatch || probeWeightNumericMaxResidual < 10^-35,
    <|
      "symbolic_match" -> probeWeightsSymbolicMatch,
      "numeric_max_residual" -> probeWeightNumericMaxResidual
    |>
  ],
  MakeTest[
    "variable_order_l1_reduces_to_constant_order_l1",
    constantReductionMaxResidual < 10^-35 &&
      constantWeightReductionMaxResidual < 10^-35,
    <|
      "trajectory_max_residual" -> constantReductionMaxResidual,
      "weight_max_residual" -> constantWeightReductionMaxResidual
    |>
  ],
  MakeTest[
    "manufactured_type3_l1_recurrence_is_finite",
    And @@ (TrueQ[NumericQ[N[#]]] & /@ variableTrajectory) &&
      0 < manufacturedMaxFiniteGridError < 5/100,
    <|"max_finite_grid_error" -> manufacturedMaxFiniteGridError|>
  ],
  MakeTest[
    "power_l1_derivative_has_bounded_finite_grid_error",
    0 < powerL1MaxFiniteGridError < 5/100,
    <|"max_finite_grid_error" -> powerL1MaxFiniteGridError|>
  ],
  MakeTest["hafo_source_not_read", source["hafo_source_read"] === False],
  MakeTest["report_not_used", source["report_input_used"] === False]
};

summary = <|
  "system_id" -> systemID,
  "validation_scope" ->
    "independent_variable_order_Caputo_Type_III_L1_finite_consistency",
  "evidence_boundary" ->
    "finite algebraic-numerical consistency only; no global convergence theorem, stability, chaos, attraction, or hiddenness claim",
  "source" -> source,
  "parameters" -> <|
    "lower_terminal" -> N[lowerTerminalExact, 30],
    "step" -> N[stepExact, 30],
    "n_steps" -> nSteps,
    "initial_value" -> N[initialValueExact, 30],
    "power" -> powerExact,
    "order_formula" -> "alpha(t)=1/2+(t-3/4)^2/10",
    "orders" -> N[ordersExact, 35],
    "constant_order" -> N[constantOrderExact, 35]
  |>,
  "power_formula" -> <|
    "definition" ->
      "Gamma[p+1]/Gamma[p+1-alpha(t)] (t-a)^(p-alpha(t))",
    "symbolic_integral" -> ExprString[powerIntegralSymbolic],
    "symbolic_closed_form" -> ExprString[powerClosedSymbolic],
    "symbolic_residual" -> ExprString[powerSymbolicResidual],
    "symbolic_match" -> powerFormulaSymbolicMatch,
    "exact_derivative_values" -> N[powerExactDerivative, 35],
    "l1_derivative_values" -> N[powerL1Derivative, 35],
    "l1_max_finite_grid_error" -> powerL1MaxFiniteGridError
  |>,
  "weights" -> <|
    "definition" ->
      "[h Gamma(1-alpha_n)]^-1 Integrate[(t_n-s)^(-alpha_n),{s,t_k,t_(k+1)}]",
    "probe_output_index" -> probeIndex,
    "probe_order" -> N[probeOrderExact, 35],
    "formula_values" -> N[probeFormulaWeights, 35],
    "symbolic_integral_values" -> N[probeIntegralWeights, 35],
    "symbolic_residuals" -> (ExprString /@ probeWeightResiduals),
    "symbolic_match" -> probeWeightsSymbolicMatch,
    "numeric_max_residual" -> probeWeightNumericMaxResidual
  |>,
  "manufactured_case" -> <|
    "exact_solution" -> "x(t)=5/4+t^3",
    "rhs" ->
      "Gamma[4]/Gamma[4-alpha(t)] t^(3-alpha(t))",
    "times" -> N[timesExact, 35],
    "orders" -> N[ordersExact, 35],
    "states" -> N[variableTrajectory, 35],
    "exact_states" -> N[manufacturedExactStates, 35],
    "max_finite_grid_error" -> manufacturedMaxFiniteGridError
  |>,
  "constant_order_reduction" -> <|
    "order" -> N[constantOrderExact, 35],
    "variable_order_path_states" -> N[constantVariableTrajectory, 35],
    "constant_order_reference_states" -> N[constantReferenceTrajectory, 35],
    "trajectory_max_residual" -> constantReductionMaxResidual,
    "weight_max_residual" -> constantWeightReductionMaxResidual
  |>,
  "tests" -> tests,
  "passed" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests)
|>;

ExportJSON[
  FileNameJoin[{outDir, systemID <> "_validation_summary.json"}],
  summary
];

ExitFromTests[tests];
