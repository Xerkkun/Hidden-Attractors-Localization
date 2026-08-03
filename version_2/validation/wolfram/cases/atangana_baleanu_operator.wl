(* ::Package:: *)

(* ============================================================= *)
(* Independent Atangana--Baleanu--Caputo operator validation     *)
(*                                                               *)
(* Mathematical source anchors (the script does not read HAFO):  *)
(*                                                               *)
(* 1. A. Atangana and D. Baleanu, "New fractional derivatives   *)
(*    with nonlocal and non-singular kernel: theory and          *)
(*    application to heat transfer model", Thermal Science 20   *)
(*    (2016), 763--769. DOI: 10.2298/TSCI160111018A              *)
(* 2. S. Yadav, R. K. Pandey, and A. K. Shukla, "Numerical      *)
(*    approximations of Atangana--Baleanu Caputo derivative and  *)
(*    its application", Chaos Solitons & Fractals 118 (2019),   *)
(*    58--64. DOI: 10.1016/j.chaos.2018.11.009                   *)
(* 3. K. Diethelm, R. Garrappa, A. Giusti, and M. Stynes,       *)
(*    "Why fractional derivatives with nonsingular kernels      *)
(*    should not be used", FCAA 23 (2020), 610--634.            *)
(*    DOI: 10.1515/fca-2020-0032                                *)
(*                                                               *)
(* The case fixes alpha=1/2 and derives interval weights directly *)
(* from Mathematica's MittagLefflerE. It checks the identity      *)
(* E_(1/2)(-sqrt(t))=Exp[t] Erfc[sqrt(t)], the zero derivative    *)
(* of a constant, and the closed form for a ramp. These are       *)
(* finite-grid operator checks: no stability, chaos, hiddenness,  *)
(* FDE-solver convergence, or initial-compatibility claim.        *)
(* ============================================================= *)

ClearAll["Global`*"];

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];

systemID = "atangana_baleanu_operator";
outDir = EnsureDirectory[
  GetCommandOption[
    "--out",
    FileNameJoin[{Directory[], "validation_outputs", systemID}]
  ]
];

source = <|
  "atangana_baleanu_doi" -> "10.2298/TSCI160111018A",
  "yadav_pandey_shukla_doi" -> "10.1016/j.chaos.2018.11.009",
  "diethelm_garrappa_giusti_stynes_doi" -> "10.1515/fca-2020-0032",
  "scope" -> "ABC kernel identity and finite uniform-grid sampled operator",
  "hafo_source_read" -> False,
  "report_input_used" -> False
|>;

alphaExact = 1/2;
kernelRateExact = alphaExact/(1 - alphaExact);
stepExact = 1/32;
nSteps = 24;
lowerTerminalExact = -2;
normalizationExact = 1 - alphaExact + alphaExact/Gamma[alphaExact];
scaleExact = normalizationExact/(1 - alphaExact);
elapsedExact = Table[index stepExact, {index, 0, nSteps}];
physicalTimes = N[lowerTerminalExact + elapsedExact, 40];

(* Direct definition of the ABC kernel. *)
ClearAll[abcKernel];
abcKernel[elapsed_?NumericQ] := N[
  MittagLefflerE[
    alphaExact,
    -kernelRateExact SetPrecision[elapsed, 80]^alphaExact
  ],
  70
];

(* The weights are obtained independently by quadrature of the kernel on
   each uniform interval. No Python coefficients or generated report data
   enter this definition. *)
intervalWeights = Table[
  N[
    NIntegrate[
      abcKernel[elapsed],
      {
        elapsed,
        N[lag stepExact, 80],
        N[(lag + 1) stepExact, 80]
      },
      WorkingPrecision -> 70,
      AccuracyGoal -> 45,
      PrecisionGoal -> 45,
      Method -> {"GlobalAdaptive", "SymbolicProcessing" -> 0}
    ]/N[stepExact, 80],
    45
  ],
  {lag, 0, nSteps - 1}
];

(* Integral identity used only as an independent cross-check of the direct
   quadrature: Integrate[E_alpha(-lambda s^alpha),s] equals
   s E_(alpha,2)(-lambda s^alpha). *)
ClearAll[mittagLefflerPrimitive];
mittagLefflerPrimitive[elapsed_] :=
  elapsed MittagLefflerE[
    alphaExact,
    2,
    -kernelRateExact elapsed^alphaExact
  ];

primitiveWeights = N[
  Table[
    (mittagLefflerPrimitive[(lag + 1) stepExact] -
      mittagLefflerPrimitive[lag stepExact])/stepExact,
    {lag, 0, nSteps - 1}
  ],
  45
];
weightPrimitiveMaxResidual = N[
  Max[Abs[intervalWeights - primitiveWeights]],
  30
];

(* Special identity for alpha=1/2. *)
identityElapsed = N[Range[0, 16]/16, 60];
mittagLefflerIdentityValues = N[
  MittagLefflerE[1/2, -Sqrt[#]] & /@ identityElapsed,
  50
];
erfcIdentityValues = N[
  Exp[#] Erfc[Sqrt[#]] & /@ identityElapsed,
  50
];
identityMaxResidual = N[
  Max[Abs[mittagLefflerIdentityValues - erfcIdentityValues]],
  30
];
identitySymbolicResidual = FullSimplify[
  FunctionExpand[MittagLefflerE[1/2, -Sqrt[t]]] -
    Exp[t] Erfc[Sqrt[t]],
  Assumptions -> t >= 0
];

ClearAll[halfOrderKernelPrimitive];
halfOrderKernelPrimitive[elapsed_] :=
  Exp[elapsed] Erfc[Sqrt[elapsed]] - 1 +
    2 Sqrt[elapsed]/Sqrt[Pi];

primitiveIdentityValues = N[
  halfOrderKernelPrimitive /@ elapsedExact,
  50
];
mittagLefflerPrimitiveValues = N[
  mittagLefflerPrimitive /@ elapsedExact,
  50
];
primitiveIdentityMaxResidual = N[
  Max[Abs[primitiveIdentityValues - mittagLefflerPrimitiveValues]],
  30
];

(* Uniform-grid piecewise-linear ABC convolution. *)
ClearAll[abcDiscrete];
abcDiscrete[samples_List, weights_List] := Module[
  {increments = Differences[samples]},
  Join[
    {0},
    Table[
      scaleExact Sum[
        increments[[historyIndex]] weights[[outputIndex - historyIndex + 1]],
        {historyIndex, 1, outputIndex}
      ],
      {outputIndex, 1, Length[increments]}
    ]
  ]
];

constantValueExact = 17/4;
constantSamples = ConstantArray[constantValueExact, nSteps + 1];
constantDerivative = N[abcDiscrete[constantSamples, intervalWeights], 40];
constantMaxAbs = N[Max[Abs[constantDerivative]], 30];

rampInterceptExact = -3;
rampSlopeExact = 11/4;
rampSamples = N[
  rampInterceptExact + rampSlopeExact elapsedExact,
  45
];
rampDerivative = N[abcDiscrete[rampSamples, intervalWeights], 40];
rampClosedForm = N[
  scaleExact rampSlopeExact (halfOrderKernelPrimitive /@ elapsedExact),
  40
];
rampClosedFormMaxResidual = N[
  Max[Abs[rampDerivative - rampClosedForm]],
  30
];

sampleValues = N[
  (3/5 + (2/7) # + #^2/5 + Sin[3 #]/11) & /@ elapsedExact,
  45
];
sampleDerivative = N[abcDiscrete[sampleValues, intervalWeights], 40];

tests = {
  MakeTest[
    "half_order_mittag_leffler_erfc_identity",
    TrueQ[identitySymbolicResidual == 0] || identityMaxResidual < 10^-35,
    <|
      "symbolic_residual" -> ExprString[identitySymbolicResidual],
      "numeric_max_residual" -> identityMaxResidual
    |>
  ],
  MakeTest[
    "half_order_kernel_primitive_closed_form",
    primitiveIdentityMaxResidual < 10^-35,
    <|"max_residual" -> primitiveIdentityMaxResidual|>
  ],
  MakeTest[
    "interval_weights_match_mittag_leffler_primitive",
    weightPrimitiveMaxResidual < 10^-35,
    <|"max_residual" -> weightPrimitiveMaxResidual|>
  ],
  MakeTest[
    "interval_weights_are_positive_and_monotone",
    And @@ Thread[intervalWeights > 0] &&
      And @@ Thread[Differences[intervalWeights] <= 0]
  ],
  MakeTest[
    "constant_abc_derivative_is_zero",
    constantMaxAbs == 0,
    <|"max_abs" -> constantMaxAbs|>
  ],
  MakeTest[
    "ramp_abc_derivative_matches_closed_form",
    rampClosedFormMaxResidual < 10^-35,
    <|"max_residual" -> rampClosedFormMaxResidual|>
  ],
  MakeTest["hafo_source_not_read", source["hafo_source_read"] === False],
  MakeTest["report_not_used", source["report_input_used"] === False]
};

summary = <|
  "system_id" -> systemID,
  "validation_scope" ->
    "independent_ABC_half_order_kernel_weights_and_sampled_operator",
  "evidence_boundary" ->
    "finite-grid operator consistency; no stability, chaos, hiddenness, FDE-solver convergence, or initial-compatibility claim",
  "source" -> source,
  "parameters" -> <|
    "alpha" -> N[alphaExact, 30],
    "kernel_rate" -> N[kernelRateExact, 30],
    "step" -> N[stepExact, 30],
    "n_steps" -> nSteps,
    "lower_terminal" -> N[lowerTerminalExact, 30],
    "normalization" -> N[normalizationExact, 30],
    "normalization_convention" ->
      "B(alpha)=1-alpha+alpha/Gamma(alpha)"
  |>,
  "kernel_identity" -> <|
    "formula" ->
      "MittagLefflerE[1/2,-Sqrt[t]]=Exp[t] Erfc[Sqrt[t]]",
    "elapsed_times" -> identityElapsed,
    "mittag_leffler_values" -> mittagLefflerIdentityValues,
    "erfc_values" -> erfcIdentityValues,
    "symbolic_residual" -> ExprString[identitySymbolicResidual],
    "max_residual" -> identityMaxResidual
  |>,
  "weights" -> <|
    "definition" ->
      "w_k=h^-1 Integrate[E_alpha(-alpha/(1-alpha) s^alpha),{s,k h,(k+1) h}]",
    "backend" -> "Wolfram NIntegrate with MittagLefflerE",
    "values" -> N[intervalWeights, 30],
    "primitive_values" -> N[primitiveWeights, 30],
    "primitive_max_residual" -> weightPrimitiveMaxResidual
  |>,
  "constant_case" -> <|
    "samples" -> N[constantSamples, 30],
    "derivative_values" -> constantDerivative,
    "max_abs" -> constantMaxAbs
  |>,
  "ramp_case" -> <|
    "intercept" -> N[rampInterceptExact, 30],
    "slope" -> N[rampSlopeExact, 30],
    "elapsed_times" -> N[elapsedExact, 30],
    "physical_times" -> physicalTimes,
    "samples" -> rampSamples,
    "derivative_values" -> rampDerivative,
    "closed_form_values" -> rampClosedForm,
    "closed_form_max_residual" -> rampClosedFormMaxResidual
  |>,
  "sample_case" -> <|
    "formula" -> "3/5+(2/7)t+t^2/5+Sin[3t]/11",
    "elapsed_times" -> N[elapsedExact, 30],
    "physical_times" -> physicalTimes,
    "samples" -> sampleValues,
    "derivative_values" -> sampleDerivative
  |>,
  "tests" -> tests,
  "passed" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests)
|>;

ExportJSON[
  FileNameJoin[{outDir, systemID <> "_validation_summary.json"}],
  summary
];

ExitFromTests[tests];
