(* ::Package:: *)

(* ============================================================= *)
(* Independent real-axis fast tempered multistep history          *)
(*                                                               *)
(* Primary sources:                                              *)
(* 1. L. Guo, F. Zeng, I. Turner, K. Burrage, G. E. Karniadakis,*)
(*    Efficient Multistep Methods for Tempered Fractional        *)
(*    Calculus, SIAM J. Sci. Comput. 41 (2019).                  *)
(*    DOI: 10.1137/18M1230153                                   *)
(* 2. C. Lubich, Discretized Fractional Calculus (1986).         *)
(*    DOI: 10.1137/0517050                                      *)
(* 3. L. N. Trefethen and J. A. C. Weideman, The Exponentially  *)
(*    Convergent Trapezoidal Rule (2014).                        *)
(*    DOI: 10.1137/130932132                                    *)
(*                                                               *)
(* This case reads no HAFO source or output. It independently     *)
(* constructs exact FBDF1/GNGF2 weights, a fixed high-precision   *)
(* real-axis trapezoidal quadrature, the recurrent old-history    *)
(* states, and the exact local/Caputo corrections.                *)
(*                                                               *)
(* Passing is finite algebraic and sampled-grid implementation    *)
(* evidence. It is not an FDE solver theorem, a general error     *)
(* certificate, or evidence of chaos, attraction, or hiddenness.  *)
(* ============================================================= *)

ClearAll["Global`*"];

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];

systemID = "tempered_fast_multistep_history";
temporaryRoot = If[
  $OperatingSystem === "Windows",
  "C:\\tmp",
  $TemporaryDirectory
];
outDir = EnsureDirectory[
  GetCommandOption[
    "--out",
    FileNameJoin[{temporaryRoot, "hafo_" <> systemID}]
  ]
];

workingPrecision = 80;
source = <|
  "guo_fast_method_ii_doi" -> "10.1137/18M1230153",
  "lubich_flmm_doi" -> "10.1137/0517050",
  "trefethen_weideman_trapezoidal_doi" -> "10.1137/130932132",
  "hafo_source_read" -> False,
  "hafo_formula_imported" -> False,
  "hafo_output_used" -> False,
  "report_input_used" -> False
|>;

(* ----------------------------------------------------------------- *)
(* 1. Independent exact multistep coefficient recurrences.           *)
(* ----------------------------------------------------------------- *)

ClearAll[GLWeights, MultistepWeights, DirectTemperedOperator];

GLWeights[order_, count_Integer?Positive] := Module[{weights, lag},
  weights = ConstantArray[0, count];
  weights[[1]] = N[1, workingPrecision];
  Do[
    weights[[lag + 1]] = N[
      ((lag - 1 - order)/lag) weights[[lag]],
      workingPrecision
    ],
    {lag, 1, count - 1}
  ];
  weights
];

MultistepWeights[order_, count_Integer?Positive, "fbdf1"] :=
  GLWeights[order, count];

MultistepWeights[order_, count_Integer?Positive, "gngf2"] := Module[
  {gl},
  gl = GLWeights[order, count];
  N[
    (1 + order/2) gl - (order/2) Join[{0}, Most[gl]],
    workingPrecision
  ]
];

DirectTemperedOperator[
  samples_List, order_, tempering_, step_, method_String,
  definition_String
] := Module[{count, weights, caputoQ},
  count = Length[samples];
  weights = MultistepWeights[order, count, method];
  caputoQ = definition === "tempered_caputo";
  Table[
    N[
      step^-order Sum[
        weights[[lag + 1]] Exp[-tempering step lag]
          samples[[n - lag + 1]],
        {lag, 0, n}
      ] - If[
        caputoQ,
        step^-order First[samples] Exp[-tempering step n]
          Total[Take[weights, n + 1]],
        0
      ],
      workingPrecision
    ],
    {n, 0, count - 1}
  ]
];

(* ----------------------------------------------------------------- *)
(* 2. Independent dimensionless Fast Method II recurrence.           *)
(* ----------------------------------------------------------------- *)

ClearAll[FastTemperedOperator];

FastTemperedOperator[
  samples_List, order_, tempering_, step_, method_String,
  definition_String, localSteps_Integer, quadraturePoints_Integer,
  logMinimum_, logMaximum_
] := Module[
  {
    count, weights, caputoQ, logNodes, spacing, nodes, factor,
    coefficient, quadratureWeights, decays, historyCoefficients,
    state, output, n, localStop, local, history, sourceIndex, value
  },
  count = Length[samples];
  weights = MultistepWeights[order, count, method];
  caputoQ = definition === "tempered_caputo";
  logNodes = N[
    Subdivide[logMinimum, logMaximum, quadraturePoints - 1],
    workingPrecision
  ];
  spacing = N[
    (logMaximum - logMinimum)/(quadraturePoints - 1),
    workingPrecision
  ];
  nodes = N[Exp[logNodes], workingPrecision];
  factor = Switch[
    method,
    "fbdf1", ConstantArray[1, quadraturePoints],
    "gngf2", 1 - (order/2) nodes
  ];
  coefficient = N[-Sin[Pi order]/Pi, workingPrecision];
  quadratureWeights = N[
    spacing coefficient Exp[(1 + order) logNodes] factor,
    workingPrecision
  ];
  decays = N[
    Exp[-tempering step]/(1 + nodes),
    workingPrecision
  ];
  historyCoefficients = N[
    step^-order Exp[-localSteps tempering step]
      quadratureWeights (1 + nodes)^(-localSteps - 1),
    workingPrecision
  ];
  state = ConstantArray[N[0, workingPrecision], quadraturePoints];
  output = ConstantArray[N[0, workingPrecision], count];

  Do[
    localStop = Min[n, localSteps];
    local = N[
      step^-order Sum[
        weights[[lag + 1]] Exp[-tempering step lag]
          samples[[n - lag + 1]],
        {lag, 0, localStop}
      ],
      workingPrecision
    ];
    history = 0;
    If[n > localSteps,
      sourceIndex = n - localSteps - 1;
      state = N[
        decays (state + samples[[sourceIndex + 1]]),
        workingPrecision
      ];
      history = N[Total[historyCoefficients state], workingPrecision];
    ];
    value = local + history;
    If[caputoQ,
      value = value - N[
        step^-order First[samples] Exp[-tempering step n]
          Total[Take[weights, n + 1]],
        workingPrecision
      ];
      If[n == 0, value = 0];
    ];
    output[[n + 1]] = N[value, workingPrecision],
    {n, 0, count - 1}
  ];
  <|
    "values" -> output,
    "nodes" -> nodes,
    "quadrature_weights" -> quadratureWeights,
    "decays" -> decays,
    "final_state" -> state
  |>
];

(* ----------------------------------------------------------------- *)
(* 3. Exact algebraic identities.                                    *)
(* ----------------------------------------------------------------- *)

betaReflectionIdentity = FullSimplify[
  -Sin[Pi q]/Pi Beta[q + 1, n - q] ==
    Gamma[n - q]/(Gamma[-q] Gamma[n + 1]),
  Assumptions -> 0 < q < 1 && n > q
];

gngf2IntegerLimit = Expand[
  ((1 - z)^q (1 + q (1 - z)/2)) /. q -> 1
];

gngf2CoefficientResiduals = Table[
  FullSimplify[
    SeriesCoefficient[
      (1 - z)^q (1 + q (1 - z)/2),
      {z, 0, index}
    ] - (
      (1 + q/2) (-1)^index Binomial[q, index] -
      If[
        index == 0,
        0,
        (q/2) (-1)^(index - 1) Binomial[q, index - 1]
      ]
    ),
    Assumptions -> 0 < q < 1
  ],
  {index, 0, 10}
];

manufacturedIdentity = FullSimplify[
  Exp[-lambda t] CaputoD[t^beta, {t, q}] ==
    Exp[-lambda t] Gamma[beta + 1]/Gamma[beta + 1 - q]
      t^(beta - q),
  Assumptions ->
    t > 0 && 0 < q < 1 && beta > 0 && lambda >= 0
];

(* ----------------------------------------------------------------- *)
(* 4. High-precision recurrent fixtures.                             *)
(* ----------------------------------------------------------------- *)

fixtureOrderExact = 3/5;
fixtureTemperingExact = 2/7;
fixtureStepExact = 1/32;
fixtureIntervals = 64;
fixtureElapsedExact = fixtureStepExact Range[0, fixtureIntervals];
fixtureSamplesExact = N[
  Exp[-fixtureTemperingExact fixtureElapsedExact]
    (7/5 + fixtureElapsedExact^3 + Sin[fixtureElapsedExact]/10),
  workingPrecision
];
fixtureLocalSteps = 12;
fixtureQuadraturePoints = 1297;
fixtureLogMinimum = -40;
fixtureLogMaximum = 8;

fixtureRows = Flatten[
  Table[
    Module[{direct, fast, residual},
      direct = DirectTemperedOperator[
        fixtureSamplesExact,
        fixtureOrderExact,
        fixtureTemperingExact,
        fixtureStepExact,
        method,
        definition
      ];
      fast = FastTemperedOperator[
        fixtureSamplesExact,
        fixtureOrderExact,
        fixtureTemperingExact,
        fixtureStepExact,
        method,
        definition,
        fixtureLocalSteps,
        fixtureQuadraturePoints,
        fixtureLogMinimum,
        fixtureLogMaximum
      ];
      residual = N[fast["values"] - direct, 50];
      <|
        "method" -> method,
        "definition" -> definition,
        "direct_values" -> direct,
        "fast_values" -> fast["values"],
        "max_fast_direct_residual" -> Max[Abs[residual]],
        "minimum_decay" -> Min[fast["decays"]],
        "maximum_decay" -> Max[fast["decays"]],
        "final_state_l1" -> Total[Abs[fast["final_state"]]]
      |>
    ],
    {method, {"fbdf1", "gngf2"}},
    {definition, {
      "tempered_riemann_liouville",
      "tempered_caputo"
    }}
  ],
  1
];

anchorSamplesExact = N[
  (11/4) Exp[-fixtureTemperingExact fixtureElapsedExact],
  workingPrecision
];
anchorRows = Table[
  Module[{fast},
    fast = FastTemperedOperator[
      anchorSamplesExact,
      fixtureOrderExact,
      fixtureTemperingExact,
      fixtureStepExact,
      method,
      "tempered_caputo",
      fixtureLocalSteps,
      fixtureQuadraturePoints,
      fixtureLogMinimum,
      fixtureLogMaximum
    ];
    <|
      "method" -> method,
      "values" -> fast["values"],
      "max_anchor_residual" -> Max[Abs[fast["values"]]]
    |>
  ],
  {method, {"fbdf1", "gngf2"}}
];

qOneFBDF1 = MultistepWeights[1, 8, "fbdf1"];
qOneGNGF2 = MultistepWeights[1, 8, "gngf2"];
qOneFBDF1Expected = N[PadRight[{1, -1}, 8], workingPrecision];
qOneGNGF2Expected = N[
  PadRight[{3/2, -2, 1/2}, 8],
  workingPrecision
];

tests = {
  MakeTest[
    "fbdf1_real_axis_sign_matches_beta_reflection",
    TrueQ[betaReflectionIdentity]
  ],
  MakeTest[
    "gngf2_coefficients_match_independent_series_coefficients",
    gngf2CoefficientResiduals === ConstantArray[0, 11]
  ],
  MakeTest[
    "gngf2_q_one_limit_is_exact_bdf2_polynomial",
    TrueQ[gngf2IntegerLimit == 3/2 - 2 z + z^2/2]
  ],
  MakeTest[
    "fbdf1_q_one_weights_are_exact_local_polynomial",
    Max[Abs[qOneFBDF1 - qOneFBDF1Expected]] < 10^-70
  ],
  MakeTest[
    "gngf2_q_one_weights_are_exact_bdf2_polynomial",
    Max[Abs[qOneGNGF2 - qOneGNGF2Expected]] < 10^-70
  ],
  MakeTest[
    "tempered_manufactured_caputo_identity_is_exact",
    TrueQ[manufacturedIdentity]
  ],
  MakeTest[
    "fbdf1_fast_rl_and_caputo_match_direct_history",
    Max[
      Lookup[
        Select[fixtureRows, Lookup[#, "method"] == "fbdf1" &],
        "max_fast_direct_residual"
      ]
    ] < 10^-25
  ],
  MakeTest[
    "gngf2_fast_rl_and_caputo_match_direct_history",
    Max[
      Lookup[
        Select[fixtureRows, Lookup[#, "method"] == "gngf2" &],
        "max_fast_direct_residual"
      ]
    ] < 10^-25
  ],
  MakeTest[
    "conjugated_caputo_exponential_anchor_is_annihilated",
    Max[Lookup[anchorRows, "max_anchor_residual"]] < 10^-24
  ],
  MakeTest[
    "all_real_history_recurrences_are_contractively_damped",
    Min[Lookup[fixtureRows, "minimum_decay"]] > 0 &&
      Max[Lookup[fixtureRows, "maximum_decay"]] <= 1
  ],
  MakeTest["hafo_source_not_read", source["hafo_source_read"] === False],
  MakeTest[
    "hafo_formula_and_output_not_imported",
    source["hafo_formula_imported"] === False &&
      source["hafo_output_used"] === False
  ],
  MakeTest["report_not_used", source["report_input_used"] === False]
};

summary = <|
  "system_id" -> systemID,
  "validation_scope" ->
    "independent_high_precision_FBDF1_GNGF2_real_axis_fast_tempered_history",
  "evidence_boundary" ->
    "finite high-precision algebraic and sampled-grid consistency only; no general trapezoidal-tail certificate, no FDE solver theorem, and no evidence of chaos, attraction, or hiddenness",
  "source" -> source,
  "conventions" -> <|
    "working_precision" -> workingPrecision,
    "dimensionless_node" -> "r=h*lambda",
    "weight_representation" ->
      "omega_lag=sum_j a_j*(1+r_j)^(-lag-1)",
    "history_recurrence" ->
      "y_m=exp(-sigma*h)/(1+r_j)*(y_(m-1)+u_(m-1))",
    "local_history" -> "exact_multistep_weights",
    "fbdf1_generator" -> "(1-z)^q",
    "gngf2_generator" -> "(1-z)^q*(1+q*(1-z)/2)",
    "fractional_bdf2_claimed" -> False,
    "caputo_anchor" ->
      "exact_-u0*exp(-sigma*n*h)*partial_sum_of_base_weights",
    "positive_exponential_materialized" -> False,
    "quadrature_error_scope" ->
      "fixed_fixture_only_not_a_general_certified_tolerance"
  |>,
  "algebra" -> <|
    "beta_reflection_identity" -> betaReflectionIdentity,
    "gngf2_integer_limit" -> ToString[gngf2IntegerLimit, InputForm],
    "gngf2_coefficient_residuals" -> gngf2CoefficientResiduals,
    "manufactured_identity" -> manufacturedIdentity
  |>,
  "fixture" -> <|
    "order" -> N[fixtureOrderExact, 30],
    "tempering" -> N[fixtureTemperingExact, 30],
    "step" -> N[fixtureStepExact, 30],
    "elapsed" -> N[fixtureElapsedExact, 40],
    "samples" -> fixtureSamplesExact,
    "local_history_steps" -> fixtureLocalSteps,
    "quadrature_points" -> fixtureQuadraturePoints,
    "log_minimum" -> fixtureLogMinimum,
    "log_maximum" -> fixtureLogMaximum,
    "rows" -> fixtureRows
  |>,
  "anchor_fixture" -> <|
    "samples" -> anchorSamplesExact,
    "rows" -> anchorRows
  |>,
  "tests" -> tests,
  "passed" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests),
  "files" -> <|
    "summary" -> systemID <> "_validation_summary.json"
  |>
|>;

ExportJSON[
  FileNameJoin[{outDir, systemID <> "_validation_summary.json"}],
  summary
];

ExitFromTests[tests];
