(* ::Package:: *)

(* ============================================================= *)
(* Independent exponentially tempered convolution quadrature     *)
(*                                                               *)
(* Mathematical source anchors (this script does not read HAFO): *)
(*                                                               *)
(* 1. C. Lubich, "Discretized Fractional Calculus", SIAM J.     *)
(*    Math. Anal. 17 (1986). DOI: 10.1137/0517050                *)
(* 2. F. Sabzikar, M. M. Meerschaert, and J. Chen, "Tempered     *)
(*    fractional calculus", J. Comput. Phys. 293 (2015).         *)
(*    DOI: 10.1016/j.jcp.2014.04.024                             *)
(* 3. B. Guo et al., tempered fractional substantial calculus    *)
(*    and convolution quadrature. DOI: 10.1137/18M1230153        *)
(* 4. M. Chen and W. Deng, high-order algorithms for the         *)
(*    tempered fractional derivative. DOI: 10.1051/m2an/2014037 *)
(*                                                               *)
(* The oracle builds BDF1/BDF2 fractional weights twice: from    *)
(* an explicit coefficient recurrence and from independent       *)
(* generalized-binomial factor expansions.  Tempering is then    *)
(* evaluated both as damped CQ weights and as exponential         *)
(* conjugation.  No HAFO source, output, or report is read.       *)
(*                                                               *)
(* Passing proves finite high-precision algebraic/numerical       *)
(* consistency for these fixtures only.  It is not a stability   *)
(* or convergence theorem and gives no evidence of chaos,         *)
(* attraction, or hiddenness.                                    *)
(* ============================================================= *)

ClearAll["Global`*"];

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];

systemID = "tempered_convolution_quadrature";
temporaryRoot = $TemporaryDirectory;
outDir = EnsureDirectory[
  GetCommandOption[
    "--out",
    FileNameJoin[{temporaryRoot, "hafo_" <> systemID}]
  ]
];

workingPrecision = 80;
source = <|
  "lubich_doi" -> "10.1137/0517050",
  "sabzikar_meerschaert_chen_doi" -> "10.1016/j.jcp.2014.04.024",
  "guo_tempered_cq_doi" -> "10.1137/18M1230153",
  "chen_deng_tempered_algorithms_doi" -> "10.1051/m2an/2014037",
  "hafo_source_read" -> False,
  "hafo_formula_imported" -> False,
  "report_input_used" -> False,
  "built_in_series_used" -> False
|>;

(* ----------------------------------------------------------------- *)
(* 1. Two independent constructions of delta(z)^q coefficients.     *)
(* ----------------------------------------------------------------- *)

ClearAll[
  BDFCoefficients, BDFDelta, BDFWeightRecurrence,
  BDFWeightExpansion, CQApply
];

BDFCoefficients[1] := {1, -1};
BDFCoefficients[2] := {3/2, -2, 1/2};
BDFDelta[bdfOrder_Integer, z_] :=
  Sum[
    BDFCoefficients[bdfOrder][[index + 1]] z^index,
    {index, 0, Length[BDFCoefficients[bdfOrder]] - 1}
  ];

(* If W(z)=delta(z)^q=sum_n w_n z^n, then delta W'=q delta' W.
   Comparing the coefficient of z^(n-1) gives the recurrence below. *)
BDFWeightRecurrence[
  order_, count_Integer?Positive, bdfOrder_Integer
] := Module[{delta, degree, weights, n, j, upper},
  delta = BDFCoefficients[bdfOrder];
  degree = Length[delta] - 1;
  weights = ConstantArray[0, count];
  weights[[1]] = N[delta[[1]]^order, workingPrecision];
  Do[
    upper = Min[degree, n];
    weights[[n + 1]] = N[
      Sum[
        (j (order + 1) - n) delta[[j + 1]] weights[[n - j + 1]],
        {j, 1, upper}
      ]/(n delta[[1]]),
      workingPrecision
    ],
    {n, 1, count - 1}
  ];
  weights
];

(* Independent factor expansions, without a built-in series expansion.  BDF2 uses
   delta(z)=(3/2)(1-z)(1-z/3). *)
BDFWeightExpansion[
  order_, count_Integer?Positive, 1
] := N[
  Table[(-1)^n Binomial[order, n], {n, 0, count - 1}],
  workingPrecision
];

BDFWeightExpansion[
  order_, count_Integer?Positive, 2
] := N[
  Table[
    (3/2)^order Sum[
      (-1)^j Binomial[order, j]
        (-1)^(n - j) Binomial[order, n - j]/3^(n - j),
      {j, 0, n}
    ],
    {n, 0, count - 1}
  ],
  workingPrecision
];

CQApply[samples_List, order_, step_, bdfOrder_Integer] := Module[
  {weights, count},
  count = Length[samples];
  weights = BDFWeightRecurrence[order, count, bdfOrder];
  Table[
    N[
      step^-order Sum[
        weights[[lag + 1]] samples[[n - lag + 1]],
        {lag, 0, n}
      ],
      workingPrecision
    ],
    {n, 0, count - 1}
  ]
];

(* ----------------------------------------------------------------- *)
(* 2. Independent direct and conjugated tempered CQ paths.           *)
(* ----------------------------------------------------------------- *)

ClearAll[
  TemperedCQDirect, TemperedCQConjugated, TemperedCQVector,
  DefinitionShiftQ
];

DefinitionShiftQ["tempered_riemann_liouville"] := False;
DefinitionShiftQ["tempered_caputo"] := True;

(* Direct damped-weight form.  The Caputo definition subtracts the
   initial value in the exponentially conjugated coordinate. *)
TemperedCQDirect[
  samples_List, order_, tempering_, step_, bdfOrder_Integer,
  definition_String
] := Module[{weights, count, shiftInitial},
  count = Length[samples];
  weights = BDFWeightRecurrence[order, count, bdfOrder];
  shiftInitial = DefinitionShiftQ[definition];
  Table[
    N[
      step^-order Sum[
        weights[[lag + 1]] (
          Exp[-tempering lag step] samples[[n - lag + 1]] -
          If[
            shiftInitial,
            Exp[-tempering n step] First[samples],
            0
          ]
        ),
        {lag, 0, n}
      ],
      workingPrecision
    ],
    {n, 0, count - 1}
  ]
];

(* Exponential-conjugation form, evaluated independently of the
   damped-weight loop above. *)
TemperedCQConjugated[
  samples_List, order_, tempering_, step_, bdfOrder_Integer,
  definition_String
] := Module[{count, elapsed, transformed, shifted, base},
  count = Length[samples];
  elapsed = step Range[0, count - 1];
  transformed = Exp[tempering elapsed] samples;
  shifted = If[
    DefinitionShiftQ[definition],
    transformed - First[transformed],
    transformed
  ];
  base = CQApply[shifted, order, step, bdfOrder];
  N[Exp[-tempering elapsed] base, workingPrecision]
];

TemperedCQVector[
  samples_?MatrixQ, orders_List, temperings_List, step_,
  bdfOrder_Integer, definition_String
] := Transpose[
  MapThread[
    TemperedCQDirect[
      #1, #2, #3, step, bdfOrder, definition
    ] &,
    {Transpose[samples], orders, temperings}
  ]
];

(* ----------------------------------------------------------------- *)
(* 3. Scalar RL and Caputo fixtures under exponential conjugation.   *)
(* ----------------------------------------------------------------- *)

sampleOrderExact = 3/5;
sampleTemperingExact = 2/5;
sampleStepExact = 1/16;
sampleIntervals = 16;
sampleElapsedExact = sampleStepExact Range[0, sampleIntervals];
sampleConstantExact = 7/5;
samplePower = 3;
sampleValuesExact = N[
  Exp[-sampleTemperingExact sampleElapsedExact] (
    sampleConstantExact + sampleElapsedExact^samplePower
  ),
  workingPrecision
];

scalarRows = Table[
  Module[{direct, conjugated, residual},
    direct = TemperedCQDirect[
      sampleValuesExact,
      sampleOrderExact,
      sampleTemperingExact,
      sampleStepExact,
      bdfOrder,
      definition
    ];
    conjugated = TemperedCQConjugated[
      sampleValuesExact,
      sampleOrderExact,
      sampleTemperingExact,
      sampleStepExact,
      bdfOrder,
      definition
    ];
    residual = N[direct - conjugated, 40];
    <|
      "definition" -> definition,
      "bdf_order" -> bdfOrder,
      "direct_values" -> direct,
      "conjugated_values" -> conjugated,
      "max_conjugation_residual" -> Max[Abs[residual]]
    |>
  ],
  {definition, {
    "tempered_riemann_liouville",
    "tempered_caputo"
  }},
  {bdfOrder, {1, 2}}
];
scalarRows = Flatten[scalarRows, 1];

(* ----------------------------------------------------------------- *)
(* 4. lambda=0 and q=1 reductions.                                  *)
(* ----------------------------------------------------------------- *)

zeroTemperingOrderExact = 7/10;
zeroTemperingStepExact = 1/12;
zeroTemperingIntervals = 12;
zeroTemperingElapsedExact =
  zeroTemperingStepExact Range[0, zeroTemperingIntervals];
zeroTemperingSamplesExact = N[
  9/8 + zeroTemperingElapsedExact^2,
  workingPrecision
];

zeroTemperingRows = Table[
  Module[{baseSamples, untempered, tempered, residual},
    baseSamples = If[
      DefinitionShiftQ[definition],
      zeroTemperingSamplesExact - First[zeroTemperingSamplesExact],
      zeroTemperingSamplesExact
    ];
    untempered = CQApply[
      baseSamples,
      zeroTemperingOrderExact,
      zeroTemperingStepExact,
      bdfOrder
    ];
    tempered = TemperedCQDirect[
      zeroTemperingSamplesExact,
      zeroTemperingOrderExact,
      0,
      zeroTemperingStepExact,
      bdfOrder,
      definition
    ];
    residual = N[tempered - untempered, 40];
    <|
      "definition" -> definition,
      "bdf_order" -> bdfOrder,
      "tempered_values" -> tempered,
      "untempered_values" -> untempered,
      "max_lambda_zero_residual" -> Max[Abs[residual]]
    |>
  ],
  {definition, {
    "tempered_riemann_liouville",
    "tempered_caputo"
  }},
  {bdfOrder, {1, 2}}
];
zeroTemperingRows = Flatten[zeroTemperingRows, 1];

qOneTemperingExact = 1/3;
qOneStepExact = 1/10;
qOneIntervals = 10;
qOneElapsedExact = qOneStepExact Range[0, qOneIntervals];
qOneSamplesExact = N[
  Exp[-qOneTemperingExact qOneElapsedExact] (5/4 + qOneElapsedExact^2),
  workingPrecision
];
qOneExpectedWeights = <|
  "1" -> N[PadRight[{1, -1}, qOneIntervals + 1], workingPrecision],
  "2" -> N[
    PadRight[{3/2, -2, 1/2}, qOneIntervals + 1],
    workingPrecision
  ]
|>;

qOneRows = Table[
  Module[{weights, direct, expected, elapsed, transformed, shifted},
    weights = BDFWeightRecurrence[1, qOneIntervals + 1, bdfOrder];
    direct = TemperedCQDirect[
      qOneSamplesExact,
      1,
      qOneTemperingExact,
      qOneStepExact,
      bdfOrder,
      definition
    ];
    elapsed = qOneElapsedExact;
    transformed = Exp[qOneTemperingExact elapsed] qOneSamplesExact;
    shifted = If[
      DefinitionShiftQ[definition],
      transformed - First[transformed],
      transformed
    ];
    expected = N[
      Exp[-qOneTemperingExact elapsed]
        Table[
          qOneStepExact^-1 Sum[
            qOneExpectedWeights[ToString[bdfOrder]][[lag + 1]]
              shifted[[n - lag + 1]],
            {lag, 0, n}
          ],
          {n, 0, qOneIntervals}
        ],
      workingPrecision
    ];
    <|
      "definition" -> definition,
      "bdf_order" -> bdfOrder,
      "weights" -> weights,
      "expected_weights" -> qOneExpectedWeights[ToString[bdfOrder]],
      "direct_values" -> direct,
      "expected_conjugated_bdf_values" -> expected,
      "max_weight_residual" -> Max[
        Abs[weights - qOneExpectedWeights[ToString[bdfOrder]]]
      ],
      "max_value_residual" -> Max[Abs[direct - expected]]
    |>
  ],
  {definition, {
    "tempered_riemann_liouville",
    "tempered_caputo"
  }},
  {bdfOrder, {1, 2}}
];
qOneRows = Flatten[qOneRows, 1];

(* ----------------------------------------------------------------- *)
(* 5. Componentwise vector fixture with distinct q and lambda.       *)
(* ----------------------------------------------------------------- *)

vectorOrdersExact = {2/5, 4/5};
vectorTemperingsExact = {1/5, 3/5};
vectorStepExact = 1/14;
vectorIntervals = 14;
vectorElapsedExact = vectorStepExact Range[0, vectorIntervals];
vectorSamplesExact = N[
  Transpose[{
    Exp[-vectorTemperingsExact[[1]] vectorElapsedExact]
      (3/2 + vectorElapsedExact^2),
    Exp[-vectorTemperingsExact[[2]] vectorElapsedExact]
      (-2/3 + vectorElapsedExact^3)
  }],
  workingPrecision
];

vectorRows = Table[
  Module[{vectorValues, scalarColumns, conjugatedColumns},
    vectorValues = TemperedCQVector[
      vectorSamplesExact,
      vectorOrdersExact,
      vectorTemperingsExact,
      vectorStepExact,
      bdfOrder,
      definition
    ];
    scalarColumns = Transpose[
      MapThread[
        TemperedCQDirect[
          #1, #2, #3, vectorStepExact, bdfOrder, definition
        ] &,
        {
          Transpose[vectorSamplesExact],
          vectorOrdersExact,
          vectorTemperingsExact
        }
      ]
    ];
    conjugatedColumns = Transpose[
      MapThread[
        TemperedCQConjugated[
          #1, #2, #3, vectorStepExact, bdfOrder, definition
        ] &,
        {
          Transpose[vectorSamplesExact],
          vectorOrdersExact,
          vectorTemperingsExact
        }
      ]
    ];
    <|
      "definition" -> definition,
      "bdf_order" -> bdfOrder,
      "values" -> vectorValues,
      "scalar_component_values" -> scalarColumns,
      "conjugated_component_values" -> conjugatedColumns,
      "max_componentwise_residual" -> Max[
        Abs[vectorValues - scalarColumns]
      ],
      "max_conjugation_residual" -> Max[
        Abs[vectorValues - conjugatedColumns]
      ]
    |>
  ],
  {definition, {
    "tempered_riemann_liouville",
    "tempered_caputo"
  }},
  {bdfOrder, {1, 2}}
];
vectorRows = Flatten[vectorRows, 1];

(* ----------------------------------------------------------------- *)
(* 6. Endpoint convergence diagnostics for manufactured functions.  *)
(* ----------------------------------------------------------------- *)

resolutionGrid = {64, 128, 256};
ClearAll[MakeConvergenceRow];
MakeConvergenceRow[bdfOrder_Integer, resolution_Integer] := Module[
  {
    step, elapsed, samples, rlValues, caputoValues,
    rlExact, caputoExact, endpoint
  },
  step = 1/resolution;
  elapsed = step Range[0, resolution];
  samples = N[
    Exp[-sampleTemperingExact elapsed]
      (sampleConstantExact + elapsed^samplePower),
    workingPrecision
  ];
  rlValues = TemperedCQDirect[
    samples,
    sampleOrderExact,
    sampleTemperingExact,
    step,
    bdfOrder,
    "tempered_riemann_liouville"
  ];
  caputoValues = TemperedCQDirect[
    samples,
    sampleOrderExact,
    sampleTemperingExact,
    step,
    bdfOrder,
    "tempered_caputo"
  ];
  endpoint = 1;
  rlExact = N[
    Exp[-sampleTemperingExact endpoint] (
      sampleConstantExact endpoint^-sampleOrderExact/
        Gamma[1 - sampleOrderExact] +
      Gamma[samplePower + 1]/
        Gamma[samplePower + 1 - sampleOrderExact]
    ),
    workingPrecision
  ];
  caputoExact = N[
    Exp[-sampleTemperingExact endpoint]
      Gamma[samplePower + 1]/
        Gamma[samplePower + 1 - sampleOrderExact],
    workingPrecision
  ];
  <|
    "bdf_order" -> bdfOrder,
    "n_steps" -> resolution,
    "step" -> N[step, 30],
    "rl_endpoint" -> Last[rlValues],
    "rl_endpoint_analytic" -> rlExact,
    "rl_endpoint_abs_error" -> Abs[Last[rlValues] - rlExact],
    "caputo_endpoint" -> Last[caputoValues],
    "caputo_endpoint_analytic" -> caputoExact,
    "caputo_endpoint_abs_error" -> Abs[Last[caputoValues] - caputoExact]
  |>
];

convergenceRows = Flatten[
  Table[
    MakeConvergenceRow[bdfOrder, resolution],
    {bdfOrder, {1, 2}},
    {resolution, resolutionGrid}
  ],
  1
];

ClearAll[ErrorsFor];
ErrorsFor[bdfOrder_Integer, key_String] := Lookup[
  Select[
    convergenceRows,
    Lookup[#, "bdf_order"] == bdfOrder &
  ],
  key
];

bdf1RLErrors = ErrorsFor[1, "rl_endpoint_abs_error"];
bdf2RLErrors = ErrorsFor[2, "rl_endpoint_abs_error"];
bdf1CaputoErrors = ErrorsFor[1, "caputo_endpoint_abs_error"];
bdf2CaputoErrors = ErrorsFor[2, "caputo_endpoint_abs_error"];

(* ----------------------------------------------------------------- *)
(* 7. Weight ledger, assertions, and portable JSON summary.          *)
(* ----------------------------------------------------------------- *)

weightOrderExact = 11/20;
weightCount = 24;
weightRows = Table[
  Module[{recurrence, expansion},
    recurrence = BDFWeightRecurrence[
      weightOrderExact, weightCount, bdfOrder
    ];
    expansion = BDFWeightExpansion[
      weightOrderExact, weightCount, bdfOrder
    ];
    <|
      "bdf_order" -> bdfOrder,
      "recurrence_weights" -> recurrence,
      "factor_expansion_weights" -> expansion,
      "max_recurrence_expansion_residual" -> Max[
        Abs[recurrence - expansion]
      ]
    |>
  ],
  {bdfOrder, {1, 2}}
];

tests = {
  MakeTest[
    "bdf1_recurrence_matches_own_binomial_expansion_at_high_precision",
    weightRows[[1, "max_recurrence_expansion_residual"]] < 10^-70,
    <|
      "max_residual" ->
        weightRows[[1, "max_recurrence_expansion_residual"]]
    |>
  ],
  MakeTest[
    "bdf2_recurrence_matches_own_factor_expansion_at_high_precision",
    weightRows[[2, "max_recurrence_expansion_residual"]] < 10^-70,
    <|
      "max_residual" ->
        weightRows[[2, "max_recurrence_expansion_residual"]]
    |>
  ],
  MakeTest[
    "bdf_generating_polynomials_are_exact",
    TrueQ[BDFDelta[1, z] == 1 - z] &&
      TrueQ[Expand[BDFDelta[2, z]] == 3/2 - 2 z + z^2/2]
  ],
  MakeTest[
    "q_one_bdf1_and_bdf2_weights_are_exact_finite_polynomials",
    Max[Lookup[qOneRows, "max_weight_residual"]] < 10^-70
  ],
  MakeTest[
    "tempered_rl_direct_weights_equal_exponential_conjugation",
    Max[
      Lookup[
        Select[
          scalarRows,
          Lookup[#, "definition"] ==
            "tempered_riemann_liouville" &
        ],
        "max_conjugation_residual"
      ]
    ] < 10^-65
  ],
  MakeTest[
    "tempered_caputo_direct_weights_equal_shifted_exponential_conjugation",
    Max[
      Lookup[
        Select[
          scalarRows,
          Lookup[#, "definition"] == "tempered_caputo" &
        ],
        "max_conjugation_residual"
      ]
    ] < 10^-65
  ],
  MakeTest[
    "lambda_zero_reduces_to_untempered_rl_and_caputo_cq",
    Max[Lookup[zeroTemperingRows, "max_lambda_zero_residual"]] < 10^-70
  ],
  MakeTest[
    "q_one_reduces_to_conjugated_bdf1_and_bdf2",
    Max[Lookup[qOneRows, "max_value_residual"]] < 10^-65
  ],
  MakeTest[
    "vector_rl_fixture_is_strictly_componentwise",
    Max[
      Lookup[
        Select[
          vectorRows,
          Lookup[#, "definition"] ==
            "tempered_riemann_liouville" &
        ],
        "max_componentwise_residual"
      ]
    ] == 0
  ],
  MakeTest[
    "vector_caputo_fixture_is_strictly_componentwise",
    Max[
      Lookup[
        Select[
          vectorRows,
          Lookup[#, "definition"] == "tempered_caputo" &
        ],
        "max_componentwise_residual"
      ]
    ] == 0
  ],
  MakeTest[
    "vector_fixture_matches_exponential_conjugation",
    Max[Lookup[vectorRows, "max_conjugation_residual"]] < 10^-65
  ],
  MakeTest[
    "bdf1_tempered_rl_and_caputo_endpoint_errors_decrease",
    And @@ Thread[Differences[bdf1RLErrors] < 0] &&
      And @@ Thread[Differences[bdf1CaputoErrors] < 0]
  ],
  MakeTest[
    "bdf2_tempered_rl_and_caputo_endpoint_errors_decrease",
    And @@ Thread[Differences[bdf2RLErrors] < 0] &&
      And @@ Thread[Differences[bdf2CaputoErrors] < 0]
  ],
  MakeTest[
    "finest_manufactured_endpoint_errors_are_bounded",
    Max[
      Last[bdf1RLErrors],
      Last[bdf2RLErrors],
      Last[bdf1CaputoErrors],
      Last[bdf2CaputoErrors]
    ] < 1/100
  ],
  MakeTest["hafo_source_not_read", source["hafo_source_read"] === False],
  MakeTest[
    "hafo_formula_not_imported",
    source["hafo_formula_imported"] === False
  ],
  MakeTest["report_not_used", source["report_input_used"] === False],
  MakeTest[
    "built_in_series_not_used",
    source["built_in_series_used"] === False
  ]
};

summary = <|
  "system_id" -> systemID,
  "validation_scope" ->
    "independent_high_precision_tempered_RL_and_Caputo_BDF1_BDF2_convolution_quadrature",
  "evidence_boundary" ->
    "finite high-precision algebraic and sampled-grid consistency only; no stability or convergence theorem, no arbitrary nonlinear FDE certification, and no evidence of chaos, attraction, or hiddenness",
  "source" -> source,
  "conventions" -> <|
    "working_precision" -> workingPrecision,
    "elapsed_coordinate" -> "tau=t-a",
    "rl_definition" ->
      "D_RL^(q,lambda) f=exp(-lambda*tau) D_RL^q[exp(lambda*tau) f]",
    "caputo_definition" ->
      "D_C^(q,lambda) f=exp(-lambda*tau) D_C^q[exp(lambda*tau) f]",
    "caputo_discrete_shift" ->
      "exp(lambda*tau_n)f_n-f_0 before untempered CQ",
    "tempered_weight_formula" ->
      "omega_k^(q,lambda)=exp(-lambda*k*h) omega_k^q",
    "tempered_generating_function" ->
      "delta(exp(-lambda*h)*z)^q/h^q",
    "no_minus_lambda_power_x" -> True,
    "shifted_symbol_family_not_evaluated" ->
      "[delta(z)/h+lambda]^q is not evaluated or identified with delta(exp(-lambda*h)*z)^q/h^q",
    "bdf1_delta" -> "1-z",
    "bdf2_delta" -> "3/2-2*z+z^2/2",
    "startup_convention" ->
      "terminal_truncated_history_no_prehistory_extrapolation",
    "starting_corrections" -> "none_implemented",
    "vector_semantics" ->
      "one scalar order and tempering parameter per component"
  |>,
  "weights" -> <|
    "order" -> N[weightOrderExact, 30],
    "count" -> weightCount,
    "rows" -> weightRows
  |>,
  "scalar_fixture" -> <|
    "order" -> N[sampleOrderExact, 30],
    "tempering" -> N[sampleTemperingExact, 30],
    "step" -> N[sampleStepExact, 30],
    "elapsed" -> N[sampleElapsedExact, 40],
    "samples" -> sampleValuesExact,
    "transformed_function" -> "constant+tau^3",
    "constant" -> N[sampleConstantExact, 30],
    "power" -> samplePower,
    "rows" -> scalarRows
  |>,
  "lambda_zero_fixture" -> <|
    "order" -> N[zeroTemperingOrderExact, 30],
    "tempering" -> 0,
    "step" -> N[zeroTemperingStepExact, 30],
    "elapsed" -> N[zeroTemperingElapsedExact, 40],
    "samples" -> zeroTemperingSamplesExact,
    "rows" -> zeroTemperingRows
  |>,
  "q_one_fixture" -> <|
    "order" -> 1,
    "tempering" -> N[qOneTemperingExact, 30],
    "step" -> N[qOneStepExact, 30],
    "elapsed" -> N[qOneElapsedExact, 40],
    "samples" -> qOneSamplesExact,
    "rows" -> qOneRows
  |>,
  "vector_fixture" -> <|
    "orders" -> N[vectorOrdersExact, 30],
    "temperings" -> N[vectorTemperingsExact, 30],
    "step" -> N[vectorStepExact, 30],
    "elapsed" -> N[vectorElapsedExact, 40],
    "samples" -> vectorSamplesExact,
    "rows" -> vectorRows
  |>,
  "convergence" -> <|
    "resolution_grid" -> resolutionGrid,
    "rows" -> convergenceRows,
    "diagnostic_only" -> True
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
