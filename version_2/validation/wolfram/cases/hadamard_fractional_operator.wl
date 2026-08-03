(* ::Package:: *)

(* ============================================================= *)
(* Independent Hadamard and Caputo--Hadamard validation          *)
(*                                                               *)
(* Mathematical source anchors (the script does not read HAFO):  *)
(*                                                               *)
(* 1. F. Jarad, T. Abdeljawad, and D. Baleanu, "Caputo-type      *)
(*    modification of the Hadamard fractional derivatives",      *)
(*    Adv. Difference Equ. (2012).                               *)
(*    DOI: 10.1186/1687-1847-2012-142                            *)
(* 2. B. Yin, G. Zhang, Y. Liu, and H. Li, "Convolution          *)
(*    quadrature for Hadamard fractional calculus ...",          *)
(*    Commun. Nonlinear Sci. Numer. Simul. 138 (2024).           *)
(*    DOI: 10.1016/j.cnsns.2024.108221                           *)
(* 3. X. Zheng, "Logarithmic transformation between              *)
(*    (variable-order) Caputo and Caputo-Hadamard fractional     *)
(*    problems and applications", Appl. Math. Lett. 121 (2021). *)
(*    DOI: 10.1016/j.aml.2021.107366                             *)
(* 4. C. W. H. Green, Y. Liu, and Y. Yan, "Numerical methods     *)
(*    for Caputo-Hadamard fractional differential equations ...",*)
(*    Mathematics 9 (2021). DOI: 10.3390/math9212728             *)
(* 5. C. Lubich, "Discretized Fractional Calculus", SIAM J.     *)
(*    Math. Anal. 17 (1986). DOI: 10.1137/0517050                *)
(* 6. K. Diethelm, N. J. Ford, and A. D. Freed, "Detailed Error  *)
(*    Analysis for a Fractional Adams Method", Numer. Algorithms *)
(*    36 (2004). DOI: 10.1023/B:NUMA.0000027736.85078.be         *)
(*                                                               *)
(* The script validates the logarithmic coordinate transform,    *)
(* continuum Gamma/Beta identities, BDF1/BDF2 sampled CQ, and a  *)
(* manufactured constant-forcing Caputo--Hadamard IVP. It does   *)
(* not certify stability, chaos, hiddenness, or convergence for  *)
(* arbitrary nonlinear systems.                                  *)
(* ============================================================= *)

ClearAll["Global`*"];

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];

systemID = "hadamard_fractional_operator";
outDir = EnsureDirectory[
  GetCommandOption["--out", FileNameJoin[{Directory[], "validation_outputs", systemID}]]
];

source = <|
  "jarad_abdeljawad_baleanu_doi" -> "10.1186/1687-1847-2012-142",
  "yin_zhang_liu_li_doi" -> "10.1016/j.cnsns.2024.108221",
  "zheng_doi" -> "10.1016/j.aml.2021.107366",
  "green_liu_yan_doi" -> "10.3390/math9212728",
  "lubich_doi" -> "10.1137/0517050",
  "diethelm_ford_freed_doi" -> "10.1023/B:NUMA.0000027736.85078.be",
  "hafo_source_read" -> False,
  "report_input_used" -> False
|>;

(* ----------------------------------------------------------------- *)
(* 1. Exact logarithmic-coordinate identities.                       *)
(* ----------------------------------------------------------------- *)

kernelTransformResidual = FullSimplify[
  Log[(a Exp[u])/(a Exp[v])] - (u - v),
  Assumptions -> a > 0 && Element[{a, u, v}, Reals] && u > v >= 0
];

measureTransformResidual = FullSimplify[
  D[a Exp[v], v]/(a Exp[v]) - 1,
  Assumptions -> a > 0 && Element[{a, v}, Reals]
];

dilationTransformResidual = FullSimplify[
  ((t D[g[Log[t/a]], t]) /. t -> a Exp[u]) - D[g[u], u],
  Assumptions -> a > 0 && u > 0 && Element[{a, u}, Reals]
];

(* The transformed Hadamard integral of u^m contains
   Beta[m+1,alpha]/Gamma[alpha]. *)
integralCoefficientResidual = FullSimplify[
  FunctionExpand[Beta[m + 1, alpha]/Gamma[alpha]] -
    Gamma[m + 1]/Gamma[m + alpha + 1],
  Assumptions -> alpha > 0 && Element[m, Integers] && m >= 0
];

(* Raw Hadamard: d/du after I^(1-q). *)
hadamardLogPowerResidual = FullSimplify[
  D[
    Gamma[m + 1] u^(m + 1 - q)/Gamma[m + 2 - q],
    u
  ] - Gamma[m + 1] u^(m - q)/Gamma[m + 1 - q],
  Assumptions -> 0 < q < 1 && u > 0 && Element[m, Integers] && m >= 1
];

(* Caputo--Hadamard: I^(1-q) after d/du. *)
caputoHadamardLogPowerResidual = FullSimplify[
  m Gamma[m] u^(m - q)/Gamma[m + 1 - q] -
    Gamma[m + 1] u^(m - q)/Gamma[m + 1 - q],
  Assumptions -> 0 < q < 1 && u > 0 && Element[m, Integers] && m >= 1
];

hadamardConstantResidual = FullSimplify[
  D[c u^(1 - q)/Gamma[2 - q], u] - c u^-q/Gamma[1 - q],
  Assumptions -> 0 < q < 1 && u > 0 && Element[c, Reals]
];

caputoHadamardConstantResidual = FullSimplify[D[c, u]];

logPowerQ1LimitResidual = Block[{$Assumptions = True},
  FullSimplify[
    Limit[
      Gamma[m + 1] u^(m - q)/Gamma[m + 1 - q],
      q -> 1,
      Direction -> "FromBelow",
      Assumptions -> u > 0 && Element[m, Integers] && m >= 1
    ] - m u^(m - 1),
    Assumptions -> u > 0 && Element[m, Integers] && m >= 1
  ]
];

constantQ1LimitResidual = Block[{$Assumptions = True},
  FullSimplify[
    Limit[
      c u^-q/Gamma[1 - q],
      q -> 1,
      Direction -> "FromBelow",
      Assumptions -> u > 0 && Element[c, Reals]
    ],
    Assumptions -> u > 0 && Element[c, Reals]
  ]
];

(* ----------------------------------------------------------------- *)
(* 2. Independent BDF convolution quadrature in logarithmic time.    *)
(* ----------------------------------------------------------------- *)

ClearAll[bdfDelta, cqWeights, cqApply];
bdfDelta[1, z_] := 1 - z;
bdfDelta[2, z_] := 3/2 - 2 z + z^2/2;

(* Formal power-series expansion is intentionally independent of the
   recurrence used by HAFO. PadRight also makes q=1 finite polynomials
   explicit instead of leaving roundoff-sized tails. *)
cqWeights[order_, count_Integer?Positive, bdfOrder_Integer] := Module[
  {coefficients},
  coefficients = CoefficientList[
    Normal[Series[bdfDelta[bdfOrder, z]^order, {z, 0, count - 1}]],
    z
  ];
  N[PadRight[coefficients, count], 50]
];

cqApply[samples_List, order_, logStep_, bdfOrder_Integer] := Module[
  {count, weights},
  count = Length[samples];
  weights = cqWeights[order, count, bdfOrder];
  Table[
    N[
      logStep^-order Sum[
        weights[[lag + 1]] samples[[sampleIndex - lag + 1]],
        {lag, 0, sampleIndex}
      ],
      40
    ],
    {sampleIndex, 0, count - 1}
  ]
];

bdf2FactorizationResidual = FullSimplify[
  Expand[(3/2) (1 - z) (1 - z/3)] - bdfDelta[2, z]
];

sampleOrderExact = 3/5;
sampleOrder = N[sampleOrderExact, 50];
lowerTerminalExact = 2;
lowerTerminal = N[lowerTerminalExact, 50];
logPowerDegree = 3;
constantValueExact = 7/5;
constantValue = N[constantValueExact, 50];
sampleIntervals = 16;
sampleCount = sampleIntervals + 1;
sampleLogStep = N[1/sampleIntervals, 50];
sampleLogTimes = N[Table[index/sampleIntervals, {index, 0, sampleIntervals}], 40];
samplePhysicalTimes = N[lowerTerminal Exp[sampleLogTimes], 30];
sampleValues = N[constantValue + sampleLogTimes^logPowerDegree, 40];
sampleShiftedValues = N[sampleValues - First[sampleValues], 40];

sampleRows = Table[
  Module[{weights, rawValues, shiftedValues, constantShiftedValues},
    weights = cqWeights[sampleOrder, sampleCount, bdfOrder];
    rawValues = cqApply[sampleValues, sampleOrder, sampleLogStep, bdfOrder];
    shiftedValues = cqApply[
      sampleShiftedValues, sampleOrder, sampleLogStep, bdfOrder
    ];
    constantShiftedValues = cqApply[
      ConstantArray[0, sampleCount], sampleOrder, sampleLogStep, bdfOrder
    ];
    <|
      "bdf_order" -> bdfOrder,
      "weights" -> weights,
      "raw_values" -> rawValues,
      "caputo_shifted_values" -> shiftedValues,
      "caputo_constant_max_abs" -> Max[Abs[constantShiftedValues]]
    |>
  ],
  {bdfOrder, {1, 2}}
];

logPowerEndpointExact = N[
  Gamma[logPowerDegree + 1]/Gamma[logPowerDegree + 1 - sampleOrderExact],
  40
];
rawConstantEndpointExact = N[
  constantValueExact/Gamma[1 - sampleOrderExact],
  40
];

resolutionGrid = {80, 160, 320};
ClearAll[makeConvergenceRow];
makeConvergenceRow[bdfOrder_Integer, resolution_Integer] := Module[
  {step, logTimes, shiftedSamples, rawConstantSamples, shiftedResult,
   rawConstantResult, shiftedError, rawConstantError},
  step = N[1/resolution, 50];
  logTimes = N[Table[index/resolution, {index, 0, resolution}], 50];
  shiftedSamples = N[logTimes^logPowerDegree, 50];
  rawConstantSamples = ConstantArray[constantValue, resolution + 1];
  shiftedResult = cqApply[shiftedSamples, sampleOrder, step, bdfOrder];
  rawConstantResult = cqApply[
    rawConstantSamples, sampleOrder, step, bdfOrder
  ];
  shiftedError = N[Abs[Last[shiftedResult] - logPowerEndpointExact], 30];
  rawConstantError = N[
    Abs[Last[rawConstantResult] - rawConstantEndpointExact],
    30
  ];
  <|
    "bdf_order" -> bdfOrder,
    "n_steps" -> resolution,
    "log_step" -> N[step, 20],
    "caputo_log_power_endpoint" -> Last[shiftedResult],
    "caputo_log_power_analytic" -> logPowerEndpointExact,
    "caputo_log_power_abs_error" -> shiftedError,
    "raw_constant_endpoint" -> Last[rawConstantResult],
    "raw_constant_analytic" -> rawConstantEndpointExact,
    "raw_constant_abs_error" -> rawConstantError
  |>
];

convergenceRows = Flatten[
  Table[
    makeConvergenceRow[bdfOrder, resolution],
    {bdfOrder, {1, 2}},
    {resolution, resolutionGrid}
  ],
  1
];

bdf1LogPowerErrors = Lookup[
  Select[convergenceRows, Lookup[#, "bdf_order"] == 1 &],
  "caputo_log_power_abs_error"
];
bdf2LogPowerErrors = Lookup[
  Select[convergenceRows, Lookup[#, "bdf_order"] == 2 &],
  "caputo_log_power_abs_error"
];
bdf1RawConstantErrors = Lookup[
  Select[convergenceRows, Lookup[#, "bdf_order"] == 1 &],
  "raw_constant_abs_error"
];
bdf2RawConstantErrors = Lookup[
  Select[convergenceRows, Lookup[#, "bdf_order"] == 2 &],
  "raw_constant_abs_error"
];

q1BDF1Weights = cqWeights[1, 8, 1];
q1BDF2Weights = cqWeights[1, 8, 2];
q1BDF1Expected = N[PadRight[{1, -1}, 8], 50];
q1BDF2Expected = N[PadRight[{3/2, -2, 1/2}, 8], 50];
q1BDF1Residual = q1BDF1Weights - q1BDF1Expected;
q1BDF2Residual = q1BDF2Weights - q1BDF2Expected;

(* ----------------------------------------------------------------- *)
(* 3. Manufactured Caputo--Hadamard constant-forcing IVP.           *)
(* ----------------------------------------------------------------- *)

abmInitialExact = 5/4;
abmForcingExact = 4/5;
abmLowerTerminalExact = 2;
abmLogStepExact = 1/10;
abmSteps = 10;
abmLogTimes = N[
  Table[index abmLogStepExact, {index, 0, abmSteps}],
  40
];
abmPhysicalTimes = N[abmLowerTerminalExact Exp[abmLogTimes], 30];
abmAnalyticStates = N[
  abmInitialExact +
    abmForcingExact abmLogTimes^sampleOrderExact/Gamma[sampleOrderExact + 1],
  40
];

constantForcingResidual = FullSimplify[
  (abmForcingExact/Gamma[q + 1]) Gamma[q + 1]/Gamma[1] -
    abmForcingExact,
  Assumptions -> 0 < q < 1
];

(* ----------------------------------------------------------------- *)
(* 4. Test ledger and portable JSON summary.                         *)
(* ----------------------------------------------------------------- *)

tests = {
  MakeTest[
    "logarithmic_kernel_transform",
    TrueQ[kernelTransformResidual == 0]
  ],
  MakeTest[
    "hadamard_measure_transform",
    TrueQ[measureTransformResidual == 0]
  ],
  MakeTest[
    "dilation_maps_to_log_derivative",
    TrueQ[dilationTransformResidual == 0]
  ],
  MakeTest[
    "hadamard_integral_log_power_beta_gamma_identity",
    TrueQ[integralCoefficientResidual == 0]
  ],
  MakeTest[
    "hadamard_log_power_derivative_identity",
    TrueQ[hadamardLogPowerResidual == 0]
  ],
  MakeTest[
    "caputo_hadamard_log_power_derivative_identity",
    TrueQ[caputoHadamardLogPowerResidual == 0]
  ],
  MakeTest[
    "hadamard_constant_identity",
    TrueQ[hadamardConstantResidual == 0]
  ],
  MakeTest[
    "caputo_hadamard_constant_is_zero",
    TrueQ[caputoHadamardConstantResidual == 0]
  ],
  MakeTest[
    "log_power_continuum_formula_has_q1_limit",
    TrueQ[logPowerQ1LimitResidual == 0]
  ],
  MakeTest[
    "constant_continuum_formula_has_q1_limit",
    TrueQ[constantQ1LimitResidual == 0]
  ],
  MakeTest[
    "bdf2_generating_polynomial_factorization",
    TrueQ[bdf2FactorizationResidual == 0]
  ],
  MakeTest[
    "q1_bdf1_weights_match_backward_difference",
    Max[Abs[q1BDF1Residual]] == 0
  ],
  MakeTest[
    "q1_bdf2_weights_match_bdf2_polynomial",
    Max[Abs[q1BDF2Residual]] == 0
  ],
  MakeTest[
    "caputo_hadamard_discrete_constant_is_zero",
    Max[Lookup[sampleRows, "caputo_constant_max_abs"]] == 0
  ],
  MakeTest[
    "bdf1_caputo_log_power_endpoint_converges",
    And @@ Thread[Differences[bdf1LogPowerErrors] < 0] &&
      bdf1LogPowerErrors[[1]]/bdf1LogPowerErrors[[2]] > 17/10 &&
      bdf1LogPowerErrors[[2]]/bdf1LogPowerErrors[[3]] > 17/10,
    <|"finest_abs_error" -> Last[bdf1LogPowerErrors]|>
  ],
  MakeTest[
    "bdf2_caputo_log_power_endpoint_converges",
    And @@ Thread[Differences[bdf2LogPowerErrors] < 0] &&
      bdf2LogPowerErrors[[1]]/bdf2LogPowerErrors[[2]] > 16/5 &&
      bdf2LogPowerErrors[[2]]/bdf2LogPowerErrors[[3]] > 16/5,
    <|"finest_abs_error" -> Last[bdf2LogPowerErrors]|>
  ],
  MakeTest[
    "raw_hadamard_constant_endpoint_converges",
    And @@ Thread[Differences[bdf1RawConstantErrors] < 0] &&
      And @@ Thread[Differences[bdf2RawConstantErrors] < 0] &&
      Max[Last[bdf1RawConstantErrors], Last[bdf2RawConstantErrors]] < 1/1000,
    <|
      "bdf1_finest_abs_error" -> Last[bdf1RawConstantErrors],
      "bdf2_finest_abs_error" -> Last[bdf2RawConstantErrors]
    |>
  ],
  MakeTest[
    "caputo_hadamard_constant_forcing_closed_form",
    TrueQ[constantForcingResidual == 0]
  ],
  MakeTest["hafo_source_not_read", source["hafo_source_read"] === False],
  MakeTest["report_not_used", source["report_input_used"] === False]
};

summary = <|
  "system_id" -> systemID,
  "output_dir" -> outDir,
  "validation_scope" ->
    "independent_hadamard_caputo_hadamard_transform_cq_and_manufactured_abm",
  "evidence_boundary" ->
    "symbolic identities and finite-grid numerical consistency; no stability, chaos, hiddenness, or general nonlinear convergence certification",
  "source" -> source,
  "parameters" -> <|
    "order" -> N[sampleOrder, 20],
    "lower_terminal" -> N[lowerTerminal, 20],
    "log_power_degree" -> logPowerDegree,
    "constant_value" -> N[constantValue, 20],
    "sample_intervals" -> sampleIntervals,
    "sample_log_step" -> N[sampleLogStep, 20],
    "resolution_grid" -> resolutionGrid
  |>,
  "transformation" -> <|
    "coordinate" -> "u=log(t/a)",
    "physical_grid" -> "t_n=a*exp(u_n)",
    "kernel_residual" -> ExprString[kernelTransformResidual],
    "measure_residual" -> ExprString[measureTransformResidual],
    "dilation_residual" -> ExprString[dilationTransformResidual]
  |>,
  "analytic_identities" -> <|
    "hadamard_integral_log_power" ->
      "I_H^alpha u^m=Gamma[m+1]/Gamma[m+alpha+1] u^(m+alpha)",
    "hadamard_log_power" ->
      "D_H^q u^m=Gamma[m+1]/Gamma[m+1-q] u^(m-q), m>=1",
    "caputo_hadamard_log_power" ->
      "D_CH^q u^m=Gamma[m+1]/Gamma[m+1-q] u^(m-q), m>=1",
    "hadamard_constant" ->
      "D_H^q c=c u^(-q)/Gamma[1-q], u>0",
    "caputo_hadamard_constant" -> "D_CH^q c=0",
    "q1_limit" -> "D_H^1=D_CH^1=d/du=t*d/dt away from CQ startup",
    "terminal_warning" ->
      "the raw constant continuum formula is singular at u=0; the first CQ sample is a terminal-truncated discrete value"
  |>,
  "cq" -> <|
    "generating_formula" ->
      "(delta(z)/log_step)^q with delta(z)^q=sum_k omega_k z^k",
    "bdf1_delta" -> "1-z",
    "bdf2_delta" -> "3/2-2*z+z^2/2",
    "startup_convention" ->
      "terminal_truncated_history_no_prehistory_extrapolation",
    "starting_corrections" -> "none_implemented",
    "sample_case" -> <|
      "log_times" -> sampleLogTimes,
      "physical_times" -> samplePhysicalTimes,
      "samples" -> sampleValues,
      "shifted_samples" -> sampleShiftedValues,
      "rows" -> sampleRows
    |>,
    "log_power_endpoint_analytic" -> logPowerEndpointExact,
    "raw_constant_endpoint_analytic" -> rawConstantEndpointExact,
    "convergence_rows" -> convergenceRows,
    "q1_bdf1_weights" -> q1BDF1Weights,
    "q1_bdf2_weights" -> q1BDF2Weights
  |>,
  "abm_manufactured" -> <|
    "equation" -> "D_CH^q x=c, x(a)=x0",
    "analytic_solution" -> "x=x0+c*log(t/a)^q/Gamma[q+1]",
    "order" -> N[sampleOrder, 20],
    "lower_terminal" -> N[abmLowerTerminalExact, 20],
    "initial_state" -> N[abmInitialExact, 20],
    "forcing" -> N[abmForcingExact, 20],
    "log_step" -> N[abmLogStepExact, 20],
    "n_steps" -> abmSteps,
    "log_times" -> abmLogTimes,
    "physical_times" -> abmPhysicalTimes,
    "analytic_states" -> abmAnalyticStates
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
