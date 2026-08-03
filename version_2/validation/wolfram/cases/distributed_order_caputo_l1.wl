(* ::Package:: *)

(* ============================================================= *)
(* Independent distributed-order Caputo L1 validation            *)
(*                                                               *)
(* For a declared positive discrete order measure, this script   *)
(* derives every coefficient of the combined L1 history kernel   *)
(* from Integrate applied to the Caputo kernel on one affine      *)
(* interpolation interval. It then advances a separately written *)
(* direct linear recurrence for a manufactured equation.          *)
(*                                                               *)
(* No HAFO source, generated report, or HAFO weight formula is    *)
(* read. The manufactured exact solution is affine, so the L1    *)
(* approximation is exact for every fractional node.             *)
(*                                                               *)
(* Evidence boundary: finite algebraic/numerical consistency      *)
(* only; no global convergence theorem, nonlinear stability,      *)
(* chaos, attraction, or hiddenness claim.                       *)
(*                                                               *)
(* Source anchors:                                               *)
(* M. Caputo, FCAA 4 (2001), 421--442; no DOI asserted.          *)
(* K. Diethelm, N. J. Ford, JCAM 225 (2009), 96--104,           *)
(* DOI: 10.1016/j.cam.2008.07.018                               *)
(* Z. Hu, F. Liu, V. Anh, I. Turner, ANZIAM J. 55 (2014),       *)
(* DOI: 10.21914/ANZIAMJ.V55I0.7888                             *)
(* Y. Lin, C. Xu, JCP 225 (2007), 1533--1552,                   *)
(* DOI: 10.1016/j.jcp.2007.02.001                               *)
(* ============================================================= *)

ClearAll["Global`*"];

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];

systemID = "distributed_order_caputo_l1";
outDir = EnsureDirectory[
  GetCommandOption[
    "--out",
    FileNameJoin[{
      Directory[], "validation", "outputs", "wolfram", systemID
    }]
  ]
];

source = <|
  "caputo_distributed_order_index" ->
    "https://www.math.bas.bg/complan/fcaa/volume4/index.html",
  "diethelm_ford_doi" -> "10.1016/j.cam.2008.07.018",
  "hu_liu_anh_turner_doi" -> "10.21914/ANZIAMJ.V55I0.7888",
  "lin_xu_doi" -> "10.1016/j.jcp.2007.02.001",
  "scope" ->
    "integrated multinode L1 kernel and one direct manufactured linear recurrence",
  "hafo_source_read" -> False,
  "report_input_used" -> False,
  "hafo_formula_imported" -> False
|>;

workingPrecision = 80;
lowerTerminalExact = 1/8;
stepExact = 1/20;
nSteps = 18;
orderNodesExact = {1/4, 3/5, 4/5};
orderMassesExact = {1/5, 1/2, 3/10};
initialValueExact = 7/5;
slopeExact = 3/4;
lambdaExact = -1/3;
timesExact = Table[
  lowerTerminalExact + index stepExact,
  {index, 0, nSteps}
];

(* Integrate constructs a primitive of the Caputo power-law kernel for
   every order node. At lag k, the physical distance t_n-s traverses
   [k h,(k+1) h], so evaluating the primitives at those endpoints gives
   the coefficient multiplying x_(n-k)-x_(n-k-1). *)
ClearAll[distance, orderSymbol];
kernelPrimitiveGeneric = Integrate[
  distance^(-orderSymbol),
  distance,
  Assumptions -> distance > 0 && 0 < orderSymbol < 1,
  GenerateConditions -> False
];
kernelPrimitives = Table[
  kernelPrimitiveGeneric /. orderSymbol -> order,
  {order, orderNodesExact}
];

ClearAll[integratedOrderCoefficient];
integratedOrderCoefficient[
  orderIndex_Integer?Positive,
  lag_Integer?NonNegative
] :=
  orderMassesExact[[orderIndex]]/
    (stepExact Gamma[1 - orderNodesExact[[orderIndex]]]) *
    (
      (kernelPrimitives[[orderIndex]] /.
        distance -> (lag + 1) stepExact) -
      (kernelPrimitives[[orderIndex]] /.
        distance -> lag stepExact)
    );

integratedPerOrderKernelExact = Table[
  integratedOrderCoefficient[orderIndex, lag],
  {orderIndex, 1, Length[orderNodesExact]},
  {lag, 0, nSteps - 1}
];
integratedCombinedKernelExact = Total[integratedPerOrderKernelExact];

(* This closed expression is written independently only after the kernel
   integrals above have been constructed. Equality is checked symbolically. *)
formulaPerOrderKernelExact = Table[
  orderMassesExact[[orderIndex]] *
    stepExact^(-orderNodesExact[[orderIndex]])/
    Gamma[2 - orderNodesExact[[orderIndex]]] *
    ((lag + 1)^(1 - orderNodesExact[[orderIndex]]) -
      lag^(1 - orderNodesExact[[orderIndex]])),
  {orderIndex, 1, Length[orderNodesExact]},
  {lag, 0, nSteps - 1}
];
formulaCombinedKernelExact = Total[formulaPerOrderKernelExact];

(* Prove the interval identity once for an arbitrary admissible order and
   positive lag, and once at lag zero. The multinode arrays below are then
   numeric evaluations of that same Integrate-derived identity. *)
ClearAll[massSymbol, stepSymbol, lagSymbol];
integratedGenericCoefficient =
  massSymbol/(stepSymbol Gamma[1 - orderSymbol]) *
    (
      (kernelPrimitiveGeneric /.
        distance -> (lagSymbol + 1) stepSymbol) -
      (kernelPrimitiveGeneric /.
        distance -> lagSymbol stepSymbol)
    );
formulaGenericCoefficient =
  massSymbol stepSymbol^(-orderSymbol)/Gamma[2 - orderSymbol] *
    ((lagSymbol + 1)^(1 - orderSymbol) -
      lagSymbol^(1 - orderSymbol));
kernelIdentityResidualPositive = FullSimplify[
  integratedGenericCoefficient - formulaGenericCoefficient,
  Assumptions ->
    0 < orderSymbol < 1 && massSymbol > 0 && stepSymbol > 0 &&
      lagSymbol >= 1 && Element[lagSymbol, Integers]
];
kernelIdentityResidualZero = FullSimplify[
  (integratedGenericCoefficient - formulaGenericCoefficient) /.
    lagSymbol -> 0,
  Assumptions ->
    0 < orderSymbol < 1 && massSymbol > 0 && stepSymbol > 0
];
perOrderKernelSymbolicMatch =
  TrueQ[kernelIdentityResidualPositive == 0] &&
    TrueQ[kernelIdentityResidualZero == 0];
combinedKernelSymbolicMatch = perOrderKernelSymbolicMatch;
perOrderKernelNumericResiduals = N[
  integratedPerOrderKernelExact - formulaPerOrderKernelExact,
  workingPrecision
];
combinedKernelNumericResiduals = N[
  integratedCombinedKernelExact - formulaCombinedKernelExact,
  workingPrecision
];
kernelNumericMaxResidual = N[
  Max[
    Max @@ Abs[Flatten[perOrderKernelNumericResiduals]],
    Max @@ Abs[combinedKernelNumericResiduals]
  ],
  40
];

combinedKernel = N[integratedCombinedKernelExact, workingPrecision];
perOrderKernel = N[integratedPerOrderKernelExact, workingPrecision];
currentCoefficient = combinedKernel[[1]];
kernelPositive = And @@ (TrueQ[# > 0] & /@ combinedKernel);
kernelDecreasing = And @@ Table[
  TrueQ[combinedKernel[[lag]] > combinedKernel[[lag + 1]]],
  {lag, 1, Length[combinedKernel] - 1}
];

(* Independent Caputo integral of the derivative of an affine state. *)
ClearAll[tSymbol, sSymbol, affineOrderSymbol];
linearCaputoIntegralGeneric = FullSimplify[
  1/Gamma[1 - affineOrderSymbol] Integrate[
    (tSymbol - sSymbol)^(-affineOrderSymbol) slopeExact,
    {sSymbol, lowerTerminalExact, tSymbol},
    Assumptions ->
      tSymbol > lowerTerminalExact && 0 < affineOrderSymbol < 1,
    GenerateConditions -> False
  ],
  Assumptions ->
    tSymbol > lowerTerminalExact && 0 < affineOrderSymbol < 1
];
linearCaputoIntegrals = Table[
  linearCaputoIntegralGeneric /. affineOrderSymbol -> order,
  {order, orderNodesExact}
];
linearCaputoClosedForms = Table[
  slopeExact (tSymbol - lowerTerminalExact)^(1 - order)/
    Gamma[2 - order],
  {order, orderNodesExact}
];
linearCaputoClosedGeneric =
  slopeExact (tSymbol - lowerTerminalExact)^
    (1 - affineOrderSymbol)/Gamma[2 - affineOrderSymbol];
linearCaputoIdentityResidual = FullSimplify[
  linearCaputoIntegralGeneric - linearCaputoClosedGeneric,
  Assumptions ->
    tSymbol > lowerTerminalExact && 0 < affineOrderSymbol < 1
];
linearCaputoResiduals = Table[
  linearCaputoIdentityResidual /. affineOrderSymbol -> order,
  {order, orderNodesExact}
];
linearCaputoSymbolicMatch = TrueQ[linearCaputoIdentityResidual == 0];

ClearAll[exactState, exactDistributedDerivative, forcing];
exactState[time_] :=
  initialValueExact + slopeExact (time - lowerTerminalExact);
exactDistributedDerivative[time_] := If[
  TrueQ[time == lowerTerminalExact],
  0,
  Sum[
    orderMassesExact[[orderIndex]] slopeExact *
      (time - lowerTerminalExact)^
        (1 - orderNodesExact[[orderIndex]])/
      Gamma[2 - orderNodesExact[[orderIndex]]],
    {orderIndex, 1, Length[orderNodesExact]}
  ]
];
forcing[time_] :=
  exactDistributedDerivative[time] - lambdaExact exactState[time];

(* Direct solution of the scalar linear discrete equation

     K0 (x_n-x_(n-1)) + H_n = lambda x_n + g_n.

   This is deliberately not a Picard implementation. *)
ClearAll[directLinearL1Recurrence];
directLinearL1Recurrence[] := Module[
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

manufacturedTrajectory = directLinearL1Recurrence[];
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
    "multinode_kernel_matches_integrated_caputo_intervals",
    perOrderKernelSymbolicMatch && combinedKernelSymbolicMatch &&
      kernelNumericMaxResidual < 10^-35,
    <|
      "per_order_symbolic_match" -> perOrderKernelSymbolicMatch,
      "combined_symbolic_match" -> combinedKernelSymbolicMatch,
      "numeric_max_residual" -> kernelNumericMaxResidual
    |>
  ],
  MakeTest[
    "combined_kernel_is_positive_and_strictly_decreasing",
    kernelPositive && kernelDecreasing,
    <|
      "positive" -> kernelPositive,
      "strictly_decreasing" -> kernelDecreasing
    |>
  ],
  MakeTest[
    "affine_caputo_integrals_match_closed_forms",
    linearCaputoSymbolicMatch,
    <|"symbolic_residuals" -> (ExprString /@ linearCaputoResiduals)|>
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
    "independent_distributed_order_Caputo_uniform_L1_finite_consistency",
  "evidence_boundary" ->
    "finite algebraic-numerical consistency only; no global convergence theorem, nonlinear stability, chaos, attraction, or hiddenness claim",
  "source" -> source,
  "parameters" -> <|
    "lower_terminal" -> N[lowerTerminalExact, 40],
    "step" -> N[stepExact, 40],
    "n_steps" -> nSteps,
    "order_nodes" -> N[orderNodesExact, 40],
    "order_masses" -> N[orderMassesExact, 40],
    "initial_value" -> N[initialValueExact, 40],
    "slope" -> N[slopeExact, 40],
    "lambda" -> N[lambdaExact, 40]
  |>,
  "kernel" -> <|
    "definition" ->
      "sum_j Omega_j/[h Gamma(1-alpha_j)] Integrate[r^(-alpha_j),{r,k h,(k+1) h}]",
    "primitives_from_integrate" -> (ExprString /@ kernelPrimitives),
    "per_order_integrated_values" -> N[perOrderKernel, 40],
    "per_order_formula_values" -> N[formulaPerOrderKernelExact, 40],
    "combined_integrated_values" -> N[combinedKernel, 40],
    "combined_formula_values" -> N[formulaCombinedKernelExact, 40],
    "generic_positive_lag_symbolic_residual" ->
      ExprString[kernelIdentityResidualPositive],
    "generic_zero_lag_symbolic_residual" ->
      ExprString[kernelIdentityResidualZero],
    "per_order_numeric_residuals" -> N[perOrderKernelNumericResiduals, 40],
    "combined_numeric_residuals" -> N[combinedKernelNumericResiduals, 40],
    "per_order_symbolic_match" -> perOrderKernelSymbolicMatch,
    "combined_symbolic_match" -> combinedKernelSymbolicMatch,
    "numeric_max_residual" -> kernelNumericMaxResidual,
    "current_step_coefficient" -> N[currentCoefficient, 40],
    "positive" -> kernelPositive,
    "strictly_decreasing" -> kernelDecreasing
  |>,
  "affine_caputo_identity" -> <|
    "integrated_values" -> (ExprString /@ linearCaputoIntegrals),
    "closed_forms" -> (ExprString /@ linearCaputoClosedForms),
    "symbolic_residuals" -> (ExprString /@ linearCaputoResiduals),
    "symbolic_match" -> linearCaputoSymbolicMatch
  |>,
  "manufactured_case" -> <|
    "exact_solution" -> "x(t)=7/5+(3/4)(t-1/8)",
    "rhs" ->
      "f(t,x)=lambda*x+sum_j Omega_j*slope*(t-a)^(1-alpha_j)/Gamma(2-alpha_j)-lambda*x_exact(t)",
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
