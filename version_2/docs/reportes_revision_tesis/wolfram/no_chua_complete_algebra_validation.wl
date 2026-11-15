(* ::Package:: *)

(* Algebraic validation for every fractional non-Chua system whose numerical
   diagnostics are reported in the master document. Integer Kalman--Fitts,
   MAVPD and PLL have dedicated source-to-seed validators under
   validation/wolfram/cases. *)

ClearAll["Global`*"];

outputDirectory = DirectoryName[$InputFileName];
outputFile = FileNameJoin[{outputDirectory, "no_chua_complete_algebra_validation.json"}];

complexPair[value_] := N[{Re[value], Im[value]}, 16];

linearizationData[matrix_, q_] := Module[
  {dimension, polynomial, coefficients, eigenvalues, polynomialResidual,
   traceResidual, criticalOrder, margin},
  dimension = Length[matrix];
  polynomial = Expand[Det[lambda IdentityMatrix[dimension] - matrix]];
  coefficients = Reverse[CoefficientList[polynomial, lambda]];
  eigenvalues = N[Eigenvalues[N[matrix, 60]], 30];
  polynomialResidual = Max[Abs[N[polynomial /. lambda -> #, 24]] & /@ eigenvalues];
  traceResidual = Abs[N[Total[eigenvalues] - Tr[matrix], 24]];
  criticalOrder = N[Min[(2 Abs[Arg[#]]/Pi) & /@ eigenvalues], 16];
  margin = N[Min[Abs[Arg[#]] & /@ eigenvalues] - q Pi/2, 16];
  <|
    "jacobian" -> N[matrix, 16],
    "trace" -> N[Tr[matrix], 16],
    "characteristic_coefficients_leading_to_constant" -> N[coefficients, 16],
    "eigenvalues_real_imag" -> (complexPair /@ eigenvalues),
    "characteristic_max_residual" -> N[polynomialResidual, 16],
    "trace_sum_residual" -> N[traceResidual, 16],
    "matignon_q_critical" -> criticalOrder,
    "matignon_margin_at_reported_q" -> margin
  |>
];

fieldCaseData[field_, variables_, q_] := Module[
  {solutions, equilibria, jacobian, residuals, linearizations},
  solutions = NSolve[Thread[field == 0], variables, Reals, WorkingPrecision -> 60];
  equilibria = N[variables /. solutions, 40];
  jacobian = D[field, {variables}];
  residuals = N[Norm[field /. Thread[variables -> #]], 30] & /@ equilibria;
  linearizations = linearizationData[
      jacobian /. Thread[variables -> #], q
    ] & /@ equilibria;
  <|
    "reported_order" -> N[q, 16],
    "equilibria" -> (N[#, 16] & /@ equilibria),
    "equilibrium_residual_norms" -> (N[#, 16] & /@ residuals),
    "linearizations" -> linearizations
  |>
];

manualCaseData[equilibria_, matrices_, q_] := <|
  "reported_order" -> N[q, 16],
  "equilibria" -> N[equilibria, 16],
  "equilibrium_residual_norms" -> ConstantArray[0., Length[equilibria]],
  "linearizations" -> (linearizationData[#, q] & /@ matrices)
|>;

(* Generalized Lorenz, Danca (2017), q=0.995. *)
rGeneralized = 34/5;
aGeneralized = -1/2;
sigmaGeneralized = -aGeneralized rGeneralized;
lorenzGeneralizedVariables = {x1, x2, x3};
lorenzGeneralizedField = {
  -sigmaGeneralized (x1 - x2) - aGeneralized x2 x3,
  rGeneralized x1 - x2 - x1 x3,
  -x3 + x1 x2
};

(* Standard fractional Lorenz benchmark, q=0.985. *)
sigmaStandard = 10;
rhoStandard = 200;
betaStandard = 8/3;
lorenzStandardVariables = {lx, ly, lz};
lorenzStandardField = {
  sigmaStandard (ly - lx),
  rhoStandard lx - ly - lx lz,
  lx ly - betaStandard lz
};

(* Rabinovich--Fabrikant, both the hidden-attractor literature parameters and
   the Lyapunov benchmark parameters. *)
rfVariables = {rx, ry, rz};
rfField[bValue_] := {
  ry (rz - 1 + rx^2) + rx/10,
  rx (3 rz + 1 - rx^2) + ry/10,
  -2 rz (bValue + rx ry)
};

(* Exponential jerk benchmark. *)
jerkEquilibria = {{0, 0, 0}};
jerkMatrices = {{{0, 1, 0}, {0, 0, 1}, {-1, -1/5200000, -1/2}}};

(* Piecewise-smooth financial benchmark. At E0 the two lateral Jacobians have
   the same characteristic polynomial; both are retained explicitly. *)
financialEquilibria = {
  {0, 20/3, 0}, {0, 20/3, 0}, {7/10, 2, -7/10}, {-7/10, 2, 7/10}
};
financialMatrices = {
  {{17/3, 0, 1}, {-1, -3/20, 0}, {-1, 0, -1}},
  {{17/3, 0, 1}, {1, -3/20, 0}, {-1, 0, -1}},
  {{1, 7/10, 1}, {-1, -3/20, 0}, {-1, 0, -1}},
  {{1, -7/10, 1}, {1, -3/20, 0}, {-1, 0, -1}}
};

(* Four-wing benchmark. *)
fourWingVariables = {wx, wy, wz};
fourWingField = {-wx + wy wz, wy - wx wz, 53/100 - 3 wz + wx wy};

(* Exact realization checks. *)
lorenzGeneralizedA = {
  {-sigmaGeneralized, sigmaGeneralized, 0},
  {rGeneralized, -1, 0}, {0, 0, -1}
};
lorenzGeneralizedB = {{-aGeneralized, 0, 0}, {0, -1, 0}, {0, 0, 1}};
lorenzGeneralizedPhi = {x2 x3, x1 x3, x1 x2};

lorenzStandardA = {
  {-sigmaStandard, sigmaStandard, 0},
  {rhoStandard, -1, 0}, {0, 0, -betaStandard}
};
lorenzStandardB = {{0, 0}, {-1, 0}, {0, 1}};
lorenzStandardPhi = {lx lz, lx ly};

rfA[bValue_] := {{1/10, -1, 0}, {1, 1/10, 0}, {0, 0, -2 bValue}};
rfB = {{1, 1, 0, 0, 0}, {0, 0, 3, -1, 0}, {0, 0, 0, 0, -2}};
rfPhi = {ry rz, rx^2 ry, rx rz, rx^3, rx ry rz};

jerkA = {{0, 1, 0}, {0, 0, 1}, {-1, 0, -1/2}};
jerkB = {0, 0, -1};
jerkC = {0, 1, 0};
jerkTransfer = FullSimplify[jerkC . Inverse[w IdentityMatrix[3] - jerkA] . jerkB];
jerkTransferExpected = -w/(w^3 + w^2/2 + 1);

financialA = {{-1, 0, 1}, {0, -3/20, 0}, {-1, 0, -1}};
financialD = {0, 1, 0};
financialB = {{1, 0}, {0, -1}, {0, 0}};
financialPhi = {fx fy, Abs[fx]};
financialField = {fz + (fy - 1) fx, 1 - 3 fy/20 - Abs[fx], -fx - fz};

fourWingA = DiagonalMatrix[{-1, 1, -3}];
fourWingD = {0, 0, 53/100};
fourWingB = DiagonalMatrix[{1, -1, 1}];
fourWingPhi = {wy wz, wx wz, wx wy};

identityChecks = <|
  "generalized_lorenz_lure_residual" ->
    Expand[lorenzGeneralizedA . lorenzGeneralizedVariables +
      lorenzGeneralizedB . lorenzGeneralizedPhi - lorenzGeneralizedField],
  "standard_lorenz_affine_residual" ->
    Expand[lorenzStandardA . lorenzStandardVariables +
      lorenzStandardB . lorenzStandardPhi - lorenzStandardField],
  "rf_hidden_parameter_lure_residual" ->
    Expand[rfA[719/2500] . rfVariables + rfB . rfPhi - rfField[719/2500]],
  "rf_benchmark_parameter_lure_residual" ->
    Expand[rfA[49/50] . rfVariables + rfB . rfPhi - rfField[49/50]],
  "jerk_transfer_residual" -> FullSimplify[jerkTransfer - jerkTransferExpected],
  "financial_affine_residual" -> FullSimplify[
    financialA . {fx, fy, fz} + financialD + financialB . financialPhi -
      financialField,
    Element[{fx, fy, fz}, Reals]
  ],
  "four_wing_affine_residual" -> Expand[
    fourWingA . fourWingVariables + fourWingD + fourWingB . fourWingPhi -
      fourWingField
  ]
|>;

cases = <|
  "generalized_lorenz_q0995" ->
    fieldCaseData[lorenzGeneralizedField, lorenzGeneralizedVariables, 199/200],
  "standard_lorenz_q0985" ->
    fieldCaseData[lorenzStandardField, lorenzStandardVariables, 197/200],
  "rabinovich_fabrikant_b02876_q0998" ->
    fieldCaseData[rfField[719/2500], rfVariables, 499/500],
  "rabinovich_fabrikant_b098_q0999" ->
    fieldCaseData[rfField[49/50], rfVariables, 999/1000],
  "exponential_jerk_commensurate_q1" ->
    manualCaseData[jerkEquilibria, jerkMatrices, 1],
  "financial_commensurate_q1" ->
    manualCaseData[financialEquilibria, financialMatrices, 1],
  "four_wing_commensurate_q1" ->
    fieldCaseData[fourWingField, fourWingVariables, 1]
|>;

allEquilibriumResiduals = Flatten[
  Lookup[Values[cases], "equilibrium_residual_norms"]
];
allCharacteristicResiduals = Flatten[
  Lookup[Flatten[Lookup[Values[cases], "linearizations"]],
    "characteristic_max_residual"]
];
allTraceResiduals = Flatten[
  Lookup[Flatten[Lookup[Values[cases], "linearizations"]],
    "trace_sum_residual"]
];

result = <|
  "schema_version" -> "1.0",
  "kernel_version" -> $Version,
  "method" -> "equations_to_equilibria_to_jacobians_to_characteristic_polynomials",
  "identity_checks" -> identityChecks,
  "cases" -> cases,
  "global_checks" -> <|
    "identity_residuals_zero" -> And @@ (
      (TrueQ[# === 0] || TrueQ[# === {0, 0, 0}] &) /@ Values[identityChecks]
    ),
    "maximum_equilibrium_residual" -> N[Max[allEquilibriumResiduals], 16],
    "maximum_characteristic_residual" -> N[Max[allCharacteristicResiduals], 16],
    "maximum_trace_sum_residual" -> N[Max[allTraceResiduals], 16]
  |>,
  "integer_non_chua_dedicated_validators" -> {
    "validation/wolfram/cases/kalman_fitts_integer.wl",
    "validation/wolfram/cases/mavpd_integer.wl",
    "validation/wolfram/cases/pll_lead_lag_integer.wl"
  }
|>;

(* RawJSON represents arbitrary-precision zero as 0.e-n, which is not accepted
   by strict JSON parsers. Normalize that serialization before writing. *)
jsonText = ExportString[result, "RawJSON"];
jsonText = StringReplace[
  jsonText,
  RegularExpression["0\\.e-[0-9]+"] -> "0.0"
];
Export[outputFile, jsonText, "Text"];
Print[outputFile];
Print[result["global_checks"]];
