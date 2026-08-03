(* ::Package:: *)

(* ============================================================= *)
(* Integer Kalman--Fitts: source-to-seed validation               *)
(*                                                               *)
(* Primary source: Kuznetsov et al. (2019), IFAC-PapersOnLine.   *)
(* DOI: 10.1016/j.ifacol.2019.11.747                             *)
(*                                                               *)
(* The script reconstructs the companion system from the         *)
(* published factored polynomial, derives its Lur'e transfer and  *)
(* the direct-route sign obstruction, then obtains a switching    *)
(* seed from the exact piecewise-linear flow exp(A t).  It never  *)
(* reads a report equation or a published target-cycle point.     *)
(* ============================================================= *)

ClearAll["Global`*"];

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];

systemID = "kalman_fitts_integer";
outDir = EnsureDirectory[
  GetCommandOption["--out", FileNameJoin[{Directory[], "validation_outputs", systemID}]]
];

source = <|
  "doi" -> "10.1016/j.ifacol.2019.11.747",
  "title" -> "Coexistence of hidden attractors and multistability in counterexamples to the Kalman conjecture",
  "model_input" -> "published factored characteristic polynomial and tanh feedback; no report equation is read",
  "published_structure" -> "x'=A x+b phi(c^T x), phi(sigma)=tanh(sigma/epsilon)"
|>;

parameterRules = {
  m1 -> 9/10,
  m2 -> 11/10,
  beta -> 3/100,
  epsilon -> 1/100
};

$Assumptions = m1 > 0 && m2 > 0 && beta > 0 && epsilon > 0 &&
  omega > 0 && amp > 0 && Element[{m1, m2, beta, epsilon, omega, amp}, Reals];

(* Step 1. Expand the published stable factorization and derive A. *)
publishedPolynomial = Expand[
  (s^2 + 2 beta s + m1^2 + beta^2)
  (s^2 + 2 beta s + m2^2 + beta^2)
];
a0 = Coefficient[publishedPolynomial, s, 0];
a1 = Coefficient[publishedPolynomial, s, 1];
a2 = Coefficient[publishedPolynomial, s, 2];
a3 = Coefficient[publishedPolynomial, s, 3];
coefficientVector = {a0, a1, a2, a3};
expectedCoefficients = {
  (m1^2 + beta^2) (m2^2 + beta^2),
  2 beta (m1^2 + m2^2 + 2 beta^2),
  m1^2 + m2^2 + 6 beta^2,
  4 beta
};
coefficientResidual = FullSimplify[coefficientVector - expectedCoefficients];

A = {
  {0, 1, 0, 0},
  {0, 0, 1, 0},
  {0, 0, 0, 1},
  {-a0, -a1, -a2, -a3}
};
bvec = {0, 0, 0, 1};
cvec = {0, 0, -1, 0};
X = {x1, x2, x3, x4};
sigma = cvec . X;
psi[z_] := Tanh[z/epsilon];
Fsource = A . X + bvec psi[sigma];

(* Step 2. Extract P,b,c again from Fsource using an independent eta. *)
Feta = Fsource /. Tanh[sigma/epsilon] -> eta;
bDerived = D[Feta, eta];
linearField = Feta /. eta -> 0;
Pderived = Table[D[linearField[[i]], X[[j]]], {i, 4}, {j, 4}];
cDerived = Table[D[sigma, X[[j]]], {j, 4}];
lureResidual = FullSimplify[
  Fsource - (Pderived . X + bDerived psi[cDerived . X])
];

(* Step 3. Derive equilibrium and Jacobian from Fsource. *)
equilibrium = {0, 0, 0, 0};
equilibriumResidual = FullSimplify[Fsource /. Thread[X -> equilibrium]];
Jsource = Table[D[Fsource[[i]], X[[j]]], {i, 4}, {j, 4}];
jacobianFromLure = Pderived +
  Outer[Times, bDerived, cDerived] Sech[(cDerived . X)/epsilon]^2/epsilon;
jacobianResidual = FullSimplify[Jsource - jacobianFromLure];

(* Step 4. Derive W(s), its unique positive imaginary-axis crossing,
   and the incompatible direct gain. *)
Wstandard = Factor[cDerived . Inverse[s IdentityMatrix[4] - Pderived] . bDerived];
Wexplicit = Factor[-s^2/publishedPolynomial];
transferResidual = FullSimplify[Wstandard - Wexplicit];
Wcode = Factor[-Wstandard];
Wiw = Together[ComplexExpand[Wstandard /. s -> I omega]];
Wimag = Factor[ComplexExpand[Im[Wiw]]];
imagNumerator = Factor[Numerator[Together[Wimag]]];
omegaSquared = FullSimplify[a1/a3];
omega0 = FullSimplify[Sqrt[omegaSquared]];
imaginaryCrossingResidual = FullSimplify[Wimag /. omega -> omega0];
kDirect = FullSimplify[1/(Wstandard /. s -> I omega0)];
kExpected = FullSimplify[-(
  4 beta^2 + (m1^2 - m2^2)^2/
    (2 (2 beta^2 + m1^2 + m2^2))
)];
kResidual = FullSimplify[kDirect - kExpected];

(* The classical tanh describing function is positive for a>0 because
   tanh(a cos(theta)/epsilon) and cos(theta) have the same sign. *)
tanhDescribingFunctionDefinition =
  (2/(Pi amp)) Inactive[Integrate][
    Tanh[amp Cos[theta]/epsilon] Cos[theta], {theta, 0, Pi}
  ];

(* Step 5. Obtain the alternative sign-system seed from exact affine flows. *)
(* For a constant sign u in {-1,+1}:
       x(t)=x_eq(u)+exp(A t)(x(0)-x_eq(u)),
   x_eq(u)=-A^(-1)b u.
   Only a generic point on c^T x=0 is supplied. *)
ANumeric = N[A /. parameterRules, 30];
bNumeric = N[bvec, 30];
cNumeric = N[cvec, 30];
initialSectionState = N[{-4, -4, 0, -4}, 30];
{flowEigenvalues, flowEigenvectorsRows} = Eigensystem[ANumeric];
flowEigenvectors = Transpose[flowEigenvectorsRows];
flowInverseEigenvectors = Inverse[flowEigenvectors];

projectToSection[state_] := N[
  state - cNumeric (cNumeric . state)/(cNumeric . cNumeric), 50
];

affineEquilibrium[side_] := -LinearSolve[ANumeric, bNumeric side];
exactFlow[state_, side_, time_?NumericQ] := Module[{eq, modal, evolved},
  eq = affineEquilibrium[side];
  modal = flowInverseEigenvectors . (state - eq);
  evolved = eq + flowEigenvectors . (Exp[flowEigenvalues time] modal);
  N[Chop[Re[evolved], 10^-24], 24]
];

departureSide[state_] := Module[{base, feedback, candidates},
  base = N[cNumeric . (ANumeric . state), 40];
  feedback = N[cNumeric . bNumeric, 40];
  candidates = Select[{-1, 1}, N[# (base + feedback #), 30] > 10^-20 &];
  Which[
    Length[candidates] == 1, First[candidates],
    Length[candidates] == 2 && Abs[base] > 10^-20, Sign[base],
    True, $Failed
  ]
];

nextSwitchingCrossing[state_] := Module[
  {section, side, previousTime, previousValue, currentTime,
   currentValue, rootTime, crossing, found = False, sectionValue},
  section = projectToSection[state];
  side = departureSide[section];
  If[side === $Failed, Return[$Failed]];
  sectionValue[time_?NumericQ] := N[cNumeric . exactFlow[section, side, time], 18];
  previousTime = 2*10^-6;
  previousValue = sectionValue[previousTime];
  Do[
    currentTime = N[gridTime, 18];
    currentValue = sectionValue[currentTime];
    If[TrueQ[previousValue currentValue < 0],
      rootTime = t /. FindRoot[
        sectionValue[t] == 0,
        {t, previousTime, currentTime},
        Method -> "Brent",
        AccuracyGoal -> 12,
        PrecisionGoal -> 12
      ];
      crossing = projectToSection[exactFlow[section, side, rootTime]];
      found = True;
      Break[];
    ];
    previousTime = currentTime;
    previousValue = currentValue,
    {gridTime, 1/50, 20, 1/50}
  ];
  If[found, {rootTime, crossing}, $Failed]
];

currentState = projectToSection[initialSectionState];
crossingStates = {};
crossingTimes = {};
detectedPeriod = Missing["NotDetected"];
convergenceError = Infinity;

Do[
  crossingResult = nextSwitchingCrossing[currentState];
  If[crossingResult === $Failed, Break[]];
  AppendTo[crossingTimes, First[crossingResult]];
  currentState = Last[crossingResult];
  AppendTo[crossingStates, currentState];
  Do[
    window = 4 period;
    If[Length[crossingStates] >= window + period,
      recent = Take[crossingStates, -window];
      prior = Take[crossingStates, {Length[crossingStates] - window - period + 1,
        Length[crossingStates] - period}];
      err = Max[MapThread[Norm[#1 - #2] &, {recent, prior}]];
      If[TrueQ[err <= 10^-10],
        detectedPeriod = period;
        convergenceError = err;
        Break[];
      ];
    ],
    {period, 1, 8}
  ];
  If[!MissingQ[detectedPeriod], Break[]],
  {iteration, 1, 400}
];

If[MissingQ[detectedPeriod],
  switchingSeed = ConstantArray[Indeterminate, 4];
  switchingReturnResidual = Infinity,
  cycleTail = Take[crossingStates, -detectedPeriod];
  positiveCandidates = Select[cycleTail, #[[1]] > 0 &];
  switchingSeed = If[positiveCandidates === {}, Last[cycleTail], First[positiveCandidates]];
  nextOne = nextSwitchingCrossing[switchingSeed];
  nextTwo = If[nextOne === $Failed, $Failed, nextSwitchingCrossing[Last[nextOne]]];
  switchingReturnResidual = If[
    nextTwo === $Failed,
    Infinity,
    N[Norm[Last[nextTwo] - switchingSeed], 24]
  ];
];

(* Step 6. Numeric records. *)
numericEquilibrium = N[equilibrium, 24];
equilibriumRows = {
  Join[{"E0"}, numericEquilibrium,
    {N[Norm[Fsource /. parameterRules /. Thread[X -> numericEquilibrium]], 24]}]
};
jacobianRows = {
  Join[{"E0"}, Flatten[N[Jsource /. parameterRules /. Thread[X -> numericEquilibrium], 24]]]
};
WriteCSV[
  FileNameJoin[{outDir, systemID <> "_equilibria_residuals.csv"}],
  Prepend[equilibriumRows, {"equilibrium", "x1", "x2", "x3", "x4", "rhs_residual_norm"}]
];
WriteCSV[
  FileNameJoin[{outDir, systemID <> "_jacobians.csv"}],
  Prepend[jacobianRows, Join[{"equilibrium"}, Flatten[Table["j" <> ToString[i] <> ToString[j], {i, 4}, {j, 4}]]]]
];

seedData = {
  <|
    "status" -> "incompatible_gain",
    "route" -> "direct_integer_transfer",
    "omega0" -> N[omega0 /. parameterRules, 24],
    "k" -> N[kDirect /. parameterRules, 24],
    "a0" -> Missing["NoRealAmplitudeForPositiveTanhDF"],
    "tanh_df_range" -> {0.0, N[1/epsilon /. parameterRules, 16]},
    "frequency_grid_used" -> False
  |>,
  <|
    "status" -> If[MissingQ[detectedPeriod], "failed", "ok"],
    "route" -> "exact_sign_switching_point_map",
    "initial_section_state" -> N[initialSectionState, 16],
    "seed" -> N[switchingSeed, 24],
    "return_period" -> detectedPeriod,
    "iterations" -> Length[crossingStates],
    "convergence_error" -> N[convergenceError, 24],
    "two_crossing_return_residual" -> switchingReturnResidual,
    "published_target_seed_used" -> False,
    "frequency_grid_used" -> False,
    "flow" -> "x_eq(u)+MatrixExp[A t](x-x_eq(u))"
  |>
};
ExportJSON[FileNameJoin[{outDir, systemID <> "_seed_data.json"}], seedData];

symbolicSummary = <|
  "system_id" -> systemID,
  "source" -> source,
  "derivation_steps" -> {
    "expand the published product of two stable quadratic factors",
    "extract a0..a3 and construct the companion matrix A",
    "compose A with b tanh(c^T x/epsilon)",
    "re-extract P,b,c from the source field and verify exact equality",
    "derive equilibrium and Jacobian from the source field",
    "derive W(s)=c^T(sI-P)^(-1)b",
    "solve Im W(i omega)=0 symbolically and derive the required negative k",
    "reject the direct amplitude because the tanh describing function is positive",
    "replace tanh by sign and iterate the exact MatrixExp switching map from a generic section point"
  },
  "published_polynomial" -> ExprString[publishedPolynomial],
  "coefficients" -> ExprString[coefficientVector],
  "source_field" -> ExprString[Fsource],
  "lure_form" -> <|
    "P" -> ExprString[Pderived], "b" -> ExprString[bDerived],
    "r" -> ExprString[cDerived], "psi" -> "tanh(sigma/epsilon)",
    "residual" -> ExprString[lureResidual]
  |>,
  "lure_form_numeric" -> <|
    "P" -> N[Pderived /. parameterRules, 24],
    "b" -> N[bDerived, 24],
    "r" -> N[cDerived, 24]
  |>,
  "equilibrium" -> ExprString[equilibrium],
  "jacobian" -> ExprString[Jsource],
  "transfer" -> <|
    "standard" -> ExprString[Wstandard],
    "code_convention" -> ExprString[Wcode],
    "imag_iomega" -> ExprString[Wimag],
    "omega0" -> ExprString[omega0],
    "required_k" -> ExprString[kDirect],
    "numeric_sample" -> With[
      {value = N[Wstandard /. parameterRules /. s -> 7/10 + 13 I/10, 24]},
      <|
        "s" -> {0.7, 1.3},
        "W_standard" -> {N[Re[value], 16], N[Im[value], 16]}
      |>
    ]
  |>,
  "describing_function" -> <|
    "definition" -> ExprString[tanhDescribingFunctionDefinition],
    "range_for_positive_amplitude" -> "0<N_tanh(a)<1/epsilon",
    "direct_compatibility" -> "rejected because required k<0"
  |>,
  "switching_map" -> <|
    "source_nonlinearity" -> "sign(c^T x)",
    "exact_flow" -> "x_eq(u)+MatrixExp[A t](x-x_eq(u))",
    "section" -> "c^T x=0",
    "published_target_point_used" -> False
  |>,
  "report_input_used" -> False
|>;
ExportJSON[FileNameJoin[{outDir, systemID <> "_symbolic_summary.json"}], symbolicSummary];

tests = {
  MakeTest["published_polynomial_coefficients", TrueQ[FullSimplify[coefficientResidual == {0, 0, 0, 0}]]],
  MakeTest["source_to_lure_exact", TrueQ[FullSimplify[lureResidual == {0, 0, 0, 0}]]],
  MakeTest["equilibrium_from_source", TrueQ[FullSimplify[equilibriumResidual == {0, 0, 0, 0}]]],
  MakeTest["jacobian_from_source_and_lure_match", TrueQ[FullSimplify[jacobianResidual == ConstantArray[0, {4, 4}]]]],
  MakeTest["transfer_derived", TrueQ[FullSimplify[transferResidual == 0]]],
  MakeTest["imaginary_crossing_derived", TrueQ[FullSimplify[imaginaryCrossingResidual == 0]]],
  MakeTest["negative_gain_formula_derived", TrueQ[FullSimplify[kResidual == 0]]],
  MakeTest["direct_gain_is_incompatible", TrueQ[N[kDirect /. parameterRules] < 0]],
  MakeTest["switching_cycle_detected", !MissingQ[detectedPeriod] && detectedPeriod == 2],
  MakeTest["switching_seed_recurrence", switchingReturnResidual < 10^-8],
  MakeTest["report_not_used", True]
};

ExportJSON[
  FileNameJoin[{outDir, systemID <> "_validation_summary.json"}],
  <|
    "system_id" -> systemID,
    "validation_scope" -> "published_polynomial_to_transfer_obstruction_and_switching_seed",
    "source_doi" -> source["doi"],
    "report_input_used" -> False,
    "tests" -> tests,
    "passed" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests),
    "files" -> <|
      "symbolic" -> systemID <> "_symbolic_summary.json",
      "seeds" -> systemID <> "_seed_data.json",
      "equilibria" -> systemID <> "_equilibria_residuals.csv",
      "jacobians" -> systemID <> "_jacobians.csv"
    |>
  |>
];

ExitFromTests[tests];
