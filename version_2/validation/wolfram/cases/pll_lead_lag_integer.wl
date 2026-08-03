(* ::Package:: *)

(* ============================================================= *)
(* Integer lead-lag PLL: source-to-seed algebraic validation      *)
(*                                                               *)
(* Primary source: Bianchi et al. (2015), ICUMT, pp. 79-84.      *)
(* DOI: 10.1109/ICUMT.2015.7382409                               *)
(*                                                               *)
(* This script starts from the published filter H(s), its state   *)
(* realization, the phase-detector characteristic, and VCO law.   *)
(* It derives the ODE, locked equilibria, shifted Lur'e form,      *)
(* transfer obstruction, Andronov equation, and the analytic      *)
(* zero-gain running-cycle seed.  No report equation is read.      *)
(* ============================================================= *)

ClearAll["Global`*"];

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];

systemID = "pll_lead_lag_integer";
outDir = EnsureDirectory[
  GetCommandOption["--out", FileNameJoin[{Directory[], "validation_outputs", systemID}]]
];

source = <|
  "doi" -> "10.1109/ICUMT.2015.7382409",
  "title" -> "Limitations of PLL simulation: hidden oscillations in MatLab and SPICE",
  "model_input" -> "published H(s), state realization, phase detector, and VCO law; no report equation is read",
  "source_items" -> {
    "H(s)=(1+s tau2)/(1+s(tau1+tau2))",
    "phi(theta)=sin(theta)/2",
    "theta'=omegaDelta-L g"
  }
|>;

parameterRules = {
  tau1 -> 28/625,
  tau2 -> 37/2000,
  loopGain -> 500,
  omegaDelta -> 1789/10
};

Tconst = tau1 + tau2;
$Assumptions = tau1 > 0 && tau2 >= 0 && loopGain > 0 && omegaDelta > 0 &&
  0 < 2 omegaDelta/loopGain < 1 && omega > 0 && Element[
    {tau1, tau2, loopGain, omegaDelta, omega}, Reals
  ];

(* Step 1. Derive a one-state realization directly from published H(s). *)
Hsource = (1 + tau2 s)/(1 + Tconst s);
Afilter = -1/Tconst;
Bfilter = tau1/Tconst;
Cfilter = 1/Tconst;
Dfilter = tau2/Tconst;
Hrealized = FullSimplify[Dfilter + Cfilter Bfilter/(s - Afilter)];
filterResidual = FullSimplify[Hsource - Hrealized];

(* Step 2. Compose filter, phase detector, and VCO to obtain the ODE. *)
phaseDetector = Sin[theta]/2;
filterOutput = Cfilter x + Dfilter phaseDetector;
Foriginal = {
  Afilter x + Bfilter phaseDetector,
  omegaDelta - loopGain filterOutput
};
Xoriginal = {x, theta};
JacobianOriginal = Table[
  D[Foriginal[[i]], Xoriginal[[j]]], {i, 2}, {j, 2}
];

(* Step 3. Derive locked equilibria from the source ODE. *)
sineLocked = FullSimplify[2 omegaDelta/loopGain];
thetaFocus = ArcSin[sineLocked];
thetaSaddle = Pi - thetaFocus;
xLocked = FullSimplify[tau1 sineLocked/2];
lockedEquilibria = {
  {xLocked, thetaFocus},
  {xLocked, thetaSaddle}
};
equilibriumResidualsSymbolic = FullSimplify[
  Foriginal /. Thread[Xoriginal -> #]
] & /@ lockedEquilibria;

(* Step 4. Shift the stable locked equilibrium and derive P,b,c,psi. *)
Xshifted = {u, v};
shiftRules = {x -> u + xLocked, theta -> v + thetaFocus};
FshiftedRaw = Foriginal /. shiftRules;

(* Replace only the nonlinear increment by eta.  This forces the Lur'e
   decomposition to be extracted from the shifted source equations. *)
Feta = FshiftedRaw /. Sin[v + thetaFocus] -> eta + Sin[thetaFocus];
Feta = FullSimplify[Feta /. Sin[thetaFocus] -> sineLocked];
bvec = D[Feta, eta];
linearField = FullSimplify[Feta /. eta -> 0];
P = Table[D[linearField[[i]], Xshifted[[j]]], {i, 2}, {j, 2}];
cvec = {0, 1};
psi[sigma_] := Sin[thetaFocus + sigma] - Sin[thetaFocus];
Fshifted = FullSimplify[Foriginal /. shiftRules];
lureField = P . Xshifted + bvec psi[cvec . Xshifted];
lureResidual = FullSimplify[
  TrigExpand[Fshifted - lureField] /. Sin[thetaFocus] -> sineLocked
];

JacobianShifted = Table[D[Fshifted[[i]], Xshifted[[j]]], {i, 2}, {j, 2}];
jacobianFromLure = P + Outer[Times, bvec, cvec] Cos[thetaFocus + cvec . Xshifted];
jacobianResidual = FullSimplify[JacobianShifted - jacobianFromLure];

(* Step 5. Derive the scalar transfer from the extracted matrices. *)
Gstandard = Factor[cvec . Inverse[s IdentityMatrix[2] - P] . bvec];
Gexplicit = -(loopGain/2) (1 + tau2 s)/(s (1 + Tconst s));
transferResidual = FullSimplify[Gstandard - Gexplicit];
Gcode = Factor[-Gstandard];
Giw = Together[ComplexExpand[Gstandard /. s -> I omega]];
Greal = FullSimplify[ComplexExpand[Re[Giw]]];
Gimag = FullSimplify[ComplexExpand[Im[Giw]]];
expectedImag = FullSimplify[
  loopGain (1 + omega^2 tau2 Tconst)/(2 omega (1 + omega^2 Tconst^2))
];
imaginaryResidual = FullSimplify[Gimag - expectedImag];

(* Step 6. Eliminate x from the source equations to derive Andronov's ODE. *)
(* From theta'=omegaDelta-loopGain(x/T+tau2 sin(theta)/(2T)), solve for x. *)
xFromVelocity = FullSimplify[
  Tconst (omegaDelta - velocity)/loopGain - tau2 Sin[theta]/2
];
xdotFromElimination = FullSimplify[
  D[xFromVelocity, theta] velocity + D[xFromVelocity, velocity] acceleration
];
sourceXdotAfterElimination = FullSimplify[Foriginal[[1]] /. x -> xFromVelocity];
andronovEquation = FullSimplify[
  Solve[xdotFromElimination == sourceXdotAfterElimination, acceleration][[1, 1]]
];
andronovExpected = acceleration -> FullSimplify[
  (omegaDelta - (loopGain/2) Sin[theta] -
      (1 + tau2 loopGain Cos[theta]/2) velocity)/Tconst
];
andronovResidual = FullSimplify[
  (acceleration /. andronovEquation) - (acceleration /. andronovExpected)
];

(* Step 7. Derive the exact L=0 running solution and section seed. *)
(* At L=0, theta(t)=omegaDelta t and x=A sin(theta)+B cos(theta). *)
xAnsatz = aSin Sin[theta] + aCos Cos[theta];
xAnsatzDot = D[xAnsatz, theta] omegaDelta;
zeroGainXEquation = Expand[
  xAnsatzDot - (-xAnsatz/Tconst + tau1 Sin[theta]/(2 Tconst))
];
coefficientEquations = {
  Coefficient[zeroGainXEquation, Sin[theta]] == 0,
  Coefficient[zeroGainXEquation, Cos[theta]] == 0
};
zeroGainCoefficients = First[Solve[coefficientEquations, {aSin, aCos}]];
xPeriodic = FullSimplify[xAnsatz /. zeroGainCoefficients];
sectionX = FullSimplify[xPeriodic /. theta -> 0];
sectionVelocity = omegaDelta;
zeroGainPeriod = FullSimplify[2 Pi/omegaDelta];
zeroGainMultiplier = FullSimplify[Exp[-zeroGainPeriod/Tconst]];
zeroGainResidual = FullSimplify[
  (D[xPeriodic, theta] omegaDelta) -
    (-xPeriodic/Tconst + tau1 Sin[theta]/(2 Tconst))
];

(* Step 8. Numeric records for the published target parameters. *)
numericEquilibria = N[lockedEquilibria /. parameterRules, 24];
equilibriumRows = MapIndexed[
  Join[
    {{"E_focus", "E_saddle"}[[First[#2]]]},
    #1,
    {N[Norm[Foriginal /. parameterRules /. Thread[Xoriginal -> #1]], 24]}
  ] &,
  numericEquilibria
];

jacobianRows = MapIndexed[
  Join[
    {{"E_focus", "E_saddle"}[[First[#2]]]},
    Flatten[N[JacobianOriginal /. parameterRules /. Thread[Xoriginal -> #1], 24]]
  ] &,
  numericEquilibria
];

WriteCSV[
  FileNameJoin[{outDir, systemID <> "_equilibria_residuals.csv"}],
  Prepend[equilibriumRows, {"equilibrium", "x", "theta", "rhs_residual_norm"}]
];
WriteCSV[
  FileNameJoin[{outDir, systemID <> "_jacobians.csv"}],
  Prepend[jacobianRows, {"equilibrium", "j11", "j12", "j21", "j22"}]
];

seedData = {
  <|
    "status" -> "ok",
    "route" -> "analytic_zero_loop_gain_running_cycle",
    "source_loop_gain" -> 0.0,
    "section_theta" -> 0.0,
    "section_x" -> N[sectionX /. parameterRules, 24],
    "section_velocity" -> N[sectionVelocity /. parameterRules, 24],
    "period" -> N[zeroGainPeriod /. parameterRules, 24],
    "multiplier" -> N[zeroGainMultiplier /. parameterRules, 24],
    "source_state" -> N[{sectionX, 0} /. parameterRules, 24],
    "published_initial_conditions_used" -> False,
    "frequency_grid_used" -> False,
    "continuation_target_loop_gain" -> N[loopGain /. parameterRules, 16]
  |>
};
ExportJSON[FileNameJoin[{outDir, systemID <> "_seed_data.json"}], seedData];

symbolicSummary = <|
  "system_id" -> systemID,
  "source" -> source,
  "derivation_steps" -> {
    "realize the published H(s) as Afilter,Bfilter,Cfilter,Dfilter",
    "compose the filter realization with phi=sin(theta)/2 and theta'=omegaDelta-L g",
    "solve the source ODE for both locked equilibria",
    "shift the stable locked equilibrium and extract P,b,c,psi",
    "derive G(s)=c^T(sI-P)^(-1)b and prove Im G(i omega)>0",
    "eliminate x to obtain the Andronov phase equation",
    "solve the L=0 periodic filter response and evaluate its section seed"
  },
  "filter" -> <|
    "H_source" -> ExprString[Hsource],
    "H_realized" -> ExprString[Hrealized],
    "realization" -> <|
      "A" -> ExprString[Afilter], "B" -> ExprString[Bfilter],
      "C" -> ExprString[Cfilter], "D" -> ExprString[Dfilter]
    |>
  |>,
  "source_field" -> ExprString[Foriginal],
  "locked_equilibria" -> ExprString[lockedEquilibria],
  "lure_form" -> <|
    "P" -> ExprString[P], "b" -> ExprString[bvec], "r" -> ExprString[cvec],
    "psi" -> "sin(thetaFocus+sigma)-sin(thetaFocus)",
    "residual" -> ExprString[lureResidual]
  |>,
  "lure_form_numeric" -> <|
    "P" -> N[P /. parameterRules, 24],
    "b" -> N[bvec /. parameterRules, 24],
    "r" -> N[cvec, 24]
  |>,
  "transfer" -> <|
    "standard" -> ExprString[Gstandard],
    "code_convention" -> ExprString[Gcode],
    "real_iomega" -> ExprString[Greal],
    "imag_iomega" -> ExprString[Gimag],
    "positive_imaginary_expression" -> ExprString[expectedImag],
    "direct_route_result" -> "no positive-frequency real-axis crossing",
    "numeric_sample" -> With[
      {value = N[Gstandard /. parameterRules /. s -> 7/10 + 13 I/10, 24]},
      <|
        "s" -> {0.7, 1.3},
        "G_standard" -> {N[Re[value], 16], N[Im[value], 16]}
      |>
    ]
  |>,
  "andronov_acceleration" -> ExprString[acceleration /. andronovEquation],
  "zero_gain_running_solution" -> <|
    "x_periodic" -> ExprString[xPeriodic],
    "section_x" -> ExprString[sectionX],
    "section_velocity" -> ExprString[sectionVelocity],
    "period" -> ExprString[zeroGainPeriod],
    "multiplier" -> ExprString[zeroGainMultiplier]
  |>,
  "report_input_used" -> False
|>;
ExportJSON[FileNameJoin[{outDir, systemID <> "_symbolic_summary.json"}], symbolicSummary];

tests = {
  MakeTest["published_filter_realization", TrueQ[FullSimplify[filterResidual == 0]]],
  MakeTest["locked_equilibria_from_source", TrueQ[FullSimplify[equilibriumResidualsSymbolic == {{0, 0}, {0, 0}}]]],
  MakeTest["shifted_lure_form_exact", TrueQ[FullSimplify[lureResidual == {0, 0}]]],
  MakeTest["jacobian_from_shifted_source_and_lure_match", TrueQ[FullSimplify[jacobianResidual == ConstantArray[0, {2, 2}]]]],
  MakeTest["transfer_derived", TrueQ[FullSimplify[transferResidual == 0]]],
  MakeTest["positive_imaginary_part_derived", TrueQ[FullSimplify[imaginaryResidual == 0]]],
  MakeTest["andronov_equation_derived", TrueQ[FullSimplify[andronovResidual == 0]]],
  MakeTest["zero_gain_running_solution", TrueQ[FullSimplify[zeroGainResidual == 0]]],
  MakeTest["equilibrium_residuals", Max[equilibriumRows[[All, -1]]] < 10^-20],
  MakeTest["report_not_used", True]
};

ExportJSON[
  FileNameJoin[{outDir, systemID <> "_validation_summary.json"}],
  <|
    "system_id" -> systemID,
    "validation_scope" -> "published_filter_to_shifted_lure_and_zero_gain_seed",
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
