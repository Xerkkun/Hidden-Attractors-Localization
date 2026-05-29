(* ::Package:: *)

(* ============================================================= *)
(* Chua saturation validator                                     *)
(* Builds transfer and canonical S through P0.S == S.Hq.          *)
(* The seed uses a0 times the first column of S, not eigenvectors. *)
(* ============================================================= *)

ClearAll[RunChuaSaturationValidation];

RunChuaSaturationValidation[case_Association] := Module[
  {
   systemID, outDir, nPrec, wPrec, params0, params, qCases, omegaSeeds,
   ampTol, freqTol, sTol, qord, w0, z,
   k, d, h, b1, b2, a, a0, x1, x2, x3, xs, mu, lambda,
   Delta, rho, Aeq, X, psi, F, P, qv, r, lureField, originalField, lureCheck,
   Jmu, Mchar, Nlambda, charMu, charGrouped, charCheck,
   M, Tz, Dz, WzFromInverse, WzExplicit, Wcheck, P0, D0z, D0Check,
   ar, bi, Tcomplex, Treal, Timag, Ap, Dcomplex, Dr, Di, Wreal, Wimag, Fomega,
   Nsmall, Nlarge, zeta, zr, zi, zAbsSq, desiredD0z,
   LCanonQ, kCanonQ, dCanonQ, residuoFrecuenciaQ, residuoFrecuenciaQConstante,
   v2R, v2I, etaR, etaI, s11CanonQ, s12CanonQ, s21CanonQ, s22CanonQ,
   s31CanonQ, s32CanonQ, s13CanonQ, s23CanonQ, s33CanonQ,
   SCanonQ, HCanonQ, XseedCanonQ, residuoTransformacionS,
   hCanonQ, hCanonQFull, b1CanonQ, b2CanonQ, b1CanonQFull, b2CanonQFull,
   bCanonQConParametros, SCanonQConParametros,
   x2star, x3star, xplus, xminus, Xstar,
   symbolicSummary, tests, findPositiveFrequencyRoots, amplitudeFromK,
   evaluateCase, numericRows, seedRows, equilibriumRows, jacobianRows,
   eigRows, thresholdForQ, rhsNumeric, satNumeric, eqs, maxEqResidual,
   jsonPath
  },

  systemID = Lookup[case, "SystemID", "chua_saturation"];
  outDir = EnsureDirectory[GetCommandOption["--out", FileNameJoin[{Directory[], "validation_outputs", systemID}]]];
  nPrec = Lookup[case, "NumericPrecision", 30];
  wPrec = Lookup[case, "WorkingPrecision", 70];
  ampTol = Lookup[case, "AmplitudeTolerance", 10^-20];
  freqTol = Lookup[case, "FrequencyTolerance", 10^-18];
  sTol = Lookup[case, "SimilarityTolerance", 10^-16];
  params0 = Lookup[case, "Parameters"];
  params = RulePrecision[params0, wPrec];
  qCases = SetPrecision[Lookup[case, "QCases", {1}], wPrec];
  omegaSeeds = SetPrecision[Lookup[case, "OmegaSeeds", {1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.2, 3.4, 3.8}], wPrec];

  ClearAll[qord, w0, z, k, d, h, b1, b2, a, a0, x1, x2, x3, xs, mu, lambda];
  $Assumptions = alpha > 0 && beta > 0 && gamma > 0 && 0 < qord <= 1 && w0 > 0 && a > 0 && a0 > 0 &&
    Element[{m0, m1, k, d, h, b1, b2, z, w0, ar, bi}, Reals];

  Delta = m0 - m1;
  rho = gamma/(beta + gamma);
  Aeq = 1 + m1 - rho;

  X = {x1, x2, x3};
  psi[s_] := Delta Clip[s, {-1, 1}];
  (* symbolic Lure check uses an abstract scalar psiS to avoid piecewise expansion *)
  originalField = {alpha (x2 - x1 - m1 x1 - psiS[x1]), x1 - x2 + x3, -beta x2 - gamma x3};
  P = {{-alpha (1 + m1), alpha, 0}, {1, -1, 1}, {0, -beta, -gamma}};
  qv = {-alpha, 0, 0};
  r = {1, 0, 0};
  lureField = P.X + qv psiS[r.X];
  lureCheck = CleanRat /@ (lureField - originalField);

  Jmu = {{-alpha (1 + mu), alpha, 0}, {1, -1, 1}, {0, -beta, -gamma}};
  Mchar = lambda IdentityMatrix[3] - Jmu;
  Nlambda = Expand[(lambda + 1) (lambda + gamma) + beta];
  charMu = CleanPoly[Det[Mchar], lambda];
  charGrouped = CleanPoly[(lambda + alpha (1 + mu)) Nlambda - alpha (lambda + gamma), lambda];
  charCheck = CleanRat[charMu - charGrouped];

  M = z IdentityMatrix[3] - P;
  Tz = CleanPoly[(z + 1) (z + gamma) + beta, z];
  Dz = CleanPoly[(z + alpha (1 + m1)) Tz - alpha (z + gamma), z];
  WzFromInverse = CleanRat[r . Inverse[M] . qv];
  WzExplicit = CleanRat[-alpha Tz/Dz];
  Wcheck = CleanRat[WzFromInverse - WzExplicit];

  P0 = Map[Factor, P + k Outer[Times, qv, r], {2}];
  D0z = CleanPoly[Det[z IdentityMatrix[3] - P0], z];
  D0Check = CleanRat[D0z - (Dz + alpha k Tz)];

  Tcomplex = Expand[((ar + I bi) + 1) ((ar + I bi) + gamma) + beta];
  Treal = ComplexExpand[Re[Tcomplex]];
  Timag = ComplexExpand[Im[Tcomplex]];
  Ap = alpha (1 + m1);
  Dcomplex = Expand[((ar + I bi) + Ap) (Treal + I Timag) - alpha ((ar + I bi) + gamma)];
  Dr = ComplexExpand[Re[Dcomplex]];
  Di = ComplexExpand[Im[Dcomplex]];
  Wreal = CleanRat[-alpha (Treal Dr + Timag Di)/(Dr^2 + Di^2)];
  Wimag = CleanRat[-alpha (Timag Dr - Treal Di)/(Dr^2 + Di^2)];
  Fomega = CleanPoly[Timag Dr - Treal Di, {ar, bi}];

  Nsmall = Delta;
  Nlarge = Delta (2/Pi) (ArcSin[1/a] + Sqrt[a^2 - 1]/a^2);

  zr = w0^qord Cos[qord Pi/2];
  zi = w0^qord Sin[qord Pi/2];
  zAbsSq = CleanRat[zr^2 + zi^2];
  desiredD0z = CleanPoly[(z + d) (z^2 - 2 zr z + zAbsSq), z];

  (* Canonical spectral construction in z=s^q. No eigenvector is used for S. *)
  LCanonQ = CleanRat[(zAbsSq - 2 zr (1 + gamma) - 4 zr^2 - gamma - beta + alpha)/(1 + gamma + 2 zr)];
  kCanonQ = CleanRat[LCanonQ/alpha - 1 - m1];
  dCanonQ = CleanRat[LCanonQ + 1 + gamma + 2 zr];
  residuoFrecuenciaQ = CleanPoly[Expand[(D0z /. k -> kCanonQ) - (desiredD0z /. d -> dCanonQ)], z];
  residuoFrecuenciaQConstante = CleanRat[Coefficient[residuoFrecuenciaQ, z, 0]];

  v2R = CleanRat[(alpha (1 + m1 + k) + zr)/alpha];
  v2I = CleanRat[zi/alpha];
  etaR = gamma + zr;
  etaI = zi;

  s11CanonQ = 1; s12CanonQ = 0;
  s21CanonQ = v2R; s22CanonQ = -v2I;
  s31CanonQ = CleanRat[zr s21CanonQ + zi s22CanonQ - 1 + s21CanonQ];
  s32CanonQ = CleanRat[(-beta s21CanonQ - (gamma + zr) s31CanonQ)/zi];
  s13CanonQ = -h;
  s23CanonQ = CleanRat[-h (alpha (1 + m1 + k) - d)/alpha];
  s33CanonQ = CleanRat[h + (1 - d) s23CanonQ];

  SCanonQ = {{s11CanonQ, s12CanonQ, s13CanonQ}, {s21CanonQ, s22CanonQ, s23CanonQ}, {s31CanonQ, s32CanonQ, s33CanonQ}};
  HCanonQ = {{zr, -zi, 0}, {zi, zr, 0}, {0, 0, -d}};
  XseedCanonQ = CleanRat /@ (a0 SCanonQ[[All, 1]]);
  residuoTransformacionS = CleanRat /@ Flatten[P0 . SCanonQ - SCanonQ . HCanonQ];

  hCanonQ = CleanRat[alpha (d^2 - (1 + gamma) d + gamma + beta)/((d + zr)^2 + zi^2)];
  hCanonQFull = CleanRat[hCanonQ /. d -> dCanonQ];
  b1CanonQ = CleanRat[h - alpha];
  b2CanonQ = CleanRat[((h - alpha) (d - zr) + 2 h zr + alpha (1 + gamma))/zi];
  b1CanonQFull = CleanRat[b1CanonQ /. h -> hCanonQFull];
  b2CanonQFull = CleanRat[b2CanonQ /. {d -> dCanonQ, h -> hCanonQFull}];
  bCanonQConParametros = CleanRat /@ {b1CanonQFull, b2CanonQFull, 1};
  SCanonQConParametros = CleanRat /@ (SCanonQ /. {k -> kCanonQ, d -> dCanonQ, h -> hCanonQFull});

  x2star = rho xs;
  x3star = -beta/(beta + gamma) xs;
  xplus = CleanRat[-Delta/Aeq];
  xminus = CleanRat[Delta/Aeq];
  Xstar[x_] := {x, rho x, -beta/(beta + gamma) x};

  tests = {
    MakeTest["lure_form", FullSimplify[lureCheck == {0, 0, 0}]],
    MakeTest["charpoly_grouped", TrueQ[FullSimplify[charCheck == 0]]],
    MakeTest["transfer_identity", TrueQ[FullSimplify[Wcheck == 0]]],
    MakeTest["p0_determinant_identity", TrueQ[FullSimplify[D0Check == 0]]]
  };

  symbolicSummary = <|
    "system_id" -> systemID,
    "nonlinearity" -> "saturation",
    "lure_form" -> <|"P" -> ExprString[P], "b" -> ExprString[qv], "r" -> ExprString[r], "psi" -> "(m0-m1) sat(sigma)", "residual" -> ExprString[lureCheck]|>,
    "transfer" -> <|"Tz" -> ExprString[Tz], "Dz" -> ExprString[Dz], "Wz" -> ExprString[WzExplicit], "fractional_frequency" -> "z=(j omega)^q = omega^q exp(j q pi/2)"|>,
    "canonical_construction" -> <|"method" -> "P0.S == S.Hq with r^T.S={1,0,-h}; seed=a0*S[[All,1]]", "k_q" -> ExprString[kCanonQ], "d_q" -> ExprString[dCanonQ], "h_q" -> ExprString[hCanonQFull], "b_q" -> ExprString[bCanonQConParametros], "S_q" -> ExprString[SCanonQConParametros], "frequency_residual" -> ExprString[residuoFrecuenciaQConstante]|>,
    "describing_function" -> <|"N_small" -> ExprString[Nsmall], "N_large" -> ExprString[Nlarge], "amplitude_condition" -> "N(a0)=k"|>,
    "tests" -> tests,
    "passed_symbolic" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests)
  |>;
  ExportJSON[FileNameJoin[{outDir, systemID <> "_symbolic_summary.json"}], symbolicSummary];

  satNumeric[u_] := Clip[u, {-1, 1}];
  rhsNumeric[xx_List] := N[{alpha (xx[[2]] - xx[[1]] - m1 xx[[1]] - Delta satNumeric[xx[[1]]]), xx[[1]] - xx[[2]] + xx[[3]], -beta xx[[2]] - gamma xx[[3]]} /. params, 24];

  eqs = N[{Xstar[0], Xstar[xplus], Xstar[xminus]} /. params, 24];
  equilibriumRows = MapThread[Join[{#1}, #2, {Norm[rhsNumeric[#2]]}] &, {{{"E0", "inner"}, {"E+", "outer"}, {"E-", "outer"}}[[All, 1]], eqs}];
  maxEqResidual = N[Max[equilibriumRows[[All, -1]]], 24];
  WriteCSV[FileNameJoin[{outDir, systemID <> "_equilibria_residuals.csv"}], Prepend[equilibriumRows, {"equilibrium", "x", "y", "z", "rhs_residual_norm"}]];

  jacobianRows = {
    Join[{"inner"}, Flatten[N[(Jmu /. mu -> m0) /. params, 24]]],
    Join[{"outer"}, Flatten[N[(Jmu /. mu -> m1) /. params, 24]]]
  };
  WriteCSV[FileNameJoin[{outDir, systemID <> "_jacobians.csv"}], Prepend[jacobianRows, {"region", "j11", "j12", "j13", "j21", "j22", "j23", "j31", "j32", "j33"}]];

  thresholdForQ[qq_] := N[qq Pi/2, 24];
  eigRows = Flatten[Table[
      With[{thr = thresholdForQ[qq], eigIn = N[Eigenvalues[N[(Jmu /. mu -> m0) /. params, 40]], 24], eigOut = N[Eigenvalues[N[(Jmu /. mu -> m1) /. params, 40]], 24]},
        Join[
          ({N[qq, 12], "inner", Re[#], Im[#], Abs[Arg[#]], Abs[Arg[#]] - thr} & /@ eigIn),
          ({N[qq, 12], "outer", Re[#], Im[#], Abs[Arg[#]], Abs[Arg[#]] - thr} & /@ eigOut)
        ]
      ], {qq, qCases}], 1];
  WriteCSV[FileNameJoin[{outDir, systemID <> "_eigenvalues_matignon.csv"}], Prepend[eigRows, {"q", "region", "real", "imag", "abs_argument", "matignon_margin"}]];

  findPositiveFrequencyRoots[qval_] := Module[{expr, f, roots, qnum, ww},
    qnum = SetPrecision[qval, wPrec];
    expr = N[residuoFrecuenciaQConstante /. params /. qord -> qnum, wPrec];
    f[x_?NumericQ] := Quiet@Check[N[Re[expr /. w0 -> SetPrecision[x, wPrec]], wPrec], Indeterminate];
    roots = Quiet@Table[Check[ww /. FindRoot[f[ww] == 0, {ww, seed}, WorkingPrecision -> wPrec, AccuracyGoal -> 25, PrecisionGoal -> 25, MaxIterations -> 200], Nothing], {seed, omegaSeeds}];
    roots = Select[roots, NumericQ[N[#]] && TrueQ[Abs[Im[N[#]]] < 10^-18] && TrueQ[0 < Re[#] < 50] && TrueQ[Abs[f[Re[#]]] < 10^-12] &];
    Sort@DeleteDuplicates[N[Re /@ roots, nPrec], Abs[#1 - #2] < 10^-10 &]
  ];

  amplitudeFromK[kval_] := Module[{aa, deltaVal, aGuess, amp, kvalPrec},
    kvalPrec = SetPrecision[kval, wPrec];
    deltaVal = N[Delta /. params, wPrec];
    If[! TrueQ[0 < kvalPrec < deltaVal], Return[Missing["No real amplitude on a0>1 saturation branch"]]];
    aGuess = Max[SetPrecision[1.000001, wPrec], 4 deltaVal/(Pi kvalPrec)];
    amp = aa /. FindRoot[N[Nlarge /. params /. a -> aa, wPrec] == kvalPrec, {aa, aGuess}, WorkingPrecision -> wPrec, AccuracyGoal -> 25, PrecisionGoal -> 25, MaxIterations -> 200];
    N[amp, nPrec]
  ];

  evaluateCase[qval_] := Module[{omegas, rules, kval, dval, hval, bval, aval, sval, seedPlus, seedMinus, freqResidual, ampResidual, p0val, hmatval, sResidual},
    omegas = findPositiveFrequencyRoots[qval];
    If[omegas === {}, Return[{<|"q" -> N[qval, 12], "status" -> "no_positive_frequency_roots"|>}]];
    Table[
      rules = Join[params, {qord -> SetPrecision[qval, wPrec], w0 -> SetPrecision[omegas[[j]], wPrec]}];
      kval = N[kCanonQ /. rules, nPrec];
      dval = N[dCanonQ /. rules, nPrec];
      hval = N[hCanonQFull /. rules, nPrec];
      bval = N[bCanonQConParametros /. rules, nPrec];
      aval = amplitudeFromK[kval];
      If[Head[aval] === Missing,
        <|"q" -> N[qval, 12], "branch" -> j, "omega0" -> N[omegas[[j]], nPrec], "k" -> kval, "status" -> ToString[aval]|>,
        sval = N[SCanonQConParametros /. Join[rules, {k -> SetPrecision[kval, wPrec], d -> SetPrecision[dval, wPrec], h -> SetPrecision[hval, wPrec]}], nPrec];
        seedPlus = N[XseedCanonQ /. Join[rules, {k -> SetPrecision[kval, wPrec], d -> SetPrecision[dval, wPrec], h -> SetPrecision[hval, wPrec], a0 -> SetPrecision[aval, wPrec]}], nPrec];
        seedMinus = -seedPlus;
        freqResidual = N[residuoFrecuenciaQConstante /. rules, nPrec];
        ampResidual = N[(Nlarge /. params /. a -> SetPrecision[aval, wPrec]) - SetPrecision[kval, wPrec], nPrec];
        p0val = N[P0 /. Join[rules, {k -> SetPrecision[kval, wPrec]}], nPrec];
        hmatval = N[HCanonQ /. Join[rules, {d -> SetPrecision[dval, wPrec]}], nPrec];
        sResidual = N[Norm[Flatten[p0val . sval - sval . hmatval]], nPrec];
        <|"q" -> N[qval, 12], "branch" -> j, "omega0" -> N[omegas[[j]], nPrec], "a0" -> aval, "k" -> kval, "d" -> dval, "h" -> hval, "b" -> bval, "S" -> sval, "seed_plus" -> seedPlus, "seed_minus" -> seedMinus, "frequency_residual" -> freqResidual, "amplitude_residual" -> ampResidual, "similarity_residual" -> sResidual, "seed_construction" -> "a0*S[[All,1]]", "status" -> "ok"|>
      ], {j, Length[omegas]}]
  ];

  seedRows = Flatten[evaluateCase /@ qCases, 1];
  ExportJSON[FileNameJoin[{outDir, systemID <> "_seed_data.json"}], seedRows];
  WriteCSV[FileNameJoin[{outDir, systemID <> "_seed_summary.csv"}], Prepend[
    ({Lookup[#, "q", ""], Lookup[#, "branch", ""], Lookup[#, "omega0", ""], Lookup[#, "a0", ""], Lookup[#, "k", ""], Lookup[#, "d", ""], Lookup[#, "h", ""], Lookup[#, "frequency_residual", ""], Lookup[#, "amplitude_residual", ""], Lookup[#, "similarity_residual", ""], Lookup[#, "status", ""]} & /@ seedRows),
    {"q", "branch", "omega0", "a0", "k", "d", "h", "frequency_residual", "amplitude_residual", "similarity_residual", "status"}]];

  tests = Join[tests, {
    MakeTest["equilibrium_residuals_numeric", TrueQ[maxEqResidual < 10^-20], <|"max_rhs_residual_norm" -> maxEqResidual|>],
    MakeTest["seed_similarity_residuals_numeric", AllTrue[Select[seedRows, Lookup[#, "status", ""] == "ok" &], Abs[Lookup[#, "similarity_residual", 1]] < sTol &]]
  }];

  jsonPath = FileNameJoin[{outDir, systemID <> "_validation_summary.json"}];
  ExportJSON[jsonPath, <|"system_id" -> systemID, "output_dir" -> outDir, "tests" -> tests, "passed" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests), "files" -> <|"symbolic" -> systemID <> "_symbolic_summary.json", "seeds" -> systemID <> "_seed_data.json"|>|>];

  If[TrueQ[Lookup[case, "ExitOnFailure", True]], ExitFromTests[tests]];
  tests
]
