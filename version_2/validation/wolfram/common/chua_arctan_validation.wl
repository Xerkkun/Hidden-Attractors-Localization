(* ::Package:: *)

(* ============================================================= *)
(* Chua arctan validator                                         *)
(* Builds transfer and canonical S through P0.S == S.Hq.          *)
(* The seed uses a0 times the first column of S, not eigenvectors. *)
(* ============================================================= *)

ClearAll[RunChuaArctanValidation];

RunChuaArctanValidation[case_Association] := Module[
  {
   systemID, outDir, nPrec, wPrec, params0, params, qCases, omegaSeeds, amplitudeSeeds,
   sTol, qord, w0, z, k, d, h, b1, b2, a, a0,
   x1, x2, x3, xs, lambda, X, psiCoef, psiSym, P, qv, r, originalField, lureField,
   lureCheck, M, Tz, Dz, WzFromInverse, WzExplicit, Wcheck, P0, D0z, D0Check,
   ar, bi, Tcomplex, Treal, Timag, Ap, Dcomplex, Dr, Di, Fomega,
   zr, zi, zAbsSq, desiredD0z, LCanonQ, kCanonQ, dCanonQ, residuoFrecuenciaQ,
   residuoFrecuenciaQConstante, v2R, v2I, etaR, etaI,
   s11CanonQ, s12CanonQ, s21CanonQ, s22CanonQ, s31CanonQ, s32CanonQ,
   s13CanonQ, s23CanonQ, s33CanonQ, SCanonQ, HCanonQ, XseedCanonQ,
   hCanonQ, hCanonQFull, b1CanonQ, b2CanonQ, b1CanonQFull, b2CanonQFull,
   bCanonQConParametros, SCanonQConParametros, residuoTransformacionS,
   rho, Aeq, scalarEq, J, tests, symbolicSummary, rhsNumeric, findEquilibria,
   equilibriumRows, jacobianRows, eigRows, thresholdForQ, findPositiveFrequencyRoots,
   NpsiNumeric, amplitudeFromK, evaluateCase, seedRows, maxEqResidual, jsonPath
  },

  systemID = Lookup[case, "SystemID", "chua_fractional_arctan"];
  outDir = EnsureDirectory[GetCommandOption["--out", FileNameJoin[{Directory[], "validation_outputs", systemID}]]];
  nPrec = Lookup[case, "NumericPrecision", 30];
  wPrec = Lookup[case, "WorkingPrecision", 70];
  sTol = Lookup[case, "SimilarityTolerance", 10^-16];
  params0 = Lookup[case, "Parameters"];
  params = RulePrecision[params0, wPrec];
  qCases = SetPrecision[Lookup[case, "QCases", {0.9998}], wPrec];
  omegaSeeds = SetPrecision[Lookup[case, "OmegaSeeds", {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}], wPrec];
  amplitudeSeeds = SetPrecision[Lookup[case, "AmplitudeSeeds", {0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 12, 20, 40}], wPrec];

  ClearAll[qord, w0, z, k, d, h, b1, b2, a, a0, x1, x2, x3, xs, lambda];
  $Assumptions = alpha > 0 && beta > 0 && gamma > 0 && 0 < qord <= 1 && w0 > 0 && a > 0 && a0 > 0 && Element[{m, n, k, d, h, b1, b2, z, w0, ar, bi}, Reals];

  psiCoef = n - m;
  rho = gamma/(beta + gamma);
  Aeq = 1 + m - rho;

  X = {x1, x2, x3};
  psiSym[s_] := psiCoef ArcTan[s];
  originalField = {alpha (x2 - x1) - alpha (m x1 + psiCoef ArcTan[x1]), x1 - x2 + x3, -beta x2 - gamma x3};
  P = {{-alpha (1 + m), alpha, 0}, {1, -1, 1}, {0, -beta, -gamma}};
  qv = {-alpha, 0, 0};
  r = {1, 0, 0};
  lureField = P.X + qv psiSym[r.X];
  lureCheck = CleanRat /@ (lureField - originalField);

  M = z IdentityMatrix[3] - P;
  Tz = CleanPoly[(z + 1) (z + gamma) + beta, z];
  Dz = CleanPoly[(z + alpha (1 + m)) Tz - alpha (z + gamma), z];
  WzFromInverse = CleanRat[r . Inverse[M] . qv];
  WzExplicit = CleanRat[-alpha Tz/Dz];
  Wcheck = CleanRat[WzFromInverse - WzExplicit];

  P0 = Map[Factor, P + k Outer[Times, qv, r], {2}];
  D0z = CleanPoly[Det[z IdentityMatrix[3] - P0], z];
  D0Check = CleanRat[D0z - (Dz + alpha k Tz)];

  Tcomplex = Expand[((ar + I bi) + 1) ((ar + I bi) + gamma) + beta];
  Treal = ComplexExpand[Re[Tcomplex]];
  Timag = ComplexExpand[Im[Tcomplex]];
  Ap = alpha (1 + m);
  Dcomplex = Expand[((ar + I bi) + Ap) (Treal + I Timag) - alpha ((ar + I bi) + gamma)];
  Dr = ComplexExpand[Re[Dcomplex]];
  Di = ComplexExpand[Im[Dcomplex]];
  Fomega = CleanPoly[Timag Dr - Treal Di, {ar, bi}];

  zr = w0^qord Cos[qord Pi/2];
  zi = w0^qord Sin[qord Pi/2];
  zAbsSq = CleanRat[zr^2 + zi^2];
  desiredD0z = CleanPoly[(z + d) (z^2 - 2 zr z + zAbsSq), z];

  (* Canonical spectral construction in z=s^q. No eigenvector is used for S. *)
  LCanonQ = CleanRat[(zAbsSq - 2 zr (1 + gamma) - 4 zr^2 - gamma - beta + alpha)/(1 + gamma + 2 zr)];
  kCanonQ = CleanRat[LCanonQ/alpha - 1 - m];
  dCanonQ = CleanRat[LCanonQ + 1 + gamma + 2 zr];
  residuoFrecuenciaQ = CleanPoly[Expand[(D0z /. k -> kCanonQ) - (desiredD0z /. d -> dCanonQ)], z];
  residuoFrecuenciaQConstante = CleanRat[Coefficient[residuoFrecuenciaQ, z, 0]];

  v2R = CleanRat[(alpha (1 + m + k) + zr)/alpha];
  v2I = CleanRat[zi/alpha];
  etaR = gamma + zr;
  etaI = zi;

  s11CanonQ = 1; s12CanonQ = 0;
  s21CanonQ = v2R; s22CanonQ = -v2I;
  s31CanonQ = CleanRat[zr s21CanonQ + zi s22CanonQ - 1 + s21CanonQ];
  s32CanonQ = CleanRat[(-beta s21CanonQ - (gamma + zr) s31CanonQ)/zi];
  s13CanonQ = -h;
  s23CanonQ = CleanRat[-h (alpha (1 + m + k) - d)/alpha];
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

  scalarEq = CleanRat[Aeq xs + psiSym[xs]];
  J[x_] := {{-alpha (1 + m) - alpha psiCoef/(1 + x^2), alpha, 0}, {1, -1, 1}, {0, -beta, -gamma}};

  tests = {
    MakeTest["lure_form", FullSimplify[lureCheck == {0, 0, 0}]],
    MakeTest["transfer_identity", TrueQ[FullSimplify[Wcheck == 0]]],
    MakeTest["p0_determinant_identity", TrueQ[FullSimplify[D0Check == 0]]]
  };

  symbolicSummary = <|
    "system_id" -> systemID,
    "nonlinearity" -> "arctan",
    "lure_form" -> <|"P" -> ExprString[P], "b" -> ExprString[qv], "r" -> ExprString[r], "psi" -> "(n-m) ArcTan[sigma]", "residual" -> ExprString[lureCheck]|>,
    "lure_form_numeric" -> <|
      "P" -> N[P /. params, nPrec],
      "b" -> N[qv /. params, nPrec],
      "r" -> N[r, nPrec]
    |>,
    "equilibrium_scalar_equation" -> ExprString[scalarEq],
    "transfer" -> <|"Tz" -> ExprString[Tz], "Dz" -> ExprString[Dz], "Wz" -> ExprString[WzExplicit], "fractional_frequency" -> "z=(j omega)^q = omega^q exp(j q pi/2)"|>,
    "canonical_construction" -> <|"method" -> "P0.S == S.Hq with r^T.S={1,0,-h}; seed=a0*S[[All,1]]", "k_q" -> ExprString[kCanonQ], "d_q" -> ExprString[dCanonQ], "h_q" -> ExprString[hCanonQFull], "b_q" -> ExprString[bCanonQConParametros], "S_q" -> ExprString[SCanonQConParametros], "frequency_residual" -> ExprString[residuoFrecuenciaQConstante]|>,
    "describing_function" -> <|"definition" -> "N(a)=(2/(Pi a)) Integral_0^Pi psi(a Cos[theta]) Cos[theta] dtheta", "amplitude_condition" -> "N(a0)=k"|>,
    "tests" -> tests,
    "passed_symbolic" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests)
  |>;
  ExportJSON[FileNameJoin[{outDir, systemID <> "_symbolic_summary.json"}], symbolicSummary];

  rhsNumeric[xx_List] := N[{alpha (xx[[2]] - xx[[1]]) - alpha (m xx[[1]] + psiCoef ArcTan[xx[[1]]]), xx[[1]] - xx[[2]] + xx[[3]], -beta xx[[2]] - gamma xx[[3]]} /. params, 24];

  findEquilibria[] := Module[{f, rootSeeds, roots, xx},
    f[u_?NumericQ] := N[(Aeq xs + psiSym[xs]) /. params /. xs -> SetPrecision[u, wPrec], wPrec];
    rootSeeds = SetPrecision[Lookup[case, "EquilibriumSeeds", {-20, -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 20}], wPrec];
    roots = Quiet@Table[Check[xx /. FindRoot[f[xx] == 0, {xx, seed}, WorkingPrecision -> wPrec, AccuracyGoal -> 25, PrecisionGoal -> 25, MaxIterations -> 200], Nothing], {seed, rootSeeds}];
    roots = Select[roots, NumericQ[N[#]] && TrueQ[Abs[Im[N[#]]] < 10^-18] && TrueQ[Abs[f[Re[#]]] < 10^-14] &];
    Sort@DeleteDuplicates[N[Re /@ roots, nPrec], Abs[#1 - #2] < 10^-10 &]
  ];

  With[{roots = findEquilibria[]},
    equilibriumRows = Table[
      With[{xx = roots[[i]], yy = N[(rho xs) /. params /. xs -> roots[[i]], nPrec], zz = N[(-beta/(beta + gamma) xs) /. params /. xs -> roots[[i]], nPrec]},
        Join[{"E" <> ToString[i - 1]}, {xx, yy, zz}, {Norm[rhsNumeric[{xx, yy, zz}]]}]
      ], {i, Length[roots]}]
  ];
  maxEqResidual = If[equilibriumRows === {}, N[Infinity], N[Max[equilibriumRows[[All, -1]]], 24]];
  WriteCSV[FileNameJoin[{outDir, systemID <> "_equilibria_residuals.csv"}], Prepend[equilibriumRows, {"equilibrium", "x", "y", "z", "rhs_residual_norm"}]];

  jacobianRows = (Join[{#[[1]]}, Flatten[N[J[#[[2]]] /. params, 24]]] & /@ (equilibriumRows[[All, {1, 2}]]));
  WriteCSV[FileNameJoin[{outDir, systemID <> "_jacobians.csv"}], Prepend[jacobianRows, {"equilibrium", "j11", "j12", "j13", "j21", "j22", "j23", "j31", "j32", "j33"}]];

  thresholdForQ[qq_] := N[qq Pi/2, 24];
  eigRows = Flatten[Table[
      With[{thr = thresholdForQ[qq]},
        Flatten[Table[
          With[{eig = N[Eigenvalues[N[J[equilibriumRows[[i, 2]]] /. params, 40]], 24]},
            ({N[qq, 12], equilibriumRows[[i, 1]], Re[#], Im[#], Abs[Arg[#]], Abs[Arg[#]] - thr} & /@ eig)
          ], {i, Length[equilibriumRows]}], 1]
      ], {qq, qCases}], 1];
  WriteCSV[FileNameJoin[{outDir, systemID <> "_eigenvalues_matignon.csv"}], Prepend[eigRows, {"q", "equilibrium", "real", "imag", "abs_argument", "matignon_margin"}]];

  findPositiveFrequencyRoots[qval_] := Module[{expr, f, roots, qnum, ww},
    qnum = SetPrecision[qval, wPrec];
    expr = N[residuoFrecuenciaQConstante /. params /. qord -> qnum, wPrec];
    f[x_?NumericQ] := Quiet@Check[N[Re[expr /. w0 -> SetPrecision[x, wPrec]], wPrec], Indeterminate];
    roots = Quiet@Table[Check[ww /. FindRoot[f[ww] == 0, {ww, seed}, WorkingPrecision -> wPrec, AccuracyGoal -> 25, PrecisionGoal -> 25, MaxIterations -> 200], Nothing], {seed, omegaSeeds}];
    roots = Select[roots, NumericQ[N[#]] && TrueQ[Abs[Im[N[#]]] < 10^-18] && TrueQ[0 < Re[#] < 50] && TrueQ[Abs[f[Re[#]]] < 10^-12] &];
    Sort@DeleteDuplicates[N[Re /@ roots, nPrec], Abs[#1 - #2] < 10^-10 &]
  ];

  NpsiNumeric[amp_?NumericQ] := Module[{coef = N[psiCoef /. params, wPrec], aa = SetPrecision[amp, wPrec], th},
    N[(2/(Pi aa)) NIntegrate[coef ArcTan[aa Cos[th]] Cos[th], {th, 0, Pi}, WorkingPrecision -> wPrec, AccuracyGoal -> 25, PrecisionGoal -> 25, MaxRecursion -> 30], wPrec]
  ];

  amplitudeFromK[kval_] := Module[{aa, roots, kvalPrec, f},
    kvalPrec = SetPrecision[kval, wPrec];
    f[u_?NumericQ] := N[NpsiNumeric[u] - kvalPrec, wPrec];
    roots = Quiet@Table[Check[aa /. FindRoot[f[aa] == 0, {aa, seed}, WorkingPrecision -> wPrec, AccuracyGoal -> 22, PrecisionGoal -> 22, MaxIterations -> 200], Nothing], {seed, amplitudeSeeds}];
    roots = Select[roots, NumericQ[N[#]] && TrueQ[Abs[Im[N[#]]] < 10^-16] && TrueQ[Re[#] > 0] && TrueQ[Abs[f[Re[#]]] < 10^-10] &];
    roots = Sort@DeleteDuplicates[N[Re /@ roots, nPrec], Abs[#1 - #2] < 10^-8 &];
    If[roots === {}, Missing["No real positive amplitude root for arctan describing function"], First[roots]]
  ];

  evaluateCase[qval_] := Module[{omegas, rules, kval, dval, hval, bval, aval, sval, seedPlus, seedMinus, freqResidual, ampResidual, p0val, hmatval, sResidual, zval, Wval},
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
        ampResidual = N[NpsiNumeric[aval] - SetPrecision[kval, wPrec], nPrec];
        p0val = N[P0 /. Join[rules, {k -> SetPrecision[kval, wPrec]}], nPrec];
        hmatval = N[HCanonQ /. Join[rules, {d -> SetPrecision[dval, wPrec]}], nPrec];
        zval = SetPrecision[omegas[[j]], wPrec]^SetPrecision[qval, wPrec] * Exp[I * SetPrecision[qval, wPrec] * Pi / 2];
        Wval = N[r . Inverse[zval * IdentityMatrix[3] - P] . qv /. params, nPrec];
        sResidual = N[Norm[Flatten[p0val . sval - sval . hmatval]], nPrec];
        <|"q" -> N[qval, 12], "branch" -> j, "omega0" -> N[omegas[[j]], nPrec], "a0" -> aval, "k" -> kval, "d" -> dval, "h" -> hval, "b" -> bval, "S" -> sval, "seed_plus" -> seedPlus, "seed_minus" -> seedMinus, "frequency_residual" -> freqResidual, "amplitude_residual" -> ampResidual, "similarity_residual" -> sResidual, "seed_construction" -> "a0*S[[All,1]]", "z_re" -> N[Re[zval], nPrec], "z_im" -> N[Im[zval], nPrec], "W_re" -> N[Re[Wval], nPrec], "W_im" -> N[Im[Wval], nPrec], "status" -> "ok"|>
      ], {j, Length[omegas]}]
  ];

  seedRows = Flatten[evaluateCase /@ qCases, 1];
  ExportJSON[FileNameJoin[{outDir, systemID <> "_seed_data.json"}], seedRows];
  WriteCSV[FileNameJoin[{outDir, systemID <> "_seed_summary.csv"}], Prepend[
    ({Lookup[#, "q", ""], Lookup[#, "branch", ""], Lookup[#, "omega0", ""], Lookup[#, "a0", ""], Lookup[#, "k", ""], Lookup[#, "d", ""], Lookup[#, "h", ""], Lookup[#, "frequency_residual", ""], Lookup[#, "amplitude_residual", ""], Lookup[#, "similarity_residual", ""], Lookup[#, "status", ""]} & /@ seedRows),
    {"q", "branch", "omega0", "a0", "k", "d", "h", "frequency_residual", "amplitude_residual", "similarity_residual", "status"}]];

  tests = Join[tests, {
    MakeTest["equilibrium_residuals_numeric", TrueQ[N[maxEqResidual] < 10^-12], <|"max_rhs_residual_norm" -> N[maxEqResidual, 16]|>],
    MakeTest["seed_similarity_residuals_numeric", AllTrue[Select[seedRows, Lookup[#, "status", ""] == "ok" &], Abs[Lookup[#, "similarity_residual", 1]] < sTol &]]
  }];

  jsonPath = FileNameJoin[{outDir, systemID <> "_validation_summary.json"}];
  ExportJSON[jsonPath, <|"system_id" -> systemID, "output_dir" -> outDir, "tests" -> tests, "passed" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests), "files" -> <|"symbolic" -> systemID <> "_symbolic_summary.json", "seeds" -> systemID <> "_seed_data.json"|>|>];

  If[TrueQ[Lookup[case, "ExitOnFailure", True]], ExitFromTests[tests]];
  tests
]
