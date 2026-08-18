(* ::Package:: *)

(*
  Symbolic audit for geometric/topological deductions used in the HAFO
  hidden-attractor methodology.

  Exact scope:
    A. Picard--Caputo startup coefficients through h^(2 q).
    B. Reparameterized covariant acceleration and coordinate changes.
    C. Equivariance of J_F F and the functional-to-restart implication.
    D. Exponential-kernel Markov lift and the q=1 endpoint.
    E. Case-by-case critical surfaces and PP/FPP residuals for Chua PWL,
       Chua arctan Wu/c590, MAVPD, Kalman--Fitts, PLL, generalized Lorenz,
       Rabinovich--Fabrikant and Munoz--Pacheco.

  This file validates algebraic identities and high-precision residuals. It
  does not prove attraction, chaos, basin topology or hiddenness.
*)

ClearAll["Global`*"];
$MaxExtraPrecision = 10000;

sourceDirectory = If[StringLength[$InputFileName] > 0,
  DirectoryName[ExpandFileName[$InputFileName]], Directory[]];
reportDirectory = DirectoryName[sourceDirectory];
evidenceDirectory = FileNameJoin[{reportDirectory, "wolfram_evidence",
  "topologia_geometria_hidden"}];
If[!DirectoryQ[evidenceDirectory],
  CreateDirectory[evidenceDirectory, CreateIntermediateDirectories -> True]];
evidencePath = FileNameJoin[{evidenceDirectory,
  "topologia_geometria_hidden_validation.txt"}];

lines = {};
failureCount = 0;
emit[text_] := Module[{line = If[StringQ[text], text, ToString[text, InputForm]]},
  AppendTo[lines, line]; Print[line]];
show[name_, value_] := emit[name <> " = " <> ToString[value, InputForm]];
check[name_, condition_] := Module[{ok = TrueQ[condition]},
  emit["CHECK " <> name <> " : " <> If[ok, "PASS", "FAIL"]];
  If[!ok, failureCount++]; ok];
nearZeroQ[value_, tolerance_: 10^-35] :=
  TrueQ[Max[Abs[Flatten[N[value, 70]]]] < tolerance];

emit["WOLFRAM TOPOLOGY-GEOMETRY HIDDEN-ATTRACTOR VALIDATION"];
emit["Kernel: " <> $Version];
emit["Scope: exact identities and bounded high-precision residuals only"];
emit[""];

(* ------------------------------------------------------------------ *)
(* A. Picard--Caputo expansion.                                       *)
(* ------------------------------------------------------------------ *)
emit["[A] PICARD--CAPUTO STARTUP EXPANSION"];
Clear[h, tau, q, f1, f2, j11, j12, j21, j22];
assumptionsQ = h > 0 && 0 < q <= 1;
iqConstant = FullSimplify[
  Integrate[(h - tau)^(q - 1), {tau, 0, h},
    Assumptions -> assumptionsQ, GenerateConditions -> False]/Gamma[q],
  assumptionsQ];
iqPower = FullSimplify[
  Integrate[(h - tau)^(q - 1) tau^q, {tau, 0, h},
    Assumptions -> assumptionsQ, GenerateConditions -> False]/
      (Gamma[q] Gamma[q + 1]), assumptionsQ];
f0 = {f1, f2};
j0 = {{j11, j12}, {j21, j22}};
picardSecond = f0 iqConstant + (j0 . f0) iqPower;
picardTarget = f0 h^q/Gamma[q + 1] +
  (j0 . f0) h^(2 q)/Gamma[2 q + 1];
picardResidual = FullSimplify[picardSecond - picardTarget, assumptionsQ];
show["I_q_1", iqConstant];
show["I_q_t^q_over_Gamma_qplus1", iqPower];
show["Picard_residual_through_h_2q", picardResidual];
check["Picard-Caputo coefficients", picardResidual === {0, 0}];

(* ------------------------------------------------------------------ *)
(* B. Reparameterization and covariant acceleration.                  *)
(* ------------------------------------------------------------------ *)
emit[""];
emit["[B] COVARIANT STARTUP ACCELERATION AND COORDINATE CHANGES"];
Clear[s, c, x1, x2, y1, y2];
aStartup = f0/Gamma[q + 1];
bStartup = (j0 . f0)/Gamma[2 q + 1];
curveS = {x1, x2} + aStartup s + bStartup s^2;
velocityS0 = FullSimplify[D[curveS, s] /. s -> 0];
accelerationS0 = FullSimplify[D[curveS, {s, 2}] /. s -> 0];
check["s=h^q startup velocity",
  velocityS0 === f0/Gamma[q + 1]];
check["s=h^q startup acceleration",
  accelerationS0 === 2 (j0 . f0)/Gamma[2 q + 1]];

(* Nonlinear chart y=g(x)=(x1,x2+c x1^2).  fA and fB are arbitrary smooth
   components.  The same physical flat connection is transformed to y. *)
Clear[fA, fB];
xvars = {x1, x2}; yvars = {y1, y2};
gmap = {x1, x2 + c x1^2};
ginverse = {y1, y2 - c y1^2};
dg = D[gmap, {xvars}];
fieldX = {fA[x1, x2], fB[x1, x2]};
jfieldX = D[fieldX, {xvars}];
fieldY = FullSimplify[(dg . fieldX) /. Thread[xvars -> ginverse]];
jfieldY = D[fieldY, {yvars}];
ordinaryAccelerationY = FullSimplify[jfieldY . fieldY];
hessianTermX = Table[
  fieldX . D[gmap[[i]], {xvars, 2}] . fieldX, {i, 2}];
ordinaryTransformTarget = FullSimplify[
  (dg . (jfieldX . fieldX) + hessianTermX) /.
    Thread[xvars -> ginverse]];
ordinaryTransformResidual = FullSimplify[
  ordinaryAccelerationY - ordinaryTransformTarget];
show["Nonlinear_JF_F_transform_residual", ordinaryTransformResidual];
check["nonlinear transform JF.F = Dg JF.F + Hess(g)[F,F]",
  ordinaryTransformResidual === {0, 0}];
check["affine specialization removes Hessian term",
  FullSimplify[hessianTermX /. c -> 0] === {0, 0}];

(* Fractional startup curve transformed by g.  In y coordinates the only
   nonzero Christoffel symbol of the pulled-forward flat connection is
   Gamma^2_11=-2 c. *)
xp = {x1, x2};
curveXfrac = xp + aStartup s + bStartup s^2;
curveYfrac = FullSimplify[gmap /. Thread[xvars -> curveXfrac]];
vY0 = FullSimplify[D[curveYfrac, s] /. s -> 0];
aYOrd0 = FullSimplify[D[curveYfrac, {s, 2}] /. s -> 0];
gammaYvv = {0, -2 c vY0[[1]]^2};
aYCov0 = FullSimplify[aYOrd0 + gammaYvv];
aXCov0 = 2 (j0 . f0)/Gamma[2 q + 1];
aYCovTarget = FullSimplify[dg . aXCov0];
covariantResidual = FullSimplify[aYCov0 - aYCovTarget];
show["Covariant_fractional_startup_residual", covariantResidual];
check["covariant startup acceleration transforms as a vector",
  covariantResidual === {0, 0}];
check["q=1 recovers nabla_F F",
  FullSimplify[{2/Gamma[2 q + 1], 1/Gamma[q + 1]^2} /. q -> 1]
    === {1, 1}];

(* ------------------------------------------------------------------ *)
(* C. Equivariance and basin-logic implication.                       *)
(* ------------------------------------------------------------------ *)
emit[""];
emit["[C] EQUIVARIANCE AND FUNCTIONAL-TO-RESTART LOGIC"];
Clear[rBall, fBall, basin];
functionalImpliesRestart = TautologyQ[
  Implies[Implies[rBall, fBall] && Implies[fBall, Not[basin]],
    Implies[rBall, Not[basin]]],
  {rBall, fBall, basin}];
restartDoesNotImplyFunctional = Not@TautologyQ[
  Implies[Implies[rBall, Not[basin]], Implies[fBall, Not[basin]]],
  {rBall, fBall, basin}];
check["functional ball exclusion implies restart-slice exclusion",
  functionalImpliesRestart];
check["restart exclusion does not imply functional exclusion",
  restartDoesNotImplyFunctional];

equivarianceAudit[name_, field_, vars_, symmetry_, extraRules_: {}] := Module[
  {j, fieldSym, equivResidual, accelResidual},
  j = D[field, {vars}];
  fieldSym = FullSimplify[field /. Thread[vars -> symmetry . vars] /. extraRules];
  equivResidual = FullSimplify[fieldSym - symmetry . field /. extraRules];
  accelResidual = FullSimplify[
    ((j . field) /. Thread[vars -> symmetry . vars]) -
      symmetry . (j . field) /. extraRules];
  show[name <> "_equivariance_residual", equivResidual];
  show[name <> "_JF_F_symmetry_residual", accelResidual];
  check[name <> " field symmetry", equivResidual === ConstantArray[0, Length[vars]]];
  check[name <> " JF.F symmetry", accelResidual === ConstantArray[0, Length[vars]]];
];

(* ------------------------------------------------------------------ *)
(* D. Exponential kernel lift.                                        *)
(* ------------------------------------------------------------------ *)
emit[""];
emit["[D] EXPONENTIAL-KERNEL MARKOV LIFT"];
Clear[t, eta, c0, c1];
phi[t_] := c0 + c1 t;
zExp = Integrate[Exp[-eta (t - tau)] phi[tau], {tau, 0, t},
  Assumptions -> eta > 0 && t >= 0, GenerateConditions -> False];
zOdeResidual = FullSimplify[
  D[zExp, t] - (-eta zExp + phi[t]), eta > 0 && t >= 0];
zInitialResidual = FullSimplify[zExp /. t -> 0];
show["Exponential_memory_ODE_residual", zOdeResidual];
check["z_l'=-eta_l z_l+phi", zOdeResidual === 0];
check["exponential memory initial value", zInitialResidual === 0];
check["q=1 Volterra kernel equals one",
  FullSimplify[t^(q - 1)/Gamma[q] /. q -> 1] === 1];
check["q=1 one-mode lift eta=0,w=1 gives x'=F",
  FullSimplify[-eta z + f /. {eta -> 0}] === f];

(* ------------------------------------------------------------------ *)
(* E. Case audits.                                                     *)
(* ------------------------------------------------------------------ *)
emit[""];
emit["[E] CASE-BY-CASE ALGEBRAIC GEOMETRY"];

(* Chua PWL, regional exact calculation. *)
Clear[alpha, beta, gamma, xx, yy, zz, m0, m1];
chuaVars = {xx, yy, zz};
chuaField[h_] := {alpha (yy - xx - h), xx - yy + zz,
  -beta yy - gamma zz};
chuaCentral = chuaField[m0 xx];
chuaPositive = chuaField[m1 xx + (m0 - m1)];
chuaNegative = chuaField[m1 xx - (m0 - m1)];
chuaJ[slope_] := D[chuaField[slope xx], {chuaVars}];
chuaDet[slope_] := Factor[Det[chuaJ[slope]]];
chuaRules = {alpha -> 42281/5000, beta -> 30183/2500,
  gamma -> 13/2500, m0 -> -221/1250, m1 -> -2867/2500};
chuaDetValues = FullSimplify[{chuaDet[m0], chuaDet[m1]} /. chuaRules];
show["Chua_PWL_detJ_formula", chuaDet[m]];
show["Chua_PWL_active_detJ_values", chuaDetValues];
check["Chua PWL has no active critical surface",
  And @@ Thread[chuaDetValues != 0]];
check["Chua PWL central symmetry",
  FullSimplify[(chuaCentral /. Thread[chuaVars -> -chuaVars]) + chuaCentral]
    === {0, 0, 0}];
check["Chua PWL outer-region symmetry",
  FullSimplify[(chuaPositive /. Thread[chuaVars -> -chuaVars]) + chuaNegative]
    === {0, 0, 0}];
check["Chua PWL PP elimination by invertible J",
  And @@ Thread[chuaDetValues != 0]];

(* Chua arctan exact critical surface and high-precision PP representatives. *)
Clear[a1, a2, rho];
chuaArctanField = chuaField[a1 xx + a2 ArcTan[rho xx]];
chuaArctanJ = D[chuaArctanField, {chuaVars}];
chuaArctanDet = Factor[Det[chuaArctanJ]];
chuaArctanDetTarget = -alpha (beta + (beta + gamma) (a1 +
  a2 rho/(1 + rho^2 xx^2)));
check["Chua arctan detJ identity",
  FullSimplify[chuaArctanDet - chuaArctanDetTarget] === 0];
check["Chua arctan odd symmetry",
  FullSimplify[(chuaArctanField /. Thread[chuaVars -> -chuaVars]) +
    chuaArctanField, Element[chuaVars, Reals]] === {0, 0, 0}];

chuaArctanPP[rules_] := Module[
  {crit, x2crit, xs, hv, uu, ys, zs, point, fld, acc, determinant},
  crit = -beta/(beta + gamma) /. rules;
  x2crit = FullSimplify[(a2 rho/(crit - a1) - 1)/rho^2 /. rules];
  xs = Sqrt[x2crit];
  hv = (a1 xs + a2 ArcTan[rho xs]) /. rules;
  uu = FullSimplify[-((beta /. rules) xs +
      ((beta + gamma) /. rules) hv)/
    ((gamma /. rules) + (1 + (gamma /. rules)) crit +
      ((beta + gamma)/alpha /. rules))];
  ys = FullSimplify[xs + hv + uu/(alpha /. rules)];
  zs = FullSimplify[hv + (1 + crit) uu + uu/(alpha /. rules)];
  point = N[{xs, ys, zs}, 70];
  fld = N[chuaArctanField /. rules /. Thread[chuaVars -> point], 70];
  acc = N[chuaArctanJ . chuaArctanField /. rules /.
    Thread[chuaVars -> point], 70];
  determinant = N[chuaArctanDet /. rules /. Thread[chuaVars -> point], 70];
  <|"point" -> point, "fieldNorm" -> Norm[fld],
    "accResidual" -> Max[Abs[acc]], "detResidual" -> Abs[determinant]|>
];
wuRules = {alpha -> 42281/5000, beta -> 30183/2500,
  gamma -> 13/2500, a1 -> 2/5, a2 -> -3117/2000, rho -> 1};
c590Rules = {alpha -> 109246784533/5000000000,
  beta -> 190818408409/10000000000, gamma -> 3689006/500000000,
  a1 -> 211448967/5000000000, a2 -> -33367815123/10000000000,
  rho -> 17984259333/10000000000};
wuPP = chuaArctanPP[wuRules];
c590PP = chuaArctanPP[c590Rules];
show["Chua_arctan_Wu_PP_positive", wuPP];
show["Chua_arctan_c590_PP_positive", c590PP];
check["Wu PP/FPP representative residual", wuPP["accResidual"] < 10^-50 &&
  wuPP["fieldNorm"] > 10^-3 && wuPP["detResidual"] < 10^-50];
check["c590 PP/FPP representative residual", c590PP["accResidual"] < 10^-45 &&
  c590PP["fieldNorm"] > 10^2 && c590PP["detResidual"] < 10^-45];

(* MAVPD exact. *)
Clear[u, v, w, delta, gam, rhoM, xi];
mavpdVars = {u, v, w};
mavpdField = {delta (gam u + v - u^3), u - xi v - w, rhoM v};
mavpdJ = D[mavpdField, {mavpdVars}];
mavpdDet = Factor[Det[mavpdJ]];
mavpdPP = {Sqrt[gam/3], 2 delta gam Sqrt[gam/3]/(3 (rhoM - delta)),
  Sqrt[gam/3] - xi 2 delta gam Sqrt[gam/3]/(3 (rhoM - delta))};
mavpdAccPP = FullSimplify[mavpdJ . mavpdField /. Thread[mavpdVars -> mavpdPP],
  gam > 0 && rhoM != delta];
show["MAVPD_detJ", mavpdDet];
show["MAVPD_PP_positive", mavpdPP];
check["MAVPD critical surface u^2=gam/3",
  FullSimplify[mavpdDet /. u^2 -> gam/3] === 0];
check["MAVPD exact PP", mavpdAccPP === {0, 0, 0}];
equivarianceAudit["MAVPD", mavpdField, mavpdVars, -IdentityMatrix[3]];

(* Kalman--Fitts exact determinant and PP obstruction. *)
Clear[kx1, kx2, kx3, kx4, eps, mb1, mb2, betak];
kVars = {kx1, kx2, kx3, kx4};
kPoly = Expand[(s^2 + 2 betak s + mb1^2 + betak^2)
  (s^2 + 2 betak s + mb2^2 + betak^2)];
kA = {{0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1},
  -Table[Coefficient[kPoly, s, i], {i, 0, 3}]};
kField = kA . kVars + {0, 0, 0, 1} Tanh[-kx3/eps];
kJ = D[kField, {kVars}];
kDet = FullSimplify[Det[kJ]];
kRules = {mb1 -> 9/10, mb2 -> 11/10, betak -> 3/100, eps -> 1/100};
show["Kalman_Fitts_detJ", Factor[kDet]];
show["Kalman_Fitts_detJ_numeric", N[kDet /. kRules, 30]];
check["Kalman-Fitts detJ constant nonzero",
  FreeQ[kDet, Alternatives @@ kVars] && TrueQ[(kDet /. kRules) != 0]];
check["Kalman-Fitts odd symmetry",
  FullSimplify[(kField /. Thread[kVars -> -kVars]) + kField,
    eps > 0 && Element[kVars, Reals]] === ConstantArray[0, 4]];
check["Kalman-Fitts no nonstationary PP by invertible J",
  TrueQ[(kDet /. kRules) != 0]];

(* PLL exact critical circles and PP. *)
Clear[px, ptheta, tau1p, tau2p, lp, omegaD];
pVars = {px, ptheta}; pT = tau1p + tau2p;
pField = {-px/pT + tau1p Sin[ptheta]/(2 pT),
  omegaD - lp px/pT - tau2p lp Sin[ptheta]/(2 pT)};
pJ = D[pField, {pVars}]; pDet = FullSimplify[Det[pJ]];
pPP1 = {tau1p/2, Pi/2}; pPP2 = {-tau1p/2, 3 Pi/2};
pAcc1 = FullSimplify[pJ . pField /. Thread[pVars -> pPP1], pT != 0];
pAcc2 = FullSimplify[pJ . pField /. Thread[pVars -> pPP2], pT != 0];
show["PLL_detJ", pDet];
show["PLL_PP_acceleration_1", pAcc1];
show["PLL_PP_acceleration_2", pAcc2];
check["PLL critical circles cos(theta)=0",
  FullSimplify[pDet /. Cos[ptheta] -> 0] === 0];
check["PLL exact PP representatives", pAcc1 === {0, 0} && pAcc2 === {0, 0}];
check["PLL deck symmetry",
  FullSimplify[(pField /. ptheta -> ptheta + 2 Pi k) - pField,
    Element[k, Integers]] === {0, 0}];

(* Generalized Lorenz; exact symmetry and refined numerical PP. *)
Clear[lx, ly, lz, sigL, rL, aL];
lVars = {lx, ly, lz};
lField = {-sigL (lx - ly) - aL ly lz, rL lx - ly - lx lz,
  -lz + lx ly};
lJ = D[lField, {lVars}]; lDet = Factor[Det[lJ]];
lRules = {sigL -> 17/5, rL -> 34/5, aL -> -1/2};
equivarianceAudit["Generalized_Lorenz", lField, lVars,
  DiagonalMatrix[{-1, -1, 1}]];
lStart = {-0.745966861002653`30, 12.66180451381419`30,
  -7.442305306160726`30};
lRefined = lVars /. FindRoot[
  Thread[(lJ . lField /. lRules) == {0, 0, 0}],
  Thread[{lVars, SetPrecision[lStart, 70]}], WorkingPrecision -> 70,
  AccuracyGoal -> 50, PrecisionGoal -> 50, MaxIterations -> 300];
lAccResidual = N[lJ . lField /. lRules /. Thread[lVars -> lRefined], 60];
lFieldAtPP = N[lField /. lRules /. Thread[lVars -> lRefined], 60];
show["Generalized_Lorenz_detJ", lDet];
show["Generalized_Lorenz_PP_refined", N[lRefined, 40]];
show["Generalized_Lorenz_PP_JF_F_residual", lAccResidual];
check["Generalized Lorenz refined PP", nearZeroQ[lAccResidual, 10^-45] &&
  Norm[lFieldAtPP] > 1];
check["Generalized Lorenz PP lies on detJ=0",
  Abs[N[lDet /. lRules /. Thread[lVars -> lRefined], 60]] < 10^-45];

(* Rabinovich--Fabrikant; four representatives plus symmetry partners. *)
Clear[rx, ry, rz, ar, br];
rVars = {rx, ry, rz};
rField = {ry (rz - 1 + rx^2) + ar rx,
  rx (3 rz + 1 - rx^2) + ar ry, -2 rz (br + rx ry)};
rJ = D[rField, {rVars}]; rDet = Factor[Det[rJ]];
rRules = {ar -> 1/10, br -> 719/2500};
equivarianceAudit["Rabinovich_Fabrikant", rField, rVars,
  DiagonalMatrix[{-1, -1, 1}]];
rStarts = {
  {-1.04880884817015`30, 1.14415510709471`30, 0`30},
  {0.948683298050514`30, 0.843274042711568`30, 0`30},
  {0.217907153070161`30, 0.326271267123146`30, 1.202380810758825`30},
  {-0.197739086435437`30, 0.226277627330655`30, 0.822145202940591`30}};
rRefined = Table[rVars /. FindRoot[
  Thread[(rJ . rField /. rRules) == {0, 0, 0}],
  Thread[{rVars, SetPrecision[start, 70]}], WorkingPrecision -> 70,
  AccuracyGoal -> 50, PrecisionGoal -> 50, MaxIterations -> 300],
  {start, rStarts}];
rAccResiduals = N[(rJ . rField /. rRules /. Thread[rVars -> #]) & /@
  rRefined, 60];
rFieldNorms = N[Norm[rField /. rRules /. Thread[rVars -> #]] & /@ rRefined, 50];
rDetResiduals = N[Abs[rDet /. rRules /. Thread[rVars -> #]] & /@ rRefined, 50];
show["RF_detJ", rDet];
show["RF_PP_representatives_refined", N[rRefined, 35]];
show["RF_PP_JF_F_max_residuals", Max[Abs[#]] & /@ rAccResiduals];
check["RF four refined PP representatives",
  Max[Flatten[Abs[rAccResiduals]]] < 10^-40 && Min[rFieldNorms] > 10^-2];
check["RF PP representatives lie on detJ=0", Max[rDetResiduals] < 10^-40];

(* Munoz--Pacheco, regional exact calculation. *)
Clear[mx, my, mz, am, sg];
muVars = {mx, my, mz};
muField = {my mz + mx (my - am), 1 - sg mx, -mx my - mz};
muJ = D[muField, {muVars}]; muDet = Factor[Det[muJ]];
muRad[ep_] := 2 (1 - ep Sqrt[am]);
muPP[ep_, sx_] := {sx muRad[ep], ep Sqrt[am],
  -sx muRad[ep]^2/2};
muPairs = Tuples[{{1, -1}, {1, -1}}];
muResiduals = Table[With[
  {ep = pair[[1]], sx = pair[[2]],
   point = (muPP[pair[[1]], pair[[2]]] /. am -> 7/20)},
  FullSimplify[muJ . muField /. am -> 7/20 /. sg -> sx /.
    Thread[muVars -> point]]], {pair, muPairs}];
show["Munoz_regional_detJ", muDet];
show["Munoz_exact_PP_residuals", muResiduals];
check["Munoz four exact regional PP",
  muResiduals === ConstantArray[{0, 0, 0}, 4]];
check["Munoz regional symmetry",
  FullSimplify[(muField /. Thread[muVars ->
      DiagonalMatrix[{-1, 1, -1}] . muVars] /. sg -> -sg) -
    DiagonalMatrix[{-1, 1, -1}] . muField] === {0, 0, 0}];

(* ------------------------------------------------------------------ *)
(* F. Critical surfaces, connecting curves and the PLL KCC invariant. *)
(* ------------------------------------------------------------------ *)
emit[""];
emit["[F] CRITICAL SURFACES, CONNECTING CURVES AND PLL KCC"];

(* Regional Chua acceleration.  The affine offset d changes the field but
   not the regional Jacobian. *)
Clear[mSlope, dOffset];
chuaAffineField = chuaField[mSlope xx + dOffset];
chuaAffineJ = D[chuaAffineField, {chuaVars}];
chuaAffineAcc = FullSimplify[chuaAffineJ . chuaAffineField];
chuaAffineAccTarget = {
  alpha (-(1 + mSlope) chuaAffineField[[1]] + chuaAffineField[[2]]),
  chuaAffineField[[1]] - chuaAffineField[[2]] + chuaAffineField[[3]],
  -beta chuaAffineField[[2]] - gamma chuaAffineField[[3]]};
check["Chua regional acceleration surfaces",
  FullSimplify[chuaAffineAcc - chuaAffineAccTarget] === {0, 0, 0}];

(* MAVPD acceleration surfaces and PP inclusion in its connecting curve. *)
mavpdAcc = FullSimplify[mavpdJ . mavpdField];
mavpdAccTarget = {
  delta ((gam - 3 u^2) mavpdField[[1]] + mavpdField[[2]]),
  mavpdField[[1]] - xi mavpdField[[2]] - mavpdField[[3]],
  rhoM mavpdField[[2]]};
mavpdMinors = {
  mavpdField[[1]] mavpdAcc[[2]] - mavpdField[[2]] mavpdAcc[[1]],
  mavpdField[[1]] mavpdAcc[[3]] - mavpdField[[3]] mavpdAcc[[1]],
  mavpdField[[2]] mavpdAcc[[3]] - mavpdField[[3]] mavpdAcc[[2]]};
check["MAVPD acceleration surfaces",
  FullSimplify[mavpdAcc - mavpdAccTarget] === {0, 0, 0}];
check["MAVPD PP belongs to connecting curve",
  FullSimplify[mavpdMinors /. Thread[mavpdVars -> mavpdPP],
    delta != rhoM && gam > 0] === {0, 0, 0}];

(* Kalman--Fitts companion-form acceleration and all six rank minors. *)
kAcc = FullSimplify[kJ . kField];
kCoefficients = Table[Coefficient[kPoly, s, i], {i, 0, 3}];
kKappa = kCoefficients[[3]] + Sech[kx3/eps]^2/eps;
kH = -kCoefficients[[1]] kx2 - kCoefficients[[2]] kx3 -
  kKappa kx4 - kCoefficients[[4]] kField[[4]];
kAccTarget = {kx3, kx4, kField[[4]], kH};
kMinors = Flatten@Table[
  kField[[i]] kAcc[[j]] - kField[[j]] kAcc[[i]],
  {i, 1, 3}, {j, i + 1, 4}];
kMinorsTarget = {
  kx2 kx4 - kx3^2,
  kx2 kField[[4]] - kx3 kx4,
  kx2 kH - kx3 kField[[4]],
  kx3 kField[[4]] - kx4^2,
  kx3 kH - kx4 kField[[4]],
  kx4 kH - kField[[4]]^2};
check["Kalman-Fitts acceleration surfaces",
  FullSimplify[kAcc - kAccTarget] === {0, 0, 0, 0}];
check["Kalman-Fitts six connecting-curve minors",
  FullSimplify[kMinors - kMinorsTarget] === ConstantArray[0, 6]];

(* PLL acceleration, connecting curve and KCC deviation curvature. *)
pAcc = FullSimplify[pJ . pField];
pAccTarget = {
  (-pField[[1]] + tau1p Cos[ptheta] pField[[2]]/2)/pT,
  -lp (pField[[1]] + tau2p Cos[ptheta] pField[[2]]/2)/pT};
pConnecting = FullSimplify[
  pField[[1]] pAcc[[2]] - pField[[2]] pAcc[[1]]];
pConnectingTarget = -lp pField[[1]]^2 +
  (1 - lp tau2p Cos[ptheta]/2) pField[[1]] pField[[2]] -
  tau1p Cos[ptheta] pField[[2]]^2/2;
check["PLL acceleration surfaces", FullSimplify[pAcc - pAccTarget] === {0, 0}];
check["PLL connecting-curve equation",
  FullSimplify[pT pConnecting - pConnectingTarget] === 0];

Clear[py];
pG = -omegaD/(2 pT) + lp Sin[ptheta]/(4 pT) +
  (1 + tau2p lp Cos[ptheta]/2) py/(2 pT);
pN = D[pG, py];
pKCC = FullSimplify[-2 D[pG, ptheta] + py D[pN, ptheta] + pN^2];
pKCCTarget = pN^2 - lp Cos[ptheta]/(2 pT) +
  tau2p lp py Sin[ptheta]/(4 pT);
pParameterRules = {tau1p -> 28/625, tau2p -> 37/2000,
  lp -> 500, omegaD -> 1789/10};
pThetaF = ArcSin[1789/2500];
pThetaS = Pi - pThetaF;
pKCCValues = N[{pKCC /. py -> 0 /. ptheta -> pThetaF,
    pKCC /. py -> 0 /. ptheta -> pThetaS} /. pParameterRules, 30];
show["PLL_KCC_deviation_curvature", pKCC];
show["PLL_KCC_locked_equilibrium_values", pKCCValues];
check["PLL KCC invariant identity", FullSimplify[pKCC - pKCCTarget] === 0];
check["PLL KCC signs at focus and saddle",
  TrueQ[pKCCValues[[1]] < 0 && pKCCValues[[2]] > 0]];

emit[""];
emit["SUMMARY: " <> ToString[failureCount] <> " failed checks"];
emit["INTERPRETATION: PASS validates algebra or a declared high-precision residual; it does not prove attraction, chaos, basin topology or hiddenness."];
Export[evidencePath, StringRiffle[lines, "\n"] <> "\n", "Text"];
If[failureCount > 0, Exit[1], Exit[0]];
