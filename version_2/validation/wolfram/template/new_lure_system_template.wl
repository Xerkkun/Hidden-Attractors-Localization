(* ::Package:: *)

(* ============================================================= *)
(* PLANTILLA PARA SISTEMA NUEVO EN FORMA LURE                    *)
(*                                                               *)
(* Objetivo:                                                     *)
(*   1. Validar F(X)=P.X+b psi(r.X).                             *)
(*   2. Calcular W_hat(z)=r^T (z I-P)^(-1) b.                    *)
(*   3. Evaluar W_q(j omega) con z=(j omega)^q.                  *)
(*   4. Construir S por P0.S == S.Hq, no por autovectores.        *)
(*   5. Exportar JSON/CSV para comparacion con Python.            *)
(*                                                               *)
(* Esta plantilla no puede resolver automaticamente todos los     *)
(* sistemas. Para un nuevo sistema, rellene las secciones marcadas *)
(* como USER INPUT y revise la solucion de S.                     *)
(* ============================================================= *)

root = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{root, "common", "ha_validation_common.wl"}]];

ClearAll["Global`*"];

systemID = "new_lure_system";
outDir = EnsureDirectory[GetCommandOption["--out", FileNameJoin[{Directory[], "validation_outputs", systemID}]]];

(* ======================= USER INPUT ========================== *)

nDim = 3;
X = {x1, x2, x3};

(* Campo original F(X). Sustituya por su sistema. *)
F = {
  f1[x1, x2, x3],
  f2[x1, x2, x3],
  f3[x1, x2, x3]
};

(* Forma Lure propuesta. *)
P = {
  {p11, p12, p13},
  {p21, p22, p23},
  {p31, p32, p33}
};

bvec = {b01, b02, b03};
rvec = {r1, r2, r3};
psi[s_] := psiSymbolic[s];

(* Parametros numericos para pruebas. *)
params = {
  p11 -> 0, p12 -> 0, p13 -> 0,
  p21 -> 0, p22 -> 0, p23 -> 0,
  p31 -> 0, p32 -> 0, p33 -> 0,
  b01 -> 0, b02 -> 0, b03 -> 0,
  r1 -> 1, r2 -> 0, r3 -> 0
};

qCases = {1};

(* Si usa funcion descriptiva, defina Npsi[a]. *)
Npsi[a_] := (2/(Pi a)) Integrate[psi[a Cos[theta]] Cos[theta], {theta, 0, Pi}];

(* =================== END USER INPUT ========================== *)

z = Unique["z"];
w0 = Unique["w0"];
qord = Unique["qord"];
k = Unique["k"];
d = Unique["d"];
h = Unique["h"];

lureField = P.X + bvec psi[rvec.X];
lureResidual = FullSimplify[F - lureField];

M = z IdentityMatrix[nDim] - P;
What = FullSimplify[rvec . Inverse[M] . bvec];
numWhat = FullSimplify[Numerator[Together[What]]];
denWhat = FullSimplify[Denominator[Together[What]]];

(* Evaluacion fraccionaria: nunca use z=I omega si q != 1. *)
zFractional = w0^qord (Cos[qord Pi/2] + I Sin[qord Pi/2]);
Wfractional = What /. z -> zFractional;

(* Matriz P0 por linealizacion armonica. *)
P0 = P + k Outer[Times, bvec, rvec];

(* Forma canonica 3D. Para nDim != 3, adaptar. *)
zr = w0^qord Cos[qord Pi/2];
zi = w0^qord Sin[qord Pi/2];
Hq = {{zr, -zi, 0}, {zi, zr, 0}, {0, 0, -d}};

Svars = Array[s, {3, 3}];
bCanon = {b1, b2, 1};

(* Proceso correcto para S:
   resolver simultaneamente P0.S == S.Hq, r^T.S == {1,0,-h}, bvec == S.{b1,b2,1}.
   Este es el criterio metodologico. No usar autovectores como sustituto de S.
*)
canonicalEquations = Join[
  Flatten[P0 . Svars == Svars . Hq],
  Thread[rvec . Svars == {1, 0, -h}],
  Thread[Svars . bCanon == bvec]
];

canonicalUnknowns = Join[Flatten[Svars], {b1, b2, h, k, d}];

(* En muchos sistemas Solve puede requerir restricciones o despejes manuales. *)
canonicalSolution = Quiet@Check[Solve[canonicalEquations, canonicalUnknowns], $Failed];

summary = <|
  "system_id" -> systemID,
  "lure_residual" -> ExprString[lureResidual],
  "passed_lure_form" -> TrueQ[FullSimplify[lureResidual == ConstantArray[0, nDim]]],
  "transfer" -> <|
    "W_hat_z" -> ExprString[What],
    "numerator" -> ExprString[numWhat],
    "denominator" -> ExprString[denWhat],
    "fractional_frequency_rule" -> "z=(j omega)^q=omega^q exp(j q pi/2)",
    "W_q_jomega" -> ExprString[Wfractional]
  |>,
  "canonical_S_method" -> "Solve P0.S == S.Hq, r^T.S == {1,0,-h}, bvec == S.{b1,b2,1}; do not use eigenvectors as S.",
  "canonical_solution_status" -> If[canonicalSolution === $Failed || canonicalSolution === {}, "not_solved_automatically", "solved"]
|>;

ExportJSON[FileNameJoin[{outDir, systemID <> "_template_summary.json"}], summary];

If[TrueQ[summary["passed_lure_form"]], Exit[0], Exit[1]];
