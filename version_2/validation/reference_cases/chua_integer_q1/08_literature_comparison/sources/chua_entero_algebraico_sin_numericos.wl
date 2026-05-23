(* ::Package:: *)

(* ============================================================= *)
(* CHUA ENTERO: PROCEDIMIENTO ALGEBRAICO LIMPIO                  *)
(* No realiza calculos numericos.                                *)
(* No usa FindRoot, NSolve, N, WorkingPrecision ni valores        *)
(* numericos de parametros.                                      *)
(* ============================================================= *)

ClearAll["Global`*"];

(* ------------------------------------------------------------- *)
(* Utilidades de presentacion                                    *)
(* ------------------------------------------------------------- *)

printSection[t_] := Print["\n==================== " <> t <> " ===================="];

printExpr[t_, e_] := (
  Print["\n" <> t <> ":"];
  Print[TraditionalForm[e]];
);

cleanPoly[expr_, vars_] := Collect[Expand[expr], vars, Factor];

cleanRat[expr_] := Factor[Together[expr]];

(* ------------------------------------------------------------- *)
(* Hipotesis simbolicas                                          *)
(* ------------------------------------------------------------- *)

$Assumptions =
  \[Alpha] > 0 && \[Beta] > 0 && \[Gamma] > 0 &&
  w0 > 0 && d > 0 && a > 1 &&
  Element[{m0, m1, k, p, BB, b1, b2, h}, Reals];

(* BB representa algebraicamente a B. No se asigna BB = \[Alpha](1+m1+k)
   hasta el final. Esto evita Solve::ivar. *)

Bdef = \[Alpha] (1 + m1 + k);

Np = Expand[(p + 1) (p + \[Gamma]) + \[Beta]];

P0B = {
   {-BB, \[Alpha], 0},
   {1, -1, 1},
   {0, -\[Beta], -\[Gamma]}
   };

qv = {-\[Alpha], 0, 0};
r = {1, 0, 0};

printSection["DEFINICIONES"];
printExpr["B", Bdef];
printExpr["N(p)", Np];
printExpr["P0 escrito con BB", P0B // MatrixForm];

(* ------------------------------------------------------------- *)
(* 1. Determinante de P0 - p I                                   *)
(* ------------------------------------------------------------- *)

printSection["DETERMINANTE Y ASIGNACION ESPECTRAL"];

charP0 = Expand[Det[P0B - p IdentityMatrix[3]]];

detByN = Expand[-(p + BB) Np + \[Alpha] (p + \[Gamma])];

detCheck = Expand[charP0 - detByN];

printExpr["det(P0 - p I) calculado por Det", cleanPoly[charP0, p]];
printExpr["Forma agrupada -(p+BB)N(p)+\[Alpha](p+\[Gamma])", detByN];
printExpr["Verificacion det - forma agrupada", detCheck];

targetChar = Expand[-(p + d) (p^2 + w0^2)];

polySpec = Expand[charP0 - targetChar];

printExpr[
  "det(P0-pI) - [-(p+d)(p^2+w0^2)]",
  cleanPoly[polySpec, p]
];

eqSpec = {
   Coefficient[polySpec, p, 2] == 0,
   Coefficient[polySpec, p, 1] == 0,
   Coefficient[polySpec, p, 0] == 0
   };

printExpr["Ecuaciones por coeficientes {p^2,p,p^0}", eqSpec];

solBD = First @ Solve[
    {
     Coefficient[polySpec, p, 2] == 0,
     Coefficient[polySpec, p, 1] == 0
     },
    {d, BB}
    ];

dFormula = cleanRat[d /. solBD];
BBFormula = cleanRat[BB /. solBD];

kFormula = cleanRat[BBFormula/\[Alpha] - m1 - 1];

residualFreq = cleanRat[
   Coefficient[polySpec, p, 0] /. solBD
   ];

freqNumerator = Factor[(1 + \[Gamma]) residualFreq];

printExpr["d", dFormula];
printExpr["BB", BBFormula];
printExpr["k, usando BB=\[Alpha](1+m1+k)", kFormula];
printExpr["Residual de frecuencia", residualFreq];
printExpr["Residual multiplicado por (1+\[Gamma])", freqNumerator];

(* ------------------------------------------------------------- *)
(* 2. Transferencia W0 y WA                                      *)
(* ------------------------------------------------------------- *)

printSection["TRANSFERENCIAS Y PARAMETROS h, b1, b2"];

denCan = (p + d) (p^2 + w0^2);

(* Convencion:
   W0(p)=r^T(P0-pI)^(-1)qv.
   Como det(P0-pI)=-(p+d)(p^2+w0^2),
   el numerador canonico queda \[Alpha] N(p). *)

numW0 = \[Alpha] Np;

Acan = {
   {0, -w0, 0},
   {w0, 0, 0},
   {0, 0, -d}
   };

bcan = {b1, b2, 1};
ccan = {1, 0, -h};

WA = Together[ccan . Inverse[Acan - p IdentityMatrix[3]] . bcan];

numWA = Expand[Numerator[Together[WA denCan]]];

numWAFactored = Expand[(-b1 p + b2 w0) (p + d) + h (p^2 + w0^2)];

checkNumWA = Expand[numWA - numWAFactored];

printExpr["A", Acan // MatrixForm];
printExpr["b", bcan];
printExpr["c", ccan];
printExpr["WA(p) generado desde c^T(A-pI)^(-1)b", WA];
printExpr["Numerador canonico de W0", numW0];
printExpr["Numerador de WA generado", cleanPoly[numWA, p]];
printExpr["Numerador de WA factorizado", numWAFactored];
printExpr["Verificacion numerador generado - factorizado", checkNumWA];

polyHB = Expand[numWAFactored - numW0];

eqHB = Table[Coefficient[polyHB, p, i] == 0, {i, 0, 2}];

printExpr["Ecuaciones por coeficientes {p^0,p,p^2}", eqHB];

solHB = First @ Solve[eqHB, {h, b1, b2}];

hFormula = cleanRat[h /. solHB];
b1Formula = cleanRat[b1 /. solHB];
b2Formula = cleanRat[b2 /. solHB];

printExpr["h", hFormula];
printExpr["b1", b1Formula];
printExpr["b2", b2Formula];

(* ------------------------------------------------------------- *)
(* 3. Matriz S                                                   *)
(* ------------------------------------------------------------- *)

printSection["MATRIZ S"];

(* Se genera por columnas desde P0 S = S A y r^T S = c^T.
   No se resuelve un sistema grande. *)

s21Formula = First[Solve[-BB + \[Alpha] s21 == 0, s21]][[1, 2]];

s22Formula = First[Solve[\[Alpha] s22 == -w0, s22]][[1, 2]];

s31Formula =
  First[Solve[1 - s21Formula + s31 == w0 s22Formula, s31]][[1, 2]];

s32Formula = cleanRat[(-\[Beta] s21Formula - \[Gamma] s31Formula)/w0];

s23Formula = First[Solve[BB h + \[Alpha] s23 == d h, s23]][[1, 2]];

s33Formula = First[Solve[-h - s23Formula + s33 == -d s23Formula, s33]][[1, 2]];

SBB = {
   {1, 0, -h},
   {s21Formula, s22Formula, s23Formula},
   {s31Formula, s32Formula, s33Formula}
   };

Sparams = SBB /. BB -> Bdef;

printExpr["S con BB", SBB // MatrixForm];
printExpr["S con BB=\[Alpha](1+m1+k)", Sparams // MatrixForm];

checkC = r . SBB - ccan;

printExpr["Verificacion r^T S - c^T", checkC];

checkSb = Expand[SBB . bcan - qv];

checkSbWithHB = cleanRat /@ (checkSb /. solHB);

printExpr["S b - qv", checkSb];
printExpr["S b - qv sustituyendo h,b1,b2", checkSbWithHB];

checkSimilarity = Together[P0B . SBB - SBB . Acan];

printExpr["P0 S - S A generado", checkSimilarity // MatrixForm];

(* ------------------------------------------------------------- *)
(* 4. Funcion descriptiva y Phi                                  *)
(* ------------------------------------------------------------- *)

printSection["FUNCION DESCRIPTIVA Y PHI"];

ClearAll[th0, R, ac, a0];

rad = Sqrt[1 - 1/a^2];

(* Primitivas manuales:
   int sin^2(theta) dtheta = theta/2 - sin(2 theta)/4
   int sin(theta) dtheta = -cos(theta) *)

I1manual = a (th0/2 - Sin[2 th0]/4);
I2manual = Cos[th0];

BpsiTh0 = Expand[(4/Pi) (m0 - m1) (I1manual + I2manual)];

geoRules = {
   Sin[2 th0] -> 2 rad/a,
   Cos[th0] -> rad
   };

BpsiGenerated = cleanPoly[BpsiTh0 /. geoRules, {th0, rad}];

PhiTh0 = cleanPoly[Pi (BpsiGenerated - k a), {th0, rad, k}];

PhiACsym = cleanPoly[PhiTh0 /. th0 -> Pi/2 - ac, {ac, rad, k}];

PhiDisplay = PhiACsym /. ac -> Inactive[ArcCos][1/a];

printExpr["Bpsi en funcion de th0", BpsiGenerated];
printExpr["Phi en funcion de th0", PhiTh0];
printExpr["Phi con ac=ArcCos(1/a)", PhiACsym];
printExpr["Phi para visualizar sin Log", PhiDisplay];

(* Derivada por regla de cadena:
   R = sqrt(1 - 1/a^2)
   d(th0)/da = -1/(a^2 R)
   dR/da = 1/(a^3 R)
*)

PhiStruct = PhiTh0 /. rad -> R;

dth0da = -1/(a^2 R);
dRda = 1/(a^3 R);

dPhiGenerated =
  Expand[D[PhiStruct, a] + D[PhiStruct, th0] dth0da + D[PhiStruct, R] dRda];

dPhiTh0 = 2 (m0 - m1) (th0 - R/a) - Pi k;

dPhiACsym = cleanPoly[dPhiTh0 /. th0 -> Pi/2 - ac /. R -> rad, {ac, rad, k}];

dPhiDisplay = dPhiACsym /. ac -> Inactive[ArcCos][1/a];

printExpr["dPhi generado antes de simplificar", dPhiGenerated];
printExpr["dPhi simplificado en funcion de th0", dPhiTh0];
printExpr["dPhi con ac=ArcCos(1/a)", dPhiACsym];
printExpr["dPhi para visualizar sin Log", dPhiDisplay];

a0Equation = (PhiACsym /. a -> a0) == 0;

Y0 = {a0, 0, 0};

X0Sym = Expand[Sparams . Y0];

printExpr["Ecuacion que define a0", a0Equation];
printExpr["Y(0)", Y0];
printExpr["X(0)=S Y(0)", X0Sym];

(* ------------------------------------------------------------- *)
(* 5. Resultados finales organizados                             *)
(* ------------------------------------------------------------- *)

printSection["RESULTADOS ALGEBRAICOS ORGANIZADOS"];

finalResults = {
   {"B", Bdef},
   {"N(p)", Np},
   {"det(P0-pI)", charP0},
   {"det agrupado", detByN},
   {"det espectral", targetChar},
   {"d", dFormula},
   {"BB", BBFormula},
   {"k", kFormula},
   {"residual frecuencia", freqNumerator},
   {"h", hFormula},
   {"b1", b1Formula},
   {"b2", b2Formula},
   {"S", Sparams},
   {"Phi(a)", PhiDisplay},
   {"dPhi/da", dPhiDisplay},
   {"a0", a0Equation},
   {"Y(0)", Y0},
   {"X(0)", X0Sym}
   };

Grid[
  Prepend[
   ({#[[1]], TraditionalForm[#[[2]]]} & /@ finalResults),
   {"Cantidad", "Expresion"}
   ],
  Frame -> All,
  Alignment -> Left
  ]
