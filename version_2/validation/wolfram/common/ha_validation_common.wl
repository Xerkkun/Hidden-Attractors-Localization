(* ::Package:: *)

(* ============================================================= *)
(* Hidden-attractor validation utilities                         *)
(* Wolfram Language helpers for algebraic/numeric validation.     *)
(* ============================================================= *)

ClearAll[
  EnsureDirectory, GetCommandOption, CleanRat, CleanPoly, ExprString,
  BoolString, ToJSONReady, ExportJSON, RulePrecision, RealNumber,
  MakeTest, ExitFromTests, WriteCSV, PackageRootFromInput
];

EnsureDirectory[dir_String] := Module[{},
  If[! DirectoryQ[dir], CreateDirectory[dir, CreateIntermediateDirectories -> True]];
  dir
];

GetCommandOption[name_String, default_] := Module[{args, p, envName, envVal},
  envName = "WOLFRAM_" <> ToUpperCase[StringDelete[name, "-"]];
  envVal = Environment[envName];
  If[StringQ[envVal] && envVal =!= "",
    envVal,
    args = $ScriptCommandLine;
    p = FirstPosition[args, name, Missing["NotFound"]];
    If[MissingQ[p] || p[[1]] >= Length[args], default, args[[p[[1]] + 1]]]
  ]
];

PackageRootFromInput[] := Module[{file = $InputFileName, dir},
  dir = If[StringQ[file] && file =!= "", DirectoryName[file], Directory[]];
  If[FileNameTake[dir] === "cases" || FileNameTake[dir] === "template",
    ParentDirectory[dir],
    dir
  ]
];

CleanRat[expr_] := Factor[Together[expr]];
CleanPoly[expr_, vars_] := Collect[Expand[expr], vars, Factor];
ExprString[expr_] := ToString[InputForm[expr]];
BoolString[x_] := If[TrueQ[x], "true", "false"];

RealNumber[x_, prec_: 16] := Module[{y = N[x, prec]},
  If[NumericQ[y], N[Re[y], prec], y]
];

RulePrecision[rules_List, prec_Integer] := rules /. Rule[s_, v_?NumericQ] :> Rule[s, SetPrecision[v, prec]];

ToJSONReady[expr_] := Which[
  AssociationQ[expr], Map[ToJSONReady, expr],
  ListQ[expr], ToJSONReady /@ expr,
  MatrixQ[expr], ToJSONReady /@ expr,
  (* Handle Infinity and DirectedInfinity before general NumericQ check *)
  Head[expr] === DirectedInfinity || expr === Infinity || expr === -Infinity,
    ToString[expr],
  (* Complex numeric: export real/imag parts separately *)
  TrueQ[NumericQ[expr]] && ! TrueQ[Im[N[expr]] == 0],
    <|"re" -> N[Re[expr]], "im" -> N[Im[expr]]|>,
  (* Zero numeric check to avoid 0.e-XX invalid JSON representation *)
  TrueQ[NumericQ[expr]] && TrueQ[N[expr] == 0],
    0.0,
  (* Real numeric: export as floating point *)
  TrueQ[NumericQ[expr]], N[expr],
  Head[expr] === Missing, ToString[expr],
  (* Max, Min, and other unevaluated symbols: export as string *)
  Head[expr] === Max || Head[expr] === Min, ToString[N[expr]],
  True, expr
];

ExportJSON[path_String, data_] := Export[path, ToJSONReady[data], "JSON"];
WriteCSV[path_String, rows_List] := Export[path, rows, "CSV"];

MakeTest[name_String, passed_, details_: <||>] := <|
  "name" -> name,
  "passed" -> TrueQ[passed],
  "details" -> details
|>;

ExitFromTests[tests_List] := Module[{passed},
  passed = And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests);
  If[passed, Exit[0], Exit[1]]
];
