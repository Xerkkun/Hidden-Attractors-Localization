(* ::Package:: *)

(* ============================================================= *)
(* Integer MAVPD: source-to-seed algebraic validation             *)
(*                                                               *)
(* Primary source: Matouk et al. (2023), Mathematics 11, 591.    *)
(* DOI: 10.3390/math11030591                                      *)
(*                                                               *)
(* This script does not read the comparative report or Python     *)
(* outputs.  It starts from the published vector field and derives *)
(* the Lur'e split, Jacobian, equilibria, transfer function,       *)
(* frequency polynomial, gains, amplitudes, harmonic vectors, and *)
(* phase-dependent seeds.                                         *)
(* ============================================================= *)

ClearAll["Global`*"];

root = ParentDirectory[DirectoryName[$InputFileName]];
commonFile = FileNameJoin[{root, "common", "ha_validation_common.wl"}];
Get[commonFile];

(* The equations below are embedded from the cited source.  The only files
   read by this case are this script and the generic validation helper, both
   inside validation/wolfram.  This computed boundary makes the final
   report-independence test non-tautological. *)
auditedInputFiles = DeleteDuplicates[ExpandFileName /@ {$InputFileName, commonFile}];
auditRootPrefix = ToLowerCase[ExpandFileName[root] <> $PathnameSeparator];
independentInputBoundarySatisfied = AllTrue[
  auditedInputFiles,
  StringStartsQ[ToLowerCase[#], auditRootPrefix] &
];

systemID = "mavpd_integer";
outDir = EnsureDirectory[
  GetCommandOption["--out", FileNameJoin[{Directory[], "validation_outputs", systemID}]]
];

(* Step 0. Published model/base cases; posterior candidate inputs are kept
   separate from the source parameters and from the algebraic derivation. *)
source = <|
  "doi" -> "10.3390/math11030591",
  "title" -> "Existence of Self-Excited and Hidden Attractors in the Modified Autonomous Van Der Pol-Duffing Systems",
  "model_input" -> "published vector field; no report equation is read",
  "candidate_parameter_status" -> "declared local xi endpoint plus numerically selected Hopf-relative gamma; not a published parameter tuple",
  "published_equation" -> "y1'=delta gamma y1+delta y2-delta y1^3; y2'=y1-xi y2-y3; y3'=rho y2"
|>;

X = {y1, y2, y3};
Fsource = {
  delta gamma y1 + delta y2 - delta y1^3,
  y1 - xi y2 - y3,
  rho y2
};
parameterRules = {gamma -> 1/10, delta -> 100, rho -> 200};
xiCases = {31/10, 7/2};

$Assumptions = delta > 0 && rho > 0 && gamma > 0 && xi > 0 &&
  omega > 0 && amp > 0 && Element[{delta, rho, gamma, xi, omega, amp}, Reals];

(* Step 1. Extract the scalar nonlinearity and derive P,b,c from Fsource. *)
sigma = y1;
cvec = Table[D[sigma, X[[j]]], {j, Length[X]}];
Feta = Fsource /. y1^3 -> eta;
bvec = D[Feta, eta];
linearField = Feta /. eta -> 0;
P = Table[D[linearField[[i]], X[[j]]], {i, Length[X]}, {j, Length[X]}];
psi[s_] := s^3;
lureField = P . X + bvec psi[cvec . X];
lureResidual = FullSimplify[Fsource - lureField];

(* Step 2. Derive equilibria and the analytic Jacobian from Fsource. *)
equilibriumSolutions = FullSimplify[
  Solve[Thread[Fsource == 0], X, Reals],
  Assumptions -> delta > 0 && rho > 0 && gamma > 0
];
Jsource = Table[D[Fsource[[i]], X[[j]]], {i, 3}, {j, 3}];
jacobianFromLure = P + Outer[Times, bvec, cvec] 3 (cvec . X)^2;
jacobianResidual = FullSimplify[Jsource - jacobianFromLure];

(* Step 2b. Derive the E+/- characteristic polynomial and Hopf boundary. *)
Jnonzero = FullSimplify[Jsource /. {y1 -> Sqrt[gamma], y2 -> 0, y3 -> Sqrt[gamma]}];
(* Build the monic engineering convention p(lambda)=det(lambda I-J)
   explicitly.  This avoids relying on the sign convention used by
   CharacteristicPolynomial for an odd-dimensional matrix. *)
nonzeroCharacteristic = Collect[
  Expand[Det[lambda IdentityMatrix[3] - Jnonzero]],
  lambda
];
expectedNonzeroCharacteristic =
  lambda^3 + (xi + 2 delta gamma) lambda^2 +
  (rho - delta + 2 delta gamma xi) lambda + 2 delta gamma rho;
nonzeroCharacteristicResidual = FullSimplify[
  nonzeroCharacteristic - expectedNonzeroCharacteristic
];
a1 = xi + 2 delta gamma;
a2 = rho - delta + 2 delta gamma xi;
a3 = 2 delta gamma rho;
routhHopfEquality = Expand[a1 a2 - a3];
routhPolynomialG = Factor[routhHopfEquality /. gamma -> g/(2 delta)];
expectedRouthPolynomialG = xi g^2 + (xi^2 - delta) g + xi (rho - delta);
routhPolynomialResidual = FullSimplify[routhPolynomialG - expectedRouthPolynomialG];
routhFactorizationResidual = FullSimplify[
  nonzeroCharacteristic - (lambda + a1) (lambda^2 + a2)
];
expectedRouthFactorizationResidual = a3 - a1 a2;
candidateXi = 57/20;
hopfGammaRoots = Sort[Select[
  N[gamma /. Solve[
    (routhHopfEquality /. {delta -> 100, rho -> 200, xi -> candidateXi}) == 0,
    gamma,
    Reals
  ], 50],
  TrueQ[# > 0] &
]];
highHopfGamma = Last[hopfGammaRoots];
highHopfRules = {delta -> 100, rho -> 200, xi -> candidateXi, gamma -> highHopfGamma};
highHopfOmega = N[Sqrt[a2 /. highHopfRules], 30];
highHopfCharacteristicResidual = N[
  Abs[nonzeroCharacteristic /. highHopfRules /. lambda -> I highHopfOmega],
  30
];
highHopfRouthCoefficientsPositive = And @@ Thread[
  N[{a1, a2, a3} /. highHopfRules, 30] > 0
];
candidateOffset = 1/100;
candidateGamma = highHopfGamma + candidateOffset;
candidateRules = {delta -> 100, rho -> 200, xi -> candidateXi, gamma -> candidateGamma};
candidateEigenvalues = N[Eigenvalues[Jnonzero /. candidateRules], 30];

(* Step 3. Derive W(s)=c^T(sI-P)^(-1)b; no formula is inserted.
   The Python/report convention is G(s)=c^T(P-sI)^(-1)b=-W(s), so the
   equivalent gain equations are k=1/W(i omega)=-1/G(i omega). *)
Wstandard = Factor[cvec . Inverse[s IdentityMatrix[3] - P] . bvec];
Wcode = Factor[-Wstandard];
transferNumerator = Factor[Numerator[Together[Wstandard]]];
transferDenominator = Factor[Denominator[Together[Wstandard]]];

Wiw = Together[ComplexExpand[Wstandard /. s -> I omega]];
Wreal = Factor[ComplexExpand[Re[Wiw]]];
Wimag = Factor[ComplexExpand[Im[Wiw]]];
imagNumerator = Factor[Numerator[Together[Wimag]]];
imagDenominator = Factor[Denominator[Together[Wimag]]];
frequencyPolynomialZ = Factor[
  imagNumerator/(delta omega) /. omega^2 -> z /. omega^4 -> z^2
];
expectedFrequencyPolynomialZ = z^2 + (delta - 2 rho + xi^2) z + rho (rho - delta);
frequencyPolynomialResidual = FullSimplify[
  frequencyPolynomialZ - expectedFrequencyPolynomialZ
];

(* Step 4. Derive the cubic describing function from its defining integral. *)
describingFunction = FullSimplify[
  (2/(Pi amp)) Integrate[psi[amp Cos[theta]] Cos[theta], {theta, 0, Pi}]
];

(* Step 5. Derive each direct branch without a frequency grid. *)
deriveBranches[xiValue_] := Module[
  {rules, polynomial, zRoots, positiveZRoots},
  rules = Join[parameterRules, {xi -> xiValue}];
  polynomial = frequencyPolynomialZ /. rules;
  zRoots = z /. Solve[polynomial == 0, z, Reals];
  positiveZRoots = Sort[Select[N[zRoots, 50], TrueQ[# > 0] &]];
  Table[
    Module[
      {wval, wstd, kval, aval, P0, harmonicVector, vectorResidual,
       phase, seed, closureResidual, dfResidual},
      wval = Sqrt[positiveZRoots[[branch]]];
      wstd = N[Wstandard /. rules /. s -> I wval, 40];
      kval = N[Re[1/wstd], 40];
      aval = N[Sqrt[4 kval/3], 40];
      P0 = P + kval Outer[Times, bvec, cvec];

      (* c^T v=1 fixes scale; the formula follows from the first and third
         rows of (P0-i omega I)v=0.  The second row is checked below. *)
      harmonicVector = {
        1,
        kval - gamma + I wval/delta,
        rho/delta - I rho (kval - gamma)/wval
      } /. rules;
      vectorResidual = N[
        Norm[(P0 /. rules) . harmonicVector - I wval harmonicVector], 30
      ];
      closureResidual = N[Abs[1 - kval wstd], 30];
      dfResidual = N[Abs[(describingFunction /. amp -> aval) - kval], 30];

      Table[
        phase = phaseValue;
        seed = N[aval Re[harmonicVector Exp[I phase]], 24];
        <|
          "status" -> "ok",
          "xi" -> N[xiValue, 16],
          "branch" -> branch - 1,
          "phase" -> N[phase, 16],
          "omega0" -> N[wval, 24],
          "k" -> N[kval, 24],
          "a0" -> N[aval, 24],
          "harmonic_vector" -> ({N[Re[#], 24], N[Im[#], 24]} & /@ harmonicVector),
          "seed" -> seed,
          "nyquist_closure_residual" -> closureResidual,
          "describing_function_residual" -> dfResidual,
          "eigenvector_equation_residual" -> vectorResidual,
          "frequency_grid_used" -> False,
          "published_table_used" -> False,
          "seed_construction" -> "a0 Re[v exp(i phase)], with (P0-i omega I)v=0 and c^T v=1"
        |>,
        {phaseValue, {0, Pi}}
      ]
    ],
    {branch, Length[positiveZRoots]}
  ]
];

seedRows = Flatten[deriveBranches /@ xiCases, 2];

(* Step 6. Numeric source-field, equilibrium, and Jacobian records. *)
equilibriumRows = Flatten[
  Table[
    Module[{rules, points},
      rules = Join[parameterRules, {xi -> xiValue}];
      points = {
        {0, 0, 0},
        {Sqrt[gamma], 0, Sqrt[gamma]},
        {-Sqrt[gamma], 0, -Sqrt[gamma]}
      } /. rules;
      MapIndexed[
        Join[
          {N[xiValue, 16], {"E0", "E+", "E-"}[[First[#2]]]},
          N[#1, 24],
          {N[Norm[Fsource /. rules /. Thread[X -> #1]], 24]}
        ] &,
        points
      ]
    ],
    {xiValue, xiCases}
  ],
  1
];

jacobianRows = Flatten[
  Table[
    Module[{rules, points},
      rules = Join[parameterRules, {xi -> xiValue}];
      points = {
        {0, 0, 0},
        {Sqrt[gamma], 0, Sqrt[gamma]},
        {-Sqrt[gamma], 0, -Sqrt[gamma]}
      } /. rules;
      MapIndexed[
        Join[
          {N[xiValue, 16], {"E0", "E+", "E-"}[[First[#2]]]},
          Flatten[N[Jsource /. rules /. Thread[X -> #1], 24]]
        ] &,
        points
      ]
    ],
    {xiValue, xiCases}
  ],
  1
];

WriteCSV[
  FileNameJoin[{outDir, systemID <> "_equilibria_residuals.csv"}],
  Prepend[equilibriumRows, {"xi", "equilibrium", "y1", "y2", "y3", "rhs_residual_norm"}]
];
WriteCSV[
  FileNameJoin[{outDir, systemID <> "_jacobians.csv"}],
  Prepend[jacobianRows, {"xi", "equilibrium", "j11", "j12", "j13", "j21", "j22", "j23", "j31", "j32", "j33"}]
];
ExportJSON[FileNameJoin[{outDir, systemID <> "_seed_data.json"}], seedRows];

numericRules = Join[parameterRules, {xi -> 31/10}];
symbolicSummary = <|
  "system_id" -> systemID,
  "source" -> source,
  "input_independence_audit" -> <|
    "audited_input_files" -> (FileNameTake /@ auditedInputFiles),
    "all_inputs_within_validation_wolfram" -> independentInputBoundarySatisfied,
    "source_equations_embedded_in_case" -> True
  |>,
  "derivation_steps" -> {
    "construct Fsource from the published differential equations",
    "replace y1^3 by an independent eta and differentiate to extract b",
    "differentiate the remaining linear field to obtain P and c=grad(y1)",
    "derive equilibria and Jacobian from Fsource",
    "derive det(lambda I-J) at E+/-, factor it on the Routh-Hurwitz equality, and derive the imaginary-pair frequency",
    "invert s I-P symbolically to derive W(s)",
    "factor Im W(i omega) and substitute z=omega^2",
    "derive N(a) from its Fourier integral",
    "solve the exact z polynomial, then k=1/W(i omega), a0=sqrt(4k/3)",
    "solve the normalized harmonic equation and construct each phase seed"
  },
  "lure_form" -> <|
    "P" -> ExprString[P], "b" -> ExprString[bvec], "r" -> ExprString[cvec],
    "psi" -> "sigma^3", "residual" -> ExprString[lureResidual]
  |>,
  "lure_form_numeric" -> <|
    "P" -> N[P /. numericRules, 24],
    "b" -> N[bvec /. numericRules, 24],
    "r" -> N[cvec, 24]
  |>,
  "equilibria_symbolic" -> ExprString[equilibriumSolutions],
  "jacobian" -> ExprString[Jsource],
  "nonzero_equilibrium_stability" -> <|
    "characteristic_polynomial" -> ExprString[nonzeroCharacteristic],
    "expected_characteristic_polynomial" -> ExprString[expectedNonzeroCharacteristic],
    "routh_hopf_equality" -> ExprString[routhHopfEquality],
    "routh_polynomial_g" -> ExprString[routhPolynomialG],
    "factorization_residual" -> ExprString[routhFactorizationResidual],
    "factorization_residual_expected" -> ExprString[expectedRouthFactorizationResidual],
    "factorization_at_equality" -> "p(lambda)=(lambda+a1)(lambda^2+a2) when a1 a2=a3",
    "substitution" -> "g=2 delta gamma",
    "xi_target" -> N[candidateXi, 16],
    "candidate_xi_source" -> "declared local continuation endpoint in reproducibility.yaml; supplied here only for a posteriori evaluation",
    "positive_gamma_boundaries" -> hopfGammaRoots,
    "selected_high_boundary" -> highHopfGamma,
    "selected_high_boundary_frequency" -> highHopfOmega,
    "selected_high_boundary_characteristic_residual" -> highHopfCharacteristicResidual,
    "selected_high_boundary_routh_coefficients_positive" -> highHopfRouthCoefficientsPositive,
    "candidate_gamma" -> candidateGamma,
    "candidate_offset" -> N[candidateOffset, 16],
    "candidate_offset_source" -> "numerical Hopf-offset screen in Python; supplied here only for a posteriori stability verification",
    "candidate_Eplus_Eminus_eigenvalues" -> ({N[Re[#], 18], N[Im[#], 18]} & /@ candidateEigenvalues),
    "boundary_derived_from_source_equations" -> True,
    "candidate_parameter_tuple_derived_algebraically" -> False
  |>,
  "transfer" -> <|
    "standard_definition" -> "W(s)=c^T (s I-P)^(-1) b",
    "code_definition" -> "G(s)=c^T (P-s I)^(-1) b=-W(s)",
    "gain_equivalence" -> "k=1/Re W(i omega)=-1/Re G(i omega)",
    "standard" -> ExprString[Wstandard],
    "code_convention" -> ExprString[Wcode],
    "numerator" -> ExprString[transferNumerator],
    "denominator" -> ExprString[transferDenominator],
    "real_iomega" -> ExprString[Wreal],
    "imag_iomega" -> ExprString[Wimag],
    "numeric_samples" -> Table[
      With[{value = N[Wstandard /. Join[parameterRules, {xi -> xiValue, s -> 7/10 + 13 I/10}], 24]},
        <|
          "xi" -> N[xiValue, 16],
          "s" -> {0.7, 1.3},
          "W_standard" -> {N[Re[value], 16], N[Im[value], 16]}
        |>
      ],
      {xiValue, xiCases}
    ]
  |>,
  "frequency_polynomial_z" -> ExprString[frequencyPolynomialZ],
  "describing_function" -> ExprString[describingFunction],
  "report_input_used" -> False
|>;
ExportJSON[FileNameJoin[{outDir, systemID <> "_symbolic_summary.json"}], symbolicSummary];

tests = {
  MakeTest["source_to_lure_exact", TrueQ[FullSimplify[lureResidual == {0, 0, 0}]]],
  MakeTest["jacobian_from_source_and_lure_match", TrueQ[FullSimplify[jacobianResidual == ConstantArray[0, {3, 3}]]]],
  MakeTest["nonzero_characteristic_polynomial_derived", TrueQ[nonzeroCharacteristicResidual == 0]],
  MakeTest["routh_hopf_polynomial_derived",
    TrueQ[routhPolynomialResidual == 0] &&
    TrueQ[FullSimplify[
      routhFactorizationResidual - expectedRouthFactorizationResidual
    ] == 0] &&
    TrueQ[highHopfRouthCoefficientsPositive] &&
    TrueQ[highHopfCharacteristicResidual < 10^-16]],
  MakeTest["numerically_selected_candidate_Eplus_Eminus_hurwitz", Max[Re[candidateEigenvalues]] < 0],
  MakeTest["frequency_polynomial_derived", TrueQ[FullSimplify[frequencyPolynomialResidual == 0]]],
  MakeTest["cubic_describing_function_derived", TrueQ[FullSimplify[describingFunction == 3 amp^2/4]]],
  MakeTest["two_branches_per_parameter_case", Length[seedRows] == 8],
  MakeTest["all_seed_residuals", AllTrue[seedRows,
    Lookup[#, "nyquist_closure_residual", 1] < 10^-16 &&
    Lookup[#, "describing_function_residual", 1] < 10^-16 &&
    Lookup[#, "eigenvector_equation_residual", 1] < 10^-16 &]],
  MakeTest["equilibrium_residuals", Max[equilibriumRows[[All, -1]]] < 10^-20],
  MakeTest[
    "report_not_used",
    TrueQ[independentInputBoundarySatisfied],
    <|"audited_input_files" -> (FileNameTake /@ auditedInputFiles)|>
  ]
};

ExportJSON[
  FileNameJoin[{outDir, systemID <> "_validation_summary.json"}],
  <|
    "system_id" -> systemID,
    "validation_scope" -> "source_equations_to_lure_seeds_and_hopf_boundary",
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
