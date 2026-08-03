(* ::Package:: *)

(* ============================================================= *)
(* Independent correlation-sum and log-log fit validation        *)
(*                                                               *)
(* The input is a declared exact two-dimensional point set.       *)
(* Admissible pairs, Euclidean distances, the Theiler exclusion,  *)
(* strict distance counts, normalization, least-squares fit, and  *)
(* local slopes are all constructed here. No project source or    *)
(* generated report is read.                                     *)
(*                                                               *)
(* Evidence boundary: finite exact-set algebraic/numerical        *)
(* consistency only; no scaling-region validity, estimator        *)
(* consistency, fractal-dimension, chaos, attraction, or          *)
(* hiddenness claim.                                              *)
(*                                                               *)
(* Source anchor:                                                 *)
(* P. Grassberger and I. Procaccia, Phys. Rev. Lett. 50 (1983),   *)
(* DOI: 10.1103/PhysRevLett.50.346                               *)
(* ============================================================= *)

ClearAll["Global`*"];

validationRoot = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{validationRoot, "common", "ha_validation_common.wl"}]];

systemID = "correlation_dimension";
repositoryRoot = ParentDirectory[ParentDirectory[validationRoot]];
outDir = EnsureDirectory[
  GetCommandOption[
    "--out",
    FileNameJoin[{
      repositoryRoot, "validation", "outputs", "wolfram", systemID
    }]
  ]
];

source = <|
  "grassberger_procaccia_doi" -> "10.1103/PhysRevLett.50.346",
  "dataset_origin" -> "declared exact synthetic two-dimensional point set",
  "hafo_source_read" -> False,
  "report_input_used" -> False,
  "hafo_formula_imported" -> False
|>;

workingPrecision = 80;
pointsExact = {
  {0, 0},
  {1, 0},
  {0, 1},
  {1, 1},
  {2, 0},
  {0, 2}
};
theilerWindow = 1;
radiiExact = {1, 11/10, 3/2, 2, 21/10, 23/10};
fitRadiusRangeExact = {11/10, 21/10};

(* An unordered pair {i,j}, i<j, is admissible exactly when
   j-i>w. This is a positive Theiler exclusion and gives
   Binomial[n-w,2] admissible pairs. *)
allPairs = Subsets[Range[Length[pointsExact]], {2}];
admissiblePairs = Select[
  allPairs,
  Last[#] - First[#] > theilerWindow &
];
excludedPairs = Complement[allPairs, admissiblePairs];
pairDistancesExact = (
  Norm[pointsExact[[#[[1]]]] - pointsExact[[#[[2]]]]] & /@
    admissiblePairs
);
denominatorExact = Length[admissiblePairs];
denominatorFormulaExact = Binomial[
  Length[pointsExact] - theilerWindow,
  2
];

(* Strict inequality is intentional. In particular, the four
   admissible unit-distance pairs are not counted at r=1. *)
strictCountsExact = Table[
  Count[pairDistancesExact, distance_ /; TrueQ[distance < radius]],
  {radius, radiiExact}
];
correlationSumsExact = strictCountsExact/denominatorExact;
expectedCountsExact = {0, 4, 6, 6, 8, 10};

fitIndices = Select[
  Range[Length[radiiExact]],
  fitRadiusRangeExact[[1]] <= radiiExact[[#]] <=
      fitRadiusRangeExact[[2]] &&
    0 < correlationSumsExact[[#]] < 1 &
];
fitRadii = N[radiiExact[[fitIndices]], workingPrecision];
fitCorrelationSums = N[
  correlationSumsExact[[fitIndices]],
  workingPrecision
];
fitLogRadii = Log[fitRadii];
fitLogCorrelationSums = Log[fitCorrelationSums];
fitDesignMatrix = Transpose[{
  ConstantArray[1, Length[fitLogRadii]],
  fitLogRadii
}];
fitCoefficients = LeastSquares[
  fitDesignMatrix,
  fitLogCorrelationSums
];
fitIntercept = First[fitCoefficients];
fitSlope = Last[fitCoefficients];
fitPredictedLogCorrelationSums = fitDesignMatrix . fitCoefficients;
fitResiduals = fitLogCorrelationSums - fitPredictedLogCorrelationSums;
fitResidualSumSquares = Total[fitResiduals^2];
fitCenteredSumSquares = Total[
  (fitLogCorrelationSums - Mean[fitLogCorrelationSums])^2
];
fitRSquared = 1 - fitResidualSumSquares/fitCenteredSumSquares;
fitLogRadiusCenteredSumSquares = Total[
  (fitLogRadii - Mean[fitLogRadii])^2
];
fitRegressionStandardError = Sqrt[
  fitResidualSumSquares/
    ((Length[fitLogRadii] - 2) fitLogRadiusCenteredSumSquares)
];
localSlopeRadii = Sqrt[Most[fitRadii] Rest[fitRadii]];
localSlopes = Differences[fitLogCorrelationSums]/Differences[fitLogRadii];

tests = {
  MakeTest[
    "declared_set_is_exact_and_two_dimensional",
    MatrixQ[pointsExact, IntegerQ] && Dimensions[pointsExact] == {6, 2}
  ],
  MakeTest[
    "positive_theiler_window_is_applied",
    theilerWindow > 0 &&
      And @@ ((Last[#] - First[#] > theilerWindow) & /@ admissiblePairs)
  ],
  MakeTest[
    "admissible_pair_denominator_matches_closed_count",
    denominatorExact == denominatorFormulaExact && denominatorExact == 10
  ],
  MakeTest[
    "strict_norm_less_than_radius_counts_match",
    strictCountsExact == expectedCountsExact
  ],
  MakeTest[
    "pairs_on_radius_are_excluded",
    Count[pairDistancesExact, 1] == 4 && First[strictCountsExact] == 0 &&
      Count[pairDistancesExact, 2] == 2 && strictCountsExact[[4]] == 6
  ],
  MakeTest[
    "correlation_sum_is_count_over_admissible_denominator",
    correlationSumsExact == expectedCountsExact/10
  ],
  MakeTest[
    "fit_range_is_explicit_and_uses_only_open_unit_sums",
    fitRadiusRangeExact == {11/10, 21/10} &&
      Length[fitIndices] == 4 &&
      And @@ (0 < # < 1 & /@ fitCorrelationSums)
  ],
  MakeTest[
    "least_squares_fit_is_finite",
    VectorQ[fitCoefficients, NumericQ] &&
      And @@ (TrueQ[Abs[N[#]] < Infinity] & /@ fitCoefficients) &&
      TrueQ[N[fitSlope] > 0]
  ],
  MakeTest[
    "local_log_log_slopes_are_exportable",
    Length[localSlopes] == Length[fitRadii] - 1 &&
      VectorQ[localSlopes, NumericQ]
  ],
  MakeTest[
    "project_source_and_report_are_not_read",
    source["hafo_source_read"] === False &&
      source["report_input_used"] === False &&
      source["hafo_formula_imported"] === False
  ]
};

evidenceBoundary =
  "finite exact-set pair-count and log-log regression consistency only; " <>
  "no scaling-region validity, estimator consistency, fractal-dimension, " <>
  "chaos, attraction, or hiddenness claim";

summary = <|
  "system_id" -> systemID,
  "validation_scope" ->
    "independent_exact_pair_count_and_explicit_log_log_fit",
  "evidence_boundary" -> evidenceBoundary,
  "source" -> source,
  "parameters" -> <|
    "points" -> N[pointsExact, 18],
    "dimension" -> 2,
    "radii" -> N[radiiExact, 18],
    "theiler_window" -> theilerWindow,
    "metric" -> "euclidean",
    "pair_criterion" -> "unordered_i_less_than_j_and_j_minus_i_greater_than_w",
    "radius_criterion" -> "Norm[x_i-x_j] < r (strict)",
    "fit_radius_range" -> N[fitRadiusRangeExact, 18],
    "fit_range_inclusion" -> "closed_radius_interval_then_0<C<1"
  |>,
  "pair_geometry" -> <|
    "total_unordered_pairs" -> Length[allPairs],
    "excluded_pair_count" -> Length[excludedPairs],
    "admissible_pair_count" -> denominatorExact,
    "denominator" -> denominatorExact,
    "admissible_pairs_zero_based" -> (admissiblePairs - 1),
    "excluded_pairs_zero_based" -> (excludedPairs - 1),
    "admissible_pair_distances" -> N[pairDistancesExact, 24]
  |>,
  "correlation_curve" -> <|
    "radii" -> N[radiiExact, 24],
    "strict_pair_counts" -> strictCountsExact,
    "denominator" -> denominatorExact,
    "correlation_sums" -> N[correlationSumsExact, 24]
  |>,
  "fit" -> <|
    "method" -> "LeastSquares[{1,log(r)},log(C)]",
    "fit_radius_range" -> N[fitRadiusRangeExact, 24],
    "selected_indices_zero_based" -> (fitIndices - 1),
    "radii" -> N[fitRadii, 24],
    "correlation_sums" -> N[fitCorrelationSums, 24],
    "log_radii" -> N[fitLogRadii, 24],
    "log_correlation_sums" -> N[fitLogCorrelationSums, 24],
    "intercept" -> N[fitIntercept, 24],
    "slope" -> N[fitSlope, 24],
    "predicted_log_correlation_sums" ->
      N[fitPredictedLogCorrelationSums, 24],
    "residuals" -> N[fitResiduals, 24],
    "residual_sum_squares" -> N[fitResidualSumSquares, 24],
    "r_squared" -> N[fitRSquared, 24],
    "regression_standard_error" ->
      N[fitRegressionStandardError, 24],
    "local_slope_radii" -> N[localSlopeRadii, 24],
    "local_slopes" -> N[localSlopes, 24]
  |>,
  "tests" -> tests,
  "passed" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests)
|>;

ExportJSON[
  FileNameJoin[{outDir, systemID <> "_validation_summary.json"}],
  summary
];

ExitFromTests[tests];
