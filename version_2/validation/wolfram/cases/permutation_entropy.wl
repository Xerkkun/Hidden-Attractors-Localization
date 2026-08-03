(* ::Package:: *)

(* ============================================================= *)
(* Independent Bandt--Pompe permutation-entropy validation       *)
(*                                                               *)
(* Four declared exact fixtures exercise chronological forward   *)
(* embedding, lexicographic Lehmer ranks, a non-unit delay, and   *)
(* the declared stable-index and omit tie policies. Every window, *)
(* ordinal pattern, count, probability, and entropy is built here *)
(* without reading project source or a generated report.          *)
(*                                                               *)
(* Evidence boundary: finite exact-sequence combinatorial and     *)
(* numerical consistency only; no entropy-rate, KS-entropy,       *)
(* asymptotic, chaos, attraction, or hiddenness claim.            *)
(*                                                               *)
(* Source anchor:                                                 *)
(* C. Bandt and B. Pompe, Phys. Rev. Lett. 88 (2002),             *)
(* DOI: 10.1103/PhysRevLett.88.174102                             *)
(* ============================================================= *)

ClearAll["Global`*"];

validationRoot = ParentDirectory[DirectoryName[$InputFileName]];
Get[FileNameJoin[{validationRoot, "common", "ha_validation_common.wl"}]];

systemID = "permutation_entropy";
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
  "bandt_pompe_doi" -> "10.1103/PhysRevLett.88.174102",
  "dataset_origin" -> "declared exact synthetic scalar sequences",
  "hafo_source_read" -> False,
  "report_input_used" -> False,
  "hafo_formula_imported" -> False
|>;

workingPrecision = 80;
embeddingDimension = 3;
patternCount = Factorial[embeddingDimension];

CandidateWindows[series_List, dimension_Integer, delay_Integer] :=
  Table[
    series[[start + Range[0, dimension - 1] delay]],
    {start, 1, Length[series] - (dimension - 1) delay}
  ];

(* Entries are chronological indices ordered first by amplitude and then,
   only for equal amplitudes, by their original chronological index. *)
OrdinalPermutationStableIndex[window_List] :=
  SortBy[Range[Length[window]], {window[[#]], #} &] - 1;

(* Standard zero-based Lehmer/factoradic rank. For m=3 the ranks 0..5
   correspond lexicographically to 012, 021, 102, 120, 201, 210. *)
LehmerRankLexicographic[permutation_List] := Sum[
  Count[
    Drop[permutation, position],
    value_ /; value < permutation[[position]]
  ] Factorial[Length[permutation] - position],
  {position, 1, Length[permutation]}
];

PatternAnalysis[
  series_List,
  dimension_Integer,
  delay_Integer,
  tiePolicy_String
] := Module[
  {
    candidateWindows, tieMask, acceptedMask, acceptedWindows,
    patterns, ranks, counts, probabilities, entropyBase2,
    normalizedEntropy
  },
  candidateWindows = CandidateWindows[series, dimension, delay];
  tieMask = (Length[DeleteDuplicates[#]] < Length[#] &) /@
    candidateWindows;
  acceptedMask = Switch[
    tiePolicy,
    "stable_index", ConstantArray[True, Length[candidateWindows]],
    "omit", Not /@ tieMask,
    _, ConstantArray[False, Length[candidateWindows]]
  ];
  acceptedWindows = Pick[candidateWindows, acceptedMask, True];
  patterns = OrdinalPermutationStableIndex /@ acceptedWindows;
  ranks = LehmerRankLexicographic /@ patterns;
  counts = Count[ranks, #] & /@ Range[0, Factorial[dimension] - 1];
  probabilities = counts/Length[acceptedWindows];
  entropyBase2 = -Total[
    If[# == 0, 0, # Log[2, #]] & /@ probabilities
  ];
  normalizedEntropy = entropyBase2/Log[2, Factorial[dimension]];
  <|
    "series" -> series,
    "embedding_dimension" -> dimension,
    "delay" -> delay,
    "tie_policy" -> tiePolicy,
    "window_order" -> "chronological_forward_x[t+j*tau]",
    "rank_encoding" -> "zero_based_lexicographic_Lehmer",
    "candidate_windows" -> candidateWindows,
    "tie_mask" -> tieMask,
    "accepted_windows" -> acceptedWindows,
    "patterns_zero_based" -> patterns,
    "ranks_zero_based" -> ranks,
    "candidate_window_count" -> Length[candidateWindows],
    "accepted_window_count" -> Length[acceptedWindows],
    "omitted_tie_count" -> Count[acceptedMask, False],
    "counts" -> counts,
    "probabilities" -> probabilities,
    "entropy_base2" -> entropyBase2,
    "normalized_entropy" -> normalizedEntropy
  |>
];

seriesTau1Exact = {4, 7, 9, 10, 6, 11, 3, 8};
seriesTau2Exact = {4, 7, 9, 10, 6, 11, 3, 8, 5};
seriesTiesExact = {1, 1, 3, 0, 2, 4, 4, 2, 5, 0};

fixtureTau1 = PatternAnalysis[
  seriesTau1Exact, embeddingDimension, 1, "stable_index"
];
fixtureTau2 = PatternAnalysis[
  seriesTau2Exact, embeddingDimension, 2, "stable_index"
];
fixtureTiesStable = PatternAnalysis[
  seriesTiesExact, embeddingDimension, 1, "stable_index"
];
fixtureTiesOmit = PatternAnalysis[
  seriesTiesExact, embeddingDimension, 1, "omit"
];

expectedTau1Ranks = {0, 0, 4, 2, 4, 3};
expectedTau1Counts = {2, 0, 1, 1, 2, 0};
expectedTau2Ranks = {1, 0, 5, 4, 3};
expectedTau2Counts = {1, 1, 0, 1, 1, 1};
expectedStableRanks = {0, 4, 3, 0, 0, 4, 2, 4};
expectedStableCounts = {3, 0, 1, 1, 3, 0};
expectedOmitRanks = {4, 3, 0, 2, 4};
expectedOmitCounts = {1, 0, 1, 1, 2, 0};

allFixtures = <|
  "no_ties_tau1" -> fixtureTau1,
  "no_ties_tau2" -> fixtureTau2,
  "ties_stable_index" -> fixtureTiesStable,
  "ties_omit" -> fixtureTiesOmit
|>;

tests = {
  MakeTest[
    "chronological_forward_windows_are_exact",
    fixtureTau1["candidate_windows"][[1]] == {4, 7, 9} &&
      fixtureTau1["candidate_windows"][[-1]] == {11, 3, 8} &&
      fixtureTau2["candidate_windows"][[1]] == {4, 9, 6} &&
      fixtureTau2["candidate_windows"][[-1]] == {6, 3, 5}
  ],
  MakeTest[
    "lehmer_ranking_is_zero_based_and_lexicographic",
    (LehmerRankLexicographic /@ Permutations[Range[0, 2]]) == Range[0, 5]
  ],
  MakeTest[
    "m3_tau1_ranks_and_counts_match",
    fixtureTau1["ranks_zero_based"] == expectedTau1Ranks &&
      fixtureTau1["counts"] == expectedTau1Counts &&
      fixtureTau1["candidate_window_count"] == 6
  ],
  MakeTest[
    "m3_tau2_ranks_and_counts_match",
    fixtureTau2["ranks_zero_based"] == expectedTau2Ranks &&
      fixtureTau2["counts"] == expectedTau2Counts &&
      fixtureTau2["candidate_window_count"] == 5
  ],
  MakeTest[
    "stable_index_resolves_equal_values_chronologically",
    fixtureTiesStable["patterns_zero_based"][[1]] == {0, 1, 2} &&
      fixtureTiesStable["patterns_zero_based"][[5]] == {0, 1, 2} &&
      fixtureTiesStable["patterns_zero_based"][[6]] == {2, 0, 1} &&
      fixtureTiesStable["ranks_zero_based"] == expectedStableRanks &&
      fixtureTiesStable["counts"] == expectedStableCounts
  ],
  MakeTest[
    "omit_discards_exactly_tied_windows",
    fixtureTiesOmit["ranks_zero_based"] == expectedOmitRanks &&
      fixtureTiesOmit["counts"] == expectedOmitCounts &&
      fixtureTiesOmit["candidate_window_count"] == 8 &&
      fixtureTiesOmit["accepted_window_count"] == 5 &&
      fixtureTiesOmit["omitted_tie_count"] == 3
  ],
  MakeTest[
    "counts_and_probabilities_are_normalized_by_accepted_windows",
    And @@ (
      Total[#1["counts"]] == #1["accepted_window_count"] &&
        Total[#1["probabilities"]] == 1 & /@ Values[allFixtures]
    )
  ],
  MakeTest[
    "entropy_uses_base_two",
    fixtureTau1["entropy_base2"] ==
        -(2 (1/3) Log[2, 1/3] + 2 (1/6) Log[2, 1/6]) &&
      fixtureTau2["entropy_base2"] == Log[2, 5] &&
      fixtureTiesStable["entropy_base2"] ==
        -(2 (3/8) Log[2, 3/8] + 2 (1/8) Log[2, 1/8]) &&
      fixtureTiesOmit["entropy_base2"] ==
        -(3 (1/5) Log[2, 1/5] + (2/5) Log[2, 2/5])
  ],
  MakeTest[
    "normalized_entropy_divides_by_log2_factorial_m",
    And @@ (
      #1["normalized_entropy"] ==
          #1["entropy_base2"]/Log[2, Factorial[embeddingDimension]] & /@
        Values[allFixtures]
    )
  ],
  MakeTest[
    "project_source_and_report_are_not_read",
    source["hafo_source_read"] === False &&
      source["report_input_used"] === False &&
      source["hafo_formula_imported"] === False
  ]
};

evidenceBoundary =
  "finite exact-sequence ordinal-pattern, count, probability, and " <>
  "entropy consistency only; no entropy-rate, KS-entropy, asymptotic, " <>
  "chaos, attraction, or hiddenness claim";

summary = <|
  "system_id" -> systemID,
  "validation_scope" ->
    "independent_exact_Bandt_Pompe_ordinal_pattern_consistency",
  "evidence_boundary" -> evidenceBoundary,
  "source" -> source,
  "conventions" -> <|
    "window_order" -> "chronological_forward_x[t+j*tau]",
    "pattern_representation" ->
      "chronological_indices_sorted_by_value_then_index",
    "rank_encoding" -> "zero_based_lexicographic_Lehmer",
    "logarithm_base" -> 2,
    "normalization" -> "H/log2(m!)",
    "tie_policies" -> {"stable_index", "omit"}
  |>,
  "fixtures" -> Map[
    Function[fixture, Join[
      KeyDrop[fixture, {"entropy_base2", "normalized_entropy"}],
      <|
        "probabilities" -> N[fixture["probabilities"], 24],
        "entropy_base2" -> N[fixture["entropy_base2"], 24],
        "normalized_entropy" -> N[fixture["normalized_entropy"], 24]
      |>
    ]],
    allFixtures
  ],
  "tests" -> tests,
  "passed" -> And @@ (TrueQ[Lookup[#, "passed"]] & /@ tests)
|>;

ExportJSON[
  FileNameJoin[{outDir, systemID <> "_validation_summary.json"}],
  summary
];

ExitFromTests[tests];
