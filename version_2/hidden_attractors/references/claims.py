"""Taxonomy of strong library claims and required academic references."""

from enum import Enum

class ClaimType(str, Enum):
    HIDDEN_ATTRACTOR_DEFINITION = "HIDDEN_ATTRACTOR_DEFINITION"
    SELF_EXCITED_DEFINITION = "SELF_EXCITED_DEFINITION"
    DESCRIBING_FUNCTION_CHUA_LOCALIZATION = "DESCRIBING_FUNCTION_CHUA_LOCALIZATION"
    DESCRIBING_FUNCTION_IS_HEURISTIC = "DESCRIBING_FUNCTION_IS_HEURISTIC"
    CAPUTO_ABM_INTEGRATION = "CAPUTO_ABM_INTEGRATION"
    FRACTIONAL_MATIGNON_STABILITY = "FRACTIONAL_MATIGNON_STABILITY"
    MACHADO_FDF = "MACHADO_FDF"
    ALTERNATIVE_LOCALIZATION_METHODS = "ALTERNATIVE_LOCALIZATION_METHODS"
    FRACTIONAL_CHUA_ARCTAN_WU2023 = "FRACTIONAL_CHUA_ARCTAN_WU2023"
    WEYL_CAPUTO_BRIDGE = "WEYL_CAPUTO_BRIDGE"
    NONSMOOTH_CHUA_LIPSCHITZ_ABM = "NONSMOOTH_CHUA_LIPSCHITZ_ABM"
    NUMERICAL_CONTINUATION_LOCALIZATION = "NUMERICAL_CONTINUATION_LOCALIZATION"
    HIDDENNESS_OPERATIONAL_VERIFICATION = "HIDDENNESS_OPERATIONAL_VERIFICATION"

CLAIM_REFERENCE_MATRIX = {
    ClaimType.HIDDEN_ATTRACTOR_DEFINITION: [
        "leonov_kuznetsov_hidden_definition",
        "kuznetsov_2017_chua_df"
    ],
    ClaimType.SELF_EXCITED_DEFINITION: [
        "leonov_kuznetsov_hidden_definition"
    ],
    ClaimType.DESCRIBING_FUNCTION_CHUA_LOCALIZATION: [
        "kuznetsov_2017_chua_df"
    ],
    ClaimType.DESCRIBING_FUNCTION_IS_HEURISTIC: [
        "kuznetsov_2017_chua_df"
    ],
    ClaimType.CAPUTO_ABM_INTEGRATION: [
        "diethelm_ford_freed_abm_caputo",
        "danca_2017_fractional_hidden"
    ],
    ClaimType.FRACTIONAL_MATIGNON_STABILITY: [
        "matignon_fractional_stability",
        "danca_2017_fractional_hidden"
    ],
    ClaimType.MACHADO_FDF: [
        "machado_2015_fractional_describing_functions"
    ],
    ClaimType.ALTERNATIVE_LOCALIZATION_METHODS: [
        "guan_xie_2025_review"
    ],
    ClaimType.FRACTIONAL_CHUA_ARCTAN_WU2023: [
        "wu_2023_fractional_chua_arctan"
    ],
    ClaimType.WEYL_CAPUTO_BRIDGE: [
        "danca_2017_fractional_hidden"
    ],
    ClaimType.NONSMOOTH_CHUA_LIPSCHITZ_ABM: [
        "danca_2017_fractional_hidden"
    ],
    ClaimType.NUMERICAL_CONTINUATION_LOCALIZATION: [
        "guan_xie_2025_review"
    ],
    ClaimType.HIDDENNESS_OPERATIONAL_VERIFICATION: [
        "danca_2017_fractional_hidden"
    ]
}
