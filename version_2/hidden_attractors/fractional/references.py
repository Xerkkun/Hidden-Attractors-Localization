"""Primary references used by the fractional numerical core.

Stability: experimental

The registry is deliberately machine-readable so numerical results can record
the sources that define an operator or justify a discretization.  Presence in
this registry is not an endorsement: critical analyses are included alongside
the original definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class FractionalReference:
    """Bibliographic record attached to an operator or numerical method."""

    key: str
    citation: str
    year: int
    doi: str | None
    url: str
    role: str


_REFERENCES = {
    "caputo1967": FractionalReference(
        "caputo1967",
        "M. Caputo, Linear models of dissipation whose Q is almost frequency independent II",
        1967,
        "10.1111/j.1365-246X.1967.tb02303.x",
        "https://doi.org/10.1111/j.1365-246X.1967.tb02303.x",
        "operator_definition",
    ),
    "diethelm_ford_freed2004": FractionalReference(
        "diethelm_ford_freed2004",
        "K. Diethelm, N. J. Ford, A. D. Freed, Detailed error analysis for a fractional Adams method",
        2004,
        "10.1023/B:NUMA.0000027736.85078.be",
        "https://doi.org/10.1023/B:NUMA.0000027736.85078.be",
        "method_and_error_analysis",
    ),
    "li_tao2009": FractionalReference(
        "li_tao2009",
        "C. Li, C. Tao, On the fractional Adams method",
        2009,
        "10.1016/j.camwa.2009.07.050",
        "https://doi.org/10.1016/j.camwa.2009.07.050",
        "error_analysis",
    ),
    "diethelm_ford2009": FractionalReference(
        "diethelm_ford2009",
        "K. Diethelm, N. J. Ford, Numerical analysis for distributed-order differential equations",
        2009,
        "10.1016/j.cam.2008.07.018",
        "https://doi.org/10.1016/j.cam.2008.07.018",
        "distributed_order_numerical_analysis",
    ),
    "hu_liu_anh_turner2014": FractionalReference(
        "hu_liu_anh_turner2014",
        "Z. Hu, F. Liu, V. Anh, I. Turner, Numerical methods for the time distributed-order superdiffusion equation",
        2014,
        "10.21914/ANZIAMJ.V55I0.7888",
        "https://doi.org/10.21914/ANZIAMJ.V55I0.7888",
        "distributed_order_l1_and_order_quadrature",
    ),
    "lin_xu2007": FractionalReference(
        "lin_xu2007",
        "Y. Lin, C. Xu, Finite difference/spectral approximations for the time-fractional diffusion equation",
        2007,
        "10.1016/j.jcp.2007.02.001",
        "https://doi.org/10.1016/j.jcp.2007.02.001",
        "caputo_l1_error_analysis",
    ),
    "yin_liu_li_zhang2021": FractionalReference(
        "yin_liu_li_zhang2021",
        "B. Yin, Y. Liu, H. Li, Z. Zhang, Approximation methods for the distributed order calculus using the convolution quadrature",
        2021,
        "10.3934/dcdsb.2020168",
        "https://doi.org/10.3934/dcdsb.2020168",
        "distributed_order_convolution_quadrature",
    ),
    "caputo_distributed_order2001": FractionalReference(
        "caputo_distributed_order2001",
        "M. Caputo, Distributed order differential equations modelling dielectric induction and diffusion",
        2001,
        None,
        "https://www.math.bas.bg/complan/fcaa/volume4/index.html",
        "distributed_order_operator_definition",
    ),
    "ghoreishi_ghaffari_saad2023": FractionalReference(
        "ghoreishi_ghaffari_saad2023",
        "F. Ghoreishi, R. Ghaffari, N. Saad, Fractional Order Runge-Kutta Methods",
        2023,
        "10.3390/fractalfract7030245",
        "https://doi.org/10.3390/fractalfract7030245",
        "efork_method_and_convergence",
    ),
    "lubich1986": FractionalReference(
        "lubich1986",
        "C. Lubich, Discretized fractional calculus",
        1986,
        "10.1137/0517050",
        "https://doi.org/10.1137/0517050",
        "convolution_quadrature_foundation",
    ),
    "lubich2004": FractionalReference(
        "lubich2004",
        "C. Lubich, Convolution quadrature revisited",
        2004,
        "10.1023/B:BITN.0000046813.23911.2D",
        "https://doi.org/10.1023/B:BITN.0000046813.23911.2D",
        "convolution_quadrature_analysis",
    ),
    "chen_deng2015": FractionalReference(
        "chen_deng2015",
        "M. Chen, W. Deng, Discretized fractional substantial calculus",
        2015,
        "10.1051/m2an/2014037",
        "https://doi.org/10.1051/m2an/2014037",
        "exponentially_shifted_fractional_linear_multistep_weights",
    ),
    "guo_zeng_turner_burrage_karniadakis2019": FractionalReference(
        "guo_zeng_turner_burrage_karniadakis2019",
        "L. Guo, F. Zeng, I. Turner, K. Burrage, G. E. Karniadakis, Efficient multistep methods for tempered fractional calculus",
        2019,
        "10.1137/18M1230153",
        "https://doi.org/10.1137/18M1230153",
        "tempered_fractional_linear_multistep_and_fast_convolution",
    ),
    "jin_li_zhou2017": FractionalReference(
        "jin_li_zhou2017",
        "B. Jin, B. Li, Z. Zhou, Correction of high-order BDF convolution quadrature for fractional evolution equations",
        2017,
        "10.1137/17M1118816",
        "https://doi.org/10.1137/17M1118816",
        "starting_corrections",
    ),
    "jarad_abdeljawad_baleanu2012": FractionalReference(
        "jarad_abdeljawad_baleanu2012",
        "F. Jarad, T. Abdeljawad, D. Baleanu, Caputo-type modification of the Hadamard fractional derivatives",
        2012,
        "10.1186/1687-1847-2012-142",
        "https://doi.org/10.1186/1687-1847-2012-142",
        "caputo_hadamard_operator_definition",
    ),
    "yin_zhang_liu_li2024": FractionalReference(
        "yin_zhang_liu_li2024",
        "B. Yin, G. Zhang, Y. Liu, H. Li, Convolution quadrature for Hadamard fractional calculus",
        2024,
        "10.1016/j.cnsns.2024.108221",
        "https://doi.org/10.1016/j.cnsns.2024.108221",
        "hadamard_convolution_quadrature",
    ),
    "zheng2021_caputo_hadamard_transform": FractionalReference(
        "zheng2021_caputo_hadamard_transform",
        "X. Zheng, Logarithmic transformation between Caputo and Caputo-Hadamard fractional problems",
        2021,
        "10.1016/j.aml.2021.107366",
        "https://doi.org/10.1016/j.aml.2021.107366",
        "caputo_hadamard_logarithmic_transformation",
    ),
    "green_liu_yan2021": FractionalReference(
        "green_liu_yan2021",
        "C. W. H. Green, Y. Liu, Y. Yan, Numerical methods for Caputo-Hadamard FDEs with graded meshes",
        2021,
        "10.3390/math9212728",
        "https://doi.org/10.3390/math9212728",
        "caputo_hadamard_predictor_corrector",
    ),
    "green_yan2022": FractionalReference(
        "green_yan2022",
        "C. W. H. Green, Y. Yan, Detailed error analysis for a fractional Adams method on Caputo-Hadamard fractional differential equations",
        2022,
        "10.3390/foundations2040057",
        "https://doi.org/10.3390/foundations2040057",
        "caputo_hadamard_adams_error_analysis",
    ),
    "matusiak2020": FractionalReference(
        "matusiak2020",
        "M. Matusiak, Fast evaluation of Grunwald-Letnikov variable fractional-order differentiation and integration based on the FFT convolution",
        2020,
        "10.1007/978-3-030-50936-1_74",
        "https://doi.org/10.1007/978-3-030-50936-1_74",
        "fft_fractional_convolution",
    ),
    "podlubny1999": FractionalReference(
        "podlubny1999",
        "I. Podlubny, Fractional Differential Equations, Academic Press",
        1999,
        None,
        "https://shop.elsevier.com/books/fractional-differential-equations/podlubny/978-0-12-558840-9",
        "classical_reference",
    ),
    "caputo_fabrizio2015": FractionalReference(
        "caputo_fabrizio2015",
        "M. Caputo, M. Fabrizio, A new definition of fractional derivative without singular kernel",
        2015,
        "10.12785/pfda/010201",
        "https://doi.org/10.12785/pfda/010201",
        "operator_definition",
    ),
    "atangana_baleanu2016": FractionalReference(
        "atangana_baleanu2016",
        "A. Atangana, D. Baleanu, New fractional derivatives with nonlocal and non-singular kernel",
        2016,
        "10.2298/TSCI160111018A",
        "https://doi.org/10.2298/TSCI160111018A",
        "operator_definition",
    ),
    "yadav_pandey_shukla2019": FractionalReference(
        "yadav_pandey_shukla2019",
        "S. Yadav, R. K. Pandey, A. K. Shukla, Numerical approximations of Atangana-Baleanu Caputo derivative and its application",
        2019,
        "10.1016/j.chaos.2018.11.009",
        "https://doi.org/10.1016/j.chaos.2018.11.009",
        "abc_sampled_operator_numerical_method",
    ),
    "lee_kim_jang2024": FractionalReference(
        "lee_kim_jang2024",
        "S. Lee, H. Kim, B. Jang, A novel numerical method for solving nonlinear fractional-order differential equations and its applications",
        2024,
        "10.3390/fractalfract8010065",
        "https://doi.org/10.3390/fractalfract8010065",
        "abc_predictor_corrector_and_fast_history",
    ),
    "diethelm_garrappa_giusti_stynes2020": FractionalReference(
        "diethelm_garrappa_giusti_stynes2020",
        "K. Diethelm, R. Garrappa, A. Giusti, M. Stynes, Why fractional derivatives with nonsingular kernels should not be used",
        2020,
        "10.1515/fca-2020-0032",
        "https://doi.org/10.1515/fca-2020-0032",
        "critical_consistency_analysis",
    ),
    "cao_wang_xu2020": FractionalReference(
        "cao_wang_xu2020",
        "J. Cao, Z. Wang, C. Xu, A high-order scheme for fractional ODEs with the Caputo-Fabrizio derivative",
        2020,
        "10.1007/s42967-019-00043-8",
        "https://doi.org/10.1007/s42967-019-00043-8",
        "numerical_method",
    ),
    "liu_fan_yin_li2020": FractionalReference(
        "liu_fan_yin_li2020",
        "Y. Liu, E. Fan, B. Yin, H. Li, Fast algorithm based on the novel approximation formula for the Caputo-Fabrizio fractional derivative",
        2020,
        "10.3934/math.2020117",
        "https://doi.org/10.3934/math.2020117",
        "related_caputo_fabrizio_fast_method",
    ),
    "wang_huang2017": FractionalReference(
        "wang_huang2017",
        "K. Wang, J. Huang, High order fast algorithm for the Caputo fractional derivative",
        2017,
        None,
        "https://arxiv.org/abs/1705.06101",
        "fast_history_algorithm",
    ),
    "jiang_zhang_zhang_zhang2017": FractionalReference(
        "jiang_zhang_zhang_zhang2017",
        "S. Jiang, J. Zhang, Q. Zhang, Z. Zhang, Fast evaluation of the Caputo fractional derivative and its applications to fractional diffusion equations",
        2017,
        "10.4208/cicp.OA-2016-0136",
        "https://doi.org/10.4208/cicp.OA-2016-0136",
        "sum_of_exponentials_fast_history",
    ),
    "yan_pal_ford2014": FractionalReference(
        "yan_pal_ford2014",
        "Y. Yan, K. Pal, N. J. Ford, Higher order numerical methods for solving fractional differential equations",
        2014,
        "10.1007/s10543-013-0443-3",
        "https://doi.org/10.1007/s10543-013-0443-3",
        "higher_order_method",
    ),
    "bibi_rehman2024": FractionalReference(
        "bibi_rehman2024",
        "A. Bibi, M. ur Rehman, A numerical method for solutions of tempered fractional differential equations",
        2024,
        "10.1016/j.cam.2024.115772",
        "https://doi.org/10.1016/j.cam.2024.115772",
        "tempered_product_integration_method",
    ),
    "li_deng_zhao2019": FractionalReference(
        "li_deng_zhao2019",
        "C. Li, W. Deng, L. Zhao, Well-posedness and numerical algorithm for the tempered fractional differential equations",
        2019,
        "10.3934/dcdsb.2019026",
        "https://doi.org/10.3934/dcdsb.2019026",
        "tempered_caputo_well_posedness_and_predictor_corrector",
    ),
    "ahmed_izadi_cattani2025": FractionalReference(
        "ahmed_izadi_cattani2025",
        "H. M. Ahmed, M. Izadi, C. Cattani, A spectral approach to variable-order fractional differential equations",
        2025,
        "10.3390/math13162544",
        "https://doi.org/10.3390/math13162544",
        "variable_order_spectral_method",
    ),
    "samko_ross1993": FractionalReference(
        "samko_ross1993",
        "S. G. Samko, B. Ross, Integration and differentiation to a variable fractional order",
        1993,
        "10.1080/10652469308819027",
        "https://doi.org/10.1080/10652469308819027",
        "variable_order_foundation",
    ),
    "tavares_almeida_torres2016": FractionalReference(
        "tavares_almeida_torres2016",
        "D. Tavares, R. Almeida, D. F. M. Torres, Caputo derivatives of fractional variable order: numerical approximations",
        2016,
        "10.1016/j.cnsns.2015.10.027",
        "https://doi.org/10.1016/j.cnsns.2015.10.027",
        "variable_order_caputo_types_and_power_identities",
    ),
    "fang_sun_wang2020": FractionalReference(
        "fang_sun_wang2020",
        "Z. W. Fang, H. W. Sun, H. Wang, A fast method for variable-order Caputo fractional derivative with applications to time-fractional diffusion equations",
        2020,
        "10.1016/j.camwa.2020.07.009",
        "https://doi.org/10.1016/j.camwa.2020.07.009",
        "variable_order_l1_and_fast_history",
    ),
    "sabzikar_meerschaert_chen2015": FractionalReference(
        "sabzikar_meerschaert_chen2015",
        "F. Sabzikar, M. M. Meerschaert, J. Chen, Tempered fractional calculus",
        2015,
        "10.1016/j.jcp.2014.04.024",
        "https://doi.org/10.1016/j.jcp.2014.04.024",
        "tempered_operator_definition",
    ),
    "khalil2014": FractionalReference(
        "khalil2014",
        "R. Khalil et al., A new definition of fractional derivative",
        2014,
        "10.1016/j.cam.2014.01.002",
        "https://doi.org/10.1016/j.cam.2014.01.002",
        "conformable_definition",
    ),
}

FRACTIONAL_REFERENCES: Mapping[str, FractionalReference] = MappingProxyType(_REFERENCES)


def get_fractional_reference(key: str) -> FractionalReference:
    """Return a primary or critical reference by stable identifier."""

    try:
        return FRACTIONAL_REFERENCES[str(key).strip().lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown fractional reference: {key!r}") from exc


__all__ = ["FRACTIONAL_REFERENCES", "FractionalReference", "get_fractional_reference"]
