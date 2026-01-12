""" MRDS Calibrating options on one maturity only.
"""

import datetime
import numpy as np
from typing import Tuple, List
from logging import getLogger
from scipy.optimize import minimize, Bounds

from mrds.correlations import corr_hyp_sec_mat
from mrds.mrds_maths import ComMathsMixin
from mrds.quartic.quartic_cy import QuadRoots, CubicRoots, QuarticRoots


_logger = getLogger(__name__)


class MrdsCalibMixin:
    """Calibration of one maturity only."""

    def __init__(self, mkt_date: datetime.date):

        self.mkt_date = mkt_date
        self.dcf = 252.0

    def __difference_to_market_date(self, fwd_date: datetime.date) -> float:
        """Computes the difference to market date.

        :param fwd_date: date to compute the distance to market date.
        """

        return (fwd_date - self.mkt_date).days / self.dcf

    def __V_one_factor(
        self,
        kappa: float,
        sigma: float,
        beta: float,
        # factor_nb: int,
        fwd_date: datetime.date,
        t_0: float,
        t_1: float,
    ) -> float:
        """Computes integrated volatility V only for one factor (factor_nb).

        :param factor_nb: factor to consider (0, 1,...)
        :param fwd_date: forward date
        :param t_0: integrated volatility start time (float)
        :param t_1: integrated vol end time (float)
        """

        # kappa = self._kappa_vec(asset)[factor_nb]
        # sigma = self._sigma_vec(asset)[factor_nb]
        # beta = self._beta_T(asset, [fwd_date])[0]  # number, not vector

        if kappa == 0.0:
            return beta**2 * sigma**2 * (t_1 - t_0)

        return (
            beta**2
            * sigma**2
            / (2.0 * kappa)
            * np.exp(-2.0 * kappa * self.__difference_to_market_date(fwd_date))
            * (np.exp(2.0 * kappa * t_1) - np.exp(2.0 * kappa * t_0))
        )

    def _V_cross_factor(
        self,
        # asset: str
        factor_1: int,
        factor_2: int,
        kv: Tuple[float, float],
        sv: Tuple[float, float],
        rho_12: float,
        beta_1: float,
        beta_2: float,
        fwd_date_1: datetime.date,
        fwd_date_2: datetime.date,
        t_0: float,
        t_1: float,
    ):
        """Computes cross integrated vol. V between factors factor_1 and factor_2.
             Integrate between t_0 and t_1 for forward dates fwd_date_1, fwd_date_2.

        :param kv: kappa vector, composed of kappa_1, and kappa_2
        :param sv: sigma vector, composed of sigma_1, sigma_2
        :param rho_12: correlation factor
        :param factor_1: first factor to consider (0 or 1)
        :param factor_2: second factor to consider (0 or 1)
        :param fwd_date_1: forward date_1
        :param fwd_date_2: forward date 2
        :param t_0: integrated volatility start time (float) t_0 < t_1
        :param t_1: integrated vol end time (float); t_1 > t_0
        """

        assert t_0 <= t_1, f"Integration times {t_0} and {t_1} are not ordered."

        # kv = self._kappa_vec(asset)
        # sv = self._sigma_vec(asset)

        kappa_1 = kv[factor_1]
        kappa_2 = kv[factor_2]
        kappa_12 = kappa_1 + kappa_2
        sigma_1 = sv[factor_1]
        sigma_2 = sv[factor_2]
        # rho_12 = self._factor_corr_mat(asset, asset)[factor_1, factor_2]
        # beta_1 = self._beta_T(asset, [fwd_date_1])[0]  # one forward date
        # beta_2 = self._beta_T(asset, [fwd_date_2])[0]

        if kappa_12 == 0.0:
            return rho_12 * beta_1 * beta_2 * sigma_1 * sigma_2 * (t_1 - t_0)

        T_0 = self.__difference_to_market_date(fwd_date_1)
        T_1 = self.__difference_to_market_date(fwd_date_2)

        if t_0 > T_0:
            return 0.0

        # t_0 < T_0
        # integrate two functions either until T_1 or t_1
        return (
            rho_12
            * beta_1
            * beta_2
            * sigma_1
            * sigma_2
            / kappa_12
            * np.exp(-kappa_1 * T_0 - kappa_2 * T_1)
            * (np.exp(kappa_12 * min(T_1, t_1)) - np.exp(kappa_12 * t_0))
        )

    def black_vol(
        self,
        # asset: str,
        fwd_date: datetime.date,
        kappa_vec: np.array,
        sigma_vec: np.array,
        corr_matrix: np.array,
    ):
        """Computes the model black vol until fwd_date.

        :param asset: asset for which this is computed, e.g. 'WTI'
        :param fwd_date: forward date for which the vol is computed.
        :param kappa_vec: kappa of the model.
        :param sigma_vec: sigma of the model.
        :param corr_matrix: corrlation of the model.
        """

        return np.sqrt(
            self.__fwd_square_vol(
                # asset,
                kappa_vec,
                sigma_vec,
                corr_matrix,
                fwd_date,
                self.mkt_date,
                fwd_date,
            )
            / self.__difference_to_market_date(fwd_date)
        )

    def __fwd_square_vol(
        self,
        # asset: str,
        kappa: np.array,
        sigma: np.array,
        corr_matrix: np.array,
        fwd_tenor: datetime.date,
        fwd_date_1: datetime.date,
        fwd_date_2: datetime.date,
    ):
        """Computes forward integrated square vol (function V) from fwd_date_1 to fwd_date_2.

         \int _{fwd_date_1} ^{fwd_date_2} ( e^(-kappa_1 (T-t)) * sigma_1 + e^{-kappa_2(T-t)) + cross terms)

        :param asset: asset to be considered. (e.g. 'WTI')
        :param kappa: vec of kappas
        :param sigma: vector of sigmas
        :param corr_matrix: correlation matrix for asset
        :param fwd_tenor: tenor for which this forward volatility is computed.
        :param fwd_date_1: start of forward volatility computation
        :param fwd_date_2: end of forward vol computation, fwd_date_2 > fwd_date_1
        """

        assert (
            fwd_date_1 <= fwd_date_2
        ), "Integration between {fwd_date_1} and {fwd_date_2} is wrong. {fwd_date_1} should be smaller than {fwd_date_2}"

        assert fwd_tenor >= max(
            fwd_date_1, fwd_date_2
        ), f"Forward tenor {fwd_tenor} should be bigger than both integration tenors {fwd_date_1}, {fwd_date_2}"

        nb_factors = 2  # self.nb_factors_for_asset(asset)

        t_1 = self.__difference_to_market_date(fwd_date_1)
        t_2 = self.__difference_to_market_date(fwd_date_2)
        T = self.__difference_to_market_date(fwd_tenor)

        direct_terms = sum(
            [
                sigma_i**2
                * (np.exp(-2 * kappa_i * (T - t_2)) - np.exp(-2 * kappa_i * (T - t_1)))
                for kappa_i, sigma_i in zip(kappa, sigma)
            ]
        )

        cross_terms = sum(
            [
                sigma[i]
                * sigma[j]
                * 2
                * corr_matrix[i, j]
                / (kappa[i] + kappa[j])
                * (
                    np.exp(-(kappa[i] + kappa[j]) * (T - t_2))
                    - np.exp(-(kappa[i] + kappa[j]) * (T - t_1))
                )
                for i in range(nb_factors)
                for j in range(i + 1, nb_factors)
            ]
        )

        return direct_terms + cross_terms

    # TODO: HERE SHOULD BE CHANGED TO ADD THIS CACHE
    # @lru_cache(maxsize=_BETA_T_CACHE_SIZE)
    def _beta_T(
        self,
        # asset: str, tenors=None,
        fwd_tenor: datetime.date,
        atm_vol: float,
        kappa_vec: Tuple[float, float],
        sigma_vec: Tuple[float, float],
        rho: float,
    ):
        """Adjusts beta_T so that the atm vol is fitted perfectly.
            (assuming that kappa, sigma, rho has already been calibrated).
            The results are memoized.

        :param asset: name of the asset calibrated (e.g. 'WTI')
        :param tenors: list of forward tenors for which the beta is
            calibrated (normally in form: List[datetime.date])
        """

        # tenors_used = tenors if tenors else self.vol_curve_names(asset).tenors

        black_vol = self.black_vol(
            # asset,
            fwd_tenor,
            kappa_vec,
            sigma_vec,
            np.array([[1.0, rho], [rho, 1.0]]),
            # self._kappa_vec(asset),
            # self._sigma_vec(asset),
            # self._factor_corr_mat(asset, asset),
        )

        return atm_vol / black_vol

        # return np.array(
        #     [self.vol_curve_names(asset).atm_vol(tenor) for tenor in tenors_used]
        # ) / np.array(
        #     [
        #         self.black_vol(
        #             asset,
        #             tenor,
        #             self._kappa_vec(asset),
        #             self._sigma_vec(asset),
        #             self._factor_corr_mat(asset, asset),
        #         )
        #         for tenor in tenors_used
        #     ]
        # )

    @staticmethod
    def __integr_analy(
        real_roots: np.ndarray,
        A0: float,
        A1: float,
        A2: float,
        A3: float,
        A4: float,
        V: float,
    ):
        """Integrate the polynomial between the roots, used for option calibration.
            Computes E[ p(x)_+ ]  where p(x) = A_0 x**4 + A_1 x**3 + A_2 x**2 + A_3 x + A_4

        :param real_roots: real roots of the polynomial described above.
        :returns: integrated value of the option w/ the prescribed parameters
        :rtype: float
        """

        Asigma = np.array([A0, A1, A2, A3, A4]) * np.array(
            [1.0, V, V**2, V**3, V**4]
        )  # A multiplied by sigmas
        nb_real_roots = len(real_roots)

        if nb_real_roots == 0:  # integrate polynomial function over whole of real axis
            if A4 > 0 or (A4 == 0 and A2 > 0) or (A4 == 0 and A2 == 0 and A0 > 0):
                return Asigma[0] + Asigma[2] + 3.0 * Asigma[4]

            return 0.0

        # abbreviations
        cdf_below = ComMathsMixin._trunc_normal_below
        cdf_above = ComMathsMixin._trunc_normal_above
        cdf_interval = ComMathsMixin._trunc_normal_interval

        if nb_real_roots == 1:
            if A3 > 0:
                return np.sum(cdf_below(real_roots[0]) * Asigma)

            return np.sum(cdf_above(real_roots[0]) * Asigma)

        if nb_real_roots in [2, 3]:  # integrate over 2 intervals
            if A4 > 0:
                return np.sum(cdf_above(real_roots[0]) * Asigma) + np.sum(
                    cdf_below(real_roots[1]) * Asigma
                )

            if A4 < 0.0:
                return np.sum(cdf_interval(real_roots[0], real_roots[1]) * Asigma)

            if A4 == 0.0 and A3 != 0.0:
                if A3 > 0.0:
                    return np.sum(
                        cdf_interval(real_roots[0], real_roots[1]) * Asigma
                    ) + np.sum(cdf_below(real_roots[2]) * Asigma)
                # A3 < 0
                return np.sum(cdf_above(real_roots[0]) * Asigma) + np.sum(
                    cdf_interval(real_roots[1], real_roots[2]) * Asigma
                )
            if A4 == 0.0 and A3 == 0.0:
                if A2 < 0.0:
                    return np.sum(cdf_interval(real_roots[0], real_roots[1]) * Asigma)

                return np.sum(cdf_above(real_roots[0]) * Asigma) + np.sum(
                    cdf_below(real_roots[1]) * Asigma
                )

        # elif nb_real_roots == 4:  # integrate over 3 intervals
        if A4 > 0:
            return (
                np.sum(cdf_above(real_roots[0]) * Asigma)
                + np.sum(cdf_below(real_roots[3]) * Asigma)
                + np.sum(cdf_interval(real_roots[1], real_roots[2]) * Asigma)
            )
        # A4 < 0
        return np.sum(cdf_interval(real_roots[0], real_roots[1]) * Asigma) + np.sum(
            cdf_interval(real_roots[2], real_roots[3]) * Asigma
        )

    @staticmethod
    def __integr_num(A_V: np.array, call_put_ind: int, strike: float) -> float:
        """Integrate numerically between the roots of the polynomials.
             IMPORTANT: For testing purposes only, in prod. use the code in integr_analy.

        :param A_V: array of A0, A1, A2, A3, A4, V
        :param call_put_ind: indicator for call (1) or put (-1)
        :param strike: option strike.
        :returns: value of the option for the parameters given in A_V.
        """

        from scipy.integrate import quad

        A0, A1, A2, A3, A4, V = A_V
        if call_put_ind == 1:
            A0 -= strike
        else:  # put
            A0 = strike - A0
            A1 = -A1
            A2 = -A2
            A3 = -A3
            A4 = -A4

        return quad(
            lambda x: np.max(
                [
                    A0
                    + A1 * V * x
                    + A2 * V**2 * x**2
                    + A3 * V**3 * x**3
                    + A4 * V**4 * x**4,
                    0.0,
                ]
            )
            / np.sqrt(2.0 * np.pi)
            * np.exp(-(x**2) / 2.0),
            -np.inf,
            np.inf,
        )[0]

    def _polynomial_european(
        self,
        asset: str,
        C_vec: np.array,
        fwd_date: datetime.date,
        strike: float,
        call_put_ind: int,
        ttm: float,
        debug_mode=False,
    ) -> float:
        """Value of european call option in skew model with strike.

        :param asset: asset to consider, e.g. 'WTI'.
        :param C_vec: vector of skew parameters
        :param fwd_date: option maturity.
        :param strike: strike of the option
        :param call_put_ind: indicator whether this is a call (1) or a put (-1)
        :param ttm: time to maturity of the option.
        :param debug_mode: whether to compute the polynomial european option numerically or analytically.
        """

        # obtaining the coefficients
        A0, A1, A2, A3, A4, V = self._skew_params(asset, C_vec, fwd_date)

        if call_put_ind == 1:
            A0 -= strike
        else:  # put
            A0 = strike - A0
            A1 = -A1
            A2 = -A2
            A3 = -A3
            A4 = -A4

        if debug_mode:
            poly_roots = np.sort(np.poly1d([A4, A3, A2, A1, A0]).roots)

        else:
            if A4 == 0.0 and A3 == 0.0 and A2 == 0.0:
                poly_roots = [-A0 / A1]
            elif A4 == 0.0 and A3 == 0.0:
                poly_roots = np.sort(QuadRoots(np.array([A2, A1, A0])))
            elif A4 == 0.0:
                poly_roots = np.sort(CubicRoots(np.array([A3, A2, A1, A0])))
            elif np.abs(A4) < 1e-6:
                poly_roots = np.sort(np.poly1d([A4, A3, A2, A1, A0]).roots)
            else:
                poly_roots = np.sort(QuarticRoots(np.array([A4, A3, A2, A1, A0])))

        real_roots = poly_roots[poly_roots == poly_roots.real].real  # real roots only

        # if debug_mode:  # debug, selects the numeric approach
        #    return disc_fact * self.__integr_num(Asigma, call_put_ind, strike)
        # else:  # production mode

        return self.DF(ttm) * self.__class__.__integr_analy(
            real_roots / V, A0, A1, A2, A3, A4, V
        )

    def __distance_model_market_black_vol(
        self,
        # asset: str,
        atm_vols: List[float],
        kappa_vec: np.array,
        sigma_vec: np.array,
        rho_vec: np.array,
        fwd_tenors: List[datetime.date],
    ) -> float:
        """Distance between model & market black volatility, used for calibration of the entire curve.
            Produces a sqaure sum of distance elements.

        :param asset: asset to consider, e.g. 'WTI'
        :param kappa_vec: kappa vector to calibrate
        :param sigma_vec: sigma vector to calibrate for asset asset
        :param rho_vec: correlation _vector_ to calibrate
        :param fwd_tenors: perhaps you want to restrict the forward tenors to a pre-defined set. If None,
                          fwd tenors from the forward curve are used.
        :returns: square root of the distance between market and model numbers.
        """

        # either use fwd_tenors provided, or use the fwd tenors from the fwd curve.
        # fwd_tenors_used = (
        #     fwd_tenors if fwd_tenors else self.fwd_curve_names(asset).fwd_tenors
        # )

        model_vols = [
            self.black_vol(
                # asset,
                fwd_date,
                kappa_vec,
                sigma_vec,
                rho_vec,  # self._construct_corr_asset(asset, rho_vec),
            )
            for fwd_date in fwd_tenors
        ]

        # market_vol_curve = self.vol_curve_names(asset)
        # market_vol = [
        #     market_vol_curve.atm_vol(fwd_date) for fwd_date in fwd_tenors_used
        # ]

        return sum(
            [
                (market_vol_elt - model_vol_elt) ** 2
                for market_vol_elt, model_vol_elt in zip(atm_vols, model_vols)
            ]
        )

    def __factor_corr_mat_default(self, same_asset_corr=0.98):
        """Default correlation matrix between asset_1 and asset_2.

        :param asset_1: first asset, e.g. 'WTI',
        :param asset_2: second asset, e.g. 'BRENT'
        :param same_asset_corr: default correlation between tenors on the same curve.
        :param diff_asset_corr: default correlation between different assets
        """

        return corr_hyp_sec_mat(same_asset_corr, range(2))

    def __factor_corr_mat_lb_ub(self, lb_ub_ind="ub") -> np.ndarray:
        """Sets the default factor correlation lower (lb) and upper (ub) bound between asset_1 and asset_2.

        :param asset_1: first asset for default correlation, e.g. 'WTI'
        :param asset_2: second asset, e.g. 'BRENT'
        :param lb_ub_ind: indicator whether it's upper bound 'ub' or lower bound 'lb'
        """
        # TODO: INCLUDE THIS HERE
        #        , 'corr_init': np.array([[1., 0.5], [0.5, 1.]])
        #        , 'corr_lb': np.array([[1., -0.99], [-0.99, 1.]])
        #        , 'corr_ub': np.array([[1., 0.99], [0.99, 1.]])} ):

        # lb_ub_fact = -0.999 if lb_ub_ind == "lb" else 0.999

        tmp_1 = np.ones(2, 2)
        tmp_ut = np.triu(tmp_1, 1)
        tmp_lt = np.tril(tmp_1, -1)
        return (
            tmp_1 - tmp_ut * 0.001 - tmp_lt * 0.001
            if lb_ub_ind == "ub"
            else tmp_1 - tmp_ut * 1.999 - tmp_lt * 1.999
        )

    def _kappa_sigma_rho(
        self, atm_vols: List[Tuple[datetime.date, float]]
    ) -> np.ndarray:
        """Calibrates kappa and sigma and rho parameters
            of the log-normal part of the model.

        :param atm_vols: list of maturities, and atm vols.
        :returns: vector of calibrated kappa, sigma and correlation.
            kappa, sigma are vectors,
            rho is a matrix (upper triangular).
        """

        nbf = 2  # self.nb_factors_for_asset(asset)

        # extracting the upper triangular part of the correlation matrix
        # fcm_init = (
        #     self._factor_corr_mat_default()
        # )  # initial value of the factor correlation.
        # fcm_lb = self._factor_corr_mat_lb_ub(
        #     lb_ub_ind="lb"
        # )  # lower bound of the factor corr. mtx.
        # fcm_ub = self._factor_corr_mat_default(
        #     lb_ub_ind="ub"
        # )  # upper bound of the factor corr. mtx.

        # init_kappa_sigma_rho = np.concatenate(
        #     [
        #         self._kappa_default(nbf, "init"),
        #         self._sigma_default(nbf, "init"),
        #         np.triu(fcm_init, 1)[np.triu(fcm_init, 1) != 0],
        #     ]
        # )

        # lower_bounds = np.concatenate(
        #     [
        #         self._kappa_default(nbf, "lb"),
        #         self._sigma_default(nbf, "lb"),
        #         np.triu(fcm_lb, 1)[np.triu(fcm_lb, 1) != 0],
        #     ],
        # )

        # upper_bounds = np.concatenate(
        #     [
        #         self._kappa_default(nbf, "ub"),
        #         self._sigma_default(nbf, "ub"),
        #         np.triu(fcm_ub, 1)[np.triu(fcm_ub, 1) != 0],
        #     ],
        # )

        init_kappa_sigma_rho = (0.05, 0.15, 0.5, 0.2, 0.5)
        lower_bounds = (0.001, 0.001, 0.001, 0.001, -0.999)
        upper_bounds = (10.0, 10.0, 5.0, 5.0, 0.9999)
        fwd_tenors = [x[0] for x in atm_vols]
        atm_vols_2 = [x[1] for x in atm_vols]

        pr_solve = minimize(
            lambda kappa_sigma_rho_vec: self.__distance_model_market_black_vol(
                atm_vols_2,
                kappa_sigma_rho_vec[:nbf],
                kappa_sigma_rho_vec[nbf : (2 * nbf)],
                np.array(
                    [
                        [1.0, kappa_sigma_rho_vec[(2 * nbf) :][0]],
                        [kappa_sigma_rho_vec[(2 * nbf) :][0], 1.0],
                    ]
                ),
                fwd_tenors,
            ),
            init_kappa_sigma_rho,
            bounds=Bounds(lower_bounds, upper_bounds),
        )

        if pr_solve.success:
            result = pr_solve.x
            kappa_1, kappa_2, sigma_1, sigma_2, rho = result
            return {
                "kappa_1": kappa_1,
                "kappa_2": kappa_2,
                "sigma_1": sigma_1,
                "sigma_2": sigma_2,
                "rho": rho,
            }

        # problem is not feasible: TODO: FOR NOW RETURN DEFAULT VALUES
        kappa_1, kappa_2, sigma_1, sigma_2, rho = init_kappa_sigma_rho
        return {
            "kappa_1": kappa_1,
            "kappa_2": kappa_2,
            "sigma_1": sigma_1,
            "sigma_2": sigma_2,
            "rho": rho,
        }


def _example_one():
    mkt_date = datetime.date(2026, 1, 11)
    atm_vols = [
        (datetime.date(2026, 2, 1), 0.4),
        (datetime.date(2026, 3, 1), 0.34),
        (datetime.date(2026, 4, 1), 0.32),
        (datetime.date(2026, 5, 1), 0.311),
        (datetime.date(2026, 6, 1), 0.299),
        (datetime.date(2026, 7, 1), 0.278),
        (datetime.date(2026, 8, 1), 0.27),
        (datetime.date(2026, 9, 1), 0.26),
    ]

    mrd1 = MrdsCalibMixin(mkt_date)
    r1 = mrd1._kappa_sigma_rho(atm_vols)

    print(r1)


if __name__ == "__main__":
    _example_one()
