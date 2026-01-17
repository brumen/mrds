""" MRDS Calibrating options on one maturity only.
"""

import datetime
import numpy as np
import scipy
import logging
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Tuple, List, Optional, Dict
from logging import getLogger
from scipy.optimize import minimize, Bounds, OptimizeResult
from multiprocessing import Pool, cpu_count
from pydantic import BaseModel

from mrds.correlations import corr_hyp_sec_mat
from mrds.mrds_maths import ComMathsMixin
from mrds.quartic.quartic_cy import QuadRoots, CubicRoots, QuarticRoots
from mrds.vols.vols_basic import black_vol_inverse
from mrds.discount import DiscountCurve
from mrds.mrds_discount import MrdsDiscount

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)
_logger = getLogger(__name__)


class MrdsModel(BaseModel):
    ksr: Tuple[Tuple[float, float], Tuple[float, float], float]
    expiries: List[datetime.date]
    skews: Dict[datetime.date, Tuple[float, float, float]]
    betas: Dict[datetime.date, float]

class MrdsCalibMixin(MrdsDiscount):
    """Calibration of one maturity only."""

    _MIN_OPTION_PRICE = 1.e-2

    def __init__(self, mkt_date: datetime.date, df=0.99):

        dc = DiscountCurve(
            mkt_date
        ).discount_function_2  # pre-configured discount curve.

        super().__init__(mkt_date, dc)  # from MrdsDiscount

        # (k1, k2), (s1, s2), rho = self._ksr
        self._ksr: Optional[Tuple[float, float, float, float, float]] = None
        self._c_vec = {}  # : Optional[Tuple[float, float, float]] = None
        self._beta_T = {}
        self.black_vol_inverse_tol = 1e-4

    @classmethod
    def from_mrds_model(
        cls, 
        mkt_date: datetime.date, 
        mrds_model: MrdsModel,
    ):
        """Reconstructs the MrdsCalibMixin model from the mrds_model and 
            mkt_date.
        """

        mrds_obj = cls(mkt_date=mkt_date)
        mrds_moj._ksr = (
            mrds_model.ksr[0][0], 
            mrds_model.ksr[0][1], 
            mrds_model.ksr[1][0], 
            mrds_model.ksr[1][1], 
            mrds_model.ksr[2],
        )
        mrds_obj._c_vec = mrds_model.skews
        mrds_obj._beta_T = mrds_model.betas

        return mrds_obj

    def black_vol(
        self,
        fwd_date: datetime.date,
        kappa_vec: np.array,
        sigma_vec: np.array,
        corr_matrix: np.array,
    ):
        """Computes the model black vol until fwd_date.

        :param fwd_date: forward date for which the vol is computed.
        :param kappa_vec: kappa of the model.
        :param sigma_vec: sigma of the model.
        :param corr_matrix: corrlation of the model.
        """

        return np.sqrt(
            self.__fwd_square_vol(
                kappa_vec,
                sigma_vec,
                corr_matrix,
                fwd_date,
                self.mkt_date,
                fwd_date,
            )
            / self._difference_to_market_date(fwd_date)
        )

    def black_vol_current(self, fwd_date: datetime.date) -> Optional[float]:
        """Returns current black vol if calibrated, otherwise None."""

        if self._ksr is None or self._beta_T.get(fwd_date) is None:
            return None

        k1, k2, s1, s2, rho = self._ksr
        beta_T = self._beta_T.get(fwd_date)

        return beta_T * self.black_vol(
            fwd_date,
            (k1, k2),
            (s1, s2),
            np.array([[1.0, rho], [rho, 1.0]]),
        )

    def __fwd_square_vol(
        self,
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

        nb_factors = 2

        t_1 = self._difference_to_market_date(fwd_date_1)
        t_2 = self._difference_to_market_date(fwd_date_2)
        T = self._difference_to_market_date(fwd_tenor)

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
    def beta_T_calib(
        self,
        fwd_tenor: datetime.date,
        market_atm_vol: float,
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
            fwd_tenor,
            kappa_vec,
            sigma_vec,
            np.array([[1.0, rho], [rho, 1.0]]),
        )

        self._beta_T[fwd_tenor] = market_atm_vol / black_vol
        return self._beta_T[fwd_tenor]

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

    def _skew_params(
        self,
        C_vec: np.array,
        fwd_date: datetime.date,
        fwd_value: float,
    ) -> tuple:
        """Given the C parameters, returns the parameters for the
            option value computation using the polynomial approach.

        :param asset: asset for which skew parameters are computed.
        :param C_vec: vector of calibrated skew parameters, [c0, c1, c2]
        :param fwd_value: forward value for tenor fwd_date
        :returns: a tuple of A0, A1, A2, A3, A4, V used in
            the _polynomial_european method.
        """

        cc1, cc2, cc3 = C_vec

        # integrated volatility
        ttm = self._difference_to_market_date(fwd_date)
        v = self.black_vol_current(fwd_date) * np.sqrt(ttm)
        f0t = fwd_value

        return (
            (1.0 - cc1 * v**2 / 2.0 + cc3 * v**4 / 8.0) * f0t,
            (1.0 - cc2 * v**2 / 2.0) * f0t,
            (cc1 / 2.0 - cc3 * v**2 / 4.0) * f0t,
            (cc2 / 6.0) * f0t,
            (cc3 / 24.0) * f0t,
            v,
        )

    def _polynomial_european(
        self,
        C_vec: np.array,
        fwd_date: datetime.date,
        fwd_value: float,
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
        :param debug_mode: whether to compute the polynomial european
            option numerically or analytically.
        """

        # obtaining the coefficients
        A0, A1, A2, A3, A4, V = self._skew_params(C_vec, fwd_date, fwd_value)

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

        return self.DF(fwd_date) * self.__class__.__integr_analy(
            real_roots / V, A0, A1, A2, A3, A4, V
        )

    def __distance_model_market_black_vol(
        self,
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
            self._ksr = result
            return {
                "kappa_1": kappa_1,
                "kappa_2": kappa_2,
                "sigma_1": sigma_1,
                "sigma_2": sigma_2,
                "rho": rho,
            }

        # problem is not feasible: TODO: FOR NOW RETURN DEFAULT VALUES
        kappa_1, kappa_2, sigma_1, sigma_2, rho = init_kappa_sigma_rho
        self._ksr = init_kappa_sigma_rho
        return {
            "kappa_1": kappa_1,
            "kappa_2": kappa_2,
            "sigma_1": sigma_1,
            "sigma_2": sigma_2,
            "rho": rho,
        }

    def __deltas_to_strikes(
        self,
        tenor_date: datetime.date,
        delta_vec_list: np.array,
        atm_vol: float,
        fwd_value: float,
    ) -> np.array:
        """Converts deltas to strikes for particular asset and tenor.

        :param atm_vol: atm vol for the tenor_date.
        :param tenor_date: tenor considered.
        :param fwd_value: forward value for tenor_date
        :returns: a vector of deltas from the strikes given in self.delta_vec_list
        """

        integrated_vol = atm_vol * np.sqrt(self._difference_to_market_date(tenor_date))

        return fwd_value / np.exp(
            (norm.ppf(delta_vec_list) - 0.5 * integrated_vol)
            * integrated_vol
        )

    @staticmethod
    def strike_to_delta(strike: float, stock_price: float, atm_vol: float, ttm: float):
        """Converts the strike to delta."""

        # TODO: CHECK THIS PART
        delta = norm.cdf(
            (np.log(stock_price/strike) + 0.5 * atm_vol**2 * np.sqrt(ttm))/
            (atm_vol * np.sqrt(ttm))
        )

        return delta

    def __model_vol_surface(
        self,
        C_vec,
        fwd_date: datetime.date,
        deltas: np.array,
        # fwd_value: float,
    ) -> List[float]:
        """Computes model vols for asset, C_vec and forward date fwd_date.

        :param asset: commodity considered.
        :param C_vec: skew vector
        :param fwd_date: forward date for which the model vols are computed.
        :param deltas: deltas to be used for calibration.
        :returns: list of volatilities for deltas for the particular parameters.
        """

        deltas_used = deltas  # if deltas else self._default_deltas_for_skew()
        atm_vol = self.black_vol_current(fwd_date)  # TODO: CHECK IF THIS IS CORRECT

        fwd_value = 100.0  # unimportant
        strikes = self.__deltas_to_strikes(fwd_date, deltas_used, atm_vol, fwd_value)

        # fwd_value = self.fwd_curve_names(asset).fwd_value(fwd_date)
        cp_ind = np.array([1 if strike >= fwd_value else -1 for strike in strikes])
        option_tenor = fwd_date  # self.__option_tenor_for_fwd_tenor(
        # asset, fwd_date
        # )  # option tenor corresponding to fwd_date
        ttm_numerical = self._difference_to_market_date(option_tenor)
        option_prices = [
            self._polynomial_european(
                C_vec, fwd_date, fwd_value, strike, cp, ttm_numerical
            )
            for strike, cp in zip(strikes, cp_ind)
        ]
        # if option prices are 0 -> correct to MIN_OPTION_PRICE
        option_prices = [
            option_price if option_price > 0.0 else self._MIN_OPTION_PRICE
            for option_price in option_prices
        ]
        discount_fact = self.DF(option_tenor)

        # numerical value of the option tenor
        option_tenor_num = self._difference_to_market_date(option_tenor)
        model_vols = [
            black_vol_inverse(
                fwd_value,
                strike,
                opt_price,
                option_tenor_num,
                discount_fact,
                call_put_ind,
                self.black_vol_inverse_tol,
            )
            for opt_price, strike, call_put_ind in zip(option_prices, strikes, cp_ind)
        ]
        return model_vols

    def _calibrate_skew_one_date(
        self,
        fwd_date: datetime.date,
        deltas_implied_vols: List[Tuple[float, float]],
        initial_guess=(1.0, 0.0, 0.0),
    ) -> Tuple[np.array, float]:
        """Optimization function to minimize over the fwd_dates.

        :param fwd_date: forward date for which the skew calibration is done.
        :param deltas_implied_vols: list of tuples of (delta, implied_vol)
            for fwd_date.
        :param fwd_value: forward value for that
        """

        deltas = [x[0] for x in deltas_implied_vols]
        # weights = np.array([1.0 - np.abs(delta - 0.5) for delta in deltas])
        implied_vols = [x[1] for x in deltas_implied_vols]

        _logger.debug(f"Calibrating skew params. for date {fwd_date}.")
        # initial_guess = np.array(
        #     [3.0, 100.0, 20.0]
        # )  # possible guess, try w/ multiple guesses.
        initial_guess = np.array(initial_guess)

        def model_minus_market(C_attempt):
            model_vols = np.array(self.__model_vol_surface(C_attempt, fwd_date, deltas))
            try: 
                return scipy.linalg.norm((model_vols - implied_vols))  # weighted by delta
            except Exception as e:
                _logger.error(f"Problem computing model vols {model_vols}: {e}")
                return np.inf

        c_vec_sol: OptimizeResult = minimize(
            model_minus_market,
            initial_guess,
        )

        if c_vec_sol.success:
            return (c_vec_sol.x, c_vec_sol.fun)  # 3 elements

        # solution not feasible  # TODO: MAYBE THERE IS SOMETHING MORE TO DO HERE
        return (initial_guess, np.inf)

    def calibrate_skew(
        self,
        fwd_date: datetime.date,
        deltas_implied_vols: List[Tuple[float, float]],
    ) -> np.array:

        initial_vals = [
            (1.0, 0.0, 0.0),
            (1.0, 10.0, 20.0),
            (3.0, 100.0, 20.0),
        ]

        c_res, residual_val = self._calibrate_skew_one_date(
            fwd_date,
            deltas_implied_vols,
            initial_guess=initial_vals[0],
        )
        _logger.debug(f"First attempt at C: {c_res}, {residual_val}")
        self._c_vec[fwd_date] = c_res

        for initial_val in initial_vals[1:]:
            c_res, residual_val_new = self._calibrate_skew_one_date(
                fwd_date,
                deltas_implied_vols,
                initial_guess=initial_val,
            )
            _logger.debug(f"Next attempt at C: {c_res}, {residual_val_new}")
            if residual_val_new < residual_val:
                residual_val = residual_val_new
                self._c_vec[fwd_date] = c_res

        return self._c_vec[fwd_date]

    def plot_fit(
        self, deltas_implied_vols: List[Tuple[float, float]], fwd_date: datetime.date
    ):
        """

        :param deltas_implied_vols: list of tuples of (delta, implied_vol)
            for fwd_date.
        """

        if self._c_vec.get(fwd_date) is None or self._ksr is None:
            return None

        deltas = [x[0] for x in deltas_implied_vols]
        implied_vols = [x[1] for x in deltas_implied_vols]

        model_vols = np.array(
            self.__model_vol_surface(self._c_vec.get(fwd_date), fwd_date, deltas)
        )

        plt.plot(deltas, implied_vols)
        plt.plot(deltas, model_vols)
        plt.show()

    def calibrate_dates(
        self,
        fwd_dates: List[datetime.date],
        deltas_implied_vols: Dict[datetime.date, List[Tuple[float, float]]],
        multi_threaded: bool = False,
    ) -> Dict[datetime.date, np.array]:
        """Optimization function to minimize over the fwd_dates.

        :param fwd_dates: list of forward dates for calibration of the skew.
        :param deltas_implied_vols: key is the fwd date, value is the
           deltas_implied_vols for that date.
        :param multi_threaded: indicator whether to use multi-threaded.
        """

        if not multi_threaded:
            return {
                fwd_date: self.calibrate_skew(
                    fwd_date, deltas_implied_vols=deltas_implied_vols.get(fwd_date)
                )
                for fwd_date in fwd_dates
            }

        # multi-threaded version.
        with Pool(processes=cpu_count()) as pool:
            _ = pool.map(
                self.calibrate_skew,
                zip(
                    fwd_dates,
                ),
            )


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
    ksr = mrd1._kappa_sigma_rho(atm_vols)
    # compute betas
    k1 = ksr["kappa_1"]
    k2 = ksr["kappa_2"]
    s1 = ksr["sigma_1"]
    s2 = ksr["sigma_2"]
    rho = ksr["rho"]
    fwd_tenor = datetime.date(2026, 2, 1)
    for fwd_tenor_1, atm_vol_1 in atm_vols:
        _ = mrd1._beta_T(fwd_tenor_1, atm_vol_1, (k1, k2), (s1, s2), rho)

    deltas_implied_vols = [
        (0.9, 0.6),
        (0.8, 0.5),
        (0.6, 0.41),
        (0.5, 0.4),
        (0.4, 0.43),
        (0.3, 0.47),
        (0.2, 0.58),
    ]

    skew_result = mrd1.calibrate_skew(
        fwd_tenor,
        deltas_implied_vols,
    )

    print("C_VEC:", skew_result)
    mrd1.plot_fit(
        deltas_implied_vols,
        fwd_tenor,
    )


if __name__ == "__main__":
    _example_one()
