"""
JWSS7 volatility structure - computation and display.

"""

import datetime
import numpy as np
import logging
import tkinter as tk
import scipy

from scipy.stats import norm
from scipy.optimize import minimize
from enum import Enum

from typing import Dict, Tuple, Union, List
from scipy.interpolate import splrep

from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


from mrds.ds import get_vol_curve
from mrds.forward_curve import FwdCurve
from mrds.vols.vols import ATMFVolatility
from mrds.vols.vols_draw import VolatilityDrawMixin
from mrds.pricers.pricers_fast import black_call_fast, black_put_fast


logger = logging.getLogger(__name__)


class CallPut(Enum):
    Call = 'Call'
    Put = 'Put'


class JWSS7Exception(Exception):
    pass


# structure representing JWSS7 params.
# the parameters are:
# sigma_0, A, B, C, P, alpha_C, alpha_P  # TODO: CHECK IF THIS IS THE CASE
JWSS7_STRUCT = Tuple[float, float, float, float, float, float, float]


class JWSS7Volatility(ATMFVolatility):
    """ Jump-wing parametrization.
    """

    INTERPOLATION_DEGREE = 2
    SCIPY_SOLVER = 'scipy_cobyla'

    @classmethod
    def from_db(cls, com_name: str, mkt_date: datetime.date, dcf: float = 365.25):
        """ Obtains the volatility from database.

        :param com_name: commodity name, e.g. 'WTI'
        :param mkt_date: market date, e.g. datetime.date(2015, 4, 1)
        :param dcf: day-count factor.
        """

        vol_type, vol_params = get_vol_curve(com_name, mkt_date)

        if vol_type != 'JWSS7':
            raise RuntimeError(f'Fetching the wrong curve. {com_name} has type {vol_type}')

        fwd_params = FwdCurve.from_db(mkt_date, com_name)

        return cls(
            com_name,
            mkt_date,
            fwd_params=fwd_params,
            vol_params=vol_params,
            dcf=dcf,
        )

    @property
    def _atm_vol_curve(self):
        """ Constructs the ATM vol curve.  Returns the object returned from splrep, to be used for splev.
        """

        if self._atm_vol_curve_interp:
            return self._atm_vol_curve_interp

        vol_dates = [(x - self.mkt_date).days / self._dcf for x in self.vol_dates]
        atm_vols = [x[0] for x in self._vol_params.values()]

        vol_dates_values = sorted(
            zip(vol_dates, atm_vols),
            key=lambda vol_date_val: vol_date_val[0]
        )

        self._atm_vol_curve_interp = splrep(
            [x[0] for x in vol_dates_values],
            [x[1] for x in vol_dates_values],
            k=self.INTERPOLATION_DEGREE,
        )

        return self._atm_vol_curve_interp


    @staticmethod
    def _transform_params_jwss7(
            jwss7_params: JWSS7_STRUCT,
    ) -> JWSS7_STRUCT:
        sigma_0, skew, smile, put_slope, put_bend, call_slope, call_bend = jwss7_params

        B = (2. * skew + put_slope) / (put_slope + call_slope)
        A = 0.5 * B * (1. - B) * (call_slope + put_slope)**2 / (smile + skew**2)

        # in the form of sigma_0, A, B, C, P, alphaC, alphaP
        return (
            sigma_0,
            A,
            B,
            call_slope/A,
            put_slope/A,
            call_bend,
            put_bend
        )

    @staticmethod
    def _transform_from_jwss7(
            vol_curve: Dict[datetime.date, JWSS7_STRUCT]
    ) -> Dict[datetime.date, JWSS7_STRUCT]:
        """ Returns jw7 parametrization from jwss7 for particular fwd date.

        vol_params in Jwss7: [S0, atm, skew, smile, putslope, putbend, callslope, callbend]
        vol_params in jw7  : [S0, atm, A   , B    , C       , P      , alphaC   , alphaP  ]
        """

        return {
            fwd_vol_date: JWSS7Volatility._transform_from_jwss7(jwss7_params_for_date)
            for fwd_vol_date, jwss7_params_for_date in vol_curve.items()
        }

    def _interpolate_params_for_fwd_date(
            self,
            fwd_date: datetime.date
    ) -> datetime.date:
        """ Interpolate parameters for forward date fwd_date.
        In this case I just select the next larger date, or if not, the largest date in the self._vol_params

        :param fwd_date: forward date for which parameters are requested.
        :returns: date corresponding to the fwd_date, in our case the next
           largest date.
        """

        input_dates = sorted(list(self.vol_dates))  # sort input dates
        selected_date = None
        for input_date in input_dates:
            if selected_date:
                if fwd_date < input_date <= selected_date:
                    selected_date = input_date
            else:
                if input_date > fwd_date:
                    selected_date = input_date

        if not selected_date:
            return max(input_dates)

        return selected_date

    @staticmethod
    def _vol_from_jw7(
            S0: float,
            K: float,
            ttm: float,
            jw7_params: JWSS7_STRUCT,
    ) -> float:
        """ Computes the volatility for parameters.

        :param S0: initial stock value.
        :param K: strike.
        :param ttm: time to maturity to which these parameters
            correspond.
        :param jw7_params: params in the jw7 form.
        :returns: volatility for those parameters.
        """

        sigma_0 = jw7_params[0]
        z = JWSS7Volatility.normalized_strike(S0, K, sigma_0, ttm)

        return JWSS7Volatility._vol_compute_from_jw7(z, jw7_params)

    @staticmethod
    def _vol_from_jwss7(
            S0: float,
            K: float,
            ttm: float,
            jwss7_params: JWSS7_STRUCT,
    ) -> float:
        """ Computes the volatility for parameters.

        :param S0: initial stock value.
        :param K: strike.
        :param ttm: time to maturity to which these parameters
            correspond.
        :param jw7_params: params in the jw7 form.
        :returns: volatility for those parameters.
        """

        jw7_params = JWSS7Volatility._transform_params_jwss7(jwss7_params)

        return JWSS7Volatility._vol_from_jw7(
            S0, K, ttm, jw7_params,
        )

    @staticmethod
    def _vol_compute_from_jw7(z: float, jw7_params: JWSS7_STRUCT) -> float:
        """ Compute jw7 vol from parameters.

        :param z: normalized strike, log(K/F_0) - see normalized_strike
           method on this class
        :param jw7_params: tuple of jw7 parameters.
        """

        sigma_0, A, B, C, P, alpha_C, alpha_P = jw7_params

        return sigma_0 * np.sqrt(
            1. + A * np.log(
                B * np.exp(C * (z / (1.0 + z*z) ** (alpha_C/2))) +
                (1. - B) * np.exp(- P * (z / (1.0 + z*z) ** (alpha_P/2)))
            )
        )

    def _vol_compute(
            self,
            fwd_date: datetime.date,
            normalized_strike: float,
    ) -> float:
        """ Computes the volatility given the following parameters:

        :param fwd_date: forward date on the vol curve.
        :param normalized_strike: normalized strike (log(F/S))
        """

        fwd_date_on_curve = self._interpolate_params_for_fwd_date(fwd_date)
        jw7_params = self._vol_params[fwd_date_on_curve]

        return self._vol_compute_from_jw7(normalized_strike, jw7_params)

    @staticmethod
    def calibrate_params(
            mkt_date: datetime.date,
            S0: float,
            prices_strikes_cp: List[Tuple[float, float, CallPut]],
            maturity: datetime.date,
            r: float,  # interest rate charged.
    ) -> JWSS7_STRUCT:
        """Calibrates the prices/strikes for the designated maturity
            by least-squares.

        :param mkt_date: market date
        :param S0: current spot price
        :param prices_strikes_cp: list of prices and strikes you are
            calibrating, along w/ call/put indicator.
        :param maturity: maturity of the option.
        """

        ttm = (maturity - mkt_date).days / 365.25

        def calibrate_jwss7(jwss7_p: JWSS7_STRUCT) -> float:

            diffs = 0
            for price, strike, call_put in prices_strikes_cp:
                black_pricer = black_call_fast if call_put == CallPut.Call \
                    else black_put_fast
                vol = JWSS7Volatility._vol_from_jw7(S0, strike, ttm, jwss7_p)
                black_price = black_pricer(S0, strike, r, vol, ttm)
                diffs += (black_price - price)**2

            return diffs

        res = minimize(
            calibrate_jwss7,
            x0=(0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            method='nelder-mead',  # TODO: CHECK HERE!!
            bounds=(
                (0, np.inf),  # sigma_0
                (0, np.inf),  # A - CHECK THIS BOUND
                (0, np.inf),  # B - CHECK BOUND
                (0, np.inf),  # C - CHECK THIS BOUND
                (0, np.inf),  # P - CHECK BOUND
                (0, np.inf),  # alpha_C - CHECK THIS BOUND
                (0, np.inf),  # alpha_P - CHECK BOUND
            ),
        )

        # check the results outcome

        return res.x

    def implied_vol(
            self,
            fwd_date: Union[datetime.date, int],
            strike: float,
            ttm: float
    ) -> float:
        """ Implied vol for the fwd_date.

        :param fwd_date: date for which the volatility is to be computed.
        :param strike: strike price
        :param ttm: time to maturity
        """

        atm_vol = self._vol_params[self._interpolate_params_for_fwd_date(fwd_date)][0]  # TODO: THIS IS RISKY
        normalized_strike = JWSS7Volatility.normalized_strike(
            self._fwd_params.fwd_value(fwd_date),
            np.array([strike]),
            atm_vol,  # atm vol is the first element
            ttm
        )[0]

        return self._vol_compute(
            fwd_date, normalized_strike
        )

    def local_vol(self, fwd_date: datetime.date, T: float, S: float, ttm: float) -> float:
        """ Local volatility of the JWSS7 parametrization.

        :param fwd_date: forward index that we are computing the local vol of
        :param S: value of forward at which to evaluate local vol.
        :param ttm: option time to maturity
        """

        sigma_0, A, B, C, P, alphaC, alphaP = self._vol_params(fwd_date)  # TODO: THIS DOESNT WORK, FIX LATER.
        S_0 = self._fwd_params.fwd_value(fwd_date)

        z = self.normalized_strike(S_0, S, sigma_0, ttm)  # TODO: CHECK IF THIS IS CORRECT
        sigma = self.implied_vol(S, T)

        d1 = (np.log(S / S_0) + sigma * sigma * ttm / 2.0) / (sigma * np.sqrt(ttm))
        d2 = d1 - sigma * np.sqrt(ttm)
        Xz = B * np.exp(C * z) + (1.0 - B) * np.exp(- P * z)

        sigmaK = A / (2.0 * Xz * K * np.sqrt (ttm) ) / ( np.sqrt ( 1.0 + A * np.log (Xz) ) ) * \
            (B * C * np.exp(C * z) - P * (1.0 - B) * np.exp(- P * z))

        d1K = ((- 1.0 / K + sigma * ttm * sigmaK) * sigma * np.sqrt(ttm) -
               ( np.log ( S / K ) + sigma * sigma * ttm / 2.0 ) * np.sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        d2K = ((- 1.0 / K - sigma * ttm * sigmaK) * sigma * np.sqrt(ttm) -
               ( np.log ( S / K ) - sigma * sigma * ttm / 2.0 ) * np.sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        denomin = (sigma_0 * np.sqrt(ttm) * K * Xz * np.sqrt(1.0 + A * np.log(Xz)))
        BCexpr = (B * C * np.exp(C * z) - P * (1.0 - B) * np.exp(- P * z))

        sigmaKK = A / (2.0 * np.sqrt(ttm)) * (- A / (2.0 * denomin * K * Xz * (1.0 + A * np.log(Xz))) *
                                           BCexpr * BCexpr - BCexpr * BCexpr / (denomin * K * Xz) +
                                           (B * C ** 2 * np.exp(C * z) + P ** 2 * (1.0 - B) * np.exp(- P * z)) /
                                           (denomin * K) - BCexpr *
                                           sigma_0 *
                                           np.sqrt(ttm) / (denomin * K)
                                           )

        # derivative of z wrt t
        zt = np.log(K / S) / sigma_0 * (-0.5 * ttm**(- 1.5))
        sigmat = sigma_0 ** 2 / (2.0 * sigma) * A / Xz  * \
            ( B * C * np.exp ( C * z) - P * (1 - B) * np.exp ( - P * z ) ) * \
            zt  # derivative of sigma wrt t

        up_part = sigma * sigma + 2.0 * ttm * sigma * sigmat

        down_part = (1.0 + K * d1 * np.sqrt (ttm) * sigmaK ) ** 2.0 + K * K * ttm * sigma * \
                    (sigmaKK - d1 * sigmaK * sigmaK * ttm)

        # catching nan-s
        if up_part / down_part < 0.:
            logger.info("Caution: Imaginary local vol., using ATM vol.")
            return sigma_0

        return np.sqrt(up_part / down_part)

    def callFutureT(self, fwdDate, S0, K, ttm):
        """
        Derivative of Black's call (with expirty time  T) on a futures contract
           with maturity ttm (cond: T < ttm)

        :param fwdDate: forward index that we are drawing the vol of
        :param S0:
        :param K: strike value
        :param ttm: time to maturity
        """

        sigma_0, A, B, C, P, alphaC, alphaP = self._transform_from_jwss7(fwdDate)

        z = JWSS7Volatility.normalized_strike(S0, K, sigma_0, ttm)
        sigma = self.implied_vol(S0, K, ttm)
        S0_local = S0  # TODO: CHECK IF THIS IS REALLY NECESSARY

        Xz = B * np.exp(C * z) + (1.0 - B) * np.exp(- P * z)

        d1 = (np.log(S0_local / K) + sigma * sigma * ttm / 2.0) / \
            (sigma * np.sqrt(ttm))
        d2 = d1 - sigma * np.sqrt(ttm)
        zt = np.log(K / S0_local) / sigma_0 * (-0.5 * pow(ttm, - 1.5))  # z wrt t
        sigmat = sigma_0 * sigma_0 / (2.0 * sigma) * A / Xz  * \
            ( B * C * np.exp ( C * z) - P * ( 1.0 - B) * np.exp ( - P * z ) ) * \
            zt  # derivative of sigma wrt t
        sigma2t = 2.0 * sigma * sigmat  # derivative of sigma^2 wrt t

        d1T = ((sigma2t * ttm / 2.0 + sigma * sigma / 2.0) * sigma * np.sqrt(ttm) -
               ( np.log ( S0_local / K) + sigma * sigma * ttm / 2.0 ) * ( sigmat * ttm + sigma / (2.0 * np.sqrt (ttm) ) ) ) \
            / (sigma * sigma * ttm)

        d2T = (- (sigma2t * ttm / 2.0 + sigma * sigma / 2.0) * sigma * np.sqrt(ttm) -
               ( np.log ( S0_local / K) - sigma * sigma * ttm / 2.0 ) * ( sigmat * ttm + sigma / (2.0 * np.sqrt (ttm) ) ) ) \
            / (sigma * sigma * ttm)

        return S0_local * norm.pdf(d1) * d1T - K * norm.pdf(d2) * d2T

    # first derivative of (undiscounted) Black's call wrt K
    def call_future_K(self, fwd_date : datetime.date, S0 : float, K : float, ttm :float):
        """ Derivative of a call option in this parametrization wrt strike price K.

        """

        sigma_0, A, B, C, P, alphaC, alphaP = self._transform_from_jwss7(fwd_date)

        z = JWSS7Volatility.normalized_strike(S0, K, sigma_0, ttm)
        sigma = self.implied_vol(S0, K, ttm)
        S0_local = S0  # TODO: CHECK IF THIS IS REALLY NECESSARY
        d1 = (np.log(S0_local / K) + sigma * sigma * ttm / 2.0) / \
            (sigma * np.sqrt(ttm))
        d2 = d1 - sigma * np.sqrt(ttm)

        Xz = B * np.exp(C * z) + (1.0 - B) * np.exp(- P * z)

        sigmaK = A / (2.0 * Xz * K * np.sqrt (ttm) ) / np.sqrt ( 1.0 + A * np.log (Xz) ) * \
            (B * C * np.exp(C * z) - P * (1.0 - B) * np.exp(- P * z))

        d1K = ((- 1.0 / K + sigma * ttm * sigmaK) * sigma * np.sqrt(ttm) -
               ( np.log ( S0_local / K ) + sigma * sigma * ttm / 2.0 ) * np.sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        d2K = ((- 1.0 / K - sigma * ttm * sigmaK) * sigma * np.sqrt(ttm) -
               ( np.log ( S0_local / K ) - sigma * sigma * ttm / 2.0 ) * np.sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        return S0_local * norm.pdf(d1) * d1K - norm.cdf(d2) - K * norm.pdf(d2) * d2K

    # second derivative wrt K of the undiscounted call
    def call_future_KK(self, fwdDate, S0, K, ttm):


        jw7Params = self._transform_from_jwss7(fwdDate)
        sigma_0 = jw7Params['sigma_0']
        A       = jw7Params['A']
        B       = jw7Params['B']
        C       = jw7Params['C']
        P       = jw7Params['P']
        alphaC  = jw7Params['alphaC']
        alphaP  = jw7Params['alphaP']
        S0_local = S0

        z = JWSS7Volatility.normalized_strike(S0, K, sigma_0, ttm)
        sigma = self.implied_vol(S0, K, ttm)
        S0_local = S0  # TODO: CHECK IF THIS IS REALLY NECESSARY

        d1 = (np.log(S0 / K) + sigma * sigma * ttm / 2.0) / (sigma * np.sqrt(ttm))
        d2 = d1 - sigma * np.sqrt(ttm)
        Xz = B * np.exp(C * z) + (1.0 - B) * np.exp(- P * z)

        sigmaK = A / (2.0 * Xz * K * np.sqrt (ttm) ) / ( np.sqrt ( 1.0 + A * np.log (Xz) ) ) * \
            (B * C * np.exp(C * z) - P * (1.0 - B) * np.exp(- P * z))

        d1K = ((- 1.0 / K + sigma * ttm * sigmaK) * sigma * np.sqrt(ttm) -
               ( np.log ( S0 / K ) + sigma * sigma * ttm / 2.0 ) * np.sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        d2K = ((- 1.0 / K - sigma * ttm * sigmaK) * sigma * np.sqrt(ttm) -
               ( np.log ( S0 / K ) - sigma * sigma * ttm / 2.0 ) * np.sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        denomin = (sigma_0 * np.sqrt(ttm) * K * Xz * np.sqrt(1.0 + A * np.log(Xz)))
        BCexpr = (B * C * np.exp(C * z) - P * (1.0 - B) * np.exp(- P * z))

        sigmaKK = A / (2.0 * np.sqrt(ttm)) * (
            - A / (2.0 * denomin * K * Xz * (1.0 + A * np.log(Xz))) * BCexpr * BCexpr -
            BCexpr * BCexpr / (denomin * K * Xz) +
            (B * C * C * np.exp(C * z) + P * P * (1.0 - B) * np.exp(- P * z)) /
            (denomin * K) -
            BCexpr * sigma_0 * np.sqrt(ttm) / (denomin * K))

        d1KK = ((1.0 / (K * K) + sigmaK * sigmaK * ttm + sigma * ttm * sigmaKK) * sigma * np.sqrt(ttm) -
                ( np.log ( S0 / K ) + sigma * sigma * ttm  / 2.0 ) * np.sqrt (ttm) * sigmaKK ) / ( sigma * sigma * ttm) - \
                  \
            2.0 * ((- 1.0 / K + sigma * ttm * sigmaK) * sigma * np.sqrt(ttm) -
                   ( np.log ( S0_local / K ) + sigma * sigma * ttm  / 2.0 ) * np.sqrt (ttm) * sigmaK ) * sigma * ttm * sigmaK \
            / (sigma * sigma * sigma * sigma * ttm * ttm)

        d2KK = ((1.0 / (K * K) - sigmaK * sigmaK * ttm - sigma * ttm * sigmaKK) * sigma * np.sqrt(ttm) -
                ( np.log ( S0 / K ) - sigma * sigma * ttm  / 2.0 ) * np.sqrt (ttm) * sigmaKK ) / ( sigma * sigma * ttm) - \
            2.0 * ((- 1.0 / K - sigma * ttm * sigmaK) * sigma * np.sqrt(ttm) -
                   ( np.log ( S0_local / K ) - sigma * sigma * ttm  / 2.0 ) * np.sqrt (ttm) * sigmaK ) * sigma * ttm * sigmaK \
            / (sigma * sigma * sigma * sigma * ttm * ttm)

        return (S0 * norm.pdf(d1) * d1K * d1K + S0_local * norm.pdf(d1) * d1KK -
                2.0 * norm.pdf(d2) * d2K - K * norm.pdf(d2) * (-d2) * d2K * d2K - K * norm.pdf(d2) * d2KK)

    def skewed_distribution(self, fwdDate, S0, K, ttm):
        return 1. + self.call_future_K(fwdDate, S0, K, ttm)

    def inversion_skewed_cdf(
        self,
        fwdDate,
        S0,
        ttm,
        quantile,
        lb=0.01,
        ub=np.inf,
        maxIter=150,
    ):
        """ Function finds K such that: skewed_cdf_analy(K, quantile) = 0

        :param maxIter: maximum number of iterations for the iteration solver.
        """

        try:
            return NLP(
                lambda K: (self.skewed_distribution(fwdDate, S0, K, ttm) - quantile)**2,
                S0,
                lb=lb,
                ub=ub,
                maxIter=maxIter,
            ).solve(self.__class__.SCIPY_SOLVER).xf[0]
        except Exception as e:
            raise JWSS7Exception(
                'Couldnt compute inversion_skewed_cdf: {0}'.format(str(e))
            )


class JWSS7VolatilityDisplay(JWSS7Volatility, VolatilityDrawMixin):

    def _strike_range(self, S0: float, sigma_0: float, ttm: float) -> Tuple[float, float]:
        # solves the equation: N(log(K/S0)/sigma_0/sqrt(ttm)) = -0.5
        # S0_factor = S0 * np.exp(sigma_0 * np.sqrt(ttm))
        # lower_bound = S0_factor * norm.ppf(-0.5)
        # upper_bound = S0_factor * norm.ppf(0.5)

        lower_bound = S0 / 2
        upper_bound = S0 * 2

        return (lower_bound, upper_bound)

    def _plot_data(self, nb_points: int = 50):
        """

        :param nb_points: number of points to display in the graph.
        """

        S0 = self.slider_S0.get()
        sigma_0 = self.slider_sig.get()
        A = self.slider_A.get()
        B = self.slider_B.get()
        C = self.slider_C.get()
        P = self.slider_P.get()
        alpha_C = self.slider_alpha_C.get()
        alpha_P = self.slider_alpha_P.get()
        ttm = self.slider_ttm.get()

        K_loglinear_lower, K_loglinear_upper = self._strike_range(S0, sigma_0, ttm)
        K_loglinear = np.linspace(K_loglinear_lower, K_loglinear_upper, nb_points)
        # log_strike = np.array([JWSS7Volatility.normalized_strike(S0, K, sigma_0, ttm) for K in K_v])

        jwss7_params = (sigma_0, A, B, C, P, alpha_C, alpha_P)
        sigmas = np.array([
            self._vol_from_jwss7(S0, K, ttm, jwss7_params)
            for K in K_loglinear
        ])

        self._ax.clear()
        self._ax.plot(K_loglinear, sigmas)
        self._ax.set_title(f"Plot: S0={S0:.2f}, sigma_0={sigma_0:.2f}")
        self._ax.set_xlabel("log-moneyness")
        self._ax.set_ylabel("vol")
        self._ax.grid(True)
        self._canvas.draw()

    def _update_plot(self, event):
        self._plot_data()

    def _create_variables(self):
        self.slider_S0 = tk.DoubleVar(value=100.)
        self.slider_sig = tk.DoubleVar(value=0.2)
        self.slider_A = tk.DoubleVar(value=1.)
        self.slider_B = tk.DoubleVar(value=0.5)
        self.slider_C = tk.DoubleVar(value=1.)
        self.slider_P = tk.DoubleVar(value=0.2)
        self.slider_alpha_C = tk.DoubleVar(value=1.)
        self.slider_alpha_P = tk.DoubleVar(value=1.)
        self.slider_ttm = tk.DoubleVar(value=0.1)

    def _create_sliders(self):
        slider_S0 = ttk.Scale(
            self.root,
            from_=0.1,
            to=10.0,
            # resolution=0.1,
            # label='S0',
            orient="horizontal",
            variable=self.slider_S0,
            command=self._update_plot,
        )
        slider_S0.grid(row=0, column=1)
        label_S0 = ttk.Label(text='S0')
        label_S0.grid(row=0, column=2)

        slider_sig = ttk.Scale(
            self.root,
            from_=0.05,
            to=0.8,
            # resolution=0.05,
            # label='sigma_0',
            orient="horizontal",
            variable=self.slider_sig,
            command=self._update_plot,
        )
        slider_sig.grid(row=1, column=1)
        label_sig = ttk.Label(text='sigma_0')
        label_sig.grid(row=1, column=2)

        slider_A = ttk.Scale(
            self.root,
            from_=0.0,
            to=5,
            # resolution=0.25,
            # label='A',
            orient="horizontal",
            variable=self.slider_A,
            command=self._update_plot,
        )
        slider_A.grid(row=2, column=1)
        label_A = ttk.Label(text='A')
        label_A.grid(row=2, column=2)

        slider_B = ttk.Scale(
            self.root,
            from_=0.0,
            to=1.,
            # resolution=0.05,
            # label='B',
            orient="horizontal",
            variable=self.slider_B,
            command=self._update_plot,
        )
        slider_B.grid(row=3, column=1)
        label_B = ttk.Label(text='B')
        label_B.grid(row=3, column=2)

        slider_C = ttk.Scale(
            self.root,
            from_=0.0,
            to=5.,
            # resolution=0.2,
            # label='C',
            orient="horizontal",
            variable=self.slider_C,
            command=self._update_plot,
        )
        slider_C.grid(row=4, column=1)
        label_C = ttk.Label(text='C')
        label_C.grid(row=4, column=2)

        slider_P = ttk.Scale(
            self.root,
            from_=0.0,
            to=5.,
            # resolution=0.2,
            # label='P',
            orient="horizontal",
            variable=self.slider_P,
            command=self._update_plot,
        )
        slider_P.grid(row=5, column=1)
        label_P = ttk.Label(text='P')
        label_P.grid(row=5, column=2)

        slider_alpha_C = ttk.Scale(
            self.root,
            from_=0.0,
            to=5.,
            # resolution=0.2,
            # label='alpha_C',
            orient="horizontal",
            variable=self.slider_alpha_C,
            command=self._update_plot,
        )
        slider_alpha_C.grid(row=6, column=1)
        label_alpha_C = ttk.Label(text='alpha_C')
        label_alpha_C.grid(row=6, column=2)

        slider_alpha_P = ttk.Scale(
            self.root,
            from_=0.0,
            to=5.,
            # resolution=0.2,
            # label='alpha_P',
            orient="horizontal",
            variable=self.slider_alpha_P,
            command=self._update_plot,
        )
        slider_alpha_P.grid(row=7, column=1)
        label_alpha_P = ttk.Label(text='alpha_P')
        label_alpha_P.grid(row=7, column=2)

        slider_ttm = ttk.Scale(
            self.root,
            from_=0.0,
            to=5.,
            # resolution=0.2,
            # label='ttm',
            orient="horizontal",
            variable=self.slider_ttm,
            command=self._update_plot,
        )
        slider_ttm.grid(row=8, column=1)
        label_ttm = ttk.Label(text='ttm')
        label_ttm.grid(row=8, column=2)

    def create_plot(self):
        "Creates a plot and plots initial data."

        self.root = tk.Tk()
        self._fig, self._ax = plt.subplots(figsize=(6, 4))
        self._canvas = FigureCanvasTkAgg(self._fig, master=self.root)
        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.grid(column=0, rowspan=8)

        self._create_variables()
        self._create_sliders()

        self._plot_data()
        self.root.mainloop()
