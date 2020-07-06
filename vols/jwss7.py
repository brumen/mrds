# JWSS7 volatility structure
import datetime
import numpy as np
import logging

from scipy.stats import norm

from typing  import List, Dict, Tuple, Union
from tkinter import Scale, Button, HORIZONTAL
from scipy.interpolate import splev, splrep

from mrds.ds            import get_vol_curve
from mrds.forward_curve import FwdCurve
from mrds.vols.vols     import Volatility, VolatilityDrawMixin

logger = logging.getLogger(__name__)


class JWSS7Exception(Exception):
    pass


class JWSS7Volatility(Volatility):
    """ Jump-wing parametrization.
    """

    INTERPOLATION_DEGREE = 2
    SCIPY_SOLVER = 'scipy_cobyla'

    def __init__( self
                , com_name : str
                , mkt_date : datetime.date
                , fwd_params
                , vol_params
                , dcf = 365.25 ):
        """ JWSS7 volatility init. the same as the volatility init, w/ some specific properties.
        All parameters are the same as in Volatility class, except for the following:

        :param dcf: day-count factor,
        """

        super().__init__(com_name, mkt_date, fwd_params, vol_params)
        self.__dcf = dcf

        self.__atm_vol_curve_interp = None

    @property
    def vol_dates(self) -> List[datetime.date]:
        """ Returns the curve spine points, i.e. the points on the curve from which the curve is interpolated.
        """

        return self._vol_params.keys()

    @property
    def __atm_vol_curve(self):
        """ Constructs the ATM vol curve

        Returns the object returned from splrep, to be used for splev.
        """

        if self.__atm_vol_curve_interp:
            return self.__atm_vol_curve_interp

        vol_dates = [(x - self.mkt_date).days / self.__dcf for x in self.vol_dates]
        atm_vols  = [x[0] for x in self._vol_params.values()]

        vol_dates_values = sorted(zip(vol_dates, atm_vols), key=lambda vol_date_val: vol_date_val[0])

        self.__atm_vol_curve_interp = splrep( [x[0] for x in vol_dates_values]
                                            , [x[1] for x in vol_dates_values]
                                            , k=self.INTERPOLATION_DEGREE )

        return self.__atm_vol_curve_interp

    def atm_vol(self, fwd_date : Union[datetime.date, List[datetime.date]]) -> Union[float, List[float]]:
        """ Returns the atm forward for the fwd date fwd_date.

        :param fwd_date: forward date for which the ATM is constructed.
        """

        to_return = splev((fwd_date - self.mkt_date).days / self.__dcf, self.__atm_vol_curve )

        if isinstance(fwd_date, datetime.date):
            return float(to_return)

        return to_return

    @classmethod
    def from_db(cls, com_name : str, mkt_date : datetime.date):
        vol_type, vol_params = get_vol_curve(com_name, mkt_date)
        if vol_type != 'JWSS7':
            raise RuntimeError('Fetching the wrong curve. {0} has type {1}'.format(com_name, vol_type))

        return cls( com_name
                  , mkt_date
                  , fwd_params = FwdCurve.from_db(mkt_date, com_name)  # ds.get_forward_curve(com_name, mkt_date)
                  , vol_params = vol_params)  # TODO: FIX THE LINE BELOW.
               #vol_params = cls._transform_from_jwss7(vol_params) )

    @staticmethod
    def _transform_from_jwss7( vol_curve : Dict[datetime.date, List]) -> Dict[datetime.date, Tuple]:
        """ Returns jw7 parametrization from jwss7 for particular fwd date.

        vol_params in Jwss7: [S0, atm, skew, smile, putslope, putbend, callslope, callbend]
        vol_params in jw7  : [S0, atm, A   , B    , C       , P      , alphaC   , alphaP  ]
        """

        transformed_curve = {}
        for fwd_vol_date, vol_params_for_date in vol_curve.items():
            sigma_0, skew, smile, put_slope, put_bend, call_slope, call_bend = vol_params_for_date
            B = (2. * skew + put_slope) / (put_slope + call_slope)
            A = 0.5 * B * (1. - B) * (call_slope + put_slope)**2 / (smile + skew**2)

            # in the form of sigma_0, A, B, C, P, alphaC, alphaP
            transformed_curve[fwd_vol_date] = (sigma_0, A, B, call_slope / A, put_slope / A, call_bend, put_bend )

            return transformed_curve

    def _interpolate_params_for_fwd_date(self, fwd_date : datetime.date) -> datetime.date:
        """ Interpolate parameters for forward date fwd_date.
        In this case I just select the next larger date, or if not, the largest date in the self._vol_params

        :param fwd_date: forward date for which parameters are requested.
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
            selected_date = max(input_dates)

        return selected_date

    def _vol_compute(self, fwd_date : datetime.date, normalized_strike : float) -> float:
        """ Computes the volatility given the following parameters:

        :param fwd_date: forward date on the vol curve.
        :param normalized_strike: normalized strike
        """

        sigma_0, A, B, C, P, alpha_C, alpha_P = self._vol_params[self._interpolate_params_for_fwd_date(fwd_date)]
        z = normalized_strike  # abbreviation, for simplicity

        return sigma_0 * np.sqrt(1. + A * np.log(B * np.exp(C * (z / (1.0 + z * z) ** (alpha_C/2))) +
                                          (1. - B) * np.exp(- P * (z / (1.0 + z * z) ** (alpha_P/2)))))

    def implied_vol(self, fwd_date : Union[datetime.date, int], strike : float, ttm : float) -> float:
        """ Implied vol for the fwd_date.

        :param fwd_date: date for which the volatility is to be computed.
        :param strike: strike price
        :param ttm: time to maturity
        """

        return self._vol_compute( fwd_date
                                , JWSS7Volatility.normalized_strike(self._fwd_params.fwd_value(fwd_date)
                                                                    , np.array([strike])
                                                                    , self._vol_params[self._interpolate_params_for_fwd_date(fwd_date)][0]  # atm vol is the first element
                                                                    , ttm)[0] )

    def local_vol(self, fwd_date : datetime.date, T : float, S: float, ttm : float) -> float:
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

    def inversion_skewed_cdf( self
                            , fwdDate
                            , S0
                            , ttm
                            , quantile
                            , lb = 0.01
                            , ub = np.inf
                            , maxIter = 150):
        """ Function finds K such that: skewed_cdf_analy(K, quantile) = 0

        :param maxIter: maximum number of iterations for the iteration solver.
        """

        try:
            return NLP( lambda K: (self.skewed_distribution(fwdDate, S0, K, ttm) - quantile)**2
                      , S0
                      , lb      = lb
                      , ub      = ub
                      , maxIter = maxIter).solve(self.__class__.SCIPY_SOLVER).xf[0]
        except Exception as e:
            raise JWSS7Exception('Couldnt compute inversion_skewed_cdf: {0}'.format(str(e)))


class JWSS7VolatilityDisplay(JWSS7Volatility, VolatilityDrawMixin):

    def _draw_buttons(self, fwd, root, ax, dataPlot_canvas):
        """
        # root ... Tk root
        # ax ... Axes3D object
        # dataPlot_canvas ... canvas object

        """

        fct_update = lambda cc: self.update_graph( fwd
                                                 , model
                                                 , [c1.get(), c2.get(), c3.get(), c4.get(), c5.get(), c6.get(), c7.get(), c8.get()]
                                                 , ax
                                                 , dataPlot_canvas )

        # parameter tk.SCALEs
        c1 = Scale(root, from_=80.0, to=120.0, resolution=0.1, label="S0", orient=HORIZONTAL,command=fct_update)
        c2 = Scale(root, from_=0.05, to=0.8, resolution=0.05, label="sig", orient=HORIZONTAL,command=fct_update)
        c3 = Scale(root, from_=0.0, to=5.0, resolution=0.25, label="A", orient=HORIZONTAL,command=fct_update)
        c4 = Scale(root, from_=0.0, to=1.0, resolution=0.05, label="B", orient=HORIZONTAL,command=fct_update)
        c5 = Scale(root, from_=0.0, to=5.0, resolution=0.2, label="C", orient=HORIZONTAL,command=fct_update)
        c6 = Scale(root, from_=0.0, to=5.0, resolution=0.2, label="P", orient=HORIZONTAL,command=fct_update)
        c7 = Scale(root, from_=0.0, to=5.0, resolution=0.2, label="alpha_C", orient=HORIZONTAL,command=fct_update)
        c8 = Scale(root, from_=0.0, to=5.0, resolution=0.2, label="alpha_P", orient=HORIZONTAL,command=fct_update)

        c1.grid(row=0, column=1)
        c2.grid(row=1, column=1)
        c3.grid(row=2, column=1)
        c4.grid(row=3, column=1)
        c5.grid(row=4, column=1)
        c6.grid(row=5, column=1)
        c7.grid(row=6, column=1)
        c8.grid(row=7, column=1)

        # replot button
        b1 = Button(root, text="replot", command=lambda: update_graph( fwd
                                                                     , model
                                                                     , [c1.get(), c2.get(), c3.get(), c4.get(), c5.get(), c6.get(), c7.get(), c8.get()]
                                                                     , ax
                                                                     , dataPlot_canvas )).grid(row=8, column=0)

        dataPlot_canvas.show()
        root.mainloop()
