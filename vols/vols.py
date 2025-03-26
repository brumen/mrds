# Implements base volatility class

import logging
import datetime
import scipy
import scipy.stats
import scipy.interpolate  # spline package
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict, Union
from scipy.optimize import minimize, Bounds
from scipy.interpolate import splev, splrep
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import mrds.ds as ds
from mrds.pricers.pricers import black_greeks
from mrds.forward_curve import FwdCurve

mpl.use('TkAgg')

logger = logging.Logger(__name__)


class VolatilityException(Exception):
    pass


class Volatility:
    """ Base volatility class.
    """

    SOLVER = 'scipy_cobyla'

    def __init__(
            self,
            com_name: str,
            mkt_date: datetime.date,
            fwd_params: FwdCurve,
            vol_params: Dict[datetime.date, List]
    ):
        """ Generic class for the volatility object. Most generic way of computing the volatility.

        :param com_name: name of the commodity to consider
        :param mkt_date: market date
        :param fwd_params: parameters about the forward curve, in the form of FwdCurve object
        :param vol_params: dictionary, where keys are volatility dates, and values are tuples of parameters (ATM, ....)
        """

        self.mkt_date = mkt_date
        self.com_name = com_name
        self._fwd_params = fwd_params
        self._vol_params = vol_params

    @classmethod
    def from_db(cls, com_name: str, mkt_date : datetime.date):
        """ Reads the forward and vol curve from external source.

        :param com_name: name of the commodity one wants, e.g. 'WTI', ...
        :param mkt_date: for which market date the vol is needed
        """

        _, vol_params = ds.get_vol_curve(com_name, mkt_date)

        return cls( com_name
                  , mkt_date
                  , fwd_params = ds.get_forward_curve(com_name, mkt_date)
                  , vol_params = vol_params )

    @property
    def vol_dates(self) -> List[datetime.date]:
        """ Volatility dates.
        """

        raise NotImplementedError('_vol_dates is not implemented.')

    def _vol_for_date(self, date_: datetime.date) -> float:
        """ Gets the volatility for a particular date.

        :param date_: date for which the volatility is obtained.
        """

        nearest_vol = sum([
            vol_spine_date < date_
            for vol_spine_date in self.vol_dates
        ])

        return 1.  # TODO: FIX THIS HERE!!!

    @staticmethod
    def normalized_strike(
            S0: np.double,
            K_v: np.array,
            sigma: np.double,
            ttm_v: np.array,
    ) -> np.array:
        """ Vectorized form of normalized log(S0/K)/(sigma * sqrt(T))

        :param S0: initial stock (forward) price
        :param K_v: strike price
        :param sigma: ATM volatility of the stock price
        :param ttm_v: time to maturity
        :returns: normalized strike of the option
        """

        return np.log(K_v / S0) / (sigma * np.sqrt(ttm_v))

    @staticmethod
    def normalized_strike_inv(
            delta_v: np.array,
            sigma: np.double,
            ttm: np.double,
    ) -> np.array:
        """ Inverse of the normalized strike.

        :param delta_v: vector of delta
        :param sigma: volatility of stock/forward
        :param ttm: time to maturity
        """

        return np.exp(scipy.stats.norm.ppf(delta_v) * sigma * np.sqrt(np.double(ttm)) - 0.5 * sigma ** 2 * ttm)

    def implied_vol(self, fwd_date : datetime.date, K : float, ttm : float) -> float:
        """ Implied vol needs to be implemented in the subclass.
            Interface is: Implied volatility as a function of S0, K, ttm.

        :param fwd_date: date of the forward contract for which the volatility is required.
        :param K: strike price.
        :param ttm: time to maturity
        """

        raise VolatilityException('Method implied_vol not implemented in Volatility class.')

    def option_price(self, F: float, K: float, maturity : datetime.date) -> float:

        return black_greeks(F, K, 0.01, self.implied_vol(TODO, K, maturity), )

    def delta(self, fwd_date : datetime.date, K : float, ttm : float) -> float:
        """  Computes the delta of the volatility.

        :param fwd_date: forward date for which the volatility is required.
        :param K: strike at which the delta is requested.
        :param ttm: time to maturity.
        """

        raise VolatilityException('Method delta not implemented in Volatility class.')

    def time_to_maturity(self, fwd_date: datetime.date, dcf = 365.25):

        return (fwd_date - self.mkt_date).days / dcf  # time to maturity

    # TODO: CHECK IF THIS IS ACTUALLY NEEDED!
    def black_simple(
            self,
            fwd_date: datetime.date,
            strike: float,
            dcf=365.25,
            df=1.,
    ):
        """ Simple version of the black volatility.

        :param fwd_date: black vol for that date
        :param strike: strike value
        :param dcf: day-count fraction.
        :param df: discount factor TODO: CHECK IF THIS IS NEEDED
        """

        ttm = self.time_to_maturity(fwd_date)

        return black_greeks( self._fwd_params.fwd_value(fwd_date)
                           , strike
                           , -np.log(df) / ttm
                           , self.implied_vol(fwd_date, strike, ttm)
                           , ttm)

    def call_future_K(self
                      , fwd_date : datetime.date
                      , K        : float
                      , ttm      : float
                      , delta_K = 0.01
                      , df = 1.):
        """ Computes the derivative of the call option value wrt strike price K - used for computing the skew
            distribution (dC/dK).
        """

        S0 = self._fwd_params.fwd_value(fwd_date)

        pr_0 = black_greeks( S0
                           , K
                           , -np.log(df) / ttm
                           , self.implied_vol(S0, K, ttm)
                           , ttm
                           , 0)

        pr_delta = black_greeks( S0
                               , K + delta_K
                               , -np.log(df) / ttm
                               , self.implied_vol(S0, K + delta_K, ttm)
                               , ttm
                               , 0 )

        return (pr_delta - pr_0) / delta_K

    def skewed_distribution(self, fwd_date : datetime.date, K : float, delta_K : float, ttm : float) -> float:
        """ Gives the CDF of a skewed distribution using UN-discounted call values
        """

        return 1.0 + self.call_future_K(fwd_date, K, ttm)

    def skewed_cdf_analy(self, K, quantile):
        return (self.skewed_distribution(K, ttm) - quantile)**2

    def inversion_skewed_cdf( self
                            , quantile : float
                            , ttm      : float
                              , maxIter = 150 ) -> float:
        """ Finds K such that: skewed_cdf_analy(K, quantile) = 0

        :param quantile: which quantile of the distribution you want to obtain.
        :param ttm: time to maturity
        :param maxIter: maximum number of iterations of the NLP solver.
        """

        try:
            nlp_solution = minimize( lambda K: self.skewed_cdf_analy(K, quantile)
                                   , S0
                                   , bounds = Bounds([0.001], [np.inf])
                                   , )

        except Exception as e:
            raise VolatilityException('Unable to invert the skewed cdf in inversion_skewed_cdf: {0}'.format(str(e)))

        return nlp_solution.xf[0]

    def local_vol_generic(self, K, T, dT, dK):
        """
        Generic, fairly imprecise computation of local vol
        based on difference methods
        LV^2 = 2 * DC/DT / K^2 / D^2C/DK^2

        :param dT:
        """

        sigma = self.impl_vol(K, T)  # CORRECT THIS HERE
        up_part = black_greeks(S_0, K, r, sigma, T, 0)[4]  # dC/dT
        down_part = (black_greeks(S_0, K + dK, r, sigma, T, 0)[1] -
                     black_greeks(S_0, K, r, sigma, T, 0)[1]) / dK

        return 2. * up_part / down_part / K**2

    def implied_surf( self
                    , fwd_date : datetime.date
                    , ttm_grid : List[float]
                    , K_grid   : List[float]) -> np.ndarray:
        """ Generates the implied vol surface for the following parameters:

        :param fwd: number of the forward contract
        :param ttm_grid: grid of expiry times
        :param K_grid: list of strikes
        """

        return np.array ( [ [self.implied_vol(fwd_date, K, ttm)
                             for K in K_grid]
                            for ttm in ttm_grid ] )

    def local_vol_surf(self, ttm_grid : List[float], K_grid : List[float], dT : float, dK : float) -> np.ndarray:
        """  Local-vol surface for ttm_grid, K_grid

        :param ttm_grid: time to maturity grid
        """

        return np.array ( [ [self.local_vol(K, ttm, dT, dK)
                             for K in K_grid]
                            for ttm in ttm_grid ] )


class VolatilityDrawMixin:
    """ Mixin for drawing the volatility surface.
    """

    def draw_surface( self
                      , fwd_date  : datetime.date
                      , S_min_max : Tuple[float, float, float]
                      , t_min_max : Tuple[float, float, float] ):
        """ Draws the implied/local vol surface from
        [Sd, Su] x [Tmin, Tmax] with steps Sstep, Tstep

        :param S_min_max: tuple of forward grid (low bound, high bound, step)
        :param t_min_max: tuple of time grid (low_bound, high_bound, step)
        """

        t_min, t_max, t_step = t_min_max
        S_min, S_max, S_step = S_min_max

        K_grid   = np.arange(S_min, S_max, S_step)
        K_len    = len(K_grid)
        ttm_grid = np.arange(t_min, t_max, t_step)
        ttm_len  = len(ttm_grid)

        K_mesh, ttm_mesh = np.meshgrid(K_grid, ttm_grid)
        vol_surf         = np.empty_like(K_mesh)

        for K_idx, K in enumerate(K_grid):
            for ttm_idx, ttm in enumerate(ttm_grid):
                vol_surf[ttm_idx, K_idx] = self.implied_vol(fwd_date, K, ttm)

        # plot machinery
        # root = tk.Tk()
        fig  = plt.figure()
        #dataPlot_canvas = FigureCanvasTkAgg(fig, master=root)
        #dataPlot_canvas.get_tk_widget().grid(row=0, column=0, rowspan=8)
        ax = Axes3D(fig)  # plot it
        ax.plot_surface(K_mesh, ttm_mesh, vol_surf)
        plt.show()

    def _K_ttm_grid( self
                   , S_min_max : Tuple[float, float, float]
                   , t_min_max : Tuple[float, float, float] ) -> Tuple[np.ndarray, np.ndarray]:
        t_min, t_max, t_step = t_min_max
        S_min, S_max, S_step = S_min_max

        return np.arange(S_min, S_max, S_step), np.arange(t_min, t_max, t_step)


class ATMFVolatility(Volatility):
    """ ATM volatility
    """

    INTERPOLATION_DEGREE = 1

    def __init__(
            self,
            com_name: str,
            mkt_date: datetime.date,
            fwd_params,
            vol_params,
            dcf=365.25,
    ):
        """ JWSS7 volatility init. the same as the volatility init, w/ some specific properties.
        All parameters are the same as in Volatility class, except for the following:

        :param dcf: day-count factor,
        """

        super().__init__(com_name, mkt_date, fwd_params, vol_params)
        self._dcf = dcf

        self._atm_vol_curve_interp = None

    @property
    def vol_dates(self) -> List[datetime.date]:
        """ Returns the volatility spine points, building blocks of volatility structure.

        :returns: volatility dates as a model input.
        """

        return list(self._vol_params.keys())

    @property
    def _atm_vol_curve(self):
        """ Constructs the ATM vol curve

        Returns the object returned from splrep, to be used for splev.
        """

        if self._atm_vol_curve_interp:
            return self._atm_vol_curve_interp

        vol_dates = [(x - self.mkt_date).days / self._dcf for x in self.vol_dates]

        vol_dates_values = sorted(
            zip(vol_dates, self._vol_params.values()),
            key=lambda vol_date_val: vol_date_val[0]
        )

        self._atm_vol_curve_interp = splrep(
            [x[0] for x in vol_dates_values],
            [x[1] for x in vol_dates_values],
            k=self.INTERPOLATION_DEGREE
        )

        return self._atm_vol_curve_interp

    def atm_vol(self, fwd_date : Union[datetime.date, List[datetime.date]]) -> Union[float, List[float]]:
        """ Returns the atm forward for the fwd date fwd_date.

        :param fwd_date: forward date for which the ATM is constructed.
        """

        to_return = splev((fwd_date - self.mkt_date).days / self._dcf, self._atm_vol_curve )

        if isinstance(fwd_date, datetime.date):
            return float(to_return)

        return to_return

    def implied_vol(self, fwd_date : datetime.date, K : float, ttm : float) -> float:
        """ Implied vol for ATM vol is simply atm vol.

        :param fwd_date: date of the forward contract for which the volatility is required.
        :param K: strike price.
        :param ttm: time to maturity
        :returns: implied volatility, which equals the atm volatility.
        """

        return self.atm_vol(fwd_date)
