# Implements base volatility class

import config
import logging

import datetime

import numpy as np
from numpy import double, log, exp, sqrt

from typing import List, Tuple, Dict

import scipy
import scipy.stats
import scipy.interpolate  # spline package
from openopt import NLP
import matplotlib as mpl
mpl.use('TkAgg')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

if config.CUDA_PRESENT:
    import pycuda.autoinit  # this needs to be here.
    from pycuda.gpuarray import to_gpu
    from pycuda.compiler import SourceModule

import ds
from pricers.pricers import black_greeks


logger = logging.Logger(__name__)


def extract_param_matrix(date_, fwd_name, vol_name, nb_fwds_taken=-1):
    """
    Array with forwards and vol params

    """

    fvm = ds.read_data_matched_tenors(date_, fwd_name, vol_name)
    nb_fwds = len(fvm['fwd_curve']) if nb_fwds_taken == -1 else nb_fwds_taken

    fwd_curve = fvm['fwd_curve'][:nb_fwds]
    option_tenors_dt = fvm['option_tenors_dt'][:nb_fwds]
    vol_surface_params = fvm['vol_surface_params'][:nb_fwds]
    fv_array = np.append(np.array(fwd_curve).reshape((nb_fwds, 1)),
                         np.array(vol_surface_params), axis=1)

    return fv_array, option_tenors_dt


class VolatilityException(Exception):
    pass


class Volatility:
    """ Base volatility class.
    """

    SOLVER = 'scipy_cobyla'

    def __init__(self
                 , com_name : str
                 , mkt_date : datetime.date
                 , fwd_params = None
                 , vol_params = None):
        """ Generic class for the volatility object. Most generic way of computing the volatility.

        :param com_name: name of the commodity to consider
        :param mkt_date: market date
        :param fwd_params: parameters about the forward curve
        :param vol_params: ???
        """

        self.mkt_date    = mkt_date
        self.com_name    = com_name
        self._fwd_params = fwd_params
        self._vol_params = vol_params

    @classmethod
    def from_db(cls, com_name : str, mkt_date : datetime.date):
        """ Reads the forward and vol curve from external source.

        :param com_name: name of the commodity one wants, e.g. 'WTI', ...
        :param mkt_date: for which market date the vol is needed
        """

        return cls( com_name
                  , mkt_date
                  , fwd_params = ds.get_forward_curve(com_name, mkt_date)
                  , vol_params = ds.get_vol_curve(com_name, mkt_date) )

    @property
    def _vol_dates(self) -> List[datetime.date]:
        """ Volatility dates.
        """

        raise NotImplementedError('_vol_dates is not implemented.')

    def _vol_for_date(self, date_ : datetime.date) -> float:
        """ Gets the volatility for a particular date.

        :param date_: date for which the volatility is obtained.
        """

        nearest_vol = sum([fwdDateInCurve < date_ for fwdDateInCurve in self._vol_dates])

        return 1.  # TODO: FIX THIS HERE!!!

    @staticmethod
    def normalized_strike( S0   : np.double
                         , K_v  : np.array
                         , sigma: np.double
                         , ttm_v: np.array) -> np.array:
        """ Vectorized form of normalized log(S0/K)/(sigma * sqrt(T))

        :param S0: initial stock (forward) price
        :param K_v: strike price
        :param sigma: ATM volatility of the stock price
        :param ttm_v: time to maturity
        :returns: normalized strike of the option
        """

        return log(K_v.reshape(1, len(K_v)) / S0) / (sigma * np.sqrt(ttm_v))

    @staticmethod
    def normalized_strike_inv(delta_v: np.array
                              , sigma: np.double
                              , ttm: np.double) -> np.array:
        """ Inverse of the normalized strike.

        :param delta_v: vector of delta
        :param sigma: volatility of stock/forward
        :param ttm: time to maturity
        """

        return exp(scipy.stats.norm.ppf(delta_v) * sigma * np.sqrt(np.double(ttm)) - 0.5 * sigma ** 2 * ttm)

    def implied_vol(self, fwd_date : datetime.date, K : float, ttm : float) -> float:
        """ Implied vol needs to be implemented in the subclass.
            Interface is: Implied volatility as a function of S0, K, ttm.

        :param fwd_date: date of the forward contract for which the volatility is required.
        :param K: strike price.
        :param ttm: time to maturity
        """

        raise VolatilityException('Method implied_vol not implemented in Volatility class.')

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
    def black_simple( self
                    , fwd_date : datetime.date
                    , strike   : float
                    , dcf = 365.25
                    , df = 1.):
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
                           , -log(df) / double(ttm)
                           , self.implied_vol(S0, K, ttm)
                           , ttm
                           , 0)

        pr_delta = black_greeks( S0
                               , K + delta_K
                               , -log(df) / double(ttm)
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
                            , maxIter = 150
                            , iprint  = -9 ):
        """ Finds K such that: skewed_cdf_analy(K, quantile) = 0

        :param quantile: which quantile of the distribution you want to obtain.
        :param ttm: time to maturity
        :param
        """

        return NLP( lambda K: self.skewed_cdf_analy(K, quantile)
                  , S0
                  , lb      = 0.001
                  , ub      = np.inf
                  , maxIter = maxIter
                  , iprint  = iprint ).solve(self.__class__.SOLVER).xf[0]

    def local_vol_generic(self, K, T, dT, dK):
        """
        Generic, fairly imprecise computation of local vol
        based on difference methods
        LV^2 = 2 * DC/DT / K^2 / D^2C/DK^2

        :params dT:
        """

        sigma = self.impl_vol(K, T)  # CORRECT THIS HERE
        up_part = black_greeks(S_0, K, r, sigma, T, 0)[4]  # dC/dT
        down_part = (black_greeks(S_0, K + dK, r, sigma, T, 0)[1] -
                     black_greeks(S_0, K, r, sigma, T, 0)[1]) / dK

        return 2. * up_part / down_part / K**2

    def implied_surf(self, fwd, ttm_grid : List[float], K_grid : List[float]) -> np.ndarray:
        """ Generates the implied vol surface for the following parameters:

        :param fwd: number of the forward contract
        :param ttm_grid: grid of expiry times
        :param K_grid: list of strikes
        """

        return np.array ( [ [self.implied_vol(fwd, K, ttm)
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
                    , t_min_max : Tuple[float, float, float]
                    , impl_local_ind = 'impl'
                    , cuda_ind       = False ):
        """ Draws the implied/local vol surface from
        [Sd, Su] x [Tmin, Tmax] with steps Sstep, Tstep

        :param fwd_date: forward to be considered.
        :param S_min_max: tuple of forward grid (low bound, high bound, step)
        :param
          vol = vol. parametrization: jw7, sabr, c0c1c2, ratiovol
          impl_local_ind ... indicator for implied or local volatility
          cuda_ind ... should the computations be performed on the cuda
          fwd_idx ... forward index we are trying to plot
        """

        t_min, t_max, t_step = t_min_max
        S_min, S_max, S_step = S_min_max

        K_grid   = np.arange(S_min, S_max, S_step)
        ttm_grid = np.arange(t_min, t_max, t_step)

        if cuda_ind:
            K_grid_d = to_gpu(K_grid).astype(np.float32)  # K, ttm grid on device
            ttm_grid_d = to_gpu(ttm_grid).astype(np.float32)

        K_mesh, ttm_mesh = np.meshgrid(K_grid, ttm_grid)
        impl_surf = np.zeros((len(ttm_grid), len(K_grid)))
        lv_surf = np.zeros((len(ttm_grid), len(K_grid)))

        c = model.get_params(fwd_idx)  # constructs the param array

        self.lv_impl_surf()

        root = tk.Tk()  # root canvas
        # plot market vols as initial
        fig = plt.figure()
        # construct canvas
        dataPlot_canvas = FigureCanvasTkAgg(fig, master=root)
        dataPlot_canvas.get_tk_widget().grid(row=0, column=0, rowspan=8)
        ax = Axes3D(fig)  # plot it

        if impl_local_ind == 'impl':  # cuda not important, so not implemented
            impl_surf = model.gen_impl_surf(
                fwd_idx,
                ttm_grid,
                K_grid)  # impl. vol surface
            ax.plot_surface(ttm_mesh, K_mesh, impl_surf)  # initial impl. plot
        else:
            lv_surf = gen_lv_surf()  # updating the local vol surface
            ax.plot_surface(ttm_mesh, K_mesh, lv_surf)  # initial lv. plot

        # draw graphs
        self._draw_buttons()

    def _K_ttm_grid( self
                   , S_min_max : Tuple[float, float, float]
                   , t_min_max : Tuple[float, float, float] ) -> Tuple[np.ndarray, np.ndarray]:
        t_min, t_max, t_step = t_min_max
        S_min, S_max, S_step = S_min_max

        return np.arange(S_min, S_max, S_step), np.arange(t_min, t_max, t_step)

    def _K_ttm_grid_cuda( self
                        , S_min_max : Tuple[float, float, float]
                        , t_min_max : Tuple[float, float, float] ) -> Tuple[np.ndarray, np.ndarray]:
        K_grid, ttm_grid = self._K_ttm_grid(S_min_max, t_min_max)

        return (to_gpu(K_grid), to_gpu(ttm_grid))

    def lv_impl_surf(self):

        K_size = len(K_grid)
        ttm_size = len(ttm_grid)

        impl_surf_d = to_gpu(impl_surf).astype(np.float32)  # impl. surf on cuda
        lv_surf_d = to_gpu(lv_surf).astype(np.float32)  # lv surf. on cuda

        with open(config.work_dir + "imp_vol_kern.cu") as impVolKernFile:
            imp_vol_mod    = SourceModule(impVolKernFile.read() % {'K_size': K_size, 'ttm_size': ttm_size})
            comp_imp_vol   = imp_vol_mod.get_function('comp_imp_vol')
            comp_local_vol = imp_vol_mod.get_function('comp_local_vol')

        c_d = to_gpu(c).astype(np.float32)
        # compute both local and implied vol
        comp_imp_vol(impl_surf_d, c_d, K_grid_d, ttm_grid_d,
                     block=(ttm_size, 1, 1), grid=(K_size, 1))
        impl_surf = impl_surf_d.get()  # get impl. surf from device
        comp_local_vol(lv_surf_d, c_d, K_grid_d, ttm_grid_d,
                       block=(ttm_size, 1, 1), grid=(K_size, 1))
        lv_surf = lv_surf_d.get()  # get local. surf from device

    def update_graph(self, fwd, model, c, a, canvas):

        model.set_params(fwd, c)  # sets the params in the model
        if impl_local_ind == 'impl':
            if cuda_ind:
                impl_surf = model.gen_impl_surf_cuda(fwd,
                                                     ttm_grid_d, K_grid_d,
                                                     len(ttm_grid), len(
                                                         K_grid),
                                                     impl_surf_d, comp_imp_vol)
            else:
                # TO CORRECT HERE TO CORRECT HERE
                # impl_surf = model.gen_impl_surf_v() # vol. surface on cpu
                impl_surf = model.gen_impl_surf(
                    fwd,
                    ttm_grid,
                    K_grid)  # vol. surface on cpu
            a.plot_surface(K_mesh, ttm_mesh, impl_surf)
        else:
            if cuda_ind:
                lv_surf = model.gen_lv_surf_cuda()  # local vol on cuda
            else:
                lv_surf = model.gen_lv_surf()  # local vol surface on cpu
            a.plot_surface(K_mesh, ttm_mesh, lv_surf)
        canvas.show()


class ATMFVolatility(Volatility):
    """ ATM volatility
    """

    @property
    def _vol_dates(self):
        # TODO: THIS IS WRONG, DATES, NOT string
        return 'volDates'

    def atm_vol(self, fwdDate_ : datetime.date):
        """ Returns the ATM volatility for the forward date fwd_date.

        """

        return self._volParams[self._vol_dates][self._vol_for_date(fwdDate_)]


def getVolObject(comName : str, mkt_date : datetime.date) -> Volatility:
    """ Gets the vol object for the commodity in question for the market date mkt_date

    :param mkt_date: market date
    :param

    """

    print('comname', comName)
    volType, _, _, _ = ds.vol_hash[comName]

    return {'JWSS7': JWSS7Volatility
           , 'ATM' : ATMFVolatility }[volType](mkt_date, comName, comName)
