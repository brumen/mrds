# Implements volatility classes.

import config
import logging

import datetime

import numpy as np
from numpy import double, log, exp, sqrt

from functools import lru_cache
from typing import List, Tuple, Dict

import scipy
import scipy.stats
from scipy.stats import norm
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
    import pycuda.gpuarray as gpa
    from pycuda.gpuarray import to_gpu
    from pycuda.compiler import SourceModule

import ds
from pricers.pricers import black_greeks
# import vols.vols_fast as vols_fast
from forward_curve import FwdCurve


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
    def normalized_strike(S0   : np.double
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

        return log(K_v.reshape(1, len(K_v)) / S0) / (sigma * sqrt(ttm_v))

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

    def implied_vol(self, S0, K, ttm):
        """
        Implied vol needs to be implemented in the subclass.
        Implied volatility for S0, K, ttm.

        :param ttm: time to maturity
        :type ttm: double
        """

        raise VolatilityException('Method implied_vol not implemented in Volatility class.')

    def delta(self, K : float, ttm : float) -> float:
        """  Computes the delta of the volatility.

        :param K: strike at which the delta is requested.
        :param ttm: time to maturity.
        """

        raise VolatilityException('Method delta not implemented in Volatility class.')

    def time_to_maturity(self, fwd_date: datetime.date, dcf = 365.25):

        return (fwd_date - self.mkt_date).days / dcf  # time to maturity

    def black_simple( self
                    , fwd_date : datetime.date
                    , strike   : double
                    , dcf = 365.25
                    , df = 1.):
        """ Simple version of the black volatility.

        :param fwd_date: black vol for that date
        :param strike: strike value
        :param dcf: day-count fraction.
        """

        ttm = self.time_to_maturity(fwd_date)
        S0  = self._fwd_params.fwd_value(fwd_date)

        return black_greeks( S0
                           , strike
                           , -np.log(df) / ttm
                           , self.implied_vol(S0, strike, ttm)
                           , ttm)

    def callFutureK( self
                   , K : np.double
                   , ttm : np.double
                   , delta_K = 0.01 ):
        """
        WRONG WRONG WRONG WRONG
        derivative of the call option with respect to strike dC/dK

        """


        pr_0 = black_greeks( S0
                           , K
                           , -log(disc_fact) / double(T)
                           , self.implied_vol(S0, K, ttm)
                           , T
                           , 0)

        pr_delta = black_greeks( S0
                               , K + delta_K
                               , -log(disc_fact) / double(T)
                               , self.implied_vol(S0, K + delta_K, ttm)
                               , T
                               , 0 )
        return (pr_delta - pr_0) / delta_K
        return self.delta(S0, )

    def skewed_distribution(self, K, delta_K, ttm):
        """
        gives the CDF of a skewed distribution using UN-discounted call values
        """
        return 1.0 + self.call_future_K(S0, K, ttm)

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

    def draw_surface( self
                    , model
                    , fwd_idx
                    , Sd
                    , Su
                    , Sstep
                    , Tmin
                    , Tmax
                    , Tstep
                    , impl_local_ind = 'impl'
                    , cuda_ind       = False ):
        """
        Draws the implied/local vol surface from
          model ... vol. surface model, it contains:
            name = jw7, sabr, c0c1c2, ratiovol
          [Sd, Su] x [Tmin, Tmax] with steps Sstep, Tstep
          vol = vol. parametrization: jw7, sabr, c0c1c2, ratiovol
          impl_local_ind ... indicator for implied or local volatility
          cuda_ind ... should the computations be performed on the cuda
          fwd_idx ... forward index we are trying to plot
        """

        K_grid = np.arange(Sd, Su, Sstep)
        ttm_grid = np.arange(Tmin, Tmax, Tstep)
        K_size = len(K_grid)
        ttm_size = len(ttm_grid)

        if cuda_ind:
            K_grid_d = to_gpu(K_grid).astype(np.float32)  # K, ttm grid on device
            ttm_grid_d = to_gpu(ttm_grid).astype(np.float32)

        K_mesh, ttm_mesh = np.meshgrid(K_grid, ttm_grid)
        impl_surf = np.zeros((len(ttm_grid), len(K_grid)))
        lv_surf = np.zeros((len(ttm_grid), len(K_grid)))

        c = model.get_params(fwd_idx)  # constructs the param array

        if cuda_ind:
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
        self.draw_buttons()

        if model.name == 'jw7':
            jw7_buttons(fwd_idx, root, ax, dataPlot_canvas)
        elif model.name == 'c0c1c2':
            c0c1c2_buttons(root, ax, dataPlot_canvas)
        elif model.name == 'ratiovol':
            ratiovol_buttons(root, ax, dataPlot_canvas)
        elif model.name == 'sabr':
            sabr_buttons(root, ax, dataPlot_canvas)


    # writing this testing in a form of a function
    # CHECK CHECK - HERE WE ARE DIRECTLY UPDATING THE PARAMETERS OF THE MODEL,
    # SHOULD BE SEPARATE
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
    """
    ATM volatility

    """

    @property
    def volName(self):
        return 'ATMF'

    @property
    def _vol_dates(self):
        # TODO: THIS IS WRONG, DATES, NOT string
        return 'volDates'

    def atm_vol(self, fwdDate_ : datetime.date):
        """ Returns the ATM volatility for the forward date fwd_date.

        """

        return self._volParams[self._vol_dates][self._vol_for_date(fwdDate_)]


class JWSS7Volatility(Volatility):
    """ Jump-wing parametrization.
    """

    @property
    def _vol_name(self):
        return 'JWSS7'

    @property
    def _vol_dates(self):
        return 'vol_dates'

    def atmVol( self
              , fwdDate : datetime.date ) -> np.double :
        """
        Returns the atm forward for the fwd date fwd_date.

        :param fwdDate: forward date for which the ATM is constructed
        """

        return self._volParams['vol_curve'][self._vol_for_date(fwdDate)][0]  # first elt is atm vol.

    @lru_cache(maxsize=None)
    def _transform_from_jwss7(self, fwdDate : datetime.date ):
        """
        Returns jw7 parametrization from jwss7 for particular fwd date.

        vol_params in Jwss7: [S0, atm, skew, smile, putslope, putbend, callslope, callbend]
        vol_params in jw7  : [S0, atm, A   , B    , C       , P      , alphaC   , alphaP  ]
        """

        nthContract = self._vol_for_date(fwdDate)
        volParams   = self._volParams['vol_curve'][nthContract]

        sigma_0, skew, smile, put_slope, put_bend, call_slope, call_bend = volParams

        B = (2. * skew + put_slope) / (put_slope + call_slope)
        A = 0.5 * B * (1. - B) * (call_slope + put_slope)**2 / (smile + skew**2)

        return {'sigma_0': sigma_0,
                'A'      : A,
                'B'      : B,
                'C'      : call_slope / A,
                'P'      : put_slope / A,
                'alphaC' : call_bend,
                'alphaP' : put_bend }

    def _vol_compute(self, fwdUsed : datetime.date, z : np.double ):
        """
        Computes the volatility given the following parameters:

        :param z: normalized strike
        :param alphaC, alphaP, ...: parameters for the JW7 parametrization.

        """

        volParams = self._transform_from_jwss7(fwdUsed)

        return volParams['sigma_0'] * sqrt(1. + volParams['A'] * log(volParams['B'] * exp(volParams['C'] * (z / (1.0 + z * z) ** (volParams['alphaC']/2))) + \
                                                                     (1. - volParams['B']) * exp(- volParams['P'] * (z / (1.0 + z * z) ** (volParams['alphaP']/2)))))

    def implied_vol(self, fwdDate_ : datetime.date or int, K : np.double, ttm : np.double):
        """
        Implied vol for the fwd

        :param fwd: forward tenor - could be either an integer, like 5,
                    or datetime.date
        :type fwd: int or datetime.date
        :param K: strike price
        :param ttm: time to maturity
        """

        fwdUsed = fwdDate_ if isinstance(fwdDate_, int) else self._vol_for_date(fwdDate_)
        volParams = self._transform_from_jwss7(fwdDate_)

        _, fwdValues = self._fwd_params

        return self._vol_compute( fwdUsed
                                , JWSS7Volatility.normalized_strike(fwdValues[fwdUsed]
                                                                    , np.array([K])
                                                                    , volParams['sigma_0']
                                                                    , ttm) )

    def localVol(self, fwdDate : datetime.date, S, T, ttm):
        """
        Local volatility of the JWSS7 parametrization.

        :param fwdDate: forward index that we are computing the local vol of
        :param ttm: option time to maturity
        """

        jw7Params = self._transform_from_jwss7(fwdDate)
        sigma_0 = jw7Params['sigma_0']
        A       = jw7Params['A']
        B       = jw7Params['B']
        C       = jw7Params['C']
        P       = jw7Params['P']
        alphaC  = jw7Params['alphaC']
        alphaP  = jw7Params['alphaP']

        z = self.normalized_strike(S_0, S, sigma_0, ttm)  # TODO: CHECK HERE!!
        sigma = self.implied_vol(S, T)

        d1 = (log(S / S_0) + sigma * sigma * ttm / 2.0) / (sigma * sqrt(ttm))
        d2 = d1 - sigma * sqrt(ttm)
        Xz = B * exp(C * z) + (1.0 - B) * exp(- P * z)

        sigmaK = A / (2.0 * Xz * K * sqrt (ttm) ) / ( sqrt ( 1.0 + A * log (Xz) ) ) * \
            (B * C * exp(C * z) - P * (1.0 - B) * exp(- P * z))

        d1K = ((- 1.0 / K + sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
               ( log ( S / K ) + sigma * sigma * ttm / 2.0 ) * sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        d2K = ((- 1.0 / K - sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
               ( log ( S / K ) - sigma * sigma * ttm / 2.0 ) * sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        denomin = (sigma_0 * sqrt(ttm) * K * Xz * sqrt(1.0 + A * log(Xz)))
        BCexpr = (B * C * exp(C * z) - P * (1.0 - B) * exp(- P * z))

        sigmaKK = A / (2.0 * sqrt(ttm)) * (- A / (2.0 * denomin * K * Xz * (1.0 + A * log(Xz))) *
                                           BCexpr * BCexpr - BCexpr * BCexpr / (denomin * K * Xz) +
                                           (B * C ** 2 * exp(C * z) + P ** 2 * (1.0 - B) * exp(- P * z)) /
                                           (denomin * K) - BCexpr *
                                           sigma_0 *
                                           sqrt(ttm) / (denomin * K)
                                           )

        # derivative of z wrt t
        zt = log(K / S) / sigma_0 * (-0.5 * ttm**(- 1.5))
        sigmat = sigma_0 ** 2 / (2.0 * sigma) * A / Xz  * \
            ( B * C * exp ( C * z) - P * (1 - B) * exp ( - P * z ) ) * \
            zt  # derivative of sigma wrt t

        up_part = sigma * sigma + 2.0 * ttm * sigma * sigmat

        down_part = (1.0 + K * d1 * sqrt (ttm) * sigmaK ) ** 2.0 + K * K * ttm * sigma * \
                    (sigmaKK - d1 * sigmaK * sigmaK * ttm)

        # catching nan-s
        if (up_part / down_part < 0.0):
            logger.info("Caution: Imaginary local vol., using ATM vol.")
            return sigma_0

        return sqrt(up_part / down_part)

    def callFutureT(self, fwdDate, S0, K, ttm):
        """
        Derivative of Black's call (with expirty time  T) on a futures contract 
           with maturity ttm (cond: T < ttm)

        :param fwdDate: forward index that we are drawing the vol of
        :param S0:
        :param K: strike value
        :param ttm: time to maturity
        """

        jw7Params = self._transform_from_jwss7(fwdDate)
        sigma_0 = jw7Params['sigma_0']
        A       = jw7Params['A']
        B       = jw7Params['B']
        C       = jw7Params['C']
        P       = jw7Params['P']
        alphaC  = jw7Params['alphaC']
        alphaP  = jw7Params['alphaP']

        z = JWSS7Volatility.normalized_strike(S0, K, sigma_0, ttm)
        sigma = self.implied_vol(S0, K, ttm)
        S0_local = S0  # TODO: CHECK IF THIS IS REALLY NECESSARY

        Xz = B * exp(C * z) + (1.0 - B) * exp(- P * z)

        d1 = (log(S0_local / K) + sigma * sigma * ttm / 2.0) / \
            (sigma * sqrt(ttm))
        d2 = d1 - sigma * sqrt(ttm)
        zt = log(K / S0_local) / sigma_0 * (-0.5 * pow(ttm, - 1.5))  # z wrt t
        sigmat = sigma_0 * sigma_0 / (2.0 * sigma) * A / Xz  * \
            ( B * C * exp ( C * z) - P * ( 1.0 - B) * exp ( - P * z ) ) * \
            zt  # derivative of sigma wrt t
        sigma2t = 2.0 * sigma * sigmat  # derivative of sigma^2 wrt t

        d1T = ((sigma2t * ttm / 2.0 + sigma * sigma / 2.0) * sigma * sqrt(ttm) -
               ( log ( S0_local / K) + sigma * sigma * ttm / 2.0 ) * ( sigmat * ttm + sigma / (2.0 * sqrt (ttm) ) ) ) \
            / (sigma * sigma * ttm)

        d2T = (- (sigma2t * ttm / 2.0 + sigma * sigma / 2.0) * sigma * sqrt(ttm) -
               ( log ( S0_local / K) - sigma * sigma * ttm / 2.0 ) * ( sigmat * ttm + sigma / (2.0 * sqrt (ttm) ) ) ) \
            / (sigma * sigma * ttm)

        return S0_local * \
            scipy.stats.norm.pdf(d1) * d1T - K * scipy.stats.norm.pdf(d2) * d2T

    # first derivative of (undiscounted) Black's call wrt K
    def call_future_K(self, fwd, S0, K, ttm):
        """
        Derivative of a call option in this parametrization wrt strike price K.

        """

        jw7Params = self._transform_from_jwss7(fwdDate)
        sigma_0 = jw7Params['sigma_0']
        A       = jw7Params['A']
        B       = jw7Params['B']
        C       = jw7Params['C']
        P       = jw7Params['P']
        alphaC  = jw7Params['alphaC']
        alphaP  = jw7Params['alphaP']

        z = JWSS7Volatility.normalized_strike(S0, K, sigma_0, ttm)
        sigma = self.implied_vol(S0, K, ttm)
        S0_local = S0  # TODO: CHECK IF THIS IS REALLY NECESSARY
        d1 = (log(S0_local / K) + sigma * sigma * ttm / 2.0) / \
            (sigma * sqrt(ttm))
        d2 = d1 - sigma * sqrt(ttm)

        Xz = B * exp(C * z) + (1.0 - B) * exp(- P * z)

        sigmaK = A / (2.0 * Xz * K * sqrt (ttm) ) / sqrt ( 1.0 + A * log (Xz) ) * \
            (B * C * exp(C * z) - P * (1.0 - B) * exp(- P * z))

        d1K = ((- 1.0 / K + sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
               ( log ( S0_local / K ) + sigma * sigma * ttm / 2.0 ) * sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        d2K = ((- 1.0 / K - sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
               ( log ( S0_local / K ) - sigma * sigma * ttm / 2.0 ) * sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        return (S0_local * scipy.stats.norm.pdf(d1) * d1K -
                scipy.stats.norm.cdf(d2) - K * scipy.stats.norm.pdf(d2) * d2K)

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

        d1 = (log(S0 / K) + sigma * sigma * ttm / 2.0) / (sigma * sqrt(ttm))
        d2 = d1 - sigma * sqrt(ttm)
        Xz = B * exp(C * z) + (1.0 - B) * exp(- P * z)

        sigmaK = A / (2.0 * Xz * K * sqrt (ttm) ) / ( sqrt ( 1.0 + A * log (Xz) ) ) * \
            (B * C * exp(C * z) - P * (1.0 - B) * exp(- P * z))

        d1K = ((- 1.0 / K + sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
               ( log ( S0 / K ) + sigma * sigma * ttm / 2.0 ) * sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        d2K = ((- 1.0 / K - sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
               ( log ( S0 / K ) - sigma * sigma * ttm / 2.0 ) * sqrt (ttm) * sigmaK ) / \
            (sigma * sigma * ttm)

        denomin = (sigma_0 * sqrt(ttm) * K * Xz * sqrt(1.0 + A * log(Xz)))
        BCexpr = (B * C * exp(C * z) - P * (1.0 - B) * exp(- P * z))

        sigmaKK = A / (2.0 * sqrt(ttm)) * (
            - A / (2.0 * denomin * K * Xz * (1.0 + A * log(Xz))) * BCexpr * BCexpr -
            BCexpr * BCexpr / (denomin * K * Xz) +
            (B * C * C * exp(C * z) + P * P * (1.0 - B) * exp(- P * z)) /
            (denomin * K) -
            BCexpr * sigma_0 * sqrt(ttm) / (denomin * K))

        d1KK = ((1.0 / (K * K) + sigmaK * sigmaK * ttm + sigma * ttm * sigmaKK) * sigma * sqrt(ttm) -
                ( log ( S0 / K ) + sigma * sigma * ttm  / 2.0 ) * sqrt (ttm) * sigmaKK ) / ( sigma * sigma * ttm) - \
                  \
            2.0 * ((- 1.0 / K + sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
                   ( log ( S0_local / K ) + sigma * sigma * ttm  / 2.0 ) * sqrt (ttm) * sigmaK ) * sigma * ttm * sigmaK \
            / (sigma * sigma * sigma * sigma * ttm * ttm)

        d2KK = ((1.0 / (K * K) - sigmaK * sigmaK * ttm - sigma * ttm * sigmaKK) * sigma * sqrt(ttm) -
                ( log ( S0 / K ) - sigma * sigma * ttm  / 2.0 ) * sqrt (ttm) * sigmaKK ) / ( sigma * sigma * ttm) - \
            2.0 * ((- 1.0 / K - sigma * ttm * sigmaK) * sigma * sqrt(ttm) -
                   ( log ( S0_local / K ) - sigma * sigma * ttm  / 2.0 ) * sqrt (ttm) * sigmaK ) * sigma * ttm * sigmaK \
            / (sigma * sigma * sigma * sigma * ttm * ttm)

        return (S0 * normpdfD(d1) * d1K * d1K + S0_local * scipy.stats.norm.pdf(d1) * d1KK -
                2.0 * scipy.stats.norm.pdf(d2) * d2K - K * scipy.stats.norm.pdf(d2) * (-d2) * d2K * d2K - K * scipy.stats.norm.pdf(d2) * d2KK)

    def skewed_distribution(self, fwdDate, S0, K, ttm):
        return 1. + self.call_future_K(fwdDate, S0, K, ttm)

    def inversion_skewed_cdf( self
                            , fwdDate
                            , S0
                            , ttm
                            , quantile
                            , lb = 0.01
                            , ub = np.inf
                            , maxIter = 150
                            , iprint  = -9 ):
        """ Function finds K such that: skewed_cdf_analy(K, quantile) = 0
        """

        return NLP( lambda K: (self.skewed_distribution(fwdDate, S0, K, ttm) - quantile)**2
                      , S0
                      , lb      = lb
                      , ub      = ub
                      , maxIter = maxIter
                      , iprint  = iprint ).solve('scipy_cobyla').xf[0]


class JWSS7VolatilityDisplay(JWSS7Volatility, VolatilityDrawMixin):

    def jw7_buttons(self, fwd, root, ax, dataPlot_canvas):
        fct_update = lambda cc: self.update_graph(fwd, model, array([c1.get(), c2.get(), c3.get(), c4.get(), c5.get(), c6.get(), c7.get(), c8.get()]), ax,
                                             dataPlot_canvas)
        # root ... Tk root
        # ax ... Axes3D object
        # dataPlot_canvas ... canvas object

        # parameter tk.SCALEs
        c1 = tk.Scale(root, from_=80.0, to=120.0, resolution=0.1, label="S0", orient=tk.HORIZONTAL,
                   command=fct_update)
        c2 = tk.Scale(root, from_=0.05, to=0.8, resolution=0.05, label="sig", orient=tk.HORIZONTAL,
                   command=fct_update)
        c3 = tk.Scale(root, from_=0.0, to=5.0, resolution=0.25, label="A", orient=tk.HORIZONTAL,
                   command=fct_update)
        c4 = tk.Scale(root, from_=0.0, to=1.0, resolution=0.05, label="B", orient=tk.HORIZONTAL,
                   command=fct_update)
        c5 = tk.Scale(root, from_=0.0, to=5.0, resolution=0.2, label="C", orient=tk.HORIZONTAL,
                   command=fct_update)
        c6 = tk.Scale(root, from_=0.0, to=5.0, resolution=0.2, label="P", orient=tk.HORIZONTAL,
                   command=fct_update)
        c7 = tk.Scale(root, from_=0.0, to=5.0, resolution=0.2, label="alpha_C", orient=tk.HORIZONTAL,
                   command=fct_update)
        c8 = tk.Scale(root, from_=0.0, to=5.0, resolution=0.2, label="alpha_P", orient=tk.HORIZONTAL,
                   command=fct_update)

        c1.grid(row=0, column=1)
        c2.grid(row=1, column=1)
        c3.grid(row=2, column=1)
        c4.grid(row=3, column=1)
        c5.grid(row=4, column=1)
        c6.grid(row=5, column=1)
        c7.grid(row=6, column=1)
        c8.grid(row=7, column=1)

        # replot button
        b1 = tk.Button(root, text="replot", command=lambda: update_graph(fwd, model, array([c1.get(), c2.get(), c3.get(), c4.get(), c5.get(), c6.get(), c7.get(), c8.get()]), ax,
                                                                      dataPlot_canvas)).grid(row=8, column=0)

        dataPlot_canvas.show()
        root.mainloop()


class C0C1C2Volatility(Volatility):
    """ c0-c1-c2 volatility parametrization
        _smooth_ind is the smoothness indicator
        _alpha is the smoothness factor
    """

    def __init__(self
                 , com_name : str
                 , mkt_date : datetime.date
                 , fwd_params : FwdCurve
                 , vol_params : Dict[datetime.date, List]):
        super(C0C1C2Volatility, self).__init__(com_name, mkt_date, fwd_params=fwd_params, vol_params=vol_params)
        self.__vol_dates = list(vol_params.keys())  # TODO: CHECK THIS PART

    @property
    def _vol_dates(self) -> List[datetime.date]:
        return self.__vol_dates

    def _get_next_date(self, fwd_date : datetime.date ) -> datetime.date:
        """ Returns the next date on the forward curve after fwd_date.

        :param fwd_date: the date after which we are searching on the curve.
        :returns: date after the fwd_date on the forward curve
        """

        fd_better = [fd for fd in self._vol_dates if fd > fwd_date]

        if fd_better:  # the list of larger dates is not empty
            return fd_better[0]

        # else returns the last date on the curve
        return self._vol_dates[-1]

    def _get_c0c1c2(self, ttm : datetime.date) -> Tuple[float, float, float, float, float]:
        next_date = self._get_next_date(ttm)
        return self._vol_params[next_date]

    def implied_vol(self, fwd_value : float, ttm : datetime.date, smooth_ind=True) -> float:
        """ Computes the implied volatility of quadratic volatility surface.

        :param fwd_value: forward value for which to compute
        :param ttm: time-to-maturity
        :param smooth_ind: indicator whether to smooth the curve.
        """

        atm_strike = self._fwd_params.fwd_value(ttm)
        z = np.log(fwd_value / atm_strike)

        c0, c1, c2, theta, alpha = self._get_c0c1c2(ttm)

        v = c0 + c1 * z + c2**2 * z**2
        sigma_star = c0 * theta - alpha * (c0 * theta - c0)
        a = c0 * theta - sigma_star

        # TODO: CHECK IF BELOW IS arctan or arctan2
        if smooth_ind:
            return v if v < sigma_star else 2. * a / np.pi * np.arctan ( np.pi / (2 * a) * (v - sigma_star) ) + sigma_star

        # smooth_ind == False, some additional logic
        if v >= c0 * theta:
            return c0 * theta

        return v


class CIVolatility(Volatility):
    """ CI parametrization.
    """

    def __init__(self, tenor_l, delta_mn_l, vols_l, omega_l):
        """
        tenor_l: tenors list [1., 2.]
        delta_mn: list of lists of delta in log-m: N(log-moneyn. log(F/F_0) / sigma / sqrt(T)
        vols: list of lists of vols for corr. moneyness
        omega_l: list of omegas, smoothing parameters
        """

        self.name = 'ci'
        self.nb_fwds = len(tenor_l)
        self.tenor_l = tenor_l
        self.delta_mn_l = delta_mn_l
        self.vols_l = vols_l
        self.omega_l = omega_l

        self.implied_vol = {}
        for tenor, (T, delta_mn, vol, omega) in enumerate(zip(self.tenor_l, self.delta_mn_l,
                                                              self.vols_l, self.omega_l)):
            self.implied_vol[tenor] = lambda z: self.convInterp(delta_mn, vol, omega, z)

    def gen_impl_surf(self, fwd, ttm_grid, delta_grid):
        """
        generates impl. vols surface for fwd(scalar) and K_grid(v), ttm
        """
        return self.implied_vol[0](delta_grid)  # WRONG WRONG - 0 HERE

    def Phi(self, n, x, xx, omega):
        return norm.cdf((xx[n]-x)/omega)

    def phi(self, n, x, xx, omega):
        return norm.pdf((xx[n]-x)/omega)

    def J(self, x, xx, omega):
        JJ = np.array([])
        tmp = (1.-(x - xx[0])/(xx[1] - xx[0]))*(self.Phi(1, x, xx, omega) - self.Phi(0, x, xx, omega))
        tmp += (omega/(xx[1]-xx[0]))*(self.phi(1, x, xx, omega) - self.phi(0, x, xx, omega))
        tmp += (1.-norm.cdf((x-xx[0])/omega))
        JJ = np.append(tmp, JJ)
        N = len(xx)-1

        for n in range(1, N):
            tmp1 = (1.-(x-xx[n])/(xx[n+1]-xx[n]))*(self.Phi(n+1, x, xx, omega)-self.Phi(n, x, xx, omega))
            tmp1 += (omega/(xx[n+1]-xx[n]))*(self.phi(n+1, x, xx, omega)-self.phi(n, x, xx, omega))
            tmp2 = (x-xx[n-1])/(xx[n]-xx[n-1])*(self.Phi(n, x, xx, omega)-self.Phi(n-1, x, xx, omega))
            tmp2 -= (omega/(xx[n]-xx[n-1]))*(self.phi(n, x, xx, omega)-self.phi(n-1, x, xx, omega))
            JJ = np.append(JJ, tmp1 + tmp2)

        tmp = (x-xx[-2])/(xx[-1]-xx[-2])*(self.Phi(N, x, xx, omega) - self.Phi(N-1, x, xx, omega))
        tmp -= (omega/xx[-1]-xx[-2])*(self.phi(N, x, xx, omega)-self.phi(N-1, x, xx, omega))
        tmp += norm.cdf((x-xx[-1])/omega)
        JJ = np.append(JJ, tmp)

        return JJ

    def I(self, x, xx, omega):
        II = np.zeros([len(xx), len(xx)])
        tmp = 1. - norm.cdf((x-xx[0])/omega)
        II00 = tmp
        N = len(xx) - 1
        for n in range(N):
            tmp1 = (1.-(x-xx[n])/(xx[n+1]-xx[n])) * (self.Phi(n+1, x, xx, omega) - self.Phi(n, x, xx, omega))
            tmp1 += (omega/(xx[n+1]-xx[n])) * (self.phi(n+1, x, xx, omega) - self.phi(n, x, xx, omega))
            II[n, n] += tmp1
            tmp2 = (x-xx[n])/(xx[n+1]-xx[n]) * (self.Phi(n+1, x, xx, omega) - self.Phi(n, x, xx, omega))
            tmp2 -= (omega/(xx[n+1]-xx[n])) * (self.phi(n+1, x, xx, omega) - self.phi(n, x, xx, omega))
            II[n+1, n] = II[n+1, n] + tmp2
        IInn = norm.cdf((x-xx[-1])/omega)
        return II, II00, IInn

    def convInterp(self, lm_v, vol_v, omega, lm_new_v):
        """
        convolution interpolation:
        inputs:
          (lm_v, vol_v) .. pairs of log-moneyness, vol
          ln_new_v ... log-m where you want to compute vols
          omega ... parameter
        """
        if (type(lm_new_v) is not np.ndarray) and (type(lm_new_v) is not list):
            lm_new_v_arr = [lm_new_v]
        else:
            lm_new_v_arr = lm_new_v
        lm_new_v_len = len(lm_new_v_arr)
        lm_v_len = len(lm_v)
        Jnm = np.zeros([lm_v_len, lm_v_len])
        for ii in range(lm_v_len):
            Jnm[ii, :] = self.J(lm_v[ii], lm_v, omega)
        # CONTINUE HERE

    def extract_ind(self, p_mat):
        """
        extracts the model parameters
        """
        self.name = 'ci'  # adds the name of the model
        self.sigma_0 = double(p_mat[:, 0])
        self.rr_25 = double(p_mat[:, 1])  # vector of rr_25 marks
        self.wg_25 = double(p_mat[:, 2])  # vector of wg


def getVolObject(comName : str, mkt_date : datetime.date) -> Volatility:
    """ Gets the vol object for the commodity in question for the market date mkt_date

    :param mkt_date: market date
    :param

    """

    print('comname', comName)
    volType, _, _, _ = ds.vol_hash[comName]

    return {'JWSS7': JWSS7Volatility
           , 'ATM' : ATMFVolatility }[volType](mkt_date, comName, comName)
