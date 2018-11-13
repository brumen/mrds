.# volatilities' module

from abc import ABC, abstractmethod

import config
import logging

import numpy as np
from numpy import double, log, exp, sqrt

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
import vols.vols_fast as vols_fast

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


class Volatility(ABC):
    """
    Base class for vol parametrization

    """

    def __init__( self
                , comName
                , mktDate ):
        """
        Generic class for the volatility object. Most generic way of computing the volatility.

        """

        self.mktDate    = mktDate
        self.comName    = comName
        self._volParams = ds.get_vol_curve    (comName, mktDate)
        self._fwdParams = ds.get_forward_curve(comName, mktDate)

    @staticmethod
    def normalizedStrike( S0   : np.double
                        , K_v  : np.array
                        , sigma: np.double
                        , ttm_v: np.array ) -> np.array:
        """
        Vectorized form of norm_strike

        :param S0: initial stock (forward) price
        :param K_v: strike price
        :param sigma: ATM volatility of the stock price
        :param ttm_v: time to maturity
        :returns: normalized strike of the option
        """

        return log(double(K_v.reshape(1, len(K_v))) / double(S0)) / \
               (sigma * sqrt(double(ttm_v)))

    @staticmethod
    def normalizedStrikeInv( delta_v: np.array
                           , sigma: np.double
                           , ttm: np.double) -> np.array:
        """
        Inverse of the normalized strike.

        :param delta_v: vector of delta
        :param sigma: volatility of stock/forward
        :param ttm: time to maturity
        """

        return exp(scipy.stats.norm.ppf(delta_v) * sigma * sqrt(double(ttm)) - 0.5 * sigma ** 2 * ttm)

    @abstractmethod
    def impliedVol(self, S0, K, ttm):
        """
        Implied vol needs to be implemented in the subclass.
        Implied volatility for S0, K, ttm.

        :param ttm: time to maturity
        :type ttm: double
        """

        return 0.2

    @abstractmethod
    def delta( self
             , K : np.double
             , ttm : np.double ):
        """
        Computes the delta of the volatility.

        """

        pass

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
                           , self.impliedVol(S0, K, ttm)
                           , T
                           , 0)

        pr_delta = black_greeks( S0
                               , K + delta_K
                               , -log(disc_fact) / double(T)
                               , self.impliedVol(S0, K + delta_K, ttm)
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

    def inversion_skewed_cdf(self, quantile, ttm):
        """
        find K : skewed_cdf_analy(K, quantile) = 0
        """
        optim_pr = NLP(lambda K: self.skewed_cdf_analy(K, quantile), S0, lb=0.01, ub=inf,
                       maxIter=self.max_iter, iprint=self.iprint)
        return optim_pr.solve(self.solver).xf[0]

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

    def gen_impl_surf(self, fwd, ttm_grid, K_grid):
        """
        Generates the implied vol surface for the following parameters:

        :param fwd: number of the forward contract
        :param ttm_grid: grid of expiry times
        :param K_grid: list of strikes
        """

        self.impl_surf = np.empty((len(ttm_grid), len(K_grid)))

        for ttm_ind, ttm in enumerate(ttm_grid):
            for K_ind, K in enumerate(K_grid):
                self.impl_surf[ttm_ind, K_ind] = self.impliedVol(fwd, K, ttm)

        return self.impl_surf

    def gen_lv_surf(self, ttm_grid, K_grid, dT, dK):
        """
        local-vol surface for ttm_grid, K_grid
        """

        self.lv_surf = np.empty((len(ttm_grid), len(K_grid)))
        for ttm_ind, ttm in enumerate(ttm_grid):
            for K_ind, K in enumerate(K_grid):
                self.lv_surf[ttm_ind, K_ind] = self.local_vol(K, ttm, dT, dK)

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
            c_d = to_gpu(c).astype(np.float32)
            imp_vol_kern_string = open(config.work_dir + "imp_vol_kern.cu").read()
            imp_vol_mod = SourceModule(
                imp_vol_kern_string % {
                    "K_size": K_size,
                    "ttm_size": ttm_size})
            # extracting compute vol function
            comp_imp_vol = imp_vol_mod.get_function("comp_imp_vol")
            comp_local_vol = imp_vol_mod.get_function(
                "comp_local_vol")  # extracting compute vol function
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
    Simplest flat volatility.

    """

    def __init__( self
                , comName
                , mktDate ):
        """
        reads from database and constructs vol object

        """
        super().__init__(comName, mktDate)  # date and comName are defined here

    @property
    def volName(self):
        return 'ATMF'

    def atmVol( self
              , fwdDate ):
        """
        Returns the ATM volatility for the forward date fwdDate

        """

        return self._volParams  # TODO: FINISH HERE


class JW7Volatility(Volatility):
    """
    JumpWing parametrization (inherits from vol_param)

    """

    def __init__( self
                , comName
                , mktDate ):
        """
        reads from database and constructs vol object

        """
        super().__init__(comName, mktDate)  # date and comName are defined here
        params_names = self.transform_from_jwss7(self._volParams)  # self.p_mat)

        self.S0        = params_names['S0']
        self.fwd_curve = self.S0
        self.sigma_0   = params_names['sigma_0']
        self.skew      = params_names['skew']
        self.smile     = params_names['smile']
        self.putSlope  = params_names['putSlope']
        self.putBend   = params_names['putBend']
        self.callBend  = params_names['callBend']
        self.callSlope = params_names['callSlope']

        self.B      = params_names['B']
        self.A      = params_names['A']
        self.C      = params_names['C']
        self.P      = params_names['P']
        self.alphaC = params_names['alphaC']
        self.alphaP = params_names['alphaP']
        self.ttm_opt = np.array([(tenor_dt - self.market_date_dt).days / 365.25
                                 for tenor_dt in option_tenors_dt])

    @property
    def volName(self):
        return 'jw7'

    def atmVol( self
              , fwdDate ):
        """
        Returns the atm forward for the fwd date.
        """

        return self._volParams

    def extract_tenors(self, new_market_date, tenors_list):
        """
        leaves in the object only the tenors specified in tenors_list
        :param tenors_list: list [3, 4, 5]
        """
        self.p_mat = self.p_mat[tenors_list, :]
        self.S0 = self.S0[tenors_list]
        self.fwd_curve = self.fwd_curve[tenors_list]
        self.sigma_0 = self.sigma_0[tenors_list]
        self.skew = self.skew[tenors_list]
        self.smile = self.smile[tenors_list]
        self.putSlope = self.putSlope[tenors_list]
        self.putBend = self.putBend[tenors_list]
        self.callBend = self.callBend[tenors_list]
        self.callSlope = self.callSlope[tenors_list]
        self.B = self.B[tenors_list]
        self.A = self.A[tenors_list]
        self.C = self.C[tenors_list]
        self.P = self.P[tenors_list]
        self.alphaC = self.alphaC[tenors_list]
        self.alphaP = self.alphaP[tenors_list]
        market_date_diff = (ds.convert_str_datetime(new_market_date) - self.market_date_dt).days / 365.25
        self.ttm_opt = self.ttm_opt[tenors_list] - market_date_diff
        self.market_date = new_market_date
        self.market_date_dt = ds.convert_str_datetime(self.market_date)

    @staticmethod
    def transform_from_jwss7(p_mat):
        """
        Returns jw7 parametrization from jwss7.
        p_mat in Jwss7: [S0, atm, skew, smile, putslope, putbend, callslope, callbend]
        p_mat in jw7: [S0, atm, A, B, C, P, alphaC, alphaP]

        """

        S0 = double(p_mat[:, 0])
        sigma_0 = double(p_mat[:, 1])
        skew = double(p_mat[:, 2])
        smile = double(p_mat[:, 3])
        put_slope = double(p_mat[:, 4])
        put_bend = double(p_mat[:, 5])
        call_slope = double(p_mat[:, 6])
        call_bend = double(p_mat[:, 7])

        B = (2. * skew + put_slope) / (put_slope + call_slope)
        A = 0.5 * B * \
            (1. - B) * (call_slope + put_slope)**2 / (smile + skew**2)
        C = call_slope / A
        P = put_slope / A
        alphaC = call_bend
        alphaP = put_bend

        return {'S0'     : S0,
                'sigma_0': sigma_0,
                'skew'   : skew,
                'smile': smile,
                'putSlope': put_slope,
                'putBend': put_bend,
                'callSlope': call_slope,
                'callBend': call_bend,
                'B': B,
                'A': A,
                'C': C,
                'P': P,
                'alphaC': alphaC,
                'alphaP': alphaP }

    @staticmethod
    def _vol_compute(z, alphaC, alphaP, sigma_0, A, B, C, P):
        """
        Computes the volatility given the following parameters:

        """

        return sigma_0 * sqrt(1. + A * log(B * exp(C * (z / (1.0 + z * z) ** (alphaC/2))) + \
                                           (1. - B) * exp(- P * (z / (1.0 + z * z) ** (alphaP/2)))))

    def implied_vol(self, fwd, K, ttm):
        z = norm_strike(self.S0[fwd], K, self.sigma_0[fwd], ttm)
        return self._vol_compute(z, self.alphaC[fwd], self.alphaP[fwd],
                                 self.sigma_0[fwd], self.A[fwd], self.B[fwd],
                                 self.C[fwd], self.P[fwd])

    def _norm_strike_inv_exact(self, fwd, delta_val):
        """
        solution to N(d1) = delta_val, where d1(vol)
        """

        optim_pr = NLP(lambda K: vols_fast.invert_delta( K
                                                               , delta_val
                                                               , self.ttm_opt[fwd],
                                               self.fwd_curve[fwd],
                                               self.alphaC[fwd],
                                               self.alphaP[fwd],
                                               self.sigma_0[fwd],
                                               self.A[fwd], self.B[fwd],
                                               self.C[fwd], self.P[fwd])
                              , self.fwd_curve[fwd]
                              , lb      = 0.001
                              , ub      = np.inf
                              , maxIter = 150
                              , iprint  = -9)

        return optim_pr.solve('scipy_cobyla').xf[0]

    def implied_vol_all_fwd_standard(self, delta_v):

        vol_mat = np.empty((len(self.fwd_curve), len(delta_v)))
        for tenor_nb, (fwd_v, ttm) in enumerate(zip(self.fwd_curve, self.ttm_opt)):
            perc_v = norm_strike_v_inv(delta_v, self.sigma_0[tenor_nb], ttm)
            z = norm_strike_v(fwd_v, fwd_v * perc_v, self.sigma_0[tenor_nb], ttm)
            vol_mat[tenor_nb, :] = self._vol_compute(z,
                                                     self.alphaC[tenor_nb],
                                                     self.alphaP[tenor_nb],
                                                     self.sigma_0[tenor_nb],
                                                     self.A[tenor_nb], self.B[tenor_nb],
                                                     self.C[tenor_nb], self.P[tenor_nb])
        return vol_mat

    def gen_impl_surf_v(self, fwd, ttm_grid, K_grid):
        self.impl_surf = self.implied_vol_v(fwd, K_grid, ttm_grid)
        return self.impl_surf

    def gen_impl_surf_cuda(self, fwd,
                           ttm_grid_d, K_grid_d,
                           ttm_grid_len, K_grid_len,
                           impl_surf_d,
                           comp_imp_vol):
        """
        computes the implied surface on cuda
          fwd ... forward index that we are plotting
          ttm_grid_d, K_grid_d ... grids for time to maturity, K on the _device_
          comp_imp_vol ... kernel for computing implied vol
        """
        c_d = gpa.to_gpu(self.get_params(fwd)).astype(np.float32)
        comp_imp_vol(impl_surf_d, c_d, K_grid_d, ttm_grid_d,
                     block=(K_grid_len, 1, 1), grid=(ttm_grid_len, 1))
        return impl_surf_d.get()

    def gen_lv_surf_cuda(self, fwd, ttm_grid, K_grid):
        """
        Computes local vol on cuda

        """

        c_d = gpa.to_gpu(self.get_params(fwd)).astype(np.float32)

        self.comp_local_vol(self.lv_surf_d, c_d, self.K_grid_d, self.ttm_grid_d,
                            block=(len(K_grid), 1, 1), grid=(len(ttm_grid), 1))
        self.lv_surf = self.lv_surf_d.get()

    def local_vol(self, fwd, S, T, ttm):
        """
        Local volatility of the JWSS7 parametrization.

        :param fwd: forward index that we are computing the local vol of
        :param ttm: option time to maturity
        """

        S_0     = self.S0[fwd]
        sigma_0 = self.sigma_0[fwd]
        A       = self.A[fwd]
        B       = self.B[fwd]
        C       = self.C[fwd]
        P       = self.P[fwd]
        alphaC  = self.alphaC[fwd]
        alphaP  = self.alphaP[fwd]

        z = self.normalizedStrike(S_0, S, sigma_0, ttm)  # TODO: CHECK HERE!!
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

        denomin = (self.sigma_0 * sqrt(ttm) * K * Xz * sqrt(1.0 + A * log(Xz)))
        BCexpr = (B * C * exp(C * z) - P * (1.0 - B) * exp(- P * z))

        sigmaKK = A / (2.0 * sqrt(ttm)) * (- A / (2.0 * denomin * K * Xz * (1.0 + A * log(Xz))) *
                                           BCexpr * BCexpr - BCexpr * BCexpr / (denomin * K * Xz) +
                                           (B * C ** 2 * exp(C * z) + P ** 2 * (1.0 - B) * exp(- P * z)) /
                                           (denomin * K) - BCexpr *
                                           self.sigma_0 *
                                           sqrt(ttm) / (denomin * K)
                                           )

        # derivative of z wrt t
        zt = log(K / S) / self.sigma_0 * (-0.5 * ttm**(- 1.5))
        sigmat = self.sigma_0 ** 2 / (2.0 * sigma) * A / Xz  * \
            ( B * C * exp ( C * z) - P * (1 - B) * exp ( - P * z ) ) * \
            zt  # derivative of sigma wrt t

        up_part = sigma * sigma + 2.0 * ttm * sigma * sigmat

        down_part = (1.0 + K * d1 * sqrt (ttm) * sigmaK ) ** 2.0 + K * K * ttm * sigma * \
                    (sigmaKK - d1 * sigmaK * sigmaK * ttm)

        # catching nan-s
        if (up_part / down_part < 0.0):
            logger.info("Caution: Imaginary local vol., using ATM vol.")
            return sigma_0
        else:
            return sqrt(up_part / down_part)

        # return (up_part / down_part < 0.0) * self.sigma_0 + (up_part /
        # down_part >= 0.0) * sqrt (up_part/down_part)

    def callFutureT(self, fwd, S0, K, ttm):
        """
        Derivative of Black's call (with expirty time  T) on a futures contract 
           with maturity ttm (cond: T < ttm)

        :param fwd: forward index that we are drawing the vol of
        :param S0:
        :param K: strike value
        :param ttm: time to maturity
        """

        S0      = self.S0[fwd]
        sigma_0 = self.sigma_0[fwd]
        A       = self.A[fwd]
        B       = self.B[fwd]
        C       = self.C[fwd]
        P       = self.P[fwd]

        z = norm_strike(S0, K, sigma_0, ttm)
        sigma = self.implied_vol(fwd, K, ttm)
        S0_local = S0  # CHECK IF THIS IS REALLY NECESSARY

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


        z = norm_strike(S0, K, sigma_0, maturity)
        sigma = self.implied_vol(fwd, K, ttm)  # CHECK IF THIS IS RIGHT

        d1 = (log(S0_local / K) + sigma * sigma * ttm / 2.0) / \
            (sigma * sqrt(ttm))
        d2 = d1 - sigma * sqrt(ttm)

        Xz = B * exp(C * z) + (1.0 - B) * exp(- P * z)

        sigmaK = A / (2.0 * Xz * K * sqrt (maturity) ) / sqrt ( 1.0 + A * log (Xz) ) * \
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
    def call_future_KK(self, fwd, S0, K, ttm):

        # derivative of the pdf of standard normal
        def normpdfD(x):
            return scipy.stats.norm.pdf(x) * (-x)

        S00_local, sigma_0, A, B, C, P, alphaC, alphaP = self.get_params(
            fwd).transpose()
        S0_local = S0

        z = norm_strike(S0, K, sigma_0, ttm)
        sigma = self.implied_vol(fwd, K, ttm)  # implied_vol (S0, K, ttm)

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
                2.0 * scipy.stats.norm.pdf(d2) * d2K - K * normpdfD(d2) * d2K * d2K - K * scipy.stats.norm.pdf(d2) * d2KK)

    # ARGUMENTS NOT CONSISTENT W/ PARENT CLASS, although it works
    def skewed_distribution(self, fwd, S0, K, ttm):
        return 1. + self.call_future_K(fwd, S0, K, ttm)

    def skewed_cdf(self, fwd, S0, K, ttm, quantile):
        return (self.skewed_distribution(fwd, S0, K, ttm) - quantile)**2

    # find K : skewed_cdf_analy(K, quantile) = 0
    def inversion_skewed_cdf(self, fwd, S0, ttm, quantile):
        optim_pr = NLP( lambda K: self.skewed_cdf(fwd, S0, K, ttm, quantile)
                      , S0
                      , lb      = 0.01
                      , ub      = np.inf
                      , maxIter = 150
                      , iprint  = -9 )
        return optim_pr.solve('scipy_cobyla').xf[0]

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
        b1 = Button(root, text="replot", command=lambda: update_graph(fwd, model, array([c1.get(), c2.get(), c3.get(), c4.get(), c5.get(), c6.get(), c7.get(), c8.get()]), ax,
                                                                      dataPlot_canvas)).grid(row=8, column=0)

        dataPlot_canvas.show()
        root.mainloop()



class c0c1c2Volatility(Volatility):
    """
    c0-c1-c2 volatility parametrization
    smooth_ind is the smoothness indicator
    alpha is the smoothness factor

    """

    def __init__(self, comName, mktDate):
        super().__init__(comName, mktDate)  # defines _volParams, _fwdParams
        self.volName = 'c0c1c2'

    # extracts the model parameters
    def extract_ind(self, p_mat):
        self._c0        = self._volParams[:, 0]  # vector of c0s
        self._c1        = self._volParams[:, 1]  # vector of c1s
        self._c2        = self._volParams[:, 2]
        self.theta      = self._volParams[:, 3]
        self.smooth_ind = self._volParams[:, 4]
        self.alpha      = self._volParams[:, 5]

    def implied_vol(self, F, t):
        K = 100.  # WRONG WRONG WRONG ...
        z = log(F / K)  # WRONG WRONG CHECK IF THIS IS CORRECT
        v = self.c0 + self.c1 * z + self.c2**2 * z**2
        sigma_star = self.c0 * self.theta - \
            self.alpha * (self.c0 * self.theta - self.c0)
        a = self.c0 * self.theta - sigma_star
        # CHECK IF BELOW IS arctan or arctan2
        g = lambda sigma: 2 * a / pi * arctan ( pi / (2 * a) * (sigma - sigma_star) ) + \
            sigma_star
        return  ( v * ( v < self.c0 * self.theta) + self.c0 * self.theta * (v >= self.c0 * self.theta) ) * ( self.smooth_ind == False ) + \
            (v * (v < sigma_star) + g(v) * (v >= sigma_star)) * \
            (self.smooth_ind == True)


class CIVolatility(Volatility):
    """
    CI parametrization: accepts deltas and vols

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


def interpolate_fwd_vols(fwd_tenors, fwd_prices, vol_tenors, vol_vols,
                         fwd_tenors_wanted, vol_tenors_wanted):
    """
    interpolates by spline between the tenors (both prices and vols)
      fwd_tenors ... vector of tenors
      fwd_prices ... vector of prices for tenors given in fwd_tenors
      vol_tenors, vol_vols ... same as above, just for vols,
      vol_vols is a matrix, with as many rows as vol_tenors and multiple
      columns for all parameters
    """
    fwd_function = lambda t: scipy.interpolate.splev(
        t,
        scipy.interpolate.splrep(
            fwd_tenors,
            fwd_prices))  # interpol. prices
    res = [fwd_function(fwd_tenors_wanted)]  # result, part 1
    # append an empty zero matrix
    res.append(zeros((len(vol_tenors_wanted), vol_vols.shape[1])))

    for vol_param_ind in xrange(vol_vols.shape[1]):
        params_tmp = lambda t: scipy.interpolate.splev(
            t,
            scipy.interpolate.splrep(
                vol_tenors,
                vol_vols[
                    :,
                    vol_param_ind]))
        res[:, vol_param_ind] = params_tmp(vol_tenors_wanted)

    return res


def black_vol_inverse_vec(F, K_vec, p_vec, dt, DF, theta, tol):
    """
    Inverse black vol for a vector of strikes, and a vector or prices.

    :param F: current forward vol.
    :type F: double
    :param K_vec: vector of strikes.
    :type K_vec: np.array[double]
    """

    return np.array([black_vol_inverse(F, K, p, dt, DF, theta, tol)
                     for K, p in zip(K_vec, p_vec)]).ravel()


def black_vol_inverse(F, K, p, dt, DF, theta, tol):
    """
    Computation of black vol from option prices.

    :param F: forward price
    :type F: double
    :param K: strike price
    :type K: double
    :param p: option price
    :type p: double
    :param dt: time to maturity
    :type dt: double
    :param DF: discount factor until dt
    :type DF: double
    :param theta: call/put indicator
    :type theta: TODO
    :param tol: toleranca
    :type tol: double
    """

    return vols_fast.black_vol_inverse_normalized( double(p) / (DF * sqrt(double(F) * double(K)))
                                                 , log(double(F) / double(K))
                                                 , theta
                                                 , tol) / sqrt(dt)


def black_vol_inverse_naive_vec(F, K_vec, p_vec, dt, DF, theta, tol, solver=None):
    return np.array([black_vol_inverse_naive(F, K, p, dt, DF, theta, tol, solver)
                     for K, p in zip(K_vec, p_vec)]).ravel()


def black_vol_inverse_naive(F, K, p, dt, DF, theta, tol, solver=None):
    """
    black vol computation

      theta = 1 ... call option, -1 ... put option
    """

    x = log(double(F) / double(K))  # insuring that no integer division is made
    beta = p / (DF * sqrt(F * K))

    # optimization search, initial guess = sigma_c
    optim_pr = NLP( lambda sigma: (b(x, sigma, theta) - beta)**2
                  , sqrt(2 * abs(x))  # inflection point function  (sigma_c)
                  , lb = 1e-6
                  , iprint = -1 )  # lower bound just above 0

    return optim_pr.solve('scipy_cobyla' if solver is None else solver).xf[0] / sqrt(dt)


def draw_surface( model
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
        c_d = to_gpu(c).astype(np.float32)
        imp_vol_kern_string = open(config.work_dir + "imp_vol_kern.cu").read()
        imp_vol_mod = SourceModule(
            imp_vol_kern_string % {
                "K_size": K_size,
                "ttm_size": ttm_size})
        # extracting compute vol function
        comp_imp_vol = imp_vol_mod.get_function("comp_imp_vol")
        comp_local_vol = imp_vol_mod.get_function(
            "comp_local_vol")  # extracting compute vol function
        # compute both local and implied vol
        comp_imp_vol(impl_surf_d, c_d, K_grid_d, ttm_grid_d,
                     block=(ttm_size, 1, 1), grid=(K_size, 1))
        impl_surf = impl_surf_d.get()  # get impl. surf from device
        comp_local_vol(lv_surf_d, c_d, K_grid_d, ttm_grid_d,
                       block=(ttm_size, 1, 1), grid=(K_size, 1))
        lv_surf = lv_surf_d.get()  # get local. surf from device

    # writing this testing in a form of a function
    # CHECK CHECK - HERE WE ARE DIRECTLY UPDATING THE PARAMETERS OF THE MODEL,
    # SHOULD BE SEPARATE
    def update_graph(fwd, model, c, a, canvas):
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



    # same as for jw7 buttons
    def c0c1c2_buttons(root, ax, dataPlot_canvas):
        fct_update = lambda cc: update_graph(fwd, model, array(
            [c0.get(), c1.get(), c2.get(), alpha.get()]), ax, dataPlot_canvas)

        # parameter scales
        c0 = tk.Scale(root, from_=80.0, to=120.0, resolution=0.1, label="c0", orient=tk.tk.HORIZONTAL,
                   command=fct_update)
        c1 = tk.Scale(root, from_=0.05, to=0.8, resolution=0.05, label="c1", orient=tk.HORIZONTAL,
                   command=fct_update)
        c2 = tk.Scale(root, from_=0.0, to=5.0, resolution=0.25, label="c2", orient=tk.HORIZONTAL,
                   command=fct_update)
        alpha = tk.Scale(root, from_=0.0, to=1.0, resolution=0.05, label="alpha", orient=tk.HORIZONTAL,
                      command=fct_update)

        c0.grid(row=0, column=1)
        c1.grid(row=1, column=1)
        c2.grid(row=2, column=1)
        alpha.grid(row=3, column=1)

        # replot button
        b1 = tk.Button(root, text="replot",
                       command=lambda: update_graph(fwd,
                                                    model,
                                                    np.array([c0.get(), c1.get(), c2.get(), alpha.get()]),
                                                    ax,
                                                    dataPlot_canvas)).grid(row=8, column=0)
        dataPlot_canvas.show()
        root.mainloop()


def sam_int(s, t, T_i, beta, sigma_L):
    """
    Samuelson volatility function.

    Computes the squared integral of samuelson behavior
    \int _s ^t (e^{-B(T_i - u)} + sigma_L )^2 du

    """

    t1 = exp(-2.0 * beta * (T_i - t)) / (2.0 * beta) - \
        exp(-2.0 * beta * (T_i - s)) / (2.0 * beta)
    t2 = sigma_L**2 * (t - s)
    t3 = 2.0 * sigma_L / beta * \
        (exp(-beta * (T_i - t)) - exp(-beta * (T_i - s)))

    return sqrt((t1 + t2 + t3) / (t - s))


def forward_vols_sam(sigma_v, T, Ti_v, taui_v, beta, sigma_L):
    """
    Forward vols in the Samuelson model.

    :param sigma_v: (atm) vols for maturities Ti
    :type sigma_v: np.array[double]
    :param T: forward time
    :type T: double
    :param Ti_v: array of forward tenors
    :type Ti_v: np.array[double]
    :param taui_v: option tenors
    :param beta: beta in the samuelson parametrization
    :type beta: np.double
    :param sigma_L: sigma_L in the samuelson parameters
    :type sigma_L: np.double
    :returns:
    :rtype:
    """

    return sigma_v * np.array([sam_int(0., T, Ti_v[et], beta, sigma_L) / \
        sam_int(0., taui_v[et], Ti_v[et], beta, sigma_L) for et in range(len(Ti_v))])


def getVolObject(mktDate, comName):
    """
    Gets the vol object for the commodity in question.

    """

    typeToObject = {'JWSS7': JW7Volatility
                   , 'ATM' : ATMFVolatility }

    volType, _, _ = ds.vol_hash[comName]

    return typeToObject[volType](mktDate, comName)
