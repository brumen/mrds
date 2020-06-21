# basic volatility functions

import mrds.config as config
import logging
import numpy as np

from numpy import double, log, exp, sqrt

from scipy.interpolate import splev, splrep  # spline package
from openopt import NLP
import matplotlib as mpl
mpl.use('TkAgg')

from mrds.vols.vols_fast import black_vol_inverse_normalized


if config.CUDA_PRESENT:
    import pycuda.autoinit  # this needs to be here.
    import pycuda.gpuarray as gpa
    from pycuda.gpuarray import to_gpu
    from pycuda.compiler import SourceModule


logger = logging.Logger(__name__)


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

    fwd_function = lambda t: splev(t, splrep(fwd_tenors, fwd_prices))  # interpol. prices
    res = [fwd_function(fwd_tenors_wanted)]  # result, part 1
    # append an empty zero matrix
    res.append(np.zeros((len(vol_tenors_wanted), vol_vols.shape[1])))

    for vol_param_ind in range(vol_vols.shape[1]):
        params_tmp = lambda t: splev(t, splrep(vol_tenors, vol_vols[:, vol_param_ind]))
        res[:, vol_param_ind] = params_tmp(vol_tenors_wanted)

    return res


def black_vol_inverse_vec( F : float
                         , K_vec : np.array
                         , p_vec : np.array
                         , dt : float
                         , DF : float
                         , theta
                         , tol : float ) -> np.array:
    """ Inverse black vol for a vector of strikes, and a vector or prices.

    :param F: current forward vol.
    :param K_vec: vector of strikes.
    """

    return [black_vol_inverse(F, K, p, dt, DF, theta, tol)
            for K, p in zip(K_vec, p_vec)]


def black_vol_inverse( F         : float
                     , K         : float
                     , p         : float
                     , dt        : float
                     , DF        : float
                     , theta     : int
                     , tolerance : float ):
    """ Computation of black vol from option price.

    :param F: forward price
    :param K: strike price
    :param p: option price
    :param dt: time to maturity
    :param DF: discount factor until dt
    :param theta: call/put indicator, 1 if call option, -1 for put.
    :param tolerance: tolerance for vol search.
    """

    return black_vol_inverse_normalized( double(p) / (DF * sqrt(double(F) * double(K)))
                                       , log(double(F) / double(K))
                                       , theta
                                       , tolerance) / sqrt(dt)


def black_vol_inverse_naive(F : float, K : float, p : float, dt : float, DF : float, theta : int, tol : float, solver=None):
    """ Inverse black volatility computation.

    :param F: forward price
    :param K: strike price
    :param p: option price to infer black vol from
    :param dt: time to maturity
    :param DF: discount factor
    :param theta:  = 1 ... call option, -1 ... put option
    :param tol: tolerance bound
    :param solver: which NLP solver to use, default scipy_cobyla
    """

    x = log(double(F) / double(K))  # insuring that no integer division is made
    beta = p / (DF * sqrt(F * K))

    # optimization search, initial guess = sigma_c
    optim_pr = NLP( lambda sigma: (b(x, sigma, theta) - beta)**2
                  , sqrt(2 * abs(x))  # inflection point function  (sigma_c)
                  , lb = 1e-6
                  , iprint = -1 )  # lower bound just above 0

    return optim_pr.solve('scipy_cobyla' if solver is None else solver).xf[0] / sqrt(dt)


def sam_int(s : float, t : float, T_i : float, beta : float, sigma_L : float) -> float:
    """ Samuelson volatility function.
        Computes the squared integral of samuelson behavior
           \int _s ^t (e^{-B(T_i - u)} + sigma_L )^2 du

    :param s: lower bound of integration
    :param t: upper bound of integration
    :param T_i: expiry of the forward contract
    :param beta: beta samuelson parameter, speed of decrease
    :param sigma_L: initial volatility TODO: CHECK IF THIS IS TRUE
    :returns: integrated samuelson volatility over a period.
    """

    t1 = exp(-2.0 * beta * (T_i - t)) / (2.0 * beta) - \
        exp(-2.0 * beta * (T_i - s)) / (2.0 * beta)
    t2 = sigma_L**2 * (t - s)
    t3 = 2.0 * sigma_L / beta * \
        (exp(-beta * (T_i - t)) - exp(-beta * (T_i - s)))

    return sqrt((t1 + t2 + t3) / (t - s))


def forward_vols_sam( sigma   : np.array[float]
                    , T       : float
                    , Ti      : np.array[float]
                    , taui    : np.array[float]
                    , beta    : float
                    , sigma_L : float) -> List[float]:
    """ Forward vols in the Samuelson model.

    :param sigma_v: (atm) vols for maturities Ti
    :param T: forward time
    :param Ti_v: array of forward tenors
    :param taui_v: option tenors
    :param beta: beta in the samuelson parametrization
    :param sigma_L: sigma_L in the samuelson parameters
    :returns: samuelson volatilities using the samuelson parametrization.
    """

    return [sigma * sam_int(0., T, Ti_elt, beta, sigma_L) / sam_int(0., taui_elt, Ti_elt, beta, sigma_L)
            for Ti_elt, taui_elt in zip(Ti, taui) ]
