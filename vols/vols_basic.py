# basic volatility functions

import config
import logging
import numpy as np

from numpy import double, log, exp, sqrt

from scipy.interpolate import splev, splrep  # spline package
from openopt import NLP
import matplotlib as mpl
mpl.use('TkAgg')

from vols.vols_fast import black_vol_inverse_normalized


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

    return np.array([black_vol_inverse(F, K, p, dt, DF, theta, tol)
                     for K, p in zip(K_vec, p_vec)]).ravel()


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

    # TODO: TO REMOVE THIS LATER>
    print('black vol p: {0}, {1}, {2}, {3}'.format(F, K, p, dt))
    return black_vol_inverse_normalized( double(p) / (DF * sqrt(double(F) * double(K)))
                                       , log(double(F) / double(K))
                                       , theta
                                       , tolerance) / sqrt(dt)


# TODO: REMOVE THIS FUNCTION LATER, NO PURPOSE
def black_vol_inverse_naive_vec(F, K_vec, p_vec, dt, DF, theta, tol, solver=None):
    return np.array([black_vol_inverse_naive(F, K, p, dt, DF, theta, tol, solver)
                     for K, p in zip(K_vec, p_vec)]).ravel()


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

